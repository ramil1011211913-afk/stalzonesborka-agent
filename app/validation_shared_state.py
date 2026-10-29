"""Offline multi-process state/lease/crash audit on NEW copied databases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import time
import urllib.request
import urllib.error
import urllib.parse

from app._app_state import application_store, auction_job, Lease, LeaseLost

ROOT = Path('data/validation/multiprocess-state-20261014').resolve()
PYTHON = Path('data/validation/precommit-check-20261014/venv/Scripts/python.exe').resolve()
CONFIG = json.loads(Path('data/validation/exact-test-config.json').read_text(encoding='utf-8-sig'))
SNAPSHOT = Path(CONFIG['snapshot']).resolve()


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def worker(args):
    shared = args.shared.resolve()
    from app.config import Config
    Config.DATA_DIR, Config.DB_PATH = str(shared / 'data'), str(shared / 'data/market.db')
    from app import main
    main.__file__ = str(shared / 'app/main.py')
    main.REFRESH_LEASE_SECONDS = 1.5
    main.check_auction_freshness = lambda *a: dict(is_fresh=True, last_collected=None, age_hours=0)
    if args.mode == 'transaction':
        from app import database
        lease = application_store(Config.DATA_DIR).claim(auction_job(Config.REGION), seconds=1.5)
        lease.progress(0, 1)
        with lease.guard(), closing(database.get_connection()) as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('DELETE FROM auction_lots')
            save(shared / 'transaction-open.json', dict(pid=os.getpid()))
            while True:
                time.sleep(.1)
    if args.mode == 'writes':
        (shared / f'ready-{args.index}').write_text('ready')
        while not (shared / 'start.signal').exists():
            time.sleep(.01)
        errors = []
        for i in range(60):
            try:
                main._persist_owned_artifacts([dict(instance_id=f'{args.index}-{i}', payload='artifact' * 50000)])
            except Exception as error:
                errors.append(repr(error))
        save(shared / f'writer-{args.index}.json', dict(pid=os.getpid(), errors=errors))
        return
    from app import api_client, database, _beam_search
    parse = api_client.parse_lot_additional
    api_client.parse_lot_additional = lambda value: value['audit_parsed'] if isinstance(value, dict) and 'audit_parsed' in value else parse(value)
    def fetch(self, item_id, **kwargs):
        store = application_store(Config.DATA_DIR)
        generation = store.status(auction_job(Config.REGION))['job_id']
        marker = shared / f'fetch-{generation}-{os.getpid()}.json'
        try:
            with marker.open('x') as handle:
                json.dump(dict(pid=os.getpid(), generation=generation), handle)
        except FileExistsError:
            pass
        deadline = time.monotonic() + 90
        while not (shared / 'release.signal').exists():
            assert time.monotonic() < deadline
            time.sleep(.02)
        if (shared / 'failure.signal').exists() and not (shared / 'recover.signal').exists():
            raise RuntimeError('controlled provider failure')
        with closing(database.get_connection()) as conn:
            rows = conn.execute('SELECT * FROM auction_lots WHERE item_id=? AND region=?', (item_id, Config.REGION)).fetchall()
        return [dict(buyoutPrice=r['buyout_price'], startPrice=r['start_price'], amount=r['amount'],
                     startTime=r['start_time'], endTime=r['end_time'], additional={'audit_parsed': {
                         k: json.loads(r[k]) if k == 'bonus_properties' else r[k]
                         for k in ['quality','quality_name','art_percent','stats_random','ndmg','bonus_properties','upgrade_bonus','ptn']}}) for r in rows]
    api_client.StalzoneAPI.fetch_all_lots = fetch
    os.environ['STALZONE_DANGER_LOOKAHEAD'] = '0'
    os.environ['STALZONE_BEAM_WIDTH'] = '1000'
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(main.app, host='127.0.0.1', port=args.port, log_level='error', access_log=False))
    @main.app.get('/__audit_shutdown')
    async def shutdown():
        server.should_exit = True
        return {'stopping': True}
    save(shared / f'backend-{args.port}.json', dict(pid=os.getpid(), port=args.port, gpu=_beam_search._GPU_OK))
    server.run()


def request(port, path, params=None, post=False):
    url = f'http://127.0.0.1:{port}{path}'
    if params:
        url += '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method='POST' if post else 'GET')
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def wait(predicate, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError('Audit wait timed out')


def shared_copy(output):
    shared = output / 'shared'
    (shared / 'data').mkdir(parents=True, exist_ok=False)
    (shared / 'app/templates').mkdir(parents=True, exist_ok=False)
    shutil.copy2(SNAPSHOT, shared / 'data/market.db')
    shutil.copy2('app/templates/index.html', shared / 'app/templates/index.html')
    (shared / 'data/last_owned.json').write_text('[{"instance_id":"legacy"}]')
    return shared


def spawn(mode, shared, log, **kwargs):
    command = [str(PYTHON), '-m', 'app.validation_shared_state', mode, '--shared', str(shared)]
    for key, value in kwargs.items():
        command.extend(['--' + key, str(value)])
    return subprocess.Popen(command, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)


def writes(output, reader):
    shared = shared_copy(output)
    store = application_store(shared / 'data')
    store.import_legacy(shared / 'data/last_owned.json')
    original = (shared / 'data/last_owned.json').read_bytes()
    processes, handles = [], []
    try:
        for index in range(4):
            log = (output / f'writer-{index}.txt').open('x'); handles.append(log)
            processes.append(spawn('writes', shared, log, index=index))
        wait(lambda: len(list(shared.glob('ready-*'))) == 4)
        (shared / 'start.signal').write_text('start')
        reads = 0
        while any(p.poll() is None for p in processes):
            if reader:
                value = store.owned()
                assert len(value['artifacts']) == 1
                reads += 1
            time.sleep(.001)
        assert all(p.returncode == 0 for p in processes)
        results = [json.loads((shared / f'writer-{i}.json').read_text()) for i in range(4)]
        assert not any(r['errors'] for r in results)
        assert store.owned()['revision'] == 241
        assert (shared / 'data/last_owned.json').read_bytes() == original
        save(output / 'results.json', dict(writes=240, errors=0, reads=reads, revision=241, legacy_unchanged=True))
        print('SQLite writes: 240/240, reader:', reader, flush=True)
    finally:
        for p in processes:
            if p.poll() is None:
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'], capture_output=True)
            p.wait(timeout=10)
        for log in handles:
            log.close()


def http_crash(output):
    shared = shared_copy(output)
    store = application_store(shared / 'data')
    legacy = (shared / 'data/last_owned.json').read_bytes()
    processes, ports, handles = [], [], []
    try:
        for i in range(2):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
            ports.append(port)
            log = (output / f'worker-{i}.txt').open('x'); handles.append(log)
            processes.append(spawn('serve', shared, log, port=port))
            def ready():
                assert processes[-1].poll() is None
                try:
                    request(port,'/api/stats'); return True
                except urllib.error.URLError:
                    return False
            wait(ready)
        backends = [json.loads((shared / f'backend-{port}.json').read_text()) for port in ports]
        assert all(b['gpu'] for b in backends)
        assert request(ports[0],'/api/owned-artifacts')['artifacts'] == [{'instance_id':'legacy'}]
        params = dict(budget=CONFIG['budget'], tolerance=0, container_id=CONFIG['container_id'],
                      priorities='{"sprint_speed_modifier":10,"health_bonus":8}',
                      owned=json.dumps(CONFIG['owned_artifacts']), armor_ids='["3g6p5","5lr71"]',
                      min_thresholds='{"health_bonus":0}', allow_duplicates='true', top_n=2)
        expected = request(ports[0],'/api/builds',params)['builds']
        assert expected
        def search(i):
            body = request(ports[i % 2],'/api/builds',params)
            assert body['builds'] == expected
            return True
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert all(pool.map(search,range(6)))
        assert request(ports[0],'/api/owned-artifacts') == request(ports[1],'/api/owned-artifacts')
        with ThreadPoolExecutor(max_workers=4) as pool:
            starts = list(pool.map(lambda i: request(ports[i % 2],'/api/refresh-auction',post=True),range(8)))
        assert sum(s['status'] == 'started' for s in starts) == 1
        wait(lambda: len(list(shared.glob('fetch-1-*.json'))) == 1)
        time.sleep(3.2)  # > two lease periods; heartbeat must keep ownership.
        statuses = [request(port,'/api/refresh-status') for port in ports]
        assert all(s['running'] and s['job_id'] == 1 and s['total'] > 0 for s in statuses)
        assert request(ports[1],'/api/refresh-auction',post=True)['status'] == 'already_running'
        assert len(list(shared.glob('fetch-1-*.json'))) == 1
        with store.connection() as conn:
            row = dict(conn.execute('SELECT * FROM refresh_jobs').fetchone())
        old = Lease(store, row['name'], row['token'], row['generation'], 1.5)
        owner = next(b for b in backends if b['pid'] == row['owner_pid'])
        survivor = next(b for b in backends if b['pid'] != row['owner_pid'])
        os.kill(owner['pid'], signal.SIGTERM)  # only the recorded test worker.
        wait(lambda: request(survivor['port'],'/api/refresh-status')['state'] == 'expired',5)
        (shared / 'release.signal').write_text('release')
        # Even a partially refreshed market can appear fresh by MAX(timestamp).
        # An expired shared job must be recovered by the next search request.
        assert request(survivor['port'],'/api/builds',params)['builds']
        wait(lambda: request(survivor['port'],'/api/refresh-status')['state'] == 'completed',30)
        status = request(survivor['port'],'/api/refresh-status')
        assert status['job_id'] == 2 and status['done'] == status['total']
        assert not old.heartbeat() and not old.finish('late stale completion')
        try:
            with old.guard():
                raise AssertionError('Stale owner entered market guard')
        except LeaseLost:
            pass
        after = request(survivor['port'],'/api/builds',params)
        assert after['builds'] and after['builds'][0]['priority_value'] == expected[0]['priority_value']
        (shared / 'failure.signal').write_text('failure')
        assert request(survivor['port'],'/api/refresh-auction',post=True)['status'] == 'started'
        wait(lambda: request(survivor['port'],'/api/refresh-status')['state'] == 'failed')
        assert request(survivor['port'],'/api/refresh-status')['error']
        (shared / 'recover.signal').write_text('recover')
        assert request(survivor['port'],'/api/refresh-auction',post=True)['status'] == 'started'
        wait(lambda: request(survivor['port'],'/api/refresh-status')['state'] == 'completed')
        assert request(survivor['port'],'/api/refresh-status')['job_id'] == 4
        assert (shared / 'data/last_owned.json').read_bytes() == legacy
        save(output / 'results.json', dict(gpu_workers=2, parallel_searches=6, concurrent_starts=8,
            refresh_owners_first_generation=1, heartbeat_kept_lease=True, crashed_owner_pid=owner['pid'],
            recovered_job_id=2, stale_owner_fenced=True, shared_progress=True, controlled_failure_visible=True,
            legacy_json_unchanged=True, search_after_recovery=True,
            automatic_recovery_on_search=True, retry_after_failure=True))
        print('Two GPU workers, single refresh, crash recovery and fencing: passed', flush=True)
    finally:
        for port in ports:
            try:
                request(port,'/__audit_shutdown')
            except (OSError, urllib.error.URLError):
                pass
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'], capture_output=True)
                p.wait(timeout=10)
        for log in handles:
            log.close()


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['audit','transaction-audit','transaction','writes','serve'])
    parser.add_argument('--shared',type=Path)
    parser.add_argument('--index',type=int)
    parser.add_argument('--port',type=int)
    parser.add_argument('--output-name')
    args = parser.parse_args()
    if args.output_name:
        destination = ROOT / args.output_name
        assert destination.parent == ROOT and destination.name not in ['.','..']
        destination.mkdir(exist_ok=False)
        ROOT = destination
    if args.mode == 'transaction-audit':
        transaction_crash(); return
    if args.mode != 'audit':
        worker(args); return
    digest = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    for name,fn in [('writes',lambda p:writes(p,False)),('read-writes',lambda p:writes(p,True)),('gpu-http-crash',http_crash)]:
        output = ROOT / name; output.mkdir(exist_ok=False); fn(output)
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == digest


def transaction_crash():
    output = ROOT / 'transaction-crash'; output.mkdir(exist_ok=False)
    shared = shared_copy(output)
    with sqlite3.connect(shared / 'data/market.db') as conn:
        before = conn.execute('SELECT * FROM auction_lots ORDER BY id').fetchall()
    with (output / 'worker.txt').open('x') as log:
        process = spawn('transaction', shared, log)
        try:
            wait(lambda: (shared / 'transaction-open.json').exists())
            pid = json.loads((shared / 'transaction-open.json').read_text())['pid']
            os.kill(pid, signal.SIGTERM)
            process.wait(timeout=10)
            with sqlite3.connect(shared / 'data/market.db') as conn:
                after = conn.execute('SELECT * FROM auction_lots ORDER BY id').fetchall()
            assert after == before
            store = application_store(shared / 'data')
            from app.config import Config
            wait(lambda: store.status(auction_job(Config.REGION))['state'] == 'expired',5)
            new = store.claim(auction_job(Config.REGION))
            assert new and new.generation == 2
            new.finish()
            save(output / 'results.json', dict(crashed_with_open_transactions=True,
                 market_rows_unchanged=len(after), state_lock_released=True, new_generation=2))
            print('Crash during market/state transactions: rollback and takeover passed',flush=True)
        finally:
            if process.poll() is None:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
                process.wait(timeout=10)


if __name__ == '__main__':
    main()

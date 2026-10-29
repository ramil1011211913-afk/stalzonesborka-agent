"""Clean-install and multi-process checks, isolated from original data.

Run with Python 3.12. The orchestrator uses a NEW venv for CPU/app workers.
No Git staging, production DB access, external refresh, or global pip writes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path('data/validation/precommit-check-20261014').resolve()
CONFIG = json.loads(Path('data/validation/exact-test-config.json').read_text(encoding='utf-8-sig'))
SNAPSHOT = Path(CONFIG['snapshot']).resolve()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def configure_app(shared, stale=False):
    from app.config import Config
    Config.DATA_DIR = str(shared / 'data')
    Config.DB_PATH = str(shared / 'data/stalzone.db')
    from app import main
    # Real import-time initialization runs on the NEW copied database.
    main.__file__ = str(shared / 'app/main.py')
    main.check_auction_freshness = lambda *args: dict(is_fresh=not stale,
        last_collected=None, age_hours=0, last_emission=None)
    def isolated_refresh(lease):
        save(shared / ('refresh-started-' + str(os.getpid()) + '.json'), dict(pid=os.getpid()))
        with lease.keep_alive():
            lease.progress(0, 1)
            while not (shared / 'end-refresh.signal').exists():
                time.sleep(.05)
        lease.finish()
    main._run_auction_refresh = isolated_refresh
    os.environ['STALZONE_DANGER_LOOKAHEAD'] = '0'
    os.environ['STALZONE_BEAM_WIDTH'] = '1000'
    return main


def worker(args):
    shared = args.shared.resolve()
    main = configure_app(shared, args.stale)
    if args.mode == 'serve':
        import uvicorn
        from app import _beam_search
        save(shared / ('backend-' + str(os.getpid()) + '.json'),
             dict(pid=os.getpid(), gpu_available=_beam_search._GPU_OK))
        server = uvicorn.Server(uvicorn.Config(main.app, host='127.0.0.1', port=args.port,
                                             log_level='error', access_log=False))
        @main.app.get('/__precommit_shutdown')
        async def shutdown_test_server():
            server.should_exit = True
            return {'stopping': True}
        server.run()
        return
    if args.mode == 'writes':
        (shared / ('ready-' + str(args.index))).write_text('ready')
        deadline = time.monotonic() + 60
        while not (shared / 'start.signal').exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        successes, errors = 0, []
        for iteration in range(60):
            data = [dict(instance_id=f'{args.index}-{iteration}', writer=args.index,
                         iteration=iteration, payload='artifact' * 50000)]
            try:
                main._persist_owned_artifacts(data)
                successes += 1
            except Exception as error:
                errors.append(dict(iteration=iteration, type=type(error).__name__, message=str(error)))
        save(shared / ('writer-' + str(args.index) + '.json'), dict(pid=os.getpid(),
             successes=successes, errors=errors))


def new_shared(output):
    shared = output / 'shared'
    (shared / 'data').mkdir(parents=True, exist_ok=False)
    (shared / 'app/templates').mkdir(parents=True, exist_ok=False)
    shutil.copy2(SNAPSHOT, shared / 'data/stalzone.db')
    shutil.copy2('app/templates/index.html', shared / 'app/templates/index.html')
    return shared


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def http(port, path, params=None):
    url = f'http://127.0.0.1:{port}' + path
    if params:
        url += '?' + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            body = response.read().decode('utf-8')
            return response.status, json.loads(body) if 'application/json' in response.headers.get('Content-Type', '') else body
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode('utf-8')


def servers(output, count, stale=False):
    python = ROOT / 'venv/Scripts/python.exe'
    shared = new_shared(output)
    processes, ports, handles = [], [], []
    try:
        for index in range(count):
            port = free_port()
            ports.append(port)
            log = (output / f'worker-{index}.txt').open('x', encoding='utf-8')
            handles.append(log)
            cmd = [str(python), '-m', 'app.validation_precommit', 'serve', '--shared', str(shared), '--port', str(port)]
            if stale:
                cmd.append('--stale')
            process = subprocess.Popen(cmd, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
            processes.append(process)
            deadline = time.monotonic() + 30
            while True:
                assert process.poll() is None, 'Worker startup failed; see worker log'
                try:
                    code, stats = http(port, '/api/stats')
                    if code == 200:
                        break
                except urllib.error.URLError:
                    pass
                assert time.monotonic() < deadline, 'Worker startup timed out'
                time.sleep(.1)
        status, html = http(ports[0], '/')
        assert status == 200 and 'objectiveMode' in html
        requests = []
        for mode in ['balanced', 'lexicographic', 'min_budget']:
            params = dict(budget=CONFIG['budget'], tolerance=0, container_id=CONFIG['container_id'],
                          priorities=json.dumps({'sprint_speed_modifier': 10, 'health_bonus': 8}),
                          owned=json.dumps(CONFIG['owned_artifacts']), armor_ids='["3g6p5","5lr71"]',
                          allow_duplicates='true', optimization_mode=mode, top_n=2,
                          min_thresholds='{"health_bonus":0}')
            if mode == 'lexicographic':
                params['priority_order'] = '["sprint_speed_modifier","health_bonus"]'
            if mode == 'min_budget':
                params['min_thresholds'] = '{"sprint_speed_modifier":10,"health_bonus":0}'
            code, body = http(ports[0], '/api/builds', params)
            assert code == 200 and body['builds'], (code, body)
            requests.append((params, body['builds']))
        def request(index):
            params, expected = requests[index % 3]
            code, body = http(ports[index % len(ports)], '/api/builds', params)
            assert code == 200 and body['builds'] == expected, (code, body)
            return dict(index=index, port=ports[index % len(ports)], mode=params['optimization_mode'], passed=True)
        with ThreadPoolExecutor(max_workers=count) as executor:
            rows = list(executor.map(request, range(count * 3)))
        logs = list((shared / 'data/build_logs').glob('*.json'))
        assert len(logs) == 3 + count * 3
        assert all(json.loads(log.read_text(encoding='utf-8')) for log in logs)
        refreshes = list(shared.glob('refresh-started-*.json'))
        result = dict(processes=count, pids=[p.pid for p in processes], routes_passed=True,
                      requests=rows, logs=len(logs), independent_refresh_jobs=len(refreshes),
                      stale=stale, refresh_states=[http(port, '/api/refresh-status')[1] for port in ports])
        result['backends'] = [json.loads(file.read_text(encoding='utf-8')) for file in shared.glob('backend-*.json')]
        save(output / 'results.json', result)
        print('HTTP workers:', count, 'stale:', stale, 'requests passed; refresh jobs:', len(refreshes), flush=True)
    finally:
        (shared / 'end-refresh.signal').write_text('stop')
        for port in ports:
            try:
                http(port, '/__precommit_shutdown')
            except (OSError, urllib.error.URLError):
                pass
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Windows venv redirectors may own an actual Python child.
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               capture_output=True, check=True)
                process.wait(timeout=10)
        for handle in handles:
            handle.close()


def write_stress(output, reader):
    from app._app_state import application_store
    shared = new_shared(output)
    store = application_store(shared / 'data')
    processes, handles = [], []
    python = ROOT / 'venv/Scripts/python.exe'
    try:
        for index in range(4):
            log = (output / f'worker-{index}.txt').open('x', encoding='utf-8')
            handles.append(log)
            processes.append(subprocess.Popen([str(python), '-m', 'app.validation_precommit', 'writes',
                '--shared', str(shared), '--index', str(index)], stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW))
        deadline = time.monotonic() + 45
        while len(list(shared.glob('ready-*'))) < 4:
            assert all(p.poll() is None for p in processes)
            assert time.monotonic() < deadline
            time.sleep(.02)
        (shared / 'start.signal').write_text('start')
        malformed, read_errors, reads = [], [], 0
        while any(p.poll() is None for p in processes):
            if reader:
                try:
                    data = store.owned()['artifacts']
                    if data:
                        assert len(data) == 1 and data[0]['payload'] == 'artifact' * 50000
                        reads += 1
                except json.JSONDecodeError as error:
                    malformed.append(str(error))
                except OSError as error:
                    read_errors.append(str(error))
            time.sleep(.001)
        assert all(p.returncode == 0 for p in processes), 'Worker process crashed'
        writers = [json.loads((shared / f'writer-{index}.json').read_text(encoding='utf-8')) for index in range(4)]
        final = store.owned()['artifacts']
        result = dict(processes=4, intended_writes=240, reader=reader, successful_writes=sum(w['successes'] for w in writers),
                      write_errors=sum(len(w['errors']) for w in writers), writers=writers,
                      reads=reads, malformed_json=malformed, read_errors=read_errors,
                      final_record=final[0]['instance_id'], temporary_files=len(list((shared / 'data').glob('last_owned-*.tmp'))))
        save(output / 'results.json', result)
        print('Multi-process writes:', result['successful_writes'], '/240; errors:', result['write_errors'], 'reader:', reader, flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
        for handle in handles:
            handle.close()


def audit():
    digest = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    for name, fn in [('startup', lambda output: servers(output, 1)),
                     ('multiprocess-http', lambda output: servers(output, 2)),
                     ('multiprocess-stale', lambda output: servers(output, 2, True)),
                     ('multiprocess-writes', lambda output: write_stress(output, False)),
                     ('multiprocess-read-write', lambda output: write_stress(output, True))]:
        output = ROOT / name
        output.mkdir(exist_ok=False)
        fn(output)
        assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['audit', 'gpu-http', 'startup-repeat', 'serve', 'writes'])
    parser.add_argument('--shared', type=Path)
    parser.add_argument('--port', type=int)
    parser.add_argument('--index', type=int)
    parser.add_argument('--stale', action='store_true')
    args = parser.parse_args()
    if args.mode == 'audit':
        audit()
    elif args.mode == 'gpu-http':
        output = ROOT / 'multiprocess-gpu-http'
        output.mkdir(exist_ok=False)
        servers(output, 2)
    elif args.mode == 'startup-repeat':
        output = ROOT / 'startup-cleanup-verified'
        output.mkdir(exist_ok=False)
        servers(output, 1)
    else:
        worker(args)


if __name__ == '__main__':
    main()

"""Release checks isolated from live data, Git staging, and external services."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, ExitStack
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from app import builder, _beam_search as beam
from app._objective import Objective

ROOT = Path('data/validation/release-readiness-20261014')
CONFIG = json.loads(Path('data/validation/exact-test-config.json').read_text(encoding='utf-8-sig'))
SNAPSHOT = Path(CONFIG['snapshot']).resolve()
FULL_POOL = False


def raw_connection():
    conn = sqlite3.connect(SNAPSHOT.as_uri() + '?mode=ro&immutable=1', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def connection():
    return closing(raw_connection())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def engine_request(mode):
    rng = np.random.default_rng(20261015)
    candidates = [SimpleNamespace(item_id=str(i // 2), physical_id='lot-' + str(i // 2),
                  buyout_price=int(rng.integers(1, 20)),
                  stats=dict(speed_modifier=float(rng.random() * 10),
                             health_bonus=float(rng.random() * 12 - 4),
                             radiation_accumulation=float(rng.integers(-10, 15)))) for i in range(64)]
    priorities = dict(speed_modifier=10, health_bonus=8)
    spec = Objective.create(priorities, dict(speed_modifier=40, health_bonus=60), mode,
                            list(priorities) if mode == 'lexicographic' else None)
    args = (candidates, 4, 45, 4, .4, {'health_bonus': 2, 'radiation_accumulation': -1},
            dict(zip(spec.keys, spec.scales)), priorities, {'health_bonus': 0}, 10)
    kwargs = dict(objective=spec, allow_duplicates=True, require_carry=True,
                  max_thresholds={'radiation_accumulation': 5}, beam_width=2000)
    return args, kwargs


def concurrency(output):
    assert beam._GPU_OK
    entries = []
    local = threading.local()
    original_array, original_impl = beam.cp.asarray, beam._search_impl
    def allocation(*args, **kwargs):
        if getattr(local, 'fault', False):
            local.allocations += 1
            if local.allocations == 5:
                local.fault = False
                raise beam.cp.cuda.memory.OutOfMemoryError(1024, 0, 0)
        return original_array(*args, **kwargs)
    def implementation(*args, **kwargs):
        if hasattr(local, 'attempts'):
            local.attempts.append('gpu' if kwargs['use_gpu'] else 'cpu')
        return original_impl(*args, **kwargs)
    with patch.object(beam.cp, 'asarray', allocation), patch.object(beam, '_search_impl', implementation):
        for flag in ['0', '1']:
            with patch.dict(os.environ, STALZONE_DANGER_LOOKAHEAD=flag):
                references = {}
                for mode in ['balanced', 'lexicographic', 'min_budget']:
                    args, kwargs = engine_request(mode)
                    references[mode] = dict(gpu=beam.search(*args, **kwargs),
                        cpu=original_impl(*args, **kwargs, use_gpu=False, danger_lookahead=flag == '1'))
                for streams in ['default', 'independent']:
                    for fault in [False, True]:
                        for round_id in range(3):
                            barrier = threading.Barrier(3)
                            done = threading.Event()
                            def worker(index):
                                mode = ['balanced', 'lexicographic', 'min_budget'][index]
                                local.fault = fault and index == 0
                                local.allocations, local.attempts = 0, []
                                stream = beam.cp.cuda.Stream(non_blocking=True) if streams == 'independent' else beam.cp.cuda.Stream.null
                                with stream:
                                    sentinel = beam.cp.full(32, index + 1, dtype=beam.cp.float64)
                                    barrier.wait(timeout=20)
                                    begin = time.perf_counter()
                                    try:
                                        args, kwargs = engine_request(mode)
                                        actual = beam.search(*args, **kwargs)
                                    finally:
                                        if index == 0:
                                            done.set()
                                    assert done.wait(timeout=60)
                                    np.testing.assert_array_equal(beam.cp.asnumpy(sentinel), np.full(32, index + 1))
                                    expected = references[mode]['cpu' if fault and index == 0 else 'gpu']
                                    np.testing.assert_array_equal(actual[1], expected[1])
                                    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=1e-9)
                                    return dict(mode=mode, attempts=local.attempts, seconds=time.perf_counter() - begin)
                            with ThreadPoolExecutor(max_workers=3) as pool:
                                rows = list(pool.map(worker, range(3)))
                            assert rows[0]['attempts'] == (['gpu', 'cpu'] if fault else ['gpu'])
                            assert all(row['attempts'] == ['gpu'] for row in rows[1:])
                            assert beam._GPU_OK
                            entries.append(dict(flag=flag, streams=streams, oom=fault, round=round_id, passed=True, workers=rows))
                            save(output / 'concurrency.json', entries)
                            print('parallel', flag, streams, fault, round_id, 'passed', flush=True)
    gc_used = beam.cp.get_default_memory_pool().used_bytes()
    save(output / 'summary.json', dict(groups=len(entries), parallel_searches=len(entries) * 3,
                                      pool_used_after=gc_used, global_gpu_backend=beam._GPU_OK))


def cpu(output):
    rows = []
    for flag in [False, True]:
        references = {}
        for mode in ['balanced', 'lexicographic', 'min_budget']:
            args, kwargs = engine_request(mode)
            references[mode] = beam._search_impl(*args, **kwargs, use_gpu=False, danger_lookahead=flag)
        def worker(index):
            mode = ['balanced', 'lexicographic', 'min_budget'][index % 3]
            args, kwargs = engine_request(mode)
            actual = beam._search_impl(*args, **kwargs, use_gpu=False, danger_lookahead=flag)
            expected = references[mode]
            np.testing.assert_array_equal(actual[1], expected[1])
            np.testing.assert_array_equal(actual[0], expected[0])
            return dict(mode=mode, flag=flag, passed=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            rows.extend(pool.map(worker, range(9)))
    save(output / 'cpu.json', rows)


def load_main(output, stack):
    from app import database, lunar_bridge
    (output / 'app').mkdir(exist_ok=False)
    (output / 'data').mkdir(exist_ok=False)
    stack.enter_context(patch.object(database, 'init_db', lambda: None))
    main = importlib.import_module('app.main')
    stack.enter_context(patch.object(main.Config, 'DATA_DIR', str(output / 'data')))
    stack.enter_context(patch.object(main, '__file__', str(output / 'app' / 'main.py')))
    stack.enter_context(patch.object(builder, 'get_connection', connection))
    stack.enter_context(patch.object(main, 'get_connection', raw_connection))
    stack.enter_context(patch.object(lunar_bridge, 'get_connection', connection))
    stack.enter_context(patch.object(main, 'check_auction_freshness',
                        lambda *args: {'is_fresh': True, 'last_collected': None, 'age_hours': 0}))
    loader = builder.load_artifact_options
    if not FULL_POOL:
        stack.enter_context(patch.object(builder, 'load_artifact_options', lambda **kwargs: loader(**kwargs)[:256]))
    stack.enter_context(patch.dict(os.environ, STALZONE_DANGER_LOOKAHEAD='1',
                                  STALZONE_BEAM_WIDTH='10000' if FULL_POOL else '1000'))
    return main


def api(output):
    from fastapi.testclient import TestClient
    with ExitStack() as stack:
        main = load_main(output, stack)
        client = TestClient(main.app)
        base = dict(budget=CONFIG['budget'], tolerance=0, container_id=CONFIG['container_id'],
                    priorities=json.dumps({'sprint_speed_modifier': 10, 'health_bonus': 8}),
                    armor_ids='["3g6p5","5lr71"]', allow_duplicates='true', top_n=3,
                    min_thresholds='{"health_bonus":0}')
        requests = []
        for mode in ['balanced', 'lexicographic', 'min_budget']:
            params = dict(base, optimization_mode=mode)
            if mode == 'lexicographic':
                params['priority_order'] = '["sprint_speed_modifier","health_bonus"]'
            if mode == 'min_budget':
                params['min_thresholds'] = '{"sprint_speed_modifier":3,"health_bonus":0}'
            r = client.get('/api/builds', params=params)
            assert r.status_code == 200 and r.json()['builds'], r.text
            requests.append((params, r.json()['builds']))
        before = set((output / 'data/build_logs').glob('*.json'))
        def worker(index):
            params, expected = requests[index % len(requests)]
            begin = time.perf_counter()
            r = client.get('/api/builds', params=params)
            assert r.status_code == 200 and r.json()['builds'] == expected, r.text
            return dict(index=index, mode=params['optimization_mode'], seconds=time.perf_counter() - begin, passed=True)
        with ThreadPoolExecutor(max_workers=2 if FULL_POOL else 3) as pool:
            rows = list(pool.map(worker, range(12)))
        after = set((output / 'data/build_logs').glob('*.json'))
        logs = after - before
        for file in logs:
            json.loads(file.read_text(encoding='utf-8'))
        save(output / 'api.json', dict(requests=rows, full_pool=FULL_POOL, workers=2 if FULL_POOL else 3,
                                      expected_new_logs=12, actual_new_logs=len(logs),
                                      logs_all_valid_json=True, no_collisions=len(logs) == 12))
        print('API searches passed; new logs:', len(logs), 'expected 12', flush=True)


def ui(output):
    import urllib.request
    import uvicorn
    with ExitStack() as stack:
        main = load_main(output, stack)
        template = output / 'app/templates/index.html'
        template.parent.mkdir(exist_ok=False)
        shutil.copy2('app/templates/index.html', template)
        def summary():
            with connection() as conn:
                return dict(artefacts=conn.execute('SELECT COUNT(*) FROM artefacts').fetchone()[0],
                            active_lots=conn.execute('SELECT COUNT(*) FROM auction_lots').fetchone()[0])
        stack.enter_context(patch.object(main, 'get_artefact_stats_summary', summary))
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(main.app, host='127.0.0.1', port=port, log_level='error', access_log=False))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        chrome = None
        try:
            deadline = time.monotonic() + 20
            while not server.started:
                assert time.monotonic() < deadline, 'Server failed to start'
                time.sleep(.05)
            profile = output.resolve() / 'chrome-profile'
            chrome = subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                '--headless=new', '--disable-background-networking', '--no-first-run',
                '--no-default-browser-check', '--remote-debugging-port=0',
                '--user-data-dir=' + str(profile), 'about:blank'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW)
            active = profile / 'DevToolsActivePort'
            while not active.exists():
                assert time.monotonic() < deadline and chrome.poll() is None, 'Chrome failed to start'
                time.sleep(.05)
            debug_port = active.read_text().splitlines()[0]
            with urllib.request.urlopen(f'http://127.0.0.1:{debug_port}/json/list') as response:
                pages = json.load(response)
            ws_url = next(page['webSocketDebuggerUrl'] for page in pages if page['type'] == 'page')
            run = subprocess.run(['node', 'app/validation_ui.cjs', ws_url,
                                  f'http://127.0.0.1:{port}/', str(output.resolve())],
                                 capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)
            (output / 'browser-log.txt').write_text(run.stdout + run.stderr, encoding='utf-8')
            assert run.returncode == 0, run.stdout + run.stderr
        finally:
            if chrome is not None and chrome.poll() is None:
                chrome.terminate()
                chrome.wait(timeout=10)
            server.should_exit = True
            thread.join(timeout=20)
            assert not thread.is_alive()


def main():
    global FULL_POOL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['concurrency', 'cpu', 'api', 'ui'])
    parser.add_argument('--output-name')
    parser.add_argument('--full-pool', action='store_true')
    args = parser.parse_args()
    FULL_POOL = args.full_pool
    output = ROOT / (args.output_name or args.phase)
    assert output.parent == ROOT
    output.mkdir(exist_ok=False)
    digest = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    globals()[args.phase](output)
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == digest
    save(output / 'metadata.json', dict(snapshot_sha256=digest, snapshot_unchanged=True, completed=True))


if __name__ == '__main__':
    main()

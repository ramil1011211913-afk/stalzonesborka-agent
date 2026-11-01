"""Isolated local load and search checks; never edits the live market database."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def serve(data, port):
    os.chdir(ROOT)
    from app.config import Config
    Config.DATA_DIR, Config.DB_PATH = str(data), str(data / 'stalzone.db')
    from app import main
    main.check_auction_freshness = lambda *a: dict(is_fresh=True, last_collected=None, age_hours=0, last_emission=None)
    main._start_auction_refresh = lambda: False
    import uvicorn
    uvicorn.run(main.app, host='127.0.0.1', port=port, log_level='warning')


def main():
    parser = argparse.ArgumentParser(description='Local isolated application QA')
    parser.add_argument('--requests', type=int, default=100, help='Requests per load wave')
    parser.add_argument('--waves', type=int, default=3)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--searches', type=int, default=16, help='Repeated searches for resource sampling')
    parser.add_argument('--full', action='store_true', help='Also run the complete pytest suite')
    parser.add_argument('--skip-ui', action='store_true')
    parser.add_argument('--skip-search', action='store_true')
    parser.add_argument('--server', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--data', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--port', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.server:
        serve(args.data.resolve(), args.port)
        return
    if any(n < 0 for n in (args.requests, args.waves, args.searches)) or args.workers < 1:
        parser.error('Counts must be nonnegative and workers positive')
    import psutil
    import requests
    source = ROOT / 'data' / 'stalzone.db'
    if not source.is_file():
        raise SystemExit('Market database missing: ' + str(source))
    run = ROOT / 'data' / 'local_qa' / (datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as src, sqlite3.connect(run / 'stalzone.db') as dst:
        src.backup(dst)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    report = {'run': str(run), 'server': url, 'load': [], 'inputs': [], 'search': [], 'resources': [], 'errors': []}
    print('Отчёт и копия данных:', run, flush=True)
    with (run / 'server.log').open('w', encoding='utf8') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--server', '--data', str(run), '--port', str(port)],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        try:
            deadline = time.monotonic() + 40
            while True:
                if child.poll() is not None:
                    raise RuntimeError('Test server exited; see server.log')
                try:
                    requests.get(url + '/api/stats', timeout=1).raise_for_status()
                    break
                except requests.RequestException:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Test server did not start; see server.log')
                    time.sleep(.2)
            def metrics():
                return dict(rss=process.memory_info().rss, handles=process.num_handles() if os.name == 'nt' else process.num_fds(),
                            threads=process.num_threads(), children=[p.name() for p in process.children(recursive=True)])
            def get(i):
                path = ('/api/stats', '/api/artefacts', '/api/containers', '/api/armors')[i % 4]
                try:
                    return requests.get(url + path, timeout=30).status_code
                except requests.RequestException:
                    return 0
            for wave in range(args.waves):
                start = time.monotonic()
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    codes = list(pool.map(get, range(args.requests)))
                row = dict(wave=wave + 1, count=len(codes), failures=sum(x != 200 for x in codes),
                           seconds=round(time.monotonic()-start, 2), **metrics())
                report['load'].append(row)
                if row['failures']: report['errors'].append('Failed reference requests')
                print('Нагрузка:', row, flush=True)
            invalid = [({'min_thresholds': '[1]'}, 422), ({'owned': '[{"quality":"oops"}]'}, 422),
                       ({'priorities': '{"speed_modifier":NaN}'}, 422), ({'exclude': '{"a":1}'}, 422)]
            for params, expected in invalid:
                status = requests.get(url + '/api/builds', params=params, timeout=30).status_code
                report['inputs'].append(dict(params=params, status=status, expected=expected))
                if status != expected: report['errors'].append('Unexpected input response: ' + str(params))
            common = dict(budget=11000000, tolerance=0, container_id='rgoz',
                          priorities=json.dumps({'total_speed': 10}), allow_duplicates='true',
                          carryable_only='true', top_n=3, unknown_percent='midpoint')
            if not args.skip_search:
                cases = [('balanced', {}), ('lexicographic', {'optimization_mode':'lexicographic', 'priority_order':'["total_speed"]'}),
                         ('minimum-price', {'min_budget':'true', 'min_budget_timeout':4, 'min_thresholds':'{"total_speed":10}'}),
                         ('max-quality', {'max_quality':'true', 'quality_timeout':20}),
                         ('impossible', {'min_thresholds':'{"total_speed":1000}'})]
                for name, extra in cases:
                    params = dict(common, **extra)
                    response = requests.get(url+'/api/builds', params=params, timeout=90)
                    response.raise_for_status()
                    data = response.json()
                    issues = []
                    for build in data.get('builds', []):
                        arts = build['artifacts']
                        if len({a['physical_id'] for a in arts}) != len(arts): issues.append('duplicate physical artifact')
                        if sum(a['buyout_price'] for a in arts if not a['is_owned']) != build['artifact_price']: issues.append('incorrect price')
                        if build['artifact_price'] > common['budget']: issues.append('budget exceeded')
                        if not build['carryable']: issues.append('not carryable')
                        if build['container']['slots'] != len(arts): issues.append('slot count')
                        for key, minimum in json.loads(params.get('min_thresholds', '{}')).items():
                            if build['combined_stats'].get(key, 0) + .0001 < minimum: issues.append('threshold '+key)
                    if name == 'impossible' and data.get('builds'): issues.append('impossible constraint accepted')
                    row = dict(name=name, state=(data.get('search_status') or {}).get('state'), builds=len(data.get('builds', [])), issues=issues)
                    report['search'].append(row)
                    if issues: report['errors'].append(name + ': ' + ', '.join(issues))
                    print('Поиск:', row, flush=True)
            sample = dict(budget=5000000, tolerance=0, container_id='dkln', priorities=json.dumps({'total_speed':10}), top_n=1)
            for i in range(args.searches):
                requests.get(url+'/api/builds', params=sample, timeout=60).raise_for_status()
                if i in {0, 1, 5, 10, args.searches-1}:
                    row = dict(search=i+1, **metrics())
                    report['resources'].append(row)
                    print('Ресурсы:', row, flush=True)
            if not args.skip_ui:
                try:
                    from playwright.sync_api import sync_playwright
                except ImportError:
                    report['ui'] = {'state': 'skipped', 'reason': 'playwright not installed'}
                else:
                    with sync_playwright() as playwright:
                        browser = playwright.chromium.launch(headless=True, channel='chrome')
                        page = browser.new_page(viewport={'width':1280, 'height':900})
                        errors = []
                        page.on('pageerror', lambda error: errors.append(str(error)))
                        page.route('**/*', lambda route: route.continue_() if route.request.url.startswith(url+'/') else route.abort())
                        page.goto(url+'/', wait_until='domcontentloaded')
                        page.wait_for_selector('#prio_total_speed', timeout=30000)
                        page.select_option('#containerId', 'rgoz')
                        page.fill('#budget', '11000000')
                        page.locator('#prio_total_speed').evaluate('(e) => {e.value=10; e.dispatchEvent(new Event("input", {bubbles:true}));}')
                        with page.expect_response(lambda r: '/api/builds?' in r.url, timeout=90000) as info:
                            page.click('#buildBtn')
                        result = info.value.json()
                        page.wait_for_function("!document.querySelector('#buildBtn').disabled", timeout=30000)
                        page.screenshot(path=str(run/'ui.png'), full_page=True)
                        report['ui'] = dict(state='passed' if info.value.status == 200 and result.get('builds') and not errors else 'failed',
                                            status=info.value.status, builds=result.get('builds_found'), errors=errors)
                        browser.close()
                        if report['ui']['state'] != 'passed': report['errors'].append('Browser scenario failed')
                print('Интерфейс:', report['ui'], flush=True)
        except Exception as error:
            report['errors'].append(type(error).__name__ + ': ' + str(error))
        finally:
            try:
                workers = [p for p in process.children(recursive=True) if p.name().lower() != 'conhost.exe']
            except psutil.Error:
                workers = []
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            for worker in workers:
                try: worker.terminate()
                except psutil.Error: pass
            _, alive = psutil.wait_procs(workers, timeout=5)
            for worker in alive:
                try: worker.kill()
                except psutil.Error: pass
    if args.full:
        print('Запуск полного набора pytest...', flush=True)
        project = run/'project'
        shutil.copytree(ROOT, project, ignore=shutil.ignore_patterns(
            '.git', 'data', '.env', '.env.*', '__pycache__', '.pytest_cache',
            '*.zip', '*.db', '.venv', 'venv'))
        (project/'data').mkdir()
        shutil.copy2(run/'stalzone.db', project/'data'/'stalzone.db')
        with (run/'pytest.log').open('w', encoding='utf8') as log:
            code = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-rs'], cwd=project, stdout=log, stderr=subprocess.STDOUT).returncode
        report['pytest'] = dict(returncode=code, log=str(run/'pytest.log'))
        if code: report['errors'].append('pytest failed')
    (run/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print('Готово. Ошибок:', len(report['errors']), '|', run/'report.json', flush=True)
    if report['errors']:
        for error in report['errors']: print(' -', error)
        raise SystemExit(1)


if __name__ == '__main__':
    main()

from concurrent.futures import ThreadPoolExecutor
import importlib
import json

import pytest


@pytest.fixture
def isolated_main(monkeypatch, tmp_path):
    pytest.importorskip('fastapi')
    from app import database
    monkeypatch.setattr(database, 'init_db', lambda: None)
    main = importlib.import_module('app.main')
    monkeypatch.setattr(main, '__file__', str(tmp_path / 'app' / 'main.py'))
    monkeypatch.setattr(main.Config, 'DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setattr(main, 'check_auction_freshness', lambda *args: {
        'is_fresh': True, 'last_collected': None, 'age_hours': 0})
    return main, tmp_path


def test_parallel_build_requests_keep_every_log(isolated_main, monkeypatch):
    from fastapi.testclient import TestClient
    main, directory = isolated_main
    monkeypatch.setattr(main, 'build_artifact_sets', lambda **kwargs: [])
    client = TestClient(main.app)
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: client.get('/api/builds'), range(16)))
    assert all(response.status_code == 200 for response in responses)
    logs = list((directory / 'data/build_logs').glob('*.json'))
    assert len(logs) == 16
    assert all(json.loads(file.read_text(encoding='utf-8'))['result']['builds_found'] == 0 for file in logs)


def test_parallel_owned_saves_publish_a_whole_record(isolated_main):
    main, directory = isolated_main
    inputs = [[dict(instance_id=str(index), payload='артефакт' * 10000)] for index in range(20)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(main._persist_owned_artifacts, inputs))
    from app._app_state import application_store
    final = application_store(directory / 'data').owned()['artifacts']
    assert final in inputs
    assert not list((directory / 'data').glob('last_owned-*.tmp'))


def test_rejected_search_does_not_replace_saved_owned(isolated_main, monkeypatch):
    from fastapi.testclient import TestClient
    main, directory = isolated_main
    original = [dict(instance_id='saved')]
    main._persist_owned_artifacts(original)
    def invalid(**kwargs):
        raise ValueError('Invalid armor')
    monkeypatch.setattr(main, 'build_artifact_sets', invalid)
    r = TestClient(main.app).get('/api/builds', params={'owned': '[{"instance_id":"rejected"}]'})
    assert r.status_code == 422
    from app._app_state import application_store
    assert application_store(directory / 'data').owned()['artifacts'] == original


@pytest.mark.parametrize('params', [
    {'min_thresholds': '[1]'}, {'max_thresholds': '{"speed_modifier":"5"}'},
    {'min_thresholds': '{"speed_modifier":NaN}'}, {'exclude': '{"a":1}'},
    {'owned': '[{"quality":"oops"}]'}, {'owned': '[{"item_id":"a","quality":"3"}]'},
    {'owned': '[{"item_id":"a","ptn":16}]'}, {'owned': '[{"item_id":"a","bonus_ids":"SPEED_MOD"}]'},
    {'owned': '[{"item_id":"a","art_percent":"130"}]'},
])
def test_invalid_search_input_returns_422_without_saving_owned(isolated_main, params):
    from fastapi.testclient import TestClient
    from app._app_state import application_store
    main, directory = isolated_main
    main._persist_owned_artifacts([dict(item_id='saved')])
    response = TestClient(main.app).get('/api/builds', params=params)
    assert response.status_code == 422
    assert application_store(directory / 'data').owned()['artifacts'] == [dict(item_id='saved')]


def test_budget_sweep_uses_selected_armor_at_every_budget(monkeypatch):
    from app import meta_optimizer
    calls = []
    monkeypatch.setattr(meta_optimizer, 'auto_optimize', lambda **kwargs: calls.append(kwargs) or {'builds': []})
    result = meta_optimizer.budget_sweep(min_budget=1000, max_budget=10000, steps=3,
                                          armor_id='selected', armor_level=7)
    assert len(result['steps']) == 3
    assert len(calls) == 3
    assert all(c['armor_id'] == 'selected' and c['armor_level'] == 7 for c in calls)


def test_budget_sweep_api_passes_armor_and_reports_invalid_profile(isolated_main, monkeypatch):
    from fastapi.testclient import TestClient
    from app import meta_optimizer
    main, _ = isolated_main
    calls = []
    monkeypatch.setattr(meta_optimizer, 'budget_sweep', lambda **kwargs: calls.append(kwargs) or {'steps': []})
    client = TestClient(main.app)
    r = client.get('/api/budget-sweep', params={'armor_id': 'selected', 'armor_level': 7})
    assert r.status_code == 200 and calls[0]['armor_id'] == 'selected' and calls[0]['armor_level'] == 7
    def invalid(**kwargs):
        raise ValueError('Unknown armor')
    monkeypatch.setattr(meta_optimizer, 'budget_sweep', invalid)
    assert client.get('/api/budget-sweep', params={'armor_id': 'unknown'}).status_code == 422

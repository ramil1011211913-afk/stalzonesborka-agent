import importlib
import json
from types import SimpleNamespace

import pytest

from app import builder, database, _min_budget_exact as exact
from app import _min_budget_incumbents as cache
from tests.test_objective_integration import setup_builder, container


def prepare(monkeypatch, containers=None):
    setup_builder(monkeypatch, [({'speed_modifier': 1}, 1)] * 300,
                  containers or [container()])
    monkeypatch.setattr(cache, 'INCUMBENTS', cache.Incumbents())


@pytest.mark.parametrize('timeout,elapsed,expected', [
    (None, 0, 5), (30, 0, 5), (10, 0, 5), (5, 0, 5),
    (3, 0, 3), (.25, 0, .25), (30, 26.25, 3.75),
    (30, 29.9, .1), (30, 30, 0), (30, 31, 0), (0, 0, None),
])
def test_model_time_is_capped_by_remaining_request(monkeypatch, timeout, elapsed, expected):
    prepare(monkeypatch)
    # Only builder's clock is replaced; scipy/pytest keep the real clock.
    ticks = iter([100., 100., 100. + elapsed])
    now = [100.]
    def clock():
        now[0] = next(ticks, now[0])
        return now[0]
    monkeypatch.setattr(builder, 'time', SimpleNamespace(monotonic=clock))
    calls = []
    def solve(*args, **kwargs):
        calls.append(kwargs['seconds'])
        return None, dict(state='infeasible', reason='proved_for_prepared_pool')
    monkeypatch.setattr(exact, 'solve', solve)
    kwargs = {} if timeout is None else {'min_budget_timeout': timeout}
    result = builder.build_artifact_sets(100, optimization_mode='min_budget', **kwargs)
    assert not result
    if expected is None:
        assert not calls
        assert result.search_status['models'][0]['reason'] == 'request_time_limit'
    else:
        assert calls == pytest.approx([expected])


def test_later_container_receives_only_remaining_time(monkeypatch):
    prepare(monkeypatch, [container('a'), container('b')])
    now = [100.]
    monkeypatch.setattr(builder, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    calls = []
    def solve(*args, **kwargs):
        calls.append(kwargs['seconds'])
        if len(calls) == 1:
            now[0] += 27
        return None, dict(state='infeasible', reason='proved_for_prepared_pool')
    monkeypatch.setattr(exact, 'solve', solve)
    builder.build_artifact_sets(100, optimization_mode='min_budget')
    assert calls == [5., 3.]


@pytest.mark.parametrize('timeout', [-1, 30.001, float('nan'), float('inf'), '30'])
def test_builder_rejects_invalid_request_time(timeout):
    with pytest.raises(ValueError, match='between 0 and 30'):
        builder.build_artifact_sets(100, optimization_mode='min_budget', min_budget_timeout=timeout)


@pytest.fixture
def api(monkeypatch, tmp_path):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    monkeypatch.setattr(database, 'init_db', lambda: None)
    main = importlib.import_module('app.main')
    monkeypatch.setattr(main, '__file__', str(tmp_path / 'app/main.py'))
    monkeypatch.setattr(main.Config, 'DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setattr(main, 'check_auction_freshness', lambda *args: dict(
        is_fresh=True, last_collected=None, age_hours=0))
    calls = []
    def search(**kwargs):
        calls.append(kwargs)
        return exact.SearchResults([], [dict(state='unknown', reason='time_limit')])
    monkeypatch.setattr(main, 'build_artifact_sets', search)
    return TestClient(main.app), calls, tmp_path


@pytest.mark.parametrize('timeout', [None, 0, .25, 5, 30])
def test_api_forwards_default_and_explicit_time_and_logs_it(api, timeout):
    client, calls, root = api
    params = {'optimization_mode': 'min_budget'}
    if timeout is not None:
        params['min_budget_timeout'] = timeout
    response = client.get('/api/builds', params=params)
    assert response.status_code == 200
    expected = 30 if timeout is None else timeout
    assert calls[-1]['min_budget_timeout'] == expected
    log = json.loads(next((root / 'data/build_logs').glob('*.json')).read_text(encoding='utf-8'))
    assert log['request']['min_budget_timeout'] == expected
    assert response.json()['search_status']['optimality_proven'] is False


@pytest.mark.parametrize('timeout', [-1, 30.001, 60, 'nan'])
def test_api_rejects_out_of_range_before_search(api, timeout):
    client, calls, _ = api
    response = client.get('/api/builds', params={'min_budget_timeout': timeout})
    assert response.status_code == 422 and not calls


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic'])
def test_ordinary_modes_do_not_use_min_budget_solver(monkeypatch, mode):
    setup_builder(monkeypatch, [({'speed_modifier': 10}, 2), ({'speed_modifier': 8}, 1)], [container()])
    def forbidden(*args, **kwargs):
        raise AssertionError('Price solver must not run in an ordinary mode')
    monkeypatch.setattr(exact, 'solve', forbidden)
    params = dict(budget=100, priorities={'speed_modifier': 10}, optimization_mode=mode)
    if mode == 'lexicographic':
        params['priority_order'] = ['speed_modifier']
    before = builder.build_artifact_sets(**params, min_budget_timeout=0)
    after = builder.build_artifact_sets(**params)
    assert before and [b.to_dict() for b in before] == [b.to_dict() for b in after]

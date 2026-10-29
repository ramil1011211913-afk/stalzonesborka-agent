from itertools import combinations
from types import SimpleNamespace
from unittest.mock import patch
import builtins
import importlib

import numpy as np
import pytest

from app import builder
from app._min_budget_exact import solve, SearchResults
from tests.test_objective_integration import setup_builder, container


def option(i, stats, price):
    return SimpleNamespace(item_id=str(i // 2), physical_id='p'+str(i // 2),
                           stats=stats, buyout_price=price)


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('carry', [False, True])
@pytest.mark.parametrize('duplicates', [False, True])
def test_exact_price_matches_independent_enumeration(seed, carry, duplicates):
    pytest.importorskip('scipy')
    rng = np.random.default_rng(seed)
    c = [option(i, dict(speed_modifier=int(rng.integers(0, 12)),
                        radiation_accumulation=int(rng.integers(-8, 16))), int(rng.integers(1, 10)))
         for i in range(12)]
    feasible = []
    def valid(ids):
        items = [c[i] for i in ids]
        if len({x.physical_id for x in items}) < 3 or (not duplicates and len({x.item_id for x in items}) < 3):
            return False
        price = sum(x.buyout_price for x in items)
        speed = sum(x.stats['speed_modifier'] for x in items) + 2
        rad = sum(x.stats['radiation_accumulation'] for x in items)
        rad = (rad*.4 if rad > 0 else rad) - 3
        counters = sum(x.stats['radiation_accumulation'] < 0 for x in items)
        return price <= 20 and 14 <= speed <= 28 and rad <= (10 if carry else 100) and counters <= 2
    for ids in combinations(range(12), 3):
        if valid(ids):
            feasible.append(sum(c[i].buyout_price for i in ids))
    ids, status = solve(c, 3, 20, 2, .4, {'speed_modifier': 2, 'radiation_accumulation': -3},
                        {'speed_modifier': 14}, {'speed_modifier': 28}, duplicates, carry, valid)
    if feasible:
        assert ids is not None and valid(ids)
        assert sum(c[i].buyout_price for i in ids) == min(feasible)
    else:
        assert ids is None and status['state'] == 'infeasible'


@pytest.mark.parametrize('reason,kwargs', [
    ('time_limit', {'seconds': 0}),
    ('unsupported_transform', {'guaranteed': True}),
    ('unsupported_transform', {'minimum': {'stopping_protection': 5}}),
    ('model_limit', {'protection': -1}),
])
def test_unsupported_or_exhausted_model_is_unknown(reason, kwargs):
    request = dict(candidates=[option(0, {}, 1)], slots=1, budget=2, max_counter=1,
                   protection=1, bonuses={}, minimum={}, maximum={}, duplicates=True,
                   carry=False, validate=lambda _: True)
    request.update(kwargs)
    ids, status = solve(**request)
    assert ids is None and status['state'] == 'unknown' and status['reason'] == reason


def test_missing_solver_is_unknown(monkeypatch):
    native = builtins.__import__
    def no_scipy(name, *args, **kwargs):
        if name.startswith('scipy'):
            raise ImportError('not installed')
        return native(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_scipy)
    ids, status = solve([option(0, {}, 1)], 1, 2, 1, 1, {}, {}, {}, True, False, lambda _: True)
    assert ids is None and status['reason'] == 'solver_unavailable'


@pytest.mark.parametrize('kind', ['timeout', 'error', 'exception', 'invalid', 'incumbent'])
def test_solver_failures_and_incumbents_are_distinguished(monkeypatch, kind):
    optimize = pytest.importorskip('scipy.optimize')
    def fake(*a, **k):
        if kind == 'exception':
            raise RuntimeError('solver failed')
        return SimpleNamespace(status=1 if kind in ('timeout', 'incumbent') else 4,
                               x=np.array([1.]) if kind in ('invalid', 'incumbent') else None)
    monkeypatch.setattr(optimize, 'milp', fake)
    ids, status = solve([option(0, {}, 1)], 1, 2, 1, 1, {}, {}, {}, True, False,
                        lambda _: kind != 'invalid')
    assert (ids is not None) == (kind == 'incumbent')
    assert status['state'] == ('feasible' if kind == 'incumbent' else 'unknown')


@pytest.mark.parametrize('protection', [0., 1.])
def test_signed_danger_and_container_bonus_boundary(protection):
    pytest.importorskip('scipy')
    c = [option(0, {'radiation_accumulation': -4}, 2), option(2, {'radiation_accumulation': 100}, 1)]
    ids, status = solve(c, 1, 3, 1, protection, {'radiation_accumulation': 12}, {}, {},
                        True, True, lambda ids: int(ids[0]) == 0)
    assert ids.tolist() == [0] and status['state'] == 'optimal'


def test_full_protection_zero_upper_bound_preserves_positive_raw_sum():
    pytest.importorskip('scipy')
    c = [option(0, {'radiation_accumulation': 100}, 1)]
    ids, status = solve(c, 1, 2, 1, 0., {}, {}, {'radiation_accumulation': 0},
                        True, False, lambda ids: ids.tolist() == [0])
    assert ids.tolist() == [0] and status['state'] == 'optimal'


def test_auto_visits_small_container_and_keeps_explicit_choice(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 5}, 1)]*8,
                  [container('large', 7), container('small', 1, bonuses={'speed_modifier': 20})])
    result = builder.build_artifact_sets(100, optimization_mode='min_budget', min_thresholds={'speed_modifier': 20})
    assert result and result[0].container.item_id == 'small' and result[0].total_price == 1
    explicit = builder.build_artifact_sets(100, optimization_mode='min_budget', container_id='large')
    assert explicit and all(b.container.item_id == 'large' for b in explicit)


def test_request_deadline_and_pool_infeasibility_never_claim_global_impossibility(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 1}, 1)]*4, [container()])
    result = builder.build_artifact_sets(100, optimization_mode='min_budget', min_budget_timeout=0)
    assert not result and result.search_status['state'] == 'unknown'
    assert result.search_status['infeasibility_proven'] is False
    assert result.search_status['models'][0]['reason'] == 'request_time_limit'
    empty = SearchResults([], [dict(state='infeasible')])
    assert empty.search_status['state'] == 'unknown' and not empty.search_status['infeasibility_proven']


def test_model_limit_cannot_be_bypassed_by_native_fallback(monkeypatch):
    from app import _beam_search as beam
    setup_builder(monkeypatch, [({'speed_modifier': 1}, 1)]*4, [container()])
    def forbidden(*a, **k):
        raise AssertionError('Model resource limit bypassed by Beam')
    monkeypatch.setattr(beam, 'search', forbidden)
    result = builder.build_artifact_sets(100, optimization_mode='min_budget',
                                        min_thresholds={f'constraint_{i}': 1 for i in range(257)})
    assert not result and result.search_status['models'][0]['reason'] == 'model_limit'


def test_api_propagates_unknown_and_logs_status(monkeypatch, tmp_path):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from app import database
    monkeypatch.setattr(database, 'init_db', lambda: None)
    main = importlib.import_module('app.main')
    monkeypatch.setattr(main, '__file__', str(tmp_path/'app/main.py'))
    monkeypatch.setattr(main.Config, 'DATA_DIR', str(tmp_path/'data'))
    monkeypatch.setattr(main, 'check_auction_freshness', lambda *a:dict(is_fresh=True,last_collected=None,age_hours=0))
    monkeypatch.setattr(main, 'build_artifact_sets', lambda **k:SearchResults([], [dict(state='unknown',reason='solver_error')]))
    r = TestClient(main.app).get('/api/builds', params=dict(optimization_mode='min_budget',min_budget_timeout=0))
    assert r.status_code == 200 and r.json()['search_status']['state'] == 'unknown'
    assert r.json()['builds_found'] == 0 and r.json()['search_status']['infeasibility_proven'] is False

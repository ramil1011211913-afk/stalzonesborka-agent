from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from types import SimpleNamespace
import os

import numpy as np
import pytest

from app import _beam_search as beam, builder
from app._objective import Objective
from app._physical_items import is_available
from app._portfolio_search import build_portfolio_sets, search_portfolio
from tests.test_objective_integration import setup_builder, container
from tests.test_armor_search import install_armors


def option(speed, health=0, price=1, item='a', lot=None, radiation=0):
    return SimpleNamespace(item_id=item, buyout_price=price, auction_lot_id=lot,
                           stats={'speed_modifier': speed, 'health_bonus': health,
                                  'radiation_accumulation': radiation})


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic', 'min_budget'])
def test_union_preserves_best_attempt_and_price_ties(monkeypatch, mode):
    candidates = [option(10, price=8), option(10, price=2), option(1, price=5)]
    objective = Objective.create({'speed_modifier': 10}, {'speed_modifier': 10}, mode,
                                 ['speed_modifier'] if mode == 'lexicographic' else None)
    calls = []
    def attempt(*args, **kwargs):
        calls.append((kwargs['beam_width'], kwargs['danger_lookahead'], kwargs['objective']))
        indices = [[0, 1], [2], []][len(calls) - 1]
        return np.ones(len(indices)), np.array(indices, dtype=np.int32).reshape((-1, 1))
    monkeypatch.setattr(beam, 'search', attempt)
    scores, combos = search_portfolio(candidates, 1, 20, 1, 1, {}, {}, {}, {}, 2, objective=objective)
    assert combos[0].tolist() == [1]
    assert len(combos) == 3
    assert [(w, flag) for w, flag, _ in calls] == [(2000, False), (2000, True), (10000, True)]
    assert all(spec is objective for _, _, spec in calls)
    assert scores[0] == objective.evaluate(candidates[1].stats, 2)


def test_lexicographic_bins_then_secondary_stat_then_price(monkeypatch):
    candidates = [option(10.01, health=1, price=1), option(10.02, health=2, price=8),
                  option(10.02, health=2, price=2)]
    objective = Objective.create({'speed_modifier': 10, 'health_bonus': 8}, None,
                                 'lexicographic', ['speed_modifier', 'health_bonus'], 0.1)
    monkeypatch.setattr(beam, 'search', lambda *a, **k: (np.ones(3), np.arange(3).reshape(-1, 1)))
    _, combos = search_portfolio(candidates, 1, 20, 1, 1, {}, {}, {}, {}, 3,
                                objective=objective, max_attempts=1)
    assert combos[:, 0].tolist() == [2, 1, 0]


def test_union_revalidates_physical_budget_min_max_and_carry(monkeypatch):
    candidates = [option(10, price=1, lot=1), option(9, price=1, lot=1),
                  option(8, price=2, lot=2), option(100, price=50, lot=3),
                  option(11, price=1, lot=4, radiation=30), option(0, price=0, lot=5)]
    objective = Objective.create({'speed_modifier': 10})
    combos = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [2, 5]], dtype=np.int32)
    monkeypatch.setattr(beam, 'search', lambda *a, **kw: (np.ones(5), combos))
    _, valid = search_portfolio(candidates, 2, 10, 2, 1, {}, {}, {}, {'speed_modifier': 15}, 5,
                               allow_duplicates=True, max_thresholds={'speed_modifier': 25},
                               objective=objective, max_attempts=1)
    assert valid.tolist() == [[0, 2]]


@pytest.mark.parametrize('limit', [0, 4, -1, True, 1.5, '2'])
def test_invalid_attempt_limit_fails_before_search(limit):
    with pytest.raises(ValueError, match='max_attempts'):
        build_portfolio_sets(10, max_attempts=limit)


def test_non_oom_failures_are_not_hidden(monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError('algorithm failure')
    monkeypatch.setattr(beam, 'search', failure)
    with pytest.raises(RuntimeError, match='algorithm failure'):
        search_portfolio([option(1)], 1, 10, 1, 1, {}, {}, {}, {}, 1)


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('mode', ['balanced', 'lexicographic', 'min_budget'])
@pytest.mark.parametrize('duplicates', [False, True])
def test_small_pool_matches_independent_exhaustive(monkeypatch, backend, mode, duplicates):
    if backend == 'gpu' and not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    monkeypatch.setattr(beam, '_GPU_OK', backend == 'gpu')
    candidates = [option(10, -3, 3, 'a', 1, 15), option(9, 3, 2, 'a', 1, 5),
                  option(7, 4, 3, 'a', 2, 4), option(5, 5, 2, 'b', 3, -8),
                  option(6, 2, 4, 'c', 4, 2), option(3, 8, 1, 'd', 5, 0)]
    objective = Objective.create({'speed_modifier': 10, 'health_bonus': 8},
                                 {'speed_modifier': 20, 'health_bonus': 16}, mode,
                                 ['speed_modifier', 'health_bonus'] if mode == 'lexicographic' else None)
    expected = []
    for pair in combinations(range(len(candidates)), 2):
        selected = [candidates[i] for i in pair]
        stats = {key: sum(c.stats[key] for c in selected) for key in selected[0].stats}
        if stats['radiation_accumulation'] > 0:
            stats['radiation_accumulation'] *= .5
        stats['health_bonus'] += 2
        price = sum(c.buyout_price for c in selected)
        if (price <= 6 and is_available(selected, duplicates) and stats['health_bonus'] >= 0
                and stats['radiation_accumulation'] <= 10 and stats['speed_modifier'] <= 18):
            expected.append((tuple(-v for v in objective.rank(stats, price)) + (price,), pair))
    scores, combos = search_portfolio(candidates, 2, 6, 2, .5, {'health_bonus': 2}, {}, {},
                                     {'health_bonus': 0}, 100, allow_duplicates=duplicates,
                                     max_thresholds={'speed_modifier': 18}, objective=objective)
    assert len(combos) == len(expected)
    ranks = {pair: rank for rank, pair in expected}
    assert set(map(tuple, combos)) == set(ranks)
    assert [ranks[tuple(pair)] for pair in combos] == sorted(ranks.values())


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic', 'min_budget'])
def test_builder_prepares_once_and_shares_objective_across_armor_attempts(monkeypatch, mode):
    setup_builder(monkeypatch, [({'speed_modifier': 10, 'health_bonus': -2}, 2),
                               ({'speed_modifier': 8, 'health_bonus': 1}, 1)], [container()])
    install_armors(monkeypatch, {'bad': {'health_bonus': -10}, 'good': {'health_bonus': 5}})
    loader = builder.load_artifact_options
    loads = []
    monkeypatch.setattr(builder, 'load_artifact_options', lambda **kw: (loads.append(kw) or loader(**kw)))
    specs = []
    implementation = beam.search
    def attempt(*args, **kwargs):
        specs.append(kwargs['objective'])
        return implementation(*args, **kwargs)
    monkeypatch.setattr(beam, 'search', attempt)
    result = build_portfolio_sets(10, armor_ids=['bad', 'good'], priorities={'speed_modifier': 10, 'health_bonus': 8},
                                 min_thresholds={'health_bonus': 0}, optimization_mode=mode,
                                 priority_order=['speed_modifier', 'health_bonus'] if mode == 'lexicographic' else None)
    assert len(loads) == 1
    assert len(result.attempts) == 6
    assert all(spec is specs[0] for spec in specs)
    assert result.builds[0].to_dict()['armor']['id'] == 'good'
    assert result.builds[0].combined_stats['health_bonus'] >= 0


def test_request_local_lookahead_and_concurrent_calls_leave_environment_unchanged(monkeypatch):
    monkeypatch.setattr(beam, '_GPU_OK', False)
    monkeypatch.setenv('STALZONE_DANGER_LOOKAHEAD', '0')
    flags = []
    def impl(*args, **kwargs):
        flags.append(kwargs['danger_lookahead'])
        return np.array([1.]), np.array([[0]], dtype=np.int32)
    monkeypatch.setattr(beam, '_search_impl', impl)
    def request(_):
        reports = []
        scores, combos = search_portfolio([option(1)], 1, 10, 1, 1, {}, {}, {}, {}, 1, attempts=reports)
        return reports
    with ThreadPoolExecutor(4) as executor:
        results = list(executor.map(request, range(8)))
    assert len(flags) == 24 and flags.count(False) == 8 and flags.count(True) == 16
    assert all([r['lookahead'] for r in rows] == [False, True, True] for rows in results)
    assert all([r['pool'] for r in rows] == [0, 0, 0] for rows in results)
    assert os.environ['STALZONE_DANGER_LOOKAHEAD'] == '0'


def test_plain_builder_does_not_opt_into_portfolio(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 1}, 1)] * 105, [container()])
    monkeypatch.setenv('STALZONE_BEAM_WIDTH', '17')
    monkeypatch.setenv('STALZONE_DANGER_LOOKAHEAD', '0')
    widths = []
    implementation = beam.search
    def attempt(*args, **kwargs):
        widths.append(kwargs['beam_width'])
        assert 'danger_lookahead' not in kwargs
        return implementation(*args, **kwargs)
    monkeypatch.setattr(beam, 'search', attempt)
    assert builder.build_artifact_sets(10, priorities={'speed_modifier': 10})
    assert widths == [17]


@pytest.mark.parametrize('limit', [1, 2, 3])
def test_attempt_cap_is_exact_even_when_no_build_is_feasible(monkeypatch, limit):
    calls = []
    def empty(*args, **kwargs):
        calls.append(kwargs['danger_lookahead'])
        return np.zeros(0), np.zeros((0, 1), dtype=np.int32)
    monkeypatch.setattr(beam, 'search', empty)
    reports = []
    scores, combos = search_portfolio([option(1)], 1, 0, 1, 1, {}, {}, {}, {}, 1,
                                     max_attempts=limit, attempts=reports)
    assert len(calls) == len(reports) == limit
    assert len(scores) == 0 and combos.shape == (0, 1)


def test_empty_pool_has_well_formed_result():
    scores, combos = search_portfolio([], 4, 10, 4, 1, {}, {}, {}, {}, 1)
    assert len(scores) == 0 and combos.shape == (0, 4)


def test_cuda_oom_restarts_each_strategy_with_same_local_lookahead(monkeypatch):
    if beam.cp is None:
        pytest.skip('CuPy required for CUDA OOM exception type')
    monkeypatch.setattr(beam, '_GPU_OK', True)
    monkeypatch.setenv('STALZONE_DANGER_LOOKAHEAD', '0')
    calls, cleanups = [], []
    def impl(*args, **kwargs):
        calls.append((kwargs['use_gpu'], kwargs['danger_lookahead'], kwargs['beam_width']))
        if kwargs['use_gpu']:
            raise beam.cp.cuda.memory.OutOfMemoryError(1, 2)
        return np.ones(1), np.zeros((1, 1), dtype=np.int32)
    monkeypatch.setattr(beam, '_search_impl', impl)
    monkeypatch.setattr(beam, '_release_gpu_resources', lambda: cleanups.append(True))
    scores, combos = search_portfolio([option(1)], 1, 10, 1, 1, {}, {}, {}, {}, 1)
    assert calls == [(True, False, 2000), (False, False, 2000),
                     (True, True, 2000), (False, True, 2000),
                     (True, True, 10000), (False, True, 10000)]
    assert len(cleanups) == 3 and combos.tolist() == [[0]]
    assert os.environ['STALZONE_DANGER_LOOKAHEAD'] == '0'

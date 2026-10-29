from itertools import combinations
from types import SimpleNamespace
import numpy as np
import pytest

from app import _beam_search as beam
from app import _min_budget_joint as joint
from app._objective import Objective
from app._physical_items import is_available
from tests.test_quality_search import repair_pool
from tests.test_objective_integration import setup_builder, container
from tests.test_armor_search import install_armors


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_joint_price_health_finds_20_without_widening(monkeypatch, backend):
    if backend == 'gpu' and not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    monkeypatch.setattr(beam, '_GPU_OK', backend == 'gpu')
    candidates = repair_pool()
    spec = Objective.create({'speed_modifier': 10}, {'speed_modifier': 400}, 'min_budget')
    args = (candidates, 4, 100, 4, 1, {}, {}, {'speed_modifier': 10}, {'health_bonus': 100}, 3)
    old, _ = beam.search(*args, objective=spec, beam_width=2000)
    scores, combos = beam.search(*args, objective=spec, beam_width=2000, joint_budget_tail=True)
    skipped, _ = beam.search(*args, objective=spec, beam_width=2000, joint_budget_tail=True, joint_tail_limit=1)
    assert old[0] == skipped[0] == -26
    assert scores[0] == -20 and all(2 <= i < 120 for i in combos[0])
    assert sum(candidates[i].stats['health_bonus'] for i in combos[0]) >= 100


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('slots', [2, 4])
def test_compressed_profiles_preserve_all_possible_tail_prices(monkeypatch, backend, slots):
    if backend == 'gpu' and not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    xp = beam.cp if backend == 'gpu' else np
    profiles = np.array([[12, -3, 4], [1, 15, 8], [-8, 5, 20]], dtype=float)
    raw = profiles[np.arange(30) % 3]
    prices = np.array([1, 4, 9])[np.arange(30) % 3].astype(float)
    counters = (raw[:, 0] < 0).astype(float)
    groups = np.arange(30, dtype=np.int32) // 2
    representatives = joint.tail_representatives(raw, prices, counters, groups, slots)
    assert len(representatives) <= slots * 3
    prefixes = [ids for ids in combinations(range(30), slots - 1) if len(set(groups[list(ids)])) == slots - 1]
    indices = np.array(prefixes, dtype=np.int32)
    stats = raw[indices].sum(axis=1)
    costs = prices[indices].sum(axis=1)
    count = counters[indices].sum(axis=1)
    def complete(values, total):
        radiation = xp.where(values[:, 0] > 0, values[:, 0] * .5, values[:, 0])
        possible = (radiation <= 10) & (values[:, 1] + 3 >= 10) & (values[:, 2] <= 60)
        return xp.where(possible, -total, -xp.inf)
    args = [xp.asarray(v) for v in [stats, costs, indices[:, -1], count, indices,
                                   raw, prices, counters, groups]]
    expected = joint.rank_last_slot(*args, xp.arange(len(raw)), 30, 2, complete, xp)
    actual = joint.rank_last_slot(*args, xp.asarray(representatives), 30, 2, complete, xp)
    if backend == 'gpu':
        expected, actual = xp.asnumpy(expected), xp.asnumpy(actual)
    np.testing.assert_array_equal(actual, expected)


def test_joint_tail_never_combines_price_of_one_candidate_with_stats_of_another():
    raw = np.array([[0, 0], [60, 30], [30, 60]], dtype=float)
    # Each candidate individually fails one constraint; the combined fantasy
    # (60,60) must not be treated as a physically available final artifact.
    def complete(stats, costs):
        return np.where(np.all(stats >= 50, axis=1), -costs, -np.inf)
    result = joint.rank_last_slot(np.array([[0., 0.]]), np.array([1.]), np.array([0]),
        np.array([0.]), np.array([[0]]), raw, np.array([0., 2., 20.]), np.zeros(3),
        np.arange(3), np.array([1, 2]), 100, 1, complete, np)
    assert np.isneginf(result[0])


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('seed', [5, 19, 72])
def test_joint_tail_matches_independent_exhaustive_with_armor_and_nonlinear_limits(monkeypatch, backend, seed):
    if backend == 'gpu' and not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    monkeypatch.setattr(beam, '_GPU_OK', backend == 'gpu')
    rng = np.random.default_rng(seed)
    candidates = [SimpleNamespace(item_id=str(i // 2), physical_id='lot' + str(i // 2),
        buyout_price=int(rng.integers(1, 10)), stats=dict(
            speed_modifier=float(rng.integers(1, 12)), health_bonus=float(rng.integers(-8, 10)),
            radiation_accumulation=float(rng.integers(-8, 16)),
            stopping_protection=float(rng.integers(1, 12)), tear_dmg_factor=15.)) for i in range(10)]
    expected = {}
    for ids in combinations(range(10), 4):
        combo = [candidates[i] for i in ids]
        price = sum(c.buyout_price for c in combo)
        health = sum(c.stats['health_bonus'] for c in combo) + 4
        speed = sum(c.stats['speed_modifier'] for c in combo) + 3
        radiation = sum(c.stats['radiation_accumulation'] for c in combo)
        radiation = radiation * .6 if radiation > 0 else radiation
        stopping = 100 * (1 - 100 / 160 * (1 - sum(c.stats['stopping_protection'] for c in combo) / 100))
        if (is_available(combo, True) and price <= 25 and health >= 0 and speed <= 38
                and radiation <= 10 and 20 <= stopping <= 70
                and sum(c.stats['radiation_accumulation'] < 0 for c in combo) <= 2):
            expected[ids] = -price
    spec = Objective.create({'speed_modifier': 10}, None, 'min_budget')
    scores, combos = beam.search(candidates, 4, 25, 2, .6,
        {'health_bonus': 4, 'speed_modifier': 3}, {}, {'speed_modifier': 10},
        {'health_bonus': 0, 'stopping_protection': 20}, 1000, objective=spec,
        max_thresholds={'speed_modifier': 38, 'stopping_protection': 70},
        allow_duplicates=True, beam_width=2000, joint_budget_tail=True)
    assert set(map(tuple, combos)) == set(expected)
    assert scores.tolist() == sorted(expected.values(), reverse=True)


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic'])
def test_other_modes_are_unchanged_by_joint_tail_flag(monkeypatch, mode):
    monkeypatch.setattr(beam, '_GPU_OK', False)
    candidates = repair_pool()[:5]
    spec = Objective.create({'speed_modifier': 10}, None, mode,
                            ['speed_modifier'] if mode == 'lexicographic' else None)
    args = (candidates, 2, 20, 2, 1, {}, {}, {'speed_modifier': 10}, {}, 20)
    old = beam.search(*args, objective=spec, beam_width=3)
    new = beam.search(*args, objective=spec, beam_width=3, joint_budget_tail=True)
    np.testing.assert_array_equal(old[0], new[0])
    np.testing.assert_array_equal(old[1], new[1])


def test_explicit_builder_shares_objective_and_preserves_armor(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 10, 'health_bonus': -2}, 2),
                               ({'speed_modifier': 8, 'health_bonus': 1}, 1)], [container(slots=2)])
    install_armors(monkeypatch, {'bad': {'health_bonus': -10}, 'good': {'health_bonus': 5}})
    builds = joint.build_joint_budget_sets(10, armor_ids=['bad', 'good'],
         priorities={'speed_modifier': 10}, min_thresholds={'health_bonus': 0}, allow_duplicates=True)
    assert builds and builds[0].total_price == 3
    assert builds[0].to_dict()['armor']['id'] == 'good'


@pytest.mark.parametrize('limit', [0, -1, True, '256'])
def test_invalid_cap_rejected_before_loading_data(limit):
    with pytest.raises(ValueError, match='joint_tail_limit'):
        joint.build_joint_budget_sets(1, joint_tail_limit=limit)


def test_oom_restart_keeps_joint_tail_parameters_and_inputs(monkeypatch):
    if not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    candidates = repair_pool()
    spec = Objective.create({'speed_modifier': 10}, None, 'min_budget')
    args = (candidates, 4, 100, 4, 1, {}, {}, {'speed_modifier': 10}, {'health_bonus': 100}, 3)
    options = dict(objective=spec, beam_width=2000, joint_budget_tail=True)
    with monkeypatch.context() as context:
        context.setattr(beam, '_GPU_OK', False)
        expected = beam.search(*args, **options)
    implementation = joint.rank_last_slot
    injected = []
    def fail(*args, **kwargs):
        if args[-1] is beam.cp and not injected:
            injected.append(True)
            raise beam.cp.cuda.memory.OutOfMemoryError(1, 2)
        return implementation(*args, **kwargs)
    monkeypatch.setattr(joint, 'rank_last_slot', fail)
    actual = beam.search(*args, **options)
    assert injected and actual[0][0] == -20
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])

from itertools import combinations
from types import SimpleNamespace
import numpy as np
import pytest

from app import _beam_search as beam
from app._constraint_lookahead import suffix_extrema, optimistic_constraints
from app._objective import Objective
from app._quality_search import build_quality_sets
from tests.test_objective_integration import setup_builder, container
from tests.test_armor_search import install_armors


def repair_pool():
    rows = [(100, -10, 1)] * 2 + [(35, 30, 5)] * 118 + [(0, 60, 15)] * 2
    return [SimpleNamespace(item_id=f'{i:03}', physical_id=f'p{i}', buyout_price=price,
                            stats={'speed_modifier': speed, 'health_bonus': health})
            for i, (speed, health, price) in enumerate(rows)]


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_future_health_repair_preserves_optimum_lost_by_quality_2000(monkeypatch, backend):
    if backend == 'gpu' and not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    monkeypatch.setattr(beam, '_GPU_OK', backend == 'gpu')
    candidates = repair_pool()
    spec = Objective.create({'speed_modifier': 10}, {'speed_modifier': 400})
    args = (candidates, 4, 100, 4, 1, {}, {'speed_modifier': 400}, {'speed_modifier': 10},
            {'health_bonus': 100}, 3)
    old, _ = beam.search(*args, objective=spec, beam_width=2000, danger_lookahead=True)
    improved, combos = beam.search(*args, objective=spec, beam_width=2000, danger_lookahead=True,
                                  constraint_lookahead=True, expansion_chunk_size=128)
    assert old[0] == 3.5  # Four mediocre, immediately healthy artifacts.
    assert improved[0] == 5
    assert tuple(combos[0]) == (0, 1, 120, 121)
    assert sum(candidates[i].stats['health_bonus'] for i in combos[0]) == 100


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('mode', ['balanced', 'lexicographic', 'min_budget'])
@pytest.mark.parametrize('chunk', [17, 256, 100000])
def test_chunking_preserves_complete_scores_order_and_tie_choices(monkeypatch, backend, mode, chunk):
    if backend == 'gpu' and not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    monkeypatch.setattr(beam, '_GPU_OK', backend == 'gpu')
    rng = np.random.default_rng(734)
    candidates = [SimpleNamespace(item_id=str(i // 2), physical_id='p' + str(i // 2),
                    buyout_price=int(rng.integers(1, 7)), stats=dict(
                        speed_modifier=float(rng.integers(0, 15)), health_bonus=float(rng.integers(-8, 10)),
                        radiation_accumulation=float(rng.integers(-12, 15)),
                        stopping_protection=float(rng.integers(0, 10)), tear_dmg_factor=5.)) for i in range(32)]
    weights = {'speed_modifier': 10, 'health_bonus': 8, 'stopping_protection': 2}
    spec = Objective.create(weights, dict(speed_modifier=60, health_bonus=50, stopping_protection=100), mode,
                            list(weights) if mode == 'lexicographic' else None)
    args = (candidates, 4, 18, 4, .4, {'health_bonus': 3}, {}, weights, {'health_bonus': 0}, 50)
    options = dict(objective=spec, allow_duplicates=True, beam_width=40, danger_lookahead=True,
                   max_thresholds={'speed_modifier': 45})
    scores, combos = beam.search(*args, **options)
    actual_scores, actual_combos = beam.search(*args, expansion_chunk_size=chunk,
                                              constraint_lookahead=mode != 'balanced', **options)
    np.testing.assert_array_equal(actual_scores, scores)
    np.testing.assert_array_equal(actual_combos, combos)


def test_suffix_bounds_enclose_every_distinct_completion_after_protection_and_bonuses():
    rng = np.random.default_rng(882)
    raw = rng.integers(-15, 25, size=(8, 7)).astype(float)
    low, high = suffix_extrema(raw)
    prefix = raw[:2].sum(axis=0)[None, :]
    lower, upper = optimistic_constraints(prefix, np.array([1]), 2, (low, high), .3,
                                          np.array([0, 5]), np.array([-2, 10]), np)
    for ids in combinations(range(2, len(raw)), 2):
        final = prefix + raw[list(ids)].sum(axis=0)
        for d in range(4):
            final[:, d] = np.where(final[:, d] > 0, final[:, d] * .3, final[:, d])
        final[:, 0] -= 2
        final[:, 5] += 10
        assert np.all(lower <= final) and np.all(final <= upper)
    np.testing.assert_array_equal(low[-1], np.zeros(7))
    np.testing.assert_array_equal(high[-1], np.zeros(7))


def test_optimistic_bounds_never_override_final_or_nonlinear_stopping_constraints():
    raw = np.zeros((1, 7))
    options = dict(hp_idx=-1, bul_idx=-1, ehp_w=0, max_ehp=0, prot_mult=1,
                   n_b=0, b_idx=[], b_val=[], tear_idx=6, stop_idx=5, stop_w=1, has_tear=True,
                   m_idx=np.array([5]), m_val=np.array([20]), n_m=1,
                   mx_idx=[], mx_val=[], n_mx=0, primary_idx=np.array([5]),
                   primary_sign=np.array([1]), xp=np)
    score = beam._score_vec(raw, raw, np.zeros(1), False, **options)
    hopeful = (np.full_like(raw, -100000), np.full_like(raw, 100000))
    actual = beam._score_vec(raw, raw, np.zeros(1), False, constraint_stats=hopeful, **options)
    np.testing.assert_array_equal(actual, score)
    final = beam._score_vec(raw, raw, np.zeros(1), False, constraint_stats=hopeful,
                           enforce_constraints=True, **options)
    assert np.isneginf(final[0])


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic', 'min_budget'])
def test_explicit_quality_keeps_armor_and_shared_objective(monkeypatch, mode):
    setup_builder(monkeypatch, [({'speed_modifier': 10, 'health_bonus': -2}, 2),
                               ({'speed_modifier': 8, 'health_bonus': 1}, 1)], [container()])
    install_armors(monkeypatch, {'bad': {'health_bonus': -10}, 'good': {'health_bonus': 5}})
    observed = []
    original = beam.search
    def spy(*args, **kwargs):
        observed.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(beam, 'search', spy)
    builds = build_quality_sets(10, armor_ids=['bad', 'good'], priorities={'speed_modifier': 10, 'health_bonus': 8},
                                 min_thresholds={'health_bonus': 0}, optimization_mode=mode,
                                 priority_order=['speed_modifier', 'health_bonus'] if mode == 'lexicographic' else None)
    assert builds[0].to_dict()['armor']['id'] == 'good'
    assert len(observed) == 2
    assert observed[0]['objective'] is observed[1]['objective']
    assert all(k['danger_lookahead'] and k['constraint_lookahead'] and k['beam_width'] == 2000 for k in observed)


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '100'])
def test_invalid_chunk_size_is_rejected(value):
    with pytest.raises(ValueError, match='expansion_chunk_size'):
        beam.search([], 4, 10, 4, 1, {}, {}, {}, {}, 1, expansion_chunk_size=value)


def test_oom_preserves_chunking_constraint_bounds_and_original_input(monkeypatch):
    if not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    candidates = repair_pool()[:4] + repair_pool()[-2:]
    spec = Objective.create({'speed_modifier': 10}, {'speed_modifier': 400})
    args = (candidates, 4, 100, 4, 1, {}, {}, {'speed_modifier': 10}, {'health_bonus': 100}, 3)
    options = dict(objective=spec, beam_width=2000, danger_lookahead=True,
                   constraint_lookahead=True, expansion_chunk_size=3)
    with monkeypatch.context() as context:
        context.setattr(beam, '_GPU_OK', False)
        expected = beam.search(*args, **options)
    original = beam._score_vec
    triggered = []
    def score(*args, **kwargs):
        if kwargs.get('xp') is beam.cp and not triggered:
            triggered.append(True)
            raise beam.cp.cuda.memory.OutOfMemoryError(1, 2)
        return original(*args, **kwargs)
    monkeypatch.setattr(beam, '_score_vec', score)
    actual = beam.search(*args, **options)
    assert triggered
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])

from types import SimpleNamespace
import numpy as np
import pytest

from app import _beam_search as beam, _min_budget_joint as joint
from tests import test_min_budget_joint as previous


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('slots', [2, 4, 6])
@pytest.mark.parametrize('seed', [17, 83, 121])
def test_suffix_matches_all_continuations_with_conflicts_and_constraints(backend, slots, seed):
    if backend == 'gpu' and not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    xp = beam.cp if backend == 'gpu' else np
    rng = np.random.default_rng(seed)
    profiles = rng.integers(-8, 22, (9, 4)).astype(float)
    raw = profiles[rng.integers(0, len(profiles), 90)]
    prices = rng.integers(0, 20, len(raw)).astype(float)
    counters = (raw[:, 0] < 0).astype(float)
    groups = rng.integers(0, 25, len(raw), dtype=np.int32)
    prefixes = []
    while len(prefixes) < 80:
        ids = np.sort(rng.choice(len(raw), slots - 1, replace=False))
        if len(set(groups[ids])) == slots - 1:
            prefixes.append(ids)
    indices = np.array(prefixes, dtype=np.int32)
    reps = joint.suffix_representatives(raw, prices, counters, groups, slots, len(raw))
    args = [xp.asarray(v) for v in [raw[indices].sum(axis=1), prices[indices].sum(axis=1),
        indices[:, -1], counters[indices].sum(axis=1), indices, raw, prices, counters, groups]]
    def complete(stats, total):
        radiation = xp.where(stats[:, 0] > 0, stats[:, 0] * .3, stats[:, 0]) - 1
        stopping = 100 * (1 - 100 / (100 + stats[:, 2]) * (1 - stats[:, 3] / 100))
        valid = (radiation <= 10) & (stats[:, 1] + 4 >= 5) & (stats[:, 1] <= 55)
        valid &= (stopping >= -10) & (stopping <= 95)
        return xp.where(valid, -total, -xp.inf)
    expected = joint.rank_last_slot(*args, xp.arange(len(raw)), 80, 3, complete, xp)
    actual = joint.rank_last_slot(*args, xp.asarray(reps), 80, 3, complete, xp)
    if backend == 'gpu':
        expected, actual = xp.asnumpy(expected), xp.asnumpy(actual)
    np.testing.assert_array_equal(actual, expected)


def test_k_minus_one_future_groups_do_not_cover_occupied_groups():
    raw = np.zeros((5, 1))
    # Prefix [0,1] occupies B,C; A at 2 remains essential despite cheaper B,C.
    groups = np.array([1, 2, 0, 1, 2])
    reps = joint.suffix_representatives(raw, [1, 1, 9, 1, 1], np.zeros(5), groups, 3)
    assert 2 in reps
    # Same-group cheaper later replacement is always safe when A is available.
    same = joint.suffix_representatives(np.zeros((2, 1)), [9, 1], [0, 0], [0, 0], 3)
    assert same.tolist() == [1]


def test_earlier_cheaper_item_cannot_replace_a_future_tail():
    reps = joint.suffix_representatives(np.zeros((4, 1)), [1, 2, 3, 4],
                                       np.zeros(4), np.arange(4), 2)
    assert reps.tolist() == [0, 1, 2, 3]


def test_counter_and_raw_profiles_are_not_merged():
    raw = np.array([[0.], [1.], [0.], [1.]])
    reps = joint.suffix_representatives(raw, [9, 9, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], 2)
    assert reps.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_1100_candidates_fit_1024_only_with_suffix_selection(monkeypatch, backend):
    if backend == 'gpu' and not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    monkeypatch.setattr(beam, '_GPU_OK', backend == 'gpu')
    candidates = [SimpleNamespace(item_id=str(i), physical_id='lot' + str(i),
        buyout_price=1100-i, stats={'health_bonus': 35.}) for i in range(1100)]
    raw = np.full((1100, 1), 35.)
    prices = np.arange(1100, 0, -1)
    assert joint.tail_representatives(raw, prices, np.zeros(1100), np.arange(1100), 2, 1024) is None
    assert len(joint.suffix_representatives(raw, prices, np.zeros(1100), np.arange(1100), 2, 1024)) == 2
    calls = []
    original = joint.rank_last_slot
    def record(*args, **kwargs):
        calls.append((len(args[9]), args[4].shape[1], len(args[5])))
        return original(*args, **kwargs)
    monkeypatch.setattr(joint, 'rank_last_slot', record)
    scores, combos = beam.search(candidates, 2, 3000, 2, 1, {}, {}, {},
        {'health_bonus': 60}, 1, min_budget=True, beam_width=1,
        allow_duplicates=True, joint_budget_tail=True, joint_tail_frontier=True, joint_tail_limit=1024)
    assert scores.tolist() == [-3.] and sorted(combos[0]) == [1098, 1099]
    assert calls == [(2, 1, 1100)]
    assert len(candidates) == 1100 and candidates[0].buyout_price == 1100


def test_over_limit_skips_whole_refinement(monkeypatch):
    raw = np.zeros((1100, 1))
    assert joint.suffix_representatives(raw, np.arange(1100), np.zeros(1100),
                                        np.arange(1100), 2, 1024) is None
    monkeypatch.setattr(beam, '_GPU_OK', False)
    from tests.test_quality_search import repair_pool
    args = (repair_pool(), 4, 100, 4, 1, {}, {}, {'speed_modifier': 10}, {'health_bonus': 100}, 3)
    baseline = beam.search(*args, min_budget=True)
    def fail(*args, **kwargs):
        raise AssertionError('Over-cap representatives were truncated')
    monkeypatch.setattr(joint, 'rank_last_slot', fail)
    skipped = beam.search(*args, min_budget=True, joint_budget_tail=True,
                          joint_tail_frontier=True, joint_tail_limit=1)
    np.testing.assert_array_equal(baseline[0], skipped[0])
    np.testing.assert_array_equal(baseline[1], skipped[1])


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('seed', [5, 19, 72])
def test_completed_combinations_preserved_with_armor_and_nonlinear_limits(monkeypatch, backend, seed):
    original = beam.search
    def search(*args, **kwargs):
        return original(*args, **dict(kwargs, joint_tail_frontier=True))
    monkeypatch.setattr(beam, 'search', search)
    previous.test_joint_tail_matches_independent_exhaustive_with_armor_and_nonlinear_limits(
        monkeypatch, backend, seed)


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic'])
def test_other_modes_do_not_construct_representatives(monkeypatch, mode):
    def fail(*args, **kwargs):
        raise AssertionError('Suffix selection enabled outside min_budget')
    monkeypatch.setattr(joint, 'suffix_representatives', fail)
    original = beam.search
    def search(*args, **kwargs):
        return original(*args, **dict(kwargs, joint_tail_frontier=True))
    monkeypatch.setattr(beam, 'search', search)
    previous.test_other_modes_are_unchanged_by_joint_tail_flag(monkeypatch, mode)


def test_builder_keeps_armor_and_oom_restarts_with_suffix_flag(monkeypatch):
    original = joint.build_joint_budget_sets
    def build(*args, **kwargs):
        return original(*args, **dict(kwargs, joint_tail_frontier=True))
    monkeypatch.setattr(joint, 'build_joint_budget_sets', build)
    previous.test_explicit_builder_shares_objective_and_preserves_armor(monkeypatch)


def test_oom_restart_keeps_suffix_mode(monkeypatch):
    original = beam.search
    calls = []
    implementation = joint.suffix_representatives
    def record(*args, **kwargs):
        calls.append(True)
        return implementation(*args, **kwargs)
    monkeypatch.setattr(joint, 'suffix_representatives', record)
    def search(*args, **kwargs):
        return original(*args, **dict(kwargs, joint_tail_frontier=True))
    monkeypatch.setattr(beam, 'search', search)
    previous.test_oom_restart_keeps_joint_tail_parameters_and_inputs(monkeypatch)
    assert len(calls) == 3  # reference CPU, failed GPU, full CPU retry


@pytest.mark.parametrize('value', [0, 1, None, 'true'])
def test_invalid_flag_rejected_before_loading_data(value):
    with pytest.raises(ValueError, match='joint_tail_frontier'):
        joint.build_joint_budget_sets(1, joint_tail_frontier=value)

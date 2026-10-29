from itertools import combinations
import numpy as np
import pytest

from app import _beam_search as beam, _min_budget_joint as joint
from app import _min_budget_stream as stream
from tests import test_min_budget_joint as previous


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('block,pairs', [(1, 17), (32, 512), (500, 97)])
@pytest.mark.parametrize('seed', [18, 67])
def test_full_pool_matches_exhaustive_tail_with_bounded_scorer(backend, block, pairs, seed):
    if backend == 'gpu' and not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    xp = beam.cp if backend == 'gpu' else np
    rng = np.random.default_rng(seed)
    raw = rng.integers(-12, 25, (333, 4)).astype(float)
    prices = rng.integers(0, 40, len(raw)).astype(float)
    counters = (raw[:, 0] < 0).astype(float)
    groups = np.arange(len(raw), dtype=np.int32) // 3
    indices = np.array([rng.choice(330, 3, replace=False) for _ in range(43)], dtype=np.int32)
    indices.sort(axis=1)
    args = [xp.asarray(v) for v in [raw[indices].sum(axis=1), prices[indices].sum(axis=1),
        indices[:, -1], counters[indices].sum(axis=1), indices, raw, prices, counters, groups]]
    seen = []
    def complete(stats, total):
        seen.append(len(total))
        danger = xp.where(stats[:, 0] > 0, stats[:, 0] * .3, stats[:, 0])
        stop = (1 - 100 / (100 + stats[:, 2]) * (1 - stats[:, 3] / 100)) * 100
        valid = (danger <= 10) & (stats[:, 1] + 3 >= 10) & (stats[:, 1] <= 50)
        valid &= (stop >= 0) & (stop <= 85)
        return xp.where(valid, -total, -xp.inf)
    expected = joint.rank_last_slot(*args, xp.arange(len(raw)), 100, 2, complete, xp)
    seen.clear()
    actual = stream.rank_last_slot(*args, xp.asarray(np.lexsort((np.arange(len(raw)), prices))),
        100, 2, complete, xp, block_size=block, max_pairs=pairs)
    assert seen and max(seen) <= pairs
    if backend == 'gpu':
        expected, actual = xp.asnumpy(expected), xp.asnumpy(actual)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('seed', [5, 19, 72])
def test_native_exhaustive_armor_nonlinear_and_physical_rules(monkeypatch, backend, seed):
    original = beam.search
    def search(*args, **kwargs):
        kwargs.update(joint_budget_scan=True, joint_scan_max_pairs=37, joint_scan_block=7)
        return original(*args, **kwargs)
    monkeypatch.setattr(beam, 'search', search)
    previous.test_joint_tail_matches_independent_exhaustive_with_armor_and_nonlinear_limits(
        monkeypatch, backend, seed)


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_counterexample_and_oom_keep_stream_options(monkeypatch, backend):
    if backend == 'gpu' and not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    monkeypatch.setattr(beam, '_GPU_OK', backend == 'gpu')
    from tests.test_quality_search import repair_pool
    args = (repair_pool(), 4, 100, 4, 1, {}, {}, {'speed_modifier': 10}, {'health_bonus': 100}, 3)
    options = dict(min_budget=True, joint_budget_scan=True, joint_scan_block=31,
                   joint_scan_max_pairs=8192)
    expected = beam.search(*args, **options)
    assert expected[0][0] == -20
    if backend == 'gpu':
        original = stream.rank_last_slot
        injected = []
        def fail(*args, **kwargs):
            if args[-1] is beam.cp and not injected:
                injected.append(True)
                raise beam.cp.cuda.memory.OutOfMemoryError(1, 2)
            assert kwargs == dict(block_size=31, max_pairs=8192)
            return original(*args, **kwargs)
        monkeypatch.setattr(stream, 'rank_last_slot', fail)
        actual = beam.search(*args, **options)
        assert injected
        np.testing.assert_array_equal(actual[0], expected[0])
        # Backend ties may choose different equally cheap combinations.
        assert actual[0][0] == -20


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic'])
def test_other_modes_do_not_activate_scan(monkeypatch, mode):
    def fail(*args, **kwargs):
        raise AssertionError('min_budget scan activated for another mode')
    monkeypatch.setattr(stream, 'rank_last_slot', fail)
    original = beam.search
    def search(*args, **kwargs):
        return original(*args, **dict(kwargs, joint_budget_scan=True))
    monkeypatch.setattr(beam, 'search', search)
    previous.test_other_modes_are_unchanged_by_joint_tail_flag(monkeypatch, mode)


@pytest.mark.parametrize('name', ['joint_scan_block', 'joint_scan_max_pairs'])
@pytest.mark.parametrize('value', [0, -1, True, '32'])
def test_invalid_limits_before_data_loading(name, value):
    with pytest.raises(ValueError, match=name):
        stream.build_stream_budget_sets(1, **{name: value})


def test_explicit_builder_keeps_armor_selection(monkeypatch):
    monkeypatch.setattr(joint, 'build_joint_budget_sets', stream.build_stream_budget_sets)
    previous.test_explicit_builder_shares_objective_and_preserves_armor(monkeypatch)


def test_early_stop_keeps_equal_price_block_and_skips_expensive_suffix():
    raw = np.arange(1000, dtype=float).reshape(-1, 1)
    prices = np.arange(1000, dtype=float)
    seen = []
    def complete(stats, total):
        seen.extend(stats[:, 0].tolist())
        return np.where(stats[:, 0] >= 2, -total, -np.inf)
    result = stream.rank_last_slot(np.zeros((1, 1)), np.zeros(1), np.array([0]),
        np.zeros(1), np.array([[0]]), raw, prices, np.zeros(1000), np.arange(1000),
        np.arange(1000), 10000, 1, complete, np, block_size=4, max_pairs=8)
    assert result[0] == -2 and max(seen) == 3

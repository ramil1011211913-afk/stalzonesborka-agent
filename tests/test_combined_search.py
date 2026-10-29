from types import SimpleNamespace
import numpy as np
import pytest

from app import _beam_search as beam, _combined_search as combined
from app._objective import Objective
from tests.test_quality_search import repair_pool
from tests.test_objective_integration import setup_builder, container
from tests.test_armor_search import install_armors


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic', 'min_budget'])
def test_policy_preserves_objective_inputs_and_exactly_one_attempt(monkeypatch, mode):
    calls = []
    spec = Objective.create({'health_bonus': 8, 'speed_modifier': 10}, None, mode,
                            ['speed_modifier', 'health_bonus'] if mode == 'lexicographic' else None)
    candidates = repair_pool()
    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return np.array([]), np.empty((0, 4), dtype=np.int32)
    monkeypatch.setattr(combined, 'beam', SimpleNamespace(search=capture))
    combined.search_combined(candidates, 4, 100, 2, .4, {'health_bonus': 3}, {}, {},
        {'health_bonus': 100}, 3, objective=spec, allow_duplicates=True,
        max_thresholds={'speed_modifier': 200}, joint_tail_limit=1024)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is candidates and kwargs['objective'] is spec
    assert kwargs['constraint_lookahead'] == (mode == 'balanced')
    assert kwargs['joint_budget_tail'] == kwargs['joint_tail_frontier'] == (mode == 'min_budget')
    assert kwargs['danger_lookahead'] == (mode != 'min_budget')
    assert not kwargs['joint_budget_scan'] and kwargs['joint_tail_limit'] == 1024
    assert kwargs['max_thresholds'] == {'speed_modifier': 200} and kwargs['allow_duplicates']


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('mode', ['balanced', 'lexicographic', 'min_budget'])
def test_combined_matches_manual_configuration(monkeypatch, backend, mode):
    if backend == 'gpu' and not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    monkeypatch.setattr(beam, '_GPU_OK', backend == 'gpu')
    spec = Objective.create({'speed_modifier': 10}, {'speed_modifier': 400}, mode,
                            ['speed_modifier'] if mode == 'lexicographic' else None)
    args = (repair_pool(), 4, 100, 4, 1, {}, {}, {'speed_modifier': 10}, {'health_bonus': 100}, 3)
    opts = dict(objective=spec, beam_width=2000, joint_tail_limit=256)
    expected = beam.search(*args, **opts, expansion_chunk_size=262144,
        danger_lookahead=mode != 'min_budget', constraint_lookahead=mode == 'balanced',
        joint_budget_tail=mode == 'min_budget', joint_tail_frontier=mode == 'min_budget')
    actual = combined.search_combined(*args, **opts)
    np.testing.assert_array_equal(expected[0], actual[0])
    np.testing.assert_array_equal(expected[1], actual[1])
    if mode == 'balanced':
        assert actual[0][0] == 5
    if mode == 'min_budget':
        assert actual[0][0] == -20


def test_builder_keeps_armor_and_shared_objective(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 10, 'health_bonus': -2}, 2),
                               ({'speed_modifier': 8, 'health_bonus': 1}, 1)], [container(slots=2)])
    install_armors(monkeypatch, {'bad': {'health_bonus': -10}, 'good': {'health_bonus': 5}})
    objectives = []
    original = combined.search_combined
    def search(*args, **kwargs):
        objectives.append(kwargs['objective'])
        return original(*args, **kwargs)
    monkeypatch.setattr(combined, 'search_combined', search)
    result = combined.build_combined_sets(10, optimization_mode='min_budget', min_budget=True,
        armor_ids=['bad', 'good'], priorities={'speed_modifier': 10},
        min_thresholds={'health_bonus': 0}, allow_duplicates=True)
    assert result and result[0].total_price == 3 and result[0].to_dict()['armor']['id'] == 'good'
    assert len(objectives) == 2 and objectives[0] is objectives[1]


@pytest.mark.parametrize('name', ['beam_width', 'expansion_chunk_size', 'joint_tail_limit'])
@pytest.mark.parametrize('value', [0, -1, True, '256'])
def test_invalid_limits_before_loading_data(name, value):
    with pytest.raises(ValueError, match=name):
        combined.build_combined_sets(1, **{name: value})


def test_oom_restarts_combined_configuration(monkeypatch):
    if not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    from app import _min_budget_joint as joint
    args = (repair_pool(), 4, 100, 4, 1, {}, {}, {'speed_modifier': 10}, {'health_bonus': 100}, 3)
    with monkeypatch.context() as context:
        context.setattr(beam, '_GPU_OK', False)
        expected = combined.search_combined(*args, min_budget=True)
    original = joint.rank_last_slot
    injected = []
    def fail(*args, **kwargs):
        if args[-1] is beam.cp and not injected:
            injected.append(True)
            raise beam.cp.cuda.memory.OutOfMemoryError(1, 2)
        return original(*args, **kwargs)
    monkeypatch.setattr(joint, 'rank_last_slot', fail)
    actual = combined.search_combined(*args, min_budget=True)
    assert injected
    np.testing.assert_array_equal(expected[0], actual[0])
    np.testing.assert_array_equal(expected[1], actual[1])

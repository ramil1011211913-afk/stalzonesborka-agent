import inspect

from app import _beam_search


def test_partial_danger_relaxation_is_limited_to_single_objectives():
    source = inspect.getsource(_beam_search._search_impl)

    assert 'set(objective.keys) == {"sprint_speed_modifier"}' in source
    assert 'set(objective.keys) == {"stopping_protection"}' in source
    assert "not min_budget" in source
    assert "ignore_danger_penalty=_relax_partial_danger" in source


def test_default_danger_penalty_is_enabled():
    signature = inspect.signature(_beam_search._score_vec)

    assert signature.parameters[
        "ignore_danger_penalty"
    ].default is False

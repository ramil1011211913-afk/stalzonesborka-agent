from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from app._objective import Objective, derive_scales


def test_health_weight_changes_score_and_choice():
    rows = [{"speed_modifier": 10, "health_bonus": -5},
            {"speed_modifier": 9, "health_bonus": 5}]
    scales = {"speed_modifier": 10, "health_bonus": 10}
    speed = Objective.create({"speed_modifier": 10, "health_bonus": 0}, scales)
    balance = Objective.create({"speed_modifier": 10, "health_bonus": 8}, scales)
    assert speed.evaluate(rows[0]) > speed.evaluate(rows[1])
    assert balance.evaluate(rows[1]) > balance.evaluate(rows[0])


def test_scales_are_immutable_and_zero_stats_have_positive_scale():
    weights = {"health_bonus": 8}
    scales = derive_scales([{"health_bonus": 0}], weights, 4)
    objective = Objective.create(weights, scales)
    weights["health_bonus"] = 0
    scales["health_bonus"] = 0
    assert objective.scales == (1,)
    assert objective.evaluate({"health_bonus": 2}) == 16
    with pytest.raises(FrozenInstanceError):
        objective.scales = (0,)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_explicit_scales_rejected(bad):
    with pytest.raises(ValueError):
        Objective.create({"health_bonus": 1}, {"health_bonus": bad})


def test_scalar_and_matrix_objective_are_identical():
    objective = Objective.create({"health_bonus": 8, "radiation_accumulation": 2},
                                 {"health_bonus": 10, "radiation_accumulation": 5})
    stats = np.array([[5, -2], [-3, 1]])
    keys = ["health_bonus", "radiation_accumulation"]
    scores = objective.matrix_score(stats, keys)
    for row, score in zip(stats, scores):
        assert score == pytest.approx(objective.evaluate(dict(zip(keys, row))))


def test_invalid_direct_construction_cannot_bypass_fixed_scale_contract():
    with pytest.raises(ValueError):
        Objective(('health_bonus',), (8,), (0,))
    with pytest.raises(ValueError):
        derive_scales([{'health_bonus': float('nan')}], ['health_bonus'], 4)


def test_lexicographic_order_and_quantized_equalities():
    objective = Objective.create({"speed_modifier": 10, "health_bonus": 8}, mode="lexicographic",
                                 order=["speed_modifier", "health_bonus"], tolerance=1e-4)
    a = {"speed_modifier": 10.00001, "health_bonus": 1}
    b = {"speed_modifier": 10.00002, "health_bonus": 2}
    c = {"speed_modifier": 10.001, "health_bonus": -100}
    assert objective.rank(c) > objective.rank(b) > objective.rank(a)
    builds = [SimpleNamespace(combined_stats=s, total_price=0) for s in [a, b, c]]
    assert sorted(builds, key=objective.sort_key)[0].combined_stats == c


@pytest.mark.parametrize("order", [None, [], ["health_bonus"], ["speed_modifier", "speed_modifier"]])
def test_lexicographic_requires_explicit_complete_order(order):
    with pytest.raises(ValueError):
        Objective.create({"speed_modifier": 10, "health_bonus": 8}, mode="lexicographic", order=order)

"""Exact-hundredth roll planning for fixed artifact sets.

The planner changes only unknown quality percentages.  Known auction rolls are
facts and must remain byte-for-byte unchanged, while a target roll is expressed
in hundredths (``120.47``), matching the game's roll granularity.
"""
import math

import pytest

from app import builder as b
from app._objective import Objective
from app import _roll_plan as plan


def bracelet(lot_id, *, known=False, percent=130.0, price=4_000_000):
    base = [
        {"prop_id": "health_bonus", "min": 0.425, "max": 0.5, "is_positive": True},
        {"prop_id": "speed_modifier", "min": 1.0200001, "max": 1.2, "is_positive": True},
        {"prop_id": "sprint_speed_modifier", "min": 1.36, "max": 1.6, "is_positive": True},
        {"prop_id": "stamina_regeneration_bonus", "min": 3.06, "max": 3.6, "is_positive": True},
        {"prop_id": "psycho_accumulation", "min": 42.5, "max": 50, "is_positive": False},
    ]
    bonus = [
        {"prop_id": "stamina_bonus", "min": 13.77, "max": 16.2,
         "unlock_ptn": 5, "is_positive": True},
        {"prop_id": "speed_modifier", "min": 0.32, "max": 0.4,
         "unlock_ptn": 10, "is_positive": True},
        {"prop_id": "stamina_regeneration_bonus", "min": 1.02, "max": 1.2,
         "unlock_ptn": 15, "is_positive": True},
    ]
    unlocked = ["STAMINA_REGENERATION", "STAMINA_BONUS", "SPEED_MOD"]
    stats, raw, names, _ = b.compute_artifact_stats(
        base, bonus, unlocked, percent, 15, 115, 15, None, 89.5, 105)
    return b.ArtifactOption(
        item_id="kqgy", name_ru="Браслет", name_en="Bracelet",
        subcategory="bracelets", quality=2, quality_name="Special",
        art_percent=percent, buyout_price=price, ptn=15,
        stats=stats, stats_raw=raw, stats_names=names,
        unlocked_bonus_ids=unlocked, base_stats=base, bonus_stats=bonus,
        stats_random=None, art_percent_known=known, auction_lot_id=lot_id,
    )


def container():
    return b.ContainerOption(
        item_id="dkln", name_ru="Разгрузка ADR-WRBT", name_en="Utility Vest",
        item_type="backpacks", slots=4, buyout_price=0,
        inner_protection=89.5, effectiveness=105.0,
        bonus_stats={
            "speed_modifier": {"value": 6.2, "name_ru": "Скорость передвижения"},
            "max_weight_bonus": {"value": 40.0, "name_ru": "Переносимый вес"},
            "total_speed": {"value": 6.2, "name_ru": "Общая скорость"},
        },
    )


def speed_objective():
    return Objective.create({"total_speed": 10})


def hundredths(percents):
    return all(abs(value * 100.0 - round(value * 100.0)) < 1e-9
               for value in percents)


def test_two_max_bracelets_are_not_carryable_at_baseline():
    left, right = bracelet(1), bracelet(2)
    stats, _, carryable, warnings = plan._evaluate(
        [left, right], [130.0, 130.0], container())
    assert not carryable
    assert warnings
    assert stats["psycho_accumulation"] > b.DANGER_THRESHOLDS[
        "stalker.artefact_properties.factor.psycho_accumulation"]


def test_exact_hundredth_carry_boundary():
    left, right = bracelet(1), bracelet(2)
    at_boundary, _, carryable, _ = plan._evaluate(
        [left, right], [125.23, 125.23], container())
    over_boundary, _, over, _ = plan._evaluate(
        [left, right], [125.24, 125.24], container())
    assert carryable
    assert not over
    assert at_boundary["psycho_accumulation"] <= 10.0
    assert over_boundary["psycho_accumulation"] > 10.0


def test_planner_lowers_only_unknown_percentages():
    left, right = bracelet(1), bracelet(2)
    result = plan.solve([left, right], container(), speed_objective(), carry=True)
    assert result.feasible
    assert result.carryable
    assert hundredths(result.percents)
    assert all(115.0 <= value <= 130.0 for value in result.percents)
    assert result.percents == (130.0, 120.47)
    assert result.objective_value < result.baseline_objective
    assert result.stats["psycho_accumulation"] <= 10.0


def test_known_percentage_stays_actual():
    known = bracelet(3, known=True, percent=127.595884375, price=4_600_000)
    unknown = bracelet(4, price=4_000_000)
    result = plan.solve([known, unknown], container(), speed_objective(), carry=True)
    assert result.feasible
    assert result.carryable
    assert result.percents[0] == 127.595884375
    assert result.choices[0].known
    assert not result.choices[0].changed
    assert hundredths(result.percents[1:])
    assert result.percents[1] <= 122.88


def test_without_carry_constraint_rolls_stay_at_maximum():
    left, right = bracelet(1), bracelet(2)
    result = plan.solve([left, right], container(), speed_objective(), carry=False)
    assert result.feasible
    assert result.percents == (130.0, 130.0)
    assert not result.carryable


def test_max_threshold_is_enforced_by_plan():
    left, right = bracelet(1), bracelet(2)
    result = plan.solve([left, right], container(), speed_objective(),
                        maximum={"psycho_accumulation": 9.0}, carry=False)
    assert result.feasible
    assert result.carryable
    assert result.stats["psycho_accumulation"] <= 9.0 + 1e-9
    assert hundredths(result.percents)
    assert all(value < 130.0 for value in result.percents)


def test_lexicographic_plan_is_carryable_and_exact():
    left, right = bracelet(1), bracelet(2)
    objective = Objective.create(
        {"total_speed": 10, "health_bonus": 5},
        mode="lexicographic", order=["total_speed", "health_bonus"])
    result = plan.solve([left, right], container(), objective, carry=True)
    assert result.feasible
    assert result.carryable
    assert hundredths(result.percents)
    assert result.stats["psycho_accumulation"] <= 10.0


def test_protected_accumulation_objective_is_rejected_as_nonlinear():
    left, right = bracelet(1), bracelet(2)
    objective = Objective.create({"psycho_accumulation": 10})
    result = plan.solve([left, right], container(), objective, carry=True)
    assert not result.feasible
    assert result.status == "unsupported"
    assert "nonlinear" in result.reason


def test_optimize_build_attaches_plan_and_keeps_known_roll():
    known = bracelet(3, known=True, percent=127.595884375, price=4_600_000)
    unknown = bracelet(4, price=4_000_000)
    objective = speed_objective()
    builds = []
    b._process_combo((unknown, known), container(), 8_600_000,
                     {"total_speed": 10}, None, builds, 0,
                     objective=objective, require_carry=False,
                     allow_duplicates=True)
    assert builds and not builds[0].carryable
    result = plan.optimize_build(builds[0], objective, carry=True, duplicates=True)
    assert result is not None
    assert result.carryable
    assert result.roll_plan and result.roll_plan["enabled"]
    assert result.roll_plan["exact_hundredths"]
    assert result.roll_plan["known_percent_preserved"]
    by_lot = {artifact.auction_lot_id: artifact for artifact in result.artifacts}
    assert by_lot[3].art_percent == 127.595884375
    assert not hasattr(by_lot[3], "_roll_plan_percent")
    assert getattr(by_lot[4], "_roll_plan_percent") is not None
    assert math.isclose(by_lot[4].art_percent, getattr(by_lot[4], "_roll_plan_percent"))
    choices = {choice["auction_lot_id"]: choice for choice in result.roll_plan["artifacts"]}
    assert choices[3]["known"] and not choices[3]["variable"]
    assert choices[4]["variable"]
    assert result.roll_plan["carryable"]


def test_api_forwards_roll_optimize(monkeypatch, tmp_path):
    from tests.test_stock_only import api
    main, client = api(monkeypatch, tmp_path)
    seen = []

    def engine(**kwargs):
        seen.append(kwargs)
        return []

    monkeypatch.setattr(main, 'build_artifact_sets', engine)
    response = client.get('/api/builds', params={'budget': 10, 'roll_optimize': 'true'})
    assert response.status_code == 200
    assert seen and seen[0]['roll_optimize'] is True
    assert response.json()['roll_optimize'] is True
    assert response.json()['roll_optimize_baseline'] == 'maximum'


@pytest.mark.parametrize('params', [
    {'roll_optimize': 'true', 'max_quality': 'true'},
    {'roll_optimize': 'true', 'min_budget': 'true'},
    {'roll_optimize': 'true', 'guaranteed_carryable': 'true'},
])
def test_api_rejects_roll_optimize_with_incompatible_modes(monkeypatch, tmp_path, params):
    from tests.test_stock_only import api
    main, client = api(monkeypatch, tmp_path)
    response = client.get('/api/builds', params={'budget': 10, **params})
    assert response.status_code == 422
    assert 'roll_optimize' in response.json()['error']

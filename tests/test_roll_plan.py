"""Exact-hundredth roll planning for fixed artifact sets.

The planner changes only unknown quality percentages.  Known auction rolls are
facts and must remain byte-for-byte unchanged, while a target roll is expressed
in hundredths (``120.47``), matching the game's roll granularity.
"""
import json
import math
import sqlite3
from contextlib import contextmanager

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


def plain_container(slots=2):
    return b.ContainerOption(
        item_id="plain", name_ru="Plain", name_en="Plain",
        item_type="backpacks", slots=slots, buyout_price=0,
        inner_protection=0.0, effectiveness=100.0, bonus_stats={},
    )


def artifact(item_id, lot_id, base, *, percent=145.0, quality=3,
             qlt_min=130, ptn=0, price=1_000_000):
    stats, raw, names, _ = b.compute_artifact_stats(
        base, [], [], percent, ptn, qlt_min, 15, None, 0, 100)
    return b.ArtifactOption(
        item_id=item_id, name_ru=item_id, name_en=item_id,
        subcategory="", quality=quality, quality_name="",
        art_percent=percent, buyout_price=price, ptn=ptn,
        stats=stats, stats_raw=raw, stats_names=names,
        unlocked_bonus_ids=[], base_stats=base, bonus_stats=[],
        stats_random=None, art_percent_known=False, auction_lot_id=lot_id,
    )


def speed_source(lot_id, percent=145.0, contamination=None, health=False):
    base = [{"prop_id": "speed_modifier", "min": 1.0, "max": 1.0,
             "is_positive": True}]
    if contamination is not None:
        low, high = contamination
        base.append({"prop_id": "radiation_accumulation", "min": low, "max": high,
                     "is_positive": False})
    if health:
        base.append({"prop_id": "health_bonus", "min": 1.0, "max": 1.0,
                     "is_positive": True})
    return artifact("speed", lot_id, base, percent=percent)


def counter_artifact(lot_id, percent=145.0):
    return artifact("counter", lot_id, [
        {"prop_id": "max_weight_bonus", "min": 8.0, "max": 10.0, "is_positive": True},
        {"prop_id": "stamina_regeneration_bonus", "min": 1.0, "max": 2.0,
         "is_positive": True},
        {"prop_id": "radiation_accumulation", "min": -30.0, "max": -25.0,
         "is_positive": True},
    ], percent=percent)


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


def test_feasible_set_keeps_zero_objective_counter_at_maximum():
    """A tied counter-artifact must not be rolled down just to fill the gap.

    The counter-artifact has no speed contribution, so the primary objective is
    indifferent to it.  The tie-break must still prefer the maximum roll: the
    minimum max-weight threshold is already satisfied at the baseline.
    """
    speed = speed_source(10)
    counter = counter_artifact(11)
    result = plan.solve(
        [speed, counter], plain_container(), speed_objective(),
        minimum={"max_weight_bonus": 14.0}, carry=True)
    assert result.feasible
    assert result.carryable
    assert result.percents == (145.0, 145.0)
    assert not any(choice.changed for choice in result.choices)
    assert result.objective_value == pytest.approx(result.baseline_objective)
    assert result.stats["max_weight_bonus"] == pytest.approx(14.5)


def test_lexicographic_tie_keeps_zero_objective_counter_at_maximum():
    speed = speed_source(20, health=True)
    counter = counter_artifact(21)
    objective = Objective.create(
        {"total_speed": 10, "health_bonus": 5},
        mode="lexicographic", order=["total_speed", "health_bonus"])
    result = plan.solve(
        [speed, counter], plain_container(), objective,
        minimum={"max_weight_bonus": 14.0}, carry=True)
    assert result.feasible
    assert result.carryable
    assert result.percents == (145.0, 145.0)
    assert not any(choice.changed for choice in result.choices)


def test_contamination_source_is_rolled_down_before_counter_artifact():
    """When the baseline exceeds a danger limit, lower the source, not the counter."""
    speed = speed_source(30, contamination=(50.0, 55.0))
    counter = counter_artifact(31)
    result = plan.solve(
        [speed, counter], plain_container(), speed_objective(),
        minimum={"max_weight_bonus": 14.0}, carry=True)
    assert result.feasible
    assert result.carryable
    assert result.percents[0] < 145.0
    assert result.percents[1] == 145.0
    assert result.stats["radiation_accumulation"] <= 10.0 + 1e-9
    assert result.stats["max_weight_bonus"] == pytest.approx(14.5)


def counter_market(monkeypatch, tmp_path):
    """Two-slot market: a speed source and a tied counter-artifact."""
    path = tmp_path / 'counter.db'
    speed = [{"prop_id": "speed_modifier", "min": 1.0, "max": 1.0,
              "is_positive": True}]
    counter = [
        {"prop_id": "max_weight_bonus", "min": 8.0, "max": 10.0,
         "is_positive": True},
        {"prop_id": "stamina_regeneration_bonus", "min": 1.0, "max": 2.0,
         "is_positive": True},
        {"prop_id": "radiation_accumulation", "min": -30.0, "max": -25.0,
         "is_positive": True},
    ]
    with sqlite3.connect(path) as c:
        c.executescript('''
        CREATE TABLE artefacts(id TEXT PRIMARY KEY,name_ru TEXT,name_en TEXT,subcategory TEXT,
          stats TEXT,bonus_stats TEXT,icon_url TEXT);
        CREATE TABLE auction_history(item_id TEXT,quality INTEGER,price INTEGER,region TEXT);
        CREATE TABLE auction_lots(id INTEGER PRIMARY KEY,item_id TEXT,region TEXT,
          buyout_price INTEGER,quality INTEGER,ptn INTEGER,art_percent REAL,stats_random REAL,
          bonus_properties TEXT,end_time TEXT,additional_raw TEXT,amount INTEGER);
        ''')
        c.execute('INSERT INTO artefacts VALUES(?,?,?,?,?,?,?)',
                  ('speed', 'Speed', 'Speed', '', json.dumps(speed), '[]', ''))
        c.execute('INSERT INTO artefacts VALUES(?,?,?,?,?,?,?)',
                  ('counter', 'Counter', 'Counter', '', json.dumps(counter), '[]', ''))
        for lot_id, item in ((1, 'speed'), (2, 'counter')):
            c.execute('INSERT INTO auction_lots VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                      (lot_id, item, 'EU', 1_000_000, 3, 0, 145.0, 0.0, '[]',
                       '2099-01-01T00:00:00Z', json.dumps({'qlt': 3, 'ptn': 0}), 1))
        c.commit()

    @contextmanager
    def connect():
        c = sqlite3.connect(path)
        c.row_factory = sqlite3.Row
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(b, 'get_connection', connect)
    monkeypatch.setattr(b, 'load_container_options',
                        lambda region: [plain_container()])
    return path


def test_max_quality_keeps_feasible_counter_at_maximum(monkeypatch, tmp_path):
    from tests.test_max_quality import call
    counter_market(monkeypatch, tmp_path)
    result = call(
        budget=10_000_000, priorities={'total_speed': 10},
        min_thresholds={'max_weight_bonus': 14.0},
        carryable_only=True, roll_optimize=True)
    assert result and result.search_status['state'] == 'OPTIMAL', result.search_status
    build = result[0]
    by_item = {artifact.item_id: artifact for artifact in build.artifacts}
    assert by_item['speed'].art_percent == 145.0
    assert by_item['counter'].art_percent == 145.0
    assert build.combined_stats['max_weight_bonus'] == pytest.approx(14.5)
    assert build.roll_plan_search['changed'] is False


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


def test_max_quality_attaches_plan_without_weakening_milp_winner(monkeypatch, tmp_path):
    from tests.test_max_quality import call
    from tests.test_unknown_percent_modes import UNKNOWN_LOT, helium_market
    helium_market(monkeypatch, tmp_path)
    result = call(budget=3_000_000, roll_optimize=True,
                  unknown_percent='minimum')
    assert result and result.search_status['state'] == 'OPTIMAL', result.search_status
    assert result.unknown_percent == 'maximum'
    assert result.search_status['requested_unknown_percent'] == 'minimum'
    assert result.search_status['roll_optimize_baseline'] == 'maximum'
    build = result[0]
    assert build.artifacts[0].auction_lot_id == UNKNOWN_LOT
    assert build.artifacts[0].art_percent == 130.0
    assert build.roll_plan and build.roll_plan['enabled']
    assert build.roll_plan['exact_hundredths']
    assert build.roll_plan['known_percent_preserved']
    assert build.roll_plan_search['mode'] == 'max_quality'
    assert build.roll_plan_search['baseline'] == 'maximum'
    assert build.roll_plan_search['optimized'] == 1
    assert build.roll_plan_search['failed'] == 0
    choice = build.roll_plan['artifacts'][0]
    assert choice['variable'] and not choice['changed']
    assert choice['percent'] == 130.0


def test_max_quality_changed_plan_downgrades_proof_to_feasible(monkeypatch, tmp_path):
    from tests.test_max_quality import call
    from tests.test_unknown_percent_modes import helium_market
    from app import _roll_plan as planner
    helium_market(monkeypatch, tmp_path)
    real = planner.optimize_build

    def changed_plan(*args, **kwargs):
        result, plan = real(*args, **kwargs)
        result.roll_plan['artifacts'][0]['changed'] = True
        return result, plan

    monkeypatch.setattr(planner, 'optimize_build', changed_plan)
    result = call(budget=3_000_000, roll_optimize=True)
    assert result.search_status['state'] == 'FEASIBLE'
    assert result.search_status['reason'] == 'roll_plan_applied'
    assert not result.search_status['optimality_proven']
    assert result[0].roll_plan_search['changed'] is True


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
    {'roll_optimize': 'true', 'min_budget': 'true'},
    {'roll_optimize': 'true', 'guaranteed_carryable': 'true'},
])
def test_api_rejects_roll_optimize_with_incompatible_modes(monkeypatch, tmp_path, params):
    from tests.test_stock_only import api
    main, client = api(monkeypatch, tmp_path)
    response = client.get('/api/builds', params={'budget': 10, **params})
    assert response.status_code == 422
    assert 'roll_optimize' in response.json()['error']


def test_api_forwards_roll_optimize_to_max_quality(monkeypatch, tmp_path):
    from tests.test_stock_only import api
    from app import _max_quality as quality
    main, client = api(monkeypatch, tmp_path)
    seen = []

    def engine(**kwargs):
        seen.append(kwargs)
        return quality.QualityResults(reason='test')

    monkeypatch.setattr(main, 'MAX_QUALITY_RELEASED', True)
    monkeypatch.setattr(quality, 'build_max_quality_sets', engine)
    response = client.get('/api/builds', params={
        'budget': 10, 'max_quality': 'true', 'roll_optimize': 'true'})
    assert response.status_code == 200, response.text
    assert seen and seen[0]['roll_optimize'] is True
    assert response.json()['roll_optimize'] is True
    assert response.json()['unknown_percent'] == 'maximum'

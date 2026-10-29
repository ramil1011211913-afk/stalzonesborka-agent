"""
Regression tests for builder.py v2.0.

Tests cover:
1. Reference data points (9 эталонов stalzone.wiki) — Опал, Бубльгум, stopping
2. ptn=0 vs ptn=15 behavior (no bonus simulation for unresearched)
3. Counter-arts in candidates and carryability
4. Owned artifacts integration
5. Duplicate prevention and allow_duplicates option
6. Budget boundary conditions
7. min_thresholds filtering
8. Container bonus stats applied after protection
9. Rubik (9n7z) exclusion

Run: python -m pytest app/test_builder.py -v
Or:  python -m app.test_builder
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.builder import (
    DANGER_THRESHOLDS,
    ArtifactBuild,
    ArtifactOption,
    ContainerOption,
    _apply_build_container_protection,
    _calc_stat_value,
    _n,
    _recalculate_stopping_protection,
    _recompute_is_positive,
    _score_option,
    _to_full_key,
    build_artifact_sets,
    check_carryability,
    compute_artifact_stats,
    load_artifact_options,
    load_container_options,
)
from app.api_client import QUALITY_RANGES

DB_PATH = Path(__file__).parent.parent / "data" / "stalzone.db"


def _get_artefact(item_id: str) -> tuple[list, list]:
    """Load base and bonus stats for an artefact from the DB."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT stats, bonus_stats FROM artefacts WHERE id=?", (item_id,)
    ).fetchone()
    conn.close()
    assert row is not None, f"Artefact {item_id} not found"
    return json.loads(row["stats"]), json.loads(row["bonus_stats"])


# ─── 1. Reference data points ─────────────────────────────────────────────────

class TestReferencePoints:
    """9 эталонных точек from stalzone.wiki (REBOOT_PROMPT.md §6)."""

    def test_opal_health_bonus(self):
        """Опал +15 92.5% Хитин: health_bonus = -3.2975."""
        base, bonus = _get_artefact("rn1z")
        ids = ["STAMINA_BONUS", "BULLET_DMG_FACTOR", "STOPPING_PROTECTION"]
        stats, _, _, _ = compute_artifact_stats(
            base, bonus, ids, 92.5, 15,
            QUALITY_RANGES[0]["min"], QUALITY_RANGES[0]["range"], None,
            60.0, 115.0,
        )
        assert abs(stats["health_bonus"] - (-3.2975)) < 0.01

    def test_opal_stamina_bonus(self):
        """Опал +15 92.5% Хитин: stamina_bonus = 39.8268 (base + bonus[0])."""
        base, bonus = _get_artefact("rn1z")
        ids = ["STAMINA_BONUS", "BULLET_DMG_FACTOR", "STOPPING_PROTECTION"]
        stats, _, _, _ = compute_artifact_stats(
            base, bonus, ids, 92.5, 15,
            QUALITY_RANGES[0]["min"], QUALITY_RANGES[0]["range"], None,
            60.0, 115.0,
        )
        assert abs(stats["stamina_bonus"] - 39.8268) < 0.01

    def test_opal_bullet_dmg_factor(self):
        """Опал +15 92.5% Хитин: bullet_dmg_factor = 35.6782 (base + bonus[1])."""
        base, bonus = _get_artefact("rn1z")
        ids = ["STAMINA_BONUS", "BULLET_DMG_FACTOR", "STOPPING_PROTECTION"]
        stats, _, _, _ = compute_artifact_stats(
            base, bonus, ids, 92.5, 15,
            QUALITY_RANGES[0]["min"], QUALITY_RANGES[0]["range"], None,
            60.0, 115.0,
        )
        assert abs(stats["bullet_dmg_factor"] - 35.6782) < 0.01

    def test_opal_stopping_protection(self):
        """Опал +15 92.5% Хитин: stopping_protection = 8.9887 (bonus[2])."""
        base, bonus = _get_artefact("rn1z")
        ids = ["STAMINA_BONUS", "BULLET_DMG_FACTOR", "STOPPING_PROTECTION"]
        stats, _, _, _ = compute_artifact_stats(
            base, bonus, ids, 92.5, 15,
            QUALITY_RANGES[0]["min"], QUALITY_RANGES[0]["range"], None,
            60.0, 115.0,
        )
        assert abs(stats["stopping_protection"] - 8.9887) < 0.01

    def test_opal_thermal_accumulation(self):
        """Опал +15 92.5% Хитин: thermal_accumulation = -9.2352."""
        base, bonus = _get_artefact("rn1z")
        ids = ["STAMINA_BONUS", "BULLET_DMG_FACTOR", "STOPPING_PROTECTION"]
        stats, _, _, _ = compute_artifact_stats(
            base, bonus, ids, 92.5, 15,
            QUALITY_RANGES[0]["min"], QUALITY_RANGES[0]["range"], None,
            60.0, 115.0,
        )
        assert abs(stats["thermal_accumulation"] - (-9.2352)) < 0.05

    def test_opal_frost_accumulation(self):
        """Опал +15 92.5% Хитин: frost_accumulation = 19.775."""
        base, bonus = _get_artefact("rn1z")
        ids = ["STAMINA_BONUS", "BULLET_DMG_FACTOR", "STOPPING_PROTECTION"]
        stats, _, _, _ = compute_artifact_stats(
            base, bonus, ids, 92.5, 15,
            QUALITY_RANGES[0]["min"], QUALITY_RANGES[0]["range"], None,
            60.0, 115.0,
        )
        assert abs(stats["frost_accumulation"] - 19.775) < 0.05

    def test_bubblegum_stopping_protection(self):
        """Бубльгум +15 92.5%: stopping_protection recalculated from tear_dmg_factor."""
        base, bonus = _get_artefact("qodk")
        ids = ["BULLET_DMG_FACTOR", "TEAR_DMG_FACTOR", "EXPLOSION_DMG_FACTOR"]
        stats, _, _, _ = compute_artifact_stats(
            base, bonus, ids, 92.5, 15,
            QUALITY_RANGES[0]["min"], QUALITY_RANGES[0]["range"], None,
            60.0, 115.0,
        )
        _recalculate_stopping_protection(stats)
        tear = stats["tear_dmg_factor"]
        expected = (1 - 100 / (100 + tear)) * 100
        assert abs(stats["stopping_protection"] - expected) < 0.001

    def test_stopping_protection_formula(self):
        """stopping_protection = (1 - 100/(100+tear) * (1-stopping/100)) * 100."""
        stats = {"tear_dmg_factor": 50.0, "stopping_protection": 0.0}
        _recalculate_stopping_protection(stats)
        # (1 - 100/150 * 1) * 100 = (1 - 0.6667) * 100 = 33.33
        assert abs(stats["stopping_protection"] - 33.3333) < 0.01

    def test_stopping_protection_with_existing(self):
        """stopping with existing value: tear=100, existing=50 → 66.67."""
        stats = {"tear_dmg_factor": 100.0, "stopping_protection": 50.0}
        _recalculate_stopping_protection(stats)
        # (1 - 100/200 * (1-0.5)) * 100 = (1 - 0.5*0.5) * 100 = 75
        assert abs(stats["stopping_protection"] - 75.0) < 0.01


# ─── 2. ptn=0 vs ptn=15 ──────────────────────────────────────────────────────

class TestPtnBehavior:
    """ptn=0: base stats only, no bonus simulation. ptn>0: unlocked bonuses."""
    def test_ptn0_has_no_unlocked_bonuses(self):
        opts = load_artifact_options(priorities={"health_bonus": 5}, max_price=20_000_000, region="EU")
        ptn0 = [o for o in opts if o.ptn == 0]
        assert len(ptn0) > 0, "Should have ptn=0 options"
        for o in ptn0:
            assert o.unlocked_bonus_ids == [], f"{o.name_ru} ptn=0 has bonuses {o.unlocked_bonus_ids}"

    def test_ptn15_has_unlocked_bonuses(self):
        opts = load_artifact_options(priorities={"health_bonus": 5}, max_price=20_000_000, region="EU")
        ptn15 = [o for o in opts if o.ptn == 15]
        assert len(ptn15) > 0, "Should have ptn=15 options"
        with_bonus = [o for o in ptn15 if o.unlocked_bonus_ids]
        assert len(with_bonus) > 0, "Some ptn=15 options should have unlocked bonuses"

    def test_ptn15_has_more_stats_than_ptn0(self):
        """Same artefact: ptn=15 should have >= stats than ptn=0."""
        opts = load_artifact_options(priorities={"health_bonus": 5}, max_price=20_000_000, region="EU")
        ptn0_ids = {o.item_id: o for o in opts if o.ptn == 0}
        ptn15_ids = {o.item_id: o for o in opts if o.ptn == 15}
        common = set(ptn0_ids) & set(ptn15_ids)
        assert len(common) > 0, "Should have common item_ids"
        cid = next(iter(common))
        assert len(ptn15_ids[cid].stats) >= len(ptn0_ids[cid].stats)


# ─── 3. Counter-arts and carryability ─────────────────────────────────────────

class TestCounterArts:
    """Counter-arts (negative accumulation) appear as auto-candidates, no score bonus."""

    def test_build_contains_counter_arts(self):
        builds = build_artifact_sets(
            budget=8_000_000, tolerance_pct=8.0,
            priorities={"bullet_dmg_factor": 10, "health_bonus": 8},
            container_id="rgoz", top_n=1, carryable_only=True,
        )
        assert len(builds) > 0
        b = builds[0]
        assert b.carryable
        has_counter = False
        for a in b.artifacts:
            for pid, val in a.stats.items():
                fkey = _to_full_key(pid)
                if fkey in DANGER_THRESHOLDS and val < 0:
                    has_counter = True
                    break
        assert has_counter, "Top build should contain counter-arts for carryability"

    def test_all_accumulation_within_thresholds(self):
        builds = build_artifact_sets(
            budget=8_000_000, tolerance_pct=8.0,
            priorities={"bullet_dmg_factor": 10, "health_bonus": 8},
            container_id="rgoz", top_n=3, carryable_only=True,
        )
        for b in builds:
            for pid, val in b.combined_stats.items():
                fkey = _to_full_key(pid)
                if fkey in DANGER_THRESHOLDS:
                    assert val <= DANGER_THRESHOLDS[fkey] + 0.001, \
                        f"{pid}={val} exceeds {DANGER_THRESHOLDS[fkey]}"


# ─── 4. Owned artifacts ───────────────────────────────────────────────────────

class TestOwnedArtifacts:
    """Owned artifacts integrate into builds at zero cost."""

    def test_owned_artifacts_in_build(self):
        """Owned artifacts remain selectable candidates at zero cost."""
        import json

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, stats, bonus_stats FROM artefacts"
        ).fetchall()
        conn.close()

        all_item_ids = [r["id"] for r in rows]
        eligible = []

        for r in rows:
            stats = json.loads(r["stats"] or "[]")
            bonus_stats = json.loads(r["bonus_stats"] or "[]")

            if any(
                s.get("prop_id") == "bullet_dmg_factor"
                for s in stats + bonus_stats
            ):
                eligible.append(r["id"])
                if len(eligible) == 4:
                    break

        assert len(eligible) == 4, (
            "Expected at least four artifacts with bullet_dmg_factor"
        )

        owned = [
            {
                "item_id": item_id,
                "quality": 2,
                "ptn": 15,
                "art_percent": 130.0,
                "bonus_ids": [],
            }
            for item_id in eligible
        ]

        builds = build_artifact_sets(
            budget=5_000_000,
            tolerance_pct=8.0,
            priorities={"bullet_dmg_factor": 10},
            container_id="dkln",
            top_n=1,
            carryable_only=False,
            owned_artifacts=owned,
            exclude_item_ids=all_item_ids,
        )

        assert builds, "Expected a build from four owned artifacts"

        selected = builds[0].artifacts
        assert len(selected) == 4
        assert {a.item_id for a in selected} == set(eligible)
        assert all(a.is_owned for a in selected)
        assert all(a.buyout_price == 0 for a in selected)
    def test_owned_zero_cost(self):
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT id FROM artefacts WHERE id != '9n7z' LIMIT 1").fetchone()
        conn.close()
        owned = [{"item_id": row["id"], "quality": 2, "ptn": 15, "art_percent": 130.0, "bonus_ids": []}]
        builds = build_artifact_sets(
            budget=5_000_000, tolerance_pct=8.0,
            priorities={"bullet_dmg_factor": 10},
            container_id="dkln", top_n=1, carryable_only=True,
            owned_artifacts=owned,
        )
        if builds:
            for a in builds[0].artifacts:
                if a.is_owned:
                    assert a.buyout_price == 0


# ─── 5. Duplicate prevention ─────────────────────────────────────────────────

class TestDuplicates:
    """Default: unique item_id per build. allow_duplicates=True permits them."""

    def test_no_duplicates_by_default(self):
        builds = build_artifact_sets(
            budget=20_000_000, tolerance_pct=8.0,
            priorities={"bullet_dmg_factor": 10},
            container_id="rgoz", top_n=3, carryable_only=True,
        )
        for b in builds:
            ids = [a.item_id for a in b.artifacts]
            assert len(ids) == len(set(ids)), f"Duplicate item_ids in build: {ids}"
    def test_allow_duplicates_when_enabled(self):
        """allow_duplicates=True permits repeated item_id in a build."""
        builds = build_artifact_sets(
            budget=20_000_000,
            tolerance_pct=8.0,
            priorities={"bullet_dmg_factor": 10},
            container_id="rgoz",
            top_n=10,
            carryable_only=True,
            allow_duplicates=True,
        )

        assert builds, "Expected at least one build"
        assert any(
            len([a.item_id for a in b.artifacts])
            != len(set(a.item_id for a in b.artifacts))
            for b in builds
        ), "Expected at least one build with repeated item_id when duplicates are enabled"

# ─── 6. Budget boundaries ────────────────────────────────────────────────────

class TestBudgetBoundaries:
    """Budget edge cases: zero, tiny, no crash."""

    def test_zero_budget_no_crash(self):
        builds = build_artifact_sets(
            budget=0, tolerance_pct=0,
            priorities={"health_bonus": 5},
            container_id="dkln", top_n=1,
        )
        # Should return empty, not crash
        assert isinstance(builds, list)

    def test_tiny_budget(self):
        builds = build_artifact_sets(
            budget=100_000, tolerance_pct=0,
            priorities={"health_bonus": 5},
            container_id="dkln", top_n=1,
        )
        assert isinstance(builds, list)

    def test_builds_within_budget(self):
        budget = 5_000_000
        builds = build_artifact_sets(
            budget=budget, tolerance_pct=8.0,
            priorities={"bullet_dmg_factor": 10},
            container_id="dkln", top_n=3, carryable_only=True,
        )
        effective = int(budget * 1.08)
        for b in builds:
            assert b.total_price <= effective, f"Price {b.total_price} > {effective}"


# ─── 7. min_thresholds ───────────────────────────────────────────────────────

class TestMinThresholds:
    """min_thresholds filter out builds that don't meet minimum stat values."""

    def test_min_threshold_enforced(self):
        builds = build_artifact_sets(
            budget=15_000_000, tolerance_pct=8.0,
            priorities={"bullet_dmg_factor": 10},
            min_thresholds={"bullet_dmg_factor": 50},
            container_id="rgoz", top_n=3, carryable_only=True,
        )
        for b in builds:
            assert b.combined_stats.get("bullet_dmg_factor", 0) >= 50


# ─── 8. Container bonus stats ─────────────────────────────────────────────────

class TestContainerBonus:
    """Container bonus stats (speed, weight) added to combined_stats after protection."""

    def test_container_bonus_applied(self):
        containers = load_container_options("EU")
        # Find a container with bonus_stats (e.g. Разгрузка ADR-WRBT)
        cont = next((c for c in containers if c.bonus_stats), None)
        assert cont is not None, "Should find a container with bonus_stats"
        builds = build_artifact_sets(
            budget=10_000_000, tolerance_pct=8.0,
            priorities={"bullet_dmg_factor": 10},
            container_id=cont.item_id, top_n=1, carryable_only=True,
        )
        if builds:
            b = builds[0]
            for pid, data in cont.bonus_stats.items():
                val = data.get("value", 0) if isinstance(data, dict) else data
                if val:
                    assert pid in b.combined_stats, f"Container bonus {pid} missing from combined_stats"


# ─── 9. Rubik exclusion ───────────────────────────────────────────────────────

class TestRubikExclusion:
    """Рубик (9n7z) must never appear in builds — fully random stats."""

    def test_rubik_not_in_options(self):
        opts = load_artifact_options(priorities={"health_bonus": 5}, max_price=20_000_000, region="EU")
        rubik = [o for o in opts if o.item_id == "9n7z"]
        assert len(rubik) == 0, "Рубик should be excluded from options"

    def test_rubik_not_in_builds(self):
        builds = build_artifact_sets(
            budget=15_000_000, tolerance_pct=8.0,
            priorities={"bullet_dmg_factor": 10},
            container_id="rgoz", top_n=3,
        )
        for b in builds:
            for a in b.artifacts:
                assert a.item_id != "9n7z", "Рубик found in build"


# ─── 10. n() normalization ────────────────────────────────────────────────────

class TestNormalize:
    """_n() swap rule: both negative → max = more negative."""

    def test_both_positive(self):
        max_v, min_v = _n(10, 20, 1.0)
        assert max_v == 20 and min_v == 10

    def test_both_negative_swap(self):
        max_v, min_v = _n(-5, -10, 1.0)
        # both negative: n=min(-5,-10)=-10, o=max(-5,-10)=-5
        assert max_v == -10 and min_v == -5

    def test_mixed_sign(self):
        max_v, min_v = _n(-5, 10, 1.0)
        assert max_v == 10 and min_v == -5

    def test_effectiveness_scaling(self):
        max_v, _ = _n(10, 20, 1.15)  # 115% effectiveness
        assert abs(max_v - 23.0) < 0.001


# ─── 11. Score option ──────────────────────────────────────────────────────────

class TestScoreOption:
    """_score_option: accumulation inverted, no counter-art bonus."""

    def test_positive_stat_positive_score(self):
        score = _score_option({"health_bonus": 10}, {"health_bonus": 5})
        assert score == 50  # 10 * 5

    def test_accumulation_inverted(self):
        # Negative accumulation = protection = good → positive score
        score = _score_option({"radiation_accumulation": -2}, {"radiation_accumulation": 5})
        assert score == 10  # -(-2) * 5

    def test_no_counter_art_bonus(self):
        """No abs(val)*50 bonus — pure weighted sum."""
        score = _score_option({"psycho_accumulation": -3}, {"bullet_dmg_factor": 10})
        # psycho_accumulation has no priority weight → score=0
        assert score == 0

def test_search_danger_thresholds_match_builder():
    from app.builder import DANGER_THRESHOLDS
    from app._numba_search import _DANGER_THRESH as numba_thresholds
    from app._beam_search import _DANGER_THRESH as beam_thresholds

    keys = [
        "stalker.artefact_properties.factor.radiation_accumulation",
        "stalker.artefact_properties.factor.biological_accumulation",
        "stalker.artefact_properties.factor.psycho_accumulation",
        "stalker.artefact_properties.factor.thermal_accumulation",
        "stalker.artefact_properties.factor.frost_accumulation",
    ]
    expected = [DANGER_THRESHOLDS[key] for key in keys]

    assert numba_thresholds.tolist() == expected
    assert beam_thresholds.tolist() == expected

def test_beam_threshold_deficits_are_normalized():
    import numpy as np
    from app._beam_search import _score_vec

    # Two partial candidates each satisfy one requirement:
    # healing >= 2 versus bullet resistance >= 50.
    stats = np.zeros((2, 7), dtype=np.float64)
    stats[0, 5] = 2.0
    stats[1, 6] = 50.0
    empty_i = np.zeros(0, dtype=np.int32)
    empty_f = np.zeros(0, dtype=np.float64)

    scores = _score_vec(
        stats, np.zeros_like(stats), np.zeros(2), False,
        -1, -1, 0.0, 0.0, 1.0,
        0, empty_i, empty_f,
        -1, -1, 0.0, False,
        np.array([5, 6], dtype=np.int32),
        np.array([2.0, 50.0]), 2,
        empty_i, empty_f, 0,
    )

    # Missing 100% of either threshold should receive equal penalty.
    assert np.isclose(scores[0], scores[1])

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

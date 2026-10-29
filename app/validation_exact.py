"""MILP certificate for linear objectives on the production candidate pool.

Only writes to a new output directory. Requires SciPy; run with Python 3.12.
The certificate is for this candidate model, not every physical auction build.
"""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from unittest.mock import patch

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from app import builder, _beam_search
from app._physical_items import availability_groups, is_available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--time-limit", default=60.0, type=float)
    parser.add_argument("--lookahead", choices=["0", "1"], default="1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path("data/validation/exact-test-config.json").read_text(encoding="utf-8-sig"))
    snapshot = Path(config["snapshot"]).resolve()
    wal = Path(str(snapshot) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("Snapshot contains uncheckpointed WAL data")
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()

    def connection():
        conn = sqlite3.connect(snapshot.as_uri() + "?mode=ro&immutable=1", uri=True)
        conn.row_factory = sqlite3.Row
        return closing(conn)

    scenarios = [(name, {key: 10}, {}) for name, key in [
        ("sprint", "sprint_speed_modifier"), ("movement", "speed_modifier"),
        ("health", "health_bonus"), ("bullet", "bullet_dmg_factor"),
        ("stamina", "stamina_bonus")]]
    scenarios.extend([
        ("no_duplicates", {"sprint_speed_modifier": 10}, {"allow_duplicates": False}),
        ("min_budget", {"sprint_speed_modifier": 10},
         {"min_budget": True, "min_thresholds": {"sprint_speed_modifier": 10}}),
        ("max_constraints", {"sprint_speed_modifier": 10},
         {"max_thresholds": {"bullet_dmg_factor": 100, "radiation_accumulation": 5}}),
    ])
    scenarios.append(("balanced_sprint_health", {"sprint_speed_modifier": 10, "health_bonus": 8}, {}))
    report = {"snapshot_sha256": digest, "lookahead": args.lookahead,
              "model": "production candidates; binary per candidate",
              "results": []}
    original_search = _beam_search.search
    for name, priorities, overrides in scenarios:
        captured = []

        def search(*sargs, **skwargs):
            captured.append((sargs, skwargs))
            return original_search(*sargs, **skwargs)

        kwargs = dict(budget=config["budget"], tolerance_pct=0,
                      container_id=config["container_id"], owned_artifacts=config["owned_artifacts"],
                      priorities=priorities, allow_duplicates=True, carryable_only=True,
                      conservative=False, region="EU", top_n=3)
        kwargs.update(overrides)
        with patch.object(builder, "get_connection", connection), patch.object(
            _beam_search, "search", search
        ), patch.dict(os.environ, {"STALZONE_DANGER_LOOKAHEAD": args.lookahead, "STALZONE_BEAM_WIDTH": "10000"}):
            builds = builder.build_artifact_sets(**kwargs)
        assert len(captured) == 1, "Expected exactly one Beam call"
        sargs, skwargs = captured[0]
        candidates, k, budget, max_counter, protection, bonuses = sargs[:6]
        n = len(candidates)
        assert 0 < protection <= 1
        stat = next(iter(priorities))
        spec = skwargs["objective"]
        prices = np.array([c.buyout_price for c in candidates], dtype=float)
        values = np.array([c.stats.get(stat, 0) for c in candidates])
        rows = [np.ones(n), prices,
                np.array([_beam_search._is_counter_art(c) for c in candidates], dtype=float)]
        lower = [k, -np.inf, -np.inf]
        upper = [k, budget, max_counter]

        def add_constraint(key, limit, is_min):
            target = limit - bonuses.get(key, 0)
            if key in _beam_search._DANGER_KEYS[:4] and target > 0:
                target /= protection
            rows.append(np.array([c.stats.get(key, 0) for c in candidates]))
            lower.append(target if is_min else -np.inf)
            upper.append(np.inf if is_min else target)

        for key, threshold in zip(_beam_search._DANGER_KEYS, _beam_search._DANGER_THRESH):
            add_constraint(key, threshold, False)
        for key, limit in kwargs.get("min_thresholds", {}).items():
            add_constraint(key, limit, True)
        for key, limit in kwargs.get("max_thresholds", {}).items():
            add_constraint(key, limit, False)
        # Each physical offer/owned instance contributes at most one hypothesis.
        availability = availability_groups(candidates, kwargs["allow_duplicates"])
        for group in availability:
            row = np.zeros(n)
            row[list(group)] = 1
            rows.append(row)
            lower.append(-np.inf)
            upper.append(1)
        min_budget = kwargs.get("min_budget", False)
        # These controls use linear, non-danger objectives. Rebuild every MILP
        # from the NEW fixed request coefficients, not old raw-stat certificates.
        assert all(key not in _beam_search._DANGER_KEYS and key != "stopping_protection"
                   for key in spec.keys)
        linear_values = sum((coefficient * np.array([c.stats.get(key, 0) for c in candidates])
                             for key, coefficient in zip(spec.keys, spec.coefficients)), np.zeros(n))
        objective_constant = sum(bonuses.get(key, 0) * coefficient
                                 for key, coefficient in zip(spec.keys, spec.coefficients))
        milp_objective = prices if min_budget else -linear_values
        result = milp(c=milp_objective, integrality=np.ones(n), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(csr_matrix(np.vstack(rows)), lower, upper),
                      options={"time_limit": args.time_limit, "mip_rel_gap": 0.0})
        entry = {"name": name, "candidates": n, "objective": spec.metadata(),
                 "availability_constraints": len(availability),
                 "status": int(result.status),
                 "message": result.message, "mip_gap": getattr(result, "mip_gap", None),
                 "dual_bound": getattr(result, "mip_dual_bound", None)}
        if result.x is not None:
            indices = np.flatnonzero(result.x > 0.5)
            combo = [candidates[i] for i in indices]
            assert is_available(combo, kwargs["allow_duplicates"])
            assert is_available(builds[0].artifacts, kwargs["allow_duplicates"])
            price = sum(c.buyout_price for c in combo)
            recalculated = []
            builder._process_combo(combo, builds[0].container, price, priorities,
                                   kwargs.get("min_thresholds", {}), recalculated,
                                   effective_budget=budget, min_budget=min_budget,
                                   max_thresholds=kwargs.get("max_thresholds", {}), objective=spec)
            assert len(indices) == k
            assert recalculated and recalculated[0].carryable
            best = recalculated[0]
            optimum = price if min_budget else spec.evaluate(best.combined_stats)
            beam_value = builds[0].total_price if min_budget else spec.evaluate(builds[0].combined_stats)
            assert np.isclose(best.combined_stats.get(stat, 0),
                              np.sum(values[indices]) + bonuses.get(stat, 0), rtol=0, atol=1e-9)
            if not min_budget:
                assert np.isclose(optimum, linear_values[indices].sum() + objective_constant,
                                  rtol=0, atol=1e-9)
            entry.update({"model_optimum_certified": result.status == 0,
                          "milp_value": optimum, "beam_value": beam_value,
                          "beam_shortfall": beam_value - optimum if min_budget else optimum - beam_value,
                          "price": price, "carryable": best.carryable,
                          "selected_indices": indices.tolist(),
                          "stat_value": best.combined_stats.get(stat, 0),
                          "milp_stats": {key: best.combined_stats.get(key, 0) for key in spec.keys},
                          "beam_stats": {key: builds[0].combined_stats.get(key, 0) for key in spec.keys},
                          "artifacts": [{"id": c.item_id, "physical_id": c.physical_id,
                                         "auction_lot_id": c.auction_lot_id,
                                         "owned_instance_id": c.owned_instance_id, "price": c.buyout_price,
                                         "quality": c.quality, "ptn": c.ptn,
                                         "percent": c.art_percent,
                                         "bonuses": c.unlocked_bonus_ids,
                                         "owned": c.is_owned} for c in combo]})
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        report["results"].append(entry)
        (args.output_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["snapshot_unchanged"] = hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest
    assert report["snapshot_unchanged"]
    (args.output_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

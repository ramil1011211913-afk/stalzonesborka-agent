"""Offline regression against a frozen snapshot; never opens the live database.

Run: python -m app.validation_regression --output-dir <new directory>
Use --baseline-dir <previous output directory> to compare saved source versions.
"""
import argparse
from contextlib import closing, ExitStack
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import types
from unittest.mock import patch

from app import builder, _beam_search
from app._physical_items import is_available


SCENARIOS = [
    ("sprint", {"sprint_speed_modifier": 10}, {}),
    ("movement", {"speed_modifier": 10}, {}),
    ("health", {"health_bonus": 10}, {}),
    ("bullet", {"bullet_dmg_factor": 10}, {}),
    ("stamina", {"stamina_bonus": 10}, {}),
    ("stopping", {"stopping_protection": 10}, {}),
    ("sprint_health", {"sprint_speed_modifier": 10, "health_bonus": 8}, {}),
    ("no_duplicates", {"sprint_speed_modifier": 10}, {"allow_duplicates": False}),
    ("min_budget", {"sprint_speed_modifier": 10},
     {"min_budget": True, "min_thresholds": {"sprint_speed_modifier": 10}}),
    ("min_constraints", {"speed_modifier": 10},
     {"min_thresholds": {"sprint_speed_modifier": 10, "health_bonus": 0}}),
    ("max_constraints", {"sprint_speed_modifier": 10},
     {"max_thresholds": {"bullet_dmg_factor": 100, "radiation_accumulation": 5}}),
]


def load_engine(source, name):
    module = types.ModuleType("app." + name)
    module.__package__ = "app"
    sys.modules[module.__name__] = module
    exec(compile(source, name, "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--current-only", action="store_true", help="Skip the old API for new objective modes")
    parser.add_argument("--backend", choices=["auto", "cpu", "oom"], default="auto",
                        help="Current engines: normal backend, CPU, or one injected allocation OOM per search")
    parser.add_argument("--scenario", choices=[s[0] for s in SCENARIOS])
    parser.add_argument("--stat", action="append", help="Replace the standard matrix with single-stat scenarios")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.backend != "auto" and not args.current_only:
        parser.error("Backend override requires --current-only")
    if args.backend == "oom" and not _beam_search._GPU_OK:
        parser.error("OOM benchmark requires a working CuPy/CUDA backend")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path("data/validation/exact-test-config.json").read_text(encoding="utf-8-sig"))
    snapshot = Path(config["snapshot"]).resolve()
    # immutable avoids WAL/SHM writes; this snapshot has an empty WAL.
    wal = Path(str(snapshot) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("Snapshot has uncheckpointed WAL data")
    digest_before = hashlib.sha256(snapshot.read_bytes()).hexdigest()

    def connection():
        conn = sqlite3.connect(snapshot.as_uri() + "?mode=ro&immutable=1", uri=True)
        conn.row_factory = sqlite3.Row
        return closing(conn)

    with connection() as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    stable_source = subprocess.check_output(
        ["git", "show", "371a55c:app/_beam_search.py"], text=True, encoding="utf-8")
    stable_builder_source = subprocess.check_output(
        ["git", "show", "371a55c:app/builder.py"], text=True, encoding="utf-8")
    stable_builder = load_engine(stable_builder_source, "_validation_stable_builder")
    current_source = Path("app/_beam_search.py").read_text(encoding="utf-8")
    (args.output_dir / "stable_beam.py").write_text(stable_source, encoding="utf-8")
    (args.output_dir / "stable_builder.py").write_text(stable_builder_source, encoding="utf-8")
    (args.output_dir / "current_beam.py").write_text(current_source, encoding="utf-8")
    (args.output_dir / "current_builder.py").write_text(Path("app/builder.py").read_text(encoding="utf-8"), encoding="utf-8")
    engines = [("stable_371a55c", load_engine(stable_source, "_validation_stable"), "0")]
    if args.baseline_dir:
        original = (args.baseline_dir / "current_beam.py").read_text(encoding="utf-8")
        engines.append(("original_joint", load_engine(original, "_validation_original"), "1"))
    engines.extend([
        ("current_flag_off", load_engine(current_source, "_validation_off"), "0"),
        ("current_joint", load_engine(current_source, "_validation_joint"), "1"),
    ])
    if args.current_only:
        engines = [entry for entry in engines if entry[0].startswith("current_")]
    report = {
        "snapshot": str(snapshot), "snapshot_sha256": digest_before,
        "python": sys.version, "gpu_available": _beam_search._GPU_OK,
        "repeats": args.repeats, "backend": args.backend, "scenarios": [],
    }
    with patch.object(builder, "get_connection", connection), patch.object(stable_builder, "get_connection", connection):
        # Load/warm the DB and scalar stat pipeline before measuring engines.
        builder.load_artifact_options(region="EU")
        scenarios = ([(stat, {stat: 10}, {}) for stat in args.stat]
                     if args.stat else SCENARIOS)
        for name, priorities, overrides in scenarios:
            if args.scenario and name != args.scenario:
                continue
            scenario = {"name": name, "priorities": priorities, "overrides": overrides, "runs": []}
            for mode, engine, flag in engines:
                search_calls = []

                def search(*sargs, **skwargs):
                    call = {"candidates": len(sargs[0]), "slots": sargs[1],
                            "beam_width": skwargs["beam_width"], "attempts": [], "oom_injected": False}
                    search_calls.append(call)
                    with ExitStack() as stack:
                        if hasattr(engine, "_search_impl"):
                            implementation = engine._search_impl

                            def attempt(*values, **options):
                                call["attempts"].append("gpu" if options["use_gpu"] else "cpu")
                                return implementation(*values, **options)

                            stack.enter_context(patch.object(engine, "_search_impl", attempt))
                        if args.backend == "cpu":
                            stack.enter_context(patch.object(engine, "_GPU_OK", False))
                        elif args.backend == "oom":
                            asarray = engine.cp.asarray

                            def allocation(*values, **options):
                                if not call["oom_injected"]:
                                    call["oom_injected"] = True
                                    raise engine.cp.cuda.memory.OutOfMemoryError(1, 2)
                                return asarray(*values, **options)

                            stack.enter_context(patch.object(engine.cp, "asarray", allocation))
                        return engine.search(*sargs, **skwargs)

                kwargs = dict(budget=config["budget"], tolerance_pct=0,
                              container_id=config["container_id"], region="EU",
                              owned_artifacts=config["owned_artifacts"],
                              priorities=priorities, allow_duplicates=True,
                              carryable_only=True, conservative=False, top_n=3)
                kwargs.update(overrides)
                timings = []
                with patch.object(_beam_search, "search", search), patch.dict(
                    os.environ, {"STALZONE_DANGER_LOOKAHEAD": flag, "STALZONE_BEAM_WIDTH": "10000"}
                ):
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        build_module = stable_builder if mode in ("stable_371a55c", "original_joint") else builder
                        builds = build_module.build_artifact_sets(**kwargs)
                        timings.append(time.perf_counter() - start)
                violations = []
                summaries = []
                for b in builds:
                    if len(b.artifacts) != b.container.slots:
                        violations.append("incomplete build")
                    if b.total_price > config["budget"]:
                        violations.append("over budget")
                    if not b.carryable:
                        violations.append("uncarryable")
                    if mode.startswith("current_") and not is_available(b.artifacts, kwargs["allow_duplicates"]):
                        violations.append("physical availability")
                    if not kwargs["allow_duplicates"] and len({a.item_id for a in b.artifacts}) != len(b.artifacts):
                        violations.append("duplicate item_id")
                    for key, val in kwargs.get("min_thresholds", {}).items():
                        if b.combined_stats.get(key, 0) < val:
                            violations.append("min " + key)
                    for key, val in kwargs.get("max_thresholds", {}).items():
                        if b.combined_stats.get(key, 0) > val:
                            violations.append("max " + key)
                    summaries.append({
                        "stats": {key: b.combined_stats.get(key, 0) for key in priorities},
                        "constraint_stats": {key: b.combined_stats.get(key, 0) for key in
                                             set(kwargs.get("min_thresholds", {})) | set(kwargs.get("max_thresholds", {}))},
                        "price": b.total_price, "carryable": b.carryable,
                        "priority_value": b.priority_value,
                        "objective": getattr(b, "objective_config", {}),
                        "armor": b.to_dict().get("armor") if hasattr(b, "to_dict") else None,
                        "items": [{"id": a.item_id, "physical_id": getattr(a, "physical_id", None),
                                   "auction_lot_id": getattr(a, "auction_lot_id", None),
                                   "owned_instance_id": getattr(a, "owned_instance_id", None),
                                   "price": a.buyout_price, "quality": a.quality,
                                   "ptn": a.ptn, "percent": a.art_percent,
                                   "bonuses": a.unlocked_bonus_ids, "owned": a.is_owned} for a in b.artifacts],
                    })
                run = {"mode": mode, "times_seconds": timings, "search_calls": search_calls,
                       "violations": violations, "builds": summaries}
                scenario["runs"].append(run)
                best = summaries[0] if summaries else None
                print(name, mode, "best=", best["stats"] if best else None,
                      "price=", best["price"] if best else None,
                      "times=", [round(t, 3) for t in timings], "violations=", violations, flush=True)
            report["scenarios"].append(scenario)
            (args.output_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["snapshot_unchanged"] = hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest_before
    assert report["snapshot_unchanged"]
    (args.output_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

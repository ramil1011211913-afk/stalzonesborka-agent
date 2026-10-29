"""Independent MILP benchmark for one container and one priority stat.

Reads a frozen SQLite snapshot. Does not modify the main optimizer or database.
"""
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from scipy.optimize import milp, Bounds, LinearConstraint

from app.builder import (
    ArtifactOption,
    ContainerOption,
    QUALITY_RANGES,
    QUALITY_NAMES,
    EXCLUDED_ARTIFACTS,
    compute_artifact_stats,
    _process_combo,
)

import sqlite3
from app._physical_items import availability_groups, owned_identity


CONFIG_PATH = Path("data/validation/exact-test-config.json")
STAT = "sprint_speed_modifier"

DANGER = {
    "radiation_accumulation": 10.0,
    "psycho_accumulation": 10.0,
    "thermal_accumulation": 10.0,
    "biological_accumulation": 10.0,
    "frost_accumulation": 20.0,
}


def parse_json(value, default):
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def make_option(art, quality, ptn, pct, bonuses, price, container,
                owned=False, stats_random=None, auction_lot_id=None,
                owned_instance_id=None, physical_id=None):
    qr = QUALITY_RANGES.get(quality, QUALITY_RANGES[0])
    base = parse_json(art["stats"], [])
    bonus_stats = parse_json(art["bonus_stats"], [])

    stats, raw, names, _ = compute_artifact_stats(
        base, bonus_stats, bonuses,
        pct, ptn, qr["min"], qr["range"],
        stats_random,
        container.inner_protection,
        container.effectiveness,
    )

    return ArtifactOption(
        item_id=art["id"],
        name_ru=art["name_ru"],
        name_en=art["name_en"] or "",
        subcategory=art["subcategory"],
        quality=quality,
        quality_name=QUALITY_NAMES.get(quality, "?"),
        art_percent=pct,
        buyout_price=price,
        ptn=ptn,
        stats=stats,
        stats_raw=raw,
        stats_names=names,
        unlocked_bonus_ids=bonuses,
        base_stats=base,
        bonus_stats=bonus_stats,
        stats_random=stats_random,
        is_owned=owned,
        auction_lot_id=auction_lot_id, owned_instance_id=owned_instance_id,
        physical_id=physical_id,
    )


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    snapshot = Path(config["snapshot"])
    if not snapshot.is_file():
        raise SystemExit(f"Snapshot not found: {snapshot}")
    wal = Path(str(snapshot) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("Snapshot contains uncheckpointed WAL data")

    budget = int(config["budget"])
    owned = config["owned_artifacts"]

    # Open the frozen database read-only.
    connection = sqlite3.connect(
        snapshot.resolve().as_uri() + "?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row

    try:
        check = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if check != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {check}")

        container_row = connection.execute(
            "SELECT * FROM containers WHERE id=?",
            (config["container_id"],),
        ).fetchone()

        if container_row is None:
            raise RuntimeError("Container not found in snapshot")

        container = ContainerOption(
            item_id=container_row["id"],
            name_ru=container_row["name_ru"],
            name_en=container_row["name_en"] or "",
            item_type=container_row["item_type"],
            slots=container_row["slots"],
            buyout_price=0,
            inner_protection=container_row["inner_protection"],
            effectiveness=container_row["effectiveness"],
            bonus_stats=parse_json(container_row["bonus_stats"], {}),
            icon_url=container_row["icon_url"] or "",
        )

        artefacts = {
            row["id"]: row
            for row in connection.execute("SELECT * FROM artefacts")
        }

        lots = connection.execute("""
            SELECT *
            FROM auction_lots
            WHERE region='EU'
              AND buyout_price > 0
              AND buyout_price <= ?
        """, (budget,)).fetchall()

        candidates = []
        missing_bonus_info = 0

        for lot in lots:
            art = artefacts.get(lot["item_id"])
            if art is None or art["id"] in EXCLUDED_ARTIFACTS:
                continue

            quality = int(lot["quality"] or 0)
            ptn = int(lot["ptn"] or 0)
            qr = QUALITY_RANGES.get(quality, QUALITY_RANGES[0])

            if config.get("conservative"):
                pct = qr["min"]
            elif lot["art_percent"]:
                pct = float(lot["art_percent"])
            else:
                pct = qr["min"] + qr["range"]

            bonuses = parse_json(lot["bonus_properties"], [])
            if not isinstance(bonuses, list):
                bonuses = []

            if ptn > 0 and not bonuses and parse_json(art["bonus_stats"], []):
                missing_bonus_info += 1

            candidates.append(make_option(
                art, quality, ptn, pct, bonuses,
                int(lot["buyout_price"]),
                container,
                stats_random=lot["stats_random"],
                auction_lot_id=lot["id"], physical_id=f"auction:EU:{lot['id']}",
            ))

        for own_index, own in enumerate(owned):
            art = artefacts.get(own["item_id"])
            if art is None:
                raise RuntimeError(f"Owned artifact missing: {own['item_id']}")

            quality = int(own["quality"])
            qr = QUALITY_RANGES.get(quality, QUALITY_RANGES[0])

            instance_id, physical_id = owned_identity(own, own_index)
            candidates.append(make_option(
                art,
                quality,
                int(own["ptn"]),
                float(own.get("art_percent", qr["min"])),
                own.get("bonus_ids", []),
                0,
                container,
                owned=True,
                owned_instance_id=instance_id, physical_id=physical_id,
            ))

    finally:
        connection.close()

    n = len(candidates)
    if n < container.slots:
        raise RuntimeError("Not enough candidates")

    prices = np.array([o.buyout_price for o in candidates], dtype=float)
    speeds = np.array([o.stats.get(STAT, 0.0) for o in candidates])

    rows = [
        np.ones(n),
        prices,
    ]
    lower = [float(container.slots), -np.inf]
    upper = [float(container.slots), float(budget)]

    # Accumulation constraints.
    # Container protection applies to positive accumulated danger.
    protection = 1.0 - container.inner_protection / 100.0

    for key, threshold in DANGER.items():
        values = np.array([
            o.stats.get(key, 0.0) for o in candidates
        ])

        bonus_data = container.bonus_stats.get(key, 0.0)
        bonus = (
            bonus_data.get("value", 0.0)
            if isinstance(bonus_data, dict)
            else bonus_data
        )

        factor = 1.0 if key == "frost_accumulation" else protection

        rows.append(values)
        lower.append(-np.inf)

        if factor > 0:
            upper.append((threshold - bonus) / factor)
        else:
            upper.append(np.inf)

    allow_duplicates = config.get("allow_duplicates", True)
    for group in availability_groups(candidates, allow_duplicates):
        row = np.zeros(n)
        row[list(group)] = 1
        rows.append(row)
        lower.append(-np.inf)
        upper.append(1)
    A = sparse.csr_matrix(np.vstack(rows))

    print("\n=== EXACT VALIDATOR ===")
    print("Snapshot:", snapshot)
    print("Container:", container.name_ru)
    print("Slots:", container.slots)
    print("Budget:", budget)
    print("Auction lots:", n - len(owned))
    print("Owned artifacts:", len(owned))
    print("Lots with unspecified researched bonuses:", missing_bonus_info)

    result = milp(
        c=-speeds,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(A, lower, upper),
        options={
            "time_limit": 300,
            "mip_rel_gap": 0.0,
        },
    )

    print("\nSolver status:", result.status)
    print("Solver message:", result.message)

    if result.x is None:
        print("No feasible solution returned.")
        return

    indices = np.flatnonzero(result.x > 0.5)
    combo = [candidates[int(i)] for i in indices]
    price = sum(o.buyout_price for o in combo)

    builds = []
    _process_combo(
        combo,
        container,
        price,
        {STAT: 10},
        {},
        builds,
        effective_budget=budget,
        allow_duplicates=allow_duplicates,
    )

    if not builds:
        print("WARNING: Game calculation rejected MILP solution.")
        return

    build = builds[0]
    actual = build.combined_stats.get(STAT, 0.0)

    print("\n=== BEST MILP RESULT ===")
    print("MILP speed:", round(float(speeds[indices].sum()), 6))
    print("Recalculated speed:", round(actual, 6))
    print("Price:", price)
    print("Carryable:", build.carryable)
    print("Reference:", config["beam_reference"])
    print("Difference:", round(actual - config["beam_reference"], 6))

    for o in combo:
        print(
            f"  {o.name_ru}: {o.art_percent:.3f}% "
            f"+{o.ptn}, price={o.buyout_price}, "
            f"speed={o.stats.get(STAT, 0):.6f}, "
            f"owned={o.is_owned}"
        )

    if hasattr(result, "mip_gap"):
        print("MIP gap:", result.mip_gap)

    if (
        result.status == 0
        and build.carryable
        and abs(float(speeds[indices].sum()) - actual) < 1e-4
    ):
        print("\nMODEL OPTIMUM: VERIFIED")
        print("Full game optimum: NOT YET CERTIFIED")
    else:
        print("\nMODEL OPTIMUM: NOT VERIFIED")


if __name__ == "__main__":
    main()

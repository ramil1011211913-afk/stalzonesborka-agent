"""
Meta-build optimizer: auto-selects containers and finds the best builds
for ANY budget — from 10K to billions.

Architecture:
  auto_optimize(budget, style)  → tries top-N containers, returns Pareto frontier
  budget_sweep(min, max, steps) → runs at logarithmic budget points, shows progression
  pre_compute_meta()            → overnight batch for all tiers × playstyles
  load_meta_cache()             → instant lookup from pre-computed cache

Playstyle presets: pvp, tank, speed, heal, balanced, carry.
Each preset maps prop_ids to priority weights (0-10).
"""
import json
import logging
import math
import os
import time

from .config import Config
from .database import get_connection
from .builder import build_artifact_sets
from ._physical_items import AVAILABILITY_CONTRACT

log = logging.getLogger(__name__)

# ───────────────────────── Playstyle presets ─────────────────────────
# prop_id → priority weight (0-10). 10 = critical, 0 = ignore.
PLAYSTYLES: dict[str, dict[str, float]] = {
    "pvp": {
        "bullet_dmg_factor": 10,
        "tear_dmg_factor": 10,
        "health_bonus": 7,
        "speed_modifier": 5,
    },
    "tank": {
        "health_bonus": 10,
        "bullet_dmg_factor": 10,
        "tear_dmg_factor": 5,
    },
    "speed": {
        "speed_modifier": 10,
        "stamina_bonus": 8,
        "health_bonus": 5,
    },
    "heal": {
        "heal_efficiency": 10,
        "regeneration_bonus": 10,
        "health_bonus": 5,
    },
    "balanced": {
        "health_bonus": 8,
        "bullet_dmg_factor": 8,
        "speed_modifier": 5,
        "heal_efficiency": 5,
        "tear_dmg_factor": 3,
    },
    "carry": {
        "max_weight_bonus": 10,
        "health_bonus": 5,
        "speed_modifier": 3,
    },
}

# ───────────────────────── Budget tiers ─────────────────────────
BUDGET_TIERS: list[int] = [
    10_000, 50_000, 100_000, 500_000,
    1_000_000, 5_000_000, 10_000_000, 25_000_000,
    50_000_000, 100_000_000, 250_000_000, 500_000_000,
]

# ───────────────────────── Container scoring ─────────────────────────

def _parse_container_bonuses(bonus_json) -> dict[str, float]:
    """Parse container bonus_stats JSON → {prop_id: value}."""
    if not bonus_json:
        return {}
    try:
        d = json.loads(bonus_json) if isinstance(bonus_json, str) else bonus_json
        return {k: v.get("value", 0) for k, v in d.items()}
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {}


def _score_container(
    container_row: dict, priorities: dict[str, float]
) -> float:
    """Score container suitability based on bonus magnitude × priority weight.

    Unlike the existing builder.py sort (which only checks has_prio_bonus boolean),
    this weighs the actual bonus value: КРАБ (+23 bullet) scores higher than
    a container with +5 bullet when bullet_dmg_factor is prioritized.
    """
    bonuses = _parse_container_bonuses(container_row.get("bonus_stats"))
    score = 0.0

    # Bonus alignment: bonus value × priority weight
    for prop, weight in priorities.items():
        if prop in bonuses:
            score += abs(bonuses[prop]) * weight * 0.3

    # Base container quality: more slots + higher protection + higher effectiveness
    score += container_row.get("slots", 0) * 3.0
    score += container_row.get("inner_protection", 0) * 0.02
    score += container_row.get("effectiveness", 0) * 0.02

    return score


def get_best_containers(
    priorities: dict[str, float] | None = None,
    top_n: int = 5,
    min_slots: int = 3,
) -> list[dict]:
    """Get top-N containers ranked by priority-aligned bonus scoring.

    Args:
        priorities: {prop_id: weight} — what stats matter
        top_n: Max containers to return
        min_slots: Minimum container slots (filter out tiny containers)

    Returns:
        List of container dicts with 'id', 'name', 'bonuses', 'score'
    """
    priorities = priorities or PLAYSTYLES["balanced"]
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name_ru, base_price, slots, inner_protection, "
        "effectiveness, bonus_stats FROM containers "
        "WHERE slots >= ? ORDER BY slots DESC, inner_protection DESC",
        (min_slots,),
    ).fetchall()

    scored = []
    for r in rows:
        d = dict(r)
        d["score"] = _score_container(d, priorities)
        d["bonuses"] = _parse_container_bonuses(d.get("bonus_stats"))
        scored.append(d)

    scored.sort(key=lambda x: -x["score"])
    return scored[:top_n]




# ───────────────────────── Core: auto_optimize ─────────────────────────

def auto_optimize(
    budget: int,
    priorities: dict[str, float] | None = None,
    style: str | None = None,
    top_n: int = 10,
    carryable_only: bool = True,
    allow_duplicates: bool = False,
    owned_artifacts: list[dict] | None = None,
    exclude_item_ids: list[str] | None = None,
    region: str | None = None,
    containers_override: list[str] | None = None,
    armor_id: str | None = None, armor_ids: list[str | None] | None = None, armor_level: int = 15,
    stock_only: bool = False,
) -> dict:
    """Find best builds across multiple containers automatically.

    Tries top-N containers ranked by priority-aligned bonus scoring,
    runs beam search for each, aggregates and returns Pareto-optimal builds.

    Args:
        budget: Max budget in rubles
        priorities: {prop_id: weight} or None for balanced
        style: Playstyle key ("pvp", "tank", etc.) — overrides priorities
        top_n: Max builds to return
        carryable_only: Only carryable builds
        allow_duplicates: Allow same item_id multiple times
        owned_artifacts: List of owned artifact dicts (price=0)
        exclude_item_ids: Item IDs to exclude
        region: Server region
        containers_override: Explicit list of container IDs to try
            (bypasses auto-selection)

    Returns:
        {
            "budget": int,
            "style": str,
            "priorities": dict,
            "containers_tried": list[str],
            "builds": list[dict],  # sorted by score, Pareto-optimal
            "elapsed_sec": float,
        }
    """
    if style and style in PLAYSTYLES:
        priorities = PLAYSTYLES[style]
    if not priorities:
        priorities = PLAYSTYLES["balanced"]
        style = "balanced"
    elif not style:
        style = "custom"

    region = region or Config.REGION
    t0 = time.time()

    # Select containers
    if containers_override:
        container_ids = containers_override
        containers_info = []
        for cid in container_ids:
            conn = get_connection()
            row = conn.execute(
                "SELECT id, name_ru, slots, inner_protection, effectiveness, "
                "bonus_stats FROM containers WHERE id=?",
                (cid,),
            ).fetchone()
            if row:
                d = dict(row)
                d["bonuses"] = _parse_container_bonuses(d.get("bonus_stats"))
                containers_info.append(d)
    else:
        scored = get_best_containers(priorities, top_n=5, min_slots=3)
        container_ids = [c["id"] for c in scored]
        containers_info = scored

    log.info(
        "auto_optimize: budget=%d style=%s containers=%s",
        budget, style, container_ids,
    )

    # All selected containers share ONE request-wide normalization and selection.
    containers_tried = list(container_ids)
    all_builds = build_artifact_sets(
        budget=budget, priorities=priorities, top_n=top_n, region=region,
        container_ids=container_ids, owned_artifacts=owned_artifacts,
        carryable_only=carryable_only, allow_duplicates=allow_duplicates,
        exclude_item_ids=exclude_item_ids,
        armor_id=armor_id, armor_ids=armor_ids, armor_level=armor_level,
        **({'stock_only': True} if stock_only else {}),
    )

    if not all_builds:
        return {
            "budget": budget,
            "style": style,
            "priorities": priorities,
            "containers_tried": containers_tried,
            "builds": [],
            "elapsed_sec": round(time.time() - t0, 2),
        }

    # Builder already anchors Pareto/diversity at the request objective optimum.
    result_builds = [b.to_dict() for b in all_builds[:top_n]]

    elapsed = time.time() - t0
    log.info(
        "auto_optimize done: %d builds from %d containers in %.1fs",
        len(result_builds), len(containers_tried), elapsed,
    )

    return {
        "budget": budget,
        "style": style,
        "priorities": priorities,
        "containers_tried": containers_tried,
        "builds": result_builds,
        "elapsed_sec": round(elapsed, 2),
    }


# ───────────────────────── Budget sweep ─────────────────────────

def budget_sweep(
    min_budget: int = 10_000,
    max_budget: int = 500_000_000,
    steps: int = 8,
    style: str | None = None,
    priorities: dict[str, float] | None = None,
    top_n: int = 3,
    carryable_only: bool = True,
    allow_duplicates: bool = False,
    region: str | None = None,
    armor_id: str | None = None,
    armor_level: int = 15,
    stock_only: bool = False,
) -> dict:
    """Run optimization at multiple budget points (logarithmic spacing).

    Shows how the meta evolves: at 1M you get build X, at 5M build Y, etc.

    Returns:
        {
            "style": str,
            "steps": [
                {
                    "budget": int,
                    "builds_found": int,
                    "best_score": float,
                    "best_ehp": float,
                    "best_build": dict | None,  # top build or None
                    "builds": list[dict],
                },
                ...
            ],
        }
    """
    if style and style in PLAYSTYLES:
        priorities = PLAYSTYLES[style]
    if not priorities:
        priorities = PLAYSTYLES["balanced"]
        style = "balanced"
    elif not style:
        style = "custom"

    # Logarithmic budget spacing
    log_min = math.log(max(min_budget, 1))
    log_max = math.log(max(max_budget, min_budget + 1))
    budgets = [
        int(math.exp(log_min + (log_max - log_min) * i / (steps - 1)))
        for i in range(steps)
    ]

    sweep_steps = []
    for budget in budgets:
        log.info("Budget sweep: %d (%s)", budget, style)
        result = auto_optimize(
            budget=budget,
            priorities=priorities,
            style=style,
            top_n=top_n,
            carryable_only=carryable_only,
            allow_duplicates=allow_duplicates,
            region=region,
            armor_id=armor_id, armor_level=armor_level,
            **({'stock_only': True} if stock_only else {}),
        )
        builds = result["builds"]
        best = builds[0] if builds else None
        sweep_steps.append({
            "budget": budget,
            "builds_found": len(builds),
            "best_score": best.get("total_score", 0) if best else 0,
            "best_ehp": best.get("effective_health", 0) if best else 0,
            "best_build": best,
            "builds": builds,
        })

    return {
        "style": style,
        "priorities": priorities,
        "steps": sweep_steps,
    }


# ───────────────────────── Pre-computation & cache ─────────────────────────

_META_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "meta_builds_balanced_physical_armor_v1.json"
)


def pre_compute_meta(force: bool = False) -> dict:
    """Pre-compute meta builds for all budget tiers × playstyles.

    Expensive operation (potentially hours). Run overnight via CLI:
        python -m app.meta_optimizer --precompute

    Results cached to data/meta_builds.json for instant lookup.
    """
    if not force and os.path.exists(_META_CACHE_PATH):
        try:
            with open(_META_CACHE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if (cached.get("version") == 4 and cached.get("objective_contract") == "balanced-v1"
                    and cached.get("availability_contract") == AVAILABILITY_CONTRACT
                    and cached.get("equipment_contract") == "armor-v1"):
                log.info("Meta cache exists, use --force to recompute")
                return cached
        except (json.JSONDecodeError, IOError):
            pass

    results: dict = {"version": 4, "objective_contract": "balanced-v1",
                     "availability_contract": AVAILABILITY_CONTRACT, "equipment_contract": "armor-v1",
                     "computed_at": time.strftime("%Y-%m-%d %H:%M"), "builds": {}}

    for style_name in PLAYSTYLES:
        results["builds"][style_name] = {}
        for budget in BUDGET_TIERS:
            log.info("Pre-computing %s @ %d...", style_name, budget)
            try:
                result = auto_optimize(budget, style=style_name, top_n=5)
                builds = result["builds"]
                results["builds"][style_name][str(budget)] = builds
                best = builds[0] if builds else None
                log.info(
                    "  → %d builds, best score=%s",
                    len(builds),
                    f"{best['total_score']:.1f}" if best else "none",
                )
            except Exception as e:
                log.error("Pre-compute failed for %s @ %d: %s", style_name, budget, e)
                results["builds"][style_name][str(budget)] = []

    os.makedirs(os.path.dirname(_META_CACHE_PATH), exist_ok=True)
    with open(_META_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log.info("Meta cache saved to %s", _META_CACHE_PATH)
    return results


def load_meta_cache() -> dict | None:
    """Load pre-computed meta builds from cache."""
    if not os.path.exists(_META_CACHE_PATH):
        return None
    try:
        with open(_META_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
            return cached if (cached.get("version") == 4 and cached.get("objective_contract") == "balanced-v1"
                              and cached.get("availability_contract") == AVAILABILITY_CONTRACT
                    and cached.get("equipment_contract") == "armor-v1") else None
    except (json.JSONDecodeError, IOError):
        return None


def get_cached_meta(budget: int, style: str = "balanced") -> list[dict] | None:
    """Get pre-computed meta builds for closest budget tier + style.

    Returns None if cache missing or budget too far (>2× diff) from any tier.
    """
    cache = load_meta_cache()
    if not cache:
        return None

    style_data = cache.get("builds", {}).get(style)
    if not style_data:
        return None

    cached_budgets = sorted(int(k) for k in style_data.keys())
    if not cached_budgets:
        return None

    closest = min(cached_budgets, key=lambda b: abs(b - budget))

    # Only return if within 2× of requested budget
    if closest > 0:
        ratio = max(budget, closest) / min(budget, closest) if min(budget, closest) > 0 else 999
        if ratio > 2.0:
            return None

    return style_data.get(str(closest))


# ───────────────────────── CLI ─────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Meta-build optimizer")
    parser.add_argument("--precompute", action="store_true", help="Pre-compute all tiers × styles")
    parser.add_argument("--sweep", action="store_true", help="Run budget sweep demo")
    parser.add_argument("--budget", type=int, default=50_000_000, help="Budget for single run")
    parser.add_argument("--style", default="balanced", help="Playstyle (pvp/tank/speed/heal/balanced/carry)")
    args = parser.parse_args()

    if args.precompute:
        pre_compute_meta(force=True)
    elif args.sweep:
        result = budget_sweep(style=args.style, top_n=3)
        for step in result["steps"]:
            b = step["best_build"]
            if b:
                arts = ", ".join(a["name_ru"][:10] for a in b.get("artifacts", []))
                print(
                    f"  {step['budget']:>12,} → score={step['best_score']:.0f} "
                    f"EHP={step['best_ehp']:.1f} | {arts}"
                )
            else:
                print(f"  {step['budget']:>12,} → no builds")
    else:
        result = auto_optimize(budget=args.budget, style=args.style, top_n=5)
        print(f"\nBudget: {result['budget']:,} | Style: {result['style']} | "
              f"Containers: {result['containers_tried']} | "
              f"Elapsed: {result['elapsed_sec']}s")
        print(f"Builds found: {len(result['builds'])}")
        for i, b in enumerate(result["builds"]):
            cont = b.get("container", {})
            arts = ", ".join(a["name_ru"][:10] for a in b.get("artifacts", []))
            print(
                f"  #{i+1} score={b['total_score']:.1f} EHP={b.get('effective_health', 0):.1f} "
                f"price={b['total_price']:,} | {cont.get('name_ru', '?')[:15]} | {arts}"
            )

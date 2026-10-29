"""
Autonomous build optimizer — runs continuously, testing thousands of build combinations
with varying priorities, containers, and budgets. Logs top results for review.

Usage: python -m app.auto_tester
"""
import json
import logging
import time
import random
import itertools
from datetime import datetime
from pathlib import Path

from .builder import build_artifact_sets, _to_full_key, DANGER_THRESHOLDS
from .database import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

LOG_FILE = Path(__file__).parent.parent / "data" / "auto_test_results.jsonl"
LOG_FILE.parent.mkdir(exist_ok=True)

# All stat prop_ids available in the game
ALL_STATS = [
    "speed_modifier", "sprint_speed_modifier", "stamina_bonus", "stamina_regeneration_bonus",
    "max_weight_bonus", "bullet_dmg_factor", "explosion_dmg_factor", "tear_dmg_factor",
    "stopping_protection", "health_bonus", "heal_efficiency", "regeneration_bonus",
    "artefakt_heal", "psycho_protection", "thermal_protection", "radiation_protection",
    "biological_protection", "bleeding_protection", "recoil_bonus", "wiggle_bonus",
    "reaction_to_burn", "reaction_to_tear", "reaction_to_electroshock",
    "reaction_to_chemical_burn",
]

# Containers to test
CONTAINERS = ["rgoz", "dkln", "j3ml", "512o", "yq90"]  # Бочонок, Разгрузка, КСМ, Берлога-6, Хитин

# Budgets to test
BUDGETS = [3_000_000, 5_000_000, 8_000_000, 11_000_000, 15_000_000, 20_000_000, 30_000_000]

# Predefined mode presets
MODE_PRESETS = {
    "pvp": {"bullet_dmg_factor": 10, "health_bonus": 8, "stamina_bonus": 5, "stopping_protection": 5, "tear_dmg_factor": 5},
    "speed": {"speed_modifier": 10, "sprint_speed_modifier": 10, "stamina_regeneration_bonus": 8},
    "tank": {"health_bonus": 10, "regeneration_bonus": 10, "heal_efficiency": 8, "artefakt_heal": 8, "bullet_dmg_factor": 7, "stamina_bonus": 5},
    "carry": {"max_weight_bonus": 10, "bullet_dmg_factor": 5, "health_bonus": 5, "stamina_bonus": 5},
    "universal": {"bullet_dmg_factor": 7, "health_bonus": 7, "stamina_bonus": 5, "explosion_dmg_factor": 5, "tear_dmg_factor": 5},
}

# Random priority generator
def random_priorities():
    """Generate random priority set: 1-3 stats with random weights 5-10."""
    n_stats = random.randint(1, 3)
    chosen = random.sample(ALL_STATS, n_stats)
    return {s: random.randint(5, 10) for s in chosen}

def evaluate_build(build, priorities):
    """Score a build and return detailed metrics."""
    cs = build.combined_stats
    result = {
        "score": build.total_score,
        "price": build.total_price,
        "carryable": build.carryable,
        "container": build.container.name_ru if build.container else "?",
        "container_prot": build.container.inner_protection if build.container else 0,
        "container_eff": build.container.effectiveness if build.container else 0,
        "n_artifacts": len(build.artifacts),
        "artifacts": [{"name": a.name_ru, "id": a.item_id, "qlt": a.quality_name, "ptn": a.ptn, "art_pct": a.art_percent, "bonuses": a.unlocked_bonus_ids, "price": a.buyout_price} for a in build.artifacts],
        "stats": {k: round(v, 4) for k, v in cs.items()},
        "priorities": priorities,
    }
    return result

def run_test_cycle(cycle_num):
    """Run one test cycle: test all modes + random priorities across containers and budgets."""
    results = []

    test_cases = []

    # Add mode presets
    for mode_name, prio in MODE_PRESETS.items():
        for cont_id in CONTAINERS:
            for budget in BUDGETS:
                test_cases.append((f"{mode_name}_{cont_id}_{budget//1000000}M", prio, cont_id, budget))

    # Add random priorities (20 random sets)
    for i in range(20):
        prio = random_priorities()
        cont_id = random.choice(CONTAINERS)
        budget = random.choice(BUDGETS)
        test_cases.append((f"random_{i}_{cont_id}_{budget//1000000}M", prio, cont_id, budget))

    log.info(f"Cycle {cycle_num}: {len(test_cases)} test cases")

    for case_name, prio, cont_id, budget in test_cases:
        try:
            builds = build_artifact_sets(
                budget=budget,
                tolerance_pct=8.0,
                priorities=prio,
                top_n=1,
                carryable_only=True,
                container_id=cont_id,
            )

            if builds:
                b = builds[0]
                result = evaluate_build(b, prio)
                result["test_case"] = case_name
                result["budget"] = budget
                result["timestamp"] = datetime.now().isoformat()
                results.append(result)

                # Log to file
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        except Exception as e:
            log.warning(f"  {case_name}: ERROR {e}")
            continue

    # Find best builds per category
    if results:
        best_speed = max(results, key=lambda r: r["stats"].get("speed_modifier", 0))
        best_bullet = max(results, key=lambda r: r["stats"].get("bullet_dmg_factor", 0))
        best_health = max(results, key=lambda r: r["stats"].get("health_bonus", -999))
        best_weight = max(results, key=lambda r: r["stats"].get("max_weight_bonus", 0))
        best_score = max(results, key=lambda r: r["score"])

        log.info(f"Cycle {cycle_num} BEST RESULTS:")
        log.info(f"  Speed:    +{best_speed['stats'].get('speed_modifier', 0):.2f}% | {best_speed['container']} | {best_speed['price']:,}")
        log.info(f"  Bullet:   +{best_bullet['stats'].get('bullet_dmg_factor', 0):.2f} | {best_bullet['container']} | {best_bullet['price']:,}")
        log.info(f"  Health:   {best_health['stats'].get('health_bonus', 0):+.2f} | {best_health['container']} | {best_health['price']:,}")
        log.info(f"  Weight:   +{best_weight['stats'].get('max_weight_bonus', 0):.2f} | {best_weight['container']} | {best_weight['price']:,}")
        log.info(f"  Score:    {best_score['score']:.1f} | {best_score['container']} | {best_score['price']:,}")

    return results

def main():
    log.info("=" * 60)
    log.info("AUTONOMOUS BUILD OPTIMIZER STARTED")
    log.info(f"Log file: {LOG_FILE}")
    log.info("=" * 60)

    # Clear previous log
    LOG_FILE.write_text("", encoding="utf-8")

    cycle = 1
    total_results = 0

    while True:
        t0 = time.time()
        results = run_test_cycle(cycle)
        elapsed = time.time() - t0
        total_results += len(results)

        log.info(f"Cycle {cycle} done: {len(results)} results in {elapsed:.1f}s (total: {total_results})")
        cycle += 1

        # Sleep between cycles
        sleep_time = max(10, 60 - elapsed)  # At least 10s between cycles
        log.info(f"Sleeping {sleep_time:.0f}s...")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()

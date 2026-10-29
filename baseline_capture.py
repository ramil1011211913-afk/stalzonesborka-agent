"""Capture baseline builds for regression comparison.
Saves build results (artifacts, stats, score, price, carryable) for multiple inputs.
Run BEFORE making changes, then AFTER, and diff."""
import sys, os, json, time
sys.path.insert(0, r'C:\myprofrma\stalzonesborka')
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
os.chdir(r'C:\myprofrma\stalzonesborka')

from app.builder import build_artifact_sets

# Test cases: (name, budget, priorities, container_id, flags)
test_cases = [
    ("pvp_11M_rgoz",      11_000_000, {"bullet_dmg_factor": 10, "health_bonus": 8, "stamina_bonus": 5, "stopping_protection": 5}, "rgoz", {"carryable_only": True, "allow_duplicates": False}),
    ("tank_11M_rgoz",     11_000_000, {"health_bonus": 10, "regeneration_bonus": 10, "heal_efficiency": 8, "artefakt_heal": 8, "bullet_dmg_factor": 7}, "rgoz", {"carryable_only": True, "allow_duplicates": False}),
    ("balanced_11M_rgoz", 11_000_000, {"health_bonus": 5, "bullet_dmg_factor": 5, "heal_efficiency": 5, "stamina_bonus": 5, "max_weight_bonus": 3}, "rgoz", {"carryable_only": True, "allow_duplicates": False}),
    ("pvp_11M_auto",      11_000_000, {"bullet_dmg_factor": 10, "health_bonus": 8, "stamina_bonus": 5, "stopping_protection": 5}, None, {"carryable_only": True, "allow_duplicates": False}),
    ("tank_11M_auto",     11_000_000, {"health_bonus": 10, "regeneration_bonus": 10, "heal_efficiency": 8, "bullet_dmg_factor": 7}, None, {"carryable_only": True, "allow_duplicates": False}),
    ("pvp_dup_11M_rgoz",  11_000_000, {"bullet_dmg_factor": 10, "health_bonus": 8, "stamina_bonus": 5}, "rgoz", {"carryable_only": True, "allow_duplicates": True}),
    ("tank_dup_11M_rgoz", 11_000_000, {"health_bonus": 10, "heal_efficiency": 8, "regeneration_bonus": 8}, "rgoz", {"carryable_only": True, "allow_duplicates": True}),
    ("pvp_50M_auto",      50_000_000, {"bullet_dmg_factor": 10, "health_bonus": 8, "stamina_bonus": 5, "stopping_protection": 5}, None, {"carryable_only": True, "allow_duplicates": False}),
    ("tank_50M_auto",     50_000_000, {"health_bonus": 10, "regeneration_bonus": 10, "heal_efficiency": 8, "artefakt_heal": 8, "bullet_dmg_factor": 7}, None, {"carryable_only": True, "allow_duplicates": False}),
    ("speed_11M_rgoz",   11_000_000, {"sprint_speed_modifier": 10, "speed_modifier": 10, "stamina_regeneration_bonus": 8}, "rgoz", {"carryable_only": True, "allow_duplicates": False}),
    ("heal_11M_auto",     11_000_000, {"heal_efficiency": 10, "health_bonus": 8, "regeneration_bonus": 6, "artefakt_heal": 5}, None, {"carryable_only": True, "allow_duplicates": False}),
    ("carry_11M_auto",    11_000_000, {"max_weight_bonus": 10, "bullet_dmg_factor": 5, "health_bonus": 5, "stamina_bonus": 5}, None, {"carryable_only": True, "allow_duplicates": False}),
]

results = {}
print(f"Capturing {len(test_cases)} baseline test cases...")
start = time.time()

for i, (name, budget, prio, cid, flags) in enumerate(test_cases):
    t0 = time.time()
    builds = build_artifact_sets(
        budget=budget, tolerance_pct=8.0, priorities=prio,
        top_n=5, container_id=cid,
        carryable_only=flags["carryable_only"],
        allow_duplicates=flags["allow_duplicates"],
    )
    elapsed = time.time() - t0
    # Serialize builds
    serialized = []
    for b in builds:
        serialized.append({
            "artifacts": [(a.name_ru, a.item_id, a.ptn, a.quality, a.art_percent, a.buyout_price) for a in b.artifacts],
            "container": b.container.name_ru if b.container else None,
            "total_price": b.total_price,
            "total_score": round(b.total_score, 4),
            "carryable": b.carryable,
            "stats": {k: round(v, 4) for k, v in sorted(b.combined_stats.items())},
        })
    results[name] = {
        "builds": serialized,
        "time": round(elapsed, 2),
        "count": len(builds),
    }
    # Print summary
    top = serialized[0] if serialized else None
    if top:
        s = top["stats"]
        ehp = (1 + s.get("health_bonus", 0)/100) * (100 + s.get("bullet_dmg_factor", 0))
        print(f"  [{i+1}/{len(test_cases)}] {name}: {len(builds)} builds, {elapsed:.1f}s, "
              f"top price={top['total_price']:,} score={top['total_score']:.1f} EHP={ehp:.1f} carry={top['carryable']}")
    else:
        print(f"  [{i+1}/{len(test_cases)}] {name}: 0 builds, {elapsed:.1f}s")

total = time.time() - start
print(f"\nTotal: {total:.1f}s ({total/60:.1f}min)")

# Save
outpath = os.path.join(os.path.dirname(__file__), "data", "baseline_builds.json")
with open(outpath, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Saved to {outpath}")

"""Compare baseline (old scoring) vs new (EHP/HPS + Pareto) builds.
Reports: improved / same / worse for each test case."""
import sys, os, json, time
sys.path.insert(0, r'C:\myprofrma\stalzonesborka')
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
os.chdir(r'C:\myprofrma\stalzonesborka')

from app.builder import build_artifact_sets, calculate_effective_health, calculate_healing_per_second

# Load baseline
with open("data/baseline_builds.json", "r", encoding="utf-8") as f:
    baseline = json.load(f)

# Same test cases as baseline_capture.py
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

print(f"Comparing {len(test_cases)} test cases: baseline (old) vs new (EHP/HPS + Pareto)")
print(f"{'Test':25s} | {'Old EHP':>8s} {'Old HPS':>7s} {'Old Price':>12s} | {'New EHP':>8s} {'New HPS':>7s} {'New Price':>12s} | {'Verdict':>8s}")
print("-" * 100)

improved = 0
same = 0
worse = 0
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

    # New top build
    new_b = builds[0] if builds else None
    if not new_b:
        print(f"  {name}: NO BUILDS returned!")
        worse += 1
        continue

    new_s = new_b.combined_stats
    new_ehp = calculate_effective_health(new_s)
    new_hps = calculate_healing_per_second(new_s)

    # Baseline top build
    old = baseline.get(name, {})
    old_builds = old.get("builds", [])
    old_b = old_builds[0] if old_builds else None
    if old_b:
        old_s = old_b["stats"]
        old_ehp = (1 + old_s.get("health_bonus", 0)/100) * (100 + old_s.get("bullet_dmg_factor", 0))
        old_hps = (old_s.get("regeneration_bonus", 0)+2.5)/5 + old_s.get("artefakt_heal", 0)*(1+old_s.get("heal_efficiency", 0)/100)
        old_price = old_b["total_price"]
    else:
        old_ehp = old_hps = old_price = 0

    # Verdict: compare EHP + HPS (primary metrics)
    new_metrics = new_ehp + new_hps * 10  # weight HPS less
    old_metrics = old_ehp + old_hps * 10
    if new_metrics > old_metrics + 0.01:
        verdict = "BETTER"
        improved += 1
    elif new_metrics < old_metrics - 0.01:
        verdict = "WORSE"
        worse += 1
    else:
        verdict = "SAME"
        same += 1

    new_arts = [a.name_ru for a in new_b.artifacts]
    old_arts = [a[0] for a in old_b["artifacts"]] if old_b else []
    arts_changed = new_arts != old_arts

    print(f"  {name:25s} | {old_ehp:8.1f} {old_hps:7.2f} {old_price:>12,} | {new_ehp:8.1f} {new_hps:7.2f} {new_b.total_price:>12,} | {verdict:>8s} ({elapsed:.1f}s)")
    if arts_changed:
        print(f"    arts: {old_arts}")
        print(f"     ->  {new_arts}")
    else:
        print(f"    arts: same")

total = time.time() - start
print(f"\n{'='*80}")
print(f"Results: {improved} improved, {same} same, {worse} worse (total {total:.1f}s)")
if worse > 0:
    print("⚠ SOME BUILDS GOT WORSE — needs investigation!")
else:
    print("✓ No builds got worse — all same or better!")

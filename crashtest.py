"""
Massive crash test: run build_artifact_sets across all containers, modes,
priority sets, and flag combinations. Log errors, track performance, find bugs.
"""
import sys, os, json, time, traceback, logging
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)  # line-buffered
from datetime import datetime
from itertools import product

sys.path.insert(0, r'C:\myprofrma\stalzonesborka')
os.chdir(r'C:\myprofrma\stalzonesborka')

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

from app.builder import build_artifact_sets, load_container_options
from app.database import get_connection

# Only 6-7 slot containers (most interesting, C(n,6-7) combinatorics)
all_containers = load_container_options('EU')
container_ids = [c.item_id for c in all_containers if c.slots >= 5]

# Priority presets (4 most diverse)
priority_sets = {
    'pvp': {'bullet_dmg_factor': 10, 'health_bonus': 8, 'stamina_bonus': 5, 'stopping_protection': 5, 'tear_dmg_factor': 5},
    'speed': {'sprint_speed_modifier': 10, 'speed_modifier': 10, 'stamina_regeneration_bonus': 8},
    'tank': {'health_bonus': 10, 'regeneration_bonus': 10, 'heal_efficiency': 8, 'artefakt_heal': 8, 'bullet_dmg_factor': 7},
    'balanced': {'health_bonus': 5, 'bullet_dmg_factor': 5, 'heal_efficiency': 5, 'stamina_bonus': 5, 'max_weight_bonus': 3},
}

# Min threshold sets (3 most useful)
threshold_sets = {
    'none': {},
    'mw50_h0': {'max_weight_bonus': 50, 'health_bonus': 0},
    'he30': {'heal_efficiency': 30},
}

# Budget levels (1 — 50M is too slow for C(100,7))
budgets = [11000000]

# Flag combinations (5)
flags = [
    {'carryable_only': True, 'allow_duplicates': False, 'min_budget': False},
    {'carryable_only': True, 'allow_duplicates': True, 'min_budget': False},
    {'carryable_only': False, 'allow_duplicates': False, 'min_budget': False},
    {'carryable_only': False, 'allow_duplicates': True, 'min_budget': False},
    {'carryable_only': True, 'allow_duplicates': False, 'min_budget': True},
]


# Build test matrix
tests = []
test_id = 0
for cid in container_ids:
    for pname, prio in priority_sets.items():
        for tname, thresh in threshold_sets.items():
            for budget in budgets:
                for fidx, flag in enumerate(flags):
                    if flag['min_budget'] and tname != 'none':
                        continue
                    # Skip min_budget with carryable_only=False
                    if flag['min_budget'] and not flag['carryable_only']:
                        continue
                    test_id += 1
                    tests.append({
                        'id': test_id,
                        'container': cid,
                        'priority': pname,
                        'threshold': tname,
                        'budget': budget,
                        'flag': f'flag_{fidx}',
                        'prio': prio,
                        'thresh': thresh,
                        'flag_dict': flag,
                    })

print(f"=== CRASH TEST MATRIX ===")
print(f"Containers: {len(container_ids)}")
print(f"Priority sets: {len(priority_sets)}")
print(f"Threshold sets: {len(threshold_sets)}")
print(f"Budgets: {len(budgets)}")
print(f"Flag combos: {len(flags)}")
print(f"Total tests: {len(tests)}")
print(f"Estimated time: ~{len(tests) * 15 / 60:.0f} minutes (avg 15s/test)")
print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
print()

# Results tracking
results = []
errors = []
timeouts = []
bug_count = 0
pass_count = 0
start_all = time.time()

for i, test in enumerate(tests):
    t_start = time.time()
    test_label = f"[{i+1}/{len(tests)}] C={test['container']} P={test['priority']} T={test['threshold']} B={test['budget']:,} F={test['flag']}"

    try:
        budget = test['budget']
        if test['flag_dict']['min_budget']:
            budget = 0
        builds = build_artifact_sets(
            budget=budget,
            tolerance_pct=8.0,
            priorities=test['prio'],
            min_thresholds=test['thresh'] if test['thresh'] else None,
            top_n=5,
            container_id=test['container'],
            carryable_only=test['flag_dict']['carryable_only'],
            allow_duplicates=test['flag_dict']['allow_duplicates'],
            min_budget=test['flag_dict']['min_budget'],
        )
        t_elapsed = time.time() - t_start
        if (i+1) % 10 == 0:
            elapsed = time.time() - start_all
            eta = elapsed / (i+1) * (len(tests) - i - 1) if i > 0 else 0
            print(f"  [{i+1}/{len(tests)}] {elapsed/60:.1f}min elapsed, ETA {eta/60:.1f}min, last={t_elapsed:.1f}s, {len(builds)} builds")

        # Validate results
        validation_errors = []
        for b in builds:
            cs = b.combined_stats

            # Check carryable_only flag
            if test['flag_dict']['carryable_only'] and not b.carryable:
                validation_errors.append(f"carryable_only=True but build not carryable!")

            # Check min_thresholds
            for k, v in test['thresh'].items():
                if cs.get(k, 0) < v:
                    validation_errors.append(f"min_threshold {k}={v} not met: {cs.get(k, 0):.2f}")

            # Check budget
            eff_budget = int(test['budget'] * 1.08) if not test['flag_dict']['min_budget'] else 999999999
            if b.total_price > eff_budget:
                validation_errors.append(f"budget exceeded: {b.total_price:,} > {eff_budget:,}")

            # Check for NaN/Inf in stats
            for k, v in cs.items():
                if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')):
                    validation_errors.append(f"NaN/Inf in stat {k}={v}")

            # Check danger stats consistency
            for d in ['psycho_accumulation', 'radiation_accumulation', 'thermal_accumulation', 'biological_accumulation', 'frost_accumulation']:
                v = cs.get(d, 0)
                threshold = 1.0 if d == 'frost_accumulation' else 0.5
                if b.carryable and v > threshold + 0.001:
                    validation_errors.append(f"carryable=True but {d}={v:.4f} > {threshold}")

        if validation_errors:
            bug_count += 1
            errors.append({
                'test': test_label,
                'errors': validation_errors,
                'build_count': len(builds),
                'time': t_elapsed,
            })
            print(f"  ⚠ BUG {test_label} — {len(validation_errors)} validation errors ({t_elapsed:.1f}s)")
            for ve in validation_errors[:3]:
                print(f"      {ve}")
        else:
            pass_count += 1
            if t_elapsed > 60:
                print(f"  ⏱ SLOW {test_label} — {t_elapsed:.1f}s, {len(builds)} builds")
            elif (i+1) % 50 == 0:
                print(f"  ✓ [{i+1}/{len(tests)}] passed — {len(builds)} builds ({t_elapsed:.1f}s)")

        results.append({
            'test': test_label,
            'builds': len(builds),
            'time': t_elapsed,
            'pass': len(validation_errors) == 0,
        })

    except Exception as e:
        t_elapsed = time.time() - t_start
        bug_count += 1
        errors.append({
            'test': test_label,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'time': t_elapsed,
        })
        print(f"  ✗ ERROR {test_label} — {e} ({t_elapsed:.1f}s)")

    # Early timeout check — if a single test takes >5 min, skip
    if time.time() - t_start > 300:
        timeouts.append(test_label)
        print(f"  ⏰ TIMEOUT {test_label} (>300s)")

total_time = time.time() - start_all

print(f"\n{'='*60}")
print(f"=== CRASH TEST COMPLETE ===")
print(f"{'='*60}")
print(f"Total tests: {len(tests)}")
print(f"Passed: {pass_count}")
print(f"Bugs/Errors: {bug_count}")
print(f"Timeouts: {len(timeouts)}")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Avg time/test: {total_time/len(tests):.1f}s")
print(f"End: {datetime.now().strftime('%H:%M:%S')}")

if errors:
    print(f"\n=== ERROR SUMMARY ({len(errors)}) ===")
    # Group by error type
    error_types = {}
    for e in errors:
        if 'errors' in e:
            for ve in e['errors']:
                # Simplify error type
                etype = ve.split(':')[0][:60]
                error_types[etype] = error_types.get(etype, 0) + 1
        elif 'error' in e:
            etype = e['error'][:60]
            error_types[etype] = error_types.get(etype, 0) + 1

    for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"  [{count}x] {etype}")

    # Show first 5 detailed errors
    print(f"\n=== FIRST 5 ERRORS (detailed) ===")
    for e in errors[:5]:
        print(f"\n  Test: {e['test']}")
        if 'errors' in e:
            for ve in e['errors']:
                print(f"    {ve}")
        elif 'error' in e:
            print(f"    {e['error']}")
            print(f"    {e['traceback'][:200]}")
else:
    print(f"\n✓ NO ERRORS FOUND — ALL TESTS PASSED")

# Performance stats
times = [r['time'] for r in results if r['pass']]
if times:
    print(f"\n=== PERFORMANCE ===")
    print(f"  Min: {min(times):.1f}s")
    print(f"  Max: {max(times):.1f}s")
    print(f"  Avg: {sum(times)/len(times):.1f}s")
    print(f"  Median: {sorted(times)[len(times)//2]:.1f}s")

    # Slowest tests
    slowest = sorted(results, key=lambda r: -r['time'])[:5]
    print(f"\n  Slowest 5:")
    for r in slowest:
        print(f"    {r['time']:.1f}s — {r['test']} ({r['builds']} builds)")

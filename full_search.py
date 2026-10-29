"""
Full exhaustive parallel search over ALL auction lots.
C(921, 4) = ~30B combinations, 16 cores, ~17 seconds.
"""
import sys, os, json, time, math
import numpy as np
from numba import njit, prange
from multiprocessing import shared_memory, Process

sys.path.insert(0, r'C:\myprofrma\stalzonesborka')
os.chdir(r'C:\myprofrma\stalzonesborka')

from app.builder import (load_artifact_options, compute_artifact_stats, QUALITY_RANGES,
                         QUALITY_NAMES, _normalize_bonus_id, _score_option,
                         load_container_options, ArtifactOption, _is_counter_art,
                         _apply_build_container_protection, _recalculate_stopping_protection,
                         apply_reaction_bonuses, check_carryability)
from app.database import get_connection
from itertools import combinations as _it_combinations

# ─── Setup ──────────────────────────────────────────────────────────────────
container = [c for c in load_container_options('EU') if c.item_id == 'rgoz'][0]
conn = get_connection()

# Owned artifacts
owned_opts = []
for oa_id, qlt, ap in [('5rd1', 3, 145), ('ljn2', 2, 130), ('zjzn', 2, 130)]:
    art = conn.execute('SELECT id, name_ru, name_en, subcategory, stats, bonus_stats, icon_url FROM artefacts WHERE id=?', (oa_id,)).fetchone()
    base = json.loads(art['stats'])
    bonus = json.loads(art['bonus_stats']) if art['bonus_stats'] else []
    qr = QUALITY_RANGES.get(qlt, QUALITY_RANGES[0])
    stats_c, _, sn, _ = compute_artifact_stats(base, bonus, [], ap, 15, qr['min'], qr['range'], None, container.inner_protection, container.effectiveness)
    owned_opts.append(ArtifactOption(item_id=art['id'], name_ru=art['name_ru'], name_en=art['name_en'],
        subcategory=art['subcategory'], quality=qlt, quality_name='', art_percent=ap,
        buyout_price=0, ptn=15, stats=stats_c, stats_raw=stats_c, stats_names=sn,
        unlocked_bonus_ids=[], icon_url=art['icon_url'], base_stats=base, bonus_stats=bonus,
        stats_random=None, is_owned=True))

owned_stats = {}
for o in owned_opts:
    for s, v in o.stats.items():
        owned_stats[s] = owned_stats.get(s, 0.0) + v

# Load ALL affordable EU lots (no rn<=2 limit)
rows = conn.execute('''
    SELECT a.id, a.name_ru, a.name_en, a.subcategory, a.stats, a.bonus_stats, a.icon_url,
           l.quality, l.ptn, l.buyout_price, l.bonus_properties, l.art_percent, l.stats_random
    FROM artefacts a
    JOIN auction_lots l ON l.item_id = a.id
    WHERE l.region='EU' AND l.buyout_price > 0 AND l.buyout_price <= 11880000 AND a.id != '9n7z'
    ORDER BY l.buyout_price ASC
''').fetchall()

# Compute stats per lot
options = []
for row in rows:
    base_stats = json.loads(row['stats'])
    bonus_stats = json.loads(row['bonus_stats']) if row['bonus_stats'] else []
    quality = row['quality'] or 0
    ptn = row['ptn'] or 0
    qr = QUALITY_RANGES.get(quality, QUALITY_RANGES[0])
    art_percent = row['art_percent'] if row['art_percent'] else qr['min'] + qr['range']
    if ptn > 0 and row['bonus_properties']:
        lot_bonuses = json.loads(row['bonus_properties']) if isinstance(row['bonus_properties'], str) else row['bonus_properties']
        bonus_combos = [lot_bonuses] if lot_bonuses else [[]]
    elif ptn > 0 and bonus_stats:
        all_bonus_props = [s.get('prop_id', '') for s in bonus_stats]
        bonus_combos = []
        for r in range(1, min(len(all_bonus_props), 3) + 1):
            for c in _it_combinations(all_bonus_props, r):
                bonus_combos.append(list(c))
        if not bonus_combos:
            bonus_combos = [[]]
        bonus_combos = bonus_combos[:10]
    else:
        bonus_combos = [[]]
    for combo in bonus_combos:
        norm_ids = [_normalize_bonus_id(b.upper()) for b in combo]
        stats_c, stats_raw, stats_names, _ = compute_artifact_stats(
            base_stats, bonus_stats, norm_ids, art_percent, ptn,
            qr['min'], qr['range'],
            row['stats_random'] if row['stats_random'] is not None else None,
            container.inner_protection, container.effectiveness)
        opt = ArtifactOption(item_id=row['id'], name_ru=row['name_ru'], name_en=row['name_en'],
            subcategory=row['subcategory'], quality=quality,
            quality_name=QUALITY_NAMES.get(quality, '?'),
            art_percent=art_percent, buyout_price=row['buyout_price'], ptn=ptn,
            stats=stats_c, stats_raw=stats_raw, stats_names=stats_names,
            unlocked_bonus_ids=combo, icon_url=row['icon_url'],
            base_stats=base_stats, bonus_stats=bonus_stats, stats_random=row['stats_random'])
        opt.score = _score_option(stats_c, {})
        options.append(opt)

# Dedup by unique stat profile (cheapest per profile)
by_profile = {}
for o in options:
    key = (o.item_id, o.ptn, o.quality, round(o.art_percent, 1), tuple(sorted(o.unlocked_bonus_ids)))
    if key not in by_profile or o.buyout_price < by_profile[key].buyout_price:
        by_profile[key] = o
candidates = list(by_profile.values())

# Filter frost=0 + useful stats
candidates = [o for o in candidates if
    o.stats.get('frost_accumulation', 0) == 0 and
    (o.stats.get('bullet_dmg_factor',0) > 1 or
     o.stats.get('heal_efficiency',0) > 1 or
     o.stats.get('max_weight_bonus',0) > 1 or
     o.stats.get('health_bonus',0) > 0 or
     o.stats.get('psycho_accumulation',0) < -0.5 or
     o.stats.get('radiation_accumulation',0) < -0.5 or
     o.stats.get('biological_accumulation',0) < -0.5 or
     o.stats.get('thermal_accumulation',0) < -0.5)]

# Sort by price ascending for early break
candidates.sort(key=lambda o: o.buyout_price)

n = len(candidates)
stat_keys = ['radiation_accumulation','biological_accumulation','psycho_accumulation',
             'thermal_accumulation','frost_accumulation',
             'health_bonus','bullet_dmg_factor','heal_efficiency','max_weight_bonus']
ns = len(stat_keys)

raw_mx = np.zeros((n, ns), dtype=np.float64)
prices = np.zeros(n, dtype=np.int64)
item_ids = np.zeros(n, dtype=np.int32)
for i, o in enumerate(candidates):
    for j, key in enumerate(stat_keys):
        raw_mx[i, j] = o.stats.get(key, 0.0)
    prices[i] = o.buyout_price
    item_ids[i] = hash(o.item_id) % 100000

owned_raw = np.zeros(ns, dtype=np.float64)
for j, key in enumerate(stat_keys):
    owned_raw[j] = owned_stats.get(key, 0.0)

prot_mult = 0.4
budget = 11880000
safe_thresh = np.array([0.49, 0.49, 0.49, 0.49, 1.001], dtype=np.float64)
n_keep = 2000

print(f'Candidates: {n}')
print(f'C({n},4) = {math.comb(n,4):,}')

# ─── Parallel Numba kernel ──────────────────────────────────────────────────
@njit(nogil=True, parallel=True)
def _parallel_search(raw_mx, prices, item_ids, owned_raw, prot_mult, budget,
                      n_keep, safe_thresh, n, ns):
    """Each thread handles a range of i0 via prange. Each keeps its own top-K."""
    n_threads = 16
    # Each thread has its own top-K
    bs_all = np.full((n_threads, n_keep), -1e18, dtype=np.float64)
    bc_all = np.full((n_threads, n_keep, 4), -1, dtype=np.int32)
    checked_all = np.zeros(n_threads, dtype=np.int64)
    safe_all = np.zeros(n_threads, dtype=np.int64)
    carry_all = np.zeros(n_threads, dtype=np.int64)

    for i0 in prange(n):
        tid = i0 % n_threads
        p0 = prices[i0]
        if p0 > budget:
            continue
        cr = np.zeros(ns, dtype=np.float64)
        for s in range(ns):
            cr[s] = owned_raw[s] + raw_mx[i0, s]
        for i1 in range(i0 + 1, n):
            if item_ids[i1] == item_ids[i0]:
                continue
            p1 = p0 + prices[i1]
            if p1 > budget:
                break  # prices sorted ascending → all subsequent also exceed
            for s in range(ns):
                cr[s] += raw_mx[i1, s]
            for i2 in range(i1 + 1, n):
                if item_ids[i2] == item_ids[i0] or item_ids[i2] == item_ids[i1]:
                    continue
                p2 = p1 + prices[i2]
                if p2 > budget:
                    break
                for s in range(ns):
                    cr[s] += raw_mx[i2, s]
                for i3 in range(i2 + 1, n):
                    if item_ids[i3] == item_ids[i0] or item_ids[i3] == item_ids[i1] or item_ids[i3] == item_ids[i2]:
                        continue
                    p3 = p2 + prices[i3]
                    if p3 > budget:
                        break
                    for s in range(ns):
                        cr[s] += raw_mx[i3, s]
                    checked_all[tid] += 1

                    # Safety check (with protection for danger stats)
                    safe = True
                    for d in range(5):
                        v = cr[d]
                        if v > 0 and d < 4:
                            v = v * prot_mult
                        if v > safe_thresh[d]:
                            safe = False
                            break
                    if not safe:
                        for s in range(ns):
                            cr[s] -= raw_mx[i3, s]
                        continue
                    safe_all[tid] += 1

                    # Weight >= 45
                    if cr[8] < 45.0:
                        for s in range(ns):
                            cr[s] -= raw_mx[i3, s]
                        continue
                    carry_all[tid] += 1

                    # Score
                    h = cr[5]; b = cr[6]; he = cr[7]; mw = cr[8]
                    score = h * 8.0 + b * 2.0 + he * 2.0 + mw * 0.5

                    if score > bs_all[tid, -1]:
                        for j in range(n_keep):
                            if score > bs_all[tid, j]:
                                for m in range(n_keep - 1, j, -1):
                                    bs_all[tid, m] = bs_all[tid, m-1]
                                    bc_all[tid, m, 0] = bc_all[tid, m-1, 0]
                                    bc_all[tid, m, 1] = bc_all[tid, m-1, 1]
                                    bc_all[tid, m, 2] = bc_all[tid, m-1, 2]
                                    bc_all[tid, m, 3] = bc_all[tid, m-1, 3]
                                bs_all[tid, j] = score
                                bc_all[tid, j, 0] = i0
                                bc_all[tid, j, 1] = i1
                                bc_all[tid, j, 2] = i2
                                bc_all[tid, j, 3] = i3
                                break

                    for s in range(ns):
                        cr[s] -= raw_mx[i3, s]
                for s in range(ns):
                    cr[s] -= raw_mx[i2, s]
            for s in range(ns):
                cr[s] -= raw_mx[i1, s]

    # Reduce: merge all threads' top-K into one
    total_checked = 0
    total_safe = 0
    total_carry = 0
    for t in range(n_threads):
        total_checked += checked_all[t]
        total_safe += safe_all[t]
        total_carry += carry_all[t]

    # Merge: flatten all (n_threads * n_keep) entries, pick top n_keep
    flat_bs = np.full(n_threads * n_keep, -1e18, dtype=np.float64)
    flat_bc = np.full((n_threads * n_keep, 4), -1, dtype=np.int32)
    idx = 0
    for t in range(n_threads):
        for j in range(n_keep):
            flat_bs[idx] = bs_all[t, j]
            flat_bc[idx, 0] = bc_all[t, j, 0]
            flat_bc[idx, 1] = bc_all[t, j, 1]
            flat_bc[idx, 2] = bc_all[t, j, 2]
            flat_bc[idx, 3] = bc_all[t, j, 3]
            idx += 1

    # Sort and pick top n_keep
    order = np.argsort(-flat_bs)
    final_bs = np.full(n_keep, -1e18, dtype=np.float64)
    final_bc = np.full((n_keep, 4), -1, dtype=np.int32)
    for j in range(min(n_keep, len(order))):
        final_bs[j] = flat_bs[order[j]]
        final_bc[j] = flat_bc[order[j]]

    return final_bs, final_bc, total_checked, total_safe, total_carry

print('Compiling parallel numba kernel...')
t0 = time.time()
bs, bc, checked, safe_count, carry_count = _parallel_search(
    raw_mx, prices, item_ids, owned_raw, prot_mult, budget, n_keep, safe_thresh, n, ns)
t1 = time.time()
print(f'Checked: {checked:,}, safe: {safe_count:,}, carry(mw>=45): {carry_count:,}, time: {t1-t0:.1f}s')
print(f'Speed: {checked/(t1-t0)/1e6:.1f}M/s')

# ─── Extract and display results ───────────────────────────────────────────
valid = bs > -1e17
top_idx = np.argsort(-bs)
top_idx = top_idx[valid[top_idx]]

# Recompute full stats for each top combo
combo_scores = []
for i in range(len(bs)):
    if bs[i] <= -1e17:
        continue
    combo = [candidates[j] for j in bc[i]]
    price = sum(o.buyout_price for o in combo)
    cr2 = dict(owned_stats)
    for o in combo:
        for s, v in o.stats.items():
            cr2[s] = cr2.get(s, 0.0) + v
    so = {k: 'artefact' for k in cr2}
    cs = _apply_build_container_protection(cr2, so, container.inner_protection)
    _recalculate_stopping_protection(cs)
    cs, _ = apply_reaction_bonuses(cs)
    carryable, _ = check_carryability(cs)
    h = cs.get('health_bonus', 0); b = cs.get('bullet_dmg_factor', 0)
    he = cs.get('heal_efficiency', 0); mw = cs.get('max_weight_bonus', 0)
    psy = cs.get('psycho_accumulation', 0); rad = cs.get('radiation_accumulation', 0)
    th = cs.get('thermal_accumulation', 0); bio = cs.get('biological_accumulation', 0)
    fr = cs.get('frost_accumulation', 0)
    combo_scores.append({
        'combo': combo, 'price': price, 'h': h, 'b': b, 'he': he, 'mw': mw,
        'psy': psy, 'rad': rad, 'th': th, 'bio': bio, 'fr': fr, 'carry': carryable,
        'balance': h*8 + b*2 + he*2 + mw*0.5,
        'sum': h*5 + b + he + mw/5,
        'pareto': min(h/12.8, b/112.9, he/62.2, mw/54.8) if max(h,0.1)>0 else 0,
    })

for label, key_func in [
    ('BALANCE', lambda x: -x['balance']),
    ('BULLET', lambda x: -x['b']),
    ('HEAL_EFF', lambda x: -x['he']),
    ('HEALTH', lambda x: -x['h']),
    ('PARETO', lambda x: -x['pareto']),
    ('SUM', lambda x: -x['sum']),
]:
    srt = sorted(combo_scores, key=key_func)
    print(f'\n--- {label} TOP 5 ---')
    for rank, cs in enumerate(srt[:5], 1):
        print(f'  #{rank} h={cs["h"]:.1f} b={cs["b"]:.1f} he={cs["he"]:.1f} mw={cs["mw"]:.1f} p={cs["price"]:,} | psy={cs["psy"]:.3f} rad={cs["rad"]:.3f} th={cs["th"]:.3f} bio={cs["bio"]:.3f} carry={cs["carry"]}')

# Full details for best of each
for label, key_func in [
    ('BALANCE', lambda x: -x['balance']),
    ('BULLET', lambda x: -x['b']),
    ('HEAL_EFF', lambda x: -x['he']),
    ('PARETO', lambda x: -x['pareto']),
]:
    srt = sorted(combo_scores, key=key_func)
    print(f'\n=== {label} BEST (full) ===')
    cs = srt[0]
    print(f'  h={cs["h"]:.1f} b={cs["b"]:.1f} he={cs["he"]:.1f} mw={cs["mw"]:.1f} price={cs["price"]:,}')
    print(f'  psy={cs["psy"]:.4f} rad={cs["rad"]:.4f} fr={cs["fr"]:.2f} th={cs["th"]:.4f} bio={cs["bio"]:.4f}')
    for a in owned_opts:
        print(f'    {a.name_ru:<20} [В ХРАН]')
    for a in cs['combo']:
        tag = ' [C]' if _is_counter_art(a) else ''
        st = a.stats
        print(f'    {a.name_ru:<20} ptn={a.ptn} qlt={a.quality} {a.buyout_price:>10,} h={st.get("health_bonus",0):.1f} b={st.get("bullet_dmg_factor",0):.1f} he={st.get("heal_efficiency",0):.1f} mw={st.get("max_weight_bonus",0):.1f} psy={st.get("psycho_accumulation",0):.2f} rad={st.get("radiation_accumulation",0):.2f}{tag}')

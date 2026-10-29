#!/usr/bin/env python3
"""Generate data/artifact_stats.json — pre-computed stat values per quality tier.

Pre-computes at: effectiveness=100 (raw), ptn=0 (no precision bonus).
At lookup time (4 independent cases — is_accumulation and is_positive are independent)
  accum + positive     value * calc_ptn_bonus(ptn)           [ptn applies, eff does NOT]
  accum + negative     value as-is                           [no ptn, no eff]
  non-accum + positive value * (eff/100) * calc_ptn_bonus(ptn)  [both ptn and eff apply]
  non-accum + negative value * (eff/100)                     [eff only, no ptn]
"""
import json, sqlite3, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.builder import (
    _calc_stat_value, _to_full_key, _is_accumulation,
    TIER_TO_RARITY, DANGER_THRESHOLDS,
)
from app.api_client import calc_ptn_bonus  # noqa: used in docstring

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'stalzone.db')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'artifact_stats.json')

# Tier midpoints — user-specified ranges
TIER_MIDPOINTS = {
    0: 92.5,   # Обычный [85;100]
    1: 107.5,  # Необычный [100;115]
    2: 122.5,  # Особый [115;130]
    3: 137.5,  # Редкий [130;145]
    4: 152.5,  # Исключительный [145;160]
    5: 167.5,  # Легендарный [160;175]
}
TIER_NAMES = {
    0: "common", 1: "uncommon", 2: "special",
    3: "rare", 4: "exceptional", 5: "legendary",
}
TIER_RANGES = {
    0: [85, 100], 1: [100, 115], 2: [115, 130],
    3: [130, 145], 4: [145, 160], 5: [160, 175],
}


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM artefacts WHERE id NOT IN ('9n7z', 'azq5t9so') ORDER BY name_ru")
    artifacts = cur.fetchall()
    conn.close()

    result = {}
    stats_n = 0
    bonus_n = 0

    for art in artifacts:
        art_id = art['id']
        base_stats = json.loads(art['stats']) if art['stats'] else []
        bonus_stats = json.loads(art['bonus_stats']) if art['bonus_stats'] else []

        entry = {
            "name_ru": art['name_ru'],
            "name_en": art['name_en'],
            "category": art['category'],
            "weight": art['weight'],
            "tiers": {},
        }

        # Pre-compute base stats per tier
        for tier, midpoint in TIER_MIDPOINTS.items():
            tier_name = TIER_NAMES[tier]
            rarity = TIER_TO_RARITY[tier]
            tier_data = {
                "art_percent": midpoint,
                "range": TIER_RANGES[tier],
                "stats": {},
            }
            for stat in base_stats:
                prop_id = stat.get('prop_id', '')
                full_key = _to_full_key(prop_id)
                is_positive = stat.get('is_positive', True)
                is_accum = _is_accumulation(full_key)
                val = _calc_stat_value(stat, midpoint, level=0,
                                       rarity=rarity, effectiveness=100.0)
                tier_data["stats"][prop_id] = {
                    "value": round(val, 4),
                    "name_ru": stat.get('name_ru', ''),
                    "is_positive": is_positive,
                    "is_accumulation": is_accum,
                }
                stats_n += 1
            entry["tiers"][tier_name] = tier_data

        # Pre-compute bonus stats per tier
        if bonus_stats:
            entry["bonus_stats"] = []
            for bs in bonus_stats:
                prop_id = bs.get('prop_id', '')
                full_key = _to_full_key(prop_id)
                is_positive = bs.get('is_positive', True)
                is_accum = _is_accumulation(full_key)
                unlock_ptn = bs.get('unlock_ptn', 0)
                bonus_entry = {
                    "prop_id": prop_id,
                    "name_ru": bs.get('name_ru', ''),
                    "unlock_ptn": unlock_ptn,
                    "is_positive": is_positive,
                    "is_accumulation": is_accum,
                    "tiers": {},
                }
                for tier, midpoint in TIER_MIDPOINTS.items():
                    tier_name = TIER_NAMES[tier]
                    rarity = TIER_TO_RARITY[tier]
                    val = _calc_stat_value(bs, midpoint, level=0,
                                           rarity=rarity, effectiveness=100.0)
                    bonus_entry["tiers"][tier_name] = round(val, 4)
                    bonus_n += 1
                entry["bonus_stats"].append(bonus_entry)

        # Danger thresholds for accumulation stats (carryability reference)
        entry["danger_thresholds"] = {}
        for stat in base_stats:
            prop_id = stat.get('prop_id', '')
            full_key = _to_full_key(prop_id)
            if full_key in DANGER_THRESHOLDS:
                entry["danger_thresholds"][prop_id] = DANGER_THRESHOLDS[full_key]

        result[art_id] = entry

    # Write
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Generated {OUT_PATH}")
    print(f"  Artifacts: {len(result)}")
    print(f"  Base stat entries: {stats_n}")
    print(f"  Bonus stat entries: {bonus_n}")

    # Verify: Бубльгум
    s = result.get('qodk', {})
    print(f"\n=== Бубльгум (qodk) ===")
    for tn in ['common','uncommon','special','rare','exceptional','legendary']:
        t = s['tiers'][tn]
        b = t['stats'].get('bullet_dmg_factor',{}).get('value',0)
        th = t['stats'].get('thermal_accumulation',{}).get('value',0)
        print(f"  {tn:12s} art%={t['art_percent']:5.1f}  bullet={b:6.2f}  thermal={th:.4f}")

    print(f"\n  Bonus stats:")
    for bs in s.get('bonus_stats',[]):
        vals = {t: round(v,2) for t,v in bs['tiers'].items()}
        print(f"    {bs['prop_id']:25s} unlock={bs['unlock_ptn']:>2}  is_pos={bs['is_positive']}  tiers={vals}")

    print(f"  Danger thresholds: {s.get('danger_thresholds')}")

    # Verify: Проклятая роза (has radiation)
    s2 = result.get('x3lz', {})
    if s2:
        print(f"\n=== Проклятая роза ({list(result.keys())[[k for k in result].index('x3lz') if 'x3lz' in result else 0]}) ===")
        for tn in ['common','rare','legendary']:
            t = s2['tiers'][tn]
            b = t['stats'].get('bullet_dmg_factor',{}).get('value',0)
            r = t['stats'].get('radiation_accumulation',{}).get('value',0)
            print(f"  {tn:12s} art%={t['art_percent']:5.1f}  bullet={b:6.2f}  radiation={r:.4f}")

    fsize = os.path.getsize(OUT_PATH)
    print(f"\nFile size: {fsize:,} bytes ({fsize/1024:.1f} KB)")


if __name__ == '__main__':
    main()

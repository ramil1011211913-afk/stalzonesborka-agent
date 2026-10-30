"""
STALCRAFT Artifact Build Optimizer — v2.0 (clean rewrite).

Math authority: stalzone.wiki calculator JS (data/wiki_calc.js).
All constants and the stat pipeline are extracted verbatim from that JS;
the game handbook (modassets/.../stalkerguide) corroborates every rule.

Pipeline (mirrors JS calculateResults → counterStatsByContainer(sumArtefactProperties(...))):
  1. compute_artifact_stats: per-artifact raw stat values (effectiveness in n(); ptn mult
     for is_positive=true only). Container protection is NOT applied here.
  2. _process_combo: sum raw stats across artifacts → recompute is_positive from the sum
     (canBePositiveAndNegative rules) → apply container protection to is_positive=false,
     statsContCanProtect, origin=artefact only → add container bonus stats →
     recalculate stopping_protection once → score → check carryability.

Optimizer design (JS has no optimizer; this is our spec — REBOOT_PROMPT.md v2.0):
  - art_percent = MAX of the quality tier (115 for qlt=1, 130 for qlt=2, ...).
  - ptn=0 lots: base stats only, NO bonus simulation. ptn>0: real unlocked bonuses.
  - Bonus combos: smart filter — only combos containing priority stats, max 5 per lot.
  - Scoring: weighted sum; accumulation stats inverted (negative=good). No counter-art bonus.
  - Counter-arts: auto-candidates via separate list, no score bonus.
  - Duplicates: unique item_id by default; allow_duplicates option.
  - Excluded: Рубик (9n7z, random 24 bonuses), Регалий (azq5t9so, clan currency — not artifact).
"""
import itertools as _it
import json
import logging
from dataclasses import dataclass, field, replace
import math
import time
from copy import deepcopy
from math import comb as _comb

from .api_client import QUALITY_NAMES, QUALITY_RANGES, calc_ptn_bonus
from .config import Config
from .database import get_connection
from ._objective import Objective, derive_scales
from ._physical_items import is_available, physical_key, owned_identity

log = logging.getLogger(__name__)

# Artifacts excluded from optimizer (not usable as build artifacts)
# 9n7z  Рубик   — fully random 24 bonus options (combinatorial explosion)
# azq5t9so Регалий — not an artifact, clan creation currency
EXCLUDED_ARTIFACTS = {"9n7z", "azq5t9so"}

# ─────────────────────────── Constants (from JS) ────────────────────────────
# accumulationStats — use effectiveness=100 in n()
ACCUMULATION_STATS = {
    "stalker.artefact_properties.factor.radiation_accumulation",
    "stalker.artefact_properties.factor.biological_accumulation",
    "stalker.artefact_properties.factor.psycho_accumulation",
    "stalker.artefact_properties.factor.thermal_accumulation",
    "stalker.artefact_properties.factor.frost_accumulation",
}

# statsContCanProtect — container protection applies to these (NO frost — confirmed from JS)
STATS_CONT_CAN_PROTECT = {
    "stalker.artefact_properties.factor.radiation_accumulation",
    "stalker.artefact_properties.factor.biological_accumulation",
    "stalker.artefact_properties.factor.psycho_accumulation",
    "stalker.artefact_properties.factor.thermal_accumulation",
}

# canBePositiveAndNegative — is_positive recomputed from summed value
# canBePositiveAndNegative: is_positive = (summed <= 0)  → negative/zero = protection
# all others:               is_positive = (summed > 0)  → positive = good
CAN_BE_POSITIVE_AND_NEGATIVE = {
    "stalker.artefact_properties.factor.radiation_accumulation",
    "stalker.artefact_properties.factor.biological_accumulation",
    "stalker.artefact_properties.factor.psycho_accumulation",
    "stalker.artefact_properties.factor.bleeding_accumulation",
    "stalker.artefact_properties.factor.thermal_accumulation",
    "stalker.artefact_properties.factor.frost_accumulation",
    "stalker.artefact_properties.factor.combustion_accumulation",
    "stalker.artefact_properties.factor.recoil_bonus",
    "stalker.artefact_properties.factor.wiggle_bonus",
}

# LimitValuesRecord — base output rate of each infection (per second).
# Carryable when accumulation <= this rate. frost is 20.0 (twice the others).
# Patch [30.09.2026]: all accumulation values ×20, thresholds scaled accordingly.
DANGER_THRESHOLDS = {
    "stalker.artefact_properties.factor.radiation_accumulation": 10.0,
    "stalker.artefact_properties.factor.biological_accumulation": 10.0,
    "stalker.artefact_properties.factor.psycho_accumulation": 10.0,
    "stalker.artefact_properties.factor.thermal_accumulation": 10.0,
    "stalker.artefact_properties.factor.frost_accumulation": 20.0,
}

# artefactRarityIndex — "rarity.ordinary" is absent → None → n_steps fallback
_RARITY_INDEX = {
    "rarity.unordinary": 0, "rarity.special": 1, "rarity.rare": 2,
    "rarity.exclusive": 3, "rarity.legendary": 4, "rarity.unique": 5,
}

# Rarity quality ranges (for the d==100 && rarity==unordinary special case only)
_RARITY_MAP = [
    (100, 114.99, "rarity.unordinary"),
    (115, 129.99, "rarity.special"),
    (130, 144.99, "rarity.rare"),
    (145, 159.99, "rarity.exclusive"),
    (160, 174.99, "rarity.legendary"),
    (175, 190, "rarity.unique"),
]

# Short prop_id → full API key
_SHORT_TO_FULL = {
    "radiation_accumulation": "stalker.artefact_properties.factor.radiation_accumulation",
    "biological_accumulation": "stalker.artefact_properties.factor.biological_accumulation",
    "psycho_accumulation": "stalker.artefact_properties.factor.psycho_accumulation",
    "thermal_accumulation": "stalker.artefact_properties.factor.thermal_accumulation",
    "frost_accumulation": "stalker.artefact_properties.factor.frost_accumulation",
    "health_bonus": "stalker.artefact_properties.factor.health_bonus",
    "stamina_bonus": "stalker.artefact_properties.factor.stamina_bonus",
    "stamina_regeneration_bonus": "stalker.artefact_properties.factor.stamina_regeneration_bonus",
    "bullet_dmg_factor": "stalker.artefact_properties.factor.bullet_dmg_factor",
    "tear_dmg_factor": "stalker.artefact_properties.factor.tear_dmg_factor",
    "explosion_dmg_factor": "stalker.artefact_properties.factor.explosion_dmg_factor",
    "burn_dmg_factor": "stalker.artefact_properties.factor.burn_dmg_factor",
    "electra_dmg_factor": "stalker.artefact_properties.factor.electra_dmg_factor",
    "stopping_protection": "stalker.artefact_properties.factor.stopping_protection",
    "speed_modifier": "stalker.artefact_properties.factor.speed_modifier",
    "sprint_speed_modifier": "stalker.artefact_properties.factor.sprint_speed_modifier",
    "max_weight_bonus": "stalker.artefact_properties.factor.max_weight_bonus",
    "regeneration_bonus": "stalker.artefact_properties.factor.regeneration_bonus",
    "heal_efficiency": "stalker.artefact_properties.factor.heal_efficiency",
    "artefakt_heal": "stalker.artefact_properties.factor.artefakt_heal",
    "bleeding_accumulation": "stalker.artefact_properties.factor.bleeding_accumulation",
    "bleeding_protection": "stalker.artefact_properties.factor.bleeding_protection",
    "electrical_accumulation": "stalker.artefact_properties.factor.electrical_accumulation",
    "gravitational_accumulation": "stalker.artefact_properties.factor.gravitational_accumulation",
    "chemical_accumulation": "stalker.artefact_properties.factor.chemical_accumulation",
    "combustion_accumulation": "stalker.artefact_properties.factor.combustion_accumulation",
    "radiation_protection": "stalker.artefact_properties.factor.radiation_protection",
    "biological_protection": "stalker.artefact_properties.factor.biological_protection",
    "psycho_protection": "stalker.artefact_properties.factor.psycho_protection",
    "thermal_protection": "stalker.artefact_properties.factor.thermal_protection",
    "recoil_bonus": "stalker.artefact_properties.factor.recoil_bonus",
    "wiggle_bonus": "stalker.artefact_properties.factor.wiggle_bonus",
    "reaction_to_burn": "stalker.artefact_properties.factor.reaction_to_burn",
    "reaction_to_tear": "stalker.artefact_properties.factor.reaction_to_tear",
    "reaction_to_electroshock": "stalker.artefact_properties.factor.reaction_to_electroshock",
    "reaction_to_chemical_burn": "stalker.artefact_properties.factor.reaction_to_chemical_burn",
    "chemical_burn_dmg_factor": "stalker.artefact_properties.factor.chemical_burn_dmg_factor",
    "blocking_damage": "stalker.artefact_properties.factor.blocking_damage",
    "recharge": "stalker.artefact_properties.factor.recharge",
    "trigger_damage": "stalker.artefact_properties.factor.trigger_damage",
    "electro_protection": "stalker.artefact_properties.factor.electra_dmg_factor",
    "burn_protection": "stalker.artefact_properties.factor.burn_dmg_factor",
}
# Auction API short bonus name → DB prop_id (upper)
BONUS_NAME_MAP = {
    "ARTEFAKT_HEAL": "ARTEFAKT_HEAL",
    "BIOLOGICAL_ACC": "BIOLOGICAL_ACCUMULATION",
    "BIOLOGICAL_PROTECTION": "BIOLOGICAL_PROTECTION",
    "BLEEDING_ACC": "BLEEDING_ACCUMULATION",
    "BLEEDING_PROTECTION": "BLEEDING_PROTECTION",
    "BULLET_DMG": "BULLET_DMG_FACTOR",
    "BURN_DMG": "BURN_DMG_FACTOR",
    "COMBUSTION_ACC": "COMBUSTION_ACCUMULATION",
    "ELECTRA_DMG": "ELECTRA_DMG_FACTOR",
    "EXPLOSION_DMG": "EXPLOSION_DMG_FACTOR",
    "HEALTH_BONUS": "HEALTH_BONUS",
    "HEAL_EFFICIENCY": "HEAL_EFFICIENCY",
    "MAX_WEIGHT_BONUS": "MAX_WEIGHT_BONUS",
    "PSYCHO_ACC": "PSYCHO_ACCUMULATION",
    "PSYCHO_PROTECTION": "PSYCHO_PROTECTION",
    "RADIATION_ACC": "RADIATION_ACCUMULATION",
    "RADIATION_PROTECTION": "RADIATION_PROTECTION",
    "RECOIL_BONUS": "RECOIL_BONUS",
    "REGENERATION_BONUS": "REGENERATION_BONUS",
    "SPEED_MOD": "SPEED_MODIFIER",
    "SPRINT_SPEED_MOD": "SPRINT_SPEED_MODIFIER",
    "STAMINA_BONUS": "STAMINA_BONUS",
    "STAMINA_REGENERATION": "STAMINA_REGENERATION_BONUS",
    "STOPPING_PROTECTION": "STOPPING_PROTECTION",
    "TEAR_DMG": "TEAR_DMG_FACTOR",
    "THERMAL_ACC": "THERMAL_ACCUMULATION",
    "THERMAL_PROTECTION": "THERMAL_PROTECTION",
    "WIGGLE_BONUS": "WIGGLE_BONUS",
}

# Quality tier → rarity name (from stalzone.wiki JS ArtefactRarityEnum).
# CRITICAL: determines step interpolation in _calc_stat_value for is_positive=false, d>100.
# "rarity.ordinary" (tier 0) is NOT in _RARITY_INDEX → None → n_steps fallback.
# Tiers 1-5 ARE in the index and give correct step values at quality boundaries.
TIER_TO_RARITY = {
    0: "rarity.ordinary",
    1: "rarity.unordinary",
    2: "rarity.special",
    3: "rarity.rare",
    4: "rarity.exclusive",
    5: "rarity.legendary",
}

# Safety cap on combinations per search
_MAX_COMBOS = 50_000_000

# Staleness threshold in hours — if auction data is older, trigger refresh
AUCTION_STALE_HOURS = 6

# Cache for emission schedule (5 min TTL to avoid API call on every build request)
_emission_cache: dict = {"data": None, "ts": 0}

def check_auction_freshness(region: str | None = None) -> dict:
    """Check if auction data is fresh enough for building.

    Returns dict with:
        is_fresh (bool): True if data collected within AUCTION_STALE_HOURS
                         AND after the last emission (emissions bring new lots).
        last_collected (str|None): ISO timestamp of most recent lot
        age_hours (float): hours since most recent lot (inf if no data)
        lot_count (int): total active lots in DB for this region
        last_emission (dict|None): {previousStart, previousEnd} from eAPI
    """
    from datetime import datetime, timezone
    import time as _time
    region = region or Config.REGION
    with get_connection() as conn:
        row = conn.execute(
            "SELECT MAX(collected_at) as max_ts, COUNT(*) as cnt "
            "FROM auction_lots WHERE region=? AND buyout_price > 0",
            (region,)
        ).fetchone()
    last_ts = row["max_ts"] if row else None
    lot_count = row["cnt"] if row else 0
    if not last_ts or lot_count == 0:
        return {"is_fresh": False, "last_collected": None,
                "age_hours": float("inf"), "lot_count": 0,
                "last_emission": None}
    try:
        last_dt = datetime.fromisoformat(last_ts)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_hours = (now - last_dt).total_seconds() / 3600
    except (ValueError, TypeError):
        return {"is_fresh": False, "last_collected": last_ts,
                "age_hours": float("inf"), "lot_count": lot_count,
                "last_emission": None}
    # Emission check — cached 5 min to avoid API call on every build request.
    now_ts = _time.time()
    emission_info = _emission_cache["data"]
    if _emission_cache["ts"] == 0 or now_ts - _emission_cache["ts"] > 300:
        try:
            from .api_client import StalzoneAPI
            api = StalzoneAPI(region)
            emission_info = api.get_emission()
            _emission_cache["data"] = emission_info
            _emission_cache["ts"] = now_ts
        except Exception:
            pass
    is_fresh = age_hours < AUCTION_STALE_HOURS
    # If emission ended AFTER last collect → new lots likely appeared → stale.
    if emission_info and emission_info.get("previousEnd"):
        try:
            em_end = datetime.fromisoformat(
                emission_info["previousEnd"].replace("Z", "+00:00"))
            if last_dt < em_end:
                is_fresh = False
        except (ValueError, TypeError):
            pass
    return {
        "is_fresh": is_fresh,
        "last_collected": last_ts,
        "age_hours": round(age_hours, 1),
        "lot_count": lot_count,
        "last_emission": emission_info,
    }


# ─────────────────────────── Helpers ─────────────────────────────────────────

def _normalize_bonus_id(raw: str) -> str:
    """Convert API short bonus name to DB prop_id (upper) for matching."""
    return BONUS_NAME_MAP.get(raw, raw)

def _quality_tier(art_percent: float) -> int:
    """Determine quality tier (0-5) from art_percent, matching JS quality ranges.
    Boundaries: 85-100=0, 100-115=1, 115-130=2, 130-145=3, 145-160=4, 160-175=5.
    At exact boundary (e.g. 160), returns the HIGHER tier (5, not 4).
    But JS uses the rarity from the API, which assigns the LOWER tier at boundary.
    So we match JS: d=160 → tier 4 (exclusive), not 5 (legendary).
    """
    if art_percent <= 100: return 0
    if art_percent <= 115: return 1
    if art_percent <= 130: return 2
    if art_percent <= 145: return 3
    if art_percent <= 160: return 4
    return 5


def _to_full_key(prop_id: str) -> str:
    """Convert short prop_id to full stalker.artefact_properties.factor.* key."""
    if prop_id.startswith("stalker."):
        return prop_id
    return _SHORT_TO_FULL.get(prop_id, prop_id)


def _is_accumulation(full_key: str) -> bool:
    return full_key in ACCUMULATION_STATS


def _is_counter_art(opt) -> bool:
    """Check if an option is a counter-art: has negative accumulation (protection)."""
    for pid, val in opt.stats.items():
        fkey = _to_full_key(pid)
        if fkey in DANGER_THRESHOLDS and val < 0:
            return True
    return False

def _get_max_carryable_count(opt, n_slots: int) -> int:
    """Возвращает максимальное кол-во штук артефакта без перегруза по заражению."""
    max_count = n_slots
    for pid, val in opt.stats.items():
        fkey = _to_full_key(pid)
        if fkey in DANGER_THRESHOLDS and val > 0:
            limit = DANGER_THRESHOLDS[fkey]
            allowed = int(limit // val) if val > 0 else n_slots
            max_count = min(max_count, max(1, allowed))
    return max_count

def _can_protect(full_key: str) -> bool:
    return full_key in STATS_CONT_CAN_PROTECT


# ─────────────────────────── Stat math (from JS) ────────────────────────────

def _n(stat_min: float, stat_max: float, t: float = 1.0) -> tuple[float, float]:
    """Normalize stat values, scaling by effectiveness factor t.

    Mirrors JS n(e, t):
      r = max, a = min; n=max(r,a), o=min(r,a)
      if r<=0 and a<=0: swap → n=min(r,a) [more negative], o=max(r,a)
      return {max: n*t, min: o*t, baseMax: n, baseMin: o}
    """
    r = float(stat_max)
    a = float(stat_min)
    n_val = max(r, a)
    o_val = min(r, a)
    if r <= 0 and a <= 0:
        n_val = min(r, a)
        o_val = max(r, a)
    return n_val * t, o_val * t


def _calc_stat_value(stat: dict, quality: float, level: int,
                     rarity: str = "rarity.ordinary",
                     effectiveness: float = 100.0) -> float:
    """Calculate a single stat value. Mirrors JS `calculation` function.

    is_positive=true:  max_v * (quality/100) * (1 + 2*level/100)
    is_positive=false, d<=100:
        d==100 && rarity==unordinary: 0.9*max + (max-0.9*max)/100*((d-(100+10*f))*10)
        else: min + (max-min)/100*d
    is_positive=false, d>100:
        a = rarityIndex[rarity] (None for ordinary → n_steps fallback)
        o = max(0, min(a ?? n_steps, 5)); snap o at tier boundaries
        i = max(0, min(d-(100+15*o), 15))/15
        calc = 0.85*max + (max-0.85*max)*i
    """
    full_key = _to_full_key(stat.get("prop_id", ""))
    is_positive = stat.get("is_positive", True)
    is_accum = _is_accumulation(full_key)
    t = 1.0 if is_accum else effectiveness / 100.0

    max_v, min_v = _n(stat.get("min", 0.0), stat.get("max", 0.0), t)

    if is_positive:
        ptn_mult = calc_ptn_bonus(level)
        return max_v * (quality / 100.0) * ptn_mult

    # is_positive=false
    d = quality
    f = 0
    for lo, hi, name in _RARITY_MAP:
        if lo < d <= hi:
            f = _RARITY_INDEX[name]
            break

    if d <= 100:
        if d == 100 and rarity == "rarity.unordinary":
            f = _RARITY_INDEX[rarity]
            a = 0.9 * max_v
            return a + (max_v - a) / 100.0 * ((d - (100 + 10 * f)) * 10)
        return min_v + (max_v - min_v) / 100.0 * d

    # d > 100
    a = _RARITY_INDEX.get(rarity)
    n_steps = int((d - 100) // 15)
    o = max(0, min(a if a is not None else n_steps, 5))
    if int(d) in (115, 130, 145, 160, 175, 190):
        o = a if a is not None else n_steps
    i = max(0, min(d - (100 + 15 * o), 15)) / 15.0
    s2 = 0.85 * max_v
    return s2 + (max_v - s2) * i


def _recompute_is_positive(full_key: str, summed_val: float) -> bool:
    """Recompute is_positive after summation (JS sumArtefactProperties).

    canBePositiveAndNegative: is_positive = (summed <= 0)  → protection
    all others:               is_positive = (summed > 0)
    """
    if full_key in CAN_BE_POSITIVE_AND_NEGATIVE:
        return summed_val <= 0
    return summed_val > 0


def _apply_build_container_protection(stats: dict, stat_origins: dict,
                                      inner_protection: float) -> dict:
    """Apply container protection at BUILD level after summation (JS counterStatsByContainer).

    Applies (1-prot/100) ONLY to:
      is_positive=false AND statsContCanProtect AND origin=artefact AND !armorStat
    """
    result = {}
    for prop_id, val in stats.items():
        full_key = _to_full_key(prop_id)
        is_pos = _recompute_is_positive(full_key, val)
        if (not is_pos
                and _can_protect(full_key)
                and stat_origins.get(prop_id, "artefact") == "artefact"):
            result[prop_id] = val * (1.0 - inner_protection / 100.0)
        else:
            result[prop_id] = val
    return result


def _recalculate_stopping_protection(stats: dict) -> dict:
    """Recalculate stopping_protection from summed tear_dmg_factor (JS recalculateStoppingProtection).

    stopping = (1 - 100/(100+tear) * (1 - stopping/100)) * 100
    """
    tear = stats.get("tear_dmg_factor", 0.0)
    existing = stats.get("stopping_protection", 0.0)
    if tear != 0 or existing != 0:
        i = ((1.0 - 100.0 / (100.0 + tear) * (1.0 - existing / 100.0)) * 100.0
             if 100.0 + tear != 0 else math.nan)
        stats["stopping_protection"] = i
    return stats


_REACTION_STAT_KEYS = {
    "reaction_to_burn", "reaction_to_chemical_burn",
    "reaction_to_electroshock", "reaction_to_tear",
}


def apply_reaction_bonuses(stats: dict) -> tuple[dict, float]:
    """Calculate reaction bonus sum (JS applyReactionBonuses).

    The wiki shows health_bonus and stamina_regeneration_bonus SEPARATELY from
    reaction bonuses. The reaction sum is used internally for EHP/HPS only.
    """
    reaction_sum = (
        stats.get("reaction_to_burn", 0.0) +
        stats.get("reaction_to_chemical_burn", 0.0) +
        stats.get("reaction_to_electroshock", 0.0) +
        stats.get("reaction_to_tear", 0.0)
    )
    # Don't modify stats — reaction bonus is separate, used for EHP/HPS only.
    return stats, reaction_sum


def calculate_effective_health(stats: dict) -> float:
    """Calculate effective health (JS calculateEffectiveHealth).

    EHP = (1 + health_bonus/100) * (100 + bullet_dmg_factor)

    health_bonus is a percentage modifier. bullet_dmg_factor is damage reduction.
    Returns bullet-equivalent damage-to-kill at base HP 100, not ordinary HP.
    """
    from ._survivability_math import calculate_effective_health as calculate
    return calculate(stats)


def calculate_healing_per_second(stats: dict) -> float:
    """Calculate healing per second (JS calculateHealingPerSecond).

    HPS = (regeneration_bonus + 2.5) / 5 + artefakt_heal * (1 + heal_efficiency/100)

    Returns the current Wiki percent-per-second display, not absolute HP/s.
    heal_efficiency multiplies the periodic artefakt_heal term only.
    """
    from ._survivability_math import calculate_healing_per_second as calculate
    return calculate(stats)



def compute_artifact_stats(base_stats: list, bonus_stats: list, bonus_unlocked_ids: list,
                           art_percent: float, ptn: int, qlt_min: float, qlt_range: float,
                           stats_random: float | None,
                           inner_protection: float = 0.0, effectiveness: float = 100.0,
                           other_acc: dict | None = None
                           ) -> tuple[dict, dict, dict, float | None]:
    """Compute all stats for one artifact. Container protection is deferred to build level.

    Pipeline (mirrors JS calc_stat only):
      1. _calc_stat_value: raw value with n() swap, ptn_mult for is_positive=true only,
         effectiveness applied in n() for non-accumulation stats.
      2. (container protection deferred to _process_combo)
      3. (stopping_protection recalc deferred to _process_combo)

    ptn=0 (unresearched): only base stats. No bonus simulation.
    ptn>0: unlock bonuses at ptn>=5 → [0], ptn>=10 → [0,1], ptn>=15 → [0,1,2].
    For auction lots with bonus_properties, use those directly.

    Returns (stats_container, stats_raw, stats_names, recommended_percent).
    stats_container == stats_raw (protection deferred). recommended_percent always None.
    """
    _ = (inner_protection, other_acc, qlt_range, stats_random)  # unused, kept for API compat
    # Rarity from qlt_min (the lot's actual quality tier, not derived from art_percent).
    # Common (qlt_min=85): rarity.ordinary — avoids d=100 special case for Common artifacts.
    # Uncommon+ (qlt_min>=100): correct rarity for step-interpolation boundaries.
    # FIX: at d=145 Rare (qlt_min=130), step gives max_v (step end, i=1.0),
    # not 0.85×max_v (n_steps fallback gives step start of Exceptional, i=0).
    # Previous code hardcoded rarity.ordinary for ALL tiers — wrong at boundaries ≥115.
    _QLT_MIN_TO_RARITY = {
        100: "rarity.unordinary", 115: "rarity.special", 130: "rarity.rare",
        145: "rarity.exclusive", 160: "rarity.legendary",
    }
    rarity = _QLT_MIN_TO_RARITY.get(int(qlt_min), "rarity.ordinary") if qlt_min and qlt_min >= 100 else "rarity.ordinary"
    stats_raw: dict[str, float] = {}
    stats_names: dict[str, str] = {}

    unlocked_set = {_normalize_bonus_id(b.upper()) for b in (bonus_unlocked_ids or [])}

    def _process_stat_list(stats_list, is_bonus=False):
        for s in stats_list:
            prop_id = s.get("prop_id", "")
            if is_bonus:
                # Bonus only counts if it's in the unlocked set
                bonus_id = _normalize_bonus_id(prop_id.upper())
                if bonus_id not in unlocked_set:
                    continue
            raw_val = _calc_stat_value(s, art_percent, ptn, rarity, effectiveness)
            stats_raw[prop_id] = stats_raw.get(prop_id, 0.0) + raw_val
            stats_names[prop_id] = s.get("name_ru", prop_id)

    _process_stat_list(base_stats, is_bonus=False)
    _process_stat_list(bonus_stats, is_bonus=True)
    from ._derived_stats import add_names, augment
    augment(stats_raw)
    add_names(stats_raw, stats_names)

    return dict(stats_raw), dict(stats_raw), stats_names, None


def check_carryability(combined_stats: dict) -> tuple[bool, list]:
    """Check if build is carryable. Only POSITIVE accumulation exceeds the output rate.

    val <= threshold → carryable. Negative = counter-art (safe).
    """
    warnings = []
    for prop_id, val in combined_stats.items():
        full_key = _to_full_key(prop_id)
        if full_key not in DANGER_THRESHOLDS:
            continue
        threshold = DANGER_THRESHOLDS[full_key]
        if val > threshold:
            warnings.append(f"Носибельность превышена: {prop_id} = {val:.4f} (макс {threshold})")
    return len(warnings) == 0, warnings

def _artifact_danger_bounds(option, container):
    cached = getattr(option, "_danger_bounds", None)
    if cached is not None:
        return cached
    qr = QUALITY_RANGES.get(option.quality, QUALITY_RANGES[0])
    rows = [compute_artifact_stats(
        option.base_stats, option.bonus_stats, option.unlocked_bonus_ids,
        percent, option.ptn, qr["min"], qr["range"], None,
        container.inner_protection if container else 0, container.effectiveness if container else 100,
    )[0] for percent in (qr["min"], qr["min"] + qr["range"])]
    keys = [key.replace("stalker.artefact_properties.factor.", "") for key in DANGER_THRESHOLDS]
    return ({key: min(row.get(key, 0) for row in rows) for key in keys},
            {key: max(row.get(key, 0) for row in rows) for key in keys})


def analyze_build_carryability(build) -> dict:
    """Report the existing per-artifact endpoint envelope in every search mode.

    Sum the lower/upper danger endpoints, then apply container protection,
    container bonuses and fixed armor exactly as in guaranteed search.
    These are endpoint envelopes, not a single shared artifact percentage.
    """
    analysis = {}
    all_carryable_max = True
    armor = getattr(build, "_armor", None) or getattr(build.container, "_armor", None)
    base = getattr(build.container, "_base_container", build.container)
    low, high = {}, {}
    for option in build.artifacts:
        values = _artifact_danger_bounds(option, base)
        for full_key in DANGER_THRESHOLDS:
            key = full_key.replace("stalker.artefact_properties.factor.", "")
            low[key] = low.get(key, 0) + values[0].get(key, 0)
            high[key] = high.get(key, 0) + values[1].get(key, 0)
    origins = {key: "artefact" for key in low}
    endpoints = [_apply_build_container_protection(row, origins, base.inner_protection if base else 0)
                 for row in (low, high)]
    for row in endpoints:
        for key, value in (base.bonus_stats if base else {}).items():
            row[key] = row.get(key, 0) + (value.get("value", 0) if isinstance(value, dict) else value)
        for key, value in _armor_additions(armor).items():
            row[key] = row.get(key, 0) + value

    for stat_key_full, threshold in DANGER_THRESHOLDS.items():
        short_key = stat_key_full.replace("stalker.artefact_properties.factor.", "")
        val = build.combined_stats.get(short_key, 0)
        if val <= 0 and endpoints[1].get(short_key, 0) <= 0:
            continue

        is_frost = short_key == "frost_accumulation"
        danger_min = endpoints[0].get(short_key, 0)
        danger_max = max(val, endpoints[1].get(short_key, 0))

        carryable_min = danger_min <= threshold
        carryable_max = danger_max <= threshold
        if not carryable_max:
            all_carryable_max = False

        # Separate artifact percentages/counter-effects have no defined single
        # inverse here. Keep the nullable field instead of fabricating a cutoff.
        art_pct_threshold = None

        name_ru = build.combined_stats_names.get(short_key, short_key)
        analysis[short_key] = {
            "name_ru": name_ru,
            "danger_min": round(danger_min, 4),
            "danger_max": round(danger_max, 4),
            "threshold": threshold,
            "carryable_min": carryable_min,
            "carryable_max": carryable_max,
            "art_pct_threshold": art_pct_threshold,
            "protected": not is_frost,
        }

    return {"stats": analysis, "guaranteed_carryable": all_carryable_max}


def _priority_terms(priorities: dict, stat_max=None) -> tuple:
    objective = Objective.create(priorities, stat_max)
    return tuple(zip(objective.keys, objective.coefficients))


def _priority_value(stats: dict, priorities: dict, stat_max=None) -> float:
    return Objective.create(priorities, stat_max).evaluate(stats)


def _score_build(stats: dict, priorities: dict, stat_max: dict | None = None,
                 is_build: bool = False) -> float:
    """Exact balanced F on final game stats; no hidden EHP/coverage penalties."""
    return Objective.create(priorities, stat_max).evaluate(stats)


def _fixed_objective_scales(options, containers, priorities):
    from ._derived_stats import augment, augment_bonus_stats
    scales = {key: 1.0 for key, weight in priorities.items() if weight > 0}
    if not scales:
        return scales
    for container in containers:
        rows = []
        for option in options:
            qr = QUALITY_RANGES.get(option.quality, QUALITY_RANGES[0])
            stats, _, _, _ = compute_artifact_stats(
                option.base_stats, option.bonus_stats, option.unlocked_bonus_ids,
                option.art_percent, option.ptn, qr['min'], qr['range'], option.stats_random,
                container.inner_protection, container.effectiveness,
            )
            _recalculate_stopping_protection(stats)
            augment(stats)
            rows.append(stats)
        bonuses = {key: value.get('value', 0) if isinstance(value, dict) else value
                   for key, value in container.bonus_stats.items()}
        augment_bonus_stats(bonuses)
        if getattr(container, "_armor", None) is not None and "stopping_protection" in scales:
            normalized_bonus = dict(bonuses)
            if 100 + normalized_bonus.get("tear_dmg_factor", 0) != 0:
                _recalculate_stopping_protection(normalized_bonus)
                bonuses["stopping_protection"] = normalized_bonus.get("stopping_protection", 0)
        bounds = derive_scales(rows, scales, container.slots, bonuses)
        for key in scales:
            scales[key] = max(scales[key], bounds[key])
    return scales


# Alias for backward compat with single-artifact scoring call sites
_score_option = _score_build


def _pareto_frontier(builds: list, objectives: list[str], objective: Objective | None = None) -> list:
    """Extract Pareto frontier: builds not dominated on any objective.

    A build A dominates B if A is >= B on ALL objectives and > B on at least one.
    Returns the list of non-dominated builds (the Pareto frontier).
    """
    if not builds:
        return []
    # Extract objective vectors
    vecs = []
    for i, b in enumerate(builds):
        s = b.combined_stats
        if objective is None:
            vec = tuple(s.get(obj, 0.0) for obj in objectives)
        elif objective.mode == "lexicographic":
            vec = objective.rank(s, b.total_price)
        else:
            vec = tuple(s.get(key, 0.0) * coefficient
                        for key, coefficient in zip(objective.keys, objective.coefficients))
        vecs.append((vec, i))

    frontier = []
    for vec_i, idx_i in vecs:
        dominated = False
        for vec_j, idx_j in vecs:
            if idx_i == idx_j:
                continue
            # j dominates i if j >= i on all and j > i on at least one
            ge_all = all(vj >= vi for vj, vi in zip(vec_j, vec_i))
            gt_any = any(vj > vi for vj, vi in zip(vec_j, vec_i))
            if ge_all and gt_any:
                dominated = True
                break
        if not dominated:
            frontier.append(builds[idx_i])
    return frontier


# ─────────────────────────── Data classes ────────────────────────────────────

@dataclass
class ArtifactOption:
    item_id: str
    name_ru: str
    name_en: str
    subcategory: str
    quality: int
    quality_name: str
    art_percent: float
    buyout_price: int
    ptn: int
    stats: dict
    stats_raw: dict
    stats_names: dict
    unlocked_bonus_ids: list
    icon_url: str = ""
    base_stats: list = field(default_factory=list)
    bonus_stats: list = field(default_factory=list)
    stats_random: float | None = None
    recommended_percent: float | None = None
    art_percent_known: bool = True
    is_owned: bool = False
    hist_avg_price: float = 0.0  # historical avg sold price for (item_id, quality)
    score: float = 0.0
    auction_lot_id: int | None = None
    owned_instance_id: str | None = None
    physical_id: str | None = None


@dataclass
class ContainerOption:
    item_id: str
    name_ru: str
    name_en: str
    item_type: str
    slots: int
    buyout_price: int
    inner_protection: float = 0.0
    effectiveness: float = 0.0
    bonus_stats: dict = field(default_factory=dict)
    icon_url: str = ""


@dataclass
class ArtifactBuild:
    artifacts: list[ArtifactOption] = field(default_factory=list)
    container: ContainerOption | None = None
    total_price: int = 0
    total_score: float = 0.0
    priority_value: float = 0.0
    combined_stats: dict = field(default_factory=dict)
    combined_stats_names: dict = field(default_factory=dict)
    carryable: bool = True
    carryability_warnings: list = field(default_factory=list)
    carryability_analysis: dict = field(default_factory=dict)
    guaranteed_carryable: bool = True
    hidden_effect_warnings: list = field(default_factory=list)
    priorities: dict = field(default_factory=dict)
    objective_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        container_dict = None
        if self.container:
            container_dict = {
                "item_id": self.container.item_id,
                "name_ru": self.container.name_ru,
                "name_en": self.container.name_en,
                "item_type": self.container.item_type,
                "slots": self.container.slots,
                "buyout_price": self.container.buyout_price,
                "inner_protection": round(self.container.inner_protection, 1),
                "effectiveness": round(self.container.effectiveness, 1),
                "bonus_stats": self.container.bonus_stats,
                "icon_url": self.container.icon_url,
            }
        # Extract priority stats for prominent display in UI
        priority_stats = {}
        for pid, weight in self.priorities.items():
            if weight == 0:
                continue
            val = self.combined_stats.get(pid, 0)
            name = self.combined_stats_names.get(pid, pid)
            priority_stats[pid] = {"value": round(val, 4), "weight": weight, "name": name}
        derived_export = {}
        if 'effective_health' in self.combined_stats:
            from ._survivability import REGISTRY, provenance
            derived_export = dict(derived_stats={k:self.combined_stats[k] for k in REGISTRY if k in self.combined_stats},
                                  calculation_provenance=provenance())
        armor_dict = getattr(self, '_armor', None)
        if armor_dict:
            armor_dict = {
                "id": armor_dict["id"],
                "name_ru": armor_dict["name_ru"],
                "rank": armor_dict["rank"],
                "level": armor_dict["level"],
                "stats": {k: round(v, 4) for k, v in armor_dict["stats"].items()},
            }
        return {
            "total_price": self.total_price,
            "artifact_price": self.total_price,
            "budget_scope": "artifacts",
            "total_score": round(self.total_score, 2),
            "priority_value": round(self.priority_value, 4),
            "objective": self.objective_config,
            "priority_stats": priority_stats,
            "combined_stats": {k: round(v, 4) for k, v in self.combined_stats.items()},
            **derived_export,
            "effective_health": round(calculate_effective_health(self.combined_stats), 1),
            "healing_per_second": round(calculate_healing_per_second(self.combined_stats), 1),
            "combined_stats_names": self.combined_stats_names,
            "container": container_dict,
            "armor": armor_dict,
            "carryable": self.carryable,
            "carryability_warnings": self.carryability_warnings,
            "carryability_analysis": self.carryability_analysis,
            "guaranteed_carryable": self.guaranteed_carryable,
            "hidden_effect_warnings": self.hidden_effect_warnings,
            "artifacts": [
                {
                    "item_id": a.item_id,
                    "auction_lot_id": a.auction_lot_id,
                    "owned_instance_id": a.owned_instance_id,
                    "physical_id": a.physical_id,
                    "name_ru": a.name_ru,
                    "name_en": a.name_en,
                    "subcategory": a.subcategory,
                    "quality": a.quality,
                    "quality_name": a.quality_name,
                    "art_percent": round(a.art_percent, 1),
                    **({"calculation": {
                        "mode": "minimum_quality_point_scenario",
                        "scenario_percent": a.art_percent,
                        "source_stored_percent": a._quality_source.art_percent,
                        "source_stats_random": a._quality_source.stats_random,
                    }} if hasattr(a, '_quality_source') else {}),
                    "ptn": a.ptn,
                    "buyout_price": a.buyout_price,
                    "is_owned": a.is_owned,
                    "recommended_percent": round(a.recommended_percent, 1) if a.recommended_percent else None,
                    "stats": {k: round(v, 4) for k, v in a.stats.items() if not k.startswith("__guaranteed_")},
                    "stats_names": a.stats_names,
                    "unlocked_bonus_ids": a.unlocked_bonus_ids,
                    "icon_url": a.icon_url,
                    "bonus_stats": [
                        {"prop_id": b.get("prop_id",""), "name_ru": b.get("name_ru",""),
                         "min": b.get("min",0), "max": b.get("max",0),
                         "unlock_ptn": b.get("unlock_ptn",0),
                         "is_positive": b.get("is_positive", True)}
                        for b in (a.bonus_stats or [])
                    ],
                }
                for a in self.artifacts
            ],
            "tips": _generate_build_tips(self),
        }


# ─────────────────────────── Armor integration ──────────────────────────────

def load_armor(armor_id: str, level: int = 15) -> dict | None:
    """Load armor stats from DB. Returns stats dict or None."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, name_ru, category, rank, level, stats, icon_url FROM armors WHERE id=? AND level=?",
            (armor_id, level)
        ).fetchone()
    if not row:
        return None
    import json as _json
    return {
        "id": row["id"],
        "name_ru": row["name_ru"],
        "category": row["category"],
        "rank": row["rank"],
        "level": row["level"],
        "stats": _json.loads(row["stats"]) if row["stats"] else {},
        "icon_url": row["icon_url"],
    }


ARMOR_ADDITIVE_KEYS = {
    "bullet_dmg_factor", "tear_dmg_factor", "stopping_protection", "explosion_dmg_factor",
    "speed_modifier", "sprint_speed_modifier", "max_weight_bonus", "health_bonus",
    "stamina_bonus", "stamina_regeneration_bonus", "regeneration_bonus", "heal_efficiency",
    "bleeding_accumulation", "radiation_accumulation", "biological_accumulation",
    "psycho_accumulation", "thermal_accumulation", "frost_accumulation",
    "radiation_protection", "biological_protection", "psycho_protection", "thermal_protection",
    "electro_protection", "burn_protection", "bleeding_protection",
    "reaction_to_burn", "reaction_to_chemical_burn", "reaction_to_electroshock", "reaction_to_tear",
}


def _armor_additions(armor):
    return {key: value for key, value in (armor or {}).get("stats", {}).items()
            if key in ARMOR_ADDITIVE_KEYS}


def _armor_containers(containers, armors):
    from ._derived_stats import augment_bonus_stats
    result = []
    for base in containers:
        for armor in armors:
            if armor is None:
                result.append(base)
                continue
            bonuses = dict(base.bonus_stats)
            for key, value in _armor_additions(armor).items():
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("Armor stats must be finite numbers")
                previous = bonuses.get(key, 0)
                previous = previous.get("value", 0) if isinstance(previous, dict) else previous
                bonuses[key] = previous + value
            augment_bonus_stats(bonuses)
            effective = replace(base, bonus_stats=bonuses)
            effective._base_container = base
            effective._armor = armor
            result.append(effective)
    return result


def _without_armor(stats, armor):
    raw = dict(stats)
    if "stopping_protection" in raw:
        raw["stopping_protection"] = 100 - (100 - raw["stopping_protection"]) * (100 + raw.get("tear_dmg_factor", 0)) / 100
    for key, value in _armor_additions(armor).items():
        raw[key] = raw.get(key, 0) - value
    from ._derived_stats import augment
    augment(raw)
    if 'effective_health' in raw:
        from ._survivability import values
        raw.update(values(raw))
    return _recalculate_stopping_protection(raw)


def _decorate_armor(build):
    effective = build.container
    if effective is not None and hasattr(effective, "_base_container"):
        build._armor = effective._armor
        build._guaranteed_search = getattr(effective, "_guaranteed_search", False)
        build._unarmored_stats = _without_armor(build.combined_stats, build._armor)
        build.container = effective._base_container
    for option in build.artifacts:
        if any(key.startswith("__guaranteed_") for key in option.stats):
            option.stats = {key: value for key, value in option.stats.items() if not key.startswith("__guaranteed_")}
            option.stats_raw = {key: value for key, value in option.stats_raw.items() if not key.startswith("__guaranteed_")}
    return build


def _final_valid(build, budget, minimum, maximum, carry, guaranteed, duplicates):
    return (build.total_price <= budget and is_available(build.artifacts, duplicates)
            and all(math.isfinite(value) for value in build.combined_stats.values())
            and all(build.combined_stats.get(key, 0) >= value for key, value in (minimum or {}).items())
            and all(build.combined_stats.get(key, 0) <= value for key, value in (maximum or {}).items())
            and (not carry or build.carryable)
            and (not guaranteed or build.guaranteed_carryable))


def _ordinary_result(result, profiles, complete, budget, minimum, maximum, carry,
                     guaranteed, duplicates, priorities, objective, region, request, primary_pool=(),
                     refine_lexicographic=False):
    """Verify ordinary hits, preserve/refine their incumbent, or rescue an empty result."""
    from . import _feasibility_fallback as rescue
    from ._quality_scenario import current_views
    started = time.monotonic()
    models = []
    # Recheck real auction descriptors, including expiry, even for primary hits.
    # Anonymous programmatic fixtures retain their previous non-market contract.
    def available(builds):
        real = [o for build in builds for o in build.artifacts if o.auction_lot_id is not None]
        valid = {id(o) for o in current_views(real, get_connection, region, _normalize_bonus_id)}
        return [build for build in builds if all(o.auction_lot_id is None or id(o) in valid for o in build.artifacts)]

    if result or primary_pool:
        try:
            result = available(result)
            if not result and primary_pool:
                result = available(primary_pool)
                result.sort(key=objective.sort_key)
                result = result[:max(1, request['top_n'])]
            elif result and primary_pool:
                # Diversity was selected before market validation. An expired
                # anchor must not hide a stronger valid build already found.
                incumbent = min(result, key=objective.sort_key)
                better = [b for b in primary_pool
                          if objective.sort_key(b) < objective.sort_key(incumbent)]
                verified = available(better)
                if verified:
                    best = min(verified, key=objective.sort_key)
                    result = [best] + [b for b in result if b is not best]
                    result = result[:max(1, request['top_n'])]
        except Exception:
            return rescue.SearchResults([], [dict(state='UNKNOWN', reason='market_validation_error')], complete=False)
        if result:
            if refine_lexicographic and objective.mode == 'lexicographic' and not (guaranteed or request.get('conservative')):
                from ._beam_refine import refine_incumbent
                try:
                    best = refine_incumbent(profiles, min(result, key=objective.sort_key),
                        budget, minimum, maximum, carry, duplicates, priorities, objective, region, request)
                    if best is not None:
                        result = [best] + [b for b in result if b is not best]
                        result = result[:max(1, request['top_n'])]
                except Exception:
                    # A secondary strategy must never discard the verified
                    # primary incumbent, including on allocation/solver errors.
                    log.exception('Secondary Beam refinement failed; retaining verified incumbent')
            refinement_models = []
            if refine_lexicographic and not (guaranteed or request.get('conservative')):
                from ._ordinary_refine import refine_incumbent as refine_linear
                try:
                    best, refinement_models = refine_linear(profiles, min(result, key=objective.sort_key),
                        budget, minimum, maximum, carry, duplicates, priorities, objective, region, request)
                    if best is not None:
                        result = [best] + [b for b in result if b is not best]
                except Exception:
                    log.exception('Linear refinement failed; retaining verified incumbent')
            if refinement_models:
                try:
                    result = available(result)
                except Exception:
                    return rescue.SearchResults([], [dict(state='UNKNOWN', reason='market_validation_error')], complete=False)
            result = result[:max(1, request['top_n'])]
            wrapped = rescue.SearchResults(result, refinement_models)
            wrapped.search_status['availability_verified'] = all(
                o.is_owned or o.auction_lot_id is not None for build in result for o in build.artifacts)
            return wrapped
    deadline = time.monotonic() + rescue.MAX_SECONDS
    for container, prepared in profiles:
        info = dict(container_id=container.item_id,
                    armor_id=(getattr(container, '_armor', None) or {}).get('id'),
                    state='UNKNOWN', reason='request_time_limit', prepared_candidates=len(prepared))
        if time.monotonic() >= deadline:
            models.append(info)
            continue
        if guaranteed or request.get('conservative'):
            models.append(dict(info, reason='unsupported_transform'))
            continue
        try:
            current = rescue.current_candidates(prepared, get_connection, region, _normalize_bonus_id)
        except Exception:
            models.append(dict(info, reason='market_validation_error'))
            continue
        checked = []

        def validate(ids):
            checked.clear()
            if (len(ids) != container.slots or len(set(map(int, ids))) != container.slots
                    or any(i < 0 or i >= len(current) for i in ids)):
                return False
            combo = [current[int(i)] for i in ids]
            # A lot can expire or be replaced while HiGHS is running.
            if len(rescue.current_candidates(combo, get_connection, region, _normalize_bonus_id)) != len(combo):
                return False
            _process_combo(combo, container, sum(o.buyout_price for o in combo), priorities,
                           minimum, checked, budget, max_thresholds=maximum, objective=objective,
                           require_carry=carry, require_guaranteed=guaranteed, allow_duplicates=duplicates)
            return bool(checked and _final_valid(checked[0], budget, minimum, maximum, carry, guaranteed, duplicates))

        bonuses = {key: val.get('value', 0) if isinstance(val, dict) else val
                   for key, val in container.bonus_stats.items()}
        ids, model = rescue.solve(current, container.slots, budget,
                                  1 - container.inner_protection / 100, bonuses,
                                  minimum or {}, maximum or {}, duplicates, carry, validate,
                                  seconds=deadline - time.monotonic(), guaranteed=guaranteed)
        models.append(dict(info, **model, omitted_candidates=len(prepared) - len(current)))
        if ids is not None and checked:
            build = _decorate_armor(checked[0])
            build._search_request = request
            wrapped = rescue.SearchResults([build], models, attempted=True, complete=complete)
            wrapped.search_status.update(elapsed_seconds=time.monotonic() - started, availability_verified=True)
            return wrapped
    wrapped = rescue.SearchResults([], models, attempted=True, complete=complete)
    wrapped.search_status['elapsed_seconds'] = time.monotonic() - started
    return wrapped


def apply_armor_to_builds(builds: list[ArtifactBuild], armor_id: str | None,
                          armor_level: int = 15, *, min_thresholds=None,
                          max_thresholds=None, carryable_only=None,
                          guaranteed_carryable=None, allow_duplicates=None):
    """Replace armor. Optimized builds rerun their full query; previews recompute.

    An unchanged armor is idempotent. Passing None removes the previous armor.
    Explicit constraints on standalone previews are checked on final stats.
    """
    if not builds:
        return builds
    request = getattr(builds[0], '_search_request', None)
    if request is not None:
        same = all((getattr(b, '_armor', None) or {}).get('id') == armor_id and
                   (armor_id is None or b._armor['level'] == armor_level) for b in builds)
        if same and all(value is None for value in (min_thresholds, max_thresholds,
                                                     carryable_only, guaranteed_carryable, allow_duplicates)):
            return builds
        params = dict(request, armor_id=armor_id, armor_ids=None, armor_level=armor_level)
        if min_thresholds is not None:
            params['min_thresholds'] = min_thresholds
        if max_thresholds is not None:
            params['max_thresholds'] = max_thresholds
        for key, value in (("carryable_only", carryable_only),
                           ("guaranteed_carryable", guaranteed_carryable), ("allow_duplicates", allow_duplicates)):
            if value is not None:
                params[key] = value
        if params.pop('max_quality', False):
            from ._max_quality import build_max_quality_sets
            return build_max_quality_sets(**params)
        return build_artifact_sets(**params)
    armor = load_armor(armor_id, armor_level) if armor_id is not None else None
    if armor_id is not None and armor is None:
        raise ValueError(f'Armor {armor_id} level {armor_level} not found')
    ranking, valid = {}, []
    for build in builds:
        base = getattr(build, '_unarmored_stats', None)
        if base is None:
            old_armor = getattr(build, '_armor', None)
            base = _without_armor(build.combined_stats, old_armor) if old_armor else dict(build.combined_stats)
            build._unarmored_stats = dict(base)
        raw = dict(base)
        # Undo derived stopping once for the new raw sum.
        if 'stopping_protection' in raw:
            raw['stopping_protection'] = 100 - (100 - raw['stopping_protection']) * (100 + raw.get('tear_dmg_factor', 0)) / 100
        for key, value in _armor_additions(armor).items():
            raw[key] = raw.get(key, 0) + value
        build.combined_stats = _recalculate_stopping_protection(raw)
        from ._derived_stats import add_names, augment
        augment(build.combined_stats)
        add_names(build.combined_stats, build.combined_stats_names)
        build._armor = armor
        build.carryable, build.carryability_warnings = check_carryability(build.combined_stats)
        build.carryability_analysis = analyze_build_carryability(build)
        build.guaranteed_carryable = build.carryability_analysis.get('guaranteed_carryable', True)
        if build.objective_config:
            config = build.objective_config
            objective = Objective.create(config['weights'], config['scales'], config['mode'],
                                         config['order'], config['tolerance'])
            build.total_score = objective.evaluate(build.combined_stats, build.total_price)
            build.priority_value = build.total_score
            ranking[id(build)] = objective.sort_key(build)
        if _final_valid(build, math.inf, min_thresholds, max_thresholds, carryable_only,
                        guaranteed_carryable, True if allow_duplicates is None else allow_duplicates):
            valid.append(build)
    if len(ranking) == len(builds):
        valid.sort(key=lambda b: ranking[id(b)])
    return valid


def _generate_build_tips(build: "ArtifactBuild") -> list[dict]:
    """Analyze build and generate improvement tips."""
    tips = []
    priorities = build.priorities or {}

    # 1. Research tips: artifacts with unlockable bonuses (ptn < unlock_ptn)
    for a in build.artifacts:
        if a.is_owned or not a.bonus_stats:
            continue
        unlockable = []
        for b in a.bonus_stats:
            unlock_ptn = b.get("unlock_ptn", 0)
            if unlock_ptn > a.ptn:
                prop_id = b.get("prop_id", "")
                name = b.get("name_ru", prop_id)
                min_val = b.get("min", 0)
                max_val = b.get("max", 0)
                is_pos = b.get("is_positive", True)
                # For negative stats (is_positive=False), show as reduction
                if not is_pos:
                    desc = f"{name}: {min_val:.1f}…{max_val:.1f}"
                else:
                    desc = f"{name}: +{min_val:.1f}…+{max_val:.1f}"
                unlockable.append({"name": name, "desc": desc, "unlock_ptn": unlock_ptn})
        if unlockable:
            tips.append({
                "type": "research",
                "icon": "🔬",
                "artifact": a.name_ru,
                "current_ptn": a.ptn,
                "bonuses": unlockable,
            })

    # 2. Counter-art warning: artifact with 0 priority stats but counter function
    prio_keys = list(priorities.keys())
    for a in build.artifacts:
        has_prio = any(a.stats.get(p, 0) > 0 for p in prio_keys)
        has_counter = any(a.stats.get(d, 0) < -0.5 for d in
                          ("psycho_accumulation", "biological_accumulation",
                           "thermal_accumulation", "radiation_accumulation"))
        if not has_prio and has_counter and not a.is_owned:
            tips.append({
                "type": "counter_art",
                "icon": "⚠️",
                "artifact": a.name_ru,
                "price": a.buyout_price,
            })

    # 3. Danger threshold warnings: stat within 80% of limit
    for prop_id, val in build.combined_stats.items():
        fkey = _to_full_key(prop_id)
        if fkey in DANGER_THRESHOLDS:
            threshold = DANGER_THRESHOLDS[fkey]
            if 0 < val <= threshold and val > threshold * 0.8:
                name = build.combined_stats_names.get(prop_id, prop_id)
                tips.append({
                    "type": "threshold",
                    "icon": "⚡",
                    "stat": name,
                    "value": round(val, 2),
                    "threshold": threshold,
                })

    # 4. Frost note: not protectable by container
    frost = build.combined_stats.get("frost_accumulation", 0)
    if frost > 0.01:
        tips.append({
            "type": "frost",
            "icon": "❄️",
            "value": round(frost, 2),
            "threshold": 1.0,
        })

    # 5. Price efficiency: owned vs auction
    owned_count = sum(1 for a in build.artifacts if a.is_owned)
    auction_price = sum(a.buyout_price for a in build.artifacts if not a.is_owned)
    if owned_count > 0:
        tips.append({
            "type": "owned",
            "icon": "📦",
            "count": owned_count,
            "auction_price": auction_price,
        })

    return tips


# ─────────────────────────── Loading ────────────────────────────────────────

def load_artifact_options(priorities: dict | None = None,
                          max_price: int | None = None,
                          region: str | None = None,
                          conservative: bool = False,
                          *, stock_only: bool = False, _quality_audit: dict | None = None) -> list[ArtifactOption]:
    """Load artifact options from active auction lots — single JOIN query, no N+1.

    Per REBOOT_PROMPT v2.0:
    - art_percent = MAX of quality tier (115 for qlt=1, 130 for qlt=2, ...).
    - ptn>0 with bonus_properties → use the lot's actual bonuses.
    - ptn=0 → base stats only, no bonus simulation (don't fake research).
    - Smart bonus-combo filter: only combos containing priority stats, max 5 per lot.
    - Score: _score_option only — no counter-art bonus.
    """
    if _quality_audit is not None:
        from ._quality_candidates import load
        from . import builder as current_builder
        return load(current_builder, region or Config.REGION,
                    max_price, stock_only, _quality_audit)
    priorities = priorities or {}
    region = region or Config.REGION
    options: list[ArtifactOption] = []

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT a.id, a.name_ru, a.name_en, a.subcategory, a.stats, a.bonus_stats, a.icon_url,
                   l.quality, l.ptn, l.buyout_price as min_price,
                   l.bonus_properties, l.art_percent, l.stats_random, l.id AS auction_lot_id
            FROM artefacts a
            LEFT JOIN (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY item_id, quality, ptn, bonus_properties
                    ORDER BY buyout_price ASC
                ) as rn
                FROM auction_lots WHERE region=? AND buyout_price > 0
            ) l ON l.item_id = a.id AND l.rn <= 7
            ORDER BY a.id, l.buyout_price ASC
        """, (region,)).fetchall()

        # Load historical avg prices per (item_id, quality) — at least 3 sales for reliability
        hist_rows = conn.execute("""
            SELECT item_id, quality, AVG(price) as avg_price, COUNT(*) as cnt
            FROM auction_history
            WHERE region=? AND price > 0
            GROUP BY item_id, quality
            HAVING cnt >= 3
        """, (region,)).fetchall()
        _hist_avg: dict[tuple[str, int], float] = {}
        _hist_by_item: dict[str, dict[int, float]] = {}
        for hr in hist_rows:
            _hist_avg[(hr["item_id"], hr["quality"])] = hr["avg_price"]
            _hist_by_item.setdefault(hr["item_id"], {})[hr["quality"]] = hr["avg_price"]
        # Fallback: for qualities without history, use nearest lower quality
        for item_id, qmap in _hist_by_item.items():
            for q in range(6):
                if (item_id, q) not in _hist_avg:
                    # Find nearest lower quality with data
                    for q_lo in range(q - 1, -1, -1):
                        if q_lo in qmap:
                            # Extrapolate: higher quality ≈ 2× lower quality
                            _hist_avg[(item_id, q)] = qmap[q_lo] * (2.0 ** (q - q_lo))
                            break
    # Priority prop_ids (upper) for bonus combo filtering
    prio_props = {_normalize_bonus_id(p.upper()) for p in priorities.keys()}

    for art_id, group in _it.groupby(rows, key=lambda r: r["id"]):
        group_list = list(group)
        art = group_list[0]
        if not art["stats"]:
            continue
        base_stats = json.loads(art["stats"])
        bonus_stats = json.loads(art["bonus_stats"]) if art["bonus_stats"] else []
        # Exclude non-artifact items (Рубик, Регалий)
        if art["id"] in EXCLUDED_ARTIFACTS:
            continue

        for lot_index, lot in enumerate(group_list):
            if lot["min_price"] is None:
                continue
            if stock_only and lot["ptn"] != 0:
                continue  # Check the original SQL value before any defaults/views.
            price = lot["min_price"]
            if max_price and price > max_price:
                continue
            quality = lot["quality"] or 0
            ptn = lot["ptn"] or 0
            _hist = _hist_avg.get((art["id"], quality), 0)
            # Price fairness: only filter UNRESEARCHED lots (ptn=0).
            # Researched lots (ptn>0) are rare, have bonuses, and are legitimately
            # more expensive — don't filter them against ptn=0 averages.
            if ptn == 0:
                if _hist > 0 and price > _hist * 2.0:
                    log.debug("Skip overpriced: %s q=%d ptn=0 price=%d avg=%.0f",
                             art["name_ru"], quality, price, _hist)
                    continue
            qr = QUALITY_RANGES.get(quality, QUALITY_RANGES[0])
            art_percent_known = bool(lot["art_percent"])
            if art_percent_known:
                art_percent = lot["art_percent"]  # base from calc_art_percent(stats_random); ptn applied in _calc_stat_value via calc_ptn_bonus
            else:
                art_percent = qr["min"] + qr["range"]

            # Determine which bonus combos to generate
            if ptn > 0 and lot["bonus_properties"]:
                lot_bonus_raw = lot["bonus_properties"] if lot["bonus_properties"] else "[]"
                try:
                    lot_bonuses = json.loads(lot_bonus_raw) if isinstance(lot_bonus_raw, str) else lot_bonus_raw
                except (json.JSONDecodeError, TypeError):
                    lot_bonuses = []
                bonus_combos = [lot_bonuses] if lot_bonuses else [[]]
            elif ptn > 0 and bonus_stats:
                available_bonuses = [s for s in bonus_stats if s.get("unlock_ptn", 0) <= ptn]
                all_bonus_props = [s.get("prop_id", "") for s in available_bonuses]
                bonus_combos = []
                for r in range(1, min(len(all_bonus_props), 3) + 1):
                    for c in _it.combinations(all_bonus_props, r):
                        combo_set = set(c)
                        has_prio = bool(combo_set & prio_props)
                        has_counter = any(
                            p.endswith("_accumulation") or p.endswith("_protection")
                            for p in combo_set
                        )
                        if has_prio or has_counter or not prio_props:
                            bonus_combos.append(list(c))
                if not bonus_combos:
                    bonus_combos = [[]]
                bonus_combos = bonus_combos[:10]
            else:
                # ptn=0: no bonuses (unresearched). Base stats only.
                bonus_combos = [[]]

            for combo in bonus_combos:
                norm_ids = [_normalize_bonus_id(b.upper()) for b in combo]
                stats_container, stats_raw, stats_names, _ = compute_artifact_stats(
                    base_stats, bonus_stats, norm_ids,
                    art_percent, ptn, qr["min"], qr["range"],
                    lot["stats_random"] if lot["stats_random"] is not None else None,
                )
                opt = ArtifactOption(
                    item_id=art["id"], name_ru=art["name_ru"], name_en=art["name_en"],
                    subcategory=art["subcategory"], quality=quality,
                    quality_name=QUALITY_NAMES.get(quality, "?"),
                    art_percent=art_percent, buyout_price=price, ptn=ptn,
                    stats=stats_container, stats_raw=stats_raw, stats_names=stats_names,
                    unlocked_bonus_ids=combo, icon_url=art["icon_url"],
                    base_stats=base_stats, bonus_stats=bonus_stats,
                    hist_avg_price=_hist,
                    stats_random=lot["stats_random"],
                    art_percent_known=art_percent_known,
                    auction_lot_id=lot["auction_lot_id"] if "auction_lot_id" in lot.keys() else None,
                    physical_id=(f"auction:{region}:{lot['auction_lot_id']}"
                                 if "auction_lot_id" in lot.keys() else f"anonymous-row:{art_id}:{lot_index}"),
                )
                opt.score = _score_option(stats_container, priorities)
                options.append(opt)

    if conservative:
        from ._quality_scenario import minimum_views
        options = minimum_views(options, priorities)
    options.sort(key=lambda o: o.score, reverse=True)
    log.info("Loaded %d artifact options from %d base artefacts", len(options), len(rows))
    return options


def load_container_options(region: str | None = None) -> list[ContainerOption]:
    from ._derived_stats import augment_bonus_stats
    region = region or Config.REGION
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name_ru, name_en, item_type, slots, base_price, "
            "inner_protection, effectiveness, bonus_stats, icon_url "
            "FROM containers ORDER BY slots DESC"
        ).fetchall()
        options = []
        for c in rows:
            lot = conn.execute(
                "SELECT MIN(buyout_price) as min_price FROM auction_lots "
                "WHERE item_id=? AND region=? AND buyout_price > 0",
                (c["id"], region),
            ).fetchone()
            price = lot["min_price"] if lot and lot["min_price"] else 0
            try:
                bonus = json.loads(c["bonus_stats"]) if c["bonus_stats"] else {}
            except (json.JSONDecodeError, TypeError):
                bonus = {}
            augment_bonus_stats(bonus)
            options.append(ContainerOption(
                item_id=c["id"], name_ru=c["name_ru"], name_en=c["name_en"] if "name_en" in c.keys() else "",
                item_type=c["item_type"], slots=c["slots"], buyout_price=price,
                inner_protection=c["inner_protection"] if "inner_protection" in c.keys() else 0.0,
                effectiveness=c["effectiveness"] if "effectiveness" in c.keys() else 0.0,
                bonus_stats=bonus, icon_url=c["icon_url"],
            ))
    return options


# ─────────────────────────── Build search ────────────────────────────────────

def _stock_options(options):
    """Restrict sources, never turn an upgraded physical instance into +0."""
    from ._quality_scenario import source
    return [o for o in options if source(o).ptn == 0]


def build_artifact_sets(
    budget: int,
    tolerance_pct: float = 8.0,
    priorities: dict | None = None,
    min_thresholds: dict | None = None,
    max_thresholds: dict | None = None,
    top_n: int = 10,
    region: str | None = None,
    container_id: str | None = None,
    owned_artifacts: list[dict] | None = None,
    carryable_only: bool = True,
    allow_duplicates: bool = False,
    min_budget: bool = False,
    conservative: bool = False,
    guaranteed_carryable: bool = False,
    exclude_item_ids: list[str] | None = None,
    template_filter: list[dict] | None = None,  # [{item_id, bonus_ids}] — restrict pool to these
    optimization_mode: str = "balanced",
    priority_order: list[str] | None = None,
    objective_scales: dict | None = None,
    objective_tolerance: float = 1e-6,
    container_ids: list[str] | None = None,
    armor_id: str | None = None,
    armor_ids: list[str | None] | None = None,
    armor_level: int = 15,
    min_budget_timeout: float = 30.0,
    *, stock_only: bool = False, _beam_search_fn=None, _quality_context=None,
) -> list[ArtifactBuild]:
    """Find top artifact builds within the artifact purchase budget (plus tolerance).

    Armor and container prices are excluded. Fixed or explicit alternative armor
    profiles enter final stats, constraints and the objective during search.

    carryable_only=True: return only carryable builds.
    allow_duplicates=False (default): unique item_id per build. True: allow same item_id.
    min_budget=True: minimize price subject to constraints; budget=0 removes the cap.
    Scales are fixed once across all selected containers. Lexicographic mode
    requires an explicit priority_order and uses fixed tolerance bins.
    """
    from ._survivability import validate as validate_survivability
    validate_survivability(priorities, min_thresholds, max_thresholds, max_quality=_quality_context is not None, mode=optimization_mode)
    _request_started = time.monotonic()
    if not isinstance(min_budget_timeout, (int, float)) or not math.isfinite(min_budget_timeout) or not 0 <= min_budget_timeout <= 30:
        raise ValueError('min_budget_timeout must be between 0 and 30 seconds')
    _min_deadline = _request_started + min_budget_timeout
    _min_models = []
    if not priorities and (min_thresholds or min_budget):
        # Auto-derive priorities from ALL positive min_thresholds.
        _mt_keys = [k for k, v in (min_thresholds or {}).items() if v > 0]
        if _mt_keys:
            priorities = {k: 10 for k in _mt_keys}
        elif min_thresholds:
            priorities = {list(min_thresholds)[0]: 10}
    priorities = dict(sorted((priorities or {}).items(), key=lambda x: -x[1]))
    region = region or Config.REGION
    # Price minimization is now explicit; zero balanced budget means free builds.
    min_budget = min_budget or optimization_mode == "min_budget"
    optimization_mode = "min_budget" if min_budget else optimization_mode
    Objective.create(priorities, objective_scales, optimization_mode, priority_order, objective_tolerance)
    # Ignored weights must not change candidate preparation or container choice.
    priorities = {key: weight for key, weight in priorities.items() if weight > 0}
    for limits in (min_thresholds, max_thresholds):
        if limits is not None and (not isinstance(limits, dict) or any(
                not isinstance(key, str) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for key, value in limits.items())):
            raise ValueError("Constraints must map stat keys to finite numbers")
    if armor_id is not None and armor_ids is not None:
        raise ValueError("Choose armor_id or armor_ids")
    selected_armors = armor_ids if armor_ids is not None else [armor_id]
    if not isinstance(selected_armors, list) or not selected_armors:
        raise ValueError("armor_ids must be a nonempty list")
    armors = []
    if any(selected is not None and (not isinstance(selected, str) or not selected) for selected in selected_armors):
        raise ValueError("Invalid armor ID")
    for selected in dict.fromkeys(selected_armors):
        if selected is None:
            armors.append(None)
        elif not isinstance(selected, str) or not selected:
            raise ValueError("Invalid armor ID")
        else:
            armor = load_armor(selected, armor_level)
            if armor is None:
                raise ValueError(f"Armor {selected} level {armor_level} not found")
            armors.append(armor)
    if min_budget and budget == 0:
        # No budget limit — find cheapest meeting thresholds.
        # Use max_lot_price * max_slots instead of 10B to avoid
        # loading extremely expensive lots that skew stat_max
        # and cause the search to miss cheap builds.
        _all_opts = load_artifact_options(priorities=priorities, region=region, conservative=conservative,
                                         **({'stock_only': True} if stock_only else {}))
        if stock_only:
            _all_opts = _stock_options(_all_opts)
        _max_lot = max((o.buyout_price for o in _all_opts), default=100_000_000)
        effective_budget = int(_max_lot * 7) + 1  # 7 = max container slots
    else:
        effective_budget = int(budget * (1 + tolerance_pct / 100.0))

    search_request = deepcopy(dict(budget=budget, tolerance_pct=tolerance_pct, priorities=dict(priorities),
                          min_thresholds=min_thresholds, max_thresholds=max_thresholds, top_n=top_n,
                          region=region, container_id=container_id, owned_artifacts=owned_artifacts,
                          carryable_only=carryable_only, allow_duplicates=allow_duplicates,
                          min_budget=min_budget, conservative=conservative,
                          guaranteed_carryable=guaranteed_carryable, exclude_item_ids=exclude_item_ids,
                          template_filter=template_filter, optimization_mode=optimization_mode,
                          priority_order=priority_order, objective_scales=objective_scales,
                          objective_tolerance=objective_tolerance, container_ids=container_ids,
                          armor_id=armor_id, armor_ids=armor_ids, armor_level=armor_level))
    search_request['min_budget_timeout'] = min_budget_timeout
    if stock_only:
        search_request['stock_only'] = True

    # Containers
    all_containers = load_container_options(region)
    if container_ids is not None:
        containers = [c for c in all_containers if c.item_id in set(container_ids)]
    elif container_id:
        containers = [c for c in all_containers if c.item_id == container_id]
    else:
        if min_budget or _quality_context is not None:
            # Slot count does not dominate effectiveness or container bonuses.
            # Visit every allowed profile; deadline skips remain explicitly unknown.
            containers = sorted(all_containers, key=lambda c: (
                not any(key in c.bonus_stats for key in priorities),
                -c.slots, -c.inner_protection, -c.effectiveness, c.item_id))
        else:
            def _container_sort_key(c):
                has_prio_bonus = False
                if c.bonus_stats:
                    for pid in priorities:
                        if pid in c.bonus_stats:
                            has_prio_bonus = True
                            break
                return (has_prio_bonus, c.inner_protection, c.effectiveness, c.slots)
            containers = sorted(all_containers, key=_container_sort_key, reverse=True)
            containers = containers[:5]  # cap — avoid searching all 54 containers


    containers = _armor_containers(containers, armors)
    if _quality_context is not None and len(containers) > _quality_context['max_profiles']:
        _quality_context['error'] = 'profile_limit'
        return []
    if guaranteed_carryable:
        effective_containers = []
        for current in containers:
            effective = replace(current)
            effective._base_container = getattr(current, "_base_container", current)
            effective._armor = getattr(current, "_armor", None)
            effective._guaranteed_search = True
            effective_containers.append(effective)
        containers = effective_containers

    # Max Quality opts into full physical loading; ordinary providers are unchanged.
    _quality_load = {}
    if _quality_context is not None:
        _quality_context['candidate_audit'] = dict(deadline=_quality_context['deadline'],
            max_input_candidates=_quality_context['max_input_candidates'],
            source='injected_candidate_provider', coverage='provider_scope_unverified')
        _quality_load['_quality_audit'] = _quality_context['candidate_audit']

    # Auction options
    options = load_artifact_options(priorities=priorities, max_price=effective_budget, region=region,
                                    conservative=conservative and _quality_context is None,
                                    **({'stock_only': True} if stock_only else {}), **_quality_load)
    if _quality_context is not None and _quality_context['candidate_audit'].get('error'):
        _quality_context['error'] = _quality_context['candidate_audit']['error']
        return []
    if stock_only:
        options = _stock_options(options)
    if _quality_context is not None and conservative:
        from ._quality_scenario import minimum_views
        options = minimum_views(options, priorities)
    if _quality_context is not None and len(options) > _quality_context['max_input_candidates']:
        _quality_context['error'] = 'input_candidate_limit'
        return []

    # Template mode: restrict candidate pool to only template-matching artifacts
    if template_filter:
        # Build set of (item_id, sorted(bonus_ids)) for matching
        _tpl_keys = set()
        for tf in template_filter:
            key = (tf["item_id"], tuple(sorted(tf.get("bonus_ids", []))))
            _tpl_keys.add(key)
        _before = len(options)
        options = [
            o for o in options
            if (o.item_id, tuple(sorted(o.unlocked_bonus_ids))) in _tpl_keys
        ]
        if _quality_context is not None:
            _quality_context['candidate_audit']['user_template_excluded_profiles'] = _before - len(options)
        log.info("Template filter: %d → %d candidates (kept only template-matching)", _before, len(options))


    # Fixed request-wide scales are computed after owned options are loaded.
    # Load hidden effects (изнанка) for warning generation
    _hidden_effects: dict[str, dict] = {}
    with get_connection() as _he_conn:
        _table_exists = _he_conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hidden_effects'"
        ).fetchone()
        if _table_exists:
            for _r in _he_conn.execute(
                "SELECT item_id, effect, severity FROM hidden_effects"
            ).fetchall():
                _hidden_effects[_r["item_id"]] = {
                    "effect": _r["effect"],
                    "severity": _r["severity"],
                }

    # Owned artifacts
    owned_opts: list[ArtifactOption] = []
    if _quality_context is not None:
        eligible_owned = sum(oa.get('ptn', 0) == 0 for oa in (owned_artifacts or [])) if stock_only else len(owned_artifacts or [])
        if len(options) + eligible_owned > _quality_context['max_input_candidates']:
            _quality_context['error'] = 'input_candidate_limit'
            _quality_context['candidate_audit'].update(coverage='incomplete_owned_preparation', owned_input_limit=True)
            return []
    if owned_artifacts:
        with get_connection() as conn:
            for owned_index, oa in enumerate(owned_artifacts):
                if stock_only and oa.get('ptn', 0) != 0:
                    continue  # Owned +0 is allowed; keep original positional IDs.
                art = conn.execute(
                    "SELECT id, name_ru, name_en, subcategory, stats, bonus_stats, icon_url "
                    "FROM artefacts WHERE id=?",
                    (oa["item_id"],),
                ).fetchone()
                if not art:
                    continue
                base_stats = json.loads(art["stats"]) if art["stats"] else []
                bonus_stats = json.loads(art["bonus_stats"]) if art["bonus_stats"] else []
                quality = oa.get("quality", 0)
                ptn = oa.get("ptn", 0)
                _known_default = ('art_percent' in oa
                                  and isinstance(oa.get('art_percent'), (int, float))
                                  and oa.get('art_percent') not in (None, 0))
                art_percent_known = bool(oa.get('_art_percent_known', _known_default))
                art_percent = oa.get("art_percent", QUALITY_RANGES.get(quality, QUALITY_RANGES[0])["min"])
                bonus_ids = oa.get("bonus_ids", [])
                if _quality_context is not None and (
                        type(quality) is not int or not 0 <= quality <= 5
                        or type(ptn) is not int or not 0 <= ptn <= 15
                        or not isinstance(art_percent, (int, float)) or not math.isfinite(art_percent)
                        or not 0 < art_percent <= 190 or not isinstance(bonus_ids, list)
                        or any(not isinstance(v, str) for v in bonus_ids)):
                    _quality_context['error'] = 'unsupported_owned_descriptor'
                    return []
                if _quality_context is not None and _quality_context['candidate_audit'].get('source') == 'full_physical_auction':
                    from ._quality_candidates import catalogue_supported
                    if not catalogue_supported(base_stats, bonus_stats, bonus_ids, _normalize_bonus_id):
                        _quality_context['error'] = 'unsupported_owned_catalogue'
                        return []
                qr = QUALITY_RANGES.get(quality, QUALITY_RANGES[0])
                stats_c, stats_r, sn, _ = compute_artifact_stats(
                    base_stats, bonus_stats, bonus_ids,
                    art_percent, ptn, qr["min"], qr["range"], None,
                )
                instance_id, physical_id = owned_identity(oa, owned_index)
                opt = ArtifactOption(
                    item_id=art["id"], name_ru=art["name_ru"], name_en=art["name_en"],
                    subcategory=art["subcategory"], quality=quality,
                    quality_name=QUALITY_NAMES.get(quality, "?"),
                    art_percent=art_percent, buyout_price=0, ptn=ptn,
                    stats=stats_c, stats_raw=stats_r, stats_names=sn,
                    unlocked_bonus_ids=bonus_ids, icon_url=art["icon_url"],
                    base_stats=base_stats, bonus_stats=bonus_stats,
                    stats_random=None,
                    art_percent_known=art_percent_known,
                    is_owned=True,
                    owned_instance_id=instance_id,
                    physical_id=physical_id,
                )
                opt.score = _score_option(stats_c, priorities)  # raw score, normalized later
                owned_opts.append(opt)

    expected_owned = sum(oa.get('ptn', 0) == 0 for oa in (owned_artifacts or [])) if stock_only else len(owned_artifacts or [])
    if _quality_context is not None and len(owned_opts) != expected_owned:
        _quality_context['error'] = 'unknown_owned_artifact'
        return []
    if _quality_context is not None:
        _quality_context['candidate_audit']['owned_candidates'] = len(owned_opts)
    if _quality_context is not None and conservative:
        owned_opts = minimum_views(owned_opts, priorities)

    # One immutable normalization for the whole request, before any beam trim.
    stat_max = _fixed_objective_scales(
        [o for o in options if o.buyout_price <= effective_budget] + owned_opts,
        containers, priorities,
    )
    if objective_scales is not None:
        stat_max.update(objective_scales)
    objective = Objective.create(priorities, stat_max, optimization_mode,
                                 priority_order, objective_tolerance)
    if _quality_context is not None:
        _quality_context.update(objective=objective, budget=effective_budget, request=search_request,
                                profiles=[], hidden_effects=_hidden_effects)
    for opt in options + owned_opts:
        # Max Quality keeps the physical descriptors immutable; its prepared
        # views are scored later and never written back to the originals.
        if _quality_context is None:
            opt.score = objective.evaluate(opt.stats, opt.buyout_price)
    options.sort(key=lambda opt: -opt.score)
    if _quality_context is not None:
        # Stop before ALL Beam group/counter/ranking caps. Each full container
        # profile is recalculated independently; originals remain immutable.
        from ._quality_scenario import source
        excluded = set(exclude_item_ids or []) | set(EXCLUDED_ARTIFACTS)
        full = [o for o in options if o.buyout_price <= effective_budget
                and not o.is_owned and o.item_id not in excluded] + owned_opts
        _quality_context['candidate_audit']['request_excluded_profiles'] = len(options) + len(owned_opts) - len(full)
        for container in containers:
            profile = []
            for o in full:
                if time.monotonic() >= _quality_context['deadline']:
                    _quality_context['error'] = 'preparation_time_limit'
                    return []
                qr = QUALITY_RANGES.get(o.quality, QUALITY_RANGES[0])
                values, raw, names, recommended = compute_artifact_stats(
                    o.base_stats, o.bonus_stats, o.unlocked_bonus_ids,
                    o.art_percent, o.ptn, qr['min'], qr['range'], o.stats_random,
                    container.inner_protection, container.effectiveness)
                view = replace(o, stats=values, stats_raw=raw, stats_names=names,
                               recommended_percent=recommended)
                if conservative and not getattr(o, 'art_percent_known', False):
                    view._quality_source = source(o)
                profile.append(view)
            _quality_context['profiles'].append((container, sorted(profile, key=lambda o: (o.item_id, o.buyout_price))))
        return []
    max_keep = max(top_n * 50, 2000)  # large — captures diverse builds across containers
    all_builds: list[ArtifactBuild] = []
    total_combos = 0
    _rescue_profiles = []
    _rescue_complete = True
    _rescue_candidates = 0

    for container in containers:
        if _quality_context is not None and time.monotonic() >= _quality_context['deadline']:
            _quality_context['error'] = 'preparation_time_limit'
            break
        if min_budget and _beam_search_fn is None and time.monotonic() >= _min_deadline:
            _min_models.append(dict(container_id=container.item_id, state='unknown', reason='request_time_limit'))
            continue
        n_slots = container.slots
        if n_slots < 1 or len(options) + len(owned_opts) < n_slots:
            _rescue_complete = False
            continue

        # Owned artifacts are now CANDIDATES (price=0), not forced offsets.
        # The search freely chooses owned OR auction for each slot.
        remaining_slots = n_slots

        # Candidates: within budget, not owned, not Рубик, not excluded — then dedup by item_id
        _exclude = set(exclude_item_ids or [])
        _exclude.update(EXCLUDED_ARTIFACTS)  # Рубик + Регалий — not build artifacts
        pool = [
            o for o in options
            if o.buyout_price <= effective_budget and not o.is_owned and o.item_id not in _exclude
        ]

        # Search engine: beam search (O(n×B×k)) for n>100, numba (O(n^k)) for n<=100.
        # With beam search, max_cand can be large — beam handles 800+ candidates in seconds.
        _k = remaining_slots
        max_cand = 99999  # use all lots
        # Deduplicate by item_id — keep ALL variants (no top-3 limit).
        # Each unique (item_id, ptn, quality, bonus combo) is a separate candidate.
        # This ensures cheap variants with slightly lower stats survive alongside
        # expensive ones — the search picks the best price/performance ratio.
        # Threshold searches have candidate-count based widening/multi-pass
        # policies. Keep their pool unchanged until those policies are budgeted.
        _profile_pool = not (min_budget or min_thresholds or max_thresholds)
        seen_ids: dict[str, list[ArtifactOption]] = {}
        for o in pool:
            if o.item_id not in seen_ids:
                seen_ids[o.item_id] = []
            seen_ids[o.item_id].append(o)
        pool = []
        for item_id, variants in seen_ids.items():
            # Budget/threshold searches retain their grouping. Other searches cap
            # only identical full profiles, so useful bonuses/percentages survive
            # while equivalent offers do not crowd out other Beam prefixes.
            _seen_qptn: dict[tuple, list] = {}
            for v in sorted(variants, key=lambda o: -o.score):
                # Distinct percentages/bonuses are distinct search choices.
                # Only exactly equivalent calculation profiles share a cap.
                _qkey = ((v.quality, v.ptn) if not _profile_pool else (
                    v.quality, v.ptn, float(v.art_percent).hex(), v.stats_random,
                    tuple(sorted(_normalize_bonus_id(x.upper()) for x in v.unlocked_bonus_ids)),
                    tuple(sorted((key, float(value).hex()) for key, value in v.stats.items())),
                    tuple(sorted((key, float(value).hex()) for key, value in v.stats_raw.items())),
                ))
                _seen_qptn.setdefault(_qkey, []).append(v)
            for _qkey, qvars in _seen_qptn.items():
                _cap = 5 if min_budget else (max(n_slots, 2) if allow_duplicates else (1 if _profile_pool else 2))
                # Count physical offers, not their mutually exclusive hypotheses.
                selected_lots = set()
                for v in sorted(qvars, key=lambda o: o.buyout_price):
                    key = physical_key(v)
                    if key not in selected_lots and len(selected_lots) >= _cap:
                        continue
                    selected_lots.add(key)
                    pool.append(v)
        # Split: regular vs counter-arts.
        # Dual-purpose arts (has priority stats AND danger reduction) go to REGULAR.
        # Only pure counter-arts (no priority stats) go to counter pool.
        _prio_keys = list(priorities.keys())
        if "health_bonus" in priorities:
            _prio_keys.extend(["heal_efficiency", "regeneration_bonus", "artefakt_heal"])
        regular = [o for o in pool if
                   any(o.stats.get(p, 0) > 0 for p in _prio_keys) or not
                   any(o.stats.get(d, 0) < -0.5 for d in
                       ("psycho_accumulation", "biological_accumulation",
                        "thermal_accumulation", "radiation_accumulation"))]
        counter_pool_all = [o for o in pool if
                            not any(o.stats.get(p, 0) > 0 for p in _prio_keys) and
                            any(o.stats.get(d, 0) < -0.5 for d in
                                ("psycho_accumulation", "biological_accumulation",
                                 "thermal_accumulation", "radiation_accumulation"))]
        # Regular: for k≤4, use ALL candidates with priority stats > 0.
        # Search ALL lots — beam search (O(n×B×k)) handles large pools efficiently.
        # No per-stat top-N filtering — every lot gets a chance in the beam.
        _reg_cap = max_cand - 30
        candidates = list(regular)[:_reg_cap]
        # Counter-arts: dedup per item_id, include best + cheapest researched + cheapest.
        # Reserve ~30 slots — counter-arts are essential for carryability even with score=0.
        _ca_seen = {}
        for o in counter_pool_all:
            if o.item_id not in _ca_seen: _ca_seen[o.item_id] = []
            _ca_seen[o.item_id].append(o)
        counter_arts = []
        extra_counter_copies = []
        for item_id, variants in _ca_seen.items():
            by_score = sorted(variants, key=lambda o: -o.score)
            counter_arts.append(by_score[0])
            researched = [v for v in variants if v.ptn > 0]
            if researched:
                cheapest_r = min(researched, key=lambda v: v.buyout_price)
                if cheapest_r is not counter_arts[-1]:
                    counter_arts.append(cheapest_r)
            cheapest = min(variants, key=lambda v: v.buyout_price)
            if cheapest not in counter_arts:
                counter_arts.append(cheapest)
            # Equal-price copies are independently purchasable. Do not collapse
            # them to the same best/cheapest representative of an item type.
            copies = {physical_key(v) for v in counter_arts if v.item_id == item_id}
            for v in sorted(variants, key=lambda o: o.buyout_price):
                if len(copies) >= n_slots:
                    break
                if physical_key(v) not in copies:
                    extra_counter_copies.append(v)
                    copies.add(physical_key(v))
        # Sort counter-arts by danger reduction (not score — score is 0 for pure counters)
        counter_arts = sorted(counter_arts, key=lambda o: -max(
            abs(min(0, o.stats.get(d, 0))) for d in
            ("psycho_accumulation", "biological_accumulation",
             "thermal_accumulation", "radiation_accumulation")))
        # The reserve counts physical offers, not alternative hypotheses.
        selected_counter_lots = set()
        retained_counters = []
        # Preserve the existing variety before spending free reserve on copies.
        for v in counter_arts + extra_counter_copies:
            key = physical_key(v)
            if key not in selected_counter_lots and len(selected_counter_lots) >= 30:
                continue
            selected_counter_lots.add(key)
            retained_counters.append(v)
        counter_arts = retained_counters
        candidates.extend(counter_arts)

        # Owned artifacts enter as FREE candidates (price=0). The search decides
        # whether to use them (free slot) or a better auction lot (paid slot).
        # With allow_duplicates=False, the search won't pick both owned and auction
        # of the same item_id — which is correct (can't use same artifact twice).
        if owned_opts:
            candidates.extend(owned_opts)
        # to recompute with container effectiveness so non-accumulation stats scale.
        # We create a temporary copy of candidates with adjusted stats.
        _numba_candidates = []
        for o in candidates:
            qr = QUALITY_RANGES.get(o.quality, QUALITY_RANGES[0])
            stats_c, _, _, _ = compute_artifact_stats(
                o.base_stats, o.bonus_stats, o.unlocked_bonus_ids,
                o.art_percent, o.ptn, qr["min"], qr["range"], o.stats_random,
                container.inner_protection, container.effectiveness,
            )
            _o2 = ArtifactOption(
                item_id=o.item_id, name_ru=o.name_ru, name_en=o.name_en,
                subcategory=o.subcategory, quality=o.quality,
                quality_name=o.quality_name, art_percent=o.art_percent,
                buyout_price=o.buyout_price, ptn=o.ptn,
                stats=stats_c, stats_raw=stats_c, stats_names=o.stats_names,
                unlocked_bonus_ids=o.unlocked_bonus_ids, icon_url=o.icon_url,
                base_stats=o.base_stats, bonus_stats=o.bonus_stats,
                stats_random=o.stats_random, is_owned=o.is_owned, score=o.score,
                hist_avg_price=o.hist_avg_price,
                art_percent_known=o.art_percent_known,
                auction_lot_id=o.auction_lot_id, owned_instance_id=o.owned_instance_id,
                physical_id=o.physical_id,
            )
            _numba_candidates.append(_o2)
            if hasattr(o, '_quality_source'):
                _o2._quality_source = o._quality_source
            if hasattr(container, "_armor"):
                _o2._danger_bounds = _artifact_danger_bounds(_o2, container)
            if guaranteed_carryable:
                for key, value in _o2._danger_bounds[1].items():
                    _o2.stats["__guaranteed_" + key] = value
        _numba_search_candidates = sorted(_numba_candidates, key=lambda o: (o.item_id, o.buyout_price))
        if _quality_context is not None:
            _quality_context['profiles'].append((container, _numba_search_candidates))
            continue
        if not min_budget:
            from . import _feasibility_fallback as _rescue
            if (len(_rescue_profiles) < _rescue.MAX_PROFILES
                    and len(_numba_search_candidates) <= _rescue.MAX_CANDIDATES
                    and _rescue_candidates + len(_numba_search_candidates) <= _rescue.MAX_TOTAL_CANDIDATES):
                _rescue_profiles.append((container, _numba_search_candidates))
                _rescue_candidates += len(_numba_search_candidates)
            else:
                _rescue_complete = False

        # Numba JIT accelerated search — searches ALL combinations at native speed.
        # ~1.5 billion combos/sec vs ~40k/sec in pure Python (37000× faster).
        prot_mult = 1.0 - container.inner_protection / 100.0
        n_keep = 50 if min_budget else 2000  # min_budget: only need cheapest few
        cont_bonus_dict = {}
        if container.bonus_stats:
            for pid, data in container.bonus_stats.items():
                val = data.get("value", 0) if isinstance(data, dict) else data
                if val:
                    cont_bonus_dict[pid] = val
        engine_maximum = dict(max_thresholds or {})
        if guaranteed_carryable:
            for full_key, threshold in DANGER_THRESHOLDS.items():
                key = full_key.replace("stalker.artefact_properties.factor.", "")
                target = threshold - cont_bonus_dict.get(key, 0)
                if key != "frost_accumulation" and target >= 0:
                    if prot_mult == 0:
                        continue
                    target /= prot_mult
                engine_maximum["__guaranteed_" + key] = target
        # Choose search engine: beam search for large pools (n>100), numba for small k>=4
        _n_cand = len(_numba_search_candidates)
        _use_beam = _n_cand > 100 or _beam_search_fn is not None or min_budget
        if _use_beam:
            from ._beam_search import search as _search_fn
            if _beam_search_fn is not None:
                _search_fn = _beam_search_fn
            _sargs = (_numba_search_candidates, remaining_slots, effective_budget, remaining_slots,
                      prot_mult, cont_bonus_dict, stat_max, priorities, min_thresholds or {}, n_keep)
            # constraint_lookahead: use suffix extrema to prune beam paths that
            # can never satisfy min/max thresholds with the remaining slots.
            # Enabled when the user set explicit stat constraints and we're doing
            # a balanced search (not min_budget / lexicographic).
            _constraint_lookahead = (
                bool(min_thresholds or max_thresholds)
                and not min_budget
                and optimization_mode == "balanced"
            )
            _skwargs = dict(allow_duplicates=allow_duplicates, min_budget=min_budget,
                            require_carry=carryable_only or guaranteed_carryable,
                            max_thresholds=engine_maximum, objective=objective,
                            constraint_lookahead=_constraint_lookahead)
            if min_budget and _beam_search_fn is None:
                from ._min_budget_exact import solve
                from ._min_budget_incumbents import INCUMBENTS, pool_key, profile_key
                import numpy as np
                scores, combos = np.zeros(0), np.zeros((0, remaining_slots), dtype=np.int32)
                model = dict(state='unknown', reason='beam_search')
                def validate_exact(indices):
                    if (len(indices) != remaining_slots or len(set(indices)) != remaining_slots
                            or any(i < 0 or i >= _n_cand for i in indices)):
                        return False
                    chosen = [_numba_search_candidates[int(i)] for i in indices]
                    checked = []
                    _process_combo(chosen, container, sum(o.buyout_price for o in chosen),
                                   priorities, min_thresholds, checked, effective_budget,
                                   objective=objective, max_thresholds=max_thresholds,
                                   require_carry=carryable_only or guaranteed_carryable,
                                   allow_duplicates=allow_duplicates)
                    return bool(checked and _final_valid(checked[0], effective_budget,
                                min_thresholds, max_thresholds, carryable_only,
                                guaranteed_carryable, allow_duplicates))
                def exact_search():
                    return solve(_numba_search_candidates, remaining_slots, effective_budget,
                                 remaining_slots, prot_mult, cont_bonus_dict, min_thresholds or {},
                                 engine_maximum, allow_duplicates, carryable_only or guaranteed_carryable,
                                 validate_exact, seconds=min(5.0, max(0., _min_deadline-time.monotonic())),
                                 guaranteed=guaranteed_carryable)
                chosen = None
                native_allowed = (_n_cand <= 5000 and
                                  len(set(priorities) | set(min_thresholds or {}) | set(engine_maximum)) <= 256)
                witness_key = (pool_key(region, container, _numba_search_candidates,
                                        effective_budget, allow_duplicates, guaranteed_carryable)
                               if native_allowed else None)
                previous = None
                previous_match = None
                physical_profile = (profile_key(region, container, effective_budget,
                                                allow_duplicates, guaranteed_carryable, carryable_only)
                                    if witness_key is not None else None)
                if witness_key is not None:
                    reserves = [(ids, 'exact_pool') for ids in INCUMBENTS.get(witness_key)]
                    reserves.extend((ids, 'physical_remap') for ids in INCUMBENTS.remap(
                        physical_profile, _numba_search_candidates, owned_artifacts or []))
                    seen_reserves = set()
                    for ids, match_kind in reserves:
                        if ids in seen_reserves:
                            continue
                        seen_reserves.add(ids)
                        if validate_exact(ids):
                            price = sum(_numba_search_candidates[i].buyout_price for i in ids)
                            if previous is None or price < previous[0]:
                                previous = (price, ids)
                                previous_match = match_kind
                if _n_cand > 256 or not native_allowed:
                    chosen, model = exact_search()
                if chosen is not None:
                    combos = np.asarray([chosen], dtype=np.int32)
                    scores = np.asarray([-sum(_numba_search_candidates[int(i)].buyout_price for i in chosen)])
                elif (model['state'] != 'infeasible' and model['reason'] not in ('model_limit', 'unsupported_prices')
                      and time.monotonic() < _min_deadline and native_allowed):
                    # Small pools retain their established beam. Large unsupported
                    # pools get one bounded attempt, never an unbounded 50K retry.
                    scores, combos = _search_fn(*_sargs, beam_width=20000 if _n_cand <= 256 else 2000, **_skwargs)
                    if not any(s > -1e17 for s in scores) and _n_cand <= 256:
                        chosen, model = exact_search()
                        if chosen is not None:
                            combos = np.asarray([chosen], dtype=np.int32)
                            scores = np.asarray([-sum(_numba_search_candidates[int(i)].buyout_price for i in chosen)])
                best_current = None
                for rank in np.argsort(scores)[::-1]:
                    if scores[rank] > -1e17 and validate_exact(combos[rank]):
                        ids = tuple(int(i) for i in combos[rank])
                        best_current = (sum(_numba_search_candidates[i].buyout_price for i in ids), ids)
                        break
                if previous is not None and (best_current is None or previous[0] < best_current[0]):
                    price, ids = previous
                    combos = np.concatenate((combos, np.asarray([ids], dtype=np.int32)))
                    scores = np.append(scores, -price)
                    # Feasibility is known from the current game validation;
                    # preserve the solver termination separately, never upgrade
                    # a timeout to a proof of optimality or infeasibility.
                    model = dict(model, solver_state=model['state'], solver_reason=model['reason'],
                                 state='feasible', reason='revalidated_previous_incumbent',
                                 reused_incumbent=True, incumbent_match=previous_match)
                    best_current = previous
                if best_current is not None:
                    INCUMBENTS.remember(witness_key, best_current[1])
                    INCUMBENTS.remember_physical(physical_profile, _numba_search_candidates,
                                                best_current[1], owned_artifacts or [])
                _min_models.append(dict(model, container_id=container.item_id))
            else:
                import os
                _base_beam_width = (int(os.environ.get("STALZONE_BEAM_WIDTH", "10000"))
                                    if _beam_search_fn is None else 2000)
                # Widen the beam when constraints are active: the search space is
                # harder (only a fraction of combinations are feasible) so we need
                # more diverse paths to reliably find the best feasible build.
                # Cap widening at n_cand<=600 to keep latency acceptable.
                _has_constraints = bool(min_thresholds or max_thresholds)
                if _has_constraints and _beam_search_fn is None and _n_cand <= 600:
                    _beam_width = min(_base_beam_width * 2,
                                      int(os.environ.get("STALZONE_BEAM_WIDTH_MAX", "50000")))
                else:
                    _beam_width = _base_beam_width
                scores, combos = _search_fn(*_sargs, beam_width=_beam_width, **_skwargs)
                # ── Multi-pass augmentation ───────────────────────────────────
                # When the user set stat thresholds, run extra beam passes with
                # different candidate orderings (priority-stat order, constraint
                # order, random shuffles) and merge the results.  This finds
                # builds that the single-pass beam missed because the default
                # (item_id, price) ordering buried good candidates.
                # Skipped when a custom _beam_search_fn is injected (tests).
                _run_augment = (
                    _has_constraints
                    and _beam_search_fn is None
                    and _n_cand <= int(os.environ.get("STALZONE_AUGMENT_MAX_CAND", "800"))
                    and os.environ.get("STALZONE_AUGMENT", "1") != "0"
                )
                if _run_augment:
                    from ._search_augment import multi_pass_beam
                    # _sargs layout: (candidates, k, budget, max_counter,
                    #                 prot_mult, cont_bonus_dict, stat_max,
                    #                 priorities, min_thresholds, n_keep)
                    _aug_passes = int(os.environ.get("STALZONE_AUGMENT_PASSES", "2"))
                    scores, combos = multi_pass_beam(
                        scores, combos,
                        _search_fn,
                        _numba_search_candidates,
                        _sargs[1:],          # tail after candidates
                        _skwargs,
                        _beam_width,
                        n_keep,
                        n_random_passes=_aug_passes,
                    )
            # Post-process beam search results (same logic as numba)
            for rank in range(len(scores)):
                if scores[rank] <= -1e17:
                    continue
                combo_indices = combos[rank]
                if len(combo_indices) == 0 or combo_indices[0] < 0:
                    continue
                combo = tuple(_numba_search_candidates[i] for i in combo_indices if i >= 0)
                if len(combo) < remaining_slots:
                    continue
                full_combo = combo
                art_price = sum(o.buyout_price for o in full_combo)
                combined_raw = {}
                combined_names = {}
                for opt in full_combo:
                    for stat, val in opt.stats.items():
                        if stat.startswith("__guaranteed_"): continue
                        combined_raw[stat] = combined_raw.get(stat, 0.0) + val
                        combined_names[stat] = opt.stats_names.get(stat, stat)
                stat_origins = {k: "artefact" for k in combined_raw}
                combined = _apply_build_container_protection(combined_raw, stat_origins, container.inner_protection)
                if container.bonus_stats:
                    for pid, data in container.bonus_stats.items():
                        val = data.get("value", 0) if isinstance(data, dict) else data
                        if val:
                            combined[pid] = combined.get(pid, 0.0) + val
                            combined_names[pid] = data.get("name_ru", pid) if isinstance(data, dict) else pid
                _recalculate_stopping_protection(combined)
                combined, _ = apply_reaction_bonuses(combined)
                from ._derived_stats import add_names, augment
                augment(combined)
                add_names(combined, combined_names)
                skip = False
                for prop_id, min_val in (min_thresholds or {}).items():
                    if combined.get(prop_id, 0) < min_val:
                        skip = True
                        break
                if not skip:
                    for prop_id, max_val in (max_thresholds or {}).items():
                        if combined.get(prop_id, 0) > max_val:
                            skip = True
                            break
                if skip:
                    continue
                score = objective.evaluate(combined, art_price)
                priority_value = score
                carryable, warns = check_carryability(combined)
                _b = ArtifactBuild(
                    artifacts=list(full_combo),
                    container=container if container.item_id else None,
                    total_price=art_price,
                    total_score=score,
                    priority_value=priority_value,
                    combined_stats=combined,
                    combined_stats_names=combined_names,
                    carryable=carryable,
                    carryability_warnings=warns,
                    priorities=priorities,
                    objective_config=objective.metadata(),
                )
                _ca = analyze_build_carryability(_b)
                _b.carryability_analysis = _ca
                _b.guaranteed_carryable = _ca.get("guaranteed_carryable", True)
                if not _final_valid(_b, effective_budget, min_thresholds, max_thresholds,
                                    carryable_only, guaranteed_carryable, allow_duplicates):
                    continue
                all_builds.append(_b)
                total_combos += 1
        elif remaining_slots >= 4:
            from ._numba_search import search as numba_search
            scores, combos = numba_search(
                _numba_search_candidates, remaining_slots, effective_budget, remaining_slots,
                prot_mult, cont_bonus_dict, stat_max, priorities, min_thresholds or {}, n_keep,
                allow_duplicates=allow_duplicates,
                min_budget=min_budget,
                require_carry=carryable_only or guaranteed_carryable,
                max_thresholds=engine_maximum, objective=objective,
            )
            # Convert Numba results directly to ArtifactBuild — skip _process_combo
            # (stats already computed in _numba_search_candidates, no need to recompute)
            for rank in range(len(scores)):
                if scores[rank] <= -1e17:
                    continue
                combo_indices = combos[rank]
                if combo_indices[0] < 0:
                    continue
                combo = tuple(_numba_search_candidates[i] for i in combo_indices if i >= 0)
                if len(combo) < remaining_slots:
                    continue
                full_combo = combo  # owned are now in candidates, no prepend needed
                art_price = sum(o.buyout_price for o in full_combo)
                # Sum stats directly from pre-computed candidate stats
                combined_raw = {}
                combined_names = {}
                for opt in full_combo:
                    for stat, val in opt.stats.items():
                        if stat.startswith("__guaranteed_"): continue
                        combined_raw[stat] = combined_raw.get(stat, 0.0) + val
                        combined_names[stat] = opt.stats_names.get(stat, stat)
                # Apply container protection
                stat_origins = {k: "artefact" for k in combined_raw}
                combined = _apply_build_container_protection(combined_raw, stat_origins, container.inner_protection)
                # Add container bonus
                if container.bonus_stats:
                    for pid, data in container.bonus_stats.items():
                        val = data.get("value", 0) if isinstance(data, dict) else data
                        if val:
                            combined[pid] = combined.get(pid, 0.0) + val
                            combined_names[pid] = data.get("name_ru", pid) if isinstance(data, dict) else pid
                # Stopping protection recalc
                _recalculate_stopping_protection(combined)
                # Apply reaction bonuses (JS applyReactionBonuses)
                combined, _ = apply_reaction_bonuses(combined)
                from ._derived_stats import add_names, augment
                augment(combined)
                add_names(combined, combined_names)
                # Min/max thresholds
                skip = False
                for prop_id, min_val in (min_thresholds or {}).items():
                    if combined.get(prop_id, 0) < min_val:
                        skip = True
                        break
                if not skip:
                    for prop_id, max_val in (max_thresholds or {}).items():
                        if combined.get(prop_id, 0) > max_val:
                            skip = True
                            break
                if skip:
                    continue
                score = objective.evaluate(combined, art_price)
                priority_value = score
                # Carryability
                carryable, warns = check_carryability(combined)
                _b = ArtifactBuild(
                    artifacts=list(full_combo),
                    container=container if container.item_id else None,
                    total_price=art_price,
                    total_score=score,
                    priority_value=priority_value,
                    combined_stats=combined,
                    combined_stats_names=combined_names,
                    carryable=carryable,
                    carryability_warnings=warns,
                    priorities=priorities,
                    objective_config=objective.metadata(),
                )
                _ca = analyze_build_carryability(_b)
                _b.carryability_analysis = _ca
                _b.guaranteed_carryable = _ca.get("guaranteed_carryable", True)
                if not _final_valid(_b, effective_budget, min_thresholds, max_thresholds,
                                    carryable_only, guaranteed_carryable, allow_duplicates):
                    continue
                all_builds.append(_b)
                total_combos += 1

            # Python fallback: brute-force top-50 candidates to catch combos
            # that numba search misses. Only for k=4 (C(50,4)=230K — instant).
            # Skip for k>4 (C(50,7)=100M — too slow in pure Python).
            if remaining_slots == 4:
                _fallback = sorted(_numba_search_candidates,
                    key=lambda o: -o.stats.get(next(iter(priorities), ""), 0))[:50]
                for combo in _it.combinations(_fallback, remaining_slots):
                    if not is_available(combo, allow_duplicates):
                        continue
                    art_price = sum(o.buyout_price for o in combo)
                    if art_price > effective_budget: continue
                    full_combo = combo
                    combined_raw = {}
                    combined_names = {}
                    for opt in full_combo:
                        for stat, val in opt.stats.items():
                            if stat.startswith("__guaranteed_"): continue
                            combined_raw[stat] = combined_raw.get(stat, 0.0) + val
                            combined_names[stat] = opt.stats_names.get(stat, stat)
                    stat_origins = {k: "artefact" for k in combined_raw}
                    combined = _apply_build_container_protection(combined_raw, stat_origins, container.inner_protection)
                    if container.bonus_stats:
                        for pid, data in container.bonus_stats.items():
                            val = data.get("value", 0) if isinstance(data, dict) else data
                            if val:
                                combined[pid] = combined.get(pid, 0.0) + val
                                combined_names[pid] = data.get("name_ru", pid) if isinstance(data, dict) else pid
                    _recalculate_stopping_protection(combined)
                    combined, _ = apply_reaction_bonuses(combined)
                    from ._derived_stats import add_names, augment
                    augment(combined)
                    add_names(combined, combined_names)
                    skip = False
                    for prop_id, min_val in (min_thresholds or {}).items():
                        if combined.get(prop_id, 0) < min_val:
                            skip = True; break
                    if not skip:
                        for prop_id, max_val in (max_thresholds or {}).items():
                            if combined.get(prop_id, 0) > max_val:
                                skip = True; break
                    if skip: continue
                    score = objective.evaluate(combined, art_price)
                    priority_value = score
                    carryable, warns = check_carryability(combined)
                    _b = ArtifactBuild(
                        artifacts=list(full_combo),
                        container=container if container.item_id else None,
                        total_price=art_price,
                        total_score=score,
                        priority_value=priority_value,
                        combined_stats=combined,
                        combined_stats_names=combined_names,
                        carryable=carryable,
                        carryability_warnings=warns,
                        priorities=priorities,
                        objective_config=objective.metadata(),
                    )
                    _ca = analyze_build_carryability(_b)
                    _b.carryability_analysis = _ca
                    _b.guaranteed_carryable = _ca.get("guaranteed_carryable", True)
                    if not _final_valid(_b, effective_budget, min_thresholds, max_thresholds,
                                        carryable_only, guaranteed_carryable, allow_duplicates):
                        continue
                    all_builds.append(_b)
                    total_combos += 1
        else:
            # k < 4: itertools combinations (C(n,3) is small)
            for combo in _it.combinations(candidates, remaining_slots):
                if not is_available(combo, allow_duplicates):
                    continue
                art_price = sum(o.buyout_price for o in combo)
                if art_price > effective_budget:
                    continue
                full_combo = combo  # owned are now in candidates
                _process_combo(full_combo, container, art_price,
                               priorities, min_thresholds, all_builds, effective_budget,
                               stat_max=stat_max, min_budget=min_budget,
                               max_thresholds=max_thresholds, objective=objective,
                               require_carry=carryable_only or guaranteed_carryable, allow_duplicates=allow_duplicates,
                               require_guaranteed=guaranteed_carryable)
                total_combos += 1

        # Prune after each container
        if len(all_builds) > max_keep:
            _sk = objective.sort_key
            all_builds.sort(key=_sk)
            all_builds = all_builds[:max_keep]

    if _quality_context is not None:
        return []  # Preparation only; the separate backend owns solving/validation.

    # Sort: by price (min_budget) or carryable+score (normal).
    # min_budget: sort by price only — user wants cheapest, not carryable.
    all_builds = [_decorate_armor(b) for b in all_builds
                  if len(b.artifacts) == b.container.slots and _final_valid(b, effective_budget, min_thresholds, max_thresholds,
                                  carryable_only, guaranteed_carryable, allow_duplicates)]
    for build in all_builds:
        build._search_request = search_request
    _fsk = objective.sort_key
    all_builds.sort(key=_fsk)

    if carryable_only:
        carryable = [b for b in all_builds if b.carryable]
        pool = carryable
    else:
        pool = all_builds

    # guaranteed_carryable: also filter by max-art% carryability
    if guaranteed_carryable:
        gc = [b for b in pool if b.guaranteed_carryable]
        pool = gc

    # For min_budget: skip Pareto (user wants cheapest, not diverse tradeoffs).
    # Pool is already sorted by price ascending — just take top_n.
    if min_budget:
        from ._min_budget_exact import SearchResults
        return SearchResults(pool[:top_n], _min_models)

    # Pareto frontier extraction: find non-dominated builds across priority objectives.
    # This gives diverse builds — each is the best at some tradeoff.
    pareto_objs = list(objective.keys)
    pareto_pool = pool[:min(len(pool), 500)]  # limit for O(n²) Pareto
    frontier = _pareto_frontier(pareto_pool, pareto_objs, objective) if len(pareto_pool) > 1 else pareto_pool
    # Sort frontier by score (best first)
    frontier.sort(key=objective.sort_key)

    # Merge: Pareto frontier first (diverse), then fill from top-score (if room).
    # Diversity filter: each build must differ by ≥2 artifacts from all selected.
    # Applied to BOTH frontier and pool — prevents near-duplicate builds.
    seen_ids = set()
    result = []
    _selected_item_sets: list[tuple[tuple, frozenset]] = []
    # Always anchor diversity at the request's best objective, even if a Pareto
    # implementation drops an equal-score or quantized-equivalent point.
    anchor = pool[:1]
    for b in anchor + list(frontier) + [b for b in pool if id(b) not in set(id(f) for f in frontier)]:
        if len(result) >= top_n:
            break
        if id(b) in seen_ids:
            continue
        seen_ids.add(id(b))
        b_set = frozenset(a.item_id for a in b.artifacts)
        armor_key = ((getattr(b, "_armor", None) or {}).get("id"),
                     (getattr(b, "_armor", None) or {}).get("level"))
        # Diversity: for small unique sets (≤4), only filter identical builds.
        # For larger sets (≥5), allow 1 item to differ (original behavior).
        _overlap = len(b_set) if len(b_set) <= 4 else len(b_set) - 1
        _too_similar = any(
            armor_key == selected_armor and len(b_set & s) >= _overlap
            for selected_armor, s in _selected_item_sets
        ) if b_set else False
        if not _too_similar:
            result.append(b)
            _selected_item_sets.append((armor_key, b_set))
    # Fallback: fill remaining slots with distinct configurations.
    # Keep the diversity-first selection above, but do not return fewer than
    # top_n merely because the remaining builds share artifact item IDs.
    if len(result) < top_n:
        _result_obj_ids = {id(b) for b in result}

        def _build_signature(_b):
            armor = getattr(_b, "_armor", None) or {}
            return (armor.get("id"), armor.get("level"), tuple(sorted(
                (
                    a.item_id,
                    a.quality,
                    a.ptn,
                    round(float(a.art_percent or 0.0), 3),
                    bool(a.is_owned),
                )
                for a in _b.artifacts
            )))

        _result_signatures = {_build_signature(b) for b in result}

        for b in sorted(pool, key=objective.sort_key):
            if len(result) >= top_n:
                break
            if id(b) in _result_obj_ids:
                continue

            _sig = _build_signature(b)
            if _sig in _result_signatures:
                continue

            result.append(b)
            _result_obj_ids.add(id(b))
            _result_signatures.add(_sig)

    # Sort final diverse result by score (best first) — diversity may reorder.
    if not min_budget:
        result.sort(key=objective.sort_key)
    result = _ordinary_result(result, _rescue_profiles, _rescue_complete, effective_budget,
                              min_thresholds, max_thresholds, carryable_only, guaranteed_carryable,
                               allow_duplicates, priorities, objective, region, search_request, primary_pool=pool,
                               refine_lexicographic=_beam_search_fn is None)
    for b in result:
        _warned_ids: set[str] = set()
        for a in b.artifacts:
            if a.item_id in _hidden_effects and a.item_id not in _warned_ids:
                _warned_ids.add(a.item_id)
                b.hidden_effect_warnings.append({
                    "item_id": a.item_id,
                    "name_ru": a.name_ru,
                    "effect": _hidden_effects[a.item_id]["effect"],
                    "severity": _hidden_effects[a.item_id]["severity"],
                })

    # Safety-net: ensure all builds have carryability analysis
    for b in result:
        if not b.carryability_analysis:
            _ca = analyze_build_carryability(b)
            b.carryability_analysis = _ca
            b.guaranteed_carryable = _ca.get("guaranteed_carryable", True)

    log.info("Processed %d combos, %d builds kept (%d carryable), Pareto frontier %d, returning top %d",
             total_combos, len(all_builds), sum(1 for b in all_builds if b.carryable),
             len(frontier), len(result))
    return result



def _process_combo(combo, container, art_price, priorities, min_thresholds, all_builds,
                   effective_budget: int = 0, stat_max: dict | None = None,
                   min_budget: bool = False, max_thresholds: dict | None = None,
                   objective: Objective | None = None, require_carry: bool = False,
                   allow_duplicates: bool = True, require_guaranteed: bool = False):
    """Process one combination: sum → container protection → bonus stats → stopping → score."""
    if effective_budget and art_price > effective_budget:
        return
    if not is_available(combo, allow_duplicates):
        return
    objective = objective or Objective.create(priorities, stat_max,
                                              "min_budget" if min_budget else "balanced")

    # No counter-art limit — score and carryability naturally balance.
    # A build with too many counter-arts will have low priority stats → low score.

    # Sort by item_id for deterministic order
    combo = tuple(sorted(combo, key=lambda o: o.item_id))

    combined_stats_raw: dict[str, float] = {}
    combined_names: dict[str, str] = {}
    stat_origins: dict[str, str] = {}
    total_score = 0.0

    for opt in combo:
        qr = QUALITY_RANGES.get(opt.quality, QUALITY_RANGES[0])
        stats_c, stats_r, sn, _ = compute_artifact_stats(
            opt.base_stats, opt.bonus_stats, opt.unlocked_bonus_ids,
            opt.art_percent, opt.ptn, qr["min"], qr["range"], opt.stats_random,
            container.inner_protection, container.effectiveness,
        )
        for stat, val in stats_c.items():
            combined_stats_raw[stat] = combined_stats_raw.get(stat, 0.0) + val
            if stat not in combined_names:
                combined_names[stat] = sn.get(stat, stat)
            if stat not in stat_origins:
                stat_origins[stat] = "artefact"

    # 1. Container protection at BUILD level (JS counterStatsByContainer)
    combined_stats = _apply_build_container_protection(
        combined_stats_raw, stat_origins, container.inner_protection
    )

    # 2. Container bonus stats (e.g. +6.2 speed, +40 weight) — after protection
    if container.bonus_stats:
        for pid, data in container.bonus_stats.items():
            val = data.get("value", 0) if isinstance(data, dict) else data
            if val:
                combined_stats[pid] = combined_stats.get(pid, 0.0) + val
                combined_names[pid] = data.get("name_ru", pid) if isinstance(data, dict) else pid

    # 3. stopping_protection recalc (build level)
    _recalculate_stopping_protection(combined_stats)
    # Apply reaction bonuses (JS applyReactionBonuses)
    combined_stats, _ = apply_reaction_bonuses(combined_stats)
    from ._derived_stats import add_names, augment
    augment(combined_stats)
    add_names(combined_stats, combined_names)

    # Derived metrics are opt-in; old combined_stats and payloads stay unchanged.
    from ._survivability import requested, values, REGISTRY
    if requested(priorities, min_thresholds, max_thresholds):
        combined_stats.update(values(combined_stats))
        combined_names.update({k:v[0] for k,v in REGISTRY.items()})
    # 4. Min/max thresholds
    for prop_id, min_val in (min_thresholds or {}).items():
        if combined_stats.get(prop_id, 0) < min_val:
            return
    for prop_id, max_val in (max_thresholds or {}).items():
        if combined_stats.get(prop_id, 0) > max_val:
            return

    # 5. The shared exact objective on completed game stats, without penalties.
    priority_value = objective.evaluate(combined_stats, art_price)
    total_score = priority_value
    if not all(math.isfinite(value) for value in combined_stats.values()):
        return

    carryable, warnings = check_carryability(combined_stats)
    if require_carry and not carryable:
        return

    _build_obj = ArtifactBuild(
        artifacts=list(combo),
        container=container if container.item_id else None,
        total_price=art_price,
        total_score=total_score,
        priority_value=priority_value,
        combined_stats=combined_stats,
        combined_stats_names=combined_names,
        carryable=carryable,
        carryability_warnings=warnings,
        priorities=priorities,
        objective_config=objective.metadata(),
    )
    _carry_analysis = analyze_build_carryability(_build_obj)
    _build_obj.carryability_analysis = _carry_analysis
    _build_obj.guaranteed_carryable = _carry_analysis.get("guaranteed_carryable", True)
    if not require_guaranteed or _build_obj.guaranteed_carryable:
        all_builds.append(_build_obj)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=== Budget 11M, Бочонок (7 slots), priorities: health_bonus=10, bullet_dmg_factor=8 ===")
    builds = build_artifact_sets(
        budget=11_000_000,
        tolerance_pct=8.0,
        priorities={"health_bonus": 10, "bullet_dmg_factor": 8},
        container_id="rgoz",
        top_n=3,
    )
    for i, b in enumerate(builds, 1):
        status = "✓" if b.carryable else "✗"
        print(f"\nBuild #{i} {status}: score={b.total_score:.1f} price={b.total_price:,}")
        for w in b.carryability_warnings:
            print(f"  ⚠ {w}")
        if b.container:
            print(f"  Container: {b.container.name_ru} (prot={b.container.inner_protection}% eff={b.container.effectiveness}%)")
        for a in b.artifacts:
            print(f"  {a.name_ru} [{a.quality_name}] ptn={a.ptn} {a.art_percent:.1f}% {'(В ХРАН)' if a.is_owned else f'{a.buyout_price:,}'}")

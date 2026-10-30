"""
Lunar-zone bridge — export/import/verify builds with lunar-zone.com.

Export:  Convert optimizer ArtifactBuild → lunar-zone JSON format.
Import:  Fetch a lunar-zone build by UUID, convert to our format, compute stats.
Verify:  Compare our computed stats vs lunar-zone's displayed stats.

Lunar-zone API endpoints (discovered from calculator-BQe7nubd.js):
  GET  /web-api/calculator/get_public_builds.php?offset=N   — list public builds
  GET  /web-api/calculator/get_builds.php?build_id=UUID     — fetch build by UUID
  POST /web-api/calculator/save_build.php                   — save new build (auth required)
  POST /web-api/calculator/update_build.php                 — update existing (auth required)

Build URL format: https://lunar-zone.com/calculator?build_id=UUID
"""
import json
import logging
import sqlite3
from pathlib import Path

import requests

from .config import Config
from .database import get_connection

log = logging.getLogger(__name__)

LUNARZONE_BASE = "https://lunar-zone.com"
LUNARZONE_API = f"{LUNARZONE_BASE}/web-api/calculator"

# Russian stat name ↔ prop_id mapping (from artefacts.bonus_stats)
RUSSIAN_TO_PROP: dict[str, str] = {
    "Биологическое заражение": "biological_accumulation",
    "Восстановление выносливости": "stamina_regeneration_bonus",
    "Выносливость": "stamina_bonus",
    "Горение": "combustion_accumulation",
    "Живучесть": "health_bonus",
    "Защита от биозаражения": "biological_protection",
    "Защита от взрыва": "explosion_dmg_factor",
    "Защита от кровотечения": "bleeding_protection",
    "Защита от огня": "burn_dmg_factor",
    "Защита от пси-излучения": "psycho_protection",
    "Защита от радиации": "radiation_protection",
    "Защита от разрыва": "tear_dmg_factor",
    "Защита от температуры": "thermal_protection",
    "Кровотечение": "bleeding_accumulation",
    "Отдача": "recoil_bonus",
    "Переносимый вес": "max_weight_bonus",
    "Периодическое лечение": "artefakt_heal",
    "Покачивание": "wiggle_bonus",
    "Пси-излучение": "psycho_accumulation",
    "Пулестойкость": "bullet_dmg_factor",
    "Радиация": "radiation_accumulation",
    "Реакция на ожог": "reaction_to_burn",
    "Реакция на разрыв": "reaction_to_tear",
    "Реакция на хим. ожог": "reaction_to_chemical_burn",
    "Реакция на электричество": "reaction_to_electroshock",
    "Регенерация здоровья": "regeneration_bonus",
    "Скорость передвижения": "speed_modifier",
    "Стойкость": "stopping_protection",
    "Температура": "thermal_accumulation",
    "Химзащита": "chemical_burn_dmg_factor",
    "Электрозащита": "electra_dmg_factor",
    "Эффективность лечения": "heal_efficiency",
}
PROP_TO_RUSSIAN = {v: k for k, v in RUSSIAN_TO_PROP.items()}

# Quality tier ↔ lunar-zone rarity strings
# MUST match api_client.QUALITY_RANGES: q0=Common(85-100), q1=Uncommon(100-115),
# q2=Special(115-130), q3=Rare(130-145), q4=Exceptional(145-160), q5=Legendary(160-175)
QUALITY_TO_RARITY = {
    0: "common", 1: "uncommon", 2: "special", 3: "rare", 4: "exceptional", 5: "legendary",
}
RARITY_TO_QUALITY = {v: k for k, v in QUALITY_TO_RARITY.items()}
# relic maps to q5 (Legendary) — no q6 in QUALITY_RANGES yet
RARITY_TO_QUALITY["relic"] = 5


# ───────────────────────── Export ─────────────────────────

def export_to_lunarzone(build, armor_id: str | None = None) -> dict:
    """Convert optimizer ArtifactBuild → lunar-zone build JSON.

    The returned JSON matches lunar-zone's build_data format and can be:
    - POSTed to save_build.php (requires auth)
    - Used with browser relay to programmatically set up the calculator
    """
    artifacts_lz = []
    for a in build.artifacts:
        # Get Russian names for unlocked bonuses
        selected_secondary = []
        for b in (a.bonus_stats or []):
            if isinstance(b, dict) and a.ptn >= b.get("unlock_ptn", 99):
                name_ru = b.get("name_ru", "")
                if name_ru:
                    selected_secondary.append(name_ru)

        artifacts_lz.append({
            "id": a.item_id,
            "name": a.name_ru,
            "level": a.ptn,
            "rarity": QUALITY_TO_RARITY.get(a.quality, "rare"),
            "quality": round(a.art_percent, 1),
            "selectedSecondary": selected_secondary,
        })

    # Armor
    armor_data = None
    selected_armor = getattr(build, '_armor', None)
    if selected_armor is not None or armor_id:
        from .builder import load_armor
        if selected_armor is not None and armor_id and selected_armor['id'] != armor_id:
            raise ValueError('Export armor must match the computed build')
        armor = selected_armor or load_armor(armor_id)
        if armor:
            armor_id = armor['id']
            armor_data = {
                "id": armor["id"],
                "name": armor["name_ru"],
                "level": armor["level"],
                "rarity": "RANK_MASTER",  # will be overridden if we have rank
            }
            # Use actual rank from DB
            with get_connection() as conn:
                row = conn.execute("SELECT rank FROM armors WHERE id=?", (armor_id,)).fetchone()
                if row:
                    rank_map = {
                        "Отмычка": "RANK_PICKLOCK", "Бродяга": "RANK_EXPLORER",
                        "Сталкер": "RANK_STALKER", "Мастер": "RANK_MASTER",
                        "Легенда": "RANK_LEGEND", "Реликвия": "RANK_RELIC",
                    }
                    armor_data["rarity"] = rank_map.get(row["rank"], "RANK_MASTER")

    # Container
    container_data = None
    if build.container:
        container_data = {
            "id": build.container.item_id,
            "name": build.container.name_ru,
            "slots": build.container.slots,
        }

    return {
        "name": f"Optimizer {build.total_price // 1000000}M",
        "build_type": "combined",
        "tags": ["optimizer"],
        "build_data": {
            "armor": armor_data,
            "container": container_data,
            "artifacts": artifacts_lz,
        },
        # Extra metadata for our UI
        "_optimizer_meta": {
            "total_price": build.total_price,
            "total_score": round(build.total_score, 2),
            "effective_health": round(_calc_ehp(build.combined_stats), 1),
            "carryable": build.carryable,
        },
    }


# ───────────────────────── Import ─────────────────────────

def fetch_lunarzone_build(build_id: str) -> dict | None:
    """Fetch a lunar-zone build by UUID. Returns the build JSON or None."""
    url = f"{LUNARZONE_API}/get_builds.php?build_id={build_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            log.error("lunar-zone get_builds returned %d", r.status_code)
            return None
        data = r.json()
        if not data.get("success") or not data.get("build"):
            log.error("lunar-zone build not found: %s", data.get("error", "?"))
            return None
        return data["build"]
    except Exception as e:
        log.error("Failed to fetch lunar-zone build %s: %s", build_id, e)
        return None


def parse_lunarzone_build(lunar_build: dict) -> dict:
    """Parse a lunar-zone build into our format.

    Returns a dict with:
      - artifacts: [{item_id, quality, ptn, art_percent, bonus_props: [prop_id]}]
      - container_id: str | None
      - armor_id: str | None
      - name: str
      - build_type: str
      - lunar_url: str
    """
    bd = lunar_build.get("build_data", {})
    build_id = lunar_build.get("uuid", "")
    name = lunar_build.get("name", "Imported")
    build_type = lunar_build.get("build_type", "combined")

    # Container
    container_id = None
    container = bd.get("container")
    if container and isinstance(container, dict):
        container_id = container.get("id")

    # Armor
    armor_id = None
    armor_level = 15
    armor = bd.get("armor")
    if armor and isinstance(armor, dict):
        armor_id = armor.get("id")
        armor_level = armor.get("level", 15)

    # Artifacts
    artifacts = []
    for art in bd.get("artifacts", []):
        if not isinstance(art, dict):
            continue
        item_id = art.get("id", "")
        if not item_id:
            continue

        # Map rarity → quality tier
        rarity = art.get("rarity", "rare")
        quality = RARITY_TO_QUALITY.get(rarity, 2)

        # art_percent from quality value
        art_percent = art.get("quality", 100.0)

        # ptn from level
        ptn = art.get("level", 15)

        # selectedSecondary → prop_ids
        bonus_props = []
        for sec_name in art.get("selectedSecondary", []):
            prop_id = RUSSIAN_TO_PROP.get(sec_name)
            if prop_id:
                bonus_props.append(prop_id)

        artifacts.append({
            "item_id": item_id,
            "quality": quality,
            "ptn": ptn,
            "art_percent": art_percent,
            "bonus_props": bonus_props,
            "_lunar_name": art.get("name", ""),
        })

    return {
        "name": name,
        "build_type": build_type,
        "container_id": container_id,
        "armor_id": armor_id,
        "armor_level": armor_level,
        "artifacts": artifacts,
        "lunar_url": f"{LUNARZONE_BASE}/calculator?build_id={build_id}",
    }


def import_lunarzone_build(build_id: str) -> dict | None:
    """Full import: fetch lunar-zone build, parse, and compute stats.

    Returns the parsed build + computed stats using our formulas.
    """
    lunar_build = fetch_lunarzone_build(build_id)
    if not lunar_build:
        return None

    parsed = parse_lunarzone_build(lunar_build)
    return parsed


# ───────────────────────── List public builds ─────────────────────────

def list_public_builds(offset: int = 0, limit: int = 20) -> list[dict]:
    """Fetch public builds from lunar-zone. Returns list of {uuid, name, build_type}."""
    url = f"{LUNARZONE_API}/get_public_builds.php?offset={offset}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        if not data.get("success"):
            return []
        builds = data.get("builds", [])
        return [
            {
                "uuid": b.get("uuid", ""),
                "name": b.get("name", ""),
                "build_type": b.get("build_type", ""),
                "tags": b.get("tags", []),
                "views": b.get("views_count", 0),
                "likes": b.get("likes_count", 0),
                "lunar_url": f"{LUNARZONE_BASE}/calculator?build_id={b.get('uuid', '')}",
            }
            for b in builds[:limit]
        ]
    except Exception as e:
        log.error("Failed to list public builds: %s", e)
        return []


# ───────────────────────── Helpers ─────────────────────────

def _calc_ehp(stats: dict) -> float:
    """Calculate effective health from combined stats."""
    try:
        from .builder import calculate_effective_health
        return calculate_effective_health(stats)
    except Exception:
        return 0.0


# ───────────────────────── Direct stat computation ─────────────────────────

def compute_specific_build(artifacts_spec: list[dict],
                           container_id: str | None = None,
                           armor_id: str | None = None, armor_level: int = 15) -> dict | None:
    """Compute stats for an EXACT set of artifacts (no search).

    Replicates _process_combo pipeline:
      load → compute_artifact_stats → sum → container protection →
      container bonus → stopping recalc → reaction bonuses → armor

    Args:
      artifacts_spec: [{item_id, quality, ptn, art_percent, bonus_props}]
      container_id: container item_id
      armor_id: armor item_id

    Returns ArtifactBuild or None.
    """
    from .builder import (
        ArtifactBuild, ArtifactOption, ContainerOption,
        compute_artifact_stats, _apply_build_container_protection,
        _recalculate_stopping_protection, apply_reaction_bonuses,
        calculate_effective_health, calculate_healing_per_second,
    )
    from .api_client import QUALITY_RANGES, QUALITY_NAMES
    import json as _json

    # Load container FIRST (needed for effectiveness in compute_artifact_stats)
    container_opt = None
    inner_protection = 0.0
    effectiveness = 100.0
    if container_id:
        with get_connection() as conn:
            c = conn.execute(
                "SELECT * FROM containers WHERE id=?", (container_id,)
            ).fetchone()
        if c:
            container_opt = ContainerOption(
                item_id=c["id"], name_ru=c["name_ru"],
                name_en=c["name_en"] if "name_en" in c.keys() else "",
                item_type=c["item_type"], slots=c["slots"], buyout_price=0,
                inner_protection=c["inner_protection"] if "inner_protection" in c.keys() else 0.0,
                effectiveness=c["effectiveness"] if "effectiveness" in c.keys() else 100.0,
                bonus_stats=_json.loads(c["bonus_stats"]) if c["bonus_stats"] else None,
                icon_url=c["icon_url"] if "icon_url" in c.keys() else "",
            )
            inner_protection = container_opt.inner_protection
            effectiveness = container_opt.effectiveness

    art_opts: list[ArtifactOption] = []
    combined_raw: dict[str, float] = {}
    combined_names: dict[str, str] = {}

    with get_connection() as conn:
        for spec in artifacts_spec:
            item_id = spec["item_id"]
            row = conn.execute(
                "SELECT id, name_ru, name_en, subcategory, stats, bonus_stats, icon_url "
                "FROM artefacts WHERE id=?",
                (item_id,),
            ).fetchone()
            if not row:
                log.warning("Artifact %s not found in DB", item_id)
                continue

            base_stats = _json.loads(row["stats"]) if row["stats"] else []
            bonus_stats = _json.loads(row["bonus_stats"]) if row["bonus_stats"] else []
            quality = spec.get("quality", 0)
            ptn = spec.get("ptn", 0)
            art_percent = spec.get("art_percent", 100.0)
            bonus_ids = spec.get("bonus_props", spec.get("bonus_ids", []))
            qr = QUALITY_RANGES.get(quality, QUALITY_RANGES[0])

            # Pass inner_protection + effectiveness from container (same as _process_combo)
            stats_c, stats_r, sn, _ = compute_artifact_stats(
                base_stats, bonus_stats, bonus_ids,
                art_percent, ptn, qr["min"], qr["range"], None,
                inner_protection, effectiveness,
            )

            opt = ArtifactOption(
                item_id=row["id"], name_ru=row["name_ru"], name_en=row["name_en"],
                subcategory=row["subcategory"], quality=quality,
                quality_name=QUALITY_NAMES.get(quality, "?"),
                art_percent=art_percent, buyout_price=spec.get("buyout_price", 0), ptn=ptn,
                stats=stats_c, stats_raw=stats_r, stats_names=sn,
                unlocked_bonus_ids=bonus_ids, icon_url=row["icon_url"],
                base_stats=base_stats, bonus_stats=bonus_stats,
                stats_random=None, is_owned=spec.get("is_owned", spec.get("auction_lot_id") is None),
                auction_lot_id=spec.get("auction_lot_id"), physical_id=spec.get("physical_id"),
                owned_instance_id=spec.get("owned_instance_id"),
            )
            art_opts.append(opt)

            for stat, val in stats_c.items():
                combined_raw[stat] = combined_raw.get(stat, 0.0) + val
                if stat in sn:
                    combined_names[stat] = sn[stat]

    if not art_opts:
        return None

    # 1. Container protection at BUILD level
    stat_origins = {k: "artefact" for k in combined_raw}
    combined = _apply_build_container_protection(combined_raw, stat_origins, inner_protection)

    # 2. Container bonus stats (after protection)
    if container_opt and container_opt.bonus_stats:
        for pid, data in container_opt.bonus_stats.items():
            val = data.get("value", 0) if isinstance(data, dict) else data
            combined[pid] = combined.get(pid, 0.0) + val
            name = data.get("name_ru", pid) if isinstance(data, dict) else pid
            combined_names[pid] = name

    # 3. Stopping protection recalc
    _recalculate_stopping_protection(combined)

    # 4. Reaction bonuses
    combined, _ = apply_reaction_bonuses(combined)
    from ._derived_stats import add_names, augment
    augment(combined)
    add_names(combined, combined_names)

    # 5. Create build
    # 5. Check carryability
    from .builder import check_carryability
    carryable, warnings = check_carryability(combined)
    build = ArtifactBuild(
        artifacts=art_opts,
        container=container_opt,
        total_price=sum(option.buyout_price for option in art_opts),
        combined_stats=combined,
        combined_stats_names=combined_names,
        carryable=carryable,
        carryability_warnings=warnings,
        priorities={},
    )

    from ._physical_items import is_available
    if not is_available(art_opts, True):
        return None
    # 6. Apply armor (if any)
    if armor_id:
        from .builder import apply_armor_to_builds
        applied = apply_armor_to_builds([build], armor_id, armor_level)
        if not applied:
            return None
        build = applied[0]

    return build


# ───────────────────────── Template mode: auction search ─────────────────────────

def _prop_id_to_eapi(prop_id: str) -> str:
    """Convert prop_id (e.g. 'bullet_dmg_factor') to eAPI short name (e.g. 'BULLET_DMG').

    Uses reverse of builder.BONUS_NAME_MAP.
    """
    from .builder import BONUS_NAME_MAP
    upper = prop_id.upper()
    # Build reverse map: DB prop_id (upper) → eAPI short name
    rev = {v: k for k, v in BONUS_NAME_MAP.items()}
    return rev.get(upper, upper)


def search_template_lots(build_id: str, exact_ptn: bool = True) -> dict | None:
    """Search auction for lots matching a lunar-zone build template.

    For each artifact in the template, finds auction lots with:
    - Same item_id
    - Same quality tier
    - Same bonus_properties (set match, order-independent)
    - Same ptn (if exact_ptn=True) or ptn >= template (if False)

    Returns dict with template info, per-artifact matching lots, and computed stats.
    """
    from .api_client import QUALITY_RANGES
    parsed = import_lunarzone_build(build_id)
    if not parsed:
        return None

    container_id = parsed.get("container_id")
    armor_id = parsed.get("armor_id")
    artifacts = parsed.get("artifacts", [])

    # Build artifacts_spec for compute_specific_build
    artifacts_spec = []
    for art in artifacts:
        artifacts_spec.append({
            "item_id": art["item_id"],
            "quality": art["quality"],
            "ptn": art["ptn"],
            "art_percent": art["art_percent"],
            "bonus_props": art.get("bonus_props", []),
        })

    # Compute stats for 3 scenarios: template (author's luck), best (MAX), worst (MIN)
    build = compute_specific_build(artifacts_spec, container_id, armor_id, parsed.get("armor_level", 15))
    computed = build.to_dict() if build else None

    # Best case: all artifacts at MAX art_percent for their quality tier
    best_spec = []
    worst_spec = []
    for art in artifacts:
        qr = QUALITY_RANGES.get(art["quality"], QUALITY_RANGES[0])
        q_max = qr["min"] + qr["range"]
        q_min = qr["min"]
        best_spec.append({**art, "art_percent": float(q_max)})
        worst_spec.append({**art, "art_percent": float(q_min)})
    best_build = compute_specific_build(best_spec, container_id, armor_id, parsed.get("armor_level", 15))
    worst_build = compute_specific_build(worst_spec, container_id, armor_id, parsed.get("armor_level", 15))
    best_stats = best_build.to_dict() if best_build else None
    worst_stats = worst_build.to_dict() if worst_build else None

    # Search auction for matching lots per artifact
    from .config import Config
    results_per_artifact = []
    total_min_price = 0
    all_found = True

    used_lot_ids = set()  # track lots already assigned to duplicate artifacts

    with get_connection() as conn:
        for i, art in enumerate(artifacts):
            item_id = art["item_id"]
            quality = art["quality"]
            ptn = art["ptn"]
            bonus_props = art.get("bonus_props", [])

            # Convert prop_ids to eAPI short names for matching
            eapi_bonuses = sorted(set(_prop_id_to_eapi(p) for p in bonus_props))

            # Search auction_lots (include stats_random + additional_raw for research detection)
            if exact_ptn:
                sql = (
                    "SELECT id, item_id, buyout_price, quality, quality_name, "
                    "art_percent, ptn, bonus_properties, end_time, region, "
                    "stats_random, additional_raw "
                    "FROM auction_lots "
                    "WHERE item_id=? AND quality=? AND ptn=? AND region=? "
                    "AND buyout_price > 0 "
                    "ORDER BY buyout_price ASC LIMIT 20"
                )
                rows = conn.execute(sql, (item_id, quality, ptn, Config.REGION)).fetchall()
            else:
                sql = (
                    "SELECT id, item_id, buyout_price, quality, quality_name, "
                    "art_percent, ptn, bonus_properties, end_time, region, "
                    "stats_random, additional_raw "
                    "FROM auction_lots "
                    "WHERE item_id=? AND quality=? AND ptn>=? AND region=? "
                    "AND buyout_price > 0 "
                    "ORDER BY buyout_price ASC LIMIT 20"
                )
                rows = conn.execute(sql, (item_id, quality, ptn, Config.REGION)).fetchall()

            # Filter by bonus_properties match (set comparison) + art_percent / research status
            import json as _json
            template_ap = art["art_percent"]
            qr = QUALITY_RANGES.get(quality, QUALITY_RANGES[0])
            q_min, q_max = qr["min"], qr["min"] + qr["range"]
            exact_lots = []
            near_lots = []
            for r in rows:
                lot_bonuses = _json.loads(r["bonus_properties"]) if r["bonus_properties"] else []
                lot_bonuses_sorted = sorted(set(lot_bonuses))
                if lot_bonuses_sorted != eapi_bonuses:
                    continue
                if r["id"] in used_lot_ids:
                    continue  # already assigned to a previous duplicate slot

                lot_ap = round(r["art_percent"], 1)
                add_raw = _json.loads(r["additional_raw"]) if r["additional_raw"] else {} if "additional_raw" in r.keys() else {}
                sr = r["stats_random"] if "stats_random" in r.keys() else 0.0
                md_k = add_raw.get("md_k") if add_raw else None
                # Researched = has specific stats_random or md_k value (>0)
                researched = (sr is not None and sr > 0) or (md_k is not None and md_k > 0)

                if not researched:
                    # Unresearched: art_percent will be random in [q_min, q_max] after research.
                    # If template_ap is within range → exact match (can achieve through research).
                    ap_match = "exact" if q_min <= template_ap <= q_max else "near"
                    ap_diff = 0.0 if q_min <= template_ap <= q_max else abs(lot_ap - template_ap)
                    entry = {
                        "id": r["id"],
                        "buyout_price": r["buyout_price"],
                        "art_percent": lot_ap,
                        "template_art_percent": template_ap,
                        "ap_diff": round(ap_diff, 1),
                        "ap_match": ap_match,
                        "researched": False,
                        "quality_range": [q_min, q_max],
                        "ptn": r["ptn"],
                        "end_time": r["end_time"],
                        "region": r["region"],
                    }
                else:
                    # Researched: art_percent is fixed. Must match exactly.
                    ap_diff = abs(lot_ap - template_ap)
                    ap_match = "exact" if ap_diff < 0.5 else "near"
                    entry = {
                        "id": r["id"],
                        "buyout_price": r["buyout_price"],
                        "art_percent": lot_ap,
                        "template_art_percent": template_ap,
                        "ap_diff": round(ap_diff, 1),
                        "ap_match": ap_match,
                        "researched": True,
                        "ptn": r["ptn"],
                        "end_time": r["end_time"],
                        "region": r["region"],
                    }

                if ap_match == "exact":
                    exact_lots.append(entry)
                else:
                    near_lots.append(entry)

            # Also find lots with WRONG bonuses (same item/quality/ptn but different bonuses)
            wrong_bonus_lots = []
            for r in rows:
                lot_bonuses = _json.loads(r["bonus_properties"]) if r["bonus_properties"] else []
                lot_bonuses_sorted = sorted(set(lot_bonuses))
                if lot_bonuses_sorted != eapi_bonuses:
                    wrong_bonus_lots.append({
                        "id": r["id"],
                        "buyout_price": r["buyout_price"],
                        "art_percent": round(r["art_percent"], 1),
                        "bonus_properties": lot_bonuses_sorted,
                    })

            # Only EXACT matches count toward all_found / total_min_price
            min_price = exact_lots[0]["buyout_price"] if exact_lots else None
            if min_price is None:
                all_found = False
            else:
                total_min_price += min_price
            # Mark cheapest exact (or near) lot as used for duplicate handling
            claimed = exact_lots[0]["id"] if exact_lots else (near_lots[0]["id"] if near_lots else None)
            if claimed is not None:
                used_lot_ids.add(claimed)

            # Get artifact name from DB
            name_row = conn.execute(
                "SELECT name_ru FROM artefacts WHERE id=?", (item_id,)
            ).fetchone()
            name_ru = name_row["name_ru"] if name_row else item_id

            matching_lots = exact_lots + near_lots
            results_per_artifact.append({
                "slot": i + 1,
                "item_id": item_id,
                "name_ru": name_ru,
                "quality": quality,
                "ptn": ptn,
                "art_percent": template_ap,
                "template_bonuses": bonus_props,
                "eapi_bonuses": eapi_bonuses,
                "exact_lots": exact_lots,
                "exact_count": len(exact_lots),
                "near_lots": near_lots,
                "near_count": len(near_lots),
                "matching_lots": matching_lots,
                "matching_count": len(matching_lots),
                "min_price": min_price,
                "wrong_bonus_lots": wrong_bonus_lots[:3],
            })

    return {
        "build_id": build_id,
        "build_name": parsed.get("name", ""),
        "container_id": container_id,
        "armor_id": armor_id,
        "lunar_url": parsed.get("lunar_url", ""),
        "artifacts": results_per_artifact,
        "all_found": all_found,
        "total_min_price": total_min_price if all_found else None,
        "computed_stats": computed,
        "best_case_stats": best_stats,
        "worst_case_stats": worst_stats,
    }


# ───────────────────────── Template build mode ─────────────────────────

def build_from_template(
    build_id: str,
    budget: int,
    container_id: str | None = None,
    armor_id: str | None = None,
    tolerance_pct: float = 8.0,
    relax_quality: bool = False,
) -> dict | None:
    """Greedily build from a lunar-zone template within budget.

    Picks the cheapest matching lot for each template slot (same item_id +
    quality + bonuses). If total exceeds budget, drops the most expensive
    slot and retries. Returns the best achievable build + stats.

    relax_quality=True: find cheapest lot of each item_id regardless of
    quality or bonuses — allows Q0/Q1/Q2 analogs at lower budget.
    """
    from .api_client import QUALITY_RANGES
    from .builder import _normalize_bonus_id
    from .config import Config
    import json as _json

    parsed = import_lunarzone_build(build_id)
    if not parsed:
        return None

    template_artifacts = parsed.get("artifacts", [])
    # Use template's container if not overridden
    if not container_id:
        container_id = parsed.get("container_id")
    if not armor_id:
        armor_id = parsed.get("armor_id")

    effective_budget = int(budget * (1 + tolerance_pct / 100.0))

    # For each template slot, find ALL matching lots
    slot_lots: list[list[dict]] = []  # slot_lots[i] = sorted list of matching lots for slot i
    with get_connection() as conn:
        for art in template_artifacts:
            item_id = art["item_id"]
            quality = art["quality"]
            ptn = art["ptn"]
            bonus_props = art.get("bonus_props", [])
            eapi_bonuses = sorted(set(_prop_id_to_eapi(p) for p in bonus_props))

            if relax_quality:
                # Same quality, any ptn/bonuses — realistic prices, stats from template
                rows = conn.execute(
                    "SELECT id, buyout_price, quality, ptn, art_percent, "
                    "bonus_properties, end_time, region "
                    "FROM auction_lots "
                    "WHERE item_id=? AND quality=? AND region=? AND buyout_price > 0 "
                    "ORDER BY buyout_price ASC",
                    (item_id, quality, Config.REGION),
                ).fetchall()
                matching = [{
                    "id": r["id"],
                    "item_id": item_id,
                    "buyout_price": r["buyout_price"],
                    "quality": r["quality"],
                    "ptn": r["ptn"],
                    "art_percent": round(r["art_percent"], 1),
                    "end_time": r["end_time"],
                    "bonus_properties": r["bonus_properties"],
                } for r in rows]
            else:
                # Exact match: same quality + same bonuses
                rows = conn.execute(
                    "SELECT id, buyout_price, quality, ptn, art_percent, "
                    "bonus_properties, end_time, region "
                    "FROM auction_lots "
                    "WHERE item_id=? AND quality=? AND region=? AND buyout_price > 0 "
                    "ORDER BY buyout_price ASC",
                    (item_id, quality, Config.REGION),
                ).fetchall()
                matching = []
                for r in rows:
                    lot_bonuses = _json.loads(r["bonus_properties"]) if r["bonus_properties"] else []
                    if sorted(set(lot_bonuses)) == eapi_bonuses:
                        matching.append({
                            "id": r["id"],
                            "item_id": item_id,
                            "buyout_price": r["buyout_price"],
                            "quality": r["quality"],
                            "ptn": r["ptn"],
                            "art_percent": round(r["art_percent"], 1),
                            "end_time": r["end_time"],
                        })
            slot_lots.append(matching)

    # Greedy: pick cheapest for each slot, drop most expensive if over budget
    available_slots = list(range(len(template_artifacts)))
    dropped_slots: list[dict] = []

    while True:
        # Pick cheapest lot for each available slot
        selected: list[dict] = []
        used_lot_ids = set()
        for slot_idx in available_slots:
            lots = slot_lots[slot_idx]
            # Skip lots already used by earlier slots (duplicate handling)
            available = [l for l in lots if l["id"] not in used_lot_ids]
            if available:
                lot = available[0]  # cheapest (sorted ASC)
                selected.append({**lot, "slot": slot_idx})
                used_lot_ids.add(lot["id"])
            else:
                selected.append(None)  # no matching lot for this slot

        total = sum(s["buyout_price"] for s in selected if s)
        if total <= effective_budget or len(available_slots) == 0:
            break

        # Drop the most expensive slot
        valid = [(s["buyout_price"], i) for i, s in enumerate(selected) if s]
        if not valid:
            break
        valid.sort(reverse=True)
        drop_slot_idx = available_slots[valid[0][1]]  # actual slot index
        dropped = template_artifacts[drop_slot_idx]
        name_row = None
        with get_connection() as conn:
            name_row = conn.execute(
                "SELECT name_ru FROM artefacts WHERE id=?", (dropped["item_id"],)
            ).fetchone()
        dropped_slots.append({
            "slot": drop_slot_idx + 1,
            "item_id": dropped["item_id"],
            "name_ru": name_row["name_ru"] if name_row else dropped["item_id"],
            "quality": dropped["quality"],
            "reason": "over_budget",
        })
        available_slots.remove(drop_slot_idx)
    # Budget optimization: upgrade slots to better quality with remaining budget
    if relax_quality:
        while True:
            current_total = sum(s["buyout_price"] for s in selected if s)
            remaining = effective_budget - current_total
            if remaining <= 0:
                break
            best_upgrade = None  # (idx_in_selected, new_lot, quality_score)
            for i, s in enumerate(selected):
                if s is None:
                    continue
                slot_idx = s["slot"]
                lots = slot_lots[slot_idx]
                used_ids = {x["id"] for x in selected if x and x["id"] != s["id"]}
                # Find affordable upgrade: higher quality within remaining budget
                for lot in lots:
                    cost = lot["buyout_price"] - s["buyout_price"]
                    if cost <= 0 or cost > remaining or lot["id"] in used_ids:
                        continue
                    # Quality score: art_percent + ptn*5 (more ptn = more bonuses)
                    q_score = lot["art_percent"] + lot["ptn"] * 5
                    cur_score = s["art_percent"] + s["ptn"] * 5
                    gain = q_score - cur_score
                    if gain > 0:
                        ratio = gain / cost if cost > 0 else 999
                        if best_upgrade is None or ratio > best_upgrade[2]:
                            best_upgrade = (i, lot, ratio)
            if best_upgrade is None:
                break
            i, new_lot, _ = best_upgrade
            selected[i] = {**new_lot, "slot": selected[i]["slot"]}
    # Recalculate total after budget optimization upgrades
    total = sum(s["buyout_price"] for s in selected if s)

    # Build artifacts_spec from selected lots
    artifacts_spec = []
    selected_lots_info = []
    with get_connection() as conn:
        for s in selected:
            if s is None:
                continue
            template_art = template_artifacts[s["slot"]]
            # Look up artifact name
            name_row = conn.execute(
                "SELECT name_ru FROM artefacts WHERE id=?", (s["item_id"],)
            ).fetchone()
            artifacts_spec.append({
                "item_id": s["item_id"],
                "auction_lot_id": s["id"], "physical_id": f"auction:{Config.REGION}:{s['id']}",
                "buyout_price": s["buyout_price"], "is_owned": False,
                "quality": s["quality"],
                "ptn": s["ptn"],
                "art_percent": s["art_percent"],
                "bonus_props": [_normalize_bonus_id(b) for b in (_json.loads(s.get("bonus_properties") or "[]"))] or template_art.get("bonus_props", []),
            })
            selected_lots_info.append({**s, "name_ru": name_row["name_ru"] if name_row else s["item_id"]})

    if not artifacts_spec:
        return {
            "build_id": build_id,
            "build_name": parsed.get("name", ""),
            "budget": budget,
            "effective_budget": effective_budget,
            "container_id": container_id,
            "armor_id": armor_id,
            "error": "No affordable template artifacts within budget",
            "dropped_slots": dropped_slots,
        }

    # Compute stats
    build = compute_specific_build(artifacts_spec, container_id, armor_id, parsed.get("armor_level", 15))
    computed = build.to_dict() if build else None

    # Template stats (what lunar-zone shows)
    template_spec = [
        {"item_id": a["item_id"], "quality": a["quality"], "ptn": a["ptn"],
         "art_percent": a["art_percent"], "bonus_props": a.get("bonus_props", [])}
        for a in template_artifacts
    ]
    template_build = compute_specific_build(template_spec, container_id, armor_id, parsed.get("armor_level", 15))
    template_stats = template_build.to_dict() if template_build else None

    return {
        "build_id": build_id,
        "build_name": parsed.get("name", ""),
        "budget": budget,
        "effective_budget": effective_budget,
        "container_id": container_id,
        "armor_id": armor_id,
        "lunar_url": parsed.get("lunar_url", ""),
        "template_slots": len(template_artifacts),
        "filled_slots": len(artifacts_spec),
        "dropped_slots": dropped_slots,
        "total_price": total,
        "selected_lots": selected_lots_info,
        "computed_stats": computed,
        "template_stats": template_stats,
    }

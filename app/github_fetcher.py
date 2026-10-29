"""
Fetch artifact data from the official Stalzone database on GitHub.

Downloads the entire repository as a single ZIP archive (one request, no API
rate limits) and processes artefact JSONs locally.

Repo: EXBO-Studio/stalzone-database
Structure: stalzone-database-main/{realm}/items/artefact/{subcategory}/*.json
Each JSON: id, category, name.lines.ru, color, infoBlocks (range elements with stat properties)
"""
import io
import json
import logging
import os
import time
import zipfile

import requests

from .config import Config

log = logging.getLogger(__name__)

# Artifact subcategories in the database
ARTEFACT_SUBCATEGORIES = [
    "biochemical",
    "electrophysical",
    "gravity",
    "thermal",
    "other_arts",
]

# Mapping of stat property keys (from JSON name.key) to short names
STAT_KEY_MAP = {
    "stamina_bonus": "stamina",
    "biological_accumulation": "bio_acc",
    "bleeding_accumulation": "bleed_acc",
    "thermal_accumulation": "thermal_acc",
    "chemical_accumulation": "chem_acc",
    "electrical_accumulation": "electr_acc",
    "gravitational_accumulation": "grav_acc",
    "health_regeneration": "regen",
    "radiation_accumulation": "rad_acc",
    "weight": "weight",
    "carry_weight_bonus": "carry_weight",
    "max_health_bonus": "max_health",
    "armor_piercing_bonus": "armor_pierce",
    "projectile_speed_bonus": "proj_speed",
    "reload_speed_bonus": "reload_speed",
    "movement_speed_bonus": "move_speed",
    "melee_damage_bonus": "melee_dmg",
    "fear_resistance": "fear_resist",
    "telepathy_resistance": "telep_resist",
    "psychedelic_resistance": "psy_resist",
    "exposure_to_anomalies_bonus": "anomaly_bonus",
    "durability_of_artifacts_bonus": "art_durability",
}

# ZIP download URL — single request, no rate limit
_ZIP_URL = (
    f"https://codeload.github.com/{Config.DB_REPO_OWNER}/{Config.DB_REPO_NAME}/"
    f"zip/refs/heads/{Config.DB_REPO_BRANCH}"
)


def _zip_cache_path() -> str:
    return os.path.join(Config.DATA_DIR, f"{Config.DB_REPO_NAME}.zip")


def download_db_zip(force: bool = False) -> str:
    """Download the Stalzone database repo as a ZIP archive.

    Caches to data/stalzone-database.zip. Returns path to the cached ZIP.
    """
    cache = _zip_cache_path()
    os.makedirs(Config.DATA_DIR, exist_ok=True)

    # Use cache if it exists and is less than 24 hours old
    if not force and os.path.exists(cache):
        age = time.time() - os.path.getmtime(cache)
        if age < 86400:
            log.info("Using cached ZIP (%s, %.1f hours old)", cache, age / 3600)
            return cache

    log.info("Downloading Stalzone database ZIP from GitHub...")
    for attempt in range(3):
        try:
            r = requests.get(_ZIP_URL, timeout=120, stream=True)
            if r.status_code == 200:
                # Write to cache file
                with open(cache, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                size_mb = os.path.getsize(cache) / (1024 * 1024)
                log.info("Downloaded %.1f MB to %s", size_mb, cache)
                return cache
            log.error("ZIP download failed: HTTP %d", r.status_code)
        except requests.RequestException as e:
            log.warning("Download attempt %d failed: %s", attempt + 1, e)
            time.sleep(3 * (attempt + 1))

    # Fallback: if we have a cached copy even if stale, use it
    if os.path.exists(cache):
        log.warning("Using stale cache as fallback: %s", cache)
        return cache

    raise RuntimeError("Failed to download Stalzone database ZIP")


def _read_artefact_jsons_from_zip(zip_path: str) -> dict[str, dict]:
    """Read all artefact JSONs from the ZIP archive.

    Returns dict: {item_id: raw_json_dict}
    """
    result = {}
    # Expected path prefix inside ZIP:
    # stalzone-database-main/{realm}/items/artefact/{subcategory}/{id}.json
    prefix = f"{Config.DB_REPO_NAME}-{Config.DB_REPO_BRANCH}/{Config.DB_REALM}/items/artefact/"

    with zipfile.ZipFile(zip_path, "r") as zf:
        artefact_files = [
            n for n in zf.namelist()
            if n.startswith(prefix) and n.endswith(".json") and "/_variants/" not in n
        ]
        log.info("Found %d artefact JSONs in ZIP", len(artefact_files))

        for name in artefact_files:
            with zf.open(name) as f:
                try:
                    raw = json.loads(f.read().decode("utf-8"))
                    item_id = raw.get("id", os.path.basename(name).replace(".json", ""))
                    result[item_id] = raw
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    log.warning("Failed to parse %s: %s", name, e)

    return result


def extract_artefact_info(raw: dict) -> dict:
    """Parse the raw artefact JSON into a flat dict with name, category, stats.

    Extracts:
    - id, name (ru/en), category, color/quality, base_price, weight
    - stat properties from range elements — uses exact name.lines.ru translation
    - stat_key for correlation with auction bonus_properties
    """
    name_obj = raw.get("name", {})
    lines = name_obj.get("lines", {}) if isinstance(name_obj, dict) else {}
    name_ru = lines.get("ru", raw.get("id", ""))
    name_en = lines.get("en", name_ru)

    info = {
        "id": raw["id"],
        "name_ru": name_ru,
        "name_en": name_en,
        "category": raw.get("category", ""),
        "color": raw.get("color", ""),
        "base_price": 0.0,
        "weight": 0.0,
        "stats": {},
        "description": "",
    }

    for block in raw.get("infoBlocks", []):
        if block.get("type") != "list":
            # text block → description
            if block.get("type") == "text":
                text_obj = block.get("text", {})
                if isinstance(text_obj, dict) and text_obj.get("type") == "translation":
                    info["description"] = text_obj.get("lines", {}).get("ru", "")
            continue

        for elem in block.get("elements", []):
            etype = elem.get("type")

            if etype == "numeric":
                name_obj = elem.get("name", {})
                key = name_obj.get("key", "") if isinstance(name_obj, dict) else ""
                # Extract base_price and weight by key
                if "base_price" in key:
                    info["base_price"] = elem.get("value", 0.0)
                elif "tooltip.info.weight" in key:
                    info["weight"] = elem.get("value", 0.0)

            elif etype == "range":
                name_obj = elem.get("name", {})
                if not isinstance(name_obj, dict):
                    continue
                key = name_obj.get("key", "")
                prop_lines = name_obj.get("lines", {})
                # Use the exact Russian translation from JSON
                prop_name_ru = prop_lines.get("ru", key)

                # Extract the property identifier from key for correlation
                # e.g. "stalker.artefact_properties.factor.stamina_bonus" → "stamina_bonus"
                # Also "stalker.artefact_properties.factor.biological_accumulation" → "biological_accumulation"
                prop_id = key.split(".")[-1] if key else prop_name_ru
                # Convert to uppercase for matching with auction bonus_properties
                # e.g. "stamina_bonus" → "STAMINA_BONUS"
                prop_id_upper = prop_id.upper() if prop_id else ""

                info["stats"][prop_id] = {
                    "min": elem.get("min", 0.0),
                    "max": elem.get("max", 0.0),
                    "name_ru": prop_name_ru,
                    "key": key,
                    "bonus_id": prop_id_upper,
                }

    return info


def _subcategory_from_path(path: str) -> str:
    """Extract subcategory from a ZIP path like:
    stalzone-database-main/ru/items/artefact/biochemical/04yr.json → biochemical
    """
    parts = path.split("/")
    # Find "artefact" in parts, take next element
    for i, p in enumerate(parts):
        if p == "artefact" and i + 1 < len(parts):
            return parts[i + 1]
    return ""

def _read_jsons_from_zip(zip_path: str, item_category: str) -> dict[str, dict]:
    """Read all JSONs for a given item category (backpacks, containers) from ZIP.

    Returns dict: {item_id: raw_json_dict}
    """
    result = {}
    prefix = f"{Config.DB_REPO_NAME}-{Config.DB_REPO_BRANCH}/{Config.DB_REALM}/items/{item_category}/"

    with zipfile.ZipFile(zip_path, "r") as zf:
        files = [
            n for n in zf.namelist()
            if n.startswith(prefix) and n.endswith(".json") and "/_variants/" not in n
        ]
        log.info("Found %d %s JSONs in ZIP", len(files), item_category)

        for name in files:
            with zf.open(name) as f:
                try:
                    raw = json.loads(f.read().decode("utf-8"))
                    item_id = raw.get("id", os.path.basename(name).replace(".json", ""))
                    result[item_id] = raw
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    log.warning("Failed to parse %s: %s", name, e)

    return result


def extract_container_info(raw: dict, item_type: str) -> dict:
    """Parse a backpack or container JSON into a flat dict.

    Extracts: id, name, color/rank, weight, slots (artifact capacity), base_price.
    """
    name_obj = raw.get("name", {})
    lines = name_obj.get("lines", {}) if isinstance(name_obj, dict) else {}
    name_ru = lines.get("ru", raw.get("id", ""))
    name_en = lines.get("en", name_ru)

    info = {
        "id": raw["id"],
        "name_ru": name_ru,
        "name_en": name_en,
        "category": raw.get("category", item_type),
        "item_type": item_type,  # "backpacks" or "containers"
        "color": raw.get("color", ""),
        "base_price": 0.0,
        "weight": 0.0,
        "slots": 0,
        "inner_protection": 0.0,
        "effectiveness": 0.0,
        "bonus_stats": {},  # {prop_id: {value, name_ru}}
        "icon_url": "",
    }

    # Known non-bonus keys — everything else is a bonus stat
    _skip_keys = ("backpack.info.size", "tooltip.info.weight", "base_price",
                  "inner_protection", "effectiveness")

    for block in raw.get("infoBlocks", []):
        if block.get("type") != "list":
            continue
        for elem in block.get("elements", []):
            etype = elem.get("type")
            if etype not in ("numeric", "range"):
                continue
            name_obj = elem.get("name", {})
            if not isinstance(name_obj, dict):
                continue
            key = name_obj.get("key", "")

            if any(s in key for s in _skip_keys):
                if "backpack.info.size" in key:
                    info["slots"] = int(elem.get("value", 0))
                elif "tooltip.info.weight" in key:
                    info["weight"] = elem.get("value", 0.0)
                elif "base_price" in key:
                    info["base_price"] = elem.get("value", 0.0)
                elif "inner_protection" in key:
                    info["inner_protection"] = elem.get("value", 0.0)
                elif "effectiveness" in key:
                    info["effectiveness"] = elem.get("value", 0.0)
            else:
                # Bonus stat — extract value and Russian name
                prop_id = key.split(".")[-1] if key else ""
                prop_name_ru = name_obj.get("lines", {}).get("ru", prop_id)
                if etype == "numeric":
                    val = elem.get("value", 0.0)
                else:  # range
                    val = (elem.get("min", 0.0) + elem.get("max", 0.0)) / 2
                if prop_id:
                    info["bonus_stats"][prop_id] = {
                        "value": val,
                        "name_ru": prop_name_ru,
                    }

    # Build icon URL
    info["icon_url"] = (
        f"{Config.RAW_GITHUB_BASE}/{Config.DB_REPO_OWNER}/{Config.DB_REPO_NAME}/"
        f"{Config.DB_REPO_BRANCH}/{Config.DB_REALM}/icons/{item_type}/{info['id']}.png"
    )

    return info


def fetch_all_containers() -> list[dict]:
    """Fetch all backpacks and containers from the GitHub database ZIP.

    Returns list of dicts with: id, name_ru, name_en, category, item_type,
    color, base_price, weight, slots, icon_url.
    """
    zip_path = download_db_zip()
    all_items = []

    for item_type in ("backpacks", "containers"):
        log.info("Reading %s from ZIP...", item_type)
        raw_items = _read_jsons_from_zip(zip_path, item_type)
        for item_id, raw in raw_items.items():
            info = extract_container_info(raw, item_type)
            all_items.append(info)
            log.debug("Parsed %s (%s): %d slots", info["name_ru"], item_id, info["slots"])
        log.info("Total %s fetched: %d", item_type, len(raw_items))

    log.info("Total containers+backpacks fetched: %d", len(all_items))
    return all_items


def fetch_all_artefacts() -> list[dict]:
    """Fetch all artefacts from the GitHub database ZIP, parse into flat dicts.

    Downloads a single ZIP archive (one HTTP request, no GitHub API rate limits)
    and processes all artefact JSONs locally.

    Returns list of dicts with: id, name_ru, name_en, category, color, base_price,
    weight, stats (dict of {short_name: {min, max, key}}), description, subcategory.
    """
    zip_path = download_db_zip()

    log.info("Reading artefact JSONs from ZIP...")
    raw_artefacts = _read_artefact_jsons_from_zip(zip_path)
    log.info("Parsed %d artefact JSONs from ZIP", len(raw_artefacts))

    # Determine subcategory for each from the ZIP path
    with zipfile.ZipFile(zip_path, "r") as zf:
        prefix = f"{Config.DB_REPO_NAME}-{Config.DB_REPO_BRANCH}/{Config.DB_REALM}/items/artefact/"
        path_to_subcat = {}
        for name in zf.namelist():
            if name.startswith(prefix) and name.endswith(".json") and "/_variants/" not in name:
                subcat = _subcategory_from_path(name)
                item_id = os.path.basename(name).replace(".json", "")
                path_to_subcat[item_id] = subcat

    all_artefacts = []
    for item_id, raw in raw_artefacts.items():
        info = extract_artefact_info(raw)
        info["subcategory"] = path_to_subcat.get(item_id, "")
        all_artefacts.append(info)
        log.debug("Parsed %s (%s): %d stats", info["name_ru"], item_id, len(info["stats"]))

    log.info("Total artefacts fetched: %d", len(all_artefacts))
    return all_artefacts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    arts = fetch_all_artefacts()
    print(f"\n=== {len(arts)} artefacts total ===")
    for a in arts[:5]:
        print(f"  {a['id']} {a['name_ru']} ({a['subcategory']}) color={a['color']} price={a['base_price']}")
        print(f"    stats: {list(a['stats'].keys())}")

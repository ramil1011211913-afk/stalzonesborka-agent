"""
Fetch artifact data from stalzone.wiki builds calculator API.

API: https://stalzone.wiki/api/builds-calculator/items/?category=artefact
Single request returns all 106 artifacts with:
  - Base stats (stats): range values with Russian names
  - Additional/bonus stats (additionalStats): up to 3 bonus properties
    ordered by unlock level (index 0 → ptn=5, 1 → ptn=10, 2 → ptn=15)
  - Category, color, rarity
"""
import json
import logging
import os
import time

import requests

from .config import Config

log = logging.getLogger(__name__)

WIKI_API_URL = "https://stalzone.wiki/api/builds-calculator/items/?category=artefact"


def _wiki_get(url: str, retries: int = 3) -> list | None:
    """GET from stalzone.wiki API with retry."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            log.error("Wiki API %s -> %d: %s", url[:60], r.status_code, r.text[:200])
            return None
        except requests.RequestException as e:
            log.warning("Wiki request failed (attempt %d): %s", attempt + 1, e)
            time.sleep(2 * (attempt + 1))
    return None


def fetch_all_artefacts() -> list[dict]:
    """Fetch all artefacts from stalzone.wiki API.

    Returns list of dicts with:
      id, name_ru, name_en, category, subcategory, color,
      base_stats (list of {prop_id, name_ru, min, max}),
      bonus_stats (list of {prop_id, name_ru, min, max, unlock_ptn}),
      icon_url, description
    """
    log.info("Fetching artefacts from stalzone.wiki API...")
    data = _wiki_get(WIKI_API_URL)
    if not isinstance(data, list):
        log.error("Expected list from wiki API, got %s", type(data))
        return []

    all_artefacts = []
    for item in data:
        exbo_id = item.get("exbo_id", "")
        name_obj = item.get("name", {})
        lines = name_obj.get("lines", {}) if isinstance(name_obj, dict) else {}
        name_ru = lines.get("ru", exbo_id)
        name_en = lines.get("en", name_ru)
        category = item.get("category", "")
        subcategory = category.split("/")[-1] if "/" in category else ""
        color = item.get("color", "")

        # Parse base stats
        base_stats = []
        for s in item.get("stats", []):
            sname = s.get("name", {})
            skey = sname.get("key", "") if isinstance(sname, dict) else ""
            sru = sname.get("lines", {}).get("ru", skey) if isinstance(sname, dict) else skey
            val = s.get("value", {})
            prop_id = skey.split(".")[-1] if skey else ""
            base_stats.append({
                "prop_id": prop_id,
                "name_ru": sru,
                "min": val.get("min", 0.0),
                "max": val.get("max", 0.0),
                "is_positive": s.get("is_positive", True),
            })

        # Parse bonus/additional stats — order = unlock order by ptn
        bonus_stats = []
        for i, s in enumerate(item.get("additionalStats", [])):
            sname = s.get("name", {})
            skey = sname.get("key", "") if isinstance(sname, dict) else ""
            sru = sname.get("lines", {}).get("ru", skey) if isinstance(sname, dict) else skey
            val = s.get("value", {})
            prop_id = skey.split(".")[-1] if skey else ""
            # Unlock ptn: index 0 → 5, index 1 → 10, index 2 → 15
            unlock_ptn = (i + 1) * 5
            bonus_stats.append({
                "prop_id": prop_id,
                "name_ru": sru,
                "min": val.get("min", 0.0),
                "max": val.get("max", 0.0),
                "unlock_ptn": unlock_ptn,
                "is_positive": s.get("is_positive", True),
            })

        # Build icon URL from GitHub raw (wiki CDN also works)
        icon_url = (
            f"https://cdn.stalcraft.wiki/exbo_item_parser/artefact/{subcategory}/{exbo_id}.png"
        )

        artefact = {
            "id": exbo_id,
            "name_ru": name_ru,
            "name_en": name_en,
            "category": category,
            "subcategory": subcategory,
            "color": color,
            "base_stats": base_stats,
            "bonus_stats": bonus_stats,
            "icon_url": icon_url,
        }
        all_artefacts.append(artefact)

    log.info("Fetched %d artefacts from wiki API", len(all_artefacts))
    return all_artefacts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    arts = fetch_all_artefacts()
    print(f"\n=== {len(arts)} artefacts ===")
    for a in arts[:3]:
        print(f"\n{a['id']} {a['name_ru']} ({a['subcategory']})")
        print(f"  Base stats:")
        for s in a["base_stats"]:
            print(f"    {s['name_ru']:25s} [{s['min']}, {s['max']}]")
        print(f"  Bonus stats:")
        for s in a["bonus_stats"]:
            print(f"    {s['name_ru']:25s} [{s['min']}, {s['max']}] unlock_ptn={s['unlock_ptn']}")

"""
Armor fetcher — parses exoskeleton data from stalzone-database.zip.

Reads armor JSONs (clothes, combat, combined, scientist, device) and extracts
numeric stats for each armor at each level (1-15). Stores in `armors` table.

Usage:
    python -m app.armor_fetcher          # full run
    python -m app.armor_fetcher --combined  # combined only (exoskeletons)
"""
import json
import logging
import zipfile
from pathlib import Path

from .database import get_connection
from .config import Config

log = logging.getLogger(__name__)

DB_PATH = Path(Config.DATA_DIR) / "stalzone.db"
ZIP_PATH = Path(Config.DATA_DIR) / "stalzone-database.zip"

# Armor categories in stalzone-database
ARMOR_CATEGORIES = ["clothes", "combat", "combined", "scientist", "device"]

# Map Russian stat names to our prop_ids
STAT_NAME_MAP = {
    "Пулестойкость": "bullet_dmg_factor",
    "Защита от разрыва": "tear_dmg_factor",
    "Защита от взрыва": "explosion_dmg_factor",
    "Скорость передвижения": "speed_modifier",
    "Скорость бега": "sprint_speed_modifier",
    "Переносимый вес": "max_weight_bonus",
    "Живучесть": "health_bonus",
    "Выносливость": "stamina_bonus",
    "Восстановление выносливости": "stamina_regeneration_bonus",
    "Регенерация здоровья": "regeneration_bonus",
    "Эффективность лечения": "heal_efficiency",
    "Кровотечение": "bleeding_accumulation",
    "Радиация": "radiation_accumulation",
    "Пси-излучение": "psycho_accumulation",
    "Биологическое заражение": "biological_accumulation",
    "Температура": "thermal_accumulation",
    "Мороз": "frost_accumulation",
    "Защита от кровотечения": "bleeding_protection",
    "Защита от радиации": "radiation_protection",
    "Защита от температуры": "thermal_protection",
    "Защита от биозаражения": "biological_protection",
    "Защита от пси-излучения": "psycho_protection",
    "Электрозащита": "electro_protection",
    "Защита от огня": "burn_protection",
    "Защита от химического ожога": "chemical_burn_protection",
}


def _extract_numeric_stats(json_data: dict) -> dict[str, float]:
    """Extract numeric stats from armor JSON infoBlocks."""
    stats = {}
    for block in json_data.get("infoBlocks", []):
        for el in block.get("elements", []):
            if el.get("type") != "numeric":
                continue
            name_obj = el.get("name", {})
            name_ru = name_obj.get("lines", {}).get("ru", "")
            value = el.get("value", 0)
            prop_id = STAT_NAME_MAP.get(name_ru)
            if prop_id and prop_id not in stats:
                # First occurrence wins (some stats appear twice: base + enhanced)
                stats[prop_id] = float(value)
    return stats


def _get_name_ru(json_data: dict) -> str:
    """Extract Russian name from armor JSON."""
    name_obj = json_data.get("name", {})
    return name_obj.get("lines", {}).get("ru", json_data.get("id", "?"))


def _get_rank(json_data: dict) -> str:
    """Extract rank (Мастер, Ветеран, etc.) from armor JSON."""
    for block in json_data.get("infoBlocks", []):
        for el in block.get("elements", []):
            if el.get("type") == "key-value":
                key_obj = el.get("key", {})
                key_ru = key_obj.get("lines", {}).get("ru", "")
                if key_ru == "Ранг":
                    val_obj = el.get("value", {})
                    return val_obj.get("lines", {}).get("ru", "")
    return ""


def fetch_armors(categories: list[str] | None = None, levels: list[int] | None = None):
    """Parse armor data from stalzone-database.zip and store in DB.

    Args:
        categories: armor categories to parse (default: all)
        levels: armor levels to parse (default: [15] = max level only)
    """
    categories = categories or ARMOR_CATEGORIES
    levels = levels or [15]

    if not ZIP_PATH.exists():
        log.error("stalzone-database.zip not found at %s", ZIP_PATH)
        return 0

    z = zipfile.ZipFile(str(ZIP_PATH))
    all_names = set(z.namelist())

    armors = []

    for category in categories:
        for level in levels:
            # Pattern: stalzone-database-main/ru/items/armor/{category}/_variants/{id}/{level}.json
            prefix = f"stalzone-database-main/ru/items/armor/{category}/_variants/"
            suffix = f"/{level}.json"

            for name in sorted(all_names):
                if not name.startswith(prefix) or not name.endswith(suffix):
                    continue
                armor_id = name[len(prefix):].rsplit("/", 1)[0]
                try:
                    data = z.read(name)
                    j = json.loads(data)
                except (json.JSONDecodeError, KeyError):
                    continue

                name_ru = _get_name_ru(j)
                rank = _get_rank(j)
                stats = _extract_numeric_stats(j)

                # Also read base file for icon_url
                base_path = f"stalzone-database-main/ru/items/armor/{category}/{armor_id}.json"
                icon_url = ""
                if base_path in all_names:
                    try:
                        base_j = json.loads(z.read(base_path))
                        icon_url = f"stalzone-database-main/global/icons/armor/{category}/{armor_id}.png"
                    except (json.JSONDecodeError, KeyError):
                        pass

                armors.append({
                    "id": armor_id,
                    "name_ru": name_ru,
                    "category": category,
                    "rank": rank,
                    "level": level,
                    "stats": json.dumps(stats, ensure_ascii=False),
                    "icon_url": icon_url,
                })

    # Store in DB
    with get_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS armors")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS armors (
                id TEXT NOT NULL,
                name_ru TEXT NOT NULL,
                category TEXT NOT NULL,
                rank TEXT NOT NULL DEFAULT '',
                level INTEGER NOT NULL DEFAULT 15,
                stats TEXT NOT NULL DEFAULT '{}',
                icon_url TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (id, level)
            )
        """)

        for armor in armors:
            conn.execute("""
                INSERT OR REPLACE INTO armors (id, name_ru, category, rank, level, stats, icon_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                armor["id"], armor["name_ru"], armor["category"],
                armor["rank"], armor["level"], armor["stats"], armor["icon_url"],
            ))
        conn.commit()

    log.info("Loaded %d armor variants (%d categories, levels=%s)", len(armors), len(categories), levels)

    # Print summary
    by_cat = {}
    for a in armors:
        by_cat.setdefault(a["category"], []).append(a)
    for cat, items in sorted(by_cat.items()):
        log.info("  %s: %d armors", cat, len(items))

    return len(armors)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import sys
    if "--combined" in sys.argv:
        count = fetch_armors(categories=["combined"])
    else:
        count = fetch_armors()
    print(f"\nLoaded {count} armor variants into stalzone.db")

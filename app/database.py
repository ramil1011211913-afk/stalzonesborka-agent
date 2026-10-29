"""
SQLite database layer for Stalzone artifact collector.

Tables:
  - artefacts: static artifact data (id, name, category, base_price, weight, stats JSON)
  - auction_history: sold lot prices (item_id, price, quality, art_percent, time, additional JSON)
  - auction_lots: active auction lots (item_id, buyout_price, quality, art_percent, additional JSON)
"""
import json
import logging
import os
import sqlite3
import time
from contextlib import closing, nullcontext
from datetime import datetime, timezone

from .config import Config

log = logging.getLogger(__name__)

# --- FIX: Retry configuration for SQLite "database is locked" ---
_DB_LOCKED_RETRIES = 5
_DB_LOCKED_RETRY_DELAY = 0.5  # seconds


def get_db_path() -> str:
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    return Config.DB_PATH


def get_connection() -> sqlite3.Connection:
    """Open a new SQLite connection with WAL mode and busy_timeout.

    Returns a connection that can be used as a context manager:
        with get_connection() as conn:
            conn.execute(...)

    Note: SQLite uses OS-level file locking, so concurrent reads are fast.
    Multiple short-lived connections are fine for this workload (read-heavy,
    single-writer). For high-throughput scenarios, consider using a shared
    connection or connection pool.
    """
    conn = sqlite3.connect(get_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # --- FIX: busy_timeout makes SQLite wait up to 5s for a lock instead of
    # immediately raising "database is locked" ---
    conn.execute("PRAGMA busy_timeout=5000")
    # FK disabled: auction item_id can reference artefacts OR containers
    conn.execute("PRAGMA foreign_keys=OFF")
    return conn


def execute_with_retry(conn: sqlite3.Connection, sql: str, params: tuple = ()):
    """Execute SQL with retry on database lock.

    Retries up to _DB_LOCKED_RETRIES times with exponential backoff
    when SQLite reports the database is locked.
    """
    for attempt in range(_DB_LOCKED_RETRIES):
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < _DB_LOCKED_RETRIES - 1:
                delay = _DB_LOCKED_RETRY_DELAY * (2 ** attempt)
                log.warning("Database locked, retry %d/%d in %.1fs: %s",
                             attempt + 1, _DB_LOCKED_RETRIES, delay, e)
                time.sleep(delay)
            else:
                raise


def init_db():
    """Create all tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS artefacts (
                id          TEXT PRIMARY KEY,
                name_ru     TEXT NOT NULL,
                name_en     TEXT NOT NULL DEFAULT '',
                category    TEXT NOT NULL DEFAULT '',
                subcategory TEXT NOT NULL DEFAULT '',
                color       TEXT NOT NULL DEFAULT '',
                base_price  REAL NOT NULL DEFAULT 0,
                weight      REAL NOT NULL DEFAULT 0,
                stats       TEXT NOT NULL DEFAULT '{}',   -- JSON base stats
                bonus_stats TEXT NOT NULL DEFAULT '[]',   -- JSON bonus/additional stats
                description TEXT NOT NULL DEFAULT '',
                icon_url    TEXT NOT NULL DEFAULT '',
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS auction_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id      TEXT NOT NULL,
                price        INTEGER NOT NULL,
                amount       INTEGER NOT NULL DEFAULT 1,
                quality      INTEGER NOT NULL DEFAULT 0,
                quality_name TEXT NOT NULL DEFAULT '',
                art_percent  REAL NOT NULL DEFAULT 0,
                stats_random REAL NOT NULL DEFAULT 0,
                ndmg         REAL NOT NULL DEFAULT 0,
                bonus_properties TEXT NOT NULL DEFAULT '[]',
                spawn_time   INTEGER NOT NULL DEFAULT 0,
                sold_time    TEXT NOT NULL,
                additional_raw TEXT NOT NULL DEFAULT '{}',
                region       TEXT NOT NULL DEFAULT 'EU',
                collected_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS auction_lots (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id      TEXT NOT NULL,
                start_price  INTEGER NOT NULL DEFAULT 0,
                buyout_price INTEGER NOT NULL DEFAULT 0,
                amount       INTEGER NOT NULL DEFAULT 1,
                quality      INTEGER NOT NULL DEFAULT 0,
                quality_name TEXT NOT NULL DEFAULT '',
                art_percent  REAL NOT NULL DEFAULT 0,
                stats_random REAL NOT NULL DEFAULT 0,
                ndmg         REAL NOT NULL DEFAULT 0,
                bonus_properties TEXT NOT NULL DEFAULT '[]',
                upgrade_bonus REAL NOT NULL DEFAULT 0,
                start_time   TEXT NOT NULL DEFAULT '',
                end_time     TEXT NOT NULL DEFAULT '',
                additional_raw TEXT NOT NULL DEFAULT '{}',
                region       TEXT NOT NULL DEFAULT 'EU',
                collected_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS containers (
                id          TEXT PRIMARY KEY,
                name_ru     TEXT NOT NULL,
                name_en     TEXT NOT NULL DEFAULT '',
                category    TEXT NOT NULL DEFAULT '',
                item_type   TEXT NOT NULL DEFAULT '',  -- backpacks | containers
                color       TEXT NOT NULL DEFAULT '',
                base_price  REAL NOT NULL DEFAULT 0,
                weight      REAL NOT NULL DEFAULT 0,
                slots       INTEGER NOT NULL DEFAULT 0,
                inner_protection REAL NOT NULL DEFAULT 0,
                effectiveness REAL NOT NULL DEFAULT 0,
                bonus_stats TEXT NOT NULL DEFAULT '{}',  -- JSON {prop_id: {value, name_ru}}
                icon_url    TEXT NOT NULL DEFAULT '',
                updated_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_history_item ON auction_history(item_id);
            CREATE INDEX IF NOT EXISTS idx_history_quality ON auction_history(quality);
            CREATE INDEX IF NOT EXISTS idx_lots_item ON auction_lots(item_id);
            CREATE INDEX IF NOT EXISTS idx_lots_quality ON auction_lots(quality);
            CREATE INDEX IF NOT EXISTS idx_lots_buyout ON auction_lots(buyout_price);
            CREATE INDEX IF NOT EXISTS idx_containers_slots ON containers(slots);
            CREATE TABLE IF NOT EXISTS armors (
                id TEXT NOT NULL,
                name_ru TEXT NOT NULL,
                category TEXT NOT NULL,
                rank TEXT NOT NULL DEFAULT '',
                level INTEGER NOT NULL DEFAULT 15,
                stats TEXT NOT NULL DEFAULT '{}',
                icon_url TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (id, level)
            );
        """)
        conn.commit()

    # Migrations: add columns if they don't exist (safe to re-run)
    # --- FIX: Use context manager for migration queries too ---
    with get_connection() as conn:
        existing_cols = {r["name"] for r in conn.execute("PRAGMA table_info(auction_history)")}
        if "ptn" not in existing_cols:
            conn.execute("ALTER TABLE auction_history ADD COLUMN ptn INTEGER DEFAULT 0")
        if "bonus_properties" not in existing_cols:
            conn.execute("ALTER TABLE auction_history ADD COLUMN bonus_properties TEXT DEFAULT '[]'")

        existing_lots_cols = {r["name"] for r in conn.execute("PRAGMA table_info(auction_lots)")}
        if "ptn" not in existing_lots_cols:
            conn.execute("ALTER TABLE auction_lots ADD COLUMN ptn INTEGER DEFAULT 0")

        existing_art_cols = {r["name"] for r in conn.execute("PRAGMA table_info(artefacts)")}

        if "icon_url" not in existing_art_cols:
            conn.execute("ALTER TABLE artefacts ADD COLUMN icon_url TEXT NOT NULL DEFAULT ''")
        if "bonus_stats" not in existing_art_cols:
            conn.execute("ALTER TABLE artefacts ADD COLUMN bonus_stats TEXT DEFAULT '[]'")
        existing_cont_cols = {r["name"] for r in conn.execute("PRAGMA table_info(containers)")}
        if "inner_protection" not in existing_cont_cols:
            conn.execute("ALTER TABLE containers ADD COLUMN inner_protection REAL DEFAULT 0")
        if "effectiveness" not in existing_cont_cols:
            conn.execute("ALTER TABLE containers ADD COLUMN effectiveness REAL DEFAULT 0")
        if "bonus_stats" not in existing_cont_cols:
            conn.execute("ALTER TABLE containers ADD COLUMN bonus_stats TEXT DEFAULT '{}'")
        conn.commit()

    log.info("Database initialized at %s", get_db_path())


def upsert_artefact(info: dict):
    """Insert or update a single artefact record from wiki API data."""
    # --- FIX: context manager ensures conn.close() on any exception ---
    with get_connection() as conn:
        now = datetime.now(timezone.utc).isoformat()
        icon_url = info.get("icon_url", "")
        subcat = info.get("subcategory", "")

        conn.execute("""
            INSERT INTO artefacts (id, name_ru, name_en, category, subcategory, color,
                                   base_price, weight, stats, bonus_stats,
                                   description, icon_url, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name_ru=excluded.name_ru, name_en=excluded.name_en,
                category=excluded.category, subcategory=excluded.subcategory,
                color=excluded.color, base_price=excluded.base_price,
                weight=excluded.weight, stats=excluded.stats,
                bonus_stats=excluded.bonus_stats,
                description=excluded.description, icon_url=excluded.icon_url,
                updated_at=excluded.updated_at
        """, (
            info["id"], info["name_ru"], info.get("name_en", ""),
            info.get("category", ""), subcat, info.get("color", ""),
            info.get("base_price", 0.0), info.get("weight", 0.0),
            json.dumps(info.get("base_stats", info.get("stats", [])), ensure_ascii=False),
            json.dumps(info.get("bonus_stats", []), ensure_ascii=False),
            info.get("description", ""), icon_url, now,
        ))
        conn.commit()


def upsert_container(info: dict):
    """Insert or update a single container/backpack record."""
    # --- FIX: context manager ensures conn.close() on any exception ---
    with get_connection() as conn:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            INSERT INTO containers (id, name_ru, name_en, category, item_type, color,
                                   base_price, weight, slots, inner_protection,
                                   effectiveness, bonus_stats, icon_url, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name_ru=excluded.name_ru, name_en=excluded.name_en,
                category=excluded.category, item_type=excluded.item_type,
                color=excluded.color, base_price=excluded.base_price,
                weight=excluded.weight, slots=excluded.slots,
                inner_protection=excluded.inner_protection,
                effectiveness=excluded.effectiveness,
                bonus_stats=excluded.bonus_stats,
                icon_url=excluded.icon_url, updated_at=excluded.updated_at
        """, (
            info["id"], info["name_ru"], info.get("name_en", ""),
            info.get("category", ""), info.get("item_type", ""),
            info.get("color", ""), info.get("base_price", 0.0),
            info.get("weight", 0.0), info.get("slots", 0),
            info.get("inner_protection", 0.0), info.get("effectiveness", 0.0),
            json.dumps(info.get("bonus_stats", {}), ensure_ascii=False),
            info.get("icon_url", ""), now,
        ))
        conn.commit()


def insert_history_record(item_id: str, price_entry: dict, parsed: dict, region: str):
    """Insert a single auction history (sold lot) record."""
    # --- FIX: context manager ensures conn.close() on any exception ---
    with get_connection() as conn:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            INSERT INTO auction_history (item_id, price, amount, quality, quality_name,
                art_percent, stats_random, ndmg, bonus_properties, ptn, spawn_time,
                sold_time, additional_raw, region, collected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item_id, price_entry.get("price", 0), price_entry.get("amount", 1),
            parsed.get("quality", 0), parsed.get("quality_name", ""),
            parsed.get("art_percent", 0.0), parsed.get("stats_random", 0.0),
            parsed.get("ndmg", 0.0), json.dumps(parsed.get("bonus_properties", []), ensure_ascii=False),
            parsed.get("ptn", 0), parsed.get("spawn_time", 0), price_entry.get("time", ""),
            json.dumps(price_entry.get("additional", {}), ensure_ascii=False), region, now,
        ))
        conn.commit()


def merge_history_for_item(item_id: str, prices: list[dict], region: str,
                           lease=None, keep: int = 500) -> int:
    """Merge sold-lot history for one item, skipping records already stored.

    The eAPI history endpoint returns a rolling window. Re-fetching it must not
    inflate price averages, so a sold lot is identified by its full immutable
    content (time, price, amount, quality, ptn, art_percent, raw additional).
    Only the newest ``keep`` rows per item/region are retained.
    """
    from .api_client import parse_lot_additional

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for entry in prices:
        parsed = parse_lot_additional(entry.get("additional"))
        additional_raw = json.dumps(entry.get("additional", {}), ensure_ascii=False)
        key = (entry.get("time", ""), entry.get("price", 0), entry.get("amount", 1),
               parsed.get("quality", 0), parsed.get("ptn", 0),
               parsed.get("art_percent", 0.0), additional_raw)
        rows.append((key, (
            item_id, entry.get("price", 0), entry.get("amount", 1),
            parsed.get("quality", 0), parsed.get("quality_name", ""),
            parsed.get("art_percent", 0.0), parsed.get("stats_random", 0.0),
            parsed.get("ndmg", 0.0),
            json.dumps(parsed.get("bonus_properties", []), ensure_ascii=False),
            parsed.get("ptn", 0), parsed.get("spawn_time", 0), entry.get("time", ""),
            additional_raw, region, now,
        )))

    guard = lease.guard() if lease is not None else nullcontext()
    with guard, closing(get_connection()) as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = {tuple(r) for r in conn.execute(
                "SELECT sold_time, price, amount, quality, ptn, art_percent, additional_raw "
                "FROM auction_history WHERE item_id=? AND region=?",
                (item_id, region))}
            fresh = [values for key, values in rows if key not in existing]
            if fresh:
                conn.executemany("""
                    INSERT INTO auction_history (item_id, price, amount, quality, quality_name,
                        art_percent, stats_random, ndmg, bonus_properties, ptn, spawn_time,
                        sold_time, additional_raw, region, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, fresh)
            conn.execute("""
                DELETE FROM auction_history WHERE item_id=? AND region=? AND id NOT IN (
                    SELECT id FROM auction_history WHERE item_id=? AND region=?
                    ORDER BY sold_time DESC, id DESC LIMIT ?
                )
            """, (item_id, region, item_id, region, max(1, int(keep))))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return len(fresh)


def clear_lots_for_item(item_id: str, region: str):
    """Delete existing active lots for an item before re-inserting (lots are ephemeral)."""
    # --- FIX: context manager ensures conn.close() on any exception ---
    with get_connection() as conn:
        conn.execute("DELETE FROM auction_lots WHERE item_id=? AND region=?", (item_id, region))
        conn.commit()


def insert_lot_record(item_id: str, lot: dict, parsed: dict, region: str):
    """Insert a single active auction lot record."""
    # --- FIX: context manager ensures conn.close() on any exception ---
    with get_connection() as conn:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            INSERT INTO auction_lots (item_id, start_price, buyout_price, amount, quality,
                quality_name, art_percent, stats_random, ndmg, bonus_properties,
                upgrade_bonus, ptn, start_time, end_time, additional_raw, region, collected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item_id, lot.get("startPrice", 0), lot.get("buyoutPrice", 0), lot.get("amount", 1),
            parsed.get("quality", 0), parsed.get("quality_name", ""),
            parsed.get("art_percent", 0.0), parsed.get("stats_random", 0.0),
            parsed.get("ndmg", 0.0), json.dumps(parsed.get("bonus_properties", []), ensure_ascii=False),
            parsed.get("upgrade_bonus", 0.0), parsed.get("ptn", 0),
            lot.get("startTime", ""), lot.get("endTime", ""),
            json.dumps(lot.get("additional", {}), ensure_ascii=False), region, now,
        ))
        conn.commit()


def replace_lots_for_item(item_id, lots, region, lease):
    """Publish a complete item under live refresh ownership, or change nothing."""
    from .api_client import parse_lot_additional
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for lot in lots:
        parsed = parse_lot_additional(lot.get('additional'))
        rows.append((item_id, lot.get('startPrice', 0), lot.get('buyoutPrice', 0), lot.get('amount', 1),
                     parsed.get('quality', 0), parsed.get('quality_name', ''),
                     parsed.get('art_percent', 0.0), parsed.get('stats_random', 0.0),
                     parsed.get('ndmg', 0.0), json.dumps(parsed.get('bonus_properties', []), ensure_ascii=False),
                     parsed.get('upgrade_bonus', 0.0), parsed.get('ptn', 0),
                     lot.get('startTime', ''), lot.get('endTime', ''),
                     json.dumps(lot.get('additional', {}), ensure_ascii=False), region, now))
    # Ownership cannot change while this market transaction commits. The market
    # replacement itself is atomic even if a worker dies between state updates.
    with lease.guard(), closing(get_connection()) as conn:
        try:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('DELETE FROM auction_lots WHERE item_id=? AND region=?', (item_id, region))
            conn.executemany('''INSERT INTO auction_lots
                (item_id,start_price,buyout_price,amount,quality,quality_name,art_percent,
                 stats_random,ndmg,bonus_properties,upgrade_bonus,ptn,start_time,end_time,
                 additional_raw,region,collected_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', rows)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise


def get_artefact_stats_summary() -> dict:
    """Return summary statistics of the database."""
    # --- FIX: context manager ensures conn.close() on any exception ---
    with get_connection() as conn:
        stats = {}
        stats["artefacts"] = conn.execute("SELECT COUNT(*) as c FROM artefacts").fetchone()["c"]
        stats["history_records"] = conn.execute("SELECT COUNT(*) as c FROM auction_history").fetchone()["c"]
        stats["active_lots"] = conn.execute("SELECT COUNT(*) as c FROM auction_lots").fetchone()["c"]
        stats["items_with_history"] = conn.execute(
            "SELECT COUNT(DISTINCT item_id) as c FROM auction_history"
        ).fetchone()["c"]
        stats["items_with_lots"] = conn.execute(
            "SELECT COUNT(DISTINCT item_id) as c FROM auction_lots"
        ).fetchone()["c"]
        stats["containers"] = conn.execute("SELECT COUNT(*) as c FROM containers").fetchone()["c"]
        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    init_db()
    print("DB stats:", get_artefact_stats_summary())

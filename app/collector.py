"""
Full data collection pipeline.

1. Fetch all artefacts from stalzone.wiki API (base + bonus stats) → SQLite
2. Fetch all containers/backpacks from GitHub DB → SQLite
3. For each artefact + container, fetch active lots from Stalzone eAPI → SQLite

Usage:
    python -m app.collector              # full run (artefacts + containers + auction)
    python -m app.collector --artefacts  # only fetch item database from wiki/GitHub
    python -m app.collector --auction    # only fetch auction data (for existing items)
"""
import argparse
import logging
import sys
import time

from .api_client import StalzoneAPI
from .config import Config
from .database import (
    get_artefact_stats_summary, init_db, merge_history_for_item,
    upsert_artefact, upsert_container,
)
from ._app_state import application_store, auction_job, LeaseLost, RefreshAlreadyRunning
from .github_fetcher import fetch_all_containers
from .wiki_fetcher import fetch_all_artefacts

log = logging.getLogger(__name__)


def collect_artefacts():
    """Fetch all artefacts from stalzone.wiki API and store in SQLite."""
    log.info("=== Phase 1: Fetching artefact database from stalzone.wiki ===")
    artefacts = fetch_all_artefacts()
    log.info("Storing %d artefacts in database...", len(artefacts))
    for art in artefacts:
        upsert_artefact(art)
    log.info("Artefact database populated: %d items", len(artefacts))
    return artefacts

def collect_containers():
    """Fetch all backpacks and containers from GitHub database and store in SQLite."""
    log.info("=== Phase 1b: Fetching containers/backpacks from GitHub ===")
    containers = fetch_all_containers()
    log.info("Storing %d containers/backpacks in database...", len(containers))
    for c in containers:
        upsert_container(c)
    log.info("Container database populated: %d items", len(containers))
    return containers


def collect_auction_data(item_ids: list[str] | None = None, max_lots: int = 200,
                         item_source: str = "all", lease=None,
                         include_history: bool = True, max_history: int = 200):
    """One coordinated refresh, whether invoked by API or CLI."""
    lease = lease or application_store(Config.DATA_DIR).claim(auction_job(Config.REGION))
    if lease is None:
        raise RefreshAlreadyRunning('Auction refresh already running')
    error = None
    try:
        with lease.keep_alive() as lost:
            result = _collect_auction_data(item_ids, max_lots, item_source, lease, lost,
                                           include_history, max_history)
            if lost.is_set():
                raise LeaseLost('Auction refresh heartbeat lost')
            return result
    except BaseException as failure:
        error = str(failure) or type(failure).__name__
        raise
    finally:
        if not lease.finish(error) and error is None:
            raise LeaseLost('Auction refresh completion ownership lost')


def _collect_auction_data(item_ids, max_lots, item_source, lease, lost,
                          include_history=True, max_history=200):
    """Fetch active auction lots (and optionally sold history) for all items.

    Args:
        item_ids: list of item IDs; None = load from DB
        max_lots: max active lots per item
        item_source: "all" (artefacts+containers), "artefacts", "containers"
        include_history: also merge the rolling sold-lot window
        max_history: max sold-lot records per item
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .database import get_connection
    from .database import replace_lots_for_item

    log.info("=== Phase 2: Fetching active lots from Stalzone eAPI (concurrent) ===")

    if item_ids is None:
        conn = get_connection()
        if item_source == "artefacts":
            rows = conn.execute("SELECT id FROM artefacts WHERE length(id) = 4 ORDER BY id").fetchall()
        elif item_source == "containers":
            rows = conn.execute("SELECT id FROM containers WHERE length(id) = 4 ORDER BY id").fetchall()
        else:
            rows = conn.execute("SELECT id FROM artefacts WHERE length(id) = 4 UNION SELECT id FROM containers WHERE length(id) = 4 ORDER BY id").fetchall()
        conn.close()
        item_ids = [r["id"] for r in rows]

    region = Config.REGION


    total_lots = 0
    items_with_data = 0
    total_history = 0
    items_with_history = 0
    history_failed = [0]
    _done = [0]
    failed = [0]
    lease.progress(0, len(item_ids))

    def _fetch_one(item_id: str, client_id: str = "", client_secret: str = ""):
        """Fetch lots and (optionally) sold history for one item."""
        try:
            if lost.is_set():
                raise LeaseLost('Auction refresh heartbeat lost')
            if client_id and client_secret:
                api = StalzoneAPI(client_id=client_id, client_secret=client_secret)
            else:
                api = StalzoneAPI()
            lots = api.fetch_all_lots(item_id, max_records=max_lots)
            history = None
            if include_history:
                try:
                    history = api.fetch_all_history(item_id, max_records=max_history)
                except Exception as e:
                    log.warning("History fetch failed for %s: %s", item_id, e)
                    history = None
            return item_id, lots, history
        except Exception as e:
            log.warning("Lots fetch failed for %s: %s", item_id, e)
            return item_id, None, None

    def _store_result(item_id: str, lots: list[dict] | None,
                      history: list[dict] | None):
        """Store fetched lots/history in DB (single writer thread)."""
        nonlocal total_lots, items_with_data, total_history, items_with_history
        if lost.is_set():
            raise LeaseLost('Auction refresh heartbeat lost')
        if lots is None:
            failed[0] += 1
            _done[0] += 1
            lease.progress(_done[0], len(item_ids))
            return
        replace_lots_for_item(item_id, lots, region, lease)
        if lots:
            total_lots += len(lots)
            items_with_data += 1
        if history is not None:
            try:
                inserted = merge_history_for_item(item_id, history, region, lease)
                total_history += inserted
                if history:
                    items_with_history += 1
            except LeaseLost:
                raise
            except Exception as e:
                # History is additive analytics: never fail the lot refresh for it.
                history_failed[0] += 1
                log.warning("History store failed for %s: %s", item_id, e)
        _done[0] += 1
        lease.progress(_done[0], len(item_ids))
        if _done[0] % 20 == 0 or _done[0] == len(item_ids):
            log.info("[%d/%d] Processed %s (%d lots, %d new history)",
                     _done[0], len(item_ids), item_id, len(lots or []),
                     total_history)

    # Multi-key parallel: distribute items across available API keys
    api_keys = getattr(Config, 'API_KEYS', [(Config.CLIENT_ID, Config.CLIENT_SECRET)])
    n_keys = len(api_keys)
    if n_keys > 1:
        # Split items round-robin across keys
        groups = [item_ids[i::n_keys] for i in range(n_keys)]
        log.info("Region: %s | Items: %d | Max lots/item: %d | Multi-key: %d keys (%d items/key)",
                 region, len(item_ids), max_lots, n_keys, len(item_ids) // n_keys)
        with ThreadPoolExecutor(max_workers=n_keys) as executor:
            futures = []
            for i, (cid, csec) in enumerate(api_keys):
                for item_id in groups[i]:
                    futures.append(executor.submit(_fetch_one, item_id, cid, csec))
            for future in as_completed(futures):
                item_id, lots, history = future.result()
                _store_result(item_id, lots, history)
    else:
        log.info("Region: %s | Items: %d | Max lots/item: %d | Workers: 8",
                 region, len(item_ids), max_lots)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_fetch_one, iid): iid for iid in item_ids}
            for future in as_completed(futures):
                item_id, lots, history = future.result()
                _store_result(item_id, lots, history)

    log.info("Auction collection done: %d active lots (%d items had data); "
             "%d new history records (%d items, %d failed)",
             total_lots, items_with_data, total_history, items_with_history,
             history_failed[0])
    if failed[0]:
        raise RuntimeError(f'Auction fetch failed for {failed[0]} items')
    return total_lots, items_with_data


def main():
    parser = argparse.ArgumentParser(description="Stalzone artifact data collector")
    parser.add_argument("--artefacts", action="store_true", help="Only fetch item database (artefacts+containers) from GitHub")
    parser.add_argument("--auction", action="store_true", help="Only fetch auction data from eAPI")
    parser.add_argument("--containers-only", action="store_true", help="Only fetch auction data for containers/backpacks")
    parser.add_argument("--max-lots", type=int, default=100, help="Max active lots per item")
    parser.add_argument("--max-history", type=int, default=200, help="Max sold-lot records per item")
    parser.add_argument("--no-history", action="store_true", help="Skip sold-lot history collection")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    init_db()

    run_items = not args.auction or args.artefacts
    run_auction = not args.artefacts or args.auction

    if run_items:
        collect_artefacts()
        collect_containers()

    if run_auction:
        source = "containers" if args.containers_only else "all"
        collect_auction_data(max_lots=args.max_lots, item_source=source,
                             include_history=not args.no_history,
                             max_history=args.max_history)

    stats = get_artefact_stats_summary()
    log.info("=== Final database stats ===")
    for k, v in stats.items():
        log.info("  %s: %s", k, v)

if __name__ == "__main__":
    main()

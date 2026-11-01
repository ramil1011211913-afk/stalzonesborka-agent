"""
Stalzone eAPI client — auction data via Client-Id/Client-Secret header auth.

Uses header-based auth (no exbo.net OAuth needed — that host is firewalled).
Endpoints (region-scoped, e.g. /EU/...):
  GET /regions                         — list of regions
  GET /{region}/auction/{item}/history — sold lot history
  GET /{region}/auction/{item}/lots    — active auction lots
"""
import logging
import time

import requests

from .config import Config

log = logging.getLogger(__name__)


class AuctionAPIError(RuntimeError):
    """The auction endpoint did not confirm a complete, usable response."""


# Quality mapping from eapi-wiki art_stats.md
QUALITY_NAMES = {
    0: "Common",
    1: "Uncommon",
    2: "Special",
    3: "Rare",
    4: "Exceptional",
    5: "Legendary",
}

# Quality ranges — each step is 15% wide, starting from 85% for Common
QUALITY_RANGES = {
    0: {"min": 85, "range": 15},   # Common: 85-100%
    1: {"min": 100, "range": 15},  # Uncommon: 100-115%
    2: {"min": 115, "range": 15},  # Special: 115-130%
    3: {"min": 130, "range": 15},  # Rare: 130-145%
    4: {"min": 145, "range": 15},  # Exceptional: 145-160%
    5: {"min": 160, "range": 15},  # Legendary: 160-175%
}


def calc_art_percent(qlt: int, stats_random: float | None) -> float:
    """Calculate artifact quality percentage from qlt + stats_random.

    stats_random is in [-2, 2], normalized to [0, 1].
    artPercent = minValue + range * normalized

    If stats_random is None (not returned by API), returns MAX of the range
    (matching lunar-zone calculator default). If stats_random is 0, returns midpoint.
    """
    meta = QUALITY_RANGES.get(qlt, QUALITY_RANGES[0])
    if stats_random is None:
        # API didn't return stats_random — use MAX (lunar-zone default)
        return meta["min"] + meta["range"]
    if stats_random == 0:
        # stats_random=0 means average — use midpoint
        return meta["min"] + meta["range"] / 2
    normalized = (stats_random + 2) / 4
    return meta["min"] + meta["range"] * normalized


def calc_ptn_bonus(ptn: int) -> float:
    """Calculate research points bonus multiplier.

    ptn ranges 0-15. Each point gives +2% bonus to stat values.
    ptn=15 → +30% bonus → multiplier = 1.30

    Example: Морозец Common base=-0.3840 temperature
      ptn=0:  -0.3840 × 1.00 = -0.3840
      ptn=15: -0.3840 × 1.30 = -0.4992
    """
    return 1.0 + (ptn or 0) * 0.02


class StalzoneAPI:
    """Client for Stalzone eAPI using Client-Id/Client-Secret header auth."""

    def __init__(self, region: str | None = None,
                 client_id: str | None = None,
                 client_secret: str | None = None):
        self.region = (region or Config.REGION).upper()
        self.base_url = Config.API_BASE
        self._session = requests.Session()
        headers = {
            "Client-Id": client_id or Config.CLIENT_ID,
            "Client-Secret": client_secret or Config.CLIENT_SECRET,
        }
        self._session.headers.update(headers)

    def _get(self, path: str, params: dict | None = None, retries: int = 3) -> dict | list | None:
        """GET request with retry and rate-limit handling."""
        url = f"{self.base_url}{path}"
        for attempt in range(retries):
            try:
                r = self._session.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
                # Check rate limit headers — sleep if running low
                remaining = r.headers.get("x-ratelimit-remaining")
                if remaining is not None and int(remaining) < 20:
                    reset_ts = r.headers.get("x-ratelimit-reset")
                    if reset_ts:
                        import time as _time
                        wait = max(0, int(reset_ts) - int(_time.time()))
                        if wait > 0 and wait < 60:
                            log.info("Rate limit low: %s remaining, sleeping %ds", remaining, wait)
                            time.sleep(wait + 1)
                        else:
                            log.debug("Rate limit low: %s remaining (reset %s)", remaining, reset_ts)

                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    log.warning("Rate limited (429). Sleeping 30s...")
                    time.sleep(30)
                    continue
                if r.status_code == 404:
                    log.debug("404: %s", path)
                    return None
                log.error("API %s -> HTTP %d (response omitted for security)", path[:80], r.status_code)
                return None
            except requests.RequestException as e:
                log.warning("Request failed (attempt %d): %s", attempt + 1, e)
                time.sleep(2 * (attempt + 1))
        return None

    def get_regions(self) -> list[dict]:
        """List available game regions."""
        data = self._get("/regions")
        return data if isinstance(data, list) else []

    def get_auction_history(self, item_id: str, limit: int = 200, offset: int = 0,
                            additional: bool = True) -> dict:
        """Fetch sold lot history for an item.

        Returns: {"total": int, "prices": [{"amount", "price", "time", "additional": {...}}]}
        """
        path = f"/{self.region}/auction/{item_id}/history"
        params = {"limit": min(limit, 200), "offset": offset, "additional": str(additional).lower()}
        data = self._get(path, params)
        if not isinstance(data, dict):
            raise AuctionAPIError("Auction history request failed; empty history not confirmed")
        return data

    def get_auction_lots(self, item_id: str, limit: int = 200, offset: int = 0,
                         sort: str = "time_created", order: str = "asc",
                         additional: bool = True) -> dict:
        """Fetch active auction lots for an item.

        sort: time_created | time_left | current_price | buyout_price
        order: asc | desc
        Returns: {"total": int, "lots": [{...}]}
        """
        path = f"/{self.region}/auction/{item_id}/lots"
        params = {
            "limit": min(limit, 200), "offset": offset,
            "sort": sort, "order": order,
            "additional": str(additional).lower(),
        }
        data = self._get(path, params)
        if not isinstance(data, dict):
            raise AuctionAPIError("Auction lots request failed; empty market not confirmed")
        return data

    def fetch_all_history(self, item_id: str, max_records: int = 200) -> list[dict]:
        """Fetch full price history with pagination, up to max_records."""
        all_prices = []
        offset = 0
        while offset < max_records:
            data = self.get_auction_history(item_id, limit=200, offset=offset, additional=True)
            prices = data.get("prices", [])
            if not prices:
                break
            all_prices.extend(prices)
            total = data.get("total", 0)
            offset += len(prices)
            if offset >= total:
                break
        log.debug("Fetched %d history records for %s", len(all_prices), item_id)
        return all_prices[:max_records]

    def fetch_all_lots(self, item_id: str, max_records: int = 200) -> list[dict]:
        """Fetch all active lots with pagination.

        A partial page, a changing total, or a result larger than max_records
        rejects the item instead of publishing an incomplete market.
        """
        all_lots = []
        offset = 0
        expected_total = None
        while True:
            data = self.get_auction_lots(item_id, limit=200, offset=offset, additional=True)
            lots = data.get("lots")
            total = data.get("total")
            if (type(total) is not int or total < 0 or not isinstance(lots, list)
                    or any(not isinstance(lot, dict) for lot in lots)):
                raise AuctionAPIError(f"Invalid auction page for {item_id}")
            if expected_total is None:
                expected_total = total
                if expected_total > max_records:
                    raise AuctionAPIError(
                        f"Auction for {item_id} has {expected_total} lots, "
                        f"above max_records={max_records}")
            elif total != expected_total:
                raise AuctionAPIError(
                    f"Auction total changed for {item_id}: "
                    f"{expected_total} -> {total}")
            if not lots:
                if offset < expected_total:
                    raise AuctionAPIError(
                        f"Auction pagination ended early for {item_id}: "
                        f"{offset}/{expected_total}")
                break
            all_lots.extend(lots)
            offset += len(lots)
            if offset >= expected_total:
                break
        if expected_total is not None and offset != expected_total:
            raise AuctionAPIError(
                f"Auction pagination incomplete for {item_id}: "
                f"{offset}/{expected_total}")
        log.debug("Fetched %d lots for %s", len(all_lots), item_id)
        return all_lots

    def get_emission(self) -> dict | None:
        """Fetch last emission schedule. App auth (works with Client-Id/Secret).
        Returns: {"previousStart": ISO, "previousEnd": ISO} or None.
        """
        data = self._get(f"/{self.region}/emission")
        return data if isinstance(data, dict) else None


def parse_lot_additional(additional: dict | None) -> dict:
    """Parse the 'additional' field from auction lot/history into structured data.

    Fields:
    - qlt: quality (0-5)
    - stats_random: deviation from average (-2..2)
    - ndmg: normalized damage/quality value
    - bonus_properties: list of bonus stat names
    - upgrade_bonus: upgrade level bonus
    - it_transf_count: item transformation count
    - ptn: points
    - spawn_time: epoch ms
    """
    if not additional:
        return {}

    qlt = additional.get("qlt", 0)
    # stats_random is in [-2, 2] range, used for art_percent calculation.
    # md_k is a DIFFERENT field (not a replacement) — do NOT use it as stats_random.
    # When stats_random is absent, calc_art_percent returns MAX (matches lunar-zone default).
    stats_random: float | None = additional.get("stats_random")
    if stats_random is not None and not isinstance(stats_random, (int, float)):
        stats_random = None

    return {
        "quality": qlt,
        "quality_name": QUALITY_NAMES.get(qlt, "Unknown"),
        "art_percent": calc_art_percent(qlt, stats_random),
        "stats_random": stats_random if stats_random is not None else 0.0,
        "ndmg": additional.get("ndmg", 0.0),
        "md_k": additional.get("md_k", 0.0),
        "bonus_properties": additional.get("bonus_properties", []),
        "upgrade_bonus": additional.get("upgrade_bonus", 0.0),
        "it_transf_count": additional.get("it_transf_count", 0),
        "ptn": additional.get("ptn", 0),
        "spawn_time": additional.get("spawn_time", 0),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    api = StalzoneAPI()
    print("Regions:", api.get_regions())
    print()

    # Test with known artefact
    print("=== History for 04yr (Виток) ===")
    hist = api.get_auction_history("04yr", limit=3, additional=True)
    print(f"Total records: {hist.get('total', 0)}")
    for p in hist.get("prices", [])[:3]:
        parsed = parse_lot_additional(p.get("additional"))
        print(f"  price={p['price']:,}  qlt={parsed['quality_name']}  pct={parsed['art_percent']:.1f}%")

    print()
    print("=== Active lots for 04yr ===")
    lots = api.get_auction_lots("04yr", limit=3, additional=True)
    print(f"Total lots: {lots.get('total', 0)}")
    for lot in lots.get("lots", [])[:3]:
        parsed = parse_lot_additional(lot.get("additional"))
        print(f"  buyout={lot.get('buyoutPrice', 0):,}  qlt={parsed['quality_name']}  pct={parsed['art_percent']:.1f}%")

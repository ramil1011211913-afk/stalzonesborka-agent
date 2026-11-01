import pytest

from app import database
from app.api_client import QUALITY_RANGES, calc_art_percent, parse_lot_additional
from app.config import Config


@pytest.mark.parametrize('quality', sorted(QUALITY_RANGES))
def test_random_is_bounded_by_quality_tier(quality):
    tier = QUALITY_RANGES[quality]
    assert calc_art_percent(quality, -3) == tier['min']
    assert calc_art_percent(quality, 3) == tier['min'] + tier['range']
    assert calc_art_percent(quality, 0) == tier['min'] + tier['range'] / 2


def test_observed_lot_has_known_maximum_percentage():
    from app._quality_scenario import percent_known
    raw = {'qlt': 3, 'ptn': 15, 'stats_random': 2.472403}
    parsed = parse_lot_additional(raw)
    assert parsed['art_percent'] == 145
    assert parsed['stats_random'] == 2.472403
    assert percent_known(raw, parsed['stats_random'], 3, parsed['art_percent'])


def test_existing_only_legacy_derived_values_are_repaired(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, 'DATA_DIR', str(tmp_path))
    monkeypatch.setattr(Config, 'DB_PATH', str(tmp_path / 'market.db'))
    database.init_db()
    raw = {'qlt': 3, 'ptn': 15, 'stats_random': 2.472403}
    old = 130 + 15 * (raw['stats_random'] + 2) / 4
    parsed = dict(quality=3, ptn=15, art_percent=old, stats_random=raw['stats_random'])
    database.insert_lot_record('a', {'buyoutPrice': 10, 'additional': raw}, parsed, 'EU')
    database.insert_lot_record('b', {'buyoutPrice': 11, 'additional': raw}, dict(parsed, art_percent=141), 'EU')
    database.insert_history_record('a', {'price': 10, 'additional': raw}, parsed, 'EU')
    database.init_db()
    database.init_db()
    with database.get_connection() as conn:
        assert [r['art_percent'] for r in conn.execute('SELECT art_percent FROM auction_lots ORDER BY item_id')] == [145, 141]
        row = conn.execute('SELECT art_percent, stats_random FROM auction_history').fetchone()
        assert (row['art_percent'], row['stats_random']) == (145, 2.472403)

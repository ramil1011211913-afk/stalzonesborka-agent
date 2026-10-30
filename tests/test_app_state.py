from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3
import time

import pytest

from app._app_state import StateStore, LeaseLost, auction_job


def test_legacy_import_once_preserves_original_and_cleared_state(tmp_path):
    legacy = tmp_path / 'last_owned.json'
    legacy.write_text('[{"instance_id":"old"}]', encoding='utf-8')
    digest = hashlib.sha256(legacy.read_bytes()).hexdigest()
    store = StateStore(tmp_path / 'state.db')
    assert store.import_legacy(legacy)
    assert store.owned()['artifacts'] == [{'instance_id': 'old'}]
    store.save_owned([])
    assert not store.import_legacy(legacy)
    assert store.owned()['artifacts'] == [] and store.owned()['revision'] == 2
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == digest


def test_owned_revisions_are_transactional_and_namespaces_independent(tmp_path):
    store = StateStore(tmp_path / 'state.db')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: store.save_owned([{'instance_id': str(i)}]), range(32)))
    assert store.owned()['revision'] == 32
    assert len(store.owned()['artifacts']) == 1
    store.save_owned([{'instance_id': 'other'}], owner='future-authenticated-user')
    assert store.owned()['artifacts'] != store.owned('future-authenticated-user')['artifacts']
    with pytest.raises(ValueError):
        store.save_owned([float('nan')])
    assert store.owned()['revision'] == 32


def test_concurrent_claims_have_one_owner_and_shared_progress(tmp_path):
    stores = [StateStore(tmp_path / 'state.db') for _ in range(12)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        claims = list(pool.map(lambda s: s.claim('auction:EU'), stores))
    leases = [lease for lease in claims if lease]
    assert len(leases) == 1
    leases[0].progress(3, 9)
    assert all(s.status('auction:EU')['done'] == 3 for s in stores)
    assert leases[0].finish()
    assert all(not s.status('auction:EU')['running'] for s in stores)
    assert stores[0].claim('auction:EU').generation == 2


def test_expired_owner_is_fenced_from_progress_finish_renew_and_market_writes(tmp_path, monkeypatch):
    from app import database
    from app.config import Config
    monkeypatch.setattr(Config, 'DATA_DIR', str(tmp_path))
    monkeypatch.setattr(Config, 'DB_PATH', str(tmp_path / 'market.db'))
    database.init_db()
    now = [100.0]
    store = StateStore(tmp_path / 'state.db', clock=lambda: now[0])
    old = store.claim(auction_job('EU'), seconds=2)
    database.replace_lots_for_item('item', [{'buyoutPrice': 100}], 'EU', old)
    now[0] = 103
    assert store.status('auction:EU')['state'] == 'expired'
    new = store.claim('auction:EU', seconds=2)
    new.progress(1, 2)
    assert not old.heartbeat() and not old.finish('stale error')
    with pytest.raises(LeaseLost):
        old.progress(2, 2)
    with pytest.raises(LeaseLost):
        database.replace_lots_for_item('item', [{'buyoutPrice': 999}], 'EU', old)
    with sqlite3.connect(Config.DB_PATH) as conn:
        assert conn.execute('SELECT buyout_price FROM auction_lots').fetchone()[0] == 100
    assert store.status('auction:EU')['done'] == 1 and store.status('auction:EU')['error'] is None


def test_market_replacement_rolls_back_whole_item_on_bad_row(tmp_path, monkeypatch):
    from app import database
    from app.config import Config
    monkeypatch.setattr(Config, 'DATA_DIR', str(tmp_path))
    monkeypatch.setattr(Config, 'DB_PATH', str(tmp_path / 'market.db'))
    database.init_db()
    lease = StateStore(tmp_path / 'state.db').claim('auction:EU')
    database.replace_lots_for_item('item', [{'buyoutPrice': 10}], 'EU', lease)
    with pytest.raises(sqlite3.IntegrityError):
        database.replace_lots_for_item('item', [{'buyoutPrice': 20}, {'amount': None}], 'EU', lease)
    with sqlite3.connect(Config.DB_PATH) as conn:
        assert conn.execute('SELECT buyout_price FROM auction_lots').fetchall() == [(10,)]


def test_heartbeat_keeps_lease_past_original_deadline(tmp_path):
    store = StateStore(tmp_path / 'state.db')
    lease = store.claim('auction:EU', seconds=.3)
    with lease.keep_alive() as lost:
        time.sleep(.65)
        assert not lost.is_set() and store.claim('auction:EU') is None
    assert lease.finish('controlled failure')
    assert store.status('auction:EU')['state'] == 'failed'


def test_collector_uses_fenced_atomic_publish_and_shared_terminal_state(tmp_path, monkeypatch):
    from app import collector, database
    from app.config import Config
    monkeypatch.setattr(Config, 'DATA_DIR', str(tmp_path))
    monkeypatch.setattr(Config, 'DB_PATH', str(tmp_path / 'market.db'))
    database.init_db()
    monkeypatch.setattr(collector.StalzoneAPI, 'fetch_all_lots',
                        lambda self, item_id, **kwargs: [{'buyoutPrice': 42}])
    monkeypatch.setattr(collector.StalzoneAPI, 'fetch_all_history',
                        lambda self, item_id, **kwargs: [])
    assert collector.collect_auction_data(item_ids=['a', 'b']) == (2, 2)
    store = StateStore(tmp_path / 'app_state.sqlite3')
    status = store.status(auction_job(Config.REGION))
    assert status['done'] == status['total'] == 2 and status['state'] == 'completed'
    def failed(*args, **kwargs):
        raise RuntimeError('network failed')
    monkeypatch.setattr(collector.StalzoneAPI, 'fetch_all_lots', failed)
    with pytest.raises(RuntimeError, match='failed for 1 items'):
        collector.collect_auction_data(item_ids=['a'])
    assert store.status(auction_job(Config.REGION))['state'] == 'failed'
    recovery = store.claim(auction_job(Config.REGION))
    assert recovery and store.status(auction_job(Config.REGION))['error'] is None
    recovery.finish()


def test_collector_api_failure_preserves_existing_lots(tmp_path, monkeypatch):
    from app import collector, database
    from app.api_client import AuctionAPIError
    from app.config import Config
    monkeypatch.setattr(Config, 'DATA_DIR', str(tmp_path))
    monkeypatch.setattr(Config, 'DB_PATH', str(tmp_path / 'market.db'))
    database.init_db()
    store = StateStore(tmp_path / 'app_state.sqlite3')
    lease = store.claim(auction_job(Config.REGION))
    database.replace_lots_for_item('a', [{'buyoutPrice': 42}], Config.REGION, lease)
    lease.finish()

    def failed(*args, **kwargs):
        raise AuctionAPIError('network failed')
    monkeypatch.setattr(collector.StalzoneAPI, 'fetch_all_lots', failed)
    with pytest.raises(RuntimeError, match='failed for 1 items'):
        collector.collect_auction_data(item_ids=['a'], include_history=False)
    with sqlite3.connect(Config.DB_PATH) as conn:
        assert conn.execute(
            'SELECT buyout_price FROM auction_lots WHERE item_id=?', ('a',)
        ).fetchall() == [(42,)]


@pytest.mark.parametrize('pages,max_records,message', [
    ([{'total': 3, 'lots': [{'id': 1}]}, {'total': 3, 'lots': []}], 10, 'ended early'),
    ([{'total': 2, 'lots': [{'id': 1}]},
      {'total': 3, 'lots': [{'id': 2}, {'id': 3}]}], 10, 'changed'),
    ([{'total': 5, 'lots': [{'id': 1}]}], 2, 'above max_records'),
])
def test_fetch_all_lots_rejects_incomplete_pagination(monkeypatch, pages, max_records, message):
    from app.api_client import AuctionAPIError, StalzoneAPI
    api = StalzoneAPI.__new__(StalzoneAPI)
    monkeypatch.setattr(api, 'get_auction_lots', lambda *args, **kwargs: pages.pop(0))
    with pytest.raises(AuctionAPIError, match=message):
        api.fetch_all_lots('x', max_records=max_records)


def test_fetch_all_lots_accepts_complete_pagination(monkeypatch):
    from app.api_client import StalzoneAPI
    api = StalzoneAPI.__new__(StalzoneAPI)
    pages = [{'total': 2, 'lots': [{'id': 1}]},
             {'total': 2, 'lots': [{'id': 2}]}]
    monkeypatch.setattr(api, 'get_auction_lots', lambda *args, **kwargs: pages.pop(0))
    assert api.fetch_all_lots('x', max_records=2) == [{'id': 1}, {'id': 2}]


def test_history_merge_deduplicates_and_keeps_window(tmp_path, monkeypatch):
    from app import database
    from app.config import Config
    monkeypatch.setattr(Config, 'DATA_DIR', str(tmp_path))
    monkeypatch.setattr(Config, 'DB_PATH', str(tmp_path / 'market.db'))
    database.init_db()

    def sale(price, minute):
        return {'price': price, 'amount': 1,
                'time': f'2026-10-29T10:{minute:02d}:00+00:00',
                'additional': {'qlt': 1, 'ptn': 0, 'stats_random': 0}}

    first = [sale(100, 1), sale(200, 2)]
    assert database.merge_history_for_item('a', first, 'EU', keep=3) == 2
    assert database.merge_history_for_item('a', first, 'EU', keep=3) == 0
    assert database.merge_history_for_item('a', [sale(300, 3), sale(400, 4)],
                                           'EU', keep=3) == 2
    with database.get_connection() as conn:
        rows = conn.execute(
            "SELECT price FROM auction_history WHERE item_id='a' ORDER BY price"
        ).fetchall()
    assert [r['price'] for r in rows] == [200, 300, 400]


def test_owned_api_import_and_explicit_empty_save(tmp_path, monkeypatch):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    import importlib
    from app import database
    monkeypatch.setattr(database, 'init_db', lambda: None)
    main = importlib.import_module('app.main')
    monkeypatch.setattr(main.Config, 'DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setattr(main, '__file__', str(tmp_path / 'app/main.py'))
    monkeypatch.setattr(main, 'check_auction_freshness', lambda *args: {
        'is_fresh': True, 'last_collected': None, 'age_hours': 0})
    monkeypatch.setattr(main, 'build_artifact_sets', lambda **kwargs: [])
    (tmp_path / 'data').mkdir()
    legacy = tmp_path / 'data/last_owned.json'
    legacy.write_text('[{"instance_id":"legacy"}]', encoding='utf-8')
    digest = hashlib.sha256(legacy.read_bytes()).hexdigest()
    client = TestClient(main.app)
    assert client.get('/api/owned-artifacts').json()['artifacts'] == [{'instance_id': 'legacy'}]
    assert client.get('/api/builds', params={'owned': '[]'}).status_code == 200
    assert client.get('/api/owned-artifacts').json()['artifacts'] == []
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == digest


def test_thread_start_failure_releases_owner_for_retry(tmp_path, monkeypatch):
    pytest.importorskip('fastapi')
    import importlib
    from app import database
    monkeypatch.setattr(database, 'init_db', lambda: None)
    main = importlib.import_module('app.main')
    monkeypatch.setattr(main.Config, 'DATA_DIR', str(tmp_path))
    class BrokenThread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            raise RuntimeError('thread allocation failed')
    monkeypatch.setattr(main.threading, 'Thread', BrokenThread)
    with pytest.raises(RuntimeError, match='thread allocation failed'):
        main._start_auction_refresh()
    store = StateStore(tmp_path / 'app_state.sqlite3')
    assert store.status(auction_job(main.Config.REGION))['state'] == 'failed'
    assert store.claim(auction_job(main.Config.REGION)) is not None

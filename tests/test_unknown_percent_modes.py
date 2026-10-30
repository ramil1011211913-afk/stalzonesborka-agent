"""Three-point handling of unknown quality percentages.

Only lots whose physical percentage is actually unknown may be recalculated at
the selected tier point. Known auction rolls stay actual in every mode.
"""
from contextlib import contextmanager
import copy
import dataclasses
import json
import sqlite3

import pytest

from app import builder as b, _max_quality as quality, _quality_scenario as scenario
from tests.test_max_quality import call
from tests.test_objective_integration import container
from tests.test_stock_only import api

SPEED = 'speed_modifier'
BONUSES = ['SPEED_MOD', 'STAMINA_REGENERATION', 'REGENERATION_BONUS']
UNKNOWN_LOT = 107188
KNOWN_LOT = 107183
UNKNOWN_PRICE = 1_988_888
KNOWN_PRICE = 2_075_000
KNOWN_PERCENT = 117.019983625
KNOWN_SPEED = 4.56377936175


def helium_market(monkeypatch, tmp_path, *, unknown=True, known=True):
    """Synthetic q2/+15 Helium pair with the real price and roll asymmetry."""
    path = tmp_path / 'helium.db'
    path.unlink(missing_ok=True)
    base = [dict(prop_id=SPEED, min=3.0, max=3.0, is_positive=True)]
    bonus = [dict(prop_id='speed_modifier', min=0.0, max=0.0, is_positive=True),
             dict(prop_id='stamina_regeneration', min=0.0, max=0.0, is_positive=True),
             dict(prop_id='regeneration_bonus', min=0.0, max=0.0, is_positive=True)]
    with sqlite3.connect(path) as c:
        c.executescript('''
        CREATE TABLE artefacts(id TEXT PRIMARY KEY,name_ru TEXT,name_en TEXT,subcategory TEXT,
          stats TEXT,bonus_stats TEXT,icon_url TEXT);
        CREATE TABLE auction_history(item_id TEXT,quality INTEGER,price INTEGER,region TEXT);
        CREATE TABLE auction_lots(id INTEGER PRIMARY KEY,item_id TEXT,region TEXT,
          buyout_price INTEGER,quality INTEGER,ptn INTEGER,art_percent REAL,stats_random REAL,
          bonus_properties TEXT,end_time TEXT,additional_raw TEXT,amount INTEGER);
        ''')
        c.execute('INSERT INTO artefacts VALUES(?,?,?,?,?,?,?)',
                  ('kqr0', 'Гелий', 'Helium', '', json.dumps(base), json.dumps(bonus), ''))
        rows = []
        if unknown:
            rows.append((UNKNOWN_LOT, UNKNOWN_PRICE, 130.0, 0.0,
                         json.dumps({'qlt': 2, 'ptn': 15})))
        if known:
            rows.append((KNOWN_LOT, KNOWN_PRICE, KNOWN_PERCENT, -1.4613377,
                         json.dumps({'qlt': 2, 'ptn': 15, 'stats_random': -1.4613377})))
        for lot_id, price, percent, random, raw in rows:
            c.execute('INSERT INTO auction_lots VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                      (lot_id, 'kqr0', 'EU', price, 2, 15, percent, random,
                       json.dumps(BONUSES), '2099-01-01T00:00:00Z', raw, 1))
        c.commit()
    c.close()

    @contextmanager
    def connect():
        c = sqlite3.connect(path)
        c.row_factory = sqlite3.Row
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(b, 'get_connection', connect)
    monkeypatch.setattr(b, 'load_container_options', lambda region: [container(slots=1)])
    return path


def loaded(monkeypatch, tmp_path, **kwargs):
    helium_market(monkeypatch, tmp_path)
    return {scenario.source(o).auction_lot_id: o for o in
            b.load_artifact_options(priorities={SPEED: 10}, region='EU', **kwargs)}


def test_mode_resolution_is_explicit_and_backward_compatible():
    assert scenario.resolve_mode() == 'maximum'
    assert scenario.resolve_mode(conservative=True) == 'minimum'
    assert scenario.resolve_mode(False, 'minimum') == 'minimum'
    assert scenario.resolve_mode(False, 'midpoint') == 'midpoint'
    assert scenario.resolve_mode(False, 'maximum') == 'maximum'
    # An explicit request wins over the legacy alias.
    assert scenario.resolve_mode(True, 'midpoint') == 'midpoint'
    assert scenario.resolve_mode(True, 'maximum') == 'maximum'
    with pytest.raises(ValueError):
        scenario.resolve_mode(False, 'average')


@pytest.mark.parametrize('mode,scope,expected', [
    ('minimum', 'quality_minimum_for_unknown_candidates', 115.0),
    ('midpoint', 'quality_midpoint_for_unknown_candidates', 122.5),
    ('maximum', 'quality_maximum_for_unknown_candidates', 130.0),
])
def test_metadata_names_each_unknown_percent_mode(mode, scope, expected):
    meta = scenario.metadata(mode)
    assert meta['unknown_percent'] == mode
    assert meta['percentage_scope'] == scope
    assert meta['conservative'] == (mode == 'minimum')
    assert meta['known_percent_preserved'] is True
    assert meta['robust'] is False
    assert scenario.percent_range(2) == dict(min=115, midpoint=122.5, maximum=130)
    assert scenario.percent_range(2)[{'minimum': 'min'}.get(mode, mode)] == expected


@pytest.mark.parametrize('mode,expected', [
    ('minimum', 115.0), ('midpoint', 122.5), ('maximum', 130.0),
])
def test_percent_views_change_only_unknown_lots(monkeypatch, tmp_path, mode, expected):
    options = list(loaded(monkeypatch, tmp_path).values())
    before = copy.deepcopy([dataclasses.asdict(o) for o in options])
    views = {scenario.source(o).auction_lot_id: o
             for o in scenario.percent_views(options, {SPEED: 10}, mode)}

    unknown, known = views[UNKNOWN_LOT], views[KNOWN_LOT]
    assert unknown.art_percent == expected
    assert unknown.stats[SPEED] == pytest.approx(3.0 * expected / 100 * 1.3)
    assert known.art_percent == KNOWN_PERCENT
    assert known.stats[SPEED] == pytest.approx(KNOWN_SPEED)
    assert scenario.source(known) is known
    if mode == 'maximum':
        assert scenario.source(unknown) is unknown
    else:
        assert scenario.source(unknown).art_percent == 130.0
        assert unknown._quality_percent_mode == mode
    assert [dataclasses.asdict(o) for o in options] == before


@pytest.mark.parametrize('mode,expected', [
    ('minimum', 115.0), ('midpoint', 122.5), ('maximum', 130.0),
])
def test_ordinary_loader_applies_selected_unknown_mode(monkeypatch, tmp_path, mode, expected):
    options = loaded(monkeypatch, tmp_path, unknown_percent=mode)
    assert options[UNKNOWN_LOT].art_percent == expected
    assert options[UNKNOWN_LOT].stats[SPEED] == pytest.approx(3.0 * expected / 100 * 1.3)
    assert options[KNOWN_LOT].art_percent == KNOWN_PERCENT
    assert options[KNOWN_LOT].stats[SPEED] == pytest.approx(KNOWN_SPEED)


def test_ordinary_loader_defaults_and_alias(monkeypatch, tmp_path):
    assert loaded(monkeypatch, tmp_path)[UNKNOWN_LOT].art_percent == 130.0
    assert loaded(monkeypatch, tmp_path, conservative=True)[UNKNOWN_LOT].art_percent == 115.0
    assert loaded(monkeypatch, tmp_path, conservative=True,
                  unknown_percent='midpoint')[UNKNOWN_LOT].art_percent == 122.5


@pytest.mark.parametrize('mode,winner,percent,speed', [
    ('minimum', KNOWN_LOT, KNOWN_PERCENT, KNOWN_SPEED),
    ('midpoint', UNKNOWN_LOT, 122.5, 4.7775),
    ('maximum', UNKNOWN_LOT, 130.0, 5.07),
])
def test_max_quality_helium_choice_matches_mode(monkeypatch, tmp_path, mode, winner, percent, speed):
    helium_market(monkeypatch, tmp_path)
    result = call(budget=3_000_000, unknown_percent=mode)
    assert result and result.search_status['state'] == 'OPTIMAL', result.search_status
    chosen = result[0].artifacts[0]
    assert chosen.auction_lot_id == winner
    assert chosen.art_percent == percent
    assert result[0].combined_stats[SPEED] == pytest.approx(speed)
    assert result.search_status['calculation']['unknown_percent'] == mode
    data = result[0].to_dict()['artifacts'][0]
    if mode == 'maximum' or winner == KNOWN_LOT:
        assert 'calculation' not in data
    else:
        assert data['calculation']['unknown_percent'] == mode
        assert data['calculation']['unknown_percent_range']['min'] == 115
        assert data['calculation']['unknown_percent_range']['maximum'] == 130


@pytest.mark.parametrize('mode', ['minimum', 'midpoint', 'maximum'])
def test_max_quality_keeps_known_roll_actual(monkeypatch, tmp_path, mode):
    helium_market(monkeypatch, tmp_path, unknown=False)
    result = call(budget=3_000_000, unknown_percent=mode)
    assert result and result.search_status['state'] == 'OPTIMAL', result.search_status
    chosen = result[0].artifacts[0]
    assert chosen.auction_lot_id == KNOWN_LOT
    assert chosen.art_percent == KNOWN_PERCENT
    assert result[0].combined_stats[SPEED] == pytest.approx(KNOWN_SPEED)
    assert 'calculation' not in result[0].to_dict()['artifacts'][0]


@pytest.mark.parametrize('mode,winner,percent,speed', [
    ('minimum', KNOWN_LOT, KNOWN_PERCENT, KNOWN_SPEED),
    ('midpoint', UNKNOWN_LOT, 122.5, 4.7775),
    ('maximum', UNKNOWN_LOT, 130.0, 5.07),
])
def test_ordinary_search_helium_choice_matches_mode(monkeypatch, tmp_path, mode, winner, percent, speed):
    helium_market(monkeypatch, tmp_path)
    result = b.build_artifact_sets(
        3_000_000, tolerance_pct=0, priorities={SPEED: 10}, region='EU',
        unknown_percent=mode, top_n=1)
    assert result
    chosen = result[0].artifacts[0]
    assert chosen.auction_lot_id == winner
    assert chosen.art_percent == percent
    assert result[0].combined_stats[SPEED] == pytest.approx(speed)
    assert result[0]._search_request['unknown_percent'] == mode
    data = result[0].to_dict()['artifacts'][0]
    assert data['art_percent_known'] is (winner == KNOWN_LOT)
    if winner == UNKNOWN_LOT:
        assert data['unknown_percent_range'] == dict(min=115, midpoint=122.5, maximum=130)
        if mode == 'maximum':
            assert 'calculation' not in data
        else:
            assert data['calculation']['unknown_percent'] == mode
    else:
        assert 'unknown_percent_range' not in data
        assert 'calculation' not in data


def test_ordinary_known_roll_preserved_in_every_mode(monkeypatch, tmp_path):
    helium_market(monkeypatch, tmp_path, unknown=False)
    for mode in scenario.UNKNOWN_PERCENT_MODES:
        result = b.build_artifact_sets(
            3_000_000, tolerance_pct=0, priorities={SPEED: 10}, region='EU',
            unknown_percent=mode, top_n=1)
        assert result and result[0].artifacts[0].art_percent == KNOWN_PERCENT
        assert result[0].combined_stats[SPEED] == pytest.approx(KNOWN_SPEED)


def test_api_forwards_unknown_mode_and_keeps_historical_default(monkeypatch, tmp_path):
    main, client = api(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(quality, 'build_max_quality_sets',
                        lambda **kw: seen.append(kw) or quality.QualityResults(reason='test'))
    base = dict(budget=10, max_quality='true', priorities='{"speed_modifier":10}')

    def request(**extra):
        response = client.get('/api/builds', params=dict(base, **extra))
        assert response.status_code == 200, response.text
        return response.json(), seen[-1]

    body, call = request()
    assert call['unknown_percent'] == 'maximum' and call['conservative'] is False
    assert body['unknown_percent'] == 'maximum' and body['conservative'] is False

    body, call = request(conservative='1')
    assert call['unknown_percent'] == 'minimum' and call['conservative'] is True
    assert body['unknown_percent'] == 'minimum' and body['conservative'] is True

    body, call = request(unknown_percent='midpoint')
    assert call['unknown_percent'] == 'midpoint' and call['conservative'] is False
    assert body['unknown_percent'] == 'midpoint'

    # Explicit mode wins over the legacy alias.
    body, call = request(conservative='1', unknown_percent='maximum')
    assert call['unknown_percent'] == 'maximum' and call['conservative'] is False
    assert body['unknown_percent'] == 'maximum'

    response = client.get('/api/builds', params=dict(base, unknown_percent='median'))
    assert response.status_code == 422


def test_api_ordinary_engine_receives_unknown_mode(monkeypatch, tmp_path):
    main, client = api(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(main, 'build_artifact_sets', lambda **kw: seen.append(kw) or [])
    response = client.get('/api/builds', params=dict(
        budget=10, priorities='{"speed_modifier":10}', unknown_percent='midpoint'))
    assert response.status_code == 200, response.text
    assert seen[-1]['unknown_percent'] == 'midpoint'
    assert seen[-1]['conservative'] is False
    assert response.json()['unknown_percent'] == 'midpoint'


def test_export_endpoint_forwards_unknown_mode(monkeypatch, tmp_path):
    import app.lunar_bridge as lunar
    main, client = api(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(main, 'build_artifact_sets', lambda **kw: seen.append(kw) or [object()])
    monkeypatch.setattr(lunar, 'export_to_lunarzone', lambda build, armor: {'ok': True})
    response = client.get('/api/export-lunarzone', params=dict(
        budget=10, priorities='{"speed_modifier":10}', unknown_percent='midpoint'))
    assert response.status_code == 200, response.text
    assert seen[-1]['unknown_percent'] == 'midpoint'
    assert seen[-1]['conservative'] is False

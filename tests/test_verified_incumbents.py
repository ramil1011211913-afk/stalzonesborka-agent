from contextlib import contextmanager
import json
import sqlite3
from types import SimpleNamespace
import pytest
from app import builder as b, _feasibility_fallback as fallback
from app._objective import Objective
from app._quality_scenario import source
from tests.test_objective_integration import container


@pytest.fixture
def conservative_market(monkeypatch,tmp_path):
    path=tmp_path/'market.db'
    with sqlite3.connect(path) as c:
        c.executescript('''
        CREATE TABLE artefacts(id TEXT,name_ru TEXT,name_en TEXT,subcategory TEXT,stats TEXT,bonus_stats TEXT,icon_url TEXT);
        CREATE TABLE auction_history(item_id TEXT,quality INTEGER,price INTEGER,region TEXT);
        CREATE TABLE auction_lots(id INTEGER,item_id TEXT,region TEXT,quality INTEGER,ptn INTEGER,buyout_price INTEGER,
          bonus_properties TEXT,art_percent REAL,stats_random REAL,end_time TEXT,additional_raw TEXT);
        ''')
        c.execute('INSERT INTO artefacts VALUES(?,?,?,?,?,?,?)',('a','A','','',json.dumps([dict(prop_id='speed_modifier',min=0,max=10,is_positive=True)]),'[]',''))
        c.execute('INSERT INTO auction_lots VALUES(?,?,?,?,?,?,?,?,?,?,?)',(1,'a','EU',2,0,5,'[]',127.5,0.,'2099-01-01T00:00:00Z','{}'))
    @contextmanager
    def connect():
        c=sqlite3.connect(path);c.row_factory=sqlite3.Row
        try:yield c
        finally:c.close()
    monkeypatch.setattr(b,'get_connection',connect)
    monkeypatch.setattr(b,'load_container_options',lambda region:[container(slots=1)])
    return path


def test_ordinary_conservative_keeps_known_physical_descriptor(conservative_market):
    view=b.load_artifact_options(priorities={'speed_modifier':10},region='EU',conservative=True)[0]
    assert view.art_percent==127.5
    assert source(view) is view
    assert view.stats['speed_modifier']==12.75


def test_ordinary_conservative_can_purchase_a_known_real_lot(conservative_market):
    result=b.build_artifact_sets(10,tolerance_pct=0,priorities={'speed_modifier':10},region='EU',conservative=True)
    assert result and result[0].combined_stats['speed_modifier']==12.75
    assert result[0].total_price==5
    assert source(result[0].artifacts[0]).art_percent==127.5


def test_ordinary_conservative_minimizes_missing_stats_random(conservative_market):
    with sqlite3.connect(conservative_market) as c:
        c.execute("UPDATE auction_lots SET quality=3, art_percent=145, stats_random=0, additional_raw=?",
                  ('{"qlt":3}',))
    view=b.load_artifact_options(priorities={'speed_modifier':10},region='EU',conservative=True)[0]
    assert view.art_percent==130 and view.stats['speed_modifier']==13
    assert source(view).art_percent==145


def test_ordinary_conservative_keeps_real_max_with_stats_random(conservative_market):
    with sqlite3.connect(conservative_market) as c:
        c.execute("UPDATE auction_lots SET quality=3, art_percent=145, stats_random=2, additional_raw=?",
                  ('{"qlt":3,"stats_random":2}',))
    view=b.load_artifact_options(priorities={'speed_modifier':10},region='EU',conservative=True)[0]
    assert view.art_percent==145 and view.stats['speed_modifier']==14.5
    assert source(view) is view


def test_ordinary_conservative_minimizes_only_unknown_stored_percent(conservative_market):
    with sqlite3.connect(conservative_market) as c:
        c.execute('UPDATE auction_lots SET art_percent=NULL, stats_random=NULL')
    view=b.load_artifact_options(priorities={'speed_modifier':10},region='EU',conservative=True)[0]
    assert view.art_percent==115 and view.stats['speed_modifier']==11.5
    assert source(view).art_percent==130
    result=b.build_artifact_sets(10,tolerance_pct=0,priorities={'speed_modifier':10},region='EU',conservative=True)
    assert result and result[0].combined_stats['speed_modifier']==11.5
    assert source(result[0].artifacts[0]).art_percent==130


@pytest.mark.parametrize('field,value',[('buyout_price',6),('art_percent',128),('stats_random',1),('quality',3),('ptn',1),('bonus_properties','["SPEED_MOD"]'),('region','NA'),('end_time','2000-01-01T00:00:00Z')])
def test_conservative_view_never_weakens_original_market_validation(monkeypatch,conservative_market,field,value):
    saved=b.load_artifact_options(priorities={'speed_modifier':10},region='EU',conservative=True)
    with sqlite3.connect(conservative_market) as c:c.execute('UPDATE auction_lots SET '+field+'=?',(value,))
    monkeypatch.setattr(b,'load_artifact_options',lambda **kwargs:saved)
    assert not b.build_artifact_sets(10,tolerance_pct=0,priorities={'speed_modifier':10},region='EU',conservative=True)


def test_expired_diversity_anchor_does_not_hide_a_better_valid_incumbent(monkeypatch):
    def build(value,identity):
        option=SimpleNamespace(auction_lot_id=identity,is_owned=False,physical_id=str(identity))
        return SimpleNamespace(artifacts=[option],combined_stats={'speed_modifier':value},total_price=1,carryable=True)
    weak,strong,expired=build(5,1),build(10,2),build(11,3)
    monkeypatch.setattr(fallback,'current_candidates',lambda options,*a,**kw:[o for o in options if o.auction_lot_id!=3])
    objective=Objective.create({'speed_modifier':10})
    result=b._ordinary_result([expired,weak],[],True,10,{},{},True,False,False,
        {'speed_modifier':10},objective,'EU',{'top_n':2},primary_pool=[expired,strong,weak])
    assert result[0] is strong


def test_two_equal_price_real_copies_survive_conservative_views(monkeypatch,conservative_market):
    with sqlite3.connect(conservative_market) as c:
        c.execute('INSERT INTO auction_lots SELECT 2,item_id,region,quality,ptn,buyout_price,bonus_properties,art_percent,stats_random,end_time,additional_raw FROM auction_lots WHERE id=1')
    monkeypatch.setattr(b,'load_container_options',lambda region:[container(slots=2)])
    options=b.load_artifact_options(priorities={'speed_modifier':10},region='EU',conservative=True)
    assert {source(o).auction_lot_id for o in options}=={1,2}
    builds=b.build_artifact_sets(10,tolerance_pct=0,priorities={'speed_modifier':10},region='EU',conservative=True,allow_duplicates=True)
    assert builds and builds[0].combined_stats['speed_modifier']==25.5
    assert len({o.auction_lot_id for o in builds[0].artifacts})==2
    assert not b.build_artifact_sets(10,tolerance_pct=0,priorities={'speed_modifier':10},region='EU',conservative=True,allow_duplicates=False)


@pytest.mark.parametrize('failure',[RuntimeError,MemoryError])
def test_secondary_failure_preserves_verified_incumbent(monkeypatch,failure):
    from app import _beam_refine
    option=SimpleNamespace(auction_lot_id=1,is_owned=False,physical_id='1')
    incumbent=SimpleNamespace(artifacts=[option],combined_stats={'speed_modifier':5},total_price=1,carryable=True)
    monkeypatch.setattr(fallback,'current_candidates',lambda options,*a,**kw:options)
    def fail(*a,**kw):raise failure('controlled secondary failure')
    monkeypatch.setattr(_beam_refine,'refine_incumbent',fail)
    objective=Objective.create({'speed_modifier':10},mode='lexicographic',order=['speed_modifier'])
    result=b._ordinary_result([incumbent],[],True,10,{},{},True,False,False,{'speed_modifier':10},objective,'EU',{'top_n':1},refine_lexicographic=True)
    assert result[0] is incumbent

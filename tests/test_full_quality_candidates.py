"""Full SQL path, exact physical multiplicity and scoped completeness contract."""
from contextlib import contextmanager
from dataclasses import replace
import json
import math
import sqlite3
import time
import pytest
from app import builder as b, _max_quality as q, _max_quality_model as mm
from app import _quality_candidates as full
from tests.test_objective_integration import container
from tests.test_max_quality import option

def sql_market(monkeypatch, tmp_path, offers=None, slots=1):
    path=tmp_path/'full.db'
    with sqlite3.connect(path) as c:
        c.execute('CREATE TABLE artefacts(id TEXT PRIMARY KEY,name_ru TEXT,name_en TEXT,subcategory TEXT,stats TEXT,bonus_stats TEXT,icon_url TEXT)')
        c.execute('CREATE TABLE auction_lots(id INTEGER PRIMARY KEY,item_id TEXT,region TEXT,buyout_price INTEGER,quality INTEGER,ptn INTEGER,art_percent REAL,stats_random REAL,bonus_properties TEXT,end_time TEXT,additional_raw TEXT,amount INTEGER)')
        c.execute('CREATE TABLE auction_history(item_id TEXT,region TEXT,quality INTEGER,price INTEGER)')
        for i,(stats,price) in enumerate(offers or [({'speed_modifier':10},2)],1):
            base=[dict(prop_id=k,min=v,max=v,is_positive=v>=0) for k,v in stats.items()]
            c.execute('INSERT INTO artefacts VALUES(?,?,?,?,?,?,?)',(str(i),str(i),'','',json.dumps(base),'[]',''))
            c.execute('INSERT INTO auction_lots VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(i,str(i),'EU',price,0,0,100,None,'[]','2099-01-01T00:00:00Z','{"qlt":0,"ptn":0}',1))
    @contextmanager
    def connection():
        c=sqlite3.connect(path);c.row_factory=sqlite3.Row
        try:yield c
        finally:c.close()
    monkeypatch.setattr(b,'get_connection',connection)
    monkeypatch.setattr(b,'load_container_options',lambda region:[container(slots=slots)])
    return path,connection

def call(**kwargs):
    params=dict(budget=1000,tolerance_pct=0,priorities={'speed_modifier':10},region='EU',allow_duplicates=True,quality_timeout=10)
    params.update(kwargs)
    return q.build_max_quality_sets(**params)

@pytest.mark.parametrize('conservative',[False,True])
@pytest.mark.parametrize('stock',[False,True])
def test_actual_full_loader_and_search(monkeypatch,tmp_path,conservative,stock):
    sql_market(monkeypatch,tmp_path)
    result=call(conservative=conservative,stock_only=stock)
    assert result and result.search_status['state']=='OPTIMAL'
    audit=result.search_status['candidate_preparation']
    assert audit['original_physical_offers']==audit['accepted_physical_offers']==1
    assert audit['coverage']=='all_supported_physical_offers'
    assert result.search_status['models'][0]['candidate_coverage']['before_exact_reduction']==1
    assert not result.search_status['global_optimality_proven']

def test_sql_window_and_fairness_do_not_limit_quality(monkeypatch,tmp_path):
    path,_=sql_market(monkeypatch,tmp_path)
    with sqlite3.connect(path) as c:
        c.execute('delete from auction_lots')
        for i in range(1,21):
            c.execute('INSERT INTO auction_lots VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(i,'1','EU',i,0,0,80+i,None,'[]','2099-01-01T00:00:00Z','{"qlt":0}',1))
        for i in range(3):c.execute("INSERT INTO auction_history VALUES('1','EU',0,1)")
    ordinary=b.load_artifact_options(region='EU')
    assert len(ordinary)==2  # Historical price-fairness still applies ordinarily.
    result=call()
    assert result and result[0].artifacts[0].auction_lot_id==20
    assert result.search_status['candidate_preparation']['accepted_physical_offers']==20
    assert result.search_status['models'][0]['candidate_coverage']['provably_removed']==0

def test_counter_reserve_cannot_hide_feasibility(monkeypatch,tmp_path):
    offers=[({'speed_modifier':10,'health_bonus':-5},1)]
    offers += [({'biological_accumulation':-100},1) for _ in range(35)]
    offers += [({'biological_accumulation':-5,'health_bonus':10},2)]
    sql_market(monkeypatch,tmp_path,offers,2)
    result=call(min_thresholds={'health_bonus':0})
    assert result and result.search_status['state']=='OPTIMAL'
    assert {o.auction_lot_id for o in result[0].artifacts}=={1,37}
    assert result.search_status['models'][0]['candidate_coverage']['before_exact_reduction']==37

@pytest.mark.parametrize('column,value,reason',[
    ('quality',None,'unknown_or_inconsistent_quality'),
    ('ptn',None,'unknown_or_invalid_ptn'),
    ('additional_raw','{}','unknown_or_inconsistent_quality'),
    ('additional_raw','{"qlt":null}','unknown_or_inconsistent_quality'),
    ('additional_raw','{"qlt":1}','unknown_or_inconsistent_quality'),
    ('bonus_properties',None,'missing_bonus_descriptor'),
    ('amount',2,'ambiguous_physical_quantity')])
def test_unknown_source_fields_are_not_defaults(monkeypatch,tmp_path,column,value,reason):
    path,_=sql_market(monkeypatch,tmp_path)
    with sqlite3.connect(path) as c:c.execute('UPDATE auction_lots SET '+column+'=?',(value,))
    result=call()
    assert not result
    audit=result.search_status['candidate_preparation']
    assert audit['accepted_physical_offers']==0 and audit['unknown_exclusions'][reason]==1

@pytest.mark.parametrize('ids,reason',[([], 'incomplete_or_duplicate_bonuses'),(['UNKNOWN'],'unsupported_bonus_ids')])
def test_unknown_open_bonuses_never_simulated(monkeypatch,tmp_path,ids,reason):
    path,_=sql_market(monkeypatch,tmp_path)
    with sqlite3.connect(path) as c:c.execute("UPDATE auction_lots SET ptn=5,bonus_properties=?,additional_raw=?",(json.dumps(ids),'{"qlt":0,"ptn":5}'))
    result=call()
    assert not result and result.search_status['candidate_preparation']['unknown_exclusions'][reason]==1

def test_expired_cheap_copies_cannot_displace_live_offer(monkeypatch,tmp_path):
    path,_=sql_market(monkeypatch,tmp_path)
    with sqlite3.connect(path) as c:
        for i in range(2,12):
            c.execute('INSERT INTO auction_lots SELECT ?,item_id,region,1,quality,ptn,art_percent,stats_random,bonus_properties,?,additional_raw,amount FROM auction_lots WHERE id=1',(i,'2000-01-01T00:00:00Z'))
    result=call()
    assert result and result[0].artifacts[0].auction_lot_id==1
    assert result.search_status['candidate_preparation']['mandatory_exclusions']['expired']==10

def test_model_and_stream_caps_are_explicit_unknown(monkeypatch,tmp_path):
    sql_market(monkeypatch,tmp_path,[({'speed_modifier':10},1),({'speed_modifier':9},1)])
    monkeypatch.setattr(mm,'MAX_CANDIDATES',1)
    result=call()
    assert not result and result.search_status['state']=='UNKNOWN'
    assert result.search_status['models'][0]['reason']=='model_limit'
    assert result.search_status['models'][0]['candidate_coverage']['solver_candidates']==2
    monkeypatch.setattr(mm,'MAX_INPUT_CANDIDATES',1)
    result=call()
    assert not result and result.search_status['reason']=='input_candidate_limit'
    assert not result.search_status['candidate_preparation']['enumeration_complete']

@pytest.mark.parametrize('owned',[False,True])
def test_exact_reduction_respects_all_bits_and_ownership(owned):
    values=[option(i,{'speed_modifier':1.,'health_bonus':2.},i+1,item='same') for i in range(12)]
    if owned:values[0].is_owned=True
    reduced,meta=full.exact_reduce(values,6,True)
    assert len(reduced)==(12 if owned else 6)
    assert meta['provably_removed']==(0 if owned else 6)
    values[-1].stats={'speed_modifier':1.,'health_bonus':math.nextafter(2.,3.)}
    reduced,meta=full.exact_reduce(values,6,True)
    assert len(reduced)==(12 if owned else 7)
    assert values[-1] in reduced
    assert full.exact_reduce(values,6,False)[0] is values

def test_conflicting_physical_hypotheses_not_collapsed():
    values=[option(i,{'speed_modifier':1.},1,item='same') for i in range(12)]
    values[1].auction_lot_id=values[0].auction_lot_id
    reduced,meta=full.exact_reduce(values,2,True)
    assert values[0] in reduced and values[1] in reduced and len(reduced)==4

@pytest.mark.parametrize('owned_only',[False,True])
def test_real_owned_only_and_mixed_sources(monkeypatch,tmp_path,owned_only):
    path,_=sql_market(monkeypatch,tmp_path,slots=2 if not owned_only else 1)
    if owned_only:
        with sqlite3.connect(path) as c:c.execute('DELETE FROM auction_lots')
    owned=[dict(item_id='1',quality=0,ptn=0,art_percent=100,instance_id='own')]
    result=call(owned_artifacts=owned)
    assert result and result.search_status['state']=='OPTIMAL'
    assert any(o.is_owned for o in result[0].artifacts)
    assert result.search_status['models'][0]['candidate_coverage']['skipped_reason']=='type_uniqueness_or_owned_semantics'
    assert len({o.physical_id for o in result[0].artifacts})==len(result[0].artifacts)

@pytest.mark.parametrize('key,value',[('quality',None),('ptn',None),('art_percent',None),('bonus_ids',None)])
def test_owned_unknown_fields_are_explicit_unknown(monkeypatch,tmp_path,key,value):
    sql_market(monkeypatch,tmp_path)
    owned=dict(item_id='1',quality=0,ptn=0,art_percent=100,bonus_ids=[])
    owned[key]=value
    result=call(owned_artifacts=[owned])
    assert not result and result.search_status['state']=='UNKNOWN'
    assert result.search_status['reason']=='unsupported_owned_descriptor'

def test_owned_input_bound_has_no_silent_trim(monkeypatch,tmp_path):
    sql_market(monkeypatch,tmp_path)
    monkeypatch.setattr(mm,'MAX_INPUT_CANDIDATES',2)
    result=call(owned_artifacts=[dict(item_id='1',quality=0,ptn=0,art_percent=100) for _ in range(3)])
    assert not result and result.search_status['state']=='UNKNOWN'
    assert result.search_status['candidate_preparation']['owned_input_limit']

@pytest.mark.parametrize('changes',[1,2])
def test_actual_full_loader_refresh_retries_once_with_new_ids(monkeypatch,tmp_path,changes):
    path,_=sql_market(monkeypatch,tmp_path)
    with sqlite3.connect(path.parent/'app_state.sqlite3') as c:
        c.execute('CREATE TABLE refresh_jobs(name TEXT,generation INTEGER,running INTEGER)')
        c.execute("INSERT INTO refresh_jobs VALUES('auction:EU',0,0)")
    native=q.run_model;calls=[]
    def run(*args,**kw):
        result=native(*args,**kw);calls.append(result)
        if len(calls)<=changes:
            with sqlite3.connect(path) as c:
                c.execute('UPDATE auction_lots SET id=id+10,buyout_price=buyout_price+1')
            with sqlite3.connect(path.parent/'app_state.sqlite3') as c:
                c.execute('UPDATE refresh_jobs SET generation=generation+1')
        return result
    monkeypatch.setattr(q,'run_model',run)
    result=call()
    assert len(calls)==2 and result.search_status['retry_count']==1
    if changes==1:
        assert result and result[0].artifacts[0].auction_lot_id==11 and result[0].total_price==3
    else:
        assert not result and result.search_status['state']=='UNKNOWN'
        assert result.search_status['validation_reason']=='market_changed'

@pytest.mark.parametrize('mode',['balanced','lexicographic'])
def test_full_sql_equal_profile_multiplicity_keeps_cheapest_exactly(monkeypatch,tmp_path,mode):
    path,_=sql_market(monkeypatch,tmp_path,slots=2)
    with sqlite3.connect(path) as c:
        for i in range(2,15):
            c.execute('INSERT INTO auction_lots SELECT ?,item_id,region,?,quality,ptn,art_percent,stats_random,bonus_properties,end_time,additional_raw,amount FROM auction_lots WHERE id=1',(i,i+1))
    result=call(conservative=True,optimization_mode=mode,priority_order=['speed_modifier'] if mode=='lexicographic' else None)
    assert result and result.search_status['state']=='OPTIMAL'
    assert {o.auction_lot_id for o in result[0].artifacts}=={1,2}
    info=result.search_status['models'][0]['candidate_coverage']
    assert info['before_exact_reduction']==14 and info['solver_candidates']==2 and info['provably_removed']==12

@pytest.mark.parametrize('opened',[False,True])
def test_only_opened_bonus_coefficients_are_required(monkeypatch,tmp_path,opened):
    path,_=sql_market(monkeypatch,tmp_path)
    with sqlite3.connect(path) as c:
        c.execute('UPDATE artefacts SET bonus_stats=?',(json.dumps([dict(prop_id='health_bonus')]),))
        if opened:c.execute('UPDATE auction_lots SET ptn=5,bonus_properties=?,additional_raw=?',('["HEALTH_BONUS"]','{"qlt":0,"ptn":5}'))
    result=call()
    if opened:
        assert not result and result.search_status['candidate_preparation']['unknown_exclusions']['invalid_catalogue_coefficients']==1
    else:assert result and result.search_status['state']=='OPTIMAL'

def test_owned_missing_coefficients_are_never_invented(monkeypatch,tmp_path):
    path,_=sql_market(monkeypatch,tmp_path)
    with sqlite3.connect(path) as c:c.execute('UPDATE artefacts SET stats=?',(json.dumps([dict(prop_id='speed_modifier',max=10)]),))
    result=call(owned_artifacts=[dict(item_id='1',quality=0,ptn=0,art_percent=100)])
    assert not result and result.search_status['state']=='UNKNOWN'
    assert result.search_status['reason']=='unsupported_owned_catalogue'

def test_real_sql_API_exposes_optional_coverage(monkeypatch,tmp_path):
    from tests.test_stock_only import api
    sql_market(monkeypatch,tmp_path)
    main,client=api(monkeypatch,tmp_path)
    response=client.get('/api/builds',params=dict(budget=1000,tolerance=0,max_quality='true',deep_optimization='true',stock_only='true',conservative='true',priorities='{"speed_modifier":10}'))
    assert response.status_code==200
    body=response.json();assert body['builds'] and body['search_status']['state']=='OPTIMAL'
    assert body['search_status']['candidate_preparation']['original_physical_offers']==1
    assert body['search_status']['models'][0]['candidate_coverage']['solver_candidates']==1

@pytest.mark.parametrize('column,reason',[('ptn','unknown_or_invalid_ptn'),('quality','unknown_or_inconsistent_quality')])
def test_missing_SQL_columns_never_become_zero(monkeypatch,tmp_path,column,reason):
    path,_=sql_market(monkeypatch,tmp_path)
    with sqlite3.connect(path) as c:c.execute('ALTER TABLE auction_lots DROP COLUMN '+column)
    result=call()
    assert not result and not result.search_status['global_optimality_proven']
    assert result.search_status['candidate_preparation']['unknown_exclusions'][reason]==1

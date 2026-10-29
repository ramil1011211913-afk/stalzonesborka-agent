"""Minimum-quality point scenario and strict ORIGINAL descriptor validation."""
import copy
import dataclasses
import itertools
import json
import sqlite3
import time

import pytest
from app import builder as b, _max_quality as quality, _quality_scenario as scenario
from app._objective import Objective
from tests.test_feasibility_fallback import market
from tests.test_max_quality import call

REAL_COMPUTE = b.compute_artifact_stats


def real_market(monkeypatch, tmp_path, rows=None, slots=1):
    opts, path, connect = market(monkeypatch, tmp_path, rows or [({'sprint_speed_modifier': 10}, 2)], slots)
    monkeypatch.setattr(b, 'compute_artifact_stats', REAL_COMPUTE)
    for opt in opts:
        opt.quality = 2
        opt.art_percent = 127.5
        opt.stats_random = 1.3333333333
        opt.base_stats = [dict(prop_id=k, min=v*.85, max=v, is_positive=not k.endswith('_accumulation'))
                          for k,v in opt.stats.items()]
        opt.stats, opt.stats_raw, opt.stats_names, _ = REAL_COMPUTE(
            opt.base_stats, [], [], opt.art_percent, 0, 115, 15, opt.stats_random)
    with sqlite3.connect(path) as c:
        c.execute('update auction_lots set quality=2, art_percent=127.5, stats_random=1.3333333333')
    return opts, path, connect


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic'])
@pytest.mark.parametrize('known', [False, True])
def test_conservative_known_and_missing_percent_use_scenario_not_source(monkeypatch,tmp_path,mode,known):
    opts,path,_=real_market(monkeypatch,tmp_path)
    if not known:
        opts[0].art_percent=130;opts[0].stats_random=0
        with sqlite3.connect(path) as c:c.execute('update auction_lots set art_percent=130, stats_random=0')
    before=copy.deepcopy(dataclasses.asdict(opts[0]))
    result=call(conservative=True,priorities={'sprint_speed_modifier':10},optimization_mode=mode,
                priority_order=['sprint_speed_modifier'] if mode=='lexicographic' else None)
    assert result and result.search_status['state']=='OPTIMAL',result.search_status
    assert result[0].combined_stats['sprint_speed_modifier']==pytest.approx(11.5)
    view=result[0].artifacts[0]
    assert view.art_percent==115 and scenario.source(view) is opts[0]
    assert dataclasses.asdict(opts[0])==before
    assert result[0].to_dict()['artifacts'][0]['calculation']['source_stored_percent']==before['art_percent']
    assert not result.search_status['calculation']['robust']
    info=result.search_status['models'][0]
    assert info['physical_pool_sha256']!=info['pool_sha256']
    assert b._final_valid(result[0],10,{},{},True,False,True)


@pytest.mark.parametrize('change',['price','percent','random','quality','ptn','bonus','region','expired','missing'])
def test_original_descriptor_rechecked_after_conservative_solver(monkeypatch,tmp_path,change):
    _,path,_=real_market(monkeypatch,tmp_path)
    def fake(model,*args):
        with sqlite3.connect(path) as c:
            if change=='missing':c.execute('delete from auction_lots')
            else:
                column,value={'price':('buyout_price',3),'percent':('art_percent',129),
                    'random':('stats_random',2),'quality':('quality',3),'ptn':('ptn',1),
                    'bonus':('bonus_properties','["UNKNOWN"]'),'region':('region','RU'),
                    'expired':('end_time','2000-01-01T00:00:00Z')}[change]
                c.execute('update auction_lots set '+column+'=?',(value,))
        return dict(stages=[dict(indices=[0])],optimal=False,reason='time_or_node_limit')
    monkeypatch.setattr(quality,'run_model',fake)
    result=call(conservative=True,priorities={'sprint_speed_modifier':10})
    assert not result and result.search_status['state']=='UNKNOWN'
    assert result.search_status['models'][0]['validation_failed']


def test_candidates_feasible_only_at_minimum_survive_preparation(monkeypatch,tmp_path):
    opts,path,_=real_market(monkeypatch,tmp_path,[({'sprint_speed_modifier':10,'radiation_accumulation':11.5},2)])
    opts[0].art_percent=130
    with sqlite3.connect(path) as c:c.execute('update auction_lots set art_percent=130')
    assert REAL_COMPUTE(opts[0].base_stats,[],[],130,0,115,15,None)[0]['radiation_accumulation']>10
    result=call(conservative=True,priorities={'sprint_speed_modifier':10})
    assert result and result[0].carryable
    assert result[0].combined_stats['radiation_accumulation']==pytest.approx(9.775)


@pytest.mark.parametrize('kind',['budget','min','max','carry'])
def test_scenario_hard_constraints_not_relaxed(monkeypatch,tmp_path,kind):
    real_market(monkeypatch,tmp_path,[({'sprint_speed_modifier':10,'radiation_accumulation':20},2)])
    args=dict(carryable_only=False)
    if kind=='budget':args['budget']=1
    if kind=='min':args['min_thresholds']={'sprint_speed_modifier':12}
    if kind=='max':args['max_thresholds']={'sprint_speed_modifier':11}
    if kind=='carry':args['carryable_only']=True
    result=call(conservative=True,priorities={'sprint_speed_modifier':10},**args)
    assert not result


def test_owned_source_percent_preserved_but_scenario_uses_quality_minimum(monkeypatch,tmp_path):
    opts,path,_=real_market(monkeypatch,tmp_path)
    original=opts[0];original.is_owned=True;original.buyout_price=0
    original.auction_lot_id=None;original.physical_id='owned:one';original.owned_instance_id='one'
    views=scenario.minimum_views([original],{'sprint_speed_modifier':10})
    assert views[0].art_percent==115 and scenario.source(views[0]).art_percent==127.5
    assert scenario.current_views(views,b.get_connection,'EU',b._normalize_bonus_id)==views


@pytest.mark.parametrize('kind',['before','incumbent','error','dependency_unavailable','cancelled','memory_limit'])
def test_conservative_timeouts_and_solver_error_status(monkeypatch,tmp_path,kind):
    real_market(monkeypatch,tmp_path)
    def fake(*args):
        return dict(stages=[] if kind!='incumbent' else [dict(indices=[0])],optimal=False,
                    reason='solver_error' if kind=='error' else kind if kind in ('dependency_unavailable','cancelled','memory_limit') else 'time_or_node_limit')
    monkeypatch.setattr(quality,'run_model',fake)
    result=call(conservative=True,priorities={'sprint_speed_modifier':10})
    assert result.search_status['state']==('FEASIBLE' if kind=='incumbent' else 'UNKNOWN')
    assert not result.search_status['optimality_proven']
    assert result.search_status['calculation']['conservative']


def test_owned_conservative_end_to_end_and_conflicting_source_percent(monkeypatch,tmp_path):
    opts,path,_=real_market(monkeypatch,tmp_path)
    base=opts[0].base_stats
    with sqlite3.connect(path) as c:
        c.execute('create table artefacts(id text,name_ru text,name_en text,subcategory text,stats text,bonus_stats text,icon_url text)')
        c.execute('insert into artefacts values(?,?,?,?,?,?,?)',('owned','own','','',json.dumps(base),'[]',''))
    own=dict(item_id='owned',quality=2,ptn=0,art_percent=129,bonus_ids=[],instance_id='mine')
    result=call(conservative=True,priorities={'sprint_speed_modifier':10},owned_artifacts=[own])
    assert result and result[0].total_price==0 and result[0].artifacts[0].art_percent==115
    assert scenario.source(result[0].artifacts[0]).art_percent==129
    bad=call(conservative=True,priorities={'sprint_speed_modifier':10},owned_artifacts=[own,dict(own,art_percent=128)])
    assert not bad and bad.search_status['models'][0]['reason']=='ambiguous_owned_identity'


def test_conservative_container_and_armor_profiles(monkeypatch,tmp_path):
    from tests.test_objective_integration import container
    real_market(monkeypatch,tmp_path)
    monkeypatch.setattr(b,'load_container_options',lambda region:[container('a',1,100),container('b',1,50)])
    armor=dict(id='armor',level=15,stats={'sprint_speed_modifier':2,'health_bonus':2},name_ru='armor',category='',rank='',icon_url='')
    monkeypatch.setattr(b,'load_armor',lambda *a:armor)
    result=call(conservative=True,priorities={'sprint_speed_modifier':10},armor_ids=[None,'armor'])
    assert result and result.search_status['state']=='OPTIMAL'
    assert result[0].container.item_id=='a' and result[0]._armor['id']=='armor'
    assert result[0].combined_stats['sprint_speed_modifier']==pytest.approx(13.5)
    assert len(result.search_status['models'])==4


def test_conservative_does_not_enable_guaranteed_carryability():
    result=call(conservative=True,guaranteed_carryable=True)
    assert not result and result.search_status['state']=='UNKNOWN'


def test_scenario_model_matches_independent_small_full_enumeration(monkeypatch,tmp_path):
    rows=[({'sprint_speed_modifier':v,'health_bonus':h},p) for v,h,p in [(10,-2,2),(8,3,3),(7,1,1),(9,-1,2),(3,2,1)]]
    opts,_,_=real_market(monkeypatch,tmp_path,rows,slots=2)
    feasible=[]
    for pair in itertools.combinations(opts,2):
        stats={'sprint_speed_modifier':sum(o.base_stats[0]['max']*1.15 for o in pair),
               'health_bonus':sum((s['max']*1.15 if s['is_positive'] else 0) for o in pair for s in o.base_stats if s['prop_id']=='health_bonus')}
        # Health endpoints can be negative even with is_positive=True; native uses
        # the more-negative endpoint as max. Compute that sign explicitly.
        health=sum(min(s['min'],s['max'])*1.15 if s['max']<0 else s['max']*1.15 for o in pair for s in o.base_stats if s['prop_id']=='health_bonus')
        if health>=0 and sum(o.buyout_price for o in pair)<=10:
            feasible.append((stats['sprint_speed_modifier'],-sum(o.buyout_price for o in pair)))
    result=call(conservative=True,priorities={'sprint_speed_modifier':10},min_thresholds={'health_bonus':0})
    assert result and result.search_status['state']=='OPTIMAL'
    assert (result[0].combined_stats['sprint_speed_modifier'],-result[0].total_price)==max(feasible)


def test_real_conservative_api_uses_scenario_and_never_refreshes_market(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from app import database
    from fastapi.testclient import TestClient
    import importlib
    real_market(monkeypatch,tmp_path)
    monkeypatch.setattr(database,'init_db',lambda:None)
    main=importlib.import_module('app.main')
    monkeypatch.setattr(main.Config,'DATA_DIR',str(tmp_path/'api-data'))
    monkeypatch.setattr(main,'MAX_QUALITY_RELEASED',True)
    monkeypatch.setattr(main,'check_auction_freshness',lambda *a:dict(is_fresh=True,last_collected=None,age_hours=0))
    monkeypatch.setattr(main,'_start_auction_refresh',lambda:pytest.fail('No auction refresh permitted'))
    response=TestClient(main.app).get('/api/builds',params=dict(budget=10,tolerance=0,
        priorities='{"sprint_speed_modifier":10}',max_quality='true',conservative='1',allow_duplicates='true'))
    assert response.status_code==200,response.text
    data=response.json()
    assert data['search_status']['state']=='OPTIMAL'
    assert data['search_status']['calculation']['mode']=='minimum_quality_point_scenario'
    assert data['builds'][0]['artifacts'][0]['calculation']['source_stored_percent']==127.5

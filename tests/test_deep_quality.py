"""Deep time policy, full-rank proof, API propagation and unchanged safety."""
from pathlib import Path
from types import SimpleNamespace
import importlib,sqlite3,threading,time
import numpy as np
import pytest
from app import _max_quality as q,_max_quality_worker as worker,_max_quality_model as model,_max_quality_process as process
from app._quality_time import request_seconds,stage_seconds,DEEP_NODE_LIMIT
from app._app_state import StateStore
from app._objective import Objective
from tests.test_conservative_quality import real_market
from tests.test_max_quality import call,option
from tests.test_objective_integration import container


@pytest.mark.parametrize('seconds',[0,-1,61,120,300,float('nan'),float('inf')])
def test_standard_timeout_contract_unchanged(seconds):
    with pytest.raises(ValueError):request_seconds(seconds)


def test_deep_is_explicit_and_does_not_change_standard():
    assert request_seconds(45)==45 and request_seconds(60)==60
    assert request_seconds(45,True)==request_seconds(300,True)==300
    with pytest.raises(ValueError):request_seconds(301,True)
    assert stage_seconds({'model_seconds':15},0,2,299)==15
    assert stage_seconds({'model_seconds':15},0,2,.02)==.02


@pytest.mark.parametrize('count',[2,5,9])
def test_deep_reserves_later_stages_without_fixed_15s_cap(count):
    m=dict(model_seconds=300,time_policy='deep_remaining',stage_reserve_seconds=15)
    first=stage_seconds(m,0,count,299)
    assert first>15 and 299-first>=15*(count-1)
    assert stage_seconds(m,count-1,count,60)==pytest.approx(59.9)
    assert 0<stage_seconds(m,0,count,1)<1


def tiny():
    return model.make_model([option(1,{'speed_modifier':2},2)],container(slots=1),Objective.create({'speed_modifier':10}),10,{},{},True,True)


@pytest.mark.parametrize('stop',[None,'primary','cost','no_incumbent'])
def test_full_rank_requires_both_stage_certificates(monkeypatch,tmp_path,stop):
    clock=[100.];monkeypatch.setattr(worker.time,'monotonic',lambda:clock[0]);calls=[]
    m=dict(tiny(),model_seconds=300,time_policy='deep_remaining',stage_reserve_seconds=15,deadline=400.,seconds=300)
    def solver(cost,**kw):
        i=len(calls);calls.append(kw['options']);clock[0]+=kw['options']['time_limit']
        status=1 if (i==0 and stop in ('primary','no_incumbent')) or (i==1 and stop=='cost') else 0
        return SimpleNamespace(status=status,message='controlled',x=None if stop=='no_incumbent' else np.array([1.]),
            mip_dual_bound=float(cost[0]),mip_gap=0.,mip_node_count=1)
    result=worker.solve(m,tmp_path/'result.json',solver=solver)
    assert len(calls)==(1 if stop in ('primary','no_incumbent') else 2)
    assert calls[0]['time_limit']>280
    if len(calls)==2:assert 14.8<calls[1]['time_limit']<=15.1
    assert result['optimal']==(stop is None)
    assert all(x['node_limit']==20000 and x['mip_rel_gap']==0 for x in calls)
    if stop=='no_incumbent':assert 'indices' not in result['stages'][0]


def fake_worker(data,*args):
    def solve(cost,**kw):
        x=np.zeros(data['n']);x[0]=1
        return SimpleNamespace(status=0,message='controlled optimum',x=x,mip_dual_bound=float(cost@x),mip_gap=0,mip_node_count=0)
    # Does not launch a process; validates orchestration using the real worker.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        return worker.solve(dict(data,seconds=300,deadline=args[0]),Path(tmp)/'result.json',solver=solve)


@pytest.mark.parametrize('conservative',[False,True])
def test_deep_reaches_model_and_native_validation(monkeypatch,tmp_path,conservative):
    real_market(monkeypatch,tmp_path);captured=[]
    def run(data,deadline,*a):
        captured.append((data,deadline-time.monotonic()));return fake_worker(data,deadline,*a)
    monkeypatch.setattr(q,'run_model',run)
    r=call(deep_optimization=True,conservative=conservative,priorities={'sprint_speed_modifier':10})
    assert r and r.search_status['state']=='OPTIMAL'
    assert r.search_status['limits']['seconds']==300 and r.search_status['deep_optimization']
    data,left=captured[0];assert 295<left<300
    assert data['model_seconds']==300 and data['time_policy']=='deep_remaining'
    assert data['memory_bytes']==536870912 and data['node_limit']==DEEP_NODE_LIMIT
    assert r[0]._search_request['deep_optimization'] and r[0]._search_request['quality_timeout']==300


@pytest.mark.parametrize('conservative',[False,True])
def test_deep_retry_once_same_deadline(monkeypatch,tmp_path,conservative):
    _,path,_=real_market(monkeypatch,tmp_path);store=StateStore(tmp_path/'app_state.sqlite3');assert store.claim('auction:EU').finish()
    calls=[]
    def run(data,deadline,*a):
        calls.append(deadline)
        with sqlite3.connect(path) as c:c.execute("update auction_lots set end_time=?",('2099-01-%02dT00:00:00Z'%(len(calls)+1),))
        return fake_worker(data,deadline,*a)
    monkeypatch.setattr(q,'run_model',run)
    r=call(deep_optimization=True,conservative=conservative,priorities={'sprint_speed_modifier':10})
    assert not r and r.search_status['reason']=='market_changed'
    assert len(calls)==2 and calls[0]==calls[1] and r.search_status['retry_count']==1


def test_deep_active_refresh_and_cancel_do_not_start_worker(monkeypatch,tmp_path):
    real_market(monkeypatch,tmp_path);store=StateStore(tmp_path/'app_state.sqlite3');store.claim('auction:EU')
    monkeypatch.setattr(q,'run_model',lambda *a:pytest.fail('Active refresh must not start native solve'))
    r=call(deep_optimization=True,priorities={'sprint_speed_modifier':10})
    assert not r and r.search_status['reason']=='market_refresh_in_progress'
    event=threading.Event();event.set()
    r=call(deep_optimization=True,cancel_event=event)
    assert not r and r.search_status['reason']=='cancelled' and r.search_status['limits']['seconds']==300


def test_deep_native_process_cancel_no_leak():
    import psutil
    event=threading.Event();timer=threading.Timer(.2,event.set)
    data=dict(tiny(),model_seconds=300,time_policy='deep_remaining',stage_reserve_seconds=15)
    before={p.pid for p in psutil.Process().children(recursive=True)}
    timer.start()
    try:r=process.run_model(data,time.monotonic()+300,event)
    finally:timer.cancel();timer.join()
    assert r['reason']=='cancelled' and not r['optimal']
    assert not ({p.pid for p in psutil.Process().children(recursive=True)}-before)


def test_deep_node_cap_explanation_preserves_raw_solver_status(monkeypatch,tmp_path):
    real_market(monkeypatch,tmp_path)
    monkeypatch.setattr(q,'run_model',lambda *a:dict(optimal=False,infeasible=False,reason='solver_error',
        stages=[dict(stage=0,kind='priority',indices=[0],solver_status=4,solver_message='Solution limit reached',node_count=DEEP_NODE_LIMIT)]))
    result=call(deep_optimization=True,priorities={'sprint_speed_modifier':10})
    assert result and result.search_status['state']=='FEASIBLE'
    assert result.search_status['reason']=='node_limit' and result.search_status['worker_reason']=='solver_error'
    assert result.search_status['models'][0]['stages'][0]['solver_status']==4


@pytest.mark.parametrize('route',['/api/builds','/api/export-lunarzone'])
def test_api_deep_explicit_flag_and_standard_range(monkeypatch,tmp_path,route):
    from app import database
    from fastapi.testclient import TestClient
    monkeypatch.setattr(database,'init_db',lambda:None)
    main=importlib.import_module('app.main');monkeypatch.setattr(main.Config,'DATA_DIR',str(tmp_path/'api-data'))
    monkeypatch.setattr(main,'MAX_QUALITY_RELEASED',True)
    monkeypatch.setattr(main,'check_auction_freshness',lambda *a:dict(is_fresh=True,last_collected=None,age_hours=0))
    monkeypatch.setattr(main,'_start_auction_refresh',lambda:pytest.fail('Deep request must not refresh'))
    calls=[]
    monkeypatch.setattr(q,'build_max_quality_sets',lambda **kw:calls.append(kw) or q.QualityResults(reason='test'))
    client=TestClient(main.app);params=dict(max_quality='true',deep_optimization='true',priorities='{"speed_modifier":10}')
    response=client.get(route,params=params)
    assert response.status_code==(200 if route=='/api/builds' else 404)
    assert calls[-1]['quality_timeout']==300 and calls[-1]['deep_optimization']
    for bad in [dict(max_quality='false',deep_optimization='true'),dict(max_quality='true',quality_timeout=300),dict(max_quality='true',deep_optimization='true',quality_timeout=301)]:
        assert client.get(route,params=dict(priorities='{"speed_modifier":10}',**bad)).status_code==422
    client.get(route,params=dict(max_quality='true',quality_timeout=60,priorities='{"speed_modifier":10}'))
    assert calls[-1]['quality_timeout']==60 and 'deep_optimization' not in calls[-1]


def test_actual_api_deep_native_validation(monkeypatch,tmp_path):
    from app import database
    from fastapi.testclient import TestClient
    real_market(monkeypatch,tmp_path)
    monkeypatch.setattr(database,'init_db',lambda:None)
    main=importlib.import_module('app.main');monkeypatch.setattr(main.Config,'DATA_DIR',str(tmp_path/'api-data'))
    monkeypatch.setattr(main,'MAX_QUALITY_RELEASED',True)
    monkeypatch.setattr(main,'check_auction_freshness',lambda *a:dict(is_fresh=True,last_collected=None,age_hours=0))
    monkeypatch.setattr(main,'_start_auction_refresh',lambda:pytest.fail('No refresh'))
    monkeypatch.setattr(q,'run_model',fake_worker)
    r=TestClient(main.app).get('/api/builds',params=dict(budget=10,tolerance=0,max_quality='true',deep_optimization='true',conservative='true',allow_duplicates='true',priorities='{"sprint_speed_modifier":10}'))
    assert r.status_code==200
    value=r.json();assert value['search_status']['state']=='OPTIMAL' and value['search_status']['deep_optimization']
    assert value['search_status']['limits']['seconds']==300 and value['builds'][0]['carryable']

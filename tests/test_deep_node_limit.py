"""Finite deep-only node budget reaches both native stages and API diagnostics."""
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from app import _max_quality as q,_max_quality_worker as worker
from app._quality_time import DEEP_NODE_LIMIT
from tests.test_conservative_quality import real_market
from tests.test_max_quality import call
from tests.test_deep_quality import fake_worker,tiny


@pytest.mark.parametrize('deep',[False,True])
@pytest.mark.parametrize('conservative',[False,True])
def test_only_deep_model_and_status_get_larger_finite_cap(monkeypatch,tmp_path,deep,conservative):
 real_market(monkeypatch,tmp_path);captured=[]
 def run(data,*args):
  captured.append(data);return fake_worker(data,*args)
 monkeypatch.setattr(q,'run_model',run)
 result=call(quality_timeout=45,deep_optimization=deep,conservative=conservative,priorities={'sprint_speed_modifier':10})
 expected=DEEP_NODE_LIMIT if deep else 20000
 assert result and result.search_status['state']=='OPTIMAL'
 assert captured[0]['node_limit']==result.search_status['limits']['nodes']==expected
 assert captured[0]['memory_bytes']==536870912 and result.search_status['limits']['simultaneous_heavy_tasks']==1
 assert 20000<DEEP_NODE_LIMIT<=200000
 assert result.search_status['limits']['seconds']==(300 if deep else 45)
 if deep:assert result.search_status['models'][0]['node_limit']==DEEP_NODE_LIMIT


@pytest.mark.parametrize('stop_cost',[False,True])
def test_highs_options_both_stages_and_incomplete_price(monkeypatch,tmp_path,stop_cost):
 calls=[]
 def solver(cost,**kw):
  i=len(calls);calls.append(kw['options']);status=1 if stop_cost and i==1 else 0
  return SimpleNamespace(status=status,message='controlled',x=np.array([1.]),mip_dual_bound=float(cost[0]),
   mip_gap=0,mip_node_count=DEEP_NODE_LIMIT if status else 1)
 model=dict(tiny(),node_limit=DEEP_NODE_LIMIT,model_seconds=300,time_policy='deep_remaining',stage_reserve_seconds=15,seconds=300)
 result=worker.solve(model,tmp_path/'result.json',solver=solver)
 assert len(calls)==2 and all(x['node_limit']==DEEP_NODE_LIMIT and x['mip_rel_gap']==0 for x in calls)
 assert result['optimal']==(not stop_cost)
 assert result['stages'][0]['solver_status']==0
 assert result['stages'][1]['solver_status']==(1 if stop_cost else 0)


@pytest.mark.parametrize('count_kind',['old_cap','below_cap','at_cap'])
def test_reason_uses_actual_cap_not_old_20k(monkeypatch,tmp_path,count_kind):
 real_market(monkeypatch,tmp_path)
 nodes={'old_cap':20000,'below_cap':DEEP_NODE_LIMIT-1,'at_cap':DEEP_NODE_LIMIT}[count_kind]
 monkeypatch.setattr(q,'run_model',lambda *a:dict(optimal=False,infeasible=False,reason='time_or_node_limit',
  stages=[dict(stage=0,kind='priority',indices=[0],solver_status=1,solver_message='controlled stop',node_count=nodes)]))
 result=call(deep_optimization=True,priorities={'sprint_speed_modifier':10})
 assert result and result.search_status['state']=='FEASIBLE'
 assert result.search_status['reason']==('node_limit' if count_kind=='at_cap' else 'time_or_node_limit')
 assert result.search_status['worker_reason']=='time_or_node_limit'


def test_deep_cancel_reports_selected_cap_without_worker(monkeypatch,tmp_path):
 real_market(monkeypatch,tmp_path);event=threading.Event();event.set()
 monkeypatch.setattr(q,'run_model',lambda *a:pytest.fail('Cancelled'))
 result=call(deep_optimization=True,cancel_event=event)
 assert not result and result.search_status['reason']=='cancelled'
 assert result.search_status['limits']['nodes']==DEEP_NODE_LIMIT


@pytest.mark.parametrize('deep',[False,True])
def test_api_flag_to_actual_model_and_native_options(monkeypatch,tmp_path,deep):
 from app import database
 from fastapi.testclient import TestClient
 import importlib
 real_market(monkeypatch,tmp_path);monkeypatch.setattr(database,'init_db',lambda:None)
 main=importlib.import_module('app.main');monkeypatch.setattr(main.Config,'DATA_DIR',str(tmp_path/'api-data'))
 monkeypatch.setattr(main,'MAX_QUALITY_RELEASED',True)
 monkeypatch.setattr(main,'check_auction_freshness',lambda *a:dict(is_fresh=True,last_collected=None,age_hours=0))
 monkeypatch.setattr(main,'_start_auction_refresh',lambda:pytest.fail('No refresh'))
 seen=[]
 def run(data,*args):
  seen.append(data['node_limit']);return fake_worker(data,*args)
 monkeypatch.setattr(q,'run_model',run)
 response=TestClient(main.app).get('/api/builds',params=dict(budget=10,tolerance=0,max_quality='true',
  deep_optimization=str(deep).lower(),conservative='true',allow_duplicates='true',priorities='{"sprint_speed_modifier":10}'))
 assert response.status_code==200
 expected=DEEP_NODE_LIMIT if deep else 20000
 assert seen==[expected] and response.json()['search_status']['limits']['nodes']==expected
 assert response.json()['search_status']['state']=='OPTIMAL'

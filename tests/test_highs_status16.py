"""Limits are termination causes, never standalone feasibility certificates."""
import importlib
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
from app import _max_quality_worker as w, _max_quality as q, builder as b
from tests.test_deep_quality import tiny
from tests.test_feasibility_fallback import market
from tests.test_max_quality import call


def response(cost, *, kind='optimal', incumbent=True, nodes=None):
    native={'optimal':7,'infeasible':8,'node':16,'solution':16,'time':13,'error':4,'model_error':2}[kind]
    status={'optimal':0,'infeasible':2,'node':4,'solution':4,'time':1,'error':4,'model_error':2}[kind]
    x=np.ones(len(cost)) if incumbent else None
    return SimpleNamespace(status=status,message=f'controlled (HiGHS Status {native}: controlled termination)',
        x=x,mip_dual_bound=float(cost@x) if x is not None else None,
        mip_gap=0 if x is not None else None,mip_node_count=nodes)


@pytest.mark.parametrize('kind',['node','solution','time','error'])
@pytest.mark.parametrize('incumbent',[False,True])
def test_limit_and_error_do_not_prove_optimality(tmp_path,kind,incumbent):
    model=dict(tiny(),seconds=10)
    result=w.solve(model,tmp_path/'result.json',solver=lambda cost,**kw:response(cost,kind=kind,incumbent=incumbent,nodes=model['node_limit'] if kind=='node' else 4))
    assert not result['optimal'] and not result['infeasible']
    assert ('indices' in result['stages'][0])==incumbent
    assert result['reason']=={'node':'node_limit','solution':'solution_limit','time':'time_limit','error':'solver_error'}[kind]
    assert result['stages'][0]['solver_status']==(1 if kind=='time' else 4)


def test_optimal_and_infeasible_have_different_certificates(tmp_path):
    optimal=w.solve(dict(tiny(),seconds=10),tmp_path/'opt.json',solver=lambda cost,**kw:response(cost))
    assert optimal['optimal'] and len(optimal['stages'])==2
    infeasible=w.solve(dict(tiny(),seconds=10),tmp_path/'inf.json',solver=lambda cost,**kw:response(cost,kind='infeasible',incumbent=False))
    assert infeasible['infeasible'] and not infeasible['optimal']
    error=w.solve(dict(tiny(),seconds=10),tmp_path/'err.json',solver=lambda cost,**kw:response(cost,kind='model_error',incumbent=False))
    assert error['reason']=='solver_error' and not error['infeasible']


@pytest.mark.parametrize('value',[None,'not numeric','0.0',float('nan'),float('inf')])
@pytest.mark.parametrize('field',['mip_dual_bound','mip_gap','mip_node_count'])
def test_optional_metadata_is_not_fabricated_or_crashing(tmp_path,value,field):
    def solver(cost,**kw):
        r=response(cost);setattr(r,field,value);return r
    result=w.solve(dict(tiny(),seconds=10),tmp_path/'metadata.json',solver=solver)
    key={'mip_dual_bound':'bound','mip_gap':'gap','mip_node_count':'node_count'}[field]
    assert result['stages'][0][key] is None
    if field=='mip_dual_bound':assert not result['optimal']
    # Finite matching primal/dual plus status0 can certify without optional gap/count.
    else:assert result['optimal']
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('primal',[None,[float('nan')],[.5],[2.],['bad']])
def test_solution_message_does_not_invent_an_incumbent(tmp_path,primal):
    def solver(cost,**kw):
        r=response(cost,kind='solution',incumbent=False);r.x=primal;return r
    result=w.solve(dict(tiny(),seconds=10),tmp_path/'bad-primal.json',solver=solver)
    assert 'indices' not in result['stages'][0] and not result['optimal'] and not result['infeasible']
    assert result['reason']=='solution_limit'


@pytest.mark.parametrize('incumbent',[False,True])
@pytest.mark.parametrize('stock_only',[False,True])
@pytest.mark.parametrize('deep',[False,True])
def test_first_proof_and_best_validated_witness_survive_cost_limit(monkeypatch,tmp_path,incumbent,stock_only,deep):
    market(monkeypatch,tmp_path,[({'speed_modifier':5},5),({'speed_modifier':5},3)],slots=1)
    def run(model,*args):
        calls=[]
        def solver(cost,**kw):
            phase=len(calls);calls.append(phase)
            x=np.zeros(model['n']);x[np.argmax(model['prices']) if phase==0 else np.argmin(model['prices'])]=1
            return SimpleNamespace(status=0 if phase==0 else 4,message='ok' if phase==0 else '(HiGHS Status 16: model_status is Solution limit reached; primal_status is None)',
                x=x if phase==0 or incumbent else None,mip_dual_bound=float(cost@x) if phase==0 else None,mip_gap=0 if phase==0 else None,mip_node_count=1 if phase==0 else None)
        return w.solve(dict(model,seconds=10),tmp_path/'stages.json',solver=solver)
    monkeypatch.setattr(q,'run_model',run)
    result=call(stock_only=stock_only,deep_optimization=deep)
    assert result and result.search_status['state']=='FEASIBLE' and not result.search_status['optimality_proven']
    assert result[0].total_price==(3 if incumbent else 5)
    info=result.search_status['models'][0]
    assert info['priority_optimality_proven'] and not info['cost_optimality_proven']
    assert result.search_status['reason']==result.search_status['worker_reason']=='solution_limit'
    assert 'минимальная стоимость не доказана' in result.search_status['message']
    assert b._final_valid(result[0],10,{},{},True,False,True)
    assert all(o.ptn==0 for o in result[0].artifacts)


@pytest.mark.parametrize('invalid',['price','expiry','game'])
def test_limit_incumbent_still_needs_physical_and_game_validation(monkeypatch,tmp_path,invalid):
    _,path,_=market(monkeypatch,tmp_path,[({'speed_modifier':5},2)],slots=1)
    def run(model,*args):
        if invalid!='game':
            with sqlite3.connect(path) as c:
                c.execute('update auction_lots set '+('buyout_price=7' if invalid=='price' else "end_time='2000-01-01T00:00:00Z'"))
        return dict(stages=[dict(stage=0,kind='priority',indices=[0],solver_status=4,solver_message='(HiGHS Status 16: Solution limit reached)')],optimal=False,infeasible=False,reason='solution_limit')
    monkeypatch.setattr(q,'run_model',run)
    result=call(stock_only=True,min_thresholds={'speed_modifier':6} if invalid=='game' else {})
    assert not result and result.search_status['state']=='UNKNOWN' and not result.search_status['optimality_proven']


def test_unknown_code4_is_not_a_solution_limit_by_words_alone(tmp_path):
    def solver(cost,**kw):
        r=response(cost,kind='error',incumbent=False);r.message='solver failed; solution limit mentioned in unrelated text';return r
    result=w.solve(dict(tiny(),seconds=10),tmp_path/'error.json',solver=solver)
    assert result['reason']=='solver_error' and result['stages'][0]['highs_status'] is None


def test_real_scipy_status16_is_a_limit_with_and_without_solution():
    # Short actual installed SciPy/HiGHS solves, not fake OptimizeResults.
    from scipy.optimize import milp,Bounds,LinearConstraint
    rng=np.random.default_rng(16371);n=80;A=rng.integers(0,100,size=(15,n));rhs=A.sum(axis=1)*.3;c=-rng.integers(1,100,size=n)
    for limit in [0,1]:
        r=milp(c,integrality=np.ones(n),bounds=Bounds(0,1),constraints=LinearConstraint(A,-np.inf,rhs),options=dict(presolve=False,node_limit=limit,time_limit=2,mip_rel_gap=0))
        assert r.status==4 and 'HiGHS Status 16:' in r.message
        nodes=w._finite_number(getattr(r,'mip_node_count',None))
        reason,code=w._stop_reason(r.status,r.message,nodes,limit)
        assert code==16 and reason in ('node_limit','solution_limit')
        assert (r.x is not None)==bool(limit)
        if r.x is not None:
            x=np.rint(r.x)
            assert np.max(abs(r.x-x))<=1e-5 and np.all(A@x<=rhs)


def test_api_retains_stage0_and_explains_cost_is_unproved(monkeypatch,tmp_path):
    from fastapi.testclient import TestClient
    from app import database
    market(monkeypatch,tmp_path,[({'speed_modifier':5},2)],slots=1)
    monkeypatch.setattr(database,'init_db',lambda:None);main=importlib.import_module('app.main')
    monkeypatch.setattr(main.Config,'DATA_DIR',str(tmp_path/'api-data'))
    monkeypatch.setattr(main,'MAX_QUALITY_RELEASED',True)
    monkeypatch.setattr(main,'check_auction_freshness',lambda *a:dict(is_fresh=True,last_collected=None,age_hours=0))
    monkeypatch.setattr(main,'_start_auction_refresh',lambda:pytest.fail('No refresh'))
    def run(model,*args):
        phase=[]
        def solver(cost,**kw):
            kind='optimal' if not phase else 'solution';phase.append(kind)
            return response(cost,kind=kind,incumbent=kind=='optimal')
        return w.solve(dict(model,seconds=10),tmp_path/'api-stage.json',solver=solver)
    monkeypatch.setattr(q,'run_model',run)
    response_api=TestClient(main.app).get('/api/builds',params=dict(budget=10,tolerance=0,priorities='{"speed_modifier":10}',max_quality='true',deep_optimization='true',stock_only='true'))
    assert response_api.status_code==200
    data=response_api.json();status=data['search_status']
    assert data['builds_found']==1 and status['state']=='FEASIBLE' and status['reason']=='solution_limit'
    assert status['validation_reason']=='passed' and 'минимальная стоимость не доказана' in status['message']
    assert status['models'][0]['stages'][1]['solver_status']==4
    assert status['models'][0]['stages'][1]['highs_status']==16

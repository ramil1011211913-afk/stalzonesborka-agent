"""Opt-in EHP primary contract, independent of the legacy cost contract."""
import itertools,json,time,sys,subprocess,threading
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import pytest
from scipy.optimize import milp,OptimizeResult
from app import _max_quality_worker as worker,_max_quality_model as mm,_max_quality as q
from app import _ehp_primary_only as po,_ehp_lp_bounds as lp,builder as b
from app._objective import Objective
from app._survivability import values
from tests.test_max_quality import option,call
from tests.test_objective_integration import container
from tests.test_feasibility_fallback import market

def model(seed=0):
    rng=np.random.default_rng(seed)
    opts=[option(i,dict(health_bonus=float(rng.integers(-15,30)),bullet_dmg_factor=float(rng.integers(-8,35))),i+1) for i in range(8)]
    m=mm.make_model(opts,container(slots=3),Objective.create({'effective_health':10}),20,{},{},True,True)
    m.update(primary_only=True,seconds=5,node_limit=50000,time_policy='deep_remaining',model_seconds=5,stage_reserve_seconds=15)
    return m

@pytest.mark.parametrize('seed',range(12))
def test_primary_exhaustive_and_no_cost_or_lock(seed,tmp_path):
    m=model(seed);calls=[]
    def record(cost,**kw):calls.append((cost.copy(),dict(kw,options=dict(kw['options']))));return milp(cost,**kw)
    r=worker.solve(m,tmp_path/'r.json',solver=record)
    def score(ids):return values({k:sum(m['metric_inputs'][i].get(k,0) for i in ids) for k in ('health_bonus','bullet_dmg_factor')})['effective_health']
    expected=max(score(ids) for ids in itertools.combinations(range(8),3) if sum(m['prices'][i] for i in ids)<=20)
    assert max(score(s['indices']) for s in r['stages'] if 'indices' in s)==expected
    assert [s['kind'] for s in r['stages']]==['primary_seed','priority']
    assert len(calls)==2 and calls[-1][1]['constraints'].A.shape[0]==len(m['rows'])
    assert not r['optimal'] and not r['cost_optimality_proven'] and not r['cost_optimization_requested']
    assert calls[-1][1]['options']['node_limit']+r['primary_seed_node_charge']<=50000

@pytest.mark.parametrize('rows,budget,expected',[
    [([{'bullet_dmg_factor':100},{'bullet_dmg_factor':99}],[9,1]),10,200],
    [([{'bullet_dmg_factor':101},{'bullet_dmg_factor':100}],[1,10]),10,201],
    [([{'bullet_dmg_factor':100},{'bullet_dmg_factor':200}],[10,11]),10,200],
    [([{'bullet_dmg_factor':100},{'bullet_dmg_factor':100}],[9,1]),10,200],
])
def test_budget_and_primary_beats_price(rows,budget,expected,tmp_path):
    stats,prices=rows;opts=[option(i,s,p) for i,(s,p) in enumerate(zip(stats,prices))]
    m=mm.make_model(opts,container(slots=1),Objective.create({'effective_health':10}),budget,{},{},True,True)
    m.update(primary_only=True,seconds=5,time_policy='deep_remaining',node_limit=50000)
    r=worker.solve(m,tmp_path/'r.json');best=max(s['derived_metrics']['effective_health'] for s in r['stages'] if 'indices' in s)
    assert best==expected and not r['cost_optimality_proven']

def test_strict_concordant_primary_numerical_proof(tmp_path):
    opts=[option(1,{'health_bonus':25,'bullet_dmg_factor':40},10)]
    m=mm.make_model(opts,container(slots=1),Objective.create({'effective_health':10},{'effective_health':10}),10,{},{},True,True)
    m.update(primary_only=True,seconds=5,time_policy='deep_remaining',node_limit=50000)
    r=worker.solve(m,tmp_path/'r.json')
    assert r['priority_optimality_proven'] and r['stages'][-1]['derived_metrics']['effective_health']==175
    assert not r['optimal'] and not r['cost_optimality_proven']

@pytest.mark.parametrize('conservative',[False,True])
@pytest.mark.parametrize('stock',[False,True])
def test_real_parent_opt_in_default_and_minmax(monkeypatch,tmp_path,conservative,stock):
    market(monkeypatch,tmp_path,[({'health_bonus':10,'bullet_dmg_factor':50,'regeneration_bonus':3},4),({'health_bonus':5,'regeneration_bonus':5},2),({'health_bonus':1,'regeneration_bonus':2},1)],slots=2)
    r=call(priorities={'effective_health':10},conservative=conservative,stock_only=stock,deep_optimization=True,primary_only=True,min_thresholds={'health_bonus':0},max_thresholds={'health_bonus':100})
    assert r and r.search_status['state']=='FEASIBLE' and r.search_status['primary_only']
    assert not r.search_status['cost_optimality_proven'] and 'не оптимизировалась' in r.search_status['message']
    assert len({o.auction_lot_id for o in r[0].artifacts})==2 and all(o.ptn==0 for o in r[0].artifacts)
    assert b._final_valid(r[0],10,{'health_bonus':0},{'health_bonus':100},True,False,True)
    assert r[0]._search_request['primary_only']
    d=call(priorities={'effective_health':10},conservative=conservative,stock_only=stock,deep_optimization=True)
    assert d and d.search_status['state']=='OPTIMAL' and not d.search_status.get('primary_only')
    assert d.search_status['models'][0]['cost_optimality_proven']

@pytest.mark.parametrize('settings',[{},dict(max_quality=True),dict(max_quality=True,deep_optimization=True,priorities={'healing_ehp_per_second':10}),dict(max_quality=True,deep_optimization=True,priorities={'effective_health':10,'health_bonus':1}),dict(max_quality=True,deep_optimization=True,min_budget=True,priorities={'effective_health':10})])
def test_API_rejects_unsupported_before_refresh(monkeypatch,settings):
    from fastapi.testclient import TestClient
    from app import main
    monkeypatch.setattr(main,'_start_auction_refresh',lambda:pytest.fail('No refresh for invalid API'))
    params=dict(primary_only=True,priorities=json.dumps({'effective_health':10}))
    params.update({k:json.dumps(v) if k=='priorities' else v for k,v in settings.items()})
    for route in ['/api/builds','/api/export-lunarzone']:
        assert TestClient(main.app).get(route,params=params).status_code==422

@pytest.mark.parametrize('owned',[False,True])
@pytest.mark.parametrize('conservative',[False,True])
@pytest.mark.parametrize('stock',[False,True])
def test_actual_API_and_export(monkeypatch,tmp_path,owned,conservative,stock):
    from fastapi.testclient import TestClient
    from app import main
    from tests.test_full_quality_candidates import sql_market
    from app import lunar_bridge
    _,conn=sql_market(monkeypatch,tmp_path,[({'health_bonus':25,'bullet_dmg_factor':40},4),({'health_bonus':5},2),({'health_bonus':1},1)],slots=2)
    monkeypatch.setattr(lunar_bridge,'get_connection',conn,raising=False)
    from tests.test_stock_only import api
    _,client=api(monkeypatch,tmp_path)
    params=dict(budget=10,tolerance=0,priorities=json.dumps({'effective_health':10}),primary_only=True,max_quality=True,deep_optimization=True,allow_duplicates=True,stock_only=stock,conservative=conservative)
    if owned:params['owned']=json.dumps([dict(item_id='1',quality=0,ptn=0,art_percent=100,instance_id='owned-one')])
    for route in ['/api/builds','/api/export-lunarzone']:
        res=client.get(route,params=params);assert res.status_code==200,res.text
        data=res.json();status=data['search_status'] if route=='/api/builds' else data['lunarzone_json']['_optimizer_meta']['search_status']
        assert status['primary_only'] and not status['cost_optimality_proven']
        assert not any(s['kind']=='cost' for m in status['models'] for s in m['stages'])

@pytest.mark.parametrize('change',['price','bonus','expired','delete'])
def test_physical_revalidation(monkeypatch,tmp_path,change):
    import sqlite3
    _,path,_=market(monkeypatch,tmp_path,[({'health_bonus':10,'bullet_dmg_factor':50},4),({'health_bonus':5},2)],slots=2)
    original=q.run_model
    def run(*a,**kw):
        result=original(*a,**kw)
        with sqlite3.connect(path) as c:
            if change=='delete':c.execute('DELETE FROM auction_lots')
            elif change=='price':c.execute('UPDATE auction_lots SET buyout_price=999')
            elif change=='bonus':c.execute("UPDATE auction_lots SET bonus_properties='[\"UNKNOWN\"]'")
            else:c.execute("UPDATE auction_lots SET end_time='2000-01-01T00:00:00Z'")
        return result
    monkeypatch.setattr(q,'run_model',run)
    r=call(priorities={'effective_health':10},deep_optimization=True,primary_only=True)
    assert not r and r.search_status['state']=='UNKNOWN' and not r.search_status['primary_optimality_proven']

def test_incompatible_LP_solver_falls_back_without_cost(monkeypatch,tmp_path):
    import scipy
    monkeypatch.setattr(scipy,'__version__','unsupported')
    r=worker.solve(model(),tmp_path/'r.json')
    assert not r['lp_bounds']['applied'] and any('indices' in s for s in r['stages'])
    assert not any(s['kind']=='cost' for s in r['stages'])

@pytest.mark.parametrize('status,msg',[(1,'HiGHS Status 13: Time limit'),(4,'HiGHS Status 16: Solution limit'),(2,'HiGHS Status 8: Infeasible'),(4,'HiGHS Status 4: Error')])
def test_lost_main_incumbent_keeps_checkpoint(monkeypatch,tmp_path,status,msg):
    calls=[]
    def limited(cost,**kw):
        calls.append(1)
        if len(calls)==1:return milp(cost,**kw)
        return OptimizeResult(status=status,message=msg,x=None,mip_dual_bound=np.nan,mip_gap=None,mip_node_count=None)
    r=worker.solve(model(),tmp_path/'r.json',solver=limited)
    assert not r['optimal'] and not r['infeasible'] and not r['priority_optimality_proven']
    assert any('indices' in s for s in r['stages'])

def test_LP_certificates_are_uniform_fraction_bounds(tmp_path):
    m=model();t=lp.transform(m,time.monotonic()+2)
    for ids in itertools.combinations(range(8),3):
        if sum(m['prices'][i] for i in ids)>20:continue
        for p in t['bound_proofs']:
            total=sum((F(float(m['metric_inputs'][i].get(p['key'],0))) for i in ids),F(0))
            assert F(p['lower_exact'])<=total<=F(p['upper_exact'])

def test_seven_old_lock_disagreements_and_counter17(tmp_path):
    # Read existing research only; no regenerating historical certificates.
    root=next(p for p in Path(__file__).resolve().parents if (p/'data/validation/survivability-engine-v2-5-20261020').is_dir())
    old=root/'data/validation/survivability-engine-v2-5-20261020'
    cases=json.loads((old/'family-and-cost-lock-audit.json').read_text())['original_secondary_exact_lock_exclusions']
    assert len(cases)==7
    for case in cases:
        m=json.loads((old/'experiments'/f"probe-{case['case']}-original-model.json").read_text())
        if m['objective_keys']==['effective_health']:
            m.update(primary_only=True,time_policy='deep_remaining')
            r=worker.solve(m,tmp_path/'r.json');assert not r['cost_optimality_proven'] and len(r['stages'])==2
        else:
            with pytest.raises(ValueError):worker.solve(dict(m,primary_only=True,time_policy='deep_remaining'),tmp_path/'r.json')
        legacy_model=dict(m,primary_only=False);legacy_model.pop('time_policy',None)
        legacy=worker.solve(legacy_model,tmp_path/'legacy.json')
        assert legacy['optimal'] and legacy['cost_optimality_proven']
    m=json.loads((old/'experiments/probe-1-original-model.json').read_text())
    r=worker.solve(m,tmp_path/'counter17.json')
    assert r['optimal'] and r['stages'][-1]['indices']==[2,5,7]
    assert sum(m['prices'][i] for i in r['stages'][-1]['indices'])==17

@pytest.mark.parametrize('kind',['memory','badalloc','abort','commit','cancel','timeout','watchdog'])
def test_primary_real_child_checkpoint_recovery(monkeypatch,tmp_path,kind):
    from app import _max_quality_process as process
    m=model();m['memory_bytes']=512*1024**2;paths=[];children=[];observed=threading.Event();cancel=threading.Event()
    script='''
import os,time,ctypes
import scipy.optimize as opt
from app import _max_quality_worker as w
kind=KIND
real=opt.milp
calls=0
def fault(*a,**kw):
 global calls
 calls+=1
 if calls==1:return real(*a,**kw)
 if kind=='memory':raise MemoryError('controlled primary allocation failure')
 if kind=='badalloc':raise RuntimeError('std::bad_alloc')
 if kind=='abort':os._exit(91)
 if kind=='commit':
  k=ctypes.WinDLL('kernel32',use_last_error=True)
  k.VirtualAlloc.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_ulong,ctypes.c_ulong];k.VirtualAlloc.restype=ctypes.c_void_p
  p=k.VirtualAlloc(None,600*1024**2,0x3000,4)
  if p:raise AssertionError('Job cap did not reject allocation')
  raise MemoryError('Windows Job rejected allocation')
 time.sleep(10)
opt.milp=fault
w.main()
'''.replace('KIND',repr(kind))
    original=subprocess.Popen;resident=process.resident_bytes
    def launch(args,**kw):
        if 'app._max_quality_worker' in args:paths.append(Path(args[-1]));args=[args[0],'-c',script,*args[-2:]]
        child=original(args,**kw);children.append(child);return child
    monkeypatch.setattr(subprocess,'Popen',launch)
    def observe():
        end=time.monotonic()+6
        while time.monotonic()<end:
            try:
                d=json.loads(paths[0].read_text())
                if any('indices' in s for s in d['stages']):
                    observed.set()
                    if kind=='cancel':cancel.set()
                    return
            except (IndexError,OSError,ValueError):pass
            time.sleep(.005)
    watch=kind in ('cancel','timeout','watchdog');thread=threading.Thread(target=observe) if watch else None
    if thread:thread.start()
    deadline=time.monotonic()+6
    if kind=='timeout':
        clock=time.monotonic
        class Clock:
            @staticmethod
            def monotonic():return deadline+1 if observed.is_set() else clock()
            sleep=staticmethod(time.sleep)
        monkeypatch.setattr(process,'time',Clock)
    if kind=='watchdog':monkeypatch.setattr(process,'resident_bytes',lambda pid:m['memory_bytes']+1 if observed.is_set() else resident(pid))
    try:r=process.run_model(m,deadline,cancel)
    finally:
        if thread:thread.join(timeout=6)
    reasons={'memory':'memory_limit','badalloc':'memory_limit','commit':'memory_limit','abort':'worker_failed','cancel':'cancelled','timeout':'request_time_limit','watchdog':'memory_limit'}
    assert r['reason']==reasons[kind],r
    assert any('indices' in s and s['independent_witness_valid'] for s in r['stages'])
    assert not r['optimal'] and not r['infeasible'] and not r['priority_optimality_proven']
    assert r['memory_limit_kind']=='windows_process_commit' and all(c.poll() is not None for c in children)

@pytest.mark.parametrize('changes',[1,2])
def test_new_route_refresh_retry_and_shared_node_budget(monkeypatch,tmp_path,changes):
    import sqlite3
    from tests.test_full_quality_candidates import sql_market
    path,_=sql_market(monkeypatch,tmp_path,[(dict(health_bonus=i,bullet_dmg_factor=i),1) for i in range(8)],slots=2)
    with sqlite3.connect(path.parent/'app_state.sqlite3') as c:
        c.execute('CREATE TABLE refresh_jobs(name TEXT,generation INTEGER,running INTEGER)');c.execute("INSERT INTO refresh_jobs VALUES('auction:EU',0,0)")
    original=q.run_model;calls=[]
    def run(m,*a,**kw):
        if calls:assert m['node_limit']==50000-sum(s['node_count'] if s.get('node_count') is not None else s['node_limit'] for s in calls[0]['stages'])
        result=original(m,*a,**kw);calls.append(result)
        if len(calls)==1:
            next(s for s in result['stages'] if s['kind']=='priority')['node_count']=30000
        if len(calls)<=changes:
            with sqlite3.connect(path) as c:c.execute('UPDATE auction_lots SET id=id+100,buyout_price=buyout_price+1')
            with sqlite3.connect(path.parent/'app_state.sqlite3') as c:c.execute('UPDATE refresh_jobs SET generation=generation+1')
        return result
    monkeypatch.setattr(q,'run_model',run)
    r=call(priorities={'effective_health':10},deep_optimization=True,primary_only=True)
    assert len(calls)==2 and r.search_status['retry_count']==1
    if changes==1:assert r and all(o.auction_lot_id>=101 for o in r[0].artifacts)
    else:assert not r and r.search_status['state']=='UNKNOWN'

@pytest.mark.parametrize('conservative',[False,True])
@pytest.mark.parametrize('stock',[False,True])
def test_owned_only_new_route(monkeypatch,tmp_path,conservative,stock):
    import sqlite3
    from tests.test_full_quality_candidates import sql_market
    path,_=sql_market(monkeypatch,tmp_path,[(dict(health_bonus=i,bullet_dmg_factor=i),1) for i in range(8)],slots=2)
    with sqlite3.connect(path) as c:c.execute('DELETE FROM auction_lots')
    owned=[dict(item_id=str(i+1),quality=0,ptn=0,art_percent=100,instance_id=str(i)) for i in range(8)]
    r=call(priorities={'effective_health':10},deep_optimization=True,primary_only=True,owned_artifacts=owned,stock_only=stock,conservative=conservative)
    assert r and r[0].total_price==0 and all(o.is_owned for o in r[0].artifacts)
    assert not any(s['kind']=='cost' for m in r.search_status['models'] for s in m['stages'])

def test_default_false_and_inclusive_thresholds(monkeypatch,tmp_path):
    market(monkeypatch,tmp_path,[({'health_bonus':0,'bullet_dmg_factor':40,'biological_accumulation':10},10)],slots=1)
    a=call(priorities={'effective_health':10},deep_optimization=True)
    d=call(priorities={'effective_health':10},deep_optimization=True,primary_only=False)
    assert [x.to_dict() for x in a]==[x.to_dict() for x in d]
    r=call(priorities={'effective_health':10},deep_optimization=True,primary_only=True,min_thresholds={'effective_health':140},max_thresholds={'effective_health':140})
    assert r and r[0].combined_stats['effective_health']==140 and r[0].total_price==10
    assert r[0].combined_stats['biological_accumulation']==10

def test_strict_budget_rejects_tolerance_before_market(monkeypatch):
    from tests.test_stock_only import api
    with pytest.raises(ValueError):q.build_max_quality_sets(10,deep_optimization=True,primary_only=True,priorities={'effective_health':10},tolerance_pct=8)

def test_primary_only_cannot_be_ignored_by_meta_cache(monkeypatch,tmp_path):
    from tests.test_stock_only import api
    from app import meta_optimizer
    _,client=api(monkeypatch,tmp_path)
    monkeypatch.setattr(meta_optimizer,'get_cached_meta',lambda *a:pytest.fail('No cache fallback for incompatible mode'))
    for route in ['/api/meta-builds','/api/budget-sweep']:
        assert client.get(route,params={'primary_only':'true'}).status_code==422

@pytest.mark.parametrize('missing',[False,True])
def test_request_budget_shared_across_profiles(monkeypatch,tmp_path,missing):
    from dataclasses import replace
    market(monkeypatch,tmp_path,[({'health_bonus':10,'bullet_dmg_factor':50},4),({'health_bonus':5},2)],slots=2)
    c=container(slots=2);other=replace(c,item_id='alternative')
    monkeypatch.setattr(b,'load_container_options',lambda region:[c,other])
    original=q.run_model;calls=[]
    def run(m,*a,**kw):
        if calls:assert m['node_limit']<20001
        r=original(m,*a,**kw);calls.append(r)
        if len(calls)==1:
            if missing:
                r['stages']=[s for s in r['stages'] if s['kind']!='priority']
                r.update(priority_optimality_proven=False,reason='worker_failed')
            else:next(s for s in r['stages'] if s['kind']=='priority')['node_count']=30000
        return r
    monkeypatch.setattr(q,'run_model',run)
    r=call(priorities={'effective_health':10},deep_optimization=True,primary_only=True)
    assert r and not r.search_status['cost_optimality_proven']
    assert len(calls)==(1 if missing else 2)

@pytest.mark.parametrize('kind',['memory','badalloc'])
def test_LP_preprocessing_failure_recovers_seed(monkeypatch,tmp_path,kind):
    from app import _max_quality_process as process
    m=model();m['memory_bytes']=512*1024**2;original=subprocess.Popen;children=[]
    script='''
from app import _ehp_lp_bounds as lp,_max_quality_worker as w
def fail(*a,**k):
 if KIND=='memory':raise MemoryError('LP allocation failure')
 raise RuntimeError('std::bad_alloc')
lp.transform=fail
w.main()
'''.replace('KIND',repr(kind))
    def launch(args,**kw):
        if 'app._max_quality_worker' in args:args=[args[0],'-c',script,*args[-2:]]
        p=original(args,**kw);children.append(p);return p
    monkeypatch.setattr(subprocess,'Popen',launch)
    r=process.run_model(m,time.monotonic()+6)
    assert r['reason']=='memory_limit' and any('indices' in s for s in r['stages'])
    assert not r['optimal'] and not r['priority_optimality_proven'] and all(p.poll() is not None for p in children)

def test_invalid_reported_auxiliary_cannot_prove_primary(tmp_path):
    opts=[option(1,{'bullet_dmg_factor':0},1),option(2,{'bullet_dmg_factor':100},1)]
    m=mm.make_model(opts,container(slots=1),Objective.create({'effective_health':10},{'effective_health':10}),10,{},{},True,True)
    m.update(primary_only=True,seconds=5,time_policy='deep_remaining',node_limit=50000)
    calls=[]
    def fake(cost,**kw):
        calls.append(1)
        if len(calls)==1:return OptimizeResult(status=1,message='HiGHS Status 13: Time',x=None,mip_node_count=0)
        x=np.zeros(len(cost));x[0]=1
        return OptimizeResult(status=0,message='HiGHS Status 7: Optimal',x=x,mip_node_count=0,mip_dual_bound=-100.,mip_gap=0.)
    r=worker.solve(m,tmp_path/'r.json',solver=fake)
    assert any('indices' in s for s in r['stages']) and not r['priority_optimality_proven']

def test_uniform_native_family_guard_is_not_witness_only():
    m=model();assert not po.exact_game_family(m)
    m['metric_inputs']=[{'health_bonus':25.,'bullet_dmg_factor':40.} for _ in range(m['n'])]
    assert po.exact_game_family(m)
    m['metric_inputs'][0]['health_bonus']=25.01
    assert not po.exact_game_family(m)

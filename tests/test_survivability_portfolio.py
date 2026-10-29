import itertools,json,time,random
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import milp
from app import _survivability_portfolio as p,_max_quality_model as mm,_max_quality_worker as worker,builder as b
from app._objective import Objective
from tests.test_max_quality import option
from tests.test_objective_integration import container

def model(seed=0,n=16):
    r=random.Random(seed)
    opts=[option(i,dict(health_bonus=r.randint(-3,20),bullet_dmg_factor=r.randint(0,45),regeneration_bonus=r.randint(0,8),artefakt_heal=.5*(i%2),heal_efficiency=0.),r.randint(1,8)) for i in range(n)]
    m=mm.make_model(opts,container(slots=2),Objective.create({'healing_ehp_per_second':10}),20,{},{},True,True)
    return opts,dict(m,seconds=5,model_seconds=5,time_policy='deep_remaining',stage_reserve_seconds=.1,node_limit=50000)

def independent(m,ids):
    s={}
    for i in sorted(ids,key=lambda i:m['metric_item_ids'][i]):
        for k,v in m['metric_inputs'][i].items():s[k]=s.get(k,0.)+v
    E=(s.get('health_bonus',0)+100)/100*(s.get('bullet_dmg_factor',0)+100)
    R=(s.get('regeneration_bonus',0)+2.5)/5+s.get('artefakt_heal',0)*(1+s.get('heal_efficiency',0)/100)
    return 10*(E*R/100)

@pytest.mark.parametrize('seed',range(6))
def test_full_small_enumeration(seed,tmp_path):
    _,m=model(seed);expected=max((independent(m,ids),-sum(m['prices'][i] for i in ids)) for ids in itertools.combinations(range(16),2) if sum(m['prices'][i] for i in ids)<=20)
    result=worker.solve(m,tmp_path/'x.json');records=[s for s in result['stages'] if 'indices' in s]
    best=max((independent(m,s['indices']),-sum(m['prices'][i] for i in s['indices'])) for s in records)
    assert best[0]==expected[0]
    if result['optimal']:assert best==expected
    assert any(s['kind']=='portfolio_witness' for s in records)
    for s in records:
        assert len(s['indices'])==len(set(s['indices']))==2
    assert result['primary_seed_node_charge']<=5000

def test_v1_counterexample_keeps_cost17(tmp_path):
    opts=[option(i,dict(health_bonus=2*i-4,bullet_dmg_factor=7*(i%4),regeneration_bonus=2*(i%3),artefakt_heal=.2*(i%2),heal_efficiency=i-3),i+1) for i in range(8)]
    m=mm.make_model(opts,container(slots=3),Objective.create({'healing_ehp_per_second':10}),20,{},{},True,True)
    result=worker.solve(dict(m,seconds=5),tmp_path/'x.json')
    assert result['optimal'] and result['stages'][-1]['indices']==[2,5,7]
    assert sum(m['prices'][i] for i in result['stages'][-1]['indices'])==17

def test_standard_and_other_goals_not_enabled():
    _,m=model();assert p.enabled(m)
    assert not p.enabled(dict(m,time_policy='standard'))
    assert not p.enabled(dict(m,objective_keys=['effective_health']))
    assert not p.enabled(dict(m,objective_keys=['effective_health','healing_ehp_per_second']))
    assert not p.enabled(dict(m,n=8))

def test_main_and_cost_matrices_remain_byte_identical(tmp_path,monkeypatch):
    _,m=model();captures=[[],[]]
    def solver(which):
        def call(c,**kw):
            if len(c)==m['variables']:
                mat=kw['constraints'].A
                captures[which].append((c.tobytes(),mat.data.tobytes(),mat.indices.tobytes(),mat.indptr.tobytes(),kw['constraints'].lb.tobytes(),kw['constraints'].ub.tobytes()))
            return milp(c,**kw)
        return call
    with monkeypatch.context() as patch:
        patch.setattr(p,'enabled',lambda m:False)
        worker.solve(m,tmp_path/'old.json',solver=solver(0))
    worker.solve(m,tmp_path/'new.json',solver=solver(1))
    assert len(captures[0])==len(captures[1])==2 and captures[0]==captures[1]

def test_checkpoint_survives_main_memory_error(tmp_path):
    _,m=model();dest=tmp_path/'x.json'
    def solver(c,**kw):
        if len(c)==m['variables']:raise MemoryError('controlled synthetic main allocation')
        return milp(c,**kw)
    with pytest.raises(MemoryError):worker.solve(m,dest,solver=solver)
    state=json.loads(dest.read_text());assert not state['optimal']
    assert any(s.get('indices') and s['kind']=='portfolio_witness' for s in state['stages'])

def test_stronger_portfolio_prevents_false_primary_and_cost_proof(tmp_path):
    _,m=model();calls=[]
    def solver(c,**kw):
        result=milp(c,**kw)
        if len(c)==m['variables']:
            calls.append('main')
            # Deliberately inconsistent claimed optimum from a solver double.
            result.x[:m['n']]=0;result.x[0]=result.x[1]=1
            result.status=0;result.message='HiGHS Status 7: Optimal'
            result.mip_dual_bound=-p.native(m,[0,1])[1]
        return result
    state=worker.solve(m,tmp_path/'x.json',solver=solver)
    assert state['reason']=='portfolio_primary_not_proved'
    assert not state['optimal'] and not state['priority_optimality_proven'] and not state['cost_optimality_proven']
    assert calls==['main']

def test_actual_node_charge_is_not_truncated(tmp_path):
    _,m=model();seeds=[]
    def solver(c,**kw):
        result=milp(c,**kw)
        if len(c)==m['n']:
            if seeds:result.mip_node_count=1702
            seeds.append(result.mip_node_count)
        return result
    state=worker.solve(m,tmp_path/'x.json',solver=solver)
    assert state['primary_seed_node_charge']==sum(seeds)

@pytest.mark.parametrize('key,lo,hi',[('effective_health',160,250),('healing_percent_per_second',2,3.5),('healing_ehp_per_second',3,10)])
def test_portfolio_derived_min_max(key,lo,hi,tmp_path):
    opts,m=model();m=mm.make_model(opts,container(slots=2),Objective.create({'healing_ehp_per_second':10}),20,{key:lo},{key:hi},True,True)
    m.update(seconds=5,model_seconds=5,time_policy='deep_remaining',stage_reserve_seconds=.1,node_limit=50000)
    state=worker.solve(m,tmp_path/'x.json')
    records=[s for s in state['stages'] if 'indices' in s]
    assert records
    for s in records:
        stats,_=p.native(m,s['indices']);assert lo<=stats[key]<=hi

@pytest.mark.parametrize('conservative',[False,True])
@pytest.mark.parametrize('stock',[False,True])
@pytest.mark.parametrize('owned',[False,True])
def test_api_full_source_path(monkeypatch,tmp_path,conservative,stock,owned):
    from tests.test_full_quality_candidates import sql_market
    from tests.test_stock_only import api
    offers=[(dict(health_bonus=i%4,bullet_dmg_factor=(i*7)%30,regeneration_bonus=1+i%3),1+i) for i in range(16)]
    sql_market(monkeypatch,tmp_path,offers,slots=2)
    _,client=api(monkeypatch,tmp_path)
    owned_data=[dict(item_id='1',quality=0,ptn=0,art_percent=100,instance_id='owner')] if owned else []
    q=dict(budget=1000,tolerance=0,max_quality='true',deep_optimization='true',priorities='{"healing_ehp_per_second":10}',conservative=str(conservative).lower(),stock_only=str(stock).lower(),min_thresholds='{"health_bonus":0}',max_thresholds='{"effective_health":500}')
    if owned:q['owned']=json.dumps(owned_data)
    response=client.get('/api/builds',params=q)
    assert response.status_code==200,response.text
    data=response.json();assert data['builds'],data
    assert data['search_status']['state'] in ('FEASIBLE','OPTIMAL')
    assert any(s.get('kind')=='portfolio_witness' for m in data['search_status']['models'] for s in m.get('stages',[]))
    assert 'derived_stats' in data['builds'][0]
    assert data['owned_count']==int(owned)
    assert len({a['auction_lot_id'] for a in data['builds'][0]['artifacts'] if a.get('auction_lot_id') is not None})>=1

def test_owned_only_portfolio(monkeypatch,tmp_path):
    import sqlite3
    from tests.test_full_quality_candidates import sql_market,call
    path,_=sql_market(monkeypatch,tmp_path,[(dict(health_bonus=i,bullet_dmg_factor=2*i,regeneration_bonus=1+i%3),1) for i in range(16)],slots=2)
    with sqlite3.connect(path) as c:c.execute('DELETE FROM auction_lots')
    owned=[dict(item_id=str(i+1),quality=0,ptn=0,art_percent=100,instance_id=str(i)) for i in range(16)]
    result=call(priorities={'healing_ehp_per_second':10},deep_optimization=True,owned_artifacts=owned,stock_only=True)
    assert result and result[0].total_price==0 and all(o.is_owned for o in result[0].artifacts)
    assert len({o.physical_id for o in result[0].artifacts})==2
    assert any(s.get('kind')=='portfolio_witness' for m in result.search_status['models'] for s in m['stages'])

@pytest.mark.parametrize('changes',[1,2])
def test_portfolio_refresh_retry(monkeypatch,tmp_path,changes):
    import sqlite3
    from tests.test_full_quality_candidates import sql_market,call
    from app import _max_quality as q
    path,_=sql_market(monkeypatch,tmp_path,[(dict(health_bonus=i,regeneration_bonus=1+i%3),1) for i in range(16)],slots=2)
    with sqlite3.connect(path.parent/'app_state.sqlite3') as c:
        c.execute('CREATE TABLE refresh_jobs(name TEXT,generation INTEGER,running INTEGER)')
        c.execute("INSERT INTO refresh_jobs VALUES('auction:EU',0,0)")
    original=q.run_model;calls=[]
    def run(*a,**kw):
        state=original(*a,**kw);calls.append(state)
        assert any(s.get('kind')=='portfolio_witness' for s in state['stages'])
        if len(calls)<=changes:
            with sqlite3.connect(path) as c:c.execute('UPDATE auction_lots SET id=id+100,buyout_price=buyout_price+1')
            with sqlite3.connect(path.parent/'app_state.sqlite3') as c:c.execute('UPDATE refresh_jobs SET generation=generation+1')
        return state
    monkeypatch.setattr(q,'run_model',run)
    result=call(priorities={'healing_ehp_per_second':10},deep_optimization=True)
    assert len(calls)==2 and result.search_status['retry_count']==1
    if changes==1:assert result and all(o.auction_lot_id>=101 for o in result[0].artifacts)
    else:assert not result and result.search_status['state']=='UNKNOWN'

@pytest.mark.parametrize('kind',['cancel','timeout','memory'])
def test_real_worker_stop_after_portfolio_checkpoint(monkeypatch,tmp_path,kind):
    import threading,subprocess
    from app import _max_quality_process as process
    _,m=model();event=threading.Event();observed=threading.Event();children=[];paths=[]
    original=subprocess.Popen;resident=process.resident_bytes
    script="""
import time
from scipy.optimize import milp
from app import _max_quality_worker as w
native=w.solve
def run(m,o):
 def solver(c,**kw):
  if len(c)==m['variables']:time.sleep(2)
  return milp(c,**kw)
 return native(m,o,solver=solver)
w.solve=run
w.main()
"""
    def launch(args,**kw):
        if 'app._max_quality_worker' in args:
            paths.append(Path(args[-1]));args=[args[0],'-c',script,*args[-2:]]
        child=original(args,**kw);children.append(child);return child
    monkeypatch.setattr(subprocess,'Popen',launch)
    def observe():
        end=time.monotonic()+5
        while time.monotonic()<end:
            try:
                state=json.loads(paths[0].read_text())
                if any(s.get('kind')=='portfolio_witness' for s in state['stages']):
                    observed.set()
                    if kind=='cancel':event.set()
                    return
            except (IndexError,OSError,ValueError):pass
            time.sleep(.005)
    thread=threading.Thread(target=observe);thread.start()
    deadline=time.monotonic()+5
    if kind=='timeout':
        # Trigger the parent deadline deterministically AFTER the checkpoint,
        # rather than racing a cold Windows/SciPy import against 1.5 seconds.
        real_time=time.monotonic
        class ControlledClock:
            @staticmethod
            def monotonic():return deadline+1 if observed.is_set() else real_time()
            sleep=staticmethod(time.sleep)
        monkeypatch.setattr(process,'time',ControlledClock)
    if kind=='memory':monkeypatch.setattr(process,'resident_bytes',lambda pid:m['memory_bytes']+1 if observed.is_set() else resident(pid))
    try:result=process.run_model(m,deadline,event)
    finally:thread.join(timeout=5)
    assert observed.is_set() and not thread.is_alive()
    assert result['reason']=={'cancel':'cancelled','timeout':'request_time_limit','memory':'memory_limit'}[kind]
    assert not result['optimal'] and not result['infeasible']
    assert any(s.get('indices') for s in result['stages'] if s.get('kind')=='portfolio_witness')
    assert all(c.poll() is not None for c in children)

"""Native primary gate; original game/physical validators remain mandatory."""
import json,itertools,time,sys,subprocess,threading
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import milp
from app import _survivability_native as ns,_survivability_portfolio as portfolio
from app import _max_quality_worker as worker,_max_quality_model as mm,_max_quality_process as process
from app._objective import Objective
from tests.test_survivability_portfolio import model,independent
from tests.test_max_quality import option
from tests.test_objective_integration import container

def test_pinned_compatibility():
    assert ns.compatibility()==(True,'pinned_scipy_1.18.1_highs_1.12.0')

@pytest.mark.parametrize('field',['scipy','highs','binary','api'])
def test_version_and_api_mismatch_fallback(monkeypatch,tmp_path,field):
    import scipy
    from scipy.optimize._highspy import _core as h
    if field=='scipy':monkeypatch.setattr(scipy,'__version__','different')
    elif field=='highs':monkeypatch.setattr(h,'HIGHS_VERSION_MINOR',99)
    elif field=='binary':monkeypatch.setattr(ns,'CORE_SHA256','different')
    else:
        original=h._Highs
        monkeypatch.setattr(h,'_Highs',lambda:original()) # SciPy construction works, required class API is absent
    _,m=model();r=worker.solve(m,tmp_path/'r.json')
    assert not r['native_start']['compatible'] and not r['native_start']['used']
    assert any('indices' in s for s in r['stages']) and not r['infeasible']

@pytest.mark.parametrize('seed',range(6))
def test_native_full_enumeration(seed,tmp_path):
    _,m=model(seed);oracle=max((independent(m,ids),-sum(m['prices'][i] for i in ids)) for ids in itertools.combinations(range(16),2) if sum(m['prices'][i] for i in ids)<=20)
    r=worker.solve(m,tmp_path/'r.json');info=r['native_start']
    assert info['used'] and info['injection_status']==0 and info['variables']==m['variables']
    assert info['exact_violations']==[] and info['checkpoint_published']
    best=max((independent(m,s['indices']),-sum(m['prices'][i] for i in s['indices'])) for s in r['stages'] if 'indices' in s)
    assert best[0]==oracle[0]
    if r['optimal']:assert best==oracle
    assert r['primary_seed_node_charge']+r['stages'][-1].get('node_count',0)<=50000

def test_counterexample_active_native_price17(tmp_path):
    opts=[option(i,dict(health_bonus=2*i-4,bullet_dmg_factor=7*(i%4),regeneration_bonus=2*(i%3),artefakt_heal=.2*(i%2),heal_efficiency=i-3),i+1) for i in range(8)]
    opts += [option(i,{},21) for i in range(8,16)]
    m=mm.make_model(opts,container(slots=3),Objective.create({'healing_ehp_per_second':10}),20,{},{},True,True)
    m.update(seconds=5,model_seconds=5,time_policy='deep_remaining',stage_reserve_seconds=.1,node_limit=50000)
    r=worker.solve(m,tmp_path/'r.json')
    assert r['native_start']['used'] and r['optimal']
    assert r['stages'][-1]['indices']==[2,5,7]
    assert sum(m['prices'][i] for i in r['stages'][-1]['indices'])==17

def test_atomic_checkpoint_and_complete_native_rows(monkeypatch,tmp_path):
    _,m=model();dest=tmp_path/'r.json';actual=ns.native_solve
    def spy(cost,kw,start):
        stored=json.loads(dest.read_text())
        assert stored['native_start']['checkpoint_published'] and stored['native_start']['phase']=='ready'
        assert any(s.get('kind')=='portfolio_witness' and 'indices' in s for s in stored['stages'])
        assert len(start)==len(cost)==m['variables'] and np.isfinite(start).all()
        assert len(start)>m['n'] and np.all(start[:m['n']]==np.rint(start[:m['n']]))
        assert kw['options']['node_limit']==50000-stored['primary_seed_node_charge']
        assert 0<kw['options']['time_limit']<m['seconds']
        return actual(cost,kw,start)
    monkeypatch.setattr(ns,'native_solve',spy)
    result=worker.solve(m,dest);assert result['native_start']['used']

def test_rejected_start_falls_back_without_changing_game_thresholds(monkeypatch,tmp_path):
    _,m=model();real=ns.witness
    def reject(*a,**kw):
        x,proof=real(*a,**kw);proof['exact_violations']=[0];return x,proof
    monkeypatch.setattr(ns,'witness',reject)
    r=worker.solve(m,tmp_path/'r.json')
    assert not r['native_start']['used'] and r['native_start']['fallback']=='no_complete_certified_start'
    assert any('indices' in s for s in r['stages'])

def test_native_pre_run_api_failure_safe_fallback(monkeypatch,tmp_path):
    _,m=model()
    def unavailable(*a):raise ns.NativeUnavailable('controlled API setup failure')
    monkeypatch.setattr(ns,'native_solve',unavailable)
    r=worker.solve(m,tmp_path/'r.json')
    assert not r['native_start']['used'] and r['native_start']['phase']=='fallback'
    assert not r['infeasible']

@pytest.mark.parametrize('key,lo,hi',[('effective_health',160,250),('healing_percent_per_second',2,3.5),('healing_percent_per_second',2,5),('healing_ehp_per_second',3,10)])
def test_native_derived_limits(key,lo,hi,tmp_path):
    opts,_=model();m=mm.make_model(opts,container(slots=2),Objective.create({'healing_ehp_per_second':10}),20,{key:lo},{key:hi},True,True)
    m.update(seconds=5,model_seconds=5,time_policy='deep_remaining',stage_reserve_seconds=.1,node_limit=50000)
    r=worker.solve(m,tmp_path/'r.json')
    if key=='healing_percent_per_second' and hi==3.5:
        # Neither linear seed meets this derived upper bound: preserve V2.2
        # full search instead of passing an invalid start or changing limits.
        assert r['native_start']['fallback']=='no_complete_certified_start'
    else:assert r['native_start']['used']
    for stage in r['stages']:
        if 'indices' in stage:assert lo<=portfolio.native(m,stage['indices'])[0][key]<=hi

def test_v1_secondary_matrix_byte_exact(monkeypatch,tmp_path):
    _,m=model();original=ns.compatibility;calls=[[],[]];real=milp;current=[0]
    def spy(c,**kw):
        if len(c)==m['variables'] and np.all(c>=0):
            a=kw['constraints'].A;calls[current[0]].append((c.tobytes(),a.data.tobytes(),a.indices.tobytes(),a.indptr.tobytes(),kw['constraints'].lb.tobytes(),kw['constraints'].ub.tobytes()))
        return real(c,**kw)
    monkeypatch.setattr('scipy.optimize.milp',spy)
    monkeypatch.setattr(ns,'compatibility',lambda:(False,'controlled baseline'))
    a=worker.solve(m,tmp_path/'old.json');current[0]=1;monkeypatch.setattr(ns,'compatibility',original)
    b=worker.solve(m,tmp_path/'new.json')
    assert a['optimal'] and b['optimal'] and calls[0]==calls[1] and len(calls[0])==1

@pytest.mark.parametrize('kind',['memory','badalloc','abort','commit','cancel','timeout','watchdog'])
def test_actual_worker_native_checkpoint_recovery(monkeypatch,tmp_path,kind):
    _,m=model();event=threading.Event();observed=threading.Event();paths=[];children=[]
    script='''
import os,time,ctypes
from pathlib import Path
from app import _survivability_native as n,_max_quality_worker as w
kind=KIND
def fault(*args):
 if kind=='memory':raise MemoryError('controlled native allocation failure')
 if kind=='badalloc':raise RuntimeError('std::bad_alloc')
 if kind=='abort':os._exit(91)
 if kind=='commit':
  kernel=ctypes.WinDLL('kernel32',use_last_error=True)
  kernel.VirtualAlloc.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_ulong,ctypes.c_ulong];kernel.VirtualAlloc.restype=ctypes.c_void_p
  p=kernel.VirtualAlloc(None,600*1024**2,0x3000,4)
  if p:
   kernel.VirtualFree.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_ulong];kernel.VirtualFree(p,0,0x8000)
   raise AssertionError('512MiB Windows Job did not reject600MiB commit')
  raise MemoryError('Windows Job rejected600MiB commit')
 time.sleep(10)
n.native_solve=fault
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
                if d.get('native_start',{}).get('checkpoint_published'):
                    observed.set()
                    if kind=='cancel':event.set()
                    return
            except (IndexError,OSError,ValueError):pass
            time.sleep(.005)
    watch=kind in ('cancel','timeout','watchdog');thread=threading.Thread(target=observe) if watch else None
    if thread:thread.start()
    deadline=time.monotonic()+6
    if kind=='timeout':
        real_time=time.monotonic
        class Clock:
            @staticmethod
            def monotonic():return deadline+1 if observed.is_set() else real_time()
            sleep=staticmethod(time.sleep)
        monkeypatch.setattr(process,'time',Clock)
    if kind=='watchdog':monkeypatch.setattr(process,'resident_bytes',lambda pid:m['memory_bytes']+1 if observed.is_set() else resident(pid))
    try:r=process.run_model(m,deadline,event)
    finally:
        if thread:thread.join(timeout=6)
    expected={'memory':'memory_limit','badalloc':'memory_limit','commit':'memory_limit','abort':'worker_failed','cancel':'cancelled','timeout':'request_time_limit','watchdog':'memory_limit'}
    assert r['reason']==expected[kind],r
    assert r['native_start']['checkpoint_published'] and not r['optimal'] and not r['infeasible']
    assert any('indices' in s and s['kind']=='portfolio_witness' for s in r['stages'])
    assert r['memory_limit_kind']=='windows_process_commit'
    assert all(c.poll() is not None for c in children)

@pytest.mark.parametrize('conservative',[False,True])
@pytest.mark.parametrize('stock',[False,True])
@pytest.mark.parametrize('owned',[False,True])
def test_native_actual_API_export(monkeypatch,tmp_path,conservative,stock,owned):
    from tests.test_full_quality_candidates import sql_market
    from tests.test_stock_only import api
    from app import lunar_bridge
    offers=[(dict(health_bonus=i%4,bullet_dmg_factor=(i*7)%30,regeneration_bonus=1+i%3),1+i) for i in range(16)]
    _,conn=sql_market(monkeypatch,tmp_path,offers,slots=2);monkeypatch.setattr(lunar_bridge,'get_connection',conn,raising=False)
    _,client=api(monkeypatch,tmp_path)
    q=dict(budget=1000,max_quality='true',deep_optimization='true',stock_only=str(stock).lower(),conservative=str(conservative).lower(),priorities='{"healing_ehp_per_second":10}')
    if owned:q['owned']=json.dumps([dict(item_id='1',quality=0,ptn=0,art_percent=100,instance_id='one')])
    for route in ['/api/builds','/api/export-lunarzone']:
        r=client.get(route,params=q);assert r.status_code==200,r.text
        d=r.json();status=d['search_status'] if route=='/api/builds' else d['lunarzone_json']['_optimizer_meta']['search_status']
        assert any(m.get('native_start',{}).get('used') for m in status['models']),status

def test_native_owned_only(monkeypatch,tmp_path):
    import sqlite3
    from tests.test_full_quality_candidates import sql_market,call
    path,_=sql_market(monkeypatch,tmp_path,[(dict(health_bonus=i,bullet_dmg_factor=i,regeneration_bonus=1+i%3),1) for i in range(16)],slots=2)
    with sqlite3.connect(path) as c:c.execute('DELETE FROM auction_lots')
    owned=[dict(item_id=str(i+1),quality=0,ptn=0,art_percent=100,instance_id=str(i)) for i in range(16)]
    r=call(priorities={'healing_ehp_per_second':10},deep_optimization=True,owned_artifacts=owned,stock_only=True)
    assert r and r[0].total_price==0 and all(o.is_owned for o in r[0].artifacts)
    assert any(m.get('native_start',{}).get('used') for m in r.search_status['models'])

@pytest.mark.parametrize('changes',[1,2])
def test_native_refresh_single_retry(monkeypatch,tmp_path,changes):
    import sqlite3
    from tests.test_full_quality_candidates import sql_market,call
    from app import _max_quality as q
    path,_=sql_market(monkeypatch,tmp_path,[(dict(health_bonus=i,regeneration_bonus=1+i%3),1) for i in range(16)],slots=2)
    with sqlite3.connect(path.parent/'app_state.sqlite3') as c:
        c.execute('CREATE TABLE refresh_jobs(name TEXT,generation INTEGER,running INTEGER)');c.execute("INSERT INTO refresh_jobs VALUES('auction:EU',0,0)")
    original=q.run_model;calls=[]
    def run(*a,**kw):
        state=original(*a,**kw);calls.append(state);assert state['native_start']['used']
        if len(calls)<=changes:
            with sqlite3.connect(path) as c:c.execute('UPDATE auction_lots SET id=id+100,buyout_price=buyout_price+1')
            with sqlite3.connect(path.parent/'app_state.sqlite3') as c:c.execute('UPDATE refresh_jobs SET generation=generation+1')
        return state
    monkeypatch.setattr(q,'run_model',run)
    r=call(priorities={'healing_ehp_per_second':10},deep_optimization=True)
    assert len(calls)==2 and r.search_status['retry_count']==1
    if changes==1:assert r and all(o.auction_lot_id>=101 for o in r[0].artifacts)
    else:assert not r and r.search_status['state']=='UNKNOWN'

@pytest.mark.parametrize('limit',['time','node'])
@pytest.mark.parametrize('start',[False,True])
def test_actual_native_limit_with_and_without_primal(limit,start):
    from scipy.optimize import Bounds,LinearConstraint
    rng=np.random.default_rng(23);a=rng.integers(1,30,(4,40)).astype(float)
    cost=-rng.integers(1,30,40).astype(float)
    kw=dict(integrality=np.ones(40),bounds=Bounds(np.zeros(40),np.ones(40)),constraints=LinearConstraint(a,-np.inf,a.sum(axis=1)*.3),
            options=dict(time_limit=0. if limit=='time' else 1.,node_limit=0 if limit=='node' else 50000,presolve=False,mip_rel_gap=0.))
    r=ns.native_solve(cost,kw,np.zeros(40) if start else None)
    assert r.status!=0 and r.native_status in (13,16)
    assert (r.x is not None)==start
    if r.x is not None:assert np.all(a@r.x<=kw['constraints'].ub)

def test_actual_native_infeasible_without_witness():
    from scipy.optimize import Bounds,LinearConstraint
    kw=dict(integrality=np.ones(2),bounds=Bounds([0.,0.],[1.,1.]),constraints=LinearConstraint([[1.,1.]],3,np.inf),options=dict(time_limit=1.,node_limit=20,mip_rel_gap=0.))
    r=ns.native_solve(np.array([1.,2.]),kw)
    assert r.status==2 and r.x is None

@pytest.mark.parametrize('native_status,message',[(1,'HiGHS Status 13: Time limit reached'),(4,'HiGHS Status 16: Solution limit reached'),(2,'HiGHS Status 8: Infeasible')])
def test_lost_native_incumbent_never_loses_portfolio(monkeypatch,tmp_path,native_status,message):
    from scipy.optimize import OptimizeResult
    _,m=model()
    def lost(*a):return OptimizeResult(status=native_status,message=message,x=None,mip_dual_bound=float('nan'),mip_gap=None,mip_node_count=0,native_status=16,injection_status=0,primal_solution_status=0)
    monkeypatch.setattr(ns,'native_solve',lost)
    r=worker.solve(m,tmp_path/'r.json')
    assert not r['optimal'] and not r['infeasible']
    assert any(s.get('kind')=='portfolio_witness' and 'indices' in s for s in r['stages'])

def test_true_native_execution_error_does_not_restart_solver(monkeypatch,tmp_path):
    _,m=model();dest=tmp_path/'r.json';calls=[]
    def error(*a):calls.append(1);raise RuntimeError('controlled native execution failure')
    monkeypatch.setattr(ns,'native_solve',error)
    with pytest.raises(RuntimeError):worker.solve(m,dest)
    stored=json.loads(dest.read_text())
    assert len(calls)==1 and stored['native_start']['checkpoint_published'] and not stored['optimal']

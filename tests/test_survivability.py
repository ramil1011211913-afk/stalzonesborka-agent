"""Derived objectives are independently enumerated, not treated as additive stats."""
import itertools,json,math
from decimal import Decimal as D
from types import SimpleNamespace
import numpy as np
import pytest
from app import builder as b,_max_quality as q,_max_quality_model as mm,_max_quality_worker as worker
from app._objective import Objective
from app._survivability import values,REGISTRY,normalized_objective
from tests.test_max_quality import option,call
from tests.test_objective_integration import container
from tests.test_feasibility_fallback import market

def oracle(stats):
    h=D(str(stats.get('health_bonus',0)));bullet=D(str(stats.get('bullet_dmg_factor',0)))
    g=D(str(stats.get('regeneration_bonus',0)));t=D(str(stats.get('artefakt_heal',0)));u=D(str(stats.get('heal_efficiency',0)))
    e=(100+h)*(100+bullet)/100;r=(g+D('2.5'))/5+t*(1+u/100)
    return {'effective_health':e,'healing_percent_per_second':r,'healing_ehp_per_second':e*r/100}

@pytest.mark.parametrize('stats',[{},dict(health_bonus=10,bullet_dmg_factor=70),dict(regeneration_bonus=5),dict(artefakt_heal=2,heal_efficiency=-20),dict(regeneration_bonus=-5),dict(health_bonus=-100)])
def test_units_and_independent_oracle(stats):
    expected=oracle(stats);actual=values(stats)
    assert all(abs(actual[k]-float(v))<1e-12 for k,v in expected.items())
    assert actual['healing_percent_per_second']==b.calculate_healing_per_second(stats)

@pytest.mark.parametrize('key',list(REGISTRY))
@pytest.mark.parametrize('minimum',[{},dict(effective_health=140),dict(healing_percent_per_second=2),dict(healing_ehp_per_second=3)])
def test_sparse_model_against_complete_decimal_oracle(tmp_path,key,minimum):
    rows=[dict(health_bonus=20,bullet_dmg_factor=1),dict(health_bonus=1,bullet_dmg_factor=45),
          dict(regeneration_bonus=10,health_bonus=2),dict(artefakt_heal=2,heal_efficiency=30),
          dict(bullet_dmg_factor=15,regeneration_bonus=5),dict(health_bonus=20,bullet_dmg_factor=1)]
    options=[option(i,s,3 if i<5 else 1) for i,s in enumerate(rows)]
    cont=container(slots=2)
    obj=normalized_objective(Objective.create({key:10}),[(cont,options)])
    feasible=[]
    for ids in itertools.combinations(range(6),2):
        s={k:sum(rows[i].get(k,0) for i in ids) for k in set().union(*rows)};v=oracle(s)
        if any(v[k]<D(str(x)) for k,x in minimum.items()):continue
        feasible.append((v[key],-sum(options[i].buyout_price for i in ids),ids))
    model=mm.make_model(options,cont,obj,10,minimum,{},True,True)
    result=worker.solve(dict(model,seconds=10),tmp_path/'result.json')
    if not feasible:assert result['infeasible'];return
    assert result['optimal'],result
    picked=tuple(result['stages'][-1]['indices'])
    assert next(v[:2] for v in feasible if v[2]==picked)==max(v[:2] for v in feasible)
    assert result['priority_optimality_proven'] and result['cost_optimality_proven']

@pytest.mark.parametrize('key',list(REGISTRY))
@pytest.mark.parametrize('conservative',[False,True])
@pytest.mark.parametrize('stock',[False,True])
def test_native_full_physical_validation(monkeypatch,tmp_path,key,conservative,stock):
    market(monkeypatch,tmp_path,[({'health_bonus':10,'bullet_dmg_factor':50,'regeneration_bonus':3},4),({'health_bonus':5,'bullet_dmg_factor':5,'regeneration_bonus':5},2),({'health_bonus':1,'regeneration_bonus':2},1)],slots=2)
    result=call(priorities={key:10},conservative=conservative,stock_only=stock,deep_optimization=True)
    assert result and result.search_status['state']=='OPTIMAL',result.search_status
    assert b._final_valid(result[0],10,{},{},True,False,True)
    assert len({o.auction_lot_id for o in result[0].artifacts})==2
    data=result[0].to_dict()
    assert data['derived_stats'][key]==result[0].combined_stats[key]
    assert data['calculation_provenance']['legacy_healing_per_second_unit']=='%/с'
    assert data['objective']['contract']=='survivability-objective-v1'
    assert data['healing_per_second']==round(result[0].combined_stats['healing_percent_per_second'],1)
    assert result.search_status['models'][0]['worker_pid']==result.search_status['models'][0]['process_id']

@pytest.mark.parametrize('constraint',[dict(effective_health=161),dict(healing_percent_per_second=2),dict(healing_ehp_per_second=4)])
def test_derived_min_restricts_search_not_postfilter(monkeypatch,tmp_path,constraint):
    market(monkeypatch,tmp_path,[({'bullet_dmg_factor':60},1),({'bullet_dmg_factor':35,'health_bonus':30,'regeneration_bonus':15},5)],slots=1)
    result=call(priorities={'bullet_dmg_factor':10},min_thresholds=constraint)
    assert result and result[0].artifacts[0].auction_lot_id==2,result.search_status
    assert all(result[0].combined_stats[k]>=v for k,v in constraint.items())

def test_impossible_derived_min_is_proven_not_timeout(monkeypatch,tmp_path):
    market(monkeypatch,tmp_path,slots=1)
    result=call(priorities={'effective_health':10},min_thresholds={'effective_health':10000})
    assert not result and result.search_status['state']=='INFEASIBLE'

@pytest.mark.parametrize('status,message,has_x,reason',[(4,'HiGHS Status 16: Solution limit reached',True,'solution_limit'),(4,'HiGHS Status 16: Solution limit reached',False,'solution_limit'),(1,'HiGHS Status 13: Time limit',True,'time_limit'),(1,'HiGHS Status 13: Time limit',False,'time_limit'),(4,'HiGHS Status 4: solve error',False,'solver_error'),(2,'HiGHS Status 2: Model error',False,'solver_error')])
def test_limits_never_false_optimal(tmp_path,status,message,has_x,reason):
    o=[option(1,{'health_bonus':5},1)];model=mm.make_model(o,container(slots=1),Objective.create({'effective_health':10}),10,{},{},True,True)
    def fake(*a,**kw):
        x=np.zeros(model['variables']) if has_x else None
        if has_x:x[0]=1
        return SimpleNamespace(status=status,message=message,x=x,mip_dual_bound=None,mip_gap=None,mip_node_count=None)
    result=worker.solve(dict(model,seconds=10),tmp_path/'r.json',solver=fake)
    assert not result['optimal'] and not result['infeasible']
    assert ('indices' in result['stages'][0])==has_x
    assert result['reason']==reason

def test_secondary_limit_keeps_first_witness_and_never_cost_optimal(monkeypatch,tmp_path):
    market(monkeypatch,tmp_path,[({'health_bonus':5},3)],slots=1)
    original=q.run_model
    def run(*a,**kw):
        state=original(*a,**kw);assert state['optimal'];state['stages']=state['stages'][:1]+[dict(stage=1,kind='cost',solver_status=4,solver_message='HiGHS Status 16: Solution limit reached')]
        state.update(optimal=False,cost_optimality_proven=False,reason='solution_limit');return state
    monkeypatch.setattr(q,'run_model',run)
    result=call(priorities={'effective_health':10})
    assert result and result.search_status['state']=='FEASIBLE'

def test_physical_change_still_rejects_derived_winner(monkeypatch,tmp_path):
    import sqlite3
    _,path,_=market(monkeypatch,tmp_path,[({'health_bonus':5},3)],slots=1)
    original=q.run_model
    def run(*a,**kw):
        state=original(*a,**kw)
        with sqlite3.connect(path) as c:c.execute('UPDATE auction_lots SET buyout_price=4')
        return state
    monkeypatch.setattr(q,'run_model',run)
    result=call(priorities={'effective_health':10})
    assert not result and result.search_status['state']=='UNKNOWN'

@pytest.mark.parametrize('deep',[False,True])
def test_real_TestClient_units_and_min(monkeypatch,tmp_path,deep):
    from tests.test_stock_only import api
    market(monkeypatch,tmp_path,[({'health_bonus':5,'bullet_dmg_factor':10,'regeneration_bonus':5},3)],slots=1)
    _,client=api(monkeypatch,tmp_path)
    result=client.get('/api/builds',params=dict(budget=10,tolerance=0,max_quality='true',deep_optimization=str(deep).lower(),priorities='{"effective_health":10}',min_thresholds='{"healing_percent_per_second":1}'))
    assert result.status_code==200,result.text
    data=result.json();assert data['search_status']['state']=='OPTIMAL'
    assert data['builds'][0]['derived_stats']['healing_percent_per_second']==1.5

@pytest.mark.parametrize('bad',['NaN','Infinity','"2"','true','{}','[]'])
def test_API_nonfinite_and_ambiguous_thresholds(monkeypatch,tmp_path,bad):
    from tests.test_stock_only import api
    _,client=api(monkeypatch,tmp_path)
    response=client.get('/api/builds',params=dict(budget=10,max_quality='true',priorities='{"effective_health":10}',min_thresholds='{"effective_health":'+bad+'}'))
    assert response.status_code==422

def test_Beam_and_nonlinear_lex_explicit_unsupported(monkeypatch,tmp_path):
    from tests.test_stock_only import api
    _,client=api(monkeypatch,tmp_path)
    assert client.get('/api/builds',params=dict(priorities='{"effective_health":10}')).status_code==422
    assert client.get('/api/builds',params=dict(max_quality='true',optimization_mode='lexicographic',priority_order='["effective_health"]',priorities='{"effective_health":10}')).status_code==422

def test_constant_regen_zero_and_objectives_not_equivalent():
    assert values({'health_bonus':10})['healing_percent_per_second']==.5
    assert values({'regeneration_bonus':-2.5})['healing_percent_per_second']==0
    a=values({'bullet_dmg_factor':100});bld=values({'bullet_dmg_factor':50,'regeneration_bonus':7.5})
    assert a['effective_health']>bld['effective_health'] and a['healing_ehp_per_second']<bld['healing_ehp_per_second']

def test_equality_is_allowed_without_safety_margin(monkeypatch,tmp_path):
    market(monkeypatch,tmp_path,[({'bullet_dmg_factor':60,'radiation_accumulation':10},1)],slots=1)
    result=call(priorities={'effective_health':10},min_thresholds={'effective_health':160})
    assert result and result[0].combined_stats['effective_health']==160
    assert result[0].combined_stats['radiation_accumulation']==10 and result[0].carryable

def test_constant_healing_optimizes_price(monkeypatch,tmp_path):
    market(monkeypatch,tmp_path,[({'bullet_dmg_factor':60},5),({'health_bonus':1},1)],slots=1)
    result=call(priorities={'healing_percent_per_second':10})
    assert result.search_status['state']=='OPTIMAL' and result[0].total_price==1
    assert result[0].combined_stats['healing_percent_per_second']==.5

def test_owned_identity_and_mixed_sources(monkeypatch,tmp_path):
    opts,_,_=market(monkeypatch,tmp_path,[({'health_bonus':10,'bullet_dmg_factor':20},5),({'regeneration_bonus':5},2)],slots=2)
    # Existing mixed-source semantics: owned is free but a separate physical instance.
    native=b.get_connection
    class OwnedRow(dict):
        pass
    class Wrapper:
        def __enter__(self):self.cm=native();self.c=self.cm.__enter__();return self
        def __exit__(self,*args):return self.cm.__exit__(*args)
        def __getattr__(self,k):return getattr(self.c,k)
        def execute(self,sql,params=()):
            if 'FROM artefacts' in sql:
                return SimpleNamespace(fetchone=lambda:dict(id='own',name_ru='own',name_en='',subcategory='',stats=json.dumps([{'values':{'health_bonus':20}}]),bonus_stats='[]',icon_url=''))
            return self.c.execute(sql,params)
    monkeypatch.setattr(b,'get_connection',Wrapper)
    result=call(priorities={'effective_health':10},owned_artifacts=[dict(item_id='own',quality=0,ptn=0,art_percent=100,instance_id='owned-1')])
    assert result and any(o.is_owned for o in result[0].artifacts),result.search_status
    assert len({o.physical_id for o in result[0].artifacts})==2

@pytest.mark.parametrize('key',list(REGISTRY))
def test_fixed_armor_no_double_count(monkeypatch,tmp_path,key):
    from tests.test_armor_search import install_armors
    market(monkeypatch,tmp_path,[({'health_bonus':5,'bullet_dmg_factor':10,'regeneration_bonus':5},3)],slots=1)
    install_armors(monkeypatch,{'armor':{'bullet_dmg_factor':100,'health_bonus':10}})
    result=call(priorities={key:10},armor_id='armor')
    assert result and result.search_status['state']=='OPTIMAL',result.search_status
    stats=result[0].combined_stats
    assert stats['effective_health']==pytest.approx(1.15*210)
    assert result[0]._unarmored_stats['effective_health']==pytest.approx(1.05*110)
    assert result[0].to_dict()['armor']['id']=='armor'

@pytest.mark.parametrize('weights',[{'effective_health':10,'healing_percent_per_second':3}, {'effective_health':10,'healing_ehp_per_second':5,'speed_modifier':1}])
def test_balanced_mixed_scales_fixed_and_oracle(tmp_path,weights):
    rows=[{'health_bonus':10,'bullet_dmg_factor':40,'speed_modifier':1},{'regeneration_bonus':10,'speed_modifier':3},{'artefakt_heal':2,'heal_efficiency':20,'health_bonus':3}]
    opts=[option(i,s,i+1) for i,s in enumerate(rows)];cont=container(slots=1)
    obj=normalized_objective(Objective.create(weights),[(cont,opts)])
    model=mm.make_model(opts,cont,obj,10,{}, {},True,True)
    r=worker.solve(dict(model,seconds=10),tmp_path/'r.json')
    assert r['optimal'],r
    scores=[]
    for i,s in enumerate(rows):
        v=dict(s);v.update({k:float(x) for k,x in oracle(s).items()});scores.append((obj.evaluate(v),-opts[i].buyout_price,i))
    assert r['stages'][-1]['indices']==[max(scores)[2]]

def absolute_model(options,minimum=None):
    cont=container(slots=1);obj=normalized_objective(Objective.create({'healing_ehp_per_second':10}),[(cont,options)])
    return mm.make_model(options,cont,obj,100,minimum or {},{},True,True)

def test_seed_rejected_if_derived_min_not_met(tmp_path):
    data=absolute_model([option(0,{'regeneration_bonus':50},1),option(1,{'health_bonus':100,'bullet_dmg_factor':100},2)],{'effective_health':200})
    result=worker.solve(dict(data,seconds=10),tmp_path/'r.json')
    assert result['optimal'],result
    assert result['stages'][0]['kind']=='incumbent_seed' and 'indices' not in result['stages'][0]
    assert result['stages'][-1]['indices']==[1]

@pytest.mark.parametrize('status,message',[(2,'HiGHS Status 8: Infeasible'),(1,'HiGHS Status 13: Time limit')])
def test_seed_never_proves_primary_or_original_infeasibility(tmp_path,status,message):
    from scipy.optimize import milp
    data=absolute_model([option(0,{'regeneration_bonus':10,'health_bonus':5},1)]);calls=[]
    def solver(*a,**kw):
        if not calls:calls.append(1);return milp(*a,**kw)
        return SimpleNamespace(status=status,message=message,x=None,mip_dual_bound=None,mip_gap=None,mip_node_count=None)
    r=worker.solve(dict(data,seconds=10),tmp_path/'r.json',solver=solver)
    assert 'indices' in r['stages'][0] and r['stages'][0]['objective_upper_bound'] is None
    assert not r['optimal'] and not r['infeasible'] and not r['priority_optimality_proven']
    if status==2:assert r['reason']=='restricted_subproblem_infeasible'

def test_seed_and_primary_share_node_budget(tmp_path):
    from scipy.optimize import milp
    data=absolute_model([option(0,{'regeneration_bonus':10},1)]);data['node_limit']=7;calls=[]
    def solver(*a,**kw):
        real=milp(*a,**kw)
        if not calls:
            calls.append(1);return SimpleNamespace(status=4,message='HiGHS Status 16: Solution limit reached',x=real.x,mip_dual_bound=None,mip_gap=None,mip_node_count=None)
        return real
    r=worker.solve(dict(data,seconds=10),tmp_path/'r.json',solver=solver)
    assert r['stages'][0]['node_count'] is None and r['primary_seed_node_charge']==7
    assert r['stages'][1]['node_limit']==0

def test_tiny_but_representable_objective_is_not_lost(tmp_path):
    opts=[option(0,{'health_bonus':100},5),option(1,{},1)]
    obj=Objective.create({'effective_health':10},{'effective_health':1e308})
    data=mm.make_model(opts,container(slots=1),obj,10,{},{},True,True)
    r=worker.solve(dict(data,seconds=10),tmp_path/'r.json')
    assert r['optimal'] and r['stages'][-1]['indices']==[0],r
    assert r['stages'][0]['objective_scale']>1

def test_negative_healing_does_not_use_positive_monotonic_shortcut(tmp_path):
    data=absolute_model([option(0,{'health_bonus':100,'regeneration_bonus':-5},5),option(1,{'regeneration_bonus':-5},1)])
    r=worker.solve(dict(data,seconds=10),tmp_path/'r.json')
    assert r['optimal'] and r['stages'][-1]['indices']==[1],r

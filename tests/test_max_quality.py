"""Independent enumeration, native validation, worker isolation and legacy guards."""
import itertools
import json
import math
import random
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from app import builder as b
from app import _max_quality as quality
from app import _max_quality_model as model
from app import _max_quality_process as process
from app import _max_quality_worker as worker
from app._objective import Objective
from app._physical_items import is_available
from tests.test_feasibility_fallback import market
from tests.test_objective_integration import container


def call(**kw):
    request = dict(budget=10, tolerance_pct=0, priorities={'speed_modifier': 10},
                   region='EU', allow_duplicates=True, quality_timeout=15)
    request.update(kw)
    return quality.build_max_quality_sets(**request)


def option(i, stats, price, physical=None, item=None):
    return b.ArtifactOption(item_id=str(i) if item is None else item, name_ru=str(i), name_en='',
        subcategory='', quality=0, quality_name='', art_percent=100, buyout_price=price, ptn=0,
        stats=stats, stats_raw=stats, stats_names={}, unlocked_bonus_ids=[], base_stats=[{'values': stats}],
        auction_lot_id=physical if physical is not None else i, physical_id='auction:EU:'+str(i))


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic'])
def test_native_worker_full_rank_and_game_validation(monkeypatch, tmp_path, mode):
    market(monkeypatch, tmp_path)
    result = call(optimization_mode=mode, priority_order=['speed_modifier'] if mode=='lexicographic' else None)
    assert result and result.search_status['state']=='OPTIMAL', result.search_status
    assert result[0].combined_stats['speed_modifier']==12 and result[0].total_price==6
    assert result[0].carryable and b._final_valid(result[0],10,{},{},True,False,True)
    assert result.search_status['models'][0]['sampled_worker_peak_rss']>0
    info=result.search_status['models'][0]
    assert info['worker_pid']==info['process_id']
    assert info['sampled_worker_peak_rss']>20*1024**2


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('duplicates', [False, True])
@pytest.mark.parametrize('mode', ['balanced', 'lexicographic'])
def test_model_against_independent_complete_enumeration(tmp_path, seed, duplicates, mode):
    rng=random.Random(seed)
    options=[option(i, dict(speed_modifier=rng.randint(-3,10), health_bonus=rng.randint(-4,8),
        radiation_accumulation=rng.randint(-12,20)), rng.randint(1,7), physical=i//2 if i<4 else i,
        item='same' if i<4 else str(i)) for i in range(8)]
    priorities={'speed_modifier':10,'health_bonus':3}
    objective=Objective.create(priorities, {'speed_modifier':20,'health_bonus':10}, mode,
        ['speed_modifier','health_bonus'] if mode=='lexicographic' else None)
    cont=container(slots=3, bonuses={'health_bonus':1}); cont.inner_protection=50
    minimum={'health_bonus':0}; maximum={'health_bonus':15}
    feasible=[]
    for combo in itertools.combinations(options,3):
        if not is_available(combo,duplicates) or sum(o.buyout_price for o in combo)>12:continue
        values={k:sum(o.stats.get(k,0) for o in combo) for k in ('speed_modifier','health_bonus','radiation_accumulation')}
        values['health_bonus']+=1
        if values['radiation_accumulation']>0:values['radiation_accumulation']*=.5
        if values['radiation_accumulation']>10 or not 0<=values['health_bonus']<=15:continue
        feasible.append(SimpleNamespace(combined_stats=values,total_price=sum(o.buyout_price for o in combo)))
    data=model.make_model(options,cont,objective,12,minimum,maximum,duplicates,True)
    result=worker.solve(dict(data,seconds=10),tmp_path/'result.json')
    if not feasible:
        assert result['infeasible'];return
    assert result['optimal'],result
    ids=result['stages'][-1]['indices'];combo=[options[i] for i in ids]
    values={k:sum(o.stats.get(k,0) for o in combo) for k in ('speed_modifier','health_bonus','radiation_accumulation')}
    values['health_bonus']+=1
    selected=SimpleNamespace(combined_stats=values,total_price=sum(o.buyout_price for o in combo))
    assert objective.sort_key(selected)==min(map(objective.sort_key,feasible))


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic'])
def test_ties_preserve_existing_rank_and_price(monkeypatch, tmp_path, mode):
    rows=[({'speed_modifier':10.0000004,'health_bonus':0},8),
          ({'speed_modifier':10.0000003,'health_bonus':5},3),
          ({'speed_modifier':10.0000003,'health_bonus':5},2)]
    market(monkeypatch,tmp_path,rows,slots=1)
    result=call(optimization_mode=mode,priorities={'speed_modifier':10,'health_bonus':1},
        priority_order=['speed_modifier','health_bonus'] if mode=='lexicographic' else None)
    assert result and result[0].total_price==2,result.search_status
    assert result.search_status['state']=='OPTIMAL'


@pytest.mark.parametrize('change', ['expired','price','region','quality','percent','bonuses','missing'])
def test_current_market_changes_never_return_old_offer(monkeypatch,tmp_path,change):
    _,path,_=market(monkeypatch,tmp_path,[({'speed_modifier':10},2),({'speed_modifier':1},1)],slots=1)
    import sqlite3
    with sqlite3.connect(path) as c:
        if change=='missing':c.execute('delete from auction_lots where id=1')
        else:
            column,value={'expired':('end_time','2000-01-01T00:00:00Z'),'price':('buyout_price',3),
                'region':('region','RU'),'quality':('quality',1),'percent':('art_percent',99),
                'bonuses':('bonus_properties','["UNKNOWN"]')}[change]
            c.execute('update auction_lots set '+column+'=? where id=1',(value,))
    result=call()
    assert result and result[0].artifacts[0].auction_lot_id==2


def test_revalidation_rejects_changed_price_after_solver(monkeypatch,tmp_path):
    _,path,_=market(monkeypatch,tmp_path,[({'speed_modifier':10},2)],slots=1)
    native=quality.run_model
    def run(*a,**kw):
        r=native(*a,**kw)
        import sqlite3
        with sqlite3.connect(path) as c:c.execute('update auction_lots set buyout_price=3')
        return r
    monkeypatch.setattr(quality,'run_model',run)
    result=call()
    assert not result and result.search_status['state']=='UNKNOWN'
    assert result.search_status['models'][0]['validation_failed']


@pytest.mark.parametrize('flag',['guaranteed_carryable'])
def test_unsupported_risk_modes_are_explicit(flag):
    result=call(**{flag:True})
    assert not result and result.search_status['state']=='UNKNOWN'
    assert 'unsupported' in result.search_status['reason']


@pytest.mark.parametrize('key',['stopping_protection','__hidden','made_up'])
def test_derived_unknown_constraints_are_not_dropped(monkeypatch,tmp_path,key):
    market(monkeypatch,tmp_path,slots=1)
    result=call(min_thresholds={key:1})
    assert not result and result.search_status['models'][0]['reason']=='unsupported_or_derived_stat'


def test_protected_danger_objective_rejected_but_linear_reaction_supported(tmp_path):
    opt=option(1,{'radiation_accumulation':2,'reaction_to_burn':1},1)
    cont=container();cont.inner_protection=50
    with pytest.raises(model.Unsupported):model.make_model([opt],cont,Objective.create({'radiation_accumulation':1}),10,{},{},True,True)
    data=model.make_model([opt],cont,Objective.create({'reaction_to_burn':1}),10,{},{},True,True)
    assert worker.solve(dict(data,seconds=10),tmp_path/'r.json')['optimal']


@pytest.mark.parametrize('kind',['min','max','budget','carry'])
def test_hard_limits_can_prove_scoped_infeasibility(monkeypatch,tmp_path,kind):
    market(monkeypatch,tmp_path,[({'speed_modifier':10,'radiation_accumulation':20},2)],slots=1)
    kwargs={'carryable_only':False}
    if kind=='min':kwargs['min_thresholds']={'speed_modifier':11}
    if kind=='max':kwargs['max_thresholds']={'speed_modifier':9}
    if kind=='budget':kwargs['budget']=1
    if kind=='carry':kwargs['carryable_only']=True
    result=call(**kwargs)
    assert not result and result.search_status['state'] in ('INFEASIBLE','UNKNOWN')
    assert not result.search_status['global_optimality_proven']


def test_multiple_containers_and_armor_profiles_share_scales(monkeypatch,tmp_path):
    market(monkeypatch,tmp_path,[({'speed_modifier':5,'health_bonus':-1},2),({'speed_modifier':4,'health_bonus':1},1)],slots=1)
    monkeypatch.setattr(b,'load_container_options',lambda r:[container('a',1,100),container('b',1,50)])
    armor=dict(id='armor',level=15,stats={'speed_modifier':2,'health_bonus':2},name_ru='armor',category='',rank='',icon_url='')
    monkeypatch.setattr(b,'load_armor',lambda *a:armor)
    result=call(armor_ids=[None,'armor'],min_thresholds={'health_bonus':0})
    assert result and result[0].container.item_id=='a' and result[0]._armor['id']=='armor'
    assert result[0].combined_stats['speed_modifier']==7
    assert len(result.search_status['models'])==4
    assert result.search_status['state']=='OPTIMAL'
    monkeypatch.setattr(b,'build_artifact_sets',lambda **kw:(_ for _ in ()).throw(AssertionError('wrong rerun')))
    assert b.apply_armor_to_builds(result,'armor') is result


@pytest.mark.parametrize('timeout',[0,-1,61,float('nan')])
def test_timeout_validation(timeout):
    with pytest.raises(ValueError):call(quality_timeout=timeout)


def test_timeout_no_incumbent_and_after_primary(tmp_path):
    opt=option(1,{'speed_modifier':5},1);data=model.make_model([opt],container(),Objective.create({'speed_modifier':1}),10,{},{},True,True)
    calls=[]
    def fake(cost,**kw):
        calls.append(cost)
        if len(calls)==1:return SimpleNamespace(status=0,message='primary optimum',x=np.array([1.]),mip_dual_bound=-5.,mip_gap=0.)
        return SimpleNamespace(status=1,message='time limit',x=None,mip_dual_bound=None,mip_gap=None)
    result=worker.solve(dict(data,seconds=10),tmp_path/'after.json',solver=fake)
    assert not result['optimal'] and result['stages'][0]['indices']==[0]
    empty=worker.solve(dict(data,seconds=0),tmp_path/'empty.json')
    assert not empty['stages'] and not empty['infeasible']


@pytest.mark.parametrize('reason',['cancelled','memory_limit','worker_error','dependency_unavailable','time_or_node_limit'])
def test_failed_worker_never_proves_absence_or_optimality(monkeypatch,tmp_path,reason):
    market(monkeypatch,tmp_path,slots=1)
    monkeypatch.setattr(quality,'run_model',lambda *a:dict(stages=[],optimal=False,infeasible=False,reason=reason))
    result=call()
    assert not result and result.search_status['state']=='UNKNOWN'


def test_timeout_returns_revalidated_primary_without_false_optimality(monkeypatch,tmp_path):
    market(monkeypatch,tmp_path,[({'speed_modifier':5},1)],slots=1)
    monkeypatch.setattr(quality,'run_model',lambda *a:dict(stages=[dict(indices=[0],kind='priority',solver_status=1)],optimal=False,infeasible=False,reason='time_or_node_limit'))
    result=call()
    assert result and result.search_status['state']=='FEASIBLE'
    assert result.search_status['models'][0]['stages'][0]['solver_status']==1


def test_native_worker_cancellation_and_lock_release():
    event=threading.Event();event.set()
    assert process.run_model({},time.monotonic()+1,event)['reason']=='cancelled'
    with process.quality_slot() as first:
        assert first
        with process.quality_slot() as second:assert not second
    with process.quality_slot() as third:assert third


def test_request_busy_does_not_start_preparation(monkeypatch):
    with process.quality_slot() as acquired:
        assert acquired
        monkeypatch.setattr(b,'build_artifact_sets',lambda *a,**kw:(_ for _ in ()).throw(AssertionError()))
        assert call().search_status['reason']=='quality_worker_busy'


def test_resource_limits_are_explicit(monkeypatch,tmp_path):
    market(monkeypatch,tmp_path,slots=1)
    monkeypatch.setattr(model,'MAX_CANDIDATES',1)
    result=call()
    assert not result and result.search_status['state']=='UNKNOWN'


def test_old_successful_search_does_not_call_quality(monkeypatch):
    from tests.test_objective_integration import setup_builder
    setup_builder(monkeypatch,[({'speed_modifier':5},1)],[container()])
    monkeypatch.setattr(quality,'run_model',lambda *a,**kw:(_ for _ in ()).throw(AssertionError()))
    assert b.build_artifact_sets(10,priorities={'speed_modifier':1})


def test_free_owned_instance_is_optional_and_can_win(monkeypatch,tmp_path):
    _,path,_=market(monkeypatch,tmp_path,[({'speed_modifier':1},2)],slots=1)
    import sqlite3
    with sqlite3.connect(path) as c:
        c.execute('create table artefacts(id text,name_ru text,name_en text,subcategory text,stats text,bonus_stats text,icon_url text)')
        c.execute('insert into artefacts values(?,?,?,?,?,?,?)',('owned','own','','',json.dumps([{'values':{'speed_modifier':12}}]),'[]',''))
    owned=[dict(item_id='owned',quality=0,ptn=0,art_percent=100,bonus_ids=[],instance_id='mine')]
    result=call(owned_artifacts=owned)
    assert result and result[0].total_price==0 and result[0].artifacts[0].is_owned
    other=dict(owned[0],quality=1)
    ambiguous=call(owned_artifacts=owned+[other])
    assert not ambiguous and ambiguous.search_status['models'][0]['reason']=='ambiguous_owned_identity'
    assert call(owned_artifacts=[dict(owned[0],item_id='missing')]).search_status['reason']=='unknown_owned_artifact'


@pytest.mark.parametrize('error',[ImportError,RuntimeError,MemoryError])
def test_worker_errors_keep_earlier_incumbent_without_proof(monkeypatch,tmp_path,error):
    source=tmp_path/'input.json';output=tmp_path/'output.json'
    source.write_text(json.dumps(dict(memory_bytes=model.MEMORY_BYTES)))
    output.write_text(json.dumps(dict(stages=[dict(indices=[0])],optimal=False)))
    monkeypatch.setattr(worker,'memory_limit',lambda *a:None)
    monkeypatch.setattr(worker,'solve',lambda *a,**kw:(_ for _ in ()).throw(error()))
    monkeypatch.setattr('sys.argv',['worker',str(source),str(output)])
    worker.main();result=json.loads(output.read_text())
    assert result['stages'][0]['indices']==[0] and not result['optimal'] and not result['infeasible']
    assert result['reason']=={ImportError:'dependency_unavailable',RuntimeError:'worker_error',MemoryError:'memory_limit'}[error]


def test_isolated_process_is_killed_on_deadline_and_memory_limit():
    opt=option(1,{'speed_modifier':1},1)
    data=model.make_model([opt],container(),Objective.create({'speed_modifier':1}),10,{},{},True,True)
    early=process.run_model(data,time.monotonic()+.05)
    assert early['reason']=='request_time_limit' and not early['optimal']
    assert 'worker_pid' in early and process.resident_bytes(early['worker_pid']) in (None,0)
    small=process.run_model(dict(data,memory_bytes=1),time.monotonic()+5)
    assert not small.get('optimal') and small.get('reason') in ('memory_limit','worker_failed','worker_error')


def test_real_parallel_quality_requests_are_bounded(monkeypatch,tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    market(monkeypatch,tmp_path,slots=1);barrier=threading.Barrier(2)
    def run():barrier.wait();return call()
    with ThreadPoolExecutor(2) as executor:results=list(executor.map(lambda _:run(),range(2)))
    assert sorted(r.search_status['state'] for r in results)==['OPTIMAL','UNKNOWN']
    assert any(r.search_status['reason']=='quality_worker_busy' for r in results)


def test_os_lock_is_shared_between_processes():
    import subprocess,sys
    with process.quality_slot() as acquired:
        assert acquired
        command='from app._max_quality_process import quality_slot\nwith quality_slot() as ok: print(ok)'
        result=subprocess.run([sys.executable,'-c',command],capture_output=True,text=True,check=True)
        assert result.stdout.strip()=='False'


@pytest.mark.parametrize('enabled',[False,True])
def test_explicit_api_and_ui_gate_and_timeout_forwarding(monkeypatch,tmp_path,enabled):
    import importlib,shutil
    from pathlib import Path
    from app import database
    from fastapi.testclient import TestClient
    monkeypatch.setattr(database,'init_db',lambda:None)
    main=importlib.import_module('app.main')
    original=Path(main.__file__).parent/'templates/index.html'
    target=tmp_path/'app/templates';target.mkdir(parents=True);shutil.copy2(original,target/'index.html')
    monkeypatch.setattr(main,'__file__',str(tmp_path/'app/main.py'))
    monkeypatch.setattr(main.Config,'DATA_DIR',str(tmp_path/'data'))
    monkeypatch.setattr(main,'MAX_QUALITY_RELEASED',enabled)
    monkeypatch.setattr(main,'check_auction_freshness',lambda *a:dict(is_fresh=True,last_collected=None,age_hours=0))
    calls=[]
    monkeypatch.setattr(quality,'build_max_quality_sets',lambda **kw:calls.append(kw) or quality.QualityResults(reason='test'))
    client=TestClient(main.app)
    response=client.get('/api/builds',params=dict(max_quality='true',priorities='{"speed_modifier":10}',quality_timeout=12))
    assert response.status_code==(200 if enabled else 422)
    assert ('Максимальное качество (MILP)' in client.get('/').text)==enabled
    if enabled:
        assert calls[-1]['quality_timeout']==12
        assert calls[-1]['optimization_mode']=='balanced'
        assert response.json()['search_status']['state']=='UNKNOWN'
        exported=client.get('/api/export-lunarzone',params=dict(max_quality='true',priorities='{"speed_modifier":10}',quality_timeout=12,allow_duplicates='true',owned='[{"item_id":"owned","quality":1,"art_percent":0}]'))
        assert exported.status_code==404 and calls[-1]['quality_timeout']==12 and calls[-1]['allow_duplicates']
        assert calls[-1]['owned_artifacts'][0]['art_percent']==115
    else:assert not calls


@pytest.mark.parametrize('mode',['balanced','lexicographic'])
def test_primary_near_bin_boundary_is_not_relaxed_for_cheaper_price(monkeypatch,tmp_path,mode):
    market(monkeypatch,tmp_path,[({'speed_modifier':10.0000005,'health_bonus':0},8),
        ({'speed_modifier':10.00000049,'health_bonus':100},1)],slots=1)
    priorities={'speed_modifier':10} if mode=='balanced' else {'speed_modifier':10,'health_bonus':1}
    result=call(priorities=priorities,optimization_mode=mode,
        priority_order=['speed_modifier','health_bonus'] if mode=='lexicographic' else None)
    assert result and result[0].total_price==8,result.search_status
    assert result.search_status['state']=='OPTIMAL'


def test_auto_container_shortlist_is_bounded_and_reported(monkeypatch,tmp_path):
    market(monkeypatch,tmp_path,slots=1)
    monkeypatch.setattr(b,'load_container_options',lambda r:[container(str(i)) for i in range(9)])
    result=call()
    assert result and result.search_status['state']=='OPTIMAL'
    status=result.search_status
    auto=status['auto_selection']
    assert status['container_selection']=='auto'
    assert auto['considered_count']==9
    assert auto['selected_count']==8
    assert auto['selected_profile_count']==8
    assert auto['omitted_count']==1
    assert auto['omitted']==['8']
    assert 'Авто' in status['message']


def test_ui_warns_when_optimality_is_not_proven():
    from pathlib import Path
    html=(Path(b.__file__).parent/'templates/index.html').read_text(encoding='utf8')
    assert 'class="optimality-warning"' in html
    assert "d.search_status.backend === 'max_quality_milp'" in html
    assert 'd.builds.length > 0 && !d.search_status.optimality_proven' in html
    assert 'role="alert"' in html
    assert 'Найдена допустимая сборка, но её оптимальность не доказана.' in html
    assert 'Deep-поиск (300 с)' in html


def test_explicit_profile_limit_does_not_silently_drop_other_containers(monkeypatch,tmp_path):
    market(monkeypatch,tmp_path,slots=1)
    monkeypatch.setattr(b,'load_container_options',lambda r:[container(str(i)) for i in range(9)])
    result=call(container_ids=[str(i) for i in range(9)])
    assert not result and result.search_status['reason']=='profile_limit'
    assert result.search_status['container_selection']=='explicit'


def test_explicit_owned_container_pool_selects_strongest(monkeypatch,tmp_path):
    market(monkeypatch,tmp_path,[({'speed_modifier':5},1)],slots=1)
    monkeypatch.setattr(b,'load_container_options',lambda r:[container('a',1,100),container('b',1,200)])
    result=call(container_ids=['a','b'])
    assert result and result[0].container.item_id=='b'
    assert result.search_status['container_selection']=='explicit'
    assert {model['container_id'] for model in result.search_status['models']}=={'a','b'}


def test_auto_shortlist_respects_armor_profile_limit(monkeypatch,tmp_path):
    market(monkeypatch,tmp_path,slots=1)
    monkeypatch.setattr(b,'load_container_options',lambda r:[container(str(i)) for i in range(9)])
    armor=dict(id='armor',level=15,stats={'speed_modifier':1},name_ru='armor',category='',rank='',icon_url='')
    monkeypatch.setattr(b,'load_armor',lambda *a:armor)
    result=call(armor_ids=[None,'armor'])
    assert result
    auto=result.search_status['auto_selection']
    assert auto['armor_profiles']==2
    assert auto['selected_count']==4
    assert auto['selected_profile_count']==8
    assert len(result.search_status['models'])==8


def test_auto_selection_prioritizes_constraint_bonus():
    from app._quality_auto import select_auto_containers
    plain=container('plain',slots=6)
    weighted=container('weighted',slots=6,bonuses={'max_weight_bonus':{'value':81}})
    selected,meta=select_auto_containers(
        [plain,weighted],[option(1,{'speed_modifier':1},1)],
        {'speed_modifier':10},minimum={'max_weight_bonus':81},requested_limit=1)
    assert [c.item_id for c in selected]==['weighted']
    assert meta['selected_count']==1 and meta['omitted']==['plain']


def test_auto_selection_reports_candidate_budget_reduction():
    from app._quality_auto import select_auto_containers
    containers=[container(str(i),slots=1) for i in range(9)]
    candidates=[option(i,{'speed_modifier':1},1) for i in range(3)]
    selected,meta=select_auto_containers(
        containers,candidates,{'speed_modifier':10},requested_limit=8,candidate_limit=6)
    assert len(selected)==2
    assert meta['candidate_reduced'] is True
    assert meta['container_limit']==2
    assert meta['omitted_count']==7


def test_actual_missing_scipy_import_is_reported(monkeypatch,tmp_path):
    import builtins,sys
    opt=option(1,{'speed_modifier':1},1)
    data=model.make_model([opt],container(),Objective.create({'speed_modifier':1}),10,{},{},True,True)
    source=tmp_path/'input.json';output=tmp_path/'output.json';source.write_text(json.dumps(dict(data,seconds=1)))
    native=builtins.__import__
    def without_scipy(name,*a,**kw):
        if name.startswith('scipy'):raise ImportError('test dependency absent')
        return native(name,*a,**kw)
    monkeypatch.setattr(worker,'memory_limit',lambda *a:None)
    monkeypatch.setattr(builtins,'__import__',without_scipy)
    monkeypatch.setattr(sys,'argv',['worker',str(source),str(output)])
    worker.main();result=json.loads(output.read_text())
    assert result['reason']=='dependency_unavailable' and not result['optimal']


@pytest.mark.parametrize('setting',['MAX_ROWS','MAX_NONZEROS','MAX_TOTAL_CANDIDATES','MAX_INPUT_CANDIDATES'])
def test_matrix_and_preparation_caps_return_unknown(monkeypatch,tmp_path,setting):
    market(monkeypatch,tmp_path,slots=1)
    monkeypatch.setattr(model,setting,1)
    result=call()
    assert not result and result.search_status['state']=='UNKNOWN'


def test_cancellation_keeps_atomic_worker_checkpoint(monkeypatch):
    from pathlib import Path
    event=threading.Event()
    class FakeProcess:
        pid=123;returncode=None
        def __init__(self,command,**kwargs):
            Path(command[-1]).write_text(json.dumps(dict(stages=[dict(indices=[0])],optimal=False)))
            event.set()
        def poll(self):return self.returncode
        def terminate(self):self.returncode=-15
        def kill(self):self.returncode=-9
        def wait(self,**kw):return self.returncode
    monkeypatch.setattr(process.subprocess,'Popen',FakeProcess)
    monkeypatch.setattr(process,'resident_bytes',lambda *a:1)
    result=process.run_model(dict(memory_bytes=100),time.monotonic()+1,event)
    assert result['reason']=='cancelled' and result['stages'][0]['indices']==[0]
    assert not result['optimal']


def test_worker_imports_do_not_restart_absolute_deadline(tmp_path):
    data=model.make_model([option(1,{'speed_modifier':1},1)],container(),Objective.create({'speed_modifier':1}),10,{},{},True,True)
    def forbidden(*a,**kw):raise AssertionError('Expired deadline must prevent solver start')
    result=worker.solve(dict(data,seconds=60,deadline=time.monotonic()-1),tmp_path/'expired.json',solver=forbidden)
    assert not result['stages'] and result['reason']=='time_limit' and not result['infeasible']


def test_quality_export_keeps_actual_bonus_selection_and_percent(monkeypatch,tmp_path):
    import importlib
    from app import database
    from fastapi.testclient import TestClient
    monkeypatch.setattr(database,'init_db',lambda:None)
    main=importlib.import_module('app.main')
    monkeypatch.setattr(main,'MAX_QUALITY_RELEASED',True)
    artifact=option(1,{'speed_modifier':1},2);artifact.ptn=10;artifact.art_percent=100.123
    artifact.unlocked_bonus_ids=['STAMINA_BONUS','ELECTRA_DMG']
    artifact.bonus_stats=[dict(prop_id='electra_dmg_factor',name_ru='Электрозащита',unlock_ptn=5),
        dict(prop_id='speed_modifier',name_ru='Скорость',unlock_ptn=10),
        dict(prop_id='stamina_bonus',name_ru='Выносливость',unlock_ptn=15)]
    build=SimpleNamespace(artifacts=[artifact],container=None,total_price=2,total_score=1,combined_stats={},carryable=True)
    monkeypatch.setattr(quality,'build_max_quality_sets',lambda **kw:quality.QualityResults([build],state='FEASIBLE'))
    response=TestClient(main.app).get('/api/export-lunarzone',params=dict(max_quality='true',priorities='{"speed_modifier":1}'))
    assert response.status_code==200
    data=response.json()['lunarzone_json'];exported=data['build_data']['artifacts'][0]
    assert exported['selectedSecondary']==['Электрозащита','Выносливость']
    assert exported['quality']==100.123
    assert data['_optimizer_meta']['physical_items'][0]['auction_lot_id']==1
    assert data['_optimizer_meta']['search_status']['state']=='FEASIBLE'

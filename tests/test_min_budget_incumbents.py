from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from app import builder, _beam_search as beam, _min_budget_exact as exact
from app import _min_budget_incumbents as cache
from tests.test_objective_integration import setup_builder, container


def test_cache_is_bounded_immutable_and_thread_safe():
    c = cache.Incumbents(max_pools=2, max_per_pool=2)
    indices = [0, 1]
    c.remember(b'a', indices); indices[0] = 999
    assert c.get(b'a') == ((0, 1),)
    c.remember(b'a', [2]); c.remember(b'a', [3])
    assert c.get(b'a') == ((2,), (3,))
    c.remember(b'b', [1]); c.get(b'a'); c.remember(b'c', [2])
    assert c.get(b'b') == ()
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda i: c.remember(str(i % 4), [i]), range(200)))
    assert len(c._pools) <= 2 and all(len(rows) <= 2 for rows in c._pools.values())


@pytest.mark.parametrize('change', ['region','price','physical','stats','type','armor','effectiveness','budget','duplicates','guaranteed','order'])
def test_key_rejects_changed_pool_or_profile(change):
    candidates = [SimpleNamespace(item_id='a', physical_id='p', buyout_price=1, stats={'speed_modifier': 10}),
                  SimpleNamespace(item_id='b', physical_id='q', buyout_price=2, stats={})]
    c=container();args=['EU',c,candidates,100,True,False]
    original=cache.pool_key(*args)
    if change=='region':args[0]='NA'
    elif change=='budget':args[3]=99
    elif change=='duplicates':args[4]=False
    elif change=='guaranteed':args[5]=True
    elif change=='armor':c._armor={'id':'new','stats':{'health_bonus':1}}
    elif change=='effectiveness':c.effectiveness=90
    elif change=='price':candidates[0].buyout_price=2
    elif change=='physical':candidates[0].physical_id='other'
    elif change=='type':candidates[0].item_id='other'
    elif change=='stats':candidates[0].stats['speed_modifier']=11
    elif change=='order':candidates.reverse()
    assert cache.pool_key(*args)!=original


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setattr(cache, 'INCUMBENTS', cache.Incumbents())
    rows=[({'speed_modifier':25,'health_bonus':2},5),({'speed_modifier':20,'health_bonus':1},3)]
    rows += [({'speed_modifier':0},10)]*299
    setup_builder(monkeypatch,rows,[container()])
    options=builder.load_artifact_options()
    for i,o in enumerate(options):o.physical_id='p'+str(i)
    monkeypatch.setattr(beam,'search',lambda *a,**k:(np.zeros(0),np.zeros((0,a[1]),dtype=np.int32)))
    mode={'kind':'first'}
    def solve(*a,**k):
        if mode['kind']=='timeout':return None,dict(state='unknown',reason='time_limit',solver_status=1)
        item='0' if mode['kind']=='first' else '1'
        ids=np.array([next(i for i,c in enumerate(a[0]) if c.item_id==item)],dtype=np.int32)
        assert a[10](ids)
        return ids,dict(state='optimal',reason='validated_incumbent',solver_status=0)
    monkeypatch.setattr(exact,'solve',solve)
    def run(**kwargs):
        req=dict(budget=1000,tolerance_pct=0,container_id='c',priorities={'speed_modifier':10},
                 min_thresholds={'speed_modifier':20},optimization_mode='min_budget',allow_duplicates=True)
        req.update(kwargs)
        return builder.build_artifact_sets(**req)
    return run,mode,options


@pytest.mark.parametrize('flag',['0','1'])
def test_relaxed_threshold_keeps_valid_incumbent_on_timeout(replay,monkeypatch,flag):
    monkeypatch.setenv('STALZONE_DANGER_LOOKAHEAD',flag)
    run,mode,_=replay
    assert run()[0].total_price==5
    mode['kind']='timeout'
    result=run(min_thresholds={'speed_modifier':15})
    assert result and result[0].total_price==5
    assert result.search_status['state']=='feasible'
    assert not result.search_status['optimality_proven'] and not result.search_status['complete']
    model=result.search_status['models'][0]
    assert model['reused_incumbent'] and model['solver_reason']=='time_limit' and model['solver_status']==1


@pytest.mark.parametrize('change',['tighter_min','upper_bound','price','physical','owned_pool','region','budget','duplicates'])
def test_witness_compatibility_is_rechecked(replay,change):
    run,mode,options=replay
    assert run()
    mode['kind']='timeout';kw={}
    if change=='tighter_min':kw['min_thresholds']={'speed_modifier':30}
    elif change=='upper_bound':kw['max_thresholds']={'speed_modifier':21}
    elif change=='price':next(o for o in options if o.item_id=='0').buyout_price=6
    elif change=='physical':next(o for o in options if o.item_id=='0').physical_id='replaced'
    elif change=='owned_pool':options.pop()
    elif change=='region':kw['region']='NA'
    elif change=='budget':kw['budget']=4
    elif change=='duplicates':kw['allow_duplicates']=False
    result=run(**kw)
    if change=='owned_pool':
        # Removing an unrelated candidate now permits a physical remap.
        assert result and result[0].total_price==5
        assert result.search_status['models'][0]['incumbent_match']=='physical_remap'
    else:
        assert not result and result.search_status['state']=='unknown'


def test_better_current_solution_keeps_optimal_solver_status(replay):
    run,mode,_=replay
    assert run()[0].total_price==5
    mode['kind']='better'
    result=run()
    assert result[0].total_price==3
    assert result.search_status['models'][0]['state']=='optimal'
    assert 'reused_incumbent' not in result.search_status['models'][0]


def test_first_request_without_witness_stays_unknown(replay):
    run,mode,_=replay;mode['kind']='timeout'
    result=run()
    assert not result and result.search_status['state']=='unknown'

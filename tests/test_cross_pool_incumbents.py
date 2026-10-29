from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app import builder
from app._min_budget_incumbents import Incumbents, profile_key
from tests.test_min_budget_incumbents import replay
from tests.test_objective_integration import container
from tests.test_armor_search import install_armors


def option(identity, **changes):
    fields=dict(item_id=identity,physical_id=identity,auction_lot_id=None,
                buyout_price=1,stats={'speed_modifier':20},is_owned=False,
                quality=1,ptn=15,art_percent=115,stats_random=None,
                unlocked_bonus_ids=['SPEED_MOD'],base_stats=[{'value':1}],bonus_stats=[])
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_remap_uses_current_indices_and_keeps_exact_cache():
    a,b,c=map(option,['a','b','c']);store=Incumbents()
    store.remember(b'exact',(0,1));store.remember_physical(b'profile',[a,b],(0,1),[])
    assert store.get(b'exact')==((0,1),)
    assert store.remap(b'profile',[b,c,a],[])==((2,0),)


@pytest.mark.parametrize('change',['missing','price_up','price_down','stats','type','quality','ptn','percent','random','bonus_ids','base','bonus_stats','physical','ambiguous'])
def test_changed_selected_offer_is_rejected(change):
    old=option('a');current=deepcopy(old);store=Incumbents();store.remember_physical(b'p',[old],(0,),[])
    rows=[current]
    if change=='missing':rows=[]
    elif change=='price_up':current.buyout_price=2
    elif change=='price_down':current.buyout_price=0
    elif change=='stats':current.stats['speed_modifier']=21
    elif change=='type':current.item_id='other'
    elif change=='quality':current.quality=2
    elif change=='ptn':current.ptn=14
    elif change=='percent':current.art_percent=110
    elif change=='random':current.stats_random=.5
    elif change=='bonus_ids':current.unlocked_bonus_ids=[]
    elif change=='base':current.base_stats=[{'value':2}]
    elif change=='bonus_stats':current.bonus_stats=[{'value':2}]
    elif change=='physical':current.physical_id='other'
    elif change=='ambiguous':rows.append(deepcopy(current))
    assert store.remap(b'p',rows,[])==()


def test_independent_identical_offer_cannot_replace_missing_physical_item():
    a=option('a');b=option('b',item_id='a');store=Incumbents()
    store.remember_physical(b'p',[a],(0,),[])
    assert store.remap(b'p',[b],[])==()
    assert store.remap(b'p',[a,b],[])==((0,),)
    store.remember_physical(b'p',[a,a],(0,1),[])
    assert len(store._physical[b'p'])==1


@pytest.mark.parametrize('change',['append','reorder','modify_used','modify_prefix','remove','explicit_id_added'])
def test_positional_owned_requires_unchanged_prefix(change):
    owned=[{'item_id':'a','art_percent':110},{'item_id':'b','art_percent':115}]
    selected=option('b',physical_id='owned:position:1',is_owned=True,buyout_price=0)
    store=Incumbents();store.remember_physical(b'p',[selected],(0,),owned)
    current=deepcopy(owned)
    if change=='append':current.append({'item_id':'c'})
    elif change=='reorder':current.reverse()
    elif change=='modify_used':current[1]['art_percent']=114
    elif change=='modify_prefix':current[0]['art_percent']=109
    elif change=='remove':current.pop()
    elif change=='explicit_id_added':current[1]['instance_id']='new'
    assert bool(store.remap(b'p',[selected],current))==(change=='append')


def test_explicit_owned_identity_can_move_but_cannot_be_ambiguous():
    owned=[{'item_id':'a','instance_id':'first'},{'item_id':'b','instance_id':'second'}]
    selected=option('b',physical_id='owned:input:second',is_owned=True,buyout_price=0)
    store=Incumbents();store.remember_physical(b'p',[selected],(0,),owned)
    assert store.remap(b'p',[selected],list(reversed(owned)))==((0,),)
    assert store.remap(b'p',[selected],owned+[dict(owned[1])])==()
    assert store.remap(b'p',[selected],[dict(owned[1],item_id='other')])==()


@pytest.mark.parametrize('change',['region','container','slots','protection','effectiveness','bonus','armor','budget','duplicates','guaranteed','carry'])
def test_profile_never_mixes_incompatible_requests(change):
    c=container();args=['EU',c,100,True,False,True];key=profile_key(*args)
    if change=='region':args[0]='NA'
    elif change=='container':c.item_id='other'
    elif change=='slots':c.slots=2
    elif change=='protection':c.inner_protection=50
    elif change=='effectiveness':c.effectiveness=90
    elif change=='bonus':c.bonus_stats={'health_bonus':1}
    elif change=='armor':c._armor={'id':'other','level':15,'stats':{}}
    elif change=='budget':args[2]=99
    elif change=='duplicates':args[3]=False
    elif change=='guaranteed':args[4]=True
    elif change=='carry':args[5]=False
    assert profile_key(*args)!=key


def test_physical_cache_is_bounded_thread_safe_and_does_not_hold_mutable_candidates():
    store=Incumbents(max_pools=2,max_per_pool=2)
    def put(i):
        candidate=option(str(i));profile=str(i%4).encode()
        store.remember_physical(profile,[candidate],(0,),[])
        store.remap(profile,[candidate],[])
    with ThreadPoolExecutor(4) as pool:list(pool.map(put,range(100)))
    assert len(store._physical)<=2 and all(len(rows)<=2 for rows in store._physical.values())
    c=option('stable');store.remember_physical(b'latest',[c],(0,),[])
    c.stats['speed_modifier']=21
    assert store.remap(b'latest',[c],[])==()


def test_anonymous_unstable_or_oversized_proof_is_not_stored():
    store=Incumbents()
    for candidate,owned in [(option('x',physical_id=None),[]),
                            (option('x',physical_id='anonymous-row:x:0'),[]),
                            (option('x',physical_id='p'*513),[]),
                            (option('x',physical_id='owned:position:64',is_owned=True),[{'item_id':'x'}]*65),
                            (option('x',physical_id='owned:position:0',is_owned=True),[{'item_id':'x','extra':'x'*17000}])]:
        store.remember_physical(b'p',[candidate],(0,),owned)
    assert not store._physical


def test_changed_pool_reuses_old_solution_with_new_indices(replay):
    run,mode,options=replay;assert run()[0].total_price==5
    added=deepcopy(options[-1]);added.item_id='-before';added.physical_id='new';options.append(added)
    mode['kind']='timeout';result=run(min_thresholds={'speed_modifier':15})
    assert result[0].total_price==5 and result[0].artifacts[0].item_id=='0'
    assert result.search_status['models'][0]['incumbent_match']=='physical_remap'


def test_cheaper_current_solution_wins_after_pool_change(replay):
    run,mode,options=replay;assert run()[0].total_price==5
    options.pop();mode['kind']='better';result=run()
    assert result[0].total_price==3 and result.search_status['models'][0]['state']=='optimal'


@pytest.mark.parametrize('kind',['container','armor'])
def test_builder_rejects_cross_profile_reuse(replay,monkeypatch,kind):
    run,mode,options=replay
    if kind=='armor':install_armors(monkeypatch,{'first':{},'second':{}})
    first=dict(armor_id='first') if kind=='armor' else {}
    assert run(**first);mode['kind']='timeout';options.pop()
    if kind=='container':
        c=builder.load_container_options('EU')[0];other=replace(c,item_id='other')
        monkeypatch.setattr(builder,'load_container_options',lambda region:[c,other]);second=dict(container_id='other')
    else:second=dict(armor_id='second')
    assert not run(**second)

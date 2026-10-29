"""Feasibility rescue is separate from objective search and market estimates."""
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import combinations
from types import SimpleNamespace
import builtins
import json
import sqlite3

import numpy as np
import pytest

from app import builder
from app import _feasibility_fallback as rescue
from tests.test_objective_integration import setup_builder, container


def market(monkeypatch, tmp_path, rows=None, slots=2):
    rows = rows or [({'speed_modifier': 10, 'radiation_accumulation': 30}, 4),
                    ({'speed_modifier': 2, 'radiation_accumulation': -20}, 2),
                    ({'speed_modifier': 1}, 1)]
    setup_builder(monkeypatch, rows, [container(slots=slots)])
    options = builder.load_artifact_options()
    path = tmp_path / 'market.db'
    with sqlite3.connect(path) as c:
        c.execute('CREATE TABLE auction_lots(id INTEGER PRIMARY KEY,item_id TEXT,region TEXT,buyout_price INTEGER,quality INTEGER,ptn INTEGER,art_percent REAL,stats_random REAL,bonus_properties TEXT,end_time TEXT)')
        for i, o in enumerate(options):
            o.auction_lot_id = i + 1; o.physical_id = f'auction:EU:{i+1}'
            c.execute('INSERT INTO auction_lots VALUES(?,?,?,?,?,?,?,?,?,?)',
                      (i+1, o.item_id, 'EU', o.buyout_price, 0, 0, 100, None, '[]', '2099-01-01T00:00:00Z'))
    @contextmanager
    def connect():
        c = sqlite3.connect(path); c.row_factory = sqlite3.Row
        try: yield c
        finally: c.close()
    monkeypatch.setattr(builder, 'get_connection', connect)
    return options, path, connect


def empty(*a, **kw):
    return np.array([]), np.zeros((0, a[1]), dtype=np.int32)


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic'])
def test_empty_primary_rescued_and_game_revalidated(monkeypatch, tmp_path, mode):
    market(monkeypatch, tmp_path)
    builds = builder.build_artifact_sets(10, tolerance_pct=0, priorities={'speed_modifier': 10}, region='EU',
        optimization_mode=mode, priority_order=['speed_modifier'] if mode=='lexicographic' else None,
        allow_duplicates=True, _beam_search_fn=empty)
    assert builds and len(builds[0].artifacts)==2
    assert builds.search_status['state']=='FEASIBLE' and builds.search_status['fallback_attempted']
    assert not builds.search_status['optimality_proven']
    assert builds[0].carryable and builds[0].total_price<=10
    assert builder._final_valid(builds[0],10,{},{},True,False,True)


def test_successful_primary_never_calls_solver(monkeypatch, tmp_path):
    market(monkeypatch, tmp_path)
    monkeypatch.setattr(rescue, 'solve', lambda *a, **kw: pytest.fail('Successful primary must not invoke rescue'))
    builds=builder.build_artifact_sets(10, priorities={'speed_modifier':10}, region='EU',allow_duplicates=True)
    assert builds and not builds.search_status['fallback_attempted']


def test_other_valid_primary_results_prevent_rescue_when_top_expires(monkeypatch,tmp_path):
    _,path,_=market(monkeypatch,tmp_path)
    with sqlite3.connect(path) as c:c.execute("UPDATE auction_lots SET end_time='2000-01-01T00:00:00Z' WHERE id=1")
    monkeypatch.setattr(rescue,'solve',lambda *a,**kw:pytest.fail('A valid primary alternative exists'))
    result=builder.build_artifact_sets(10,priorities={'speed_modifier':10},region='EU',allow_duplicates=True,top_n=1)
    assert result and not result.search_status['fallback_attempted']
    assert {o.auction_lot_id for o in result[0].artifacts}=={2,3}


def test_research_below_first_unlock_does_not_require_bonuses(monkeypatch,tmp_path):
    opts,path,connect=market(monkeypatch,tmp_path)
    opts[0].ptn=3
    opts[0].bonus_stats=[dict(prop_id='speed_modifier',unlock_ptn=5)]
    with sqlite3.connect(path) as c:c.execute('UPDATE auction_lots SET ptn=3 WHERE id=1')
    assert len(rescue.current_candidates(opts,connect,'EU'))==3
    opts[0].ptn=5
    with sqlite3.connect(path) as c:c.execute('UPDATE auction_lots SET ptn=5 WHERE id=1')
    assert len(rescue.current_candidates(opts,connect,'EU'))==2


@pytest.mark.parametrize('change', ['deleted','price','quality','percent','random','bonus','expired','unknown_expiry','region','unsupported_tier'])
def test_stale_or_incorrect_market_descriptors_rejected(monkeypatch,tmp_path,change):
    opts,path,connect=market(monkeypatch,tmp_path)
    fields={'price':('buyout_price',999),'quality':('quality',3),'percent':('art_percent',0),
            'random':('stats_random',44),'bonus':('bonus_properties','["SPEED_MOD"]'),
            'expired':('end_time','2000-01-01T00:00:00Z'),'unknown_expiry':('end_time',''), 'region':('region','RU'),
            'unsupported_tier':('quality',6)}
    with sqlite3.connect(path) as c:
        if change=='deleted':c.execute('DELETE FROM auction_lots WHERE id=1')
        else:
            field,value=fields[change];c.execute(f'UPDATE auction_lots SET {field}=? WHERE id=1',(value,))
    assert [o.auction_lot_id for o in rescue.current_candidates(opts,connect,'EU')]==[2,3]


def test_unsupported_tier_is_not_silently_calculated_as_common(monkeypatch,tmp_path):
    opts,path,connect=market(monkeypatch,tmp_path)
    opts[0].quality=6
    with sqlite3.connect(path) as c:c.execute('UPDATE auction_lots SET quality=6 WHERE id=1')
    assert [o.auction_lot_id for o in rescue.current_candidates(opts,connect,'EU')]==[2,3]


def test_empty_conservative_request_is_explicitly_unsupported(monkeypatch,tmp_path):
    market(monkeypatch,tmp_path)
    result=builder.build_artifact_sets(10,priorities={'speed_modifier':10},region='EU',conservative=True,_beam_search_fn=empty)
    assert not result and result.search_status['state']=='UNKNOWN'
    assert result.search_status['models'][0]['reason']=='unsupported_transform'


def test_expiry_is_rechecked_after_solver(monkeypatch,tmp_path):
    _,path,_=market(monkeypatch,tmp_path)
    optimize=pytest.importorskip('scipy.optimize')
    def fake(*a,**kw):
        with sqlite3.connect(path) as c:c.execute("UPDATE auction_lots SET end_time='2000-01-01T00:00:00Z'")
        return SimpleNamespace(status=0,x=np.array([0.,1.,1.]))
    monkeypatch.setattr(optimize,'milp',fake)
    result=builder.build_artifact_sets(10, priorities={'speed_modifier':10},region='EU',_beam_search_fn=empty)
    assert not result and result.search_status['state']=='UNKNOWN'
    assert result.search_status['models'][0]['reason']=='invalid_incumbent'


@pytest.mark.parametrize('guaranteed,minimum', [(True,{}),(False,{'stopping_protection':1})])
def test_unsupported_models_remain_unknown(monkeypatch,tmp_path,guaranteed,minimum):
    market(monkeypatch,tmp_path)
    result=builder.build_artifact_sets(10,priorities={'speed_modifier':10},region='EU',
        guaranteed_carryable=guaranteed,min_thresholds=minimum,_beam_search_fn=empty)
    assert not result and result.search_status['state']=='UNKNOWN'
    assert result.search_status['models'][0]['reason']=='unsupported_transform'


def candidate(i,stats,price=1):
    return SimpleNamespace(item_id=str(i//2),physical_id=f'p{i//2}',auction_lot_id=None,stats=stats,buyout_price=price)


@pytest.mark.parametrize('seed',range(8))
@pytest.mark.parametrize('duplicates',[True,False])
@pytest.mark.parametrize('protection',[0.,.4,1.])
def test_linear_model_matches_independent_full_enumeration(seed,duplicates,protection):
    rng=np.random.default_rng(seed)
    c=[candidate(i,{'speed_modifier':int(rng.integers(-2,10)), 'radiation_accumulation':int(rng.integers(-10,25))},int(rng.integers(1,5))) for i in range(10)]
    def valid(ids):
        combo=[c[int(i)] for i in ids]
        raw=sum(o.stats['radiation_accumulation'] for o in combo)
        rad=(raw*protection if raw>0 else raw)+2
        return (len({o.physical_id for o in combo})==3 and (duplicates or len({o.item_id for o in combo})==3)
            and sum(o.buyout_price for o in combo)<=8 and 5<=sum(o.stats['speed_modifier'] for o in combo)+3<=17 and -3<=rad<=10)
    possible=any(valid(ids) for ids in combinations(range(10),3))
    ids,status=rescue.solve(c,3,8,protection,{'speed_modifier':3,'radiation_accumulation':2},
        {'speed_modifier':5,'radiation_accumulation':-3},{'speed_modifier':17},duplicates,True,valid,seconds=3)
    assert (ids is not None)==possible
    assert status['state']==('FEASIBLE' if possible else 'INFEASIBLE')
    if possible:assert valid(ids)


@pytest.mark.parametrize('kind',['timeout','error','exception','invalid','fractional','incumbent'])
def test_solver_outcomes_never_claim_user_optimality(monkeypatch,kind):
    opt=pytest.importorskip('scipy.optimize')
    calls=[]
    def fake(*a,**kw):
        calls.append(kw['options'])
        if kind=='exception':raise RuntimeError('solver failed')
        return SimpleNamespace(status=1 if kind in ('timeout','incumbent') else 4,
            x=np.array([.6 if kind=='fractional' else 1.]) if kind in ('invalid','fractional','incumbent') else None)
    monkeypatch.setattr(opt,'milp',fake)
    ids,status=rescue.solve([candidate(0,{})],1,3,1,{}, {},{},True,False,
                            lambda ids:kind!='invalid',seconds=.05)
    assert calls and 0<calls[0]['time_limit']<=.05, calls
    assert calls[0]['node_limit']==rescue.MAX_NODES
    assert status['state']==('FEASIBLE' if kind=='incumbent' else 'UNKNOWN'), status


def test_missing_solver_and_limits(monkeypatch):
    native=builtins.__import__
    def missing(name,*a,**kw):
        if name.startswith('scipy'):raise ImportError('unavailable')
        return native(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',missing)
    common=([candidate(0,{})],1,3,1,{}, {},{},True,False,lambda _:True)
    assert rescue.solve(*common,seconds=1)[1]['reason']=='solver_unavailable'
    assert rescue.solve(*common,seconds=0)[1]['reason']=='time_limit'
    monkeypatch.setattr(rescue,'MAX_CANDIDATES',0)
    assert rescue.solve(*common,seconds=1)[1]['reason']=='model_limit'


def test_armor_profiles_and_thresholds_are_not_mixed(monkeypatch,tmp_path):
    market(monkeypatch,tmp_path,rows=[({'speed_modifier':3},1),({'speed_modifier':3},1)])
    monkeypatch.setattr(builder,'load_armor',lambda name,level:dict(id=name,level=level,stats={'health_bonus':10 if name=='good' else -10}))
    builds=builder.build_artifact_sets(10,priorities={'speed_modifier':10},region='EU',armor_ids=['bad','good'],
        min_thresholds={'health_bonus':0},max_thresholds={'speed_modifier':7},_beam_search_fn=empty)
    assert builds and builds[0]._armor['id']=='good'
    assert builds[0].combined_stats['health_bonus']==10
    assert builds.search_status['models'][0]['state']=='INFEASIBLE'


def test_scope_and_ui_do_not_turn_unknown_into_budget_advice():
    from pathlib import Path
    assert rescue.SearchResults([]).search_status['state']=='UNKNOWN'
    proved=rescue.SearchResults([],[dict(state='INFEASIBLE')],attempted=True)
    assert proved.search_status['state']=='INFEASIBLE' and not proved.search_status['global_infeasibility_proven']
    incomplete=rescue.SearchResults([],[dict(state='INFEASIBLE')],attempted=True,complete=False)
    assert incomplete.search_status['state']=='UNKNOWN'
    assert {x.value for x in rescue.SearchState}=={'FEASIBLE','UNKNOWN','INFEASIBLE','OPTIMAL'}
    html=(Path(builder.__file__).parent/'templates/index.html').read_text(encoding='utf8')
    assert 'Не найдено. Увеличьте бюджет или снизьте приоритеты.' not in html


@pytest.mark.parametrize('mode',['balanced','lexicographic'])
def test_real_1111m_snapshot_false_empty_regression(monkeypatch,mode):
    import os
    from pathlib import Path
    path=os.environ.get('STALZONE_FEASIBILITY_TEST_DB')
    if not path:
        pytest.skip('Explicit isolated market snapshot required')
    path=Path(path).resolve()
    @contextmanager
    def connect():
        c=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
        try:yield c
        finally:c.close()
    monkeypatch.setattr(builder,'get_connection',connect)
    monkeypatch.setenv('STALZONE_BEAM_WIDTH','10000')
    native=rescue.current_candidates
    # Replay the historical snapshot at its recorded capture time, not today.
    stamp=datetime(2026,10,17,14,58,42,tzinfo=timezone.utc)
    monkeypatch.setattr(rescue,'current_candidates',lambda *a,**kw:native(*a,**dict(kw,now=stamp)))
    result=builder.build_artifact_sets(11000000,tolerance_pct=1,priorities={'sprint_speed_modifier':10},
        container_id='rgoz',region='EU',allow_duplicates=True,carryable_only=True,
        optimization_mode=mode,priority_order=['sprint_speed_modifier'] if mode=='lexicographic' else None,
        min_thresholds={},max_thresholds={})
    assert result and result.search_status['fallback_attempted']
    assert result.search_status['state']=='FEASIBLE' and not result.search_status['optimality_proven']
    build=result[0]
    assert len(build.artifacts)==7 and len({x.auction_lot_id for x in build.artifacts})==7
    assert build.total_price<=11110000 and build.carryable
    assert all(x.stats_random is not None for x in build.artifacts)
    assert len(native(build.artifacts,connect,'EU',builder._normalize_bonus_id,now=stamp))==7


def test_sparse_matrix_limit_and_deadline_are_enforced(monkeypatch):
    common=([candidate(0,{'speed_modifier':1})],1,3,1,{}, {},{},True,False,lambda _:True)
    monkeypatch.setattr(rescue,'MAX_NONZEROS',0)
    assert rescue.solve(*common,seconds=3)[1]['reason']=='model_limit'


def test_owned_instances_can_rescue_without_any_market_rows(monkeypatch,tmp_path):
    opts,_,connect=market(monkeypatch,tmp_path)
    for i,o in enumerate(opts):
        o.auction_lot_id=None;o.is_owned=True;o.buyout_price=0;o.physical_id=f'owned:{i}'
    assert len(rescue.current_candidates(opts,connect,'EU'))==3
    opts[0].physical_id=None
    assert len(rescue.current_candidates(opts,connect,'EU'))==2


@pytest.mark.parametrize('mode',['balanced','lexicographic'])
def test_http_exposes_unknown_status_without_running_server(monkeypatch,tmp_path,mode):
    import importlib
    from app import database
    from fastapi.testclient import TestClient
    monkeypatch.setattr(database,'init_db',lambda:None)
    main=importlib.import_module('app.main')
    (tmp_path/'app').mkdir()
    monkeypatch.setattr(main,'__file__',str(tmp_path/'app/main.py'))
    monkeypatch.setattr(main.Config,'DATA_DIR',str(tmp_path/'data'))
    monkeypatch.setattr(main,'check_auction_freshness',lambda *a:{'is_fresh':True,'last_collected':None,'age_hours':0})
    calls=[]
    monkeypatch.setattr(main,'build_artifact_sets',lambda **kw:calls.append(kw) or rescue.SearchResults([],[dict(state='UNKNOWN',reason='time_or_node_limit')],attempted=True))
    params={'optimization_mode':mode,'priorities':'{"sprint_speed_modifier":10}','container_id':'rgoz','budget':11000000,'tolerance':1}
    if mode=='lexicographic':params['priority_order']='["sprint_speed_modifier"]'
    r=TestClient(main.app).get('/api/builds',params=params)
    assert r.status_code==200
    assert r.json()['search_status']['state']=='UNKNOWN'
    assert r.json()['builds_found']==0
    assert calls[0]['optimization_mode']==mode

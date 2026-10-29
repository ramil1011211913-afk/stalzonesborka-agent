"""Stock sources stay physical and are filtered before all optimizers."""
import copy
import dataclasses
import importlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest
from app import builder as b, _beam_search as beam, _max_quality as quality
from tests.test_feasibility_fallback import market, empty
from tests.test_max_quality import call


def set_level(options, path, index, level):
    o = options[index]
    o.ptn = level
    o.bonus_stats = [dict(prop_id=f'bonus{i}') for i in range(3)]
    o.unlocked_bonus_ids = [f'BONUS{i}' for i in range(level // 5)]
    with sqlite3.connect(path) as c:
        c.execute('update auction_lots set ptn=?, bonus_properties=? where id=?',
                  (level, json.dumps(o.unlocked_bonus_ids), o.auction_lot_id))


@pytest.mark.parametrize('level', range(1, 16))
def test_every_upgraded_level_excluded_without_mutation(monkeypatch, tmp_path, level):
    opts, path, _ = market(monkeypatch, tmp_path, [({'speed_modifier': 20}, 1), ({'speed_modifier': 2}, 2)], slots=1)
    set_level(opts, path, 0, level)
    before = copy.deepcopy([dataclasses.asdict(o) for o in opts])
    result = b.build_artifact_sets(10, region='EU', priorities={'speed_modifier': 10}, stock_only=True)
    assert result and result[0].artifacts[0].auction_lot_id == 2
    assert all(o.ptn == 0 for build in result for o in build.artifacts)
    after = [dataclasses.asdict(o) for o in opts]
    for d in before + after:d.pop('score')  # Existing ranking mutates only this internal field.
    assert before == after
    assert result[0]._search_request['stock_only'] is True


def test_original_source_checked_not_scenario_level(monkeypatch, tmp_path):
    opts, path, _ = market(monkeypatch, tmp_path, [({'speed_modifier': 20}, 1), ({'speed_modifier': 2}, 2)], slots=1)
    set_level(opts, path, 0, 15)
    source = opts[0]
    fake_view = dataclasses.replace(source, ptn=0)
    fake_view._quality_source = source
    opts[0] = fake_view
    result = b.build_artifact_sets(10, region='EU', priorities={'speed_modifier': 10}, stock_only=True)
    assert result and result[0].artifacts[0].auction_lot_id == 2
    assert source.ptn == 15


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic', 'min_budget', 'rescue', 'quality', 'deep'])
@pytest.mark.parametrize('conservative', [False, True])
def test_search_modes_filter_before_solver_and_preserve_constraints(monkeypatch, tmp_path, mode, conservative):
    opts, path, _ = market(monkeypatch, tmp_path,
        [({'speed_modifier': 20}, 1), ({'speed_modifier': 2, 'radiation_accumulation': 10}, 2)], slots=1)
    set_level(opts, path, 0, 15)
    args = dict(budget=10, tolerance_pct=0, region='EU', priorities={'speed_modifier': 10},
                stock_only=True, conservative=conservative, min_thresholds={'speed_modifier': 2})
    if mode in ('quality', 'deep'):
        result = call(**args, deep_optimization=mode == 'deep')
    else:
        if mode == 'lexicographic':args.update(optimization_mode=mode, priority_order=['speed_modifier'])
        if mode == 'min_budget':args.update(min_budget=True)
        if mode == 'rescue':
            args['_beam_search_fn'] = empty
            if conservative:
                # Existing rescue does not support conservative; no invented answer.
                result = b.build_artifact_sets(**args)
                assert not result and result.search_status['state'] == 'UNKNOWN'
                return
        result = b.build_artifact_sets(**args)
    assert result and all(o.ptn == 0 and o.auction_lot_id == 2 for build in result for o in build.artifacts)
    assert result[0].carryable and result[0].combined_stats['radiation_accumulation'] == 10
    assert b._final_valid(result[0], 10, {'speed_modifier': 2}, {}, True, False, True)


@pytest.mark.parametrize('mode', ['beam', 'quality', 'deep'])
def test_not_enough_distinct_stock_instances_returns_no_build(monkeypatch, tmp_path, mode):
    opts, path, _ = market(monkeypatch, tmp_path, [({'speed_modifier': 20}, 1), ({'speed_modifier': 2}, 2)], slots=2)
    set_level(opts, path, 0, 15)
    engine = call if mode != 'beam' else b.build_artifact_sets
    kw = dict(stock_only=True, allow_duplicates=True)
    if mode == 'deep':kw['deep_optimization'] = True
    assert not engine(budget=10, **kw)


def test_repeated_types_allowed_but_not_repeated_physical_instance(monkeypatch, tmp_path):
    opts, path, _ = market(monkeypatch, tmp_path, [({'speed_modifier': 2}, 2), ({'speed_modifier': 3}, 2)], slots=2)
    opts[1].item_id = opts[0].item_id
    with sqlite3.connect(path) as c:c.execute('update auction_lots set item_id=?', (opts[0].item_id,))
    for engine in [b.build_artifact_sets, call]:
        result = engine(budget=10, stock_only=True, allow_duplicates=True)
        assert result and len({o.auction_lot_id for o in result[0].artifacts}) == 2
        assert not engine(budget=10, stock_only=True, allow_duplicates=False)
    opts[1].auction_lot_id = opts[0].auction_lot_id
    opts[1].physical_id = opts[0].physical_id
    assert not b.build_artifact_sets(10, stock_only=True, allow_duplicates=True)


@pytest.mark.parametrize('quality_mode', [False, True])
def test_owned_filter_preserves_input_and_positional_identity(monkeypatch, tmp_path, quality_mode):
    opts, path, _ = market(monkeypatch, tmp_path, [({'speed_modifier': 1}, 5)], slots=1)
    with sqlite3.connect(path) as c:
        c.execute('create table artefacts(id text,name_ru text,name_en text,subcategory text,stats text,bonus_stats text,icon_url text)')
        c.execute('insert into artefacts values(?,?,?,?,?,?,?)', ('owned', 'own', '', '', json.dumps(opts[0].base_stats), '[]', ''))
    owned = [dict(item_id='owned', quality=0, ptn=15, art_percent=100),
             dict(item_id='owned', quality=0, art_percent=100)]  # omitted ptn retains existing +0 contract
    before = copy.deepcopy(owned)
    engine = call if quality_mode else b.build_artifact_sets
    result = engine(budget=10, owned_artifacts=owned, stock_only=True)
    assert result and result[0].total_price == 0
    o = result[0].artifacts[0]
    assert o.is_owned and o.ptn == 0 and o.physical_id == 'owned:position:1'
    assert owned == before


def test_loader_checks_raw_sql_level_including_null(monkeypatch, tmp_path):
    path = tmp_path / 'raw.db'
    with sqlite3.connect(path) as c:
        c.execute('create table artefacts(id text,name_ru text,name_en text,subcategory text,stats text,bonus_stats text,icon_url text)')
        c.execute('insert into artefacts values(?,?,?,?,?,?,?)', ('a', 'a', '', '', json.dumps([dict(prop_id='speed_modifier', min=1, max=2, is_positive=True)]), '[]', ''))
        c.execute('create table auction_lots(id integer,item_id text,region text,quality integer,ptn integer,buyout_price integer,bonus_properties text,art_percent real,stats_random real)')
        for i, level in enumerate([None, *range(16)]):c.execute('insert into auction_lots values(?,?,?,?,?,?,?,?,?)', (i+1, 'a', 'EU', 0, level, 1, '[]', 100, None))
        c.execute('create table auction_history(item_id text,region text,quality integer,price integer)')
    @contextmanager
    def connect():
        c = sqlite3.connect(path);c.row_factory = sqlite3.Row
        try:yield c
        finally:c.close()
    monkeypatch.setattr(b, 'get_connection', connect)
    result = b.load_artifact_options(stock_only=True)
    assert len(result) == 1 and result[0].auction_lot_id == 2 and result[0].ptn == 0
    assert b.load_artifact_options()  # old loader behavior retained


def api(monkeypatch, tmp_path):
    from app import database
    from fastapi.testclient import TestClient
    monkeypatch.setattr(database, 'init_db', lambda: None)
    main = importlib.import_module('app.main')
    monkeypatch.setattr(main.Config, 'DATA_DIR', str(tmp_path / 'api-data'))
    monkeypatch.setattr(main, 'MAX_QUALITY_RELEASED', True)
    monkeypatch.setattr(main, 'check_auction_freshness', lambda *a: dict(is_fresh=True, last_collected=None, age_hours=0))
    monkeypatch.setattr(main, '_start_auction_refresh', lambda: pytest.fail('No refresh allowed'))
    return main, TestClient(main.app)


@pytest.mark.parametrize('value', [None, 'false', '0', 'true', '1', 'invalid'])
@pytest.mark.parametrize('quality_mode', [False, True])
def test_api_validates_and_forwards_opt_in_only(monkeypatch, tmp_path, value, quality_mode):
    main, client = api(monkeypatch, tmp_path);seen=[]
    def engine(**kw):seen.append(kw);return []
    monkeypatch.setattr(main, 'build_artifact_sets', engine)
    monkeypatch.setattr(quality, 'build_max_quality_sets', engine)
    params=dict(budget=10, max_quality=str(quality_mode).lower())
    if value is not None:params['stock_only']=value
    response=client.get('/api/builds',params=params)
    if value=='invalid':
        assert response.status_code==422 and not seen
    else:
        assert response.status_code==200
        assert seen and seen[0].get('stock_only', False)==(value in ('true','1'))
        if value not in ('true','1'):assert 'stock_only' not in seen[0] and 'stock_only' not in response.json()


def test_meta_stock_bypasses_unfiltered_cache_and_sweep_forwards(monkeypatch,tmp_path):
    main,client=api(monkeypatch,tmp_path)
    from app import meta_optimizer as meta
    seen=[]
    monkeypatch.setattr(meta,'get_cached_meta',lambda *a:pytest.fail('Stock request must not use generic cache'))
    def auto(**kw):
        seen.append(kw);return dict(budget=1000,style='balanced',priorities={},containers_tried=[],builds=[],elapsed_sec=0)
    monkeypatch.setattr(meta,'auto_optimize',auto)
    assert client.get('/api/meta-builds',params={'budget':1000,'stock_only':'true'}).status_code==200
    assert seen[-1]['stock_only'] is True
    assert client.get('/api/budget-sweep',params={'min_budget':1000,'max_budget':2000,'steps':2,'stock_only':'true'}).status_code==200
    assert len(seen)==3 and all(k['stock_only'] for k in seen)


@pytest.mark.parametrize('conservative',[False,True])
def test_api_real_native_stock_deep(monkeypatch,tmp_path,conservative):
    opts,path,_=market(monkeypatch,tmp_path,[({'speed_modifier':20},1),({'speed_modifier':2},2)],slots=1)
    set_level(opts,path,0,15)
    _,client=api(monkeypatch,tmp_path)
    response=client.get('/api/builds',params=dict(budget=10,tolerance=0,stock_only='true',max_quality='true',deep_optimization='true',conservative=str(conservative).lower(),priorities='{"speed_modifier":10}'))
    assert response.status_code==200
    result=response.json()
    assert result['search_status']['state']=='OPTIMAL' and result['search_status']['limits']['seconds']==300
    assert result['stock_only'] and result['builds'][0]['artifacts'][0]['auction_lot_id']==2
    log=json.loads(Path(result['log_file']).read_text(encoding='utf8'))
    assert log['request']['stock_only'] and log['request']['deep_optimization']


def test_lunar_export_forwards_stock_filter(monkeypatch,tmp_path):
    _,client=api(monkeypatch,tmp_path);seen=[]
    monkeypatch.setattr(quality,'build_max_quality_sets',lambda **kw:seen.append(kw) or [])
    response=client.get('/api/export-lunarzone',params=dict(budget=10,max_quality='true',stock_only='true',priorities='{"speed_modifier":10}'))
    assert response.status_code==404 and seen[0]['stock_only']


@pytest.mark.parametrize('engine',[b.build_artifact_sets,call])
def test_false_and_absent_flag_produce_same_payload(monkeypatch,tmp_path,engine):
    opts,path,_=market(monkeypatch,tmp_path,[({'speed_modifier':20},1),({'speed_modifier':2},2)],slots=1)
    set_level(opts,path,0,15)
    kwargs=dict(budget=10,region='EU',priorities={'speed_modifier':10})
    absent=engine(**kwargs);explicit=engine(**kwargs,stock_only=False)
    assert absent and explicit
    assert json.dumps([x.to_dict() for x in absent],sort_keys=True)==json.dumps([x.to_dict() for x in explicit],sort_keys=True)
    assert 'stock_only' not in explicit[0]._search_request


@pytest.mark.parametrize('condition',['budget','minimum','maximum','carry'])
@pytest.mark.parametrize('engine',[b.build_artifact_sets,call])
def test_stock_does_not_relax_other_constraints(monkeypatch,tmp_path,condition,engine):
    market(monkeypatch,tmp_path,[({'speed_modifier':2,'radiation_accumulation':10},2)],slots=1)
    kwargs=dict(budget=10,region='EU',stock_only=True)
    if condition=='budget':kwargs['budget']=1
    if condition=='minimum':kwargs['min_thresholds']={'speed_modifier':3}
    if condition=='maximum':kwargs['max_thresholds']={'speed_modifier':1}
    if condition=='carry':kwargs['min_thresholds']={'radiation_accumulation':11}
    assert not engine(**kwargs)


def test_stock_min_budget_cannot_reuse_upgraded_cached_incumbent(monkeypatch,tmp_path):
    opts,path,_=market(monkeypatch,tmp_path,[({'speed_modifier':5},1),({'speed_modifier':5},3)],slots=1)
    set_level(opts,path,0,15)
    args=dict(budget=10,min_budget=True,region='EU',min_thresholds={'speed_modifier':4})
    original=b.build_artifact_sets(**args)
    assert original and original[0].artifacts[0].ptn==15 and original[0].total_price==1
    stock=b.build_artifact_sets(**args,stock_only=True)
    assert stock and stock[0].artifacts[0].auction_lot_id==2 and stock[0].total_price==3


@pytest.mark.parametrize('gpu',[False,True])
def test_beam_backend_receives_stock_pool(monkeypatch,tmp_path,gpu):
    opts,path,_=market(monkeypatch,tmp_path,[({'speed_modifier':i},1) for i in [10,2,3,4]],slots=2)
    set_level(opts,path,0,15)
    if gpu:
        if beam.cp is None:pytest.skip('CuPy/GPU unavailable')
        try:
            if beam.cp.cuda.runtime.getDeviceCount()<1:pytest.skip('No GPU')
        except Exception:pytest.skip('GPU runtime unavailable')
    monkeypatch.setattr(beam,'_GPU_OK',gpu)
    result=b.build_artifact_sets(10,stock_only=True,allow_duplicates=True)
    assert result and all(o.ptn==0 and o.auction_lot_id!=1 for build in result for o in build.artifacts)

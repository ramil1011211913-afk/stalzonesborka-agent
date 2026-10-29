"""Real SQLite commits, bounded retry and strict post-solve validation."""
from contextlib import contextmanager
import dataclasses
import sqlite3
import threading

import pytest
from app import builder as b, _max_quality as quality, _quality_market as market_watch
from app._app_state import StateStore
from tests.test_conservative_quality import real_market
from tests.test_max_quality import call


def state(tmp_path, running=False):
    store = StateStore(tmp_path/'app_state.sqlite3')
    lease = store.claim('auction:EU')
    if not running:
        assert lease.finish()
    return store, lease


def solve_one(data, *args):
    return dict(optimal=True, infeasible=False, reason='finished', stages=[
        dict(stage=0,kind='priority',indices=[0],solver_status=0),
        dict(stage=1,kind='cost',indices=[0],solver_status=0)])


def commit_change(path):
    # A no-op UPDATE may write no pages and must not count as changed data.
    with sqlite3.connect(path) as c:
        c.execute("update auction_lots set end_time=case when end_time='2099-01-02T00:00:00Z' then '2099-01-03T00:00:00Z' else '2099-01-02T00:00:00Z' end")


@pytest.mark.parametrize('conservative',[False,True])
def test_stable_market_single_attempt_native_validation(monkeypatch,tmp_path,conservative):
    real_market(monkeypatch,tmp_path);state(tmp_path)
    calls=[];process=b._process_combo;final=b._final_valid
    def worker(*a):calls.append(1);return solve_one(*a)
    seen=[]
    def validate(*a,**kw):seen.append('process');return process(*a,**kw)
    def final_valid(*a,**kw):seen.append('final');return final(*a,**kw)
    monkeypatch.setattr(quality,'run_model',worker)
    monkeypatch.setattr(b,'_process_combo',validate);monkeypatch.setattr(b,'_final_valid',final_valid)
    r=call(conservative=conservative,priorities={'sprint_speed_modifier':10})
    assert r and r.search_status['state']=='OPTIMAL'
    assert len(calls)==1 and r.search_status['retry_count']==0
    assert seen==['process','final','process','final']
    assert r.search_status['worker_reason']=='finished'
    assert r.search_status['validation_reason']=='passed'


@pytest.mark.parametrize('conservative',[False,True])
@pytest.mark.parametrize('when',['preparation','before_solver','solver'])
def test_commit_forces_full_repreparation_once(monkeypatch,tmp_path,conservative,when):
    opts,path,_=real_market(monkeypatch,tmp_path);state(tmp_path)
    calls=[];preparations=[];original=b.build_artifact_sets;make=quality.limits.make_model
    replaced=[False]
    def publish():
        if replaced[0]:return
        replaced[0]=True
        # Same item/price but a new local row; never remap an old model index.
        with sqlite3.connect(path) as c:c.execute('update auction_lots set id=101 where id=1')
        opts[0]=dataclasses.replace(opts[0],auction_lot_id=101,physical_id='auction:EU:101')
    def prepare(*a,**kw):
        preparations.append(1);r=original(*a,**kw)
        if when=='preparation':publish()
        return r
    def build_model(*a,**kw):
        r=make(*a,**kw)
        if when=='before_solver':publish()
        return r
    def worker(data,*a):
        calls.append(data['pool_sha256'])
        if when=='solver':publish()
        return solve_one(data,*a)
    monkeypatch.setattr(b,'build_artifact_sets',prepare);monkeypatch.setattr(quality.limits,'make_model',build_model)
    monkeypatch.setattr(quality,'run_model',worker)
    r=call(conservative=conservative,priorities={'sprint_speed_modifier':10})
    assert r and r.search_status['state']=='OPTIMAL',r.search_status
    assert r[0].artifacts[0].auction_lot_id==101
    assert len(preparations)==2 and r.search_status['retry_count']==1
    assert len(calls)==(2 if when=='solver' else 1)
    assert r.search_status['attempts'][0]['market_changed']
    assert not r.search_status['attempts'][1]['market_changed']


@pytest.mark.parametrize('conservative',[False,True])
def test_repeated_commits_never_third_attempt(monkeypatch,tmp_path,conservative):
    _,path,_=real_market(monkeypatch,tmp_path);state(tmp_path)
    calls=[]
    def worker(*a):
        calls.append(1)
        commit_change(path)
        return solve_one(*a)
    monkeypatch.setattr(quality,'run_model',worker)
    r=call(conservative=conservative,priorities={'sprint_speed_modifier':10})
    assert not r and r.search_status['state']=='UNKNOWN'
    assert len(calls)==2 and r.search_status['retry_count']==1
    assert r.search_status['reason']=='market_changed'


def test_remaining_deadline_disallows_retry(monkeypatch,tmp_path):
    _,path,_=real_market(monkeypatch,tmp_path);state(tmp_path)
    clock=[100.];monkeypatch.setattr(quality.time,'monotonic',lambda:clock[0])
    calls=[]
    def worker(data,deadline,*a):
        calls.append(deadline)
        clock[0]=114.5
        commit_change(path)
        return solve_one(data)
    monkeypatch.setattr(quality,'run_model',worker)
    r=call(priorities={'sprint_speed_modifier':10},quality_timeout=15)
    assert not r and len(calls)==1
    assert calls[0]==114.9
    assert r.search_status['reason']=='market_changed_insufficient_time'


def test_retry_receives_original_deadline(monkeypatch,tmp_path):
    _,path,_=real_market(monkeypatch,tmp_path);state(tmp_path)
    clock=[100.];monkeypatch.setattr(quality.time,'monotonic',lambda:clock[0]);calls=[]
    def worker(data,deadline,*a):
        calls.append(deadline);clock[0]+=3
        if len(calls)==1:
            commit_change(path)
        return solve_one(data)
    monkeypatch.setattr(quality,'run_model',worker)
    r=call(priorities={'sprint_speed_modifier':10},quality_timeout=15)
    assert r and calls==[114.9,114.9]


@pytest.mark.parametrize('initial',[True,False])
def test_partial_refresh_not_accepted_or_retried(monkeypatch,tmp_path,initial):
    _,path,_=real_market(monkeypatch,tmp_path);store,lease=state(tmp_path,initial)
    calls=[]
    def worker(*a):
        calls.append(1);store.claim('auction:EU')
        commit_change(path)
        return solve_one(*a)
    monkeypatch.setattr(quality,'run_model',worker)
    r=call(priorities={'sprint_speed_modifier':10})
    assert not r and r.search_status['state']=='UNKNOWN'
    assert len(calls)==(0 if initial else 1)
    assert 'Аукцион обновился' in r.search_status['message']


def test_generation_detects_refresh_without_market_commit(monkeypatch,tmp_path):
    real_market(monkeypatch,tmp_path);store,_=state(tmp_path);calls=[]
    def worker(*a):
        calls.append(1)
        if len(calls)==1:assert store.claim('auction:EU').finish()
        return solve_one(*a)
    monkeypatch.setattr(quality,'run_model',worker)
    r=call(priorities={'sprint_speed_modifier':10})
    assert r and len(calls)==2
    before=r.search_status['attempts'][0]['market_before'];after=r.search_status['attempts'][0]['market_after']
    assert before['data_version']==after['data_version'] and before['generation']!=after['generation']


def test_per_item_commit_detected_inside_same_generation(monkeypatch,tmp_path):
    _,path,connect=real_market(monkeypatch,tmp_path);store,lease=state(tmp_path,True)
    with market_watch.MarketWatch(connect,'EU') as watch:
        assert not watch.changed()
        commit_change(path)
        assert watch.changed()
        assert watch.initial['generation']==watch.latest['generation']


@pytest.mark.parametrize('change',['price','expired','bonus','missing','region','quality','ptn','percent','random'])
@pytest.mark.parametrize('conservative',[False,True])
def test_changed_descriptors_never_return_stale_build(monkeypatch,tmp_path,change,conservative):
    _,path,_=real_market(monkeypatch,tmp_path);calls=[]
    def worker(*a):
        calls.append(1)
        with sqlite3.connect(path) as c:
            if change=='missing':c.execute('delete from auction_lots')
            else:
                col,val={'price':('buyout_price',9),'expired':('end_time','2000-01-01T00:00:00Z'),
                    'bonus':('bonus_properties','["UNKNOWN"]'),'region':('region','RU'),
                    'quality':('quality',4),'ptn':('ptn',15),'percent':('art_percent',129),'random':('stats_random',2)}[change]
                c.execute('update auction_lots set '+col+'=?',(val,))
        return solve_one(*a)
    monkeypatch.setattr(quality,'run_model',worker)
    r=call(conservative=conservative,priorities={'sprint_speed_modifier':10})
    assert not r and r.search_status['state']=='UNKNOWN' and len(calls)==1
    info=r.search_status['models'][0]
    assert info['worker_reason']=='finished' and info['validation_reason']=='physical_offer_validation_failed'


def test_no_state_file_no_creation_or_unsafe_retry(monkeypatch,tmp_path):
    _,path,_=real_market(monkeypatch,tmp_path)
    def worker(*a):
        commit_change(path)
        return solve_one(*a)
    monkeypatch.setattr(quality,'run_model',worker)
    r=call(priorities={'sprint_speed_modifier':10})
    assert not r and not (tmp_path/'app_state.sqlite3').exists()
    assert r.search_status['retry_count']==0


def test_unreadable_refresh_state_fails_closed(monkeypatch,tmp_path):
    real_market(monkeypatch,tmp_path);(tmp_path/'app_state.sqlite3').write_bytes(b'not sqlite')
    monkeypatch.setattr(quality,'run_model',lambda *a:pytest.fail('No agreed market state'))
    r=call(priorities={'sprint_speed_modifier':10})
    assert not r and r.search_status['reason']=='market_state_unavailable'


def test_native_validation_failure_is_not_solver_finished(monkeypatch,tmp_path):
    real_market(monkeypatch,tmp_path);state(tmp_path)
    monkeypatch.setattr(quality,'run_model',solve_one)
    monkeypatch.setattr(b,'_final_valid',lambda *a:False)
    r=call(priorities={'sprint_speed_modifier':10})
    assert not r and r.search_status['reason']=='game_validation_failed'
    assert r.search_status['worker_reason']=='finished' and r.search_status['retry_count']==0


def test_cancel_after_commit_never_retries(monkeypatch,tmp_path):
    _,path,_=real_market(monkeypatch,tmp_path);state(tmp_path);event=threading.Event();calls=[]
    def worker(*a):
        calls.append(1);event.set()
        commit_change(path)
        return solve_one(*a)
    monkeypatch.setattr(quality,'run_model',worker)
    r=call(priorities={'sprint_speed_modifier':10},cancel_event=event)
    assert not r and r.search_status['reason']=='cancelled' and len(calls)==1

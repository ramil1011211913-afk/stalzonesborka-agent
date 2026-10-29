import logging
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from app import _beam_search as beam
from app._objective import Objective
from app._physical_items import is_available


class PoolOOM(Exception):
    pass


class RuntimeCUDAError(Exception):
    def __init__(self, status):
        self.status = status


class DriverCUDAError(RuntimeCUDAError):
    pass


class FakeCuPy:
    """CPU-only fault injection at public CuPy operations; no CUDA required."""
    def __init__(self, fail=None, error=None):
        self.fail = fail
        self.error = error if error is not None else PoolOOM('injected OOM')
        self.events = []
        stream = SimpleNamespace(synchronize=lambda: self.events.append('sync'))
        self.cuda = SimpleNamespace(memory=SimpleNamespace(OutOfMemoryError=PoolOOM),
                                    runtime=SimpleNamespace(CUDARuntimeError=RuntimeCUDAError),
                                    driver=SimpleNamespace(CUDADriverError=DriverCUDAError),
                                    get_current_stream=lambda: stream)
        self.stream = stream

    def __getattr__(self, name):
        function = np.asarray if name == 'asnumpy' else getattr(np, name)
        if name not in ('asarray', 'where', 'lexsort', 'asnumpy'):
            return function

        def invoke(*args, **kwargs):
            self.events.append(name)
            if name == self.fail:
                self.fail = None
                raise self.error
            return function(*args, **kwargs)
        return invoke

    def get_default_memory_pool(self):
        def release(stream=None):
            assert stream is self.stream
            self.events.append('free-device')
        return SimpleNamespace(free_all_blocks=release)

    def get_default_pinned_memory_pool(self):
        return SimpleNamespace(free_all_blocks=lambda: self.events.append('free-pinned'))


def request(mode='balanced', duplicates=True):
    rows = [('a', 'auction:1', 12, -2, 18, 4), ('a', 'auction:1', 10, 1, 16, 3),
            ('a', 'auction:2', 9, 2, 16, 4), ('a', 'owned:1', 7, 0, 8, 0),
            ('b', 'auction:3', 1, 1, -10, 2), ('c', 'auction:4', 2, 2, -15, 3),
            ('d', 'auction:5', 4, 5, 0, 5), ('e', 'auction:6', 3, 3, 3, 3),
            ('f', 'auction:7', 2, 1, 0, 2)]
    candidates = [SimpleNamespace(item_id=item, physical_id=physical, buyout_price=price,
                                  stats={'speed_modifier': speed, 'health_bonus': health,
                                         'radiation_accumulation': radiation})
                  for item, physical, speed, health, radiation, price in rows]
    priorities = {'speed_modifier': 10, 'health_bonus': 8, 'radiation_accumulation': 3}
    scales = {'speed_modifier': 20, 'health_bonus': 30, 'radiation_accumulation': 20}
    objective = Objective.create(priorities, scales, mode,
                                 ['speed_modifier', 'health_bonus', 'radiation_accumulation'])
    args = (candidates, 4, 13, 2, .5, {'radiation_accumulation': 1, 'health_bonus': 2},
            scales, priorities, {'speed_modifier': 4, 'health_bonus': 0}, 8)
    kwargs = dict(objective=objective, allow_duplicates=duplicates, require_carry=True,
                  beam_width=1000, max_thresholds={'health_bonus': 20, 'radiation_accumulation': 10},
                  optimization_mode=mode, priority_order=list(objective.order), objective_tolerance=1e-6)
    return args, kwargs


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic', 'min_budget'])
@pytest.mark.parametrize('duplicates', [False, True])
@pytest.mark.parametrize('stage', ['asarray', 'where', 'lexsort', 'asnumpy'])
@pytest.mark.parametrize('lookahead', ['0', '1'])
def test_oom_at_every_stage_restarts_same_request_on_cpu(monkeypatch, caplog, mode, duplicates, stage, lookahead):
    args, kwargs = request(mode, duplicates)
    monkeypatch.setenv('STALZONE_DANGER_LOOKAHEAD', lookahead)
    monkeypatch.setattr(beam, '_GPU_OK', False)
    expected = beam.search(*args, **kwargs)
    assert len(expected[0])
    fake = FakeCuPy(stage)
    monkeypatch.setattr(beam, 'cp', fake)
    monkeypatch.setattr(beam, '_GPU_OK', True)
    impl = beam._search_impl
    calls = []

    def spy(*values, **options):
        calls.append((values, options.copy()))
        if not options['use_gpu']:
            assert fake.events[-3:] == ['sync', 'free-device', 'free-pinned']
        return impl(*values, **options)

    monkeypatch.setattr(beam, '_search_impl', spy)
    with caplog.at_level(logging.WARNING, logger=beam.__name__):
        actual = beam.search(*args, **kwargs)
    assert [options['use_gpu'] for _, options in calls] == [True, False]
    assert all(x is y for x, y in zip(calls[0][0], calls[1][0]))
    assert {k: v for k, v in calls[0][1].items() if k != 'use_gpu'} == {
        k: v for k, v in calls[1][1].items() if k != 'use_gpu'}
    assert calls[0][1]['objective'] is kwargs['objective']
    assert calls[0][1]['danger_lookahead'] == (lookahead == '1')
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=0)
    assert fake.events.count('sync') == 1 and fake.events.count(stage) >= 1
    assert beam._GPU_OK is True
    assert 'restarting full Beam search on CPU' in caplog.text
    for indices in actual[1]:
        assert is_available([args[0][i] for i in indices], duplicates)


@pytest.mark.parametrize('error', [RuntimeError('CUDA out of memory'), MemoryError('host RAM'),
                                  ValueError('bad shape'), RuntimeCUDAError(700), DriverCUDAError(1)])
def test_other_errors_are_not_hidden_or_retried(monkeypatch, error):
    fake = FakeCuPy('asarray', error)
    monkeypatch.setattr(beam, 'cp', fake)
    monkeypatch.setattr(beam, '_GPU_OK', True)
    with pytest.raises(type(error)) as caught:
        beam.search(*request()[0], **request()[1])
    assert caught.value is error
    assert 'sync' not in fake.events
    assert fake.events.count('asarray') == 1


@pytest.mark.parametrize('error', [PoolOOM(), RuntimeCUDAError(2), DriverCUDAError(2)])
def test_only_confirmed_oom_classes_trigger_retry(monkeypatch, error):
    fake = FakeCuPy('asarray', error)
    monkeypatch.setattr(beam, 'cp', fake)
    monkeypatch.setattr(beam, '_GPU_OK', True)
    args, kwargs = request()
    assert len(beam.search(*args, **kwargs)[0])
    assert fake.events.count('free-device') == 1


def test_traceback_gpu_references_released_before_cleanup_and_cpu_failure_propagates(monkeypatch):
    fake = FakeCuPy()
    monkeypatch.setattr(beam, 'cp', fake)
    monkeypatch.setattr(beam, '_GPU_OK', True)
    refs, attempts = [], []

    class Buffer:
        pass

    def allocation():
        buffer = Buffer()
        refs.append(weakref.ref(buffer))
        raise RuntimeError('original allocator failure')

    def attempt(*args, **kwargs):
        attempts.append(kwargs['use_gpu'])
        if kwargs['use_gpu']:
            try:
                allocation()
            except RuntimeError as original:
                raise PoolOOM() from original
        raise MemoryError('CPU is also full')

    def cleanup():
        assert refs[0]() is None

    monkeypatch.setattr(beam, '_search_impl', attempt)
    monkeypatch.setattr(beam, '_release_gpu_resources', cleanup)
    with pytest.raises(MemoryError, match='CPU is also full'):
        beam.search(*request()[0], **request()[1])
    assert attempts == [True, False]
    assert beam._GPU_OK is True


def test_cpu_retry_keeps_request_lookahead_even_if_environment_changes(monkeypatch):
    fake = FakeCuPy('asarray')
    monkeypatch.setattr(beam, 'cp', fake)
    monkeypatch.setattr(beam, '_GPU_OK', True)
    monkeypatch.setenv('STALZONE_DANGER_LOOKAHEAD', '1')
    release = beam._release_gpu_resources

    def cleanup():
        monkeypatch.setenv('STALZONE_DANGER_LOOKAHEAD', '0')
        release()

    impl = beam._search_impl
    flags = []

    def spy(*args, **kwargs):
        flags.append(kwargs['danger_lookahead'])
        return impl(*args, **kwargs)

    monkeypatch.setattr(beam, '_release_gpu_resources', cleanup)
    monkeypatch.setattr(beam, '_search_impl', spy)
    beam.search(*request()[0], **request()[1])
    assert flags == [True, True]


def test_preemptive_cpu_backend_never_calls_cupy(monkeypatch):
    fake = FakeCuPy('asarray', AssertionError('GPU touched'))
    monkeypatch.setattr(beam, 'cp', fake)
    monkeypatch.setattr(beam, '_GPU_OK', True)
    args, kwargs = request()
    kwargs['beam_width'] = 20001
    assert len(beam.search(*args, **kwargs)[0])
    assert fake.events == []


@pytest.mark.parametrize('status', [2, 700])
def test_cleanup_rethrows_other_cuda_errors(monkeypatch, status):
    fake = FakeCuPy('asarray')
    fake.stream.synchronize = lambda: (_ for _ in ()).throw(RuntimeCUDAError(status))
    monkeypatch.setattr(beam, 'cp', fake)
    monkeypatch.setattr(beam, '_GPU_OK', True)
    args, kwargs = request()
    if status == 2:
        assert len(beam.search(*args, **kwargs)[0])
        assert 'free-pinned' in fake.events
    else:
        with pytest.raises(RuntimeCUDAError) as error:
            beam.search(*args, **kwargs)
        assert error.value.status == 700
        assert 'free-device' not in fake.events


@pytest.mark.skipif(not beam._GPU_OK, reason='Working CUDA required')
def test_real_cupy_pool_oom_keeps_live_allocations_and_restarts_on_cpu(monkeypatch, caplog):
    cp = beam.cp
    args, kwargs = request()
    monkeypatch.setattr(beam, '_GPU_OK', False)
    expected = beam.search(*args, **kwargs)
    monkeypatch.setattr(beam, '_GPU_OK', True)
    pool = cp.get_default_memory_pool()
    old_limit = pool.get_limit()
    sentinel = cp.ones(32)
    cp.cuda.get_current_stream().synchronize()
    pool.free_all_blocks()
    live = pool.used_bytes()
    try:
        pool.set_limit(size=live + 1)
        with caplog.at_level(logging.WARNING, logger=beam.__name__):
            actual = beam.search(*args, **kwargs)
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[0], expected[0])
        assert 'restarting full Beam search on CPU' in caplog.text
        assert pool.used_bytes() == live
        np.testing.assert_array_equal(cp.asnumpy(sentinel), np.ones(32))
    finally:
        pool.set_limit(size=old_limit)
        del sentinel
        pool.free_all_blocks()


@pytest.mark.skipif(not beam._GPU_OK, reason='Working CUDA required')
def test_real_partial_gpu_allocations_die_before_cpu_retry(monkeypatch):
    cp = beam.cp
    original = cp.asarray
    refs, calls = [], []

    def allocate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 5:
            raise cp.cuda.memory.OutOfMemoryError(1, 2)
        array = original(*args, **kwargs)
        refs.append(weakref.ref(array))
        return array

    impl = beam._search_impl

    def attempt(*args, **kwargs):
        if not kwargs['use_gpu']:
            assert refs and all(ref() is None for ref in refs)
        return impl(*args, **kwargs)

    monkeypatch.setattr(cp, 'asarray', allocate)
    monkeypatch.setattr(beam, '_search_impl', attempt)
    args, kwargs = request()
    assert len(beam.search(*args, **kwargs)[0])
    assert len(calls) == 5

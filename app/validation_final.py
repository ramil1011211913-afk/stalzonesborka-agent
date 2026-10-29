"""Offline final audit. Writes only into a new audit subdirectory.

Run with Python 3.12: python -m app.validation_final quality|performance|integration
Uses immutable SQLite snapshots and never initializes or opens the live DB.
"""
import argparse
from contextlib import closing, ExitStack
from dataclasses import replace
import gc
import hashlib
import importlib
from itertools import combinations
import json
import logging
import math
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from app import builder, _beam_search as beam
from app._objective import Objective, direction
from app._physical_items import availability_groups, is_available

ROOT = Path('data/validation/final-audit-20261014')
CONFIG = json.loads(Path('data/validation/exact-test-config.json').read_text(encoding='utf-8-sig'))
SNAPSHOT = Path(CONFIG['snapshot']).resolve()


def connection():
    conn = sqlite3.connect(SNAPSHOT.as_uri() + '?mode=ro&immutable=1', uri=True)
    conn.row_factory = sqlite3.Row
    return closing(conn)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def request(**overrides):
    args = dict(budget=CONFIG['budget'], container_id=CONFIG['container_id'],
                owned_artifacts=CONFIG['owned_artifacts'], tolerance_pct=0,
                allow_duplicates=True, carryable_only=True, conservative=False,
                priorities={'sprint_speed_modifier': 10}, top_n=3, region='EU')
    args.update(overrides)
    return args


def validate(builds, args):
    for b in builds:
        assert len(b.artifacts) == b.container.slots
        assert builder._final_valid(b, args['budget'], args.get('min_thresholds'),
                                    args.get('max_thresholds'), args['carryable_only'],
                                    args.get('guaranteed_carryable', False), args['allow_duplicates'])
        assert b.to_dict()['budget_scope'] == 'artifacts'


def certificate(sargs, skwargs, container, priorities, limit=30):
    """Independent binary MILP for linear goals; one solve per armor profile.

    Lexicographic bins are locked sequentially; tolerance is 1e-4 for these
    controls to keep the bins above the MILP solver feasibility tolerance.
    Nonlinear/danger objective terms are deliberately not certified here.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix
    candidates, k, budget, max_counter, protection, bonuses = sargs[:6]
    spec = skwargs['objective']
    assert all(key not in beam._DANGER_KEYS and key != 'stopping_protection' for key in spec.keys)
    n = len(candidates)
    prices = np.array([c.buyout_price for c in candidates], dtype=float)
    rows, lower, upper = [np.ones(n), prices], [k, -np.inf], [k, budget]
    rows.append(np.array([beam._is_counter_art(c) for c in candidates], dtype=float))
    lower.append(-np.inf)
    upper.append(max_counter)

    def values(key):
        return np.array([c.stats.get(key, 0) for c in candidates], dtype=float)

    def constraint(key, bound, minimum=False):
        assert key != 'stopping_protection'
        target = bound - bonuses.get(key, 0)
        if key in beam._DANGER_KEYS[:4] and target > 0:
            assert protection > 0
            target /= protection
        rows.append(values(key))
        lower.append(target if minimum else -np.inf)
        upper.append(np.inf if minimum else target)

    if skwargs['require_carry']:
        for key, bound in zip(beam._DANGER_KEYS, beam._DANGER_THRESH):
            constraint(key, bound)
    for key, bound in sargs[8].items():
        constraint(key, bound, True)
    for key, bound in (skwargs.get('max_thresholds') or {}).items():
        constraint(key, bound)
    groups = availability_groups(candidates, skwargs['allow_duplicates'])
    for group in groups:
        row = np.zeros(n)
        row[list(group)] = 1
        rows.append(row)
        lower.append(-np.inf)
        upper.append(1)

    def solve(cost):
        return milp(c=cost, integrality=np.ones(n), bounds=Bounds(0, 1),
                    constraints=LinearConstraint(csr_matrix(np.vstack(rows)), lower, upper),
                    options={'time_limit': limit, 'mip_rel_gap': 0.0})

    statuses = []
    if spec.mode == 'lexicographic':
        for key in spec.order:
            column = values(key) * direction(key)
            result = solve(-column)
            statuses.append(int(result.status))
            if result.x is None or result.status != 0:
                break
            constant = bonuses.get(key, 0) * direction(key)
            total = column @ np.rint(result.x) + constant
            rank = math.floor(total / spec.tolerance + .5)
            rows.append(column)
            lower.append((rank - .5) * spec.tolerance - constant)
            upper.append((rank + .5) * spec.tolerance - constant - 1e-8)
    else:
        cost = prices if spec.mode == 'min_budget' else -sum(
            (coef * values(key) for key, coef in zip(spec.keys, spec.coefficients)), np.zeros(n))
        result = solve(cost)
        statuses.append(int(result.status))
    entry = dict(candidates=n, statuses=statuses, certified=all(s == 0 for s in statuses),
                 message=result.message, gap=getattr(result, 'mip_gap', None),
                 objective=spec.metadata(), availability_constraints=len(groups))
    if result.x is not None:
        indices = np.flatnonzero(result.x > .5)
        combo = [candidates[i] for i in indices]
        price = sum(c.buyout_price for c in combo)
        assert len(combo) == k and is_available(combo, skwargs['allow_duplicates'])
        effective = replace(container, bonus_stats=bonuses)
        recalculated = []
        builder._process_combo(combo, effective, price, priorities, sargs[8], recalculated,
                               effective_budget=budget, max_thresholds=skwargs.get('max_thresholds'),
                               objective=spec, require_carry=True,
                               allow_duplicates=skwargs['allow_duplicates'])
        assert recalculated, 'MILP solution failed the production final-stat check'
        best = recalculated[0]
        entry.update(rank=spec.rank(best.combined_stats, price), price=price,
                     stats={key: best.combined_stats.get(key, 0) for key in spec.keys},
                     selected_physical_ids=[c.physical_id for c in combo])
    return entry


def exhaustive_controls(output):
    """Small independent oracle including nonlinear stopping and lot conflicts."""
    from app._numba_search import search as numba_search
    rng = np.random.default_rng(20261014)
    entries = []
    for seed in range(8):
        candidates = [SimpleNamespace(item_id=str(i // 2), physical_id='p' + str(i // 2),
                      buyout_price=int(rng.integers(1, 12)),
                      stats=dict(speed_modifier=float(rng.integers(0, 12)),
                                 health_bonus=float(rng.integers(-7, 12)),
                                 radiation_accumulation=float(rng.integers(-12, 24)),
                                 tear_dmg_factor=float(rng.integers(0, 21)),
                                 stopping_protection=float(rng.integers(0, 9)))) for i in range(10)]
        armors = [dict(health_bonus=5, radiation_accumulation=-2, tear_dmg_factor=20),
                  dict(speed_modifier=3, health_bonus=-3, radiation_accumulation=1, tear_dmg_factor=100)]
        for mode in ['balanced', 'lexicographic', 'min_budget']:
            priorities = dict(speed_modifier=10, health_bonus=8, stopping_protection=2,
                              radiation_accumulation=3)
            spec = Objective.create(priorities, dict(speed_modifier=60, health_bonus=60,
                                    stopping_protection=100, radiation_accumulation=50), mode,
                                    list(priorities) if mode == 'lexicographic' else None)
            oracle = []
            for profile, bonus in enumerate(armors):
                for ids in combinations(range(10), 4):
                    opts = [candidates[i] for i in ids]
                    if len({c.physical_id for c in opts}) < 4:
                        continue
                    price = sum(c.buyout_price for c in opts)
                    raw = {key: sum(c.stats.get(key, 0) for c in opts) for key in candidates[0].stats}
                    rad = raw['radiation_accumulation']
                    raw['radiation_accumulation'] = rad * .4 if rad > 0 else rad
                    for key, value in bonus.items():
                        raw[key] = raw.get(key, 0) + value
                    raw['stopping_protection'] = 100 * (1 - 100 / (100 + raw['tear_dmg_factor']) *
                                                       (1 - raw['stopping_protection'] / 100))
                    if price > 27 or raw['health_bonus'] < 0 or raw['radiation_accumulation'] > 10:
                        continue
                    # Independent explicit formula, not spec.evaluate().
                    score = -price if mode == 'min_budget' else (
                        10 * raw['speed_modifier'] / 60 + 8 * raw['health_bonus'] / 60 +
                        2 * raw['stopping_protection'] / 100 - 3 * raw['radiation_accumulation'] / 50)
                    rank = (tuple(math.floor(((-raw[k]) if k == 'radiation_accumulation' else raw[k]) /
                                             1e-6 + .5) for k in priorities)
                            if mode == 'lexicographic' else (score,))
                    oracle.append((rank, -price, profile, tuple(ids), score))
            expected = sorted(oracle, reverse=True)
            for flag in ['0', '1']:
                for backend in ['cpu', 'gpu', 'numba']:
                    if backend == 'gpu' and not beam._GPU_OK:
                        continue
                    actual = []
                    for profile, bonus in enumerate(armors):
                        args = (candidates, 4, 27, 4, .4, bonus, dict(zip(spec.keys, spec.scales)),
                                priorities, {'health_bonus': 0}, 1000)
                        search = numba_search if backend == 'numba' else beam.search
                        with patch.dict(os.environ, STALZONE_DANGER_LOOKAHEAD=flag), patch.object(
                                beam, '_GPU_OK', backend == 'gpu'):
                            scores, combos = search(*args, objective=spec, allow_duplicates=True,
                                                  require_carry=True)
                        oracle_by_ids = {o[3]: o for o in oracle if o[2] == profile}
                        ordered = []
                        for score, combo in zip(scores, combos):
                            if combo[0] < 0:
                                continue
                            ids = tuple(sorted(int(i) for i in combo))
                            assert ids in oracle_by_ids
                            row = oracle_by_ids[ids]
                            assert math.isclose(score, row[4], abs_tol=1e-9)
                            actual.append(row)
                            ordered.append(row)
                        if ordered:
                            best = max(oracle_by_ids.values(), key=lambda row: row[:2])
                            if mode == 'balanced':
                                assert math.isclose(ordered[0][4], best[4], abs_tol=1e-9)
                                assert all(a[4] >= b[4] - 1e-9 for a, b in zip(ordered, ordered[1:]))
                            else:
                                assert ordered[0][:2] == best[:2]
                                assert [r[:2] for r in ordered] == sorted([r[:2] for r in ordered], reverse=True)
                    assert {(x[2], x[3]) for x in actual} == {(x[2], x[3]) for x in oracle}
                    assert not expected or sorted(actual, reverse=True)[0][:2] == expected[0][:2]
                    entries.append(dict(seed=seed, mode=mode, flag=flag, backend=backend,
                                        feasible=len(oracle), passed=True))
    dump(output / 'exhaustive.json', entries)


def exhaustive(output):
    exhaustive_controls(output)


def quality(output):
    from app.validation_regression import SCENARIOS
    scenarios = [s for s in SCENARIOS if s[0] not in ['stopping', 'min_constraints']]
    scenarios += [('balanced_health_floor', {'sprint_speed_modifier': 10, 'health_bonus': 8},
                   {'min_thresholds': {'health_bonus': 0}}),
                  ('lexicographic', {'sprint_speed_modifier': 10, 'health_bonus': 8},
                   dict(optimization_mode='lexicographic',
                        priority_order=['sprint_speed_modifier', 'health_bonus'],
                        objective_tolerance=1e-4, min_thresholds={'health_bonus': 0}))]
    entries = []
    with patch.object(builder, 'get_connection', connection):
        containers = {c.item_id: c for c in builder.load_container_options('EU')}
        for armor_ids in [None, ['3g6p5', '5lr71']]:
            for name, priorities, overrides in scenarios:
                for flag in ['0', '1']:
                    captured = []
                    original = beam.search
                    def spy(*args, **kwargs):
                        captured.append((args, kwargs))
                        return original(*args, **kwargs)
                    args = request(priorities=priorities, armor_ids=armor_ids, **overrides)
                    with patch.object(beam, 'search', spy), patch.dict(os.environ,
                            STALZONE_DANGER_LOOKAHEAD=flag, STALZONE_BEAM_WIDTH='10000'):
                        builds = builder.build_artifact_sets(**args)
                    validate(builds, args)
                    models = [certificate(a, k, containers[CONFIG['container_id']], priorities)
                              for a, k in captured]
                    ranks = [tuple(m['rank']) for m in models if 'rank' in m]
                    expected = max(ranks) if ranks else None
                    actual = captured[0][1]['objective'].rank(builds[0].combined_stats,
                              builds[0].total_price) if builds else None
                    match = (actual == expected if captured[0][1]['objective'].mode == 'lexicographic'
                             else actual is not None and expected is not None and
                             np.allclose(actual, expected, rtol=0, atol=1e-7))
                    entry = dict(name=name, armor_ids=armor_ids, flag=flag, models=models,
                                 certified=all(m['certified'] for m in models), match=bool(match),
                                 beam_rank=actual, model_rank=expected,
                                 beam_price=builds[0].total_price if builds else None)
                    entries.append(entry)
                    dump(output / 'milp.json', entries)
                    print(name, armor_ids, flag, 'certified', entry['certified'], 'match', match, flush=True)
    exhaustive_controls(output)


class MemorySampler:
    """Process RSS and sampled CUDA global/pool memory; cache is not a leak."""
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.rss = self.global_vram = self.pool_used = self.pool_total = 0
        self.samples = 0
        self.errors = []
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        self.rss = max(self.rss, self.process.memory_info().rss)
        if beam.cp is not None:
            try:
                free, total = beam.cp.cuda.runtime.memGetInfo()
                self.global_vram = max(self.global_vram, total - free)
                pool = beam.cp.get_default_memory_pool()
                self.pool_used = max(self.pool_used, pool.used_bytes())
                self.pool_total = max(self.pool_total, pool.total_bytes())
            except Exception as error:
                self.errors.append(str(error))
        self.samples += 1

    def loop(self):
        while not self.stop.wait(.01):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()

    def result(self):
        return dict(peak_rss_bytes=self.rss, sampled_global_vram_bytes=self.global_vram,
                    sampled_pool_used_bytes=self.pool_used, sampled_pool_reserved_bytes=self.pool_total,
                    samples=self.samples, sampling_errors=self.errors, interval_ms=10)


def memory_after():
    import psutil
    gc.collect()
    if beam.cp is not None:
        beam.cp.cuda.get_current_stream().synchronize()
    pool = beam.cp.get_default_memory_pool() if beam.cp is not None else None
    return dict(rss=psutil.Process().memory_info().rss,
                pool_used=pool.used_bytes() if pool else 0,
                pool_reserved=pool.total_bytes() if pool else 0)


def performance(output):
    """Same-process sequential requests with workload sizes, OOM, recovery."""
    entries = []
    original_loader = builder.load_artifact_options
    counts = []
    cap = None
    def loader(**kwargs):
        options = original_loader(**kwargs)
        return options if cap is None else options[:cap]
    original_search = beam.search
    def spy(*args, **kwargs):
        counts.append(len(args[0]))
        return original_search(*args, **kwargs)
    matrix = [(256, 1000, None), (768, 3000, None), (None, 10000, None),
              (None, 10000, ['3g6p5', '5lr71'])]
    with patch.object(builder, 'get_connection', connection), patch.object(
            builder, 'load_artifact_options', loader), patch.object(beam, 'search', spy):
        for backend in ['gpu', 'cpu']:
            for cap, width, armors in matrix:
                for flag in ['0', '1']:
                    args = request(armor_ids=armors)
                    times, baselines, memories, signatures = [], [], [], []
                    counts.clear()
                    with patch.object(beam, '_GPU_OK', backend == 'gpu'), patch.dict(os.environ,
                            STALZONE_DANGER_LOOKAHEAD=flag, STALZONE_BEAM_WIDTH=str(width)):
                        warm = builder.build_artifact_sets(**args)
                        validate(warm, args)
                        del warm
                        baseline = memory_after()
                        for repeat in range(6):
                            with MemorySampler() as meter:
                                start = time.perf_counter()
                                builds = builder.build_artifact_sets(**args)
                                if beam.cp is not None:
                                    beam.cp.cuda.get_current_stream().synchronize()
                                times.append(time.perf_counter() - start)
                            validate(builds, args)
                            signatures.append(json.dumps([b.to_dict() for b in builds], sort_keys=True))
                            memories.append(meter.result())
                            del builds
                            baselines.append(memory_after())
                    assert len(set(signatures)) == 1, 'Repeated queries changed serialized output'
                    entry = dict(backend=backend, loader_cap=cap, beam_width=width, flag=flag,
                                 armor_ids=armors, actual_candidate_counts=sorted(set(counts)),
                                 seconds=times, median=float(np.median(times)),
                                 p95=float(np.percentile(times, 95)), baseline=baseline,
                                 after_requests=baselines, memory=memories, deterministic=True)
                    entries.append(entry)
                    dump(output / 'performance.json', entries)
                    print(backend, cap, width, armors, flag, 'median', round(entry['median'], 3), flush=True)
        cap = 256
        args = request(armor_ids=['3g6p5', '5lr71'], priorities={'speed_modifier': 10, 'health_bonus': 8},
                       min_thresholds={'health_bonus': 0}, max_thresholds={'radiation_accumulation': 5})
        recovery = []
        with patch.dict(os.environ, STALZONE_BEAM_WIDTH='1000', STALZONE_DANGER_LOOKAHEAD='1'):
            with patch.object(beam, '_GPU_OK', False):
                cpu = json.dumps([b.to_dict() for b in builder.build_artifact_sets(**args)], sort_keys=True)
            with patch.object(beam, '_GPU_OK', True):
                initial_gpu = json.dumps([b.to_dict() for b in builder.build_artifact_sets(**args)], sort_keys=True)
                for attempt in range(10):
                    allocation = beam.cp.asarray
                    failed = []
                    def injected(*values, **kwargs):
                        if not failed:
                            failed.append(True)
                            raise beam.cp.cuda.memory.OutOfMemoryError(1024, 0, 0)
                        return allocation(*values, **kwargs)
                    with patch.object(beam.cp, 'asarray', injected), MemorySampler() as meter:
                        fallback = builder.build_artifact_sets(**args)
                    validate(fallback, args)
                    assert failed and json.dumps([b.to_dict() for b in fallback], sort_keys=True) == cpu
                    del fallback
                    restored = builder.build_artifact_sets(**args)
                    assert json.dumps([b.to_dict() for b in restored], sort_keys=True) == initial_gpu
                    assert beam._GPU_OK
                    del restored
                    recovery.append(dict(attempt=attempt, passed=True, memory=meter.result(), after=memory_after()))
                    dump(output / 'oom-recovery.json', recovery)
                def unexpected(*args, **kwargs):
                    raise RuntimeError('audit non-OOM failure')
                try:
                    with patch.object(beam.cp, 'asarray', unexpected):
                        builder.build_artifact_sets(**args)
                except RuntimeError as error:
                    assert str(error) == 'audit non-OOM failure'
                else:
                    raise AssertionError('Non-OOM error swallowed')
                assert json.dumps([b.to_dict() for b in builder.build_artifact_sets(**args)], sort_keys=True) == initial_gpu
        dump(output / 'reliability.json', dict(oom_cycles=len(recovery), non_oom_propagated=True,
                                             recovered_gpu=True, cpu_fallback_serialization_equal=True))


def integration(output):
    from fastapi.testclient import TestClient
    from app import database, meta_optimizer, lunar_bridge
    entries = []
    template = output / 'app' / 'templates' / 'index.html'
    template.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path('app/templates/index.html'), template)
    (output / 'data').mkdir(exist_ok=False)
    # Disable import-time init; route logs are redirected into this NEW directory.
    with patch.object(database, 'init_db', lambda: None):
        main = importlib.import_module('app.main')
    with ExitStack() as stack:
        stack.enter_context(patch.object(main.Config, 'DATA_DIR', str(output / 'data')))
        stack.enter_context(patch.object(builder, 'get_connection', connection))
        stack.enter_context(patch.object(lunar_bridge, 'get_connection', connection))
        stack.enter_context(patch.object(main, '__file__', str(output / 'app' / 'main.py')))
        stack.enter_context(patch.object(main, 'check_auction_freshness',
                           lambda *args: {'is_fresh': True, 'last_collected': None, 'age_hours': 0}))
        stack.enter_context(patch.dict(os.environ, STALZONE_BEAM_WIDTH='10000', STALZONE_DANGER_LOOKAHEAD='1'))
        client = TestClient(main.app)
        params = dict(budget=CONFIG['budget'], tolerance=0, container_id=CONFIG['container_id'],
                      owned=json.dumps(CONFIG['owned_artifacts']), allow_duplicates='true', top_n=3,
                      priorities=json.dumps({'sprint_speed_modifier': 10, 'health_bonus': 8}),
                      min_thresholds=json.dumps({'health_bonus': 0}), armor_id='3g6p5', armor_level=15)
        first = client.get('/api/builds', params=params)
        assert first.status_code == 200, first.text
        repeated = client.get('/api/builds', params=params)
        assert repeated.status_code == 200
        a, b = first.json(), repeated.json()
        assert a['builds'] == b['builds'] and a['builds']
        assert all(x['armor']['id'] == '3g6p5' and x['combined_stats']['health_bonus'] >= 0 for x in a['builds'])
        entries.append(dict(name='real_repeated_search', passed=True, builds=len(a['builds'])))
        other = client.get('/api/builds', params=dict(params, armor_id='5lr71', armor_level=15))
        assert other.status_code == 200 and other.json()['builds']
        assert all(x['armor']['id'] == '5lr71' and x['armor']['level'] == 15 for x in other.json()['builds'])
        entries.append(dict(name='real_armor_change', passed=True))
        exported = client.get('/api/export-lunarzone', params=params)
        assert exported.status_code == 200, exported.text
        parsed = lunar_bridge.parse_lunarzone_build(exported.json()['lunarzone_json'])
        assert parsed['armor_id'] == '3g6p5' and parsed['armor_level'] == 15
        entries.append(dict(name='real_export_upgrade_level', passed=True))
        for malformed in [{'armor_ids': '[]'}, {'armor_ids': '["5lr71"]'},
                          {'armor_id': 'does-not-exist'}, {'min_thresholds': '{'},
                          {'optimization_mode': 'lexicographic'}, {'armor_level': 16}, {'armor_level': 7}]:
            response = client.get('/api/builds', params=dict(params, **malformed))
            assert response.status_code == 422, response.text
        entries.append(dict(name='invalid_build_requests', passed=True))
        entries.append(dict(name='snapshot_armor_levels', available=[15], unavailable_level_7_status=422))
        with patch.object(meta_optimizer, 'get_cached_meta', side_effect=AssertionError('wrong cache')):
            meta = client.get('/api/meta-builds', params=dict(budget=CONFIG['budget'],
                   priorities=params['priorities'], containers=json.dumps([CONFIG['container_id']]),
                   allow_duplicates='true', armor_ids='["3g6p5","5lr71"]', top_n=2))
            assert meta.status_code == 200 and meta.json()['builds'], meta.text
        entries.append(dict(name='real_meta_armor_cache_bypass', passed=True))
        # Test cached route contract without touching/writing any old cache.
        cache_build = dict(a['builds'][0], total_price=100)
        too_expensive = dict(cache_build, total_price=10**12)
        with patch.object(meta_optimizer, 'get_cached_meta', return_value=[too_expensive, cache_build]):
            cached = client.get('/api/meta-builds', params={'budget': 1000})
            assert cached.status_code == 200 and cached.json()['cached'] is True
            assert cached.json()['builds'] == [cache_build]
        entries.append(dict(name='cached_budget_filter', passed=True))
        html = client.get('/')
        assert html.status_code == 200 and 'Бюджет артефактов' in html.text
        assert 'Цена брони и контейнера не входит' in html.text
        entries.append(dict(name='budget_scope_in_real_html', passed=True))
        # Record pre-existing permissive parsing for final audit, rather than
        # assuming that success validates malformed user constraints.
        observations = []
        with patch.object(meta_optimizer, 'auto_optimize', return_value=dict(budget=1000,
                style='balanced', priorities={}, containers_tried=[], builds=[], elapsed_sec=0)):
            for key in ['priorities', 'containers']:
                response = client.get('/api/meta-builds', params={'budget': 1000, key: '{'})
                assert response.status_code == 422
                observations.append(dict(parameter=key, malformed='{', status=response.status_code))
        entries.append(dict(name='meta_malformed_json_observation', observations=observations))
    dump(output / 'integration.json', entries)


def soak(output):
    """80 full-pool, two-armor requests, 20 per backend/Lookahead cell."""
    import psutil
    process = psutil.Process()
    report = []
    args = request(armor_ids=['3g6p5', '5lr71'])
    started = time.perf_counter()
    with patch.object(builder, 'get_connection', connection):
        for backend in ['gpu', 'cpu']:
            for flag in ['0', '1']:
                entry = dict(backend=backend, flag=flag, runs=[])
                report.append(entry)
                with patch.object(beam, '_GPU_OK', backend == 'gpu'), patch.dict(os.environ,
                        STALZONE_DANGER_LOOKAHEAD=flag, STALZONE_BEAM_WIDTH='10000'):
                    warm = builder.build_artifact_sets(**args)
                    validate(warm, args)
                    expected = hashlib.sha256(json.dumps([b.to_dict() for b in warm],
                                              sort_keys=True).encode()).hexdigest()
                    del warm
                    entry['baseline'] = memory_after()
                    for index in range(20):
                        with MemorySampler() as meter:
                            begin = time.perf_counter()
                            builds = builder.build_artifact_sets(**args)
                            if beam.cp is not None:
                                beam.cp.cuda.get_current_stream().synchronize()
                            elapsed = time.perf_counter() - begin
                        validate(builds, args)
                        assert hashlib.sha256(json.dumps([b.to_dict() for b in builds],
                                               sort_keys=True).encode()).hexdigest() == expected
                        del builds
                        entry['runs'].append(dict(seconds=elapsed, peak=meter.result(),
                                                  after=memory_after(), threads=process.num_threads()))
                        dump(output / 'soak.json', report)
                        if (index + 1) % 5 == 0:
                            print('soak', backend, flag, index + 1, entry['runs'][-1]['after'], flush=True)
                times = [r['seconds'] for r in entry['runs']]
                retained = [r['after']['rss'] for r in entry['runs']]
                entry.update(median=float(np.median(times)), p95=float(np.percentile(times, 95)),
                             retained_rss_delta=retained[-1] - retained[0],
                             retained_rss_range=max(retained) - min(retained),
                             deterministic=True)
                dump(output / 'soak.json', report)
    dump(output / 'summary.json', dict(total_requests=80, elapsed_seconds=time.perf_counter() - started))


def main():
    global SNAPSHOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['quality', 'performance', 'integration', 'soak', 'exhaustive'])
    parser.add_argument('--output-name', help='New subdirectory for a repeat; existing runs are preserved')
    parser.add_argument('--snapshot', type=Path, help='Alternative read-only auction snapshot')
    parser.add_argument('--armor-snapshot', type=Path, help='Read armor profiles from a separate immutable snapshot')
    args = parser.parse_args()
    if args.snapshot:
        SNAPSHOT = args.snapshot.resolve()
    output = ROOT / (args.output_name or args.phase)
    assert output.parent == ROOT and output.name not in ['.', '..']
    output.mkdir(exist_ok=False)
    wal = Path(str(SNAPSHOT) + '-wal')
    assert not wal.exists() or wal.stat().st_size == 0
    digest = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    metadata = dict(phase=args.phase, snapshot=str(SNAPSHOT), snapshot_sha256=digest,
                    python=platform.python_version(), platform=platform.platform(), gpu=beam._GPU_OK)
    if beam.cp is not None:
        props = beam.cp.cuda.runtime.getDeviceProperties(0)
        metadata['gpu_name'] = props['name'].decode()
        metadata['gpu_total_memory'] = props['totalGlobalMem']
    dump(output / 'metadata.json', metadata)
    logging.basicConfig(level=logging.ERROR)
    with ExitStack() as stack:
        if args.armor_snapshot:
            armor_snapshot = args.armor_snapshot.resolve()
            armor_wal = Path(str(armor_snapshot) + '-wal')
            assert not armor_wal.exists() or armor_wal.stat().st_size == 0
            armor_digest = hashlib.sha256(armor_snapshot.read_bytes()).hexdigest()
            original_armor_loader = builder.load_armor
            def separate_armor_connection():
                conn = sqlite3.connect(armor_snapshot.as_uri() + '?mode=ro&immutable=1', uri=True)
                conn.row_factory = sqlite3.Row
                return closing(conn)
            def armor_loader(*values, **kwargs):
                with patch.object(builder, 'get_connection', separate_armor_connection):
                    return original_armor_loader(*values, **kwargs)
            stack.enter_context(patch.object(builder, 'load_armor', armor_loader))
            metadata.update(armor_snapshot=str(armor_snapshot), armor_snapshot_sha256=armor_digest)
            dump(output / 'metadata.json', metadata)
        globals()[args.phase](output)
        if args.armor_snapshot:
            assert hashlib.sha256(armor_snapshot.read_bytes()).hexdigest() == armor_digest
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == digest
    metadata['snapshot_unchanged'] = True
    metadata['completed'] = True
    dump(output / 'metadata.json', metadata)


if __name__ == '__main__':
    main()

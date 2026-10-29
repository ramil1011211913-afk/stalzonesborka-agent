from itertools import combinations
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app import builder, _beam_search as beam
from app._numba_search import search as numba_search
from app._objective import Objective
from app._physical_items import availability_groups, conflict_ids, is_available, physical_key, owned_identity
from tests.test_objective_integration import setup_builder, container


def option(item, physical, value, price=1):
    return SimpleNamespace(item_id=item, physical_id=physical, buyout_price=price,
                           stats={'speed_modifier': value})


@pytest.mark.parametrize('backend', ['beam_cpu', 'beam_cuda', 'numba'])
@pytest.mark.parametrize('mode', ['balanced', 'lexicographic', 'min_budget'])
@pytest.mark.parametrize('duplicates', [False, True])
def test_physical_alternatives_match_independent_exhaustive_oracle(monkeypatch, backend, mode, duplicates):
    if backend == 'beam_cuda' and not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    monkeypatch.setattr(beam, '_GPU_OK', backend == 'beam_cuda')
    candidates = [option('a', 'auction:1', 15), option('a', 'auction:1', 14),
                  option('a', 'auction:2', 13), option('a', 'owned:1', 12, 0),
                  option('a', 'owned:2', 11, 0)]
    candidates += [option(str(i), f'auction:{i+10}', i, i+1) for i in range(1, 5)]
    objective = Objective.create({'speed_modifier': 10}, {'speed_modifier': 100}, mode,
                                 ['speed_modifier'])
    oracle = []
    for indices in combinations(range(len(candidates)), 4):
        combo = [candidates[i] for i in indices]
        # Independent rules, not the implementation helper under test.
        if len({c.physical_id for c in combo}) != 4:
            continue
        if not duplicates and len({c.item_id for c in combo}) != 4:
            continue
        price = sum(c.buyout_price for c in combo)
        stats = {'speed_modifier': sum(c.stats['speed_modifier'] for c in combo)}
        oracle.append((indices, objective.rank(stats, price), price))
    oracle.sort(key=lambda row: tuple(-v for v in row[1]) + (row[2],))
    search = numba_search if backend == 'numba' else beam.search
    scores, combos = search(candidates, 4, 100, 4, 1, {}, {'speed_modifier': 100},
                            {'speed_modifier': 10}, {}, 1000, allow_duplicates=duplicates,
                            objective=objective)
    valid = [tuple(sorted(row)) for row in combos if np.all(row >= 0)]
    assert set(valid) == {row[0] for row in oracle}
    actual = []
    for indices in valid:
        combo = [candidates[i] for i in indices]
        assert is_available(combo, duplicates)
        price = sum(c.buyout_price for c in combo)
        stats = {'speed_modifier': sum(c.stats['speed_modifier'] for c in combo)}
        actual.append((objective.rank(stats, price), price))
    assert actual == [(row[1], row[2]) for row in oracle]


@pytest.mark.parametrize('slots', [5, 6, 7])
def test_every_numba_kernel_rejects_shared_physical_identity(slots):
    candidates = [option('a', 'auction:1', 100), option('a', 'auction:1', 99)]
    candidates += [option(str(i), f'auction:{i+2}', 1) for i in range(slots - 2)]
    _, combos = numba_search(candidates, slots, 100, slots, 1, {}, {},
                             {'speed_modifier': 10}, {}, 2, allow_duplicates=True)
    assert not np.any(combos >= 0)


@pytest.mark.parametrize('slots', [2, 3, 4])
def test_builder_python_paths_and_numba_fallback_cannot_reuse_lot(monkeypatch, slots):
    setup_builder(monkeypatch, [({'speed_modifier': 10}, 1)] * slots, [container(slots=slots)])
    options = builder.load_artifact_options()
    options[1].item_id = options[0].item_id
    options[0].physical_id = options[1].physical_id = 'auction:shared'
    for i in range(2, slots):
        options[i].physical_id = f'auction:{i}'
    assert builder.build_artifact_sets(100, priorities={'speed_modifier': 10},
                                       allow_duplicates=True) == []


def test_process_combo_defends_physical_availability(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 10}, 1)] * 2, [container(slots=2)])
    options = builder.load_artifact_options()
    options[0].physical_id = options[1].physical_id = 'auction:shared'
    builds = []
    builder._process_combo(options, container(slots=2), 2, {'speed_modifier': 10}, {}, builds)
    assert builds == []


def test_builder_retains_two_independent_equal_counter_lots(monkeypatch):
    setup_builder(monkeypatch,
                  [({'speed_modifier': 10, 'radiation_accumulation': 50}, 1),
                   ({'radiation_accumulation': -20}, 1),
                   ({'radiation_accumulation': -20}, 1)], [container(slots=3)])
    options = builder.load_artifact_options()
    options[2].item_id = options[1].item_id
    for i, artifact in enumerate(options):
        artifact.physical_id = f'auction:{i}'
    builds = builder.build_artifact_sets(10, priorities={'speed_modifier': 10},
                                         allow_duplicates=True)
    assert builds and builds[0].combined_stats['radiation_accumulation'] == 10
    assert len({a.physical_id for a in builds[0].artifacts}) == 3
    assert not builder.build_artifact_sets(10, priorities={'speed_modifier': 10},
                                           allow_duplicates=False)
    options[2].physical_id = options[1].physical_id
    assert not builder.build_artifact_sets(10, priorities={'speed_modifier': 10},
                                           allow_duplicates=True)


def test_loader_keeps_source_lot_ids_across_generated_alternatives(monkeypatch):
    base = dict(id='a', name_ru='a', name_en='', subcategory='', stats='[{}]', icon_url='',
                bonus_stats=json.dumps([{'prop_id': 'health_bonus', 'unlock_ptn': 0},
                                        {'prop_id': 'speed_modifier', 'unlock_ptn': 0}]),
                quality=0, ptn=1, min_price=100, bonus_properties=None,
                art_percent=100, stats_random=None)
    rows = [dict(base, auction_lot_id=7), dict(base, auction_lot_id=8)]

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql, *args):
            if 'auction_history' in sql:
                return SimpleNamespace(fetchall=lambda: [])
            assert 'l.id AS auction_lot_id' in sql
            return SimpleNamespace(fetchall=lambda: rows)

    monkeypatch.setattr(builder, 'get_connection', Connection)
    monkeypatch.setattr(builder, 'compute_artifact_stats', lambda *args: ({}, {}, {}, None))
    options = builder.load_artifact_options(region='EU')
    assert len(options) == 6
    for lot in (7, 8):
        group = [o for o in options if o.auction_lot_id == lot]
        assert len(group) == 3 and {o.physical_id for o in group} == {f'auction:EU:{lot}'}
        assert not is_available(group[:2])
    assert is_available([options[0], options[3]])


@pytest.mark.parametrize('duplicates', [False, True])
@pytest.mark.parametrize('explicit', [False, True])
def test_owned_instances_have_independent_identity(monkeypatch, duplicates, explicit):
    setup_builder(monkeypatch, [], [container(slots=2)])
    art = dict(id='a', name_ru='a', name_en='', subcategory='',
               stats=json.dumps([{'values': {'speed_modifier': 1}}]), bonus_stats='[]', icon_url='')

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql, *args):
            return SimpleNamespace(fetchone=lambda: None if 'sqlite_master' in sql else art)

    monkeypatch.setattr(builder, 'get_connection', Connection)
    owned = [{'item_id': 'a'}, {'item_id': 'a'}]
    if explicit:
        owned[0]['instance_id'], owned[1]['instance_id'] = 'first', 'second'
    builds = builder.build_artifact_sets(0, priorities={'speed_modifier': 10},
                                         owned_artifacts=owned, allow_duplicates=duplicates)
    assert bool(builds) == duplicates
    if builds:
        assert len({a.physical_id for a in builds[0].artifacts}) == 2
        assert all(a.auction_lot_id is None for a in builds[0].artifacts)
        assert all(a['physical_id'].startswith('owned:') for a in builds[0].to_dict()['artifacts'])
    owned[0]['instance_id'] = owned[1]['instance_id'] = 'same-object'
    assert not builder.build_artifact_sets(0, priorities={'speed_modifier': 10},
                                           owned_artifacts=owned, allow_duplicates=True)


def test_physical_identity_cannot_cross_artifact_types():
    with pytest.raises(ValueError):
        conflict_ids([option('a', 'lot:1', 1), option('b', 'lot:1', 1)], False)
    first, second = option('a', 'display:1', 1), option('a', 'display:2', 1)
    first.auction_lot_id = second.auction_lot_id = 42
    assert not is_available([first, second])
    assert conflict_ids([first, second], True).tolist() == [0, 0]


def test_anonymous_candidates_are_separate_objects_not_extra_copies():
    first = SimpleNamespace(item_id='a')
    second = SimpleNamespace(item_id='a')
    assert physical_key(first) != physical_key(second)
    assert is_available([first, second])
    assert not is_available([first, first])
    assert owned_identity({'instance_id': None}, 0) == owned_identity({}, 0)
    assert owned_identity({'owned_instance_id': None}, 0) == owned_identity({}, 0)
    assert owned_identity({'instance_id': '0'}, 1)[1] != owned_identity({}, 0)[1]


@pytest.mark.parametrize('duplicates', [False, True])
def test_milp_availability_rows_exclude_alternatives_but_keep_independent_copies(duplicates):
    scipy = pytest.importorskip('scipy.optimize')
    candidates = [option('a', 'auction:1', 100), option('a', 'auction:1', 99),
                  option('a', 'auction:2', 90), option('a', 'owned:1', 80),
                  option('b', 'auction:3', 1)]
    groups = availability_groups(candidates, duplicates)
    assert (0, 1) in groups
    rows, upper = [np.ones(len(candidates))], [2]
    for group in groups:
        row = np.zeros(len(candidates))
        row[list(group)] = 1
        rows.append(row)
        upper.append(1)
    result = scipy.milp(c=-np.array([c.stats['speed_modifier'] for c in candidates]),
                        integrality=np.ones(len(candidates)), bounds=scipy.Bounds(0, 1),
                        constraints=scipy.LinearConstraint(np.array(rows),
                                                           [2] + [-np.inf] * len(groups), upper))
    assert result.status == 0
    indices = set(np.flatnonzero(result.x > .5))
    assert indices == ({0, 2} if duplicates else {0, 4})
    assert is_available([candidates[i] for i in indices], duplicates)

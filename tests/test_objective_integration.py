import importlib
import json
from types import SimpleNamespace

import pytest

from app import builder, database, _beam_search as beam, meta_optimizer
from app._objective import Objective


def setup_builder(monkeypatch, rows, containers):
    options = [builder.ArtifactOption(
        item_id=str(i), name_ru=str(i), name_en='', subcategory='', quality=0,
        quality_name='', art_percent=100, buyout_price=price, ptn=0,
        stats=stats, stats_raw=stats, stats_names={}, unlocked_bonus_ids=[],
        base_stats=[{"values": stats}],
    ) for i, (stats, price) in enumerate(rows)]

    def compute(base, bonus, ids, percent, ptn, qmin, qrange, random,
                protection=0, effectiveness=100, other_acc=None):
        stats = {key: value if key.endswith('_accumulation') else value * effectiveness / 100
                 for key, value in base[0]['values'].items()}
        return stats, stats, {}, None

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, *args): return SimpleNamespace(fetchone=lambda: None)

    monkeypatch.setattr(builder, 'load_artifact_options', lambda **kwargs: options)
    monkeypatch.setattr(builder, 'load_container_options', lambda region: containers)
    monkeypatch.setattr(builder, 'compute_artifact_stats', compute)
    monkeypatch.setattr(builder, 'get_connection', Connection)
    monkeypatch.setattr(beam, '_GPU_OK', False)


def container(name='c', slots=1, effectiveness=100, bonuses=None):
    return builder.ContainerOption(name, name, '', '', slots, 0,
                                   effectiveness=effectiveness, bonus_stats=bonuses or {})


def test_pareto_and_diversity_cannot_hide_balanced_optimum(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 10, 'radiation_accumulation': 0}, 2),
                                ({'speed_modifier': 8, 'radiation_accumulation': -20}, 1)], [container()])
    builds = builder.build_artifact_sets(10, tolerance_pct=0,
                                         priorities={'speed_modifier': 10, 'radiation_accumulation': 5}, top_n=2)
    assert builds[0].artifacts[0].item_id == '1'
    assert builds[0].priority_value == 13
    assert builds[0].total_score == 13
    assert [b.total_score for b in builds] == sorted([b.total_score for b in builds], reverse=True)
    low = SimpleNamespace(combined_stats={'speed_modifier': 8, 'radiation_accumulation': -20}, total_price=1)
    high = SimpleNamespace(combined_stats={'speed_modifier': 8, 'radiation_accumulation': 0}, total_price=1)
    objective = Objective.create({'speed_modifier': 10, 'radiation_accumulation': 5})
    assert builder._pareto_frontier([high, low], list(objective.keys), objective) == [low]


def test_carry_requirement_remains_hard_for_small_exhaustive_pools(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 10, 'radiation_accumulation': 20}, 1)], [container()])
    assert builder.build_artifact_sets(10, priorities={'speed_modifier': 10}, carryable_only=True) == []
    assert builder.build_artifact_sets(10, priorities={'speed_modifier': 10}, carryable_only=False)


def test_lexicographic_quantized_tie_survives_raw_pareto_dominance(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 10.00002, 'health_bonus': 1.00002}, 2),
                                ({'speed_modifier': 10.00001, 'health_bonus': 1.00001}, 1)], [container()])
    builds = builder.build_artifact_sets(10, priorities={'speed_modifier': 10, 'health_bonus': 8},
                                         optimization_mode='lexicographic',
                                         priority_order=['speed_modifier', 'health_bonus'],
                                         objective_tolerance=1e-4, top_n=1)
    assert builds[0].artifacts[0].item_id == '1'


def test_request_scales_shared_by_containers_and_independent_of_beam_width(monkeypatch):
    rows = [({'speed_modifier': float(i % 6), 'health_bonus': float(i % 9)}, 1) for i in range(105)]
    setup_builder(monkeypatch, rows, [container('a', 1), container('b', 2, 50, {'health_bonus': 3})])
    captured = []
    search = beam.search

    def spy(*args, **kwargs):
        captured.append(kwargs['objective'])
        return search(*args, **kwargs)

    monkeypatch.setattr(beam, 'search', spy)
    configurations = []
    for width in (1, 50):
        captured.clear()
        monkeypatch.setenv('STALZONE_BEAM_WIDTH', str(width))
        builds = builder.build_artifact_sets(10, tolerance_pct=0,
                                             priorities={'speed_modifier': 10, 'health_bonus': 8}, top_n=2)
        assert len(captured) == 2 and captured[0] is captured[1]
        configurations.append(captured[0].metadata())
        assert all(b.objective_config == captured[0].metadata() for b in builds)
    assert configurations[0] == configurations[1]


def test_zero_weight_does_not_change_candidate_request_or_heuristic(monkeypatch):
    rows = [({'sprint_speed_modifier': float(i % 6), 'health_bonus': float(i % 9)}, 1)
            for i in range(105)]
    setup_builder(monkeypatch, rows, [container('a', 4)])
    loader = builder.load_artifact_options
    calls = []

    def capture(**kwargs):
        calls.append(kwargs['priorities'])
        return loader(**kwargs)

    monkeypatch.setattr(builder, 'load_artifact_options', capture)
    monkeypatch.setenv('STALZONE_DANGER_LOOKAHEAD', '0')
    results = []
    for priorities in ({'sprint_speed_modifier': 10},
                       {'sprint_speed_modifier': 10, 'health_bonus': 0}):
        builds = builder.build_artifact_sets(10, priorities=priorities, top_n=2)
        results.append([(b.total_score, b.total_price,
                         [a.item_id for a in b.artifacts], b.objective_config) for b in builds])
    assert calls == [{'sprint_speed_modifier': 10}] * 2
    assert results[0] == results[1]


def test_api_exposes_modes_and_validates_order_before_work(monkeypatch, tmp_path):
    pytest.importorskip('fastapi', reason='API dependencies unavailable in this Python environment')
    from fastapi.testclient import TestClient
    monkeypatch.setattr(database, 'init_db', lambda: None)
    main = importlib.import_module('app.main')
    monkeypatch.setattr(main, '__file__', str(tmp_path / 'app' / 'main.py'))
    monkeypatch.setattr(main.Config, 'DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setattr(main, 'check_auction_freshness', lambda *args: {
        'is_fresh': True, 'last_collected': None, 'age_hours': 0})
    calls = []
    monkeypatch.setattr(main, 'build_artifact_sets', lambda **kwargs: calls.append(kwargs) or [])
    client = TestClient(main.app)
    bad = client.get('/api/builds', params={'optimization_mode': 'lexicographic',
                                          'priorities': json.dumps({'speed_modifier': 10})})
    assert bad.status_code == 422 and calls == []
    bad = client.get('/api/builds', params={'priorities': json.dumps({'speed_modifier': 10}),
                                          'objective_scales': json.dumps({'speed_modifier': 0})})
    assert bad.status_code == 422 and calls == []
    valid = client.get('/api/builds', params={'optimization_mode': 'lexicographic',
                                            'priorities': json.dumps({'speed_modifier': 10}),
                                            'priority_order': json.dumps(['speed_modifier'])})
    assert valid.status_code == 200
    assert calls[-1]['optimization_mode'] == 'lexicographic'
    assert calls[-1]['priority_order'] == ['speed_modifier']
    valid = client.get('/api/builds')
    assert valid.status_code == 200 and calls[-1]['optimization_mode'] == 'balanced'
    exported = client.get('/api/export-lunarzone', params={'optimization_mode': 'lexicographic',
                                                          'priorities': '{bad json'})
    assert exported.status_code == 422


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic', 'min_budget'])
def test_armor_refreshes_same_objective_without_double_counting_tear(monkeypatch, mode):
    objective = Objective.create({'stopping_protection': 10, 'health_bonus': 8},
                                 {'stopping_protection': 100, 'health_bonus': 10}, mode,
                                 ['stopping_protection', 'health_bonus'])
    builds = [builder.ArtifactBuild(combined_stats={'stopping_protection': 50,
                                                   'tear_dmg_factor': 100, 'health_bonus': 0},
                                    total_price=2, objective_config=objective.metadata())]
    monkeypatch.setattr(builder, 'load_armor', lambda *args: {'id': 'a', 'name_ru': 'a',
                                                           'stats': {'tear_dmg_factor': 100, 'health_bonus': 3}})
    builder.apply_armor_to_builds(builds, 'a')
    assert builds[0].combined_stats['stopping_protection'] == pytest.approx(100 * 2 / 3)
    assert builds[0].total_score == pytest.approx(objective.evaluate(builds[0].combined_stats, 2))
    assert builds[0].objective_config == objective.metadata()


def test_meta_uses_one_query_instead_of_comparing_different_normalizations(monkeypatch):
    monkeypatch.setattr(meta_optimizer, 'get_best_containers', lambda *args, **kwargs: [{'id': 'a'}, {'id': 'b'}])
    calls = []
    monkeypatch.setattr(meta_optimizer, 'build_artifact_sets', lambda **kwargs: calls.append(kwargs) or [])
    meta_optimizer.auto_optimize(100, priorities={'speed_modifier': 10})
    assert len(calls) == 1 and calls[0]['container_ids'] == ['a', 'b']


def test_previous_meta_cache_is_not_reused_or_deleted(monkeypatch, tmp_path):
    path = tmp_path / 'cache.json'
    old = json.dumps({'version': 1, 'builds': {}})
    path.write_text(old)
    monkeypatch.setattr(meta_optimizer, '_META_CACHE_PATH', str(path))
    assert meta_optimizer.load_meta_cache() is None
    assert path.read_text() == old
    old = json.dumps({'version': 2, 'objective_contract': 'balanced-v1', 'builds': {}})
    path.write_text(old)
    assert meta_optimizer.load_meta_cache() is None
    assert path.read_text() == old

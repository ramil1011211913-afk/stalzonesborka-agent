import json
from types import SimpleNamespace
import importlib

import pytest

from app import builder, _beam_search as beam, database, meta_optimizer
from app._objective import Objective
from tests.test_objective_integration import setup_builder, container


def install_armors(monkeypatch, rows):
    armors = {key: dict(id=key, name_ru=key, rank='master', category='combat',
                        level=15, stats=stats, icon_url='') for key, stats in rows.items()}
    monkeypatch.setattr(builder, 'load_armor', lambda key, level=15: armors.get(key))
    return armors


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic', 'min_budget'])
@pytest.mark.parametrize('lookahead', ['0', '1'])
def test_numba_search_finds_armor_enabled_build_before_final_selection(monkeypatch, mode, lookahead):
    setup_builder(monkeypatch, [({'speed_modifier': 10, 'health_bonus': -2}, 1)] * 4 +
                  [({'speed_modifier': 1, 'health_bonus': 2}, 2)] * 4, [container(slots=4)])
    install_armors(monkeypatch, {'good': {'health_bonus': 10}})
    monkeypatch.setenv('STALZONE_DANGER_LOOKAHEAD', lookahead)
    kwargs = dict(priorities={'speed_modifier': 10, 'health_bonus': 8},
                  optimization_mode=mode, priority_order=['speed_modifier', 'health_bonus'] if mode == 'lexicographic' else None,
                  min_thresholds={'health_bonus': 0}, top_n=1, tolerance_pct=0)
    before = builder.build_artifact_sets(20, **kwargs)
    builds = builder.build_artifact_sets(20, armor_id='good', **kwargs)
    assert builds and builds[0].combined_stats['speed_modifier'] == 40
    assert builds[0].combined_stats['health_bonus'] == 2
    assert builds[0].total_price == 4
    assert before[0].combined_stats['speed_modifier'] < 40
    assert builds[0].to_dict()['armor']['id'] == 'good'
    assert builds[0].container.bonus_stats == {}


@pytest.mark.parametrize('backend', ['cpu', 'cuda'])
@pytest.mark.parametrize('mode', ['balanced', 'lexicographic', 'min_budget'])
@pytest.mark.parametrize('lookahead', ['0', '1'])
def test_narrow_beam_checks_worsening_armor_before_trimming(monkeypatch, backend, mode, lookahead):
    if backend == 'cuda' and not beam._GPU_OK:
        pytest.skip('Working CUDA required')
    rows = [({'speed_modifier': 20, 'health_bonus': 0}, 1),
            ({'speed_modifier': 12, 'health_bonus': 8}, 2)] + [({'speed_modifier': 1, 'health_bonus': 20}, 3)] * 103
    setup_builder(monkeypatch, rows, [container()])
    monkeypatch.setattr(beam, '_GPU_OK', backend == 'cuda')
    install_armors(monkeypatch, {'bad': {'health_bonus': -5}})
    monkeypatch.setenv('STALZONE_BEAM_WIDTH', '1')
    monkeypatch.setenv('STALZONE_DANGER_LOOKAHEAD', lookahead)
    builds = builder.build_artifact_sets(10, armor_id='bad', priorities={'speed_modifier': 10, 'health_bonus': 8},
                                         optimization_mode=mode,
                                         priority_order=['speed_modifier', 'health_bonus'] if mode == 'lexicographic' else None,
                                         min_thresholds={'health_bonus': 0}, top_n=1)
    assert builds[0].artifacts[0].item_id == '1'
    assert builds[0].combined_stats['health_bonus'] == 3


def test_armor_changes_nonlinear_stopping_optimum(monkeypatch):
    setup_builder(monkeypatch, [({'tear_dmg_factor': 200}, 1),
                                ({'stopping_protection': 50}, 2)], [container()])
    install_armors(monkeypatch, {'tear': {'tear_dmg_factor': 200}})
    assert builder.build_artifact_sets(10, priorities={'stopping_protection': 10}, top_n=1)[0].artifacts[0].item_id == '0'
    builds = builder.build_artifact_sets(10, armor_id='tear', priorities={'stopping_protection': 10}, top_n=1)
    assert builds[0].artifacts[0].item_id == '1'
    assert builds[0].combined_stats['stopping_protection'] == pytest.approx(83.3333333333)
    assert builder.apply_armor_to_builds(builds, 'tear') is builds


def test_maximum_and_carryability_rechecked_with_armor(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 10, 'health_bonus': 4, 'radiation_accumulation': 8}, 1),
                                ({'speed_modifier': 8, 'health_bonus': 0, 'radiation_accumulation': 0}, 2)], [container()])
    install_armors(monkeypatch, {'bad': {'health_bonus': 10, 'radiation_accumulation': 4},
                                 'good': {'radiation_accumulation': -10}})
    builds = builder.build_artifact_sets(10, armor_id='bad', priorities={'speed_modifier': 10},
                                         max_thresholds={'health_bonus': 12})
    assert builds[0].artifacts[0].item_id == '1'
    assert builds[0].carryable and builds[0].combined_stats['health_bonus'] == 10
    assert builder.build_artifact_sets(1, tolerance_pct=0, armor_id='bad', priorities={'speed_modifier': 10}) == []
    assert builder.build_artifact_sets(1, tolerance_pct=0, armor_id='good', priorities={'speed_modifier': 10})[0].carryable


def test_joint_armor_variants_share_normalization_and_choose_feasible_minimum(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 10, 'health_bonus': 0}, 1)] * 105, [container()])
    install_armors(monkeypatch, {'bad': {'health_bonus': -1}, 'good': {'health_bonus': 8}})
    observed = []
    search = beam.search
    def spy(*args, **kwargs):
        observed.append(kwargs['objective'])
        return search(*args, **kwargs)
    monkeypatch.setattr(beam, 'search', spy)
    builds = builder.build_artifact_sets(10, armor_ids=['bad', 'good'],
                                         priorities={'speed_modifier': 10, 'health_bonus': 8},
                                         min_thresholds={'health_bonus': 0}, optimization_mode='min_budget', top_n=1)
    assert len(observed) >= 2 and all(o is observed[0] for o in observed)
    assert builds[0]._armor['id'] == 'good'
    assert builds[0].total_price == 1
    assert builds[0].combined_stats['health_bonus'] == 8
    assert builds[0].to_dict()['budget_scope'] == 'artifacts'


def test_replace_and_remove_armor_repeat_full_search(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 10, 'health_bonus': 0}, 1),
                                ({'speed_modifier': 8, 'health_bonus': 8}, 2)], [container()])
    install_armors(monkeypatch, {'good': {'health_bonus': 8}, 'bad': {'health_bonus': -5}})
    builds = builder.build_artifact_sets(10, priorities={'speed_modifier': 10},
                                         armor_id='good', min_thresholds={'health_bonus': 0}, top_n=1)
    assert builds[0].artifacts[0].item_id == '0'
    replaced = builder.apply_armor_to_builds(builds, 'bad')
    assert replaced[0].artifacts[0].item_id == '1'
    assert replaced[0].combined_stats['health_bonus'] == 3
    removed = builder.apply_armor_to_builds(replaced, None)
    assert removed[0].artifacts[0].item_id == '0'
    assert removed[0].to_dict()['armor'] is None


def test_guaranteed_carryability_uses_artifact_endpoints_and_fixed_armor(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 1, 'radiation_accumulation': 4}, 1)], [container()])
    install_armors(monkeypatch, {'fixed': {'radiation_accumulation': 5, 'radiation_protection': 100}})
    builds = builder.build_artifact_sets(10, armor_id='fixed', guaranteed_carryable=True,
                                         priorities={'speed_modifier': 10})
    assert builds and builds[0].combined_stats['radiation_accumulation'] == 9
    assert builds[0].carryability_analysis['stats']['radiation_accumulation']['danger_max'] == 9


def test_unknown_or_singular_armor_never_silently_returns_valid_result(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 1}, 1)], [container()])
    install_armors(monkeypatch, {'singular': {'tear_dmg_factor': -100}})
    with pytest.raises(ValueError, match='not found'):
        builder.build_artifact_sets(10, armor_id='unknown')
    assert not builder.build_artifact_sets(10, armor_id='singular', priorities={'speed_modifier': 10})


def test_armor_resistance_is_a_final_stat_not_accumulation_compensation(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 1, 'radiation_accumulation': 12}, 1)], [container()])
    install_armors(monkeypatch, {'resistance': {'radiation_protection': 100}})
    assert not builder.build_artifact_sets(10, armor_id='resistance', priorities={'speed_modifier': 10})
    builds = builder.build_artifact_sets(10, armor_id='resistance', priorities={'radiation_protection': 10},
                                         min_thresholds={'radiation_protection': 50}, carryable_only=False)
    assert builds[0].combined_stats['radiation_protection'] == 100
    assert builds[0].combined_stats['radiation_accumulation'] == 12
    assert not builds[0].carryable


def test_meta_forwards_selected_armor(monkeypatch):
    monkeypatch.setattr(meta_optimizer, 'get_best_containers', lambda *a, **k: [{'id': 'c'}])
    calls = []
    monkeypatch.setattr(meta_optimizer, 'build_artifact_sets', lambda **kwargs: calls.append(kwargs) or [])
    meta_optimizer.auto_optimize(10, armor_ids=['a', 'b'], armor_level=7)
    assert calls[0]['armor_ids'] == ['a', 'b'] and calls[0]['armor_level'] == 7


def test_diversity_preserves_distinct_armors_on_same_artifacts(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 1, 'health_bonus': 1}, 1)], [container()])
    install_armors(monkeypatch, {'speed': {'speed_modifier': 5}, 'health': {'health_bonus': 5}})
    builds = builder.build_artifact_sets(10, armor_ids=['speed', 'health'],
                                         priorities={'speed_modifier': 10, 'health_bonus': 8}, top_n=2)
    assert len(builds) == 2 and {b._armor['id'] for b in builds} == {'speed', 'health'}


def test_guaranteed_constraint_is_part_of_narrow_beam_search(monkeypatch):
    rows = [({'speed_modifier': 20, 'radiation_accumulation': 12}, 1),
            ({'speed_modifier': 10, 'radiation_accumulation': 10}, 1)] + [({'speed_modifier': 1, 'radiation_accumulation': 30}, 3)] * 103
    setup_builder(monkeypatch, rows, [container()])
    for option in builder.load_artifact_options():
        option.art_percent = 85
    compute = builder.compute_artifact_stats
    def ranged(base, bonus, ids, percent, *args, **kwargs):
        stats, raw, names, other = compute(base, bonus, ids, percent, *args, **kwargs)
        stats['radiation_accumulation'] *= percent / 100
        return stats, raw, names, other
    monkeypatch.setattr(builder, 'compute_artifact_stats', ranged)
    install_armors(monkeypatch, {'fixed': {'radiation_accumulation': -1}})
    monkeypatch.setenv('STALZONE_BEAM_WIDTH', '1')
    common = dict(armor_id='fixed', priorities={'speed_modifier': 10}, top_n=1)
    assert builder.build_artifact_sets(10, **common)[0].combined_stats['speed_modifier'] == 20
    builds = builder.build_artifact_sets(10, guaranteed_carryable=True, **common)
    assert builds and builds[0].combined_stats['speed_modifier'] == 10
    assert builds[0].guaranteed_carryable


def test_standalone_armor_replacement_is_idempotent_and_refreshes_carryability(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 1, 'radiation_accumulation': 8}, 1)], [container()])
    install_armors(monkeypatch, {'bad': {'radiation_accumulation': 5}, 'good': {'radiation_accumulation': -5}})
    previews = []
    builder._process_combo(builder.load_artifact_options(), container(), 1, {}, {}, previews)
    builder.apply_armor_to_builds(previews, 'bad')
    assert not previews[0].carryable
    builder.apply_armor_to_builds(previews, 'bad')
    assert previews[0].combined_stats['radiation_accumulation'] == 13
    builder.apply_armor_to_builds(previews, 'good')
    assert previews[0].carryable and previews[0].combined_stats['radiation_accumulation'] == 3
    assert builder.apply_armor_to_builds(previews, 'bad', carryable_only=True) == []


def test_export_preserves_selected_upgrade_level_without_reloading_armor(monkeypatch):
    from app import lunar_bridge
    setup_builder(monkeypatch, [({'speed_modifier': 1}, 1)], [container()])
    armors = install_armors(monkeypatch, {'selected': {'health_bonus': 8}})
    armors['selected']['level'] = 7
    build = builder.build_artifact_sets(10, armor_id='selected', armor_level=7)[0]
    monkeypatch.setattr(builder, 'load_armor', lambda *a: pytest.fail('reloaded armor for export'))
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, *args): return SimpleNamespace(fetchone=lambda: None)
    monkeypatch.setattr(lunar_bridge, 'get_connection', Connection)
    exported = lunar_bridge.export_to_lunarzone(build)
    assert exported['build_data']['armor']['level'] == 7
    parsed = lunar_bridge.parse_lunarzone_build(exported)
    assert parsed['armor_level'] == 7
    with pytest.raises(ValueError, match='must match'):
        lunar_bridge.export_to_lunarzone(build, 'different')


def test_exact_template_calculation_preserves_prices_and_physical_identity(monkeypatch):
    from app import lunar_bridge
    setup_builder(monkeypatch, [], [container(slots=2)])
    install_armors(monkeypatch, {'good': {'health_bonus': 8}})
    art = dict(id='a', name_ru='a', name_en='', subcategory='',
               stats=json.dumps([{'values': {'speed_modifier': 1}}]), bonus_stats='[]', icon_url='')
    cont = dict(id='c', name_ru='c', name_en='', item_type='', slots=2,
                inner_protection=0, effectiveness=100, bonus_stats='{}', icon_url='')
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql, *args):
            return SimpleNamespace(fetchone=lambda: cont if 'FROM containers' in sql else art)
    monkeypatch.setattr(lunar_bridge, 'get_connection', Connection)
    specs = [{'item_id': 'a', 'auction_lot_id': 7, 'buyout_price': 3},
             {'item_id': 'a', 'auction_lot_id': 8, 'buyout_price': 4}]
    build = lunar_bridge.compute_specific_build(specs, 'c', 'good')
    assert build.total_price == 7 and build.combined_stats['health_bonus'] == 8
    assert {a.auction_lot_id for a in build.artifacts} == {7, 8}
    specs[1]['auction_lot_id'] = 7
    assert lunar_bridge.compute_specific_build(specs, 'c', 'good') is None


def test_api_passes_armor_into_search_and_export_reuses_selected_metadata(monkeypatch, tmp_path):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    monkeypatch.setattr(database, 'init_db', lambda: None)
    main = importlib.import_module('app.main')
    monkeypatch.setattr(main, '__file__', str(tmp_path / 'app' / 'main.py'))
    monkeypatch.setattr(main.Config, 'DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setattr(main, 'check_auction_freshness', lambda *a: {'is_fresh': True, 'last_collected': None, 'age_hours': 0})
    calls = []
    monkeypatch.setattr(main, 'build_artifact_sets', lambda **kwargs: calls.append(kwargs) or [])
    monkeypatch.setattr(builder, 'apply_armor_to_builds', lambda *a, **k: pytest.fail('late armor application'))
    client = TestClient(main.app)
    assert client.get('/api/builds', params={'armor_ids': '["a","b"]', 'armor_level': 7}).status_code == 200
    assert calls[-1]['armor_ids'] == ['a', 'b'] and calls[-1]['armor_level'] == 7
    assert client.get('/api/export-lunarzone', params={'armor_id': 'a'}).status_code == 404
    assert calls[-1]['armor_id'] == 'a'
    assert client.get('/api/builds', params={'armor_ids': '[]'}).status_code == 422
    meta_calls = []
    monkeypatch.setattr(meta_optimizer, 'get_cached_meta', lambda *a: pytest.fail('armor query used no-armor cache'))
    def meta(**kwargs):
        meta_calls.append(kwargs)
        return dict(budget=kwargs['budget'], style='balanced', priorities={},
                    containers_tried=[], builds=[], elapsed_sec=0)
    monkeypatch.setattr(meta_optimizer, 'auto_optimize', meta)
    assert client.get('/api/meta-builds', params={'armor_ids': '["a","b"]', 'armor_level': 7}).status_code == 200
    assert meta_calls[0]['armor_ids'] == ['a', 'b'] and meta_calls[0]['armor_level'] == 7

"""Derived total speed = movement speed + sprint speed across every search path."""
import importlib
from types import SimpleNamespace

import pytest

from app import builder, _beam_search as beam, _max_quality_model as model
from app import _max_quality_worker as worker
from app._numba_search import search as numba_search
from app._objective import Objective
from tests.test_beam_search_validation import run
from tests.test_max_quality import option
from tests.test_objective_integration import container, setup_builder


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setattr(beam, '_GPU_OK', False)


def test_compute_artifact_stats_materializes_total_speed():
    base = [dict(prop_id='speed_modifier', min=0, max=10, is_positive=True),
            dict(prop_id='sprint_speed_modifier', min=0, max=5, is_positive=True)]
    stats, raw, names, _ = builder.compute_artifact_stats(base, [], [], 130, 0, 115, 15, None)
    assert stats['total_speed'] == pytest.approx(19.5)
    assert raw['total_speed'] == pytest.approx(19.5)
    assert names['total_speed'] == 'Общая скорость'


def test_beam_and_numba_rank_derived_total_speed():
    candidates = [SimpleNamespace(item_id=str(i), buyout_price=1,
                                  stats={'speed_modifier': float(speed),
                                         'sprint_speed_modifier': float(sprint)})
                  for i, (speed, sprint) in enumerate([(6, 4), (5, 4), (5, 3), (4, 3), (3, 3)])]
    objective = Objective.create({'total_speed': 10}, {'total_speed': 10})
    args = (candidates, 4, 100, 4, 1.0, {}, {'total_speed': 10},
            {'total_speed': 10}, {}, 1)
    for search in (beam.search, numba_search):
        scores, combos = search(*args, objective=objective, allow_duplicates=True)
        assert set(combos[0]) == {0, 1, 2, 3}
        assert scores[0] == pytest.approx(34)


def test_beam_enforces_derived_total_speed_thresholds():
    candidates = [SimpleNamespace(item_id=str(i), buyout_price=1,
                                  stats={'speed_modifier': float(speed),
                                         'sprint_speed_modifier': 0.0})
                  for i, speed in enumerate([10, 9, 8])]
    settings = dict(k=1, beam_width=10, priorities={'total_speed': 10},
                    stat_max={'total_speed': 10})
    _, combos = run(candidates, min_thresholds={'total_speed': 9.5}, **settings)
    assert combos[0].tolist() == [0]
    _, combos = run(candidates, max_thresholds={'total_speed': 9.5}, **settings)
    assert combos[0].tolist() == [1]


def test_container_and_armor_bonuses_include_total_speed():
    base = builder.ContainerOption('c', 'c', '', '', 1, 0,
                                   bonus_stats={'speed_modifier': 2,
                                                'sprint_speed_modifier': 1})
    armor = dict(id='a', name_ru='a', category='', rank='', level=15,
                 stats={'speed_modifier': 1, 'sprint_speed_modifier': 3})
    effective = builder._armor_containers([base], [armor])[0]
    assert effective.bonus_stats['total_speed'] == pytest.approx(7)
    without = builder._without_armor(
        {'speed_modifier': 5, 'sprint_speed_modifier': 5, 'total_speed': 10}, armor)
    assert without['total_speed'] == pytest.approx(6)


def test_ordinary_search_uses_derived_total_speed(monkeypatch):
    setup_builder(monkeypatch, [
        ({'speed_modifier': 6, 'sprint_speed_modifier': 4}, 2),
        ({'speed_modifier': 9, 'sprint_speed_modifier': 0}, 1),
    ], [container()])
    builds = builder.build_artifact_sets(
        10, tolerance_pct=0, priorities={'total_speed': 10}, top_n=1)
    assert builds and builds[0].artifacts[0].item_id == '0'
    assert builds[0].combined_stats['total_speed'] == pytest.approx(10)
    assert builds[0].combined_stats_names['total_speed'] == 'Общая скорость'


def test_ordinary_search_applies_total_speed_thresholds(monkeypatch):
    setup_builder(monkeypatch, [
        ({'speed_modifier': 6, 'sprint_speed_modifier': 4}, 2),
        ({'speed_modifier': 9, 'sprint_speed_modifier': 0}, 1),
    ], [container()])
    builds = builder.build_artifact_sets(
        10, tolerance_pct=0, priorities={'total_speed': 10},
        min_thresholds={'total_speed': 9.5}, top_n=1)
    assert builds and builds[0].combined_stats['total_speed'] == pytest.approx(10)
    builds = builder.build_artifact_sets(
        10, tolerance_pct=0, priorities={'total_speed': 10},
        max_thresholds={'total_speed': 9.5}, top_n=1)
    assert builds and builds[0].combined_stats['total_speed'] == pytest.approx(9)


@pytest.mark.parametrize('constraint,expected', [
    ({}, 0), ({'minimum': {'total_speed': 9.5}}, 0), ({'maximum': {'total_speed': 9.5}}, 1)])
def test_max_quality_model_uses_derived_total_speed(tmp_path, constraint, expected):
    options = [option(0, {'speed_modifier': 6, 'sprint_speed_modifier': 4}, 1),
               option(1, {'speed_modifier': 9, 'sprint_speed_modifier': 0}, 1),
               option(2, {'speed_modifier': 1, 'sprint_speed_modifier': 1}, 1)]
    objective = Objective.create({'total_speed': 10}, {'total_speed': 10})
    data = model.make_model(options, container(slots=1), objective, 10,
                            constraint.get('minimum', {}), constraint.get('maximum', {}),
                            True, True)
    result = worker.solve(dict(data, seconds=10), tmp_path / 'result.json')
    assert result['optimal'], result
    assert result['stages'][-1]['indices'] == [expected]


def test_all_stats_exposes_total_speed(monkeypatch):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from app import database
    monkeypatch.setattr(database, 'init_db', lambda: None)
    main = importlib.import_module('app.main')

    class Connection:
        def execute(self, *args, **kwargs):
            return []
        def close(self):
            pass

    monkeypatch.setattr(main, 'get_connection', lambda: Connection())
    data = TestClient(main.app).get('/api/all-stats').json()
    assert data['stats']['total_speed'] == 'Общая скорость'


def test_parse_prompt_recognizes_total_speed(monkeypatch):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from app import database
    monkeypatch.setattr(database, 'init_db', lambda: None)
    main = importlib.import_module('app.main')
    monkeypatch.setattr(main, '_llm_system_prompt', 'cached')
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    data = TestClient(main.app).post(
        '/api/parse-prompt', json={'text': 'нужна общая скорость'}).json()
    assert data['priorities'] == {'total_speed': 10}
    assert data['llm_parsed'] is False

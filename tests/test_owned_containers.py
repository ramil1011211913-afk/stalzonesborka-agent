"""Owned-container selection restricts the search pool without silent drops."""
import importlib
import json
from pathlib import Path

import pytest

from app import builder, database
from tests.test_objective_integration import container, setup_builder


def test_builder_uses_only_selected_containers_and_picks_strongest(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 10}, 1)], [
        container('a', 1, 100), container('b', 1, 150), container('c', 1, 300)])
    builds = builder.build_artifact_sets(
        10, tolerance_pct=0, priorities={'speed_modifier': 10},
        container_ids=['a', 'c'], carryable_only=False, top_n=5)
    assert builds and builds[0].container.item_id == 'c'
    assert {build.container.item_id for build in builds} <= {'a', 'c'}


def test_builder_deduplicates_selected_container_ids(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 10}, 1)], [
        container('a', 1, 100), container('b', 1, 200)])
    builds = builder.build_artifact_sets(
        10, tolerance_pct=0, priorities={'speed_modifier': 10},
        container_ids=['a', 'a', ' b '], carryable_only=False)
    assert builds and builds[0].container.item_id == 'b'
    assert {build.container.item_id for build in builds} <= {'a', 'b'}


def test_builder_rejects_unknown_container_ids(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 10}, 1)], [container('a')])
    with pytest.raises(ValueError, match='Unknown container ID'):
        builder.build_artifact_sets(10, priorities={'speed_modifier': 10},
                                    container_ids=['a', 'missing'])


@pytest.mark.parametrize('ids', [[], [''], ['a', 1]])
def test_builder_rejects_invalid_container_id_lists(monkeypatch, ids):
    setup_builder(monkeypatch, [({'speed_modifier': 10}, 1)], [container('a')])
    with pytest.raises(ValueError, match='nonempty list'):
        builder.build_artifact_sets(10, priorities={'speed_modifier': 10},
                                    container_ids=ids)


def test_builder_rejects_both_container_selectors(monkeypatch):
    setup_builder(monkeypatch, [({'speed_modifier': 10}, 1)], [container('a')])
    with pytest.raises(ValueError, match='Choose container_id or container_ids'):
        builder.build_artifact_sets(10, priorities={'speed_modifier': 10},
                                    container_id='a', container_ids=['a'])


def _api_client(monkeypatch, tmp_path):
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
    return TestClient(main.app), calls


def test_api_forwards_owned_containers_and_reports_selection(monkeypatch, tmp_path):
    client, calls = _api_client(monkeypatch, tmp_path)
    response = client.get('/api/builds', params={
        'priorities': json.dumps({'speed_modifier': 10}),
        'container_ids': json.dumps(['a', 'a', ' b ']),
    })
    assert response.status_code == 200
    assert calls[-1]['container_ids'] == ['a', 'b']
    assert calls[-1]['container_id'] is None
    data = response.json()
    assert data['container_ids'] == ['a', 'b']
    assert data['container_selection'] == 'owned'


def test_api_preserves_single_and_auto_container_behavior(monkeypatch, tmp_path):
    client, calls = _api_client(monkeypatch, tmp_path)
    assert client.get('/api/builds').status_code == 200
    assert calls[-1]['container_id'] is None and calls[-1]['container_ids'] is None
    assert client.get('/api/builds', params={'container_id': 'a'}).status_code == 200
    assert calls[-1]['container_id'] == 'a' and calls[-1]['container_ids'] is None
    assert client.get('/api/builds', params={'container_id': ''}).status_code == 200
    assert calls[-1]['container_id'] is None and calls[-1]['container_ids'] is None


@pytest.mark.parametrize('container_ids', ['[]', '[1]', '[""]', 'not-json', 'null'])
def test_api_rejects_invalid_owned_container_lists(monkeypatch, tmp_path, container_ids):
    client, calls = _api_client(monkeypatch, tmp_path)
    response = client.get('/api/builds', params={'container_ids': container_ids})
    assert response.status_code == 422
    assert calls == []


def test_api_rejects_both_container_selectors(monkeypatch, tmp_path):
    client, calls = _api_client(monkeypatch, tmp_path)
    response = client.get('/api/builds', params={
        'container_id': 'a', 'container_ids': json.dumps(['b'])})
    assert response.status_code == 422
    assert calls == []


def test_max_quality_api_forwards_owned_containers(monkeypatch, tmp_path):
    client, _ = _api_client(monkeypatch, tmp_path)
    main = importlib.import_module('app.main')
    from app import _max_quality as quality
    monkeypatch.setattr(main, 'MAX_QUALITY_RELEASED', True)
    calls = []
    monkeypatch.setattr(quality, 'build_max_quality_sets',
                        lambda **kwargs: calls.append(kwargs) or quality.QualityResults(reason='test'))
    response = client.get('/api/builds', params={
        'max_quality': 'true',
        'priorities': json.dumps({'speed_modifier': 10}),
        'container_ids': json.dumps(['a', 'b']),
    })
    assert response.status_code == 200
    assert calls[-1]['container_ids'] == ['a', 'b']
    assert response.json()['container_selection'] == 'owned'


def test_ui_exposes_owned_container_picker_and_max_quality_guard():
    html = (Path(builder.__file__).parent / 'templates/index.html').read_text(encoding='utf8')
    assert 'id="ownedContainersList"' in html
    assert 'id="ownedContainerSearch"' in html
    assert 'id="ownedContainersClear"' in html
    assert 'const MAX_QUALITY_CONTAINERS = 8;' in html
    assert "localStorage.getItem('owned_containers')" in html
    assert "params.append('container_ids', JSON.stringify(selectedContainerIds))" in html
    assert "params.append('container_id', cid)" in html
    assert 'Max Quality: не более ${MAX_QUALITY_CONTAINERS} контейнеров за запрос' in html
    assert 'Учтены выбранные контейнеры:' in html

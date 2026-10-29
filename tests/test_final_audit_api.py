"""Malformed meta requests must not silently change the requested search."""
import importlib

import pytest


@pytest.mark.parametrize('parameter,value', [
    ('priorities', '{'), ('priorities', '[]'), ('priorities', '"speed"'),
    ('priorities', ''), ('priorities', 'null'), ('containers', ''),
    ('priorities', '{"health_bonus":"8"}'), ('priorities', '{"health_bonus":NaN}'),
    ('containers', '{'), ('containers', '{}'), ('containers', '[]'),
    ('containers', '[null]'), ('containers', '[1]'), ('containers', '[""]'),
    ('containers', 'null'), ('containers', '"dkln"'),
])
def test_meta_rejects_malformed_search_parameters(monkeypatch, parameter, value):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from app import database, meta_optimizer
    monkeypatch.setattr(database, 'init_db', lambda: None)
    main = importlib.import_module('app.main')
    monkeypatch.setattr(meta_optimizer, 'auto_optimize',
                        lambda **kwargs: pytest.fail('Invalid query reached optimizer'))
    monkeypatch.setattr(meta_optimizer, 'get_cached_meta',
                        lambda *args: pytest.fail('Invalid query used default cache'))
    response = TestClient(main.app).get('/api/meta-builds', params={parameter: value})
    assert response.status_code == 422
    assert response.json()['error']

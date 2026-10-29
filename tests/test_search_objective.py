from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from app import builder, _beam_search as beam
from tests.test_beam_search_validation import option, run


@pytest.mark.parametrize("flag", ["0", "1"])
def test_lexicographic_lower_priority_cannot_outweigh_primary(monkeypatch, flag):
    monkeypatch.setenv("STALZONE_DANGER_LOOKAHEAD", flag)
    monkeypatch.setattr(beam, "_GPU_OK", False)
    candidates = [option(0, speed_modifier=10, health_bonus=-5),
                  option(1, speed_modifier=9, health_bonus=1000)]
    scores, combos = run(candidates, k=1, beam_width=1,
                         priorities={"speed_modifier": 10, "health_bonus": 8},
                         optimization_mode="lexicographic",
                         priority_order=["speed_modifier", "health_bonus"],
                         stat_max={"speed_modifier": 100, "health_bonus": 1})
    assert combos.tolist() == [[0]]
    assert scores.tolist() == [-39]


def test_balanced_uses_fixed_scales_for_equal_weights(monkeypatch):
    monkeypatch.setattr(beam, "_GPU_OK", False)
    candidates = [option(0, speed_modifier=8, health_bonus=2),
                  option(1, speed_modifier=1, health_bonus=20)]
    scores, combos = run(candidates, k=1, beam_width=1,
                         priorities={"speed_modifier": 10, "health_bonus": 10},
                         stat_max={"speed_modifier": 0.1, "health_bonus": 1000})
    assert combos.tolist() == [[0]]
    assert scores[0] == builder._priority_value(candidates[0].stats,
                                               {"speed_modifier": 10, "health_bonus": 10},
                                               {"speed_modifier": 0.1, "health_bonus": 1000})


def test_carryability_is_a_constraint_not_a_hidden_objective(monkeypatch):
    monkeypatch.setattr(beam, "_GPU_OK", False)
    candidates = [option(0, speed_modifier=100000, radiation_accumulation=20),
                  option(1, speed_modifier=1)]
    scores, combos = run(candidates, k=1, beam_width=1, require_carry=False)
    assert combos.tolist() == [[0]]
    assert scores.tolist() == [100000]
    _, combos = run(candidates, k=1, require_carry=False)
    assert combos.tolist() == [[0], [1]]


def test_constraint_on_secondary_stat_remains_mandatory(monkeypatch):
    monkeypatch.setattr(beam, "_GPU_OK", False)
    candidates = [option(0, speed_modifier=10, health_bonus=-5),
                  option(1, speed_modifier=9, health_bonus=1)]
    _, combos = run(candidates, k=1, beam_width=1,
                    priorities={"speed_modifier": 10, "health_bonus": 8},
                    min_thresholds={"health_bonus": 0})
    assert combos.tolist() == [[1]]


def test_default_stopping_search_keeps_repairable_strong_prefix(monkeypatch):
    monkeypatch.setattr(beam, "_GPU_OK", False)
    monkeypatch.setenv("STALZONE_DANGER_LOOKAHEAD", "0")
    candidates = [option(0, stopping_protection=50, radiation_accumulation=20),
                  option(1, stopping_protection=60, radiation_accumulation=20),
                  option(2, radiation_accumulation=-15)]
    scores, combos = run(candidates, beam_width=1,
                         priorities={"stopping_protection": 10})
    assert combos.tolist() == [[1, 2]]
    assert scores.tolist() == [600]


def test_danger_primary_uses_protected_sum_and_container_bonus(monkeypatch):
    monkeypatch.setattr(beam, "_GPU_OK", False)
    candidates = [option(0, radiation_accumulation=20), option(1, radiation_accumulation=-5)]
    scores, combos = run(candidates, priorities={"radiation_accumulation": 10},
                         prot_mult=0.5, cont_bonus_dict={"radiation_accumulation": -2})
    assert combos.tolist() == [[0, 1]]
    assert scores.tolist() == [-55]


@pytest.mark.parametrize("k", [4, 5, 6, 7])
@pytest.mark.parametrize("allow_duplicates", [False, True])
def test_numba_unsorted_prices_keep_original_indices_and_sentinels(k, allow_duplicates):
    from app._numba_search import search
    candidates = [option("expensive", price=100, speed_modifier=100)]
    candidates += [option(i, price=0, speed_modifier=1) for i in range(k)]
    scores, combos = search(candidates, k, 10, k, 1, {}, {"speed_modifier": 10},
                            {"speed_modifier": 10}, {}, 3,
                            allow_duplicates=allow_duplicates)
    assert scores[0] == k
    assert set(combos[0]) == set(range(1, k + 1))
    assert np.all(combos[1:] == -1)
    assert [c.buyout_price for c in candidates] == [100] + [0] * k


def test_history_is_initialized_independently_for_researched_lots():
    def row(item_id, quality, ptn):
        return dict(id=item_id, name_ru=item_id, name_en='', subcategory='', stats='[]',
                    bonus_stats='[]', icon_url='', quality=quality, ptn=ptn,
                    min_price=100, bonus_properties='[]', art_percent=130, stats_random=None)
    rows = [row('a', 2, 15), row('b', 1, 0), row('b', 2, 15)]
    history = [dict(item_id='a', quality=2, avg_price=321, cnt=3),
               dict(item_id='b', quality=1, avg_price=500, cnt=3),
               dict(item_id='b', quality=2, avg_price=999, cnt=3)]

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql, params=()):
            return SimpleNamespace(fetchall=lambda: history if 'auction_history' in sql else rows)

    with patch.object(builder, 'get_connection', Connection):
        options = builder.load_artifact_options()
    assert [o.hist_avg_price for o in options] == [321, 500, 999]

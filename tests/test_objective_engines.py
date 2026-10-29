from itertools import combinations
from types import SimpleNamespace
import math

import numpy as np
import pytest

from app import builder, _beam_search as beam
from app._numba_search import search as numba_search
from app._objective import Objective


def test_numba_keeps_fractional_costs_for_budget_feasibility():
    candidates = [SimpleNamespace(item_id=str(i), buyout_price=1.9, stats={'speed_modifier': 1})
                  for i in range(4)]
    scores, combos = numba_search(candidates, 4, 7, 4, 1, {},
                                  {'speed_modifier': 1}, {'speed_modifier': 1}, {}, 1)
    assert not np.any(combos >= 0)


@pytest.mark.parametrize('mode', ['balanced', 'lexicographic'])
def test_complete_equal_objective_prefers_lower_price_in_every_engine(monkeypatch, mode):
    monkeypatch.setattr(beam, '_GPU_OK', False)
    candidates = [SimpleNamespace(item_id=str(i), buyout_price=cost, stats={'speed_modifier': 1})
                  for i, cost in enumerate([9, 7, 1, 4, 2, 3])]
    objective = Objective.create({'speed_modifier': 10}, {'speed_modifier': 10}, mode,
                                 ['speed_modifier'])
    args = (candidates, 4, 100, 4, 1, {}, {'speed_modifier': 10}, {'speed_modifier': 10}, {}, 1)
    for search in [beam.search, numba_search]:
        scores, combos = search(*args, objective=objective)
        assert set(combos[0]) == {2, 3, 4, 5}
        assert scores[0] == 4


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["balanced", "lexicographic", "min_budget"])
@pytest.mark.parametrize("duplicates", [False, True])
def test_engines_match_independent_full_game_oracle(monkeypatch, backend, mode, duplicates):
    if backend == "cuda" and not beam._GPU_OK:
        pytest.skip("Working CUDA required")
    monkeypatch.setattr(beam, "_GPU_OK", backend == "cuda")
    monkeypatch.setenv("STALZONE_DANGER_LOOKAHEAD", "1")
    priorities = {"speed_modifier": 10, "health_bonus": 8, "stopping_protection": 2}
    scales = {"speed_modifier": 20, "health_bonus": 30, "stopping_protection": 100}
    order = ["speed_modifier", "health_bonus", "stopping_protection"]
    objective = Objective.create(priorities, scales, mode, order)
    rng = np.random.default_rng(519)
    for _ in range(6):
        candidates = [SimpleNamespace(item_id=str(i // 2), buyout_price=int(rng.integers(0, 10)),
                                      stats={"speed_modifier": float(rng.integers(0, 7)),
                                             "health_bonus": float(rng.integers(-5, 11)),
                                             "radiation_accumulation": float(rng.integers(-10, 21)),
                                             "tear_dmg_factor": float(rng.integers(0, 11)),
                                             "stopping_protection": float(rng.integers(0, 11))})
                      for i in range(8)]
        expected, ranks = {}, {}
        for combo in combinations(range(8), 4):
            opts = [candidates[i] for i in combo]
            price = sum(o.buyout_price for o in opts)
            raw = {key: sum(o.stats[key] for o in opts) for key in candidates[0].stats}
            radiation = (raw["radiation_accumulation"] * 0.25 if raw["radiation_accumulation"] > 0
                         else raw["radiation_accumulation"]) - 3
            stopping = (1 - 100 / (100 + raw["tear_dmg_factor"]) * (1 - raw["stopping_protection"] / 100)) * 100
            if (price > 18 or raw["speed_modifier"] < 4 or not 0 <= radiation <= 8
                    or raw["health_bonus"] > 25 or stopping > 70
                    or sum(o.stats["radiation_accumulation"] < 0 for o in opts) > 2):
                continue
            if not duplicates and len({o.item_id for o in opts}) != 4:
                continue
            f = 10 * raw["speed_modifier"] / 20 + 8 * raw["health_bonus"] / 30 + 2 * stopping / 100
            score = -price if mode == "min_budget" else f
            expected[combo] = score
            ranks[combo] = (tuple(math.floor(v / 1e-6 + 0.5) for v in
                                  [raw["speed_modifier"], raw["health_bonus"], stopping])
                            if mode == "lexicographic" else (score,))
            raw["radiation_accumulation"] = radiation
            raw["stopping_protection"] = stopping
            assert builder._score_build(raw, priorities, scales, is_build=True) == pytest.approx(f)
        args = (candidates, 4, 18, 2, 0.25, {"radiation_accumulation": -3}, scales,
                priorities, {"speed_modifier": 4, "radiation_accumulation": 0}, 200)
        kwargs = dict(objective=objective, allow_duplicates=duplicates, require_carry=True,
                      max_thresholds={"radiation_accumulation": 8, "health_bonus": 25,
                                      "stopping_protection": 70})
        for search in [beam.search, numba_search]:
            scores, combos = search(*args, **kwargs)
            actual = {tuple(sorted(c)): score for c, score in zip(combos, scores) if c[0] >= 0}
            assert actual.keys() == expected.keys()
            for combo in actual:
                assert actual[combo] == pytest.approx(expected[combo], abs=1e-10)
            ordered = [ranks[tuple(sorted(c))] for c in combos if c[0] >= 0]
            assert ordered == sorted(ordered, reverse=True)

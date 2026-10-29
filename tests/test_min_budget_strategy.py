from itertools import combinations

import numpy as np
import pytest

from app import _beam_search as beam
from app._min_budget import suffix_bounds, rank_prefixes, expand_level
from tests.test_beam_search_validation import option, run


@pytest.mark.parametrize("flag", ["0", "1"])
def test_cheap_feasible_build_survives_large_stat_distraction(monkeypatch, flag):
    monkeypatch.setenv("STALZONE_DANGER_LOOKAHEAD", flag)
    monkeypatch.setattr(beam, "_GPU_OK", False)
    candidates = [option(0, price=0), option(1, price=1, speed_modifier=4),
                  option(2, price=20, speed_modifier=100),
                  option(3, price=1, speed_modifier=6)]
    scores, combos = run(candidates, k=3, beam_width=2, min_budget=True,
                         min_thresholds={"speed_modifier": 10})
    assert combos[0].tolist() == [0, 1, 3]
    assert scores[0] == -2


def test_suffix_price_bound_never_overstates_completion_cost():
    rng = np.random.default_rng(802)
    for _ in range(20):
        raw = rng.uniform(-20, 30, (8, 6))
        prices = rng.integers(0, 25, 8)
        low, high, costs = suffix_bounds(raw, prices, 4)
        for last in range(-1, 8):
            for slots in range(1, 5):
                feasible = list(combinations(range(last + 1, 8), slots))
                if not feasible:
                    assert np.isinf(costs[last + 1, slots])
                for combo in feasible:
                    assert costs[last + 1, slots] <= prices[list(combo)].sum()
                    assert np.all(slots * low[last + 1] <= raw[list(combo)].sum(axis=0))
                    assert np.all(slots * high[last + 1] >= raw[list(combo)].sum(axis=0))


def test_prefix_bound_keeps_temporarily_unwearable_repairable_build():
    raw = np.zeros((3, 6))
    raw[0, 0] = 30
    raw[1, 0] = -25
    raw[2, 5] = 10
    prices = np.array([0, 2, 3])
    bounds = suffix_bounds(raw, prices, 3)
    empty_i = np.array([], dtype=np.int32)
    empty_v = np.array([])
    scores = rank_prefixes(raw[:1], prices[:1], np.array([0]), 2,
                           bounds, 10, 1, empty_i, empty_v,
                           np.array([5]), np.array([10]), empty_i, empty_v,
                           True, beam._DANGER_THRESH, -1, np)
    assert scores.tolist() == [-5]


def test_min_budget_does_not_prune_derived_stopping_constraint(monkeypatch):
    monkeypatch.setattr(beam, "_GPU_OK", False)
    candidates = [option(0, price=0, tear_dmg_factor=100), option(1, price=1),
                  option(2, price=20, stopping_protection=80)]
    scores, combos = run(candidates, min_budget=True,
                         min_thresholds={"stopping_protection": 40}, beam_width=2)
    assert combos[0].tolist() == [0, 1]
    assert scores[0] == -1


@pytest.mark.parametrize("duplicates", [False, True])
@pytest.mark.parametrize("block_limit", [2, 100000])
def test_block_expansion_matches_independent_complete_enumeration(duplicates, block_limit):
    raw = np.zeros((6, 6))
    raw[:, 0] = [12, -5, 0, 7, -10, 0]
    raw[:, 5] = [3, 5, 8, 2, 1, 7]
    prices = np.array([1, 3, 2, 5, 0, 4], dtype=float)
    ids = np.array([0, 0, 1, 2, 2, 3])
    counters = (raw[:, 0] < 0).astype(float)
    indices = np.arange(6, dtype=np.int32)[:, None]

    def score(stats, costs, last):
        valid = (stats[:, 5] >= 8) & (stats[:, 0] <= 10)
        return np.where(valid, -costs, -np.inf)

    result = expand_level(raw, prices, counters, indices[:, 0], indices,
                          ids[:, None] if not duplicates else None,
                          raw, prices, counters, ids, 7, 1, 100,
                          duplicates, score, beam._top_indices, np, block_limit)
    actual = {tuple(combo): s for combo, s in zip(result[4], result[-1])}
    expected = {}
    for combo in combinations(range(6), 2):
        i, j = combo
        if (prices[i] + prices[j] <= 7 and counters[i] + counters[j] <= 1
                and (duplicates or ids[i] != ids[j])
                and raw[i, 5] + raw[j, 5] >= 8 and raw[i, 0] + raw[j, 0] <= 10):
            expected[combo] = -prices[i] - prices[j]
    assert actual == expected

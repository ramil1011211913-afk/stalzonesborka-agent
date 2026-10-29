"""Behavioral checks against independent enumeration, rather than source text."""
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from app import _beam_search as beam
from app._danger_lookahead import build_reduction_table, lookup_reduction


def option(i, price=1, **stats):
    return SimpleNamespace(item_id=str(i), buyout_price=price, stats=stats)


def run(candidates, **kwargs):
    settings = dict(k=2, budget=100, max_counter=2, prot_mult=1.0,
                    cont_bonus_dict={}, stat_max={"speed_modifier": 10},
                    priorities={"speed_modifier": 10}, min_thresholds={},
                    n_keep=1000, beam_width=10000, allow_duplicates=True)
    settings.update(kwargs)
    return beam.search(candidates, **settings)


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setattr(beam, "_GPU_OK", False)


@pytest.mark.parametrize("flag", ["0", "1"])
def test_max_only_stat_is_enforced(monkeypatch, flag):
    monkeypatch.setenv("STALZONE_DANGER_LOOKAHEAD", flag)
    candidates = [option(0, speed_modifier=10, health_bonus=100),
                  option(1, speed_modifier=2), option(2, speed_modifier=1)]
    _, combos = run(candidates, max_thresholds={"health_bonus": 10})
    assert combos.tolist() == [[1, 2]]


def test_max_priority_stat_is_enforced():
    candidates = [option(0, speed_modifier=10), option(1, speed_modifier=2),
                  option(2, speed_modifier=1)]
    _, combos = run(candidates, max_thresholds={"speed_modifier": 4})
    assert combos.tolist() == [[1, 2]]


@pytest.mark.parametrize("k,width", [(1, 1), (2, 2)])
def test_final_invalid_builds_do_not_displace_feasible_builds(k, width):
    candidates = [option(0, speed_modifier=1000, health_bonus=10),
                  option(1, speed_modifier=10), option(2, speed_modifier=1)]
    _, combos = run(candidates, k=k, beam_width=width,
                    max_thresholds={"health_bonus": 0})
    assert combos.tolist() == ([[1]] if k == 1 else [[1, 2]])


@pytest.mark.parametrize("flag", ["0", "1"])
def test_impossible_extension_returns_empty_complete_shape(monkeypatch, flag):
    monkeypatch.setenv("STALZONE_DANGER_LOOKAHEAD", flag)
    scores, combos = run([option(0, price=70), option(1, price=70)])
    assert len(scores) == 0
    assert combos.shape == (0, 2)


def test_zero_budget_accepts_free_candidates(monkeypatch):
    monkeypatch.setenv("STALZONE_DANGER_LOOKAHEAD", "1")
    _, combos = run([option(0, price=0), option(1, price=0)], budget=0)
    assert combos.tolist() == [[0, 1]]


def test_danger_constraint_uses_protected_value():
    candidates = [option(0, radiation_accumulation=20), option(1)]
    _, combos = run(candidates, prot_mult=0.5,
                    min_thresholds={"radiation_accumulation": 15})
    assert len(combos) == 0  # actual accumulation is 10, not raw 20


def test_danger_max_uses_protection_before_container_bonus():
    candidates = [option(0, radiation_accumulation=20), option(1)]
    _, combos = run(candidates, prot_mult=0.5,
                    cont_bonus_dict={"radiation_accumulation": -2},
                    max_thresholds={"radiation_accumulation": 8})
    assert combos.tolist() == [[0, 1]]


def test_carryability_matches_builder_at_threshold():
    _, combos = run([option(0, radiation_accumulation=10.0005), option(1)])
    assert len(combos) == 0


def test_stopping_priority_includes_tear_dependency():
    candidates = [option(0, tear_dmg_factor=100), option(1), option(2)]
    _, combos = run(candidates, priorities={"stopping_protection": 10},
                    min_thresholds={"stopping_protection": 40})
    assert {tuple(c) for c in combos} == {(0, 1), (0, 2)}


@pytest.mark.parametrize("flag", ["0", "1"])
@pytest.mark.parametrize("allow_duplicates", [False, True])
@pytest.mark.parametrize("min_budget", [False, True])
def test_wide_beam_matches_independent_exhaustive_oracle(
    monkeypatch, flag, allow_duplicates, min_budget
):
    monkeypatch.setenv("STALZONE_DANGER_LOOKAHEAD", flag)
    rng = np.random.default_rng(741)
    for _ in range(12):
        candidates = [option(i // 2, price=int(rng.integers(0, 21)),
                             speed_modifier=float(rng.integers(-2, 11)),
                             health_bonus=float(rng.integers(-3, 9)),
                             radiation_accumulation=float(rng.integers(-10, 31)))
                      for i in range(8)]
        expected = {}
        for combo in combinations(range(8), 3):
            opts = [candidates[i] for i in combo]
            price = sum(o.buyout_price for o in opts)
            speed = sum(o.stats["speed_modifier"] for o in opts)
            health = sum(o.stats["health_bonus"] for o in opts)
            raw = sum(o.stats["radiation_accumulation"] for o in opts)
            danger = (raw * 0.5 if raw > 0 else raw) - 2
            if price > 35 or speed < 4 or health > 12 or danger > 8:
                continue
            if not allow_duplicates and len({o.item_id for o in opts}) != 3:
                continue
            # Completed Beam scores match builder's actual ordering objective.
            score = -price if min_budget else speed
            expected[combo] = score
        scores, combos = run(candidates, k=3, max_counter=3, budget=35,
                             prot_mult=0.5, cont_bonus_dict={"radiation_accumulation": -2},
                             min_thresholds={"speed_modifier": 4},
                             max_thresholds={"health_bonus": 12, "radiation_accumulation": 8},
                             allow_duplicates=allow_duplicates, min_budget=min_budget)
        actual = {tuple(c): s for c, s in zip(combos, scores)}
        assert actual.keys() == expected.keys()
        for combo, score in actual.items():
            assert score == pytest.approx(expected[combo])


def test_suffix_budget_bound_is_optimistic_for_all_feasible_completions():
    rng = np.random.default_rng(529)
    for _ in range(25):
        raw = rng.uniform(-50, 30, (7, 5))
        prices = rng.integers(0, 40, 7).astype(float)
        table, caps = build_reduction_table(raw, prices, budget=100, buckets=7)
        for last in range(-1, 7):
            for slots in (1, 2, 3):
                for remaining in (0, 17, 39, 70, 100):
                    bound = slots * lookup_reduction(table, caps, [last], [remaining])[0]
                    for combo in combinations(range(last + 1, 7), slots):
                        if prices[list(combo)].sum() <= remaining:
                            assert np.all(bound <= raw[list(combo)].sum(axis=0) + 1e-10)


def test_final_level_does_not_apply_optimistic_reduction(monkeypatch):
    candidates = [option(0, speed_modifier=10, radiation_accumulation=30),
                  option(1, speed_modifier=1), option(2, radiation_accumulation=-25)]
    monkeypatch.setenv("STALZONE_DANGER_LOOKAHEAD", "1")
    _, combos = run(candidates)
    assert {tuple(c) for c in combos} == {(0, 2), (1, 2)}


def test_counter_limit_and_physical_candidate_uniqueness(monkeypatch):
    monkeypatch.setenv("STALZONE_DANGER_LOOKAHEAD", "1")
    candidates = [option("same", radiation_accumulation=-10),
                  option("same", radiation_accumulation=-20), option("other")]
    _, combos = run(candidates, max_counter=1, allow_duplicates=True)
    assert {tuple(c) for c in combos} == {(0, 2), (1, 2)}
    assert all(len(set(c)) == len(c) for c in combos)


@pytest.mark.skipif(not beam._GPU_OK, reason="A working CUDA/CuPy backend is required")
@pytest.mark.parametrize("flag", ["0", "1"])
@pytest.mark.parametrize("allow_duplicates", [False, True])
@pytest.mark.parametrize("min_budget", [False, True])
def test_actual_cuda_matches_cpu(monkeypatch, flag, allow_duplicates, min_budget):
    candidates = [option(i // 2, price=i, speed_modifier=i,
                         health_bonus=i, radiation_accumulation=i - 5)
                  for i in range(9)]
    monkeypatch.setenv("STALZONE_DANGER_LOOKAHEAD", flag)
    kwargs = dict(k=3, allow_duplicates=allow_duplicates, min_budget=min_budget,
                  min_thresholds={"speed_modifier": 6},
                  max_thresholds={"health_bonus": 18, "radiation_accumulation": 8},
                  prot_mult=0.5, cont_bonus_dict={"radiation_accumulation": -2})
    cpu_scores, cpu_combos = run(candidates, **kwargs)
    monkeypatch.setattr(beam, "_GPU_OK", True)
    gpu_scores, gpu_combos = run(candidates, **kwargs)
    cpu = {tuple(c): s for c, s in zip(cpu_combos, cpu_scores)}
    gpu = {tuple(c): s for c, s in zip(gpu_combos, gpu_scores)}
    assert cpu.keys() == gpu.keys()
    for combo in cpu:
        assert gpu[combo] == pytest.approx(cpu[combo])


@pytest.mark.skipif(not beam._GPU_OK, reason="A working CUDA/CuPy backend is required")
def test_cuda_enabled_large_beam_uses_cpu_fallback(monkeypatch):
    monkeypatch.setattr(beam, "_GPU_OK", True)
    monkeypatch.setenv("STALZONE_DANGER_LOOKAHEAD", "1")
    scores, combos = run([option(0, speed_modifier=2), option(1, speed_modifier=1)],
                         beam_width=50000)
    assert isinstance(scores, np.ndarray)
    assert isinstance(combos, np.ndarray)
    assert combos.tolist() == [[0, 1]]


def test_zero_budget_includes_only_free_compensation():
    stats = np.zeros((2, 5))
    stats[:, 0] = [-5, -100]
    table, caps = build_reduction_table(stats, [0, 1], budget=0)
    result = lookup_reduction(table, caps, [-1], [0])
    assert result[0, 0] == -5


@pytest.mark.parametrize("prices", [[-1], [np.nan], [np.inf], [[1]]])
def test_invalid_prices_are_rejected(prices):
    with pytest.raises(ValueError):
        build_reduction_table(np.zeros((1, 5)), prices, budget=100)


@pytest.mark.parametrize("budget", [-1, np.nan, np.inf])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError):
        build_reduction_table(np.zeros((1, 5)), [0], budget=budget)

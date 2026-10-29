import numpy as np

from app._danger_lookahead import (
    build_reduction_table,
    lookup_reduction,
)


def test_reduction_uses_same_artifact_price_and_stats():
    # Artifact 0: strong but expensive.
    # Artifact 1: weaker but affordable.
    stats = np.zeros((2, 5))
    stats[0, 0] = -150.0
    stats[1, 0] = -20.0

    prices = np.array([5_000_000, 100_000], dtype=float)

    table, caps = build_reduction_table(
        stats,
        prices,
        budget=10_000_000,
        buckets=100,
    )

    # A last index of -1 means no candidates selected.
    reduction = lookup_reduction(
        table,
        caps,
        last_indices=np.array([-1]),
        remaining_budgets=np.array([500_000]),
    )

    assert reduction[0, 0] == -20.0
    assert reduction[0, 0] != -150.0


def test_suffix_excludes_previous_candidates():
    stats = np.zeros((3, 5))
    stats[0, 0] = -150.0
    stats[1, 0] = -30.0
    stats[2, 0] = -10.0

    prices = np.array([100, 100, 100], dtype=float)

    table, caps = build_reduction_table(
        stats, prices, budget=1000, buckets=10
    )

    reduction = lookup_reduction(
        table,
        caps,
        last_indices=np.array([0, 1, 2]),
        remaining_budgets=np.array([1000, 1000, 1000]),
    )

    assert reduction[0, 0] == -30.0
    assert reduction[1, 0] == -10.0
    assert reduction[2, 0] == 0.0


def test_unaffordable_counter_not_used():
    stats = np.zeros((2, 5))
    stats[0, 1] = -100.0
    stats[1, 1] = -50.0

    prices = np.array([5_000_000, 3_000_000], dtype=float)

    table, caps = build_reduction_table(
        stats,
        prices,
        budget=10_000_000,
        buckets=100,
    )

    reduction = lookup_reduction(
        table,
        caps,
        last_indices=np.array([-1]),
        remaining_budgets=np.array([500_000]),
    )

    assert reduction[0, 1] == 0.0


def test_budget_rounding_never_excludes_affordable_counter():
    stats = np.zeros((1, 5))
    stats[0, 0] = -15.0
    prices = np.array([151.0])

    table, caps = build_reduction_table(
        stats, prices, budget=1000, buckets=10
    )

    result = lookup_reduction(
        table,
        caps,
        last_indices=np.array([-1]),
        remaining_budgets=np.array([151.0]),
    )

    assert result[0, 0] == -15.0

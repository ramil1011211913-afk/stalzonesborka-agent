"""Budget-aware optimistic suffix danger bounds.

The price and danger reduction always come from the same artifact option.
Budget caps round UP, making the bound optimistic rather than restrictive.
Each danger dimension chooses its own best option. Multiplying this single-item
bound by remaining slots relaxes total cost, distinctness and counter limits;
this is not a multi-slot knapsack solution or a bound on the whole build score.
"""

import numpy as np


def build_reduction_table(raw_mx, prices, budget, buckets=64):
    raw_mx = np.asarray(raw_mx, dtype=np.float64)
    prices = np.asarray(prices, dtype=np.float64)

    if raw_mx.ndim != 2 or raw_mx.shape[1] < 5:
        raise ValueError("raw_mx must contain at least five danger columns")

    n = len(prices)

    if raw_mx.shape[0] != n:
        raise ValueError("Candidate count mismatch")

    if n == 0:
        raise ValueError("At least one candidate is required")

    if budget < 0 or not np.isfinite(budget) or buckets < 1:
        raise ValueError("Invalid budget or bucket count")

    if prices.ndim != 1 or not np.all(np.isfinite(prices)) or np.any(prices < 0):
        raise ValueError("Prices must be a finite nonnegative vector")
    if not np.all(np.isfinite(raw_mx[:, :5])):
        raise ValueError("Danger values must be finite")

    # Repeated zero caps are intentional: free owned candidates remain eligible.
    caps = np.linspace(0.0, float(budget), buckets + 1)

    # table[i, b, d] is the strongest negative danger value
    # available at index >= i and price <= caps[b].
    table = np.zeros((n + 1, buckets + 1, 5), dtype=np.float64)

    danger = np.minimum(raw_mx[:, :5], 0.0)

    for b, cap in enumerate(caps):
        eligible = np.where(
            prices[:, None] <= cap,
            danger,
            0.0,
        )

        table[:n, b, :] = np.minimum.accumulate(
            eligible[::-1],
            axis=0,
        )[::-1]

    return table, caps


def lookup_reduction(table, caps, last_indices, remaining_budgets):
    """Return optimistic reduction for candidates AFTER each last index."""
    last_indices = np.asarray(last_indices, dtype=np.int64)
    remaining_budgets = np.asarray(
        remaining_budgets, dtype=np.float64
    )

    if last_indices.shape != remaining_budgets.shape:
        raise ValueError("Shape mismatch")

    if np.any(last_indices < -1) or np.any(last_indices >= len(table) - 1):
        raise ValueError("Candidate index out of range")

    # Search for the first cap >= remaining budget.
    # This is deliberately optimistic.
    bucket_indices = np.searchsorted(
        caps,
        np.maximum(remaining_budgets, 0.0),
        side="left",
    )
    bucket_indices = np.clip(
        bucket_indices, 0, len(caps) - 1
    )

    return table[last_indices + 1, bucket_indices]

"""Price-first prefix ranking with optimistic feasibility checks.

These bounds relax item-family uniqueness, counter counts and coupling between
stats. They can reject impossible prefixes, but cannot certify a finite beam.
"""
import numpy as np


def suffix_bounds(raw, prices, slots):
    n, width = raw.shape
    high = np.zeros((n + 1, width))
    low = np.zeros((n + 1, width))
    high[:n] = np.maximum.accumulate(np.maximum(raw, 0)[::-1], axis=0)[::-1]
    low[:n] = np.minimum.accumulate(np.minimum(raw, 0)[::-1], axis=0)[::-1]
    costs = np.full((n + 1, slots + 1), np.inf)
    costs[:, 0] = 0
    for i in range(n - 1, -1, -1):
        costs[i, 1:] = np.minimum(costs[i + 1, 1:], prices[i] + costs[i + 1, :-1])
    return low, high, costs


def rank_prefixes(stats, prices, last, remaining, bounds, budget,
                  protection, bonus_idx, bonus_val, min_idx, min_val,
                  max_idx, max_val, require_carry, danger_thresholds,
                  stop_idx, xp):
    low, high, costs = bounds
    lower = stats + remaining * low[last + 1]
    upper = stats + remaining * high[last + 1]
    for d in range(4):
        lower[:, d] = xp.where(lower[:, d] > 0, lower[:, d] * protection, lower[:, d])
        upper[:, d] = xp.where(upper[:, d] > 0, upper[:, d] * protection, upper[:, d])
    for b in range(len(bonus_idx)):
        lower[:, bonus_idx[b]] += bonus_val[b]
        upper[:, bonus_idx[b]] += bonus_val[b]
    total_lower_cost = prices + costs[last + 1, remaining]
    possible = total_lower_cost <= budget
    # Linear per-stat extrema are unsafe for derived stopping protection.
    # Leave that constraint to the exact completed-build check.
    for m in range(len(min_idx)):
        if int(min_idx[m]) != stop_idx:
            possible &= upper[:, min_idx[m]] >= min_val[m] - 1e-9
    for m in range(len(max_idx)):
        if int(max_idx[m]) != stop_idx:
            possible &= lower[:, max_idx[m]] <= max_val[m] + 1e-9
    if require_carry:
        possible &= xp.all(lower[:, :5] <= danger_thresholds + 1e-9, axis=1)
    return xp.where(possible, -total_lower_cost, -xp.inf)


def expand_level(stats, prices, counters, last, indices, item_ids,
                 raw, candidate_prices, candidate_counters, candidate_ids,
                 budget, max_counter, width, allow_duplicates, score, select, xp,
                 max_expansions=262144, joint_score=None):
    """Merge block-local top beams; avoid allocating all beam × pool stats.

    Discarding an entry below the local top width cannot remove a strictly
    better global entry. Equal-score boundary choices remain heuristic.
    """
    n = len(raw)
    block_size = max(1, max_expansions // n)
    candidate_range = xp.arange(n)
    best = None
    for start in range(0, len(prices), block_size):
        end = min(start + block_size, len(prices))
        valid = last[start:end, None] < candidate_range[None, :]
        valid &= prices[start:end, None] + candidate_prices[None, :] <= budget
        valid &= counters[start:end, None] + candidate_counters[None, :] <= max_counter
        if not allow_duplicates:
            for slot in range(item_ids.shape[1]):
                valid &= item_ids[start:end, slot, None] != candidate_ids[None, :]
        parent, candidate = xp.where(valid)
        if not len(parent):
            continue
        parent += start
        new_stats = stats[parent] + raw[candidate]
        new_prices = prices[parent] + candidate_prices[candidate]
        if joint_score is None:
            scores = score(new_stats, new_prices, candidate)
        else:
            scores = joint_score(new_stats, new_prices, candidate,
                                 counters[parent] + candidate_counters[candidate],
                                 xp.column_stack((indices[parent], candidate.astype(xp.int32))))
        top = select(scores, width, xp)
        parent, candidate = parent[top], candidate[top]
        current = (
            new_stats[top], new_prices[top],
            counters[parent] + candidate_counters[candidate],
            candidate.astype(xp.int32),
            xp.column_stack((indices[parent], candidate.astype(xp.int32))),
            xp.column_stack((item_ids[parent], candidate_ids[candidate]))
            if not allow_duplicates else None,
            scores[top],
        )
        if best is None:
            best = current
        else:
            merged = tuple(xp.concatenate((old, new), axis=0) if old is not None else None
                           for old, new in zip(best, current))
            top = select(merged[-1], width, xp)
            best = tuple(array[top] if array is not None else None for array in merged)
    if best is not None:
        return best
    return (xp.zeros((0, stats.shape[1])), xp.zeros(0), xp.zeros(0),
            xp.zeros(0, dtype=xp.int32),
            xp.zeros((0, indices.shape[1] + 1), dtype=xp.int32),
            xp.zeros((0, indices.shape[1] + 1), dtype=xp.int32) if item_ids is not None else None,
            xp.zeros(0))

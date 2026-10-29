"""Uncoupled suffix extrema for prefix constraint penalties, not objective bounds.

Budget, physical conflicts and coupling are relaxed. Nonlinear stopping
constraints keep ordinary penalties. Only balanced prefix ranking uses these
bounds; lexicographic and min_budget retain their established ranking.
Completed feasibility is unchanged.
"""
import numpy as np


def suffix_extrema(raw):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('Suffix stats must be a finite matrix')
    low = np.zeros((len(raw) + 1, raw.shape[1]))
    high = np.zeros_like(low)
    if len(raw):
        low[:-1] = np.minimum.accumulate(np.minimum(raw, 0)[::-1], axis=0)[::-1]
        high[:-1] = np.maximum.accumulate(np.maximum(raw, 0)[::-1], axis=0)[::-1]
    return low, high


def optimistic_constraints(stats, last, remaining, bounds, protection, bonus_idx, bonus_val, xp):
    low, high = bounds
    lower = stats + remaining * low[last + 1]
    upper = stats + remaining * high[last + 1]
    for danger in range(4):
        lower[:, danger] = xp.where(lower[:, danger] > 0, lower[:, danger] * protection, lower[:, danger])
        upper[:, danger] = xp.where(upper[:, danger] > 0, upper[:, danger] * protection, upper[:, danger])
    for column, value in zip(bonus_idx, bonus_val):
        lower[:, column] += value
        upper[:, column] += value
    return lower, upper

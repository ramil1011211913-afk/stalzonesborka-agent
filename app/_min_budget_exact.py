"""Bounded exact price model on a prepared pool; never certifies the auction.

Nonlinear constraints and guaranteed carryability are deliberately unsupported.
Every solver incumbent must pass the caller's complete game validation.
"""
import math
import time
import numpy as np

from ._physical_items import availability_groups

DANGERS = ('radiation_accumulation', 'biological_accumulation',
           'psycho_accumulation', 'thermal_accumulation', 'frost_accumulation')
LIMITS = (10., 10., 10., 10., 20.)


def solve(candidates, slots, budget, max_counter, protection, bonuses,
          minimum, maximum, duplicates, carry, validate, *, seconds=1.,
          guaranteed=False):
    """Return (indices or None, metadata). None means unknown or infeasible pool."""
    deadline = time.monotonic() + max(0., seconds)
    info = dict(state='unknown', scope='prepared_pool', reason='time_limit')
    if seconds <= 0:
        return None, info
    if (guaranteed or 'stopping_protection' in set(minimum) | set(maximum)
            or any(k.startswith('__guaranteed_') for k in maximum)):
        return None, dict(info, reason='unsupported_transform')
    if (len(candidates) > 5000 or len(minimum) + len(maximum) > 256
            or not 0 <= protection <= 1 or budget > 2**53):
        return None, dict(info, reason='model_limit')
    try:
        from scipy.optimize import milp, Bounds, LinearConstraint
        from scipy.sparse import coo_matrix
    except ImportError:
        return None, dict(info, reason='solver_unavailable')
    n = len(candidates)
    prices = np.array([c.buyout_price for c in candidates], dtype=float)
    if not np.isfinite(prices).all() or np.any(prices < 0) or np.any(prices > 2**53):
        return None, dict(info, reason='unsupported_prices')
    rr, cc, vv, lower, upper = [], [], [], [], []

    def row(columns, values, lo, hi):
        index = len(lower)
        rr.extend([index] * len(columns)); cc.extend(columns); vv.extend(values)
        lower.append(lo); upper.append(hi)

    def vector(values, lo, hi):
        nonzero = np.flatnonzero(values)
        row(nonzero.tolist(), values[nonzero].tolist(), lo, hi)

    vector(np.ones(n), slots, slots)
    vector(prices, -np.inf, budget)
    vector(np.array([any(key.endswith('_accumulation') and value < 0
                         for key, value in c.stats.items()) for c in candidates], dtype=float),
           -np.inf, max_counter)

    def bound(key, value, is_min=False):
        target = value - bonuses.get(key, 0)
        if key in DANGERS[:4] and protection == 0 and not is_min and target >= 0:
            # Full protection maps every positive raw sum to zero. A
            # nonnegative upper bound therefore imposes no raw constraint.
            return
        if key in DANGERS[:4] and target > 0:
            if protection == 0:
                if is_min:
                    vector(np.zeros(n), 1, np.inf)  # Impossible positive lower bound.
                return
            target /= protection
        values = np.array([c.stats.get(key, 0) for c in candidates], dtype=float)
        if not np.isfinite(values).all() or not math.isfinite(target):
            raise ValueError('Nonfinite model')
        vector(values, target if is_min else -np.inf, np.inf if is_min else target)

    try:
        if carry:
            for key, value in zip(DANGERS, LIMITS):
                bound(key, value)
        for key, value in minimum.items():
            bound(key, value, True)
        for key, value in maximum.items():
            bound(key, value)
        for group in availability_groups(candidates, duplicates):
            row(list(group), [1.] * len(group), -np.inf, 1.)
        if len(vv) > 250000 or len(lower) > 10000:
            return None, dict(info, reason='model_limit')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, info
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(lower), n)).tocsc()
        result = milp(prices, integrality=np.ones(n), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options=dict(time_limit=remaining, mip_rel_gap=0.))
    except Exception:
        # A solver failure is not a proof of infeasibility. No user data in errors.
        return None, dict(info, reason='solver_error')
    info.update(solver_status=int(result.status))
    if result.x is not None and np.isfinite(result.x).all():
        indices = np.flatnonzero(result.x > .5).astype(np.int32)
        if (len(indices) == slots and np.max(np.abs(result.x - np.rint(result.x))) <= 1e-5
                and validate(indices)):
            return indices, dict(info, state='optimal' if result.status == 0 else 'feasible',
                                 reason='validated_incumbent')
        return None, dict(info, reason='invalid_incumbent')
    if result.status == 2:
        return None, dict(info, state='infeasible', reason='proved_for_prepared_pool')
    return None, dict(info, reason='time_limit' if result.status == 1 else 'solver_error')


class SearchResults(list):
    """Backwards compatible, request-local diagnostics, including empty results."""
    def __init__(self, builds, models):
        super().__init__(builds)
        self.search_status = dict(
            state='feasible' if builds else 'unknown',
            scope='prepared_candidate_pools', optimality_proven=False,
            infeasibility_proven=False,
            complete=all(m['state'] in ('optimal', 'infeasible') for m in models) and bool(models),
            models=models,
            message=('Допустимые сборки найдены; глобальный минимум не гарантирован.' if builds else
                     'Поиск не установил допустимую сборку. Это не доказательство невозможности выполнить ограничения.'))

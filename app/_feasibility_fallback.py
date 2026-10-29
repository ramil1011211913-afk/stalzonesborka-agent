"""Bounded feasibility rescue for empty ordinary searches, never an optimizer.

All certificates are scoped to the verified, prepared candidate pools. Time limits
are solver limits, not a hard wall-clock or native-memory guarantee.
"""
from datetime import datetime, timezone
from enum import Enum
import json
import hashlib
import math
import time

MAX_SECONDS = 5.0
MODEL_SECONDS = 3.0
MAX_CANDIDATES = 5000
MAX_PROFILES = 16
MAX_TOTAL_CANDIDATES = 20000
MAX_CONSTRAINTS = 256
MAX_ROWS = 10000
MAX_NONZEROS = 100000
MAX_NODES = 10000
DANGERS = ('radiation_accumulation', 'biological_accumulation',
           'psycho_accumulation', 'thermal_accumulation', 'frost_accumulation')
LIMITS = (10., 10., 10., 10., 20.)


class SearchState(str, Enum):
    FEASIBLE = 'FEASIBLE'
    UNKNOWN = 'UNKNOWN'
    INFEASIBLE = 'INFEASIBLE'
    OPTIMAL = 'OPTIMAL'  # Reserved: neither beam nor a feasibility model proves it.


class SearchResults(list):
    def __init__(self, builds, models=(), *, attempted=False, complete=True):
        super().__init__(builds)
        proved = (not builds and complete and bool(models)
                  and all(m['state'] == 'INFEASIBLE' for m in models))
        state = 'FEASIBLE' if builds else 'INFEASIBLE' if proved else 'UNKNOWN'
        self.search_status = dict(
            state=state, scope='verified_current_prepared_pools' if attempted else 'returned_builds',
            optimality_proven=False, infeasibility_proven=proved,
            global_infeasibility_proven=False, fallback_attempted=attempted,
            coverage_complete=complete,
            models=list(models),
            message=('Допустимая сборка найдена; оптимальность не доказана.' if builds else
                     'Несовместность доказана только для проверенных подготовленных пулов, не для всего аукциона.' if proved else
                     'Поиск не смог подтвердить наличие или отсутствие допустимой сборки. Это не доказательство недостаточного бюджета.'))
        if attempted:
            self.search_status['fallback_limits'] = dict(seconds=MAX_SECONDS, model_seconds=MODEL_SECONDS,
                candidates=MAX_CANDIDATES, profiles=MAX_PROFILES, total_candidates=MAX_TOTAL_CANDIDATES,
                rows=MAX_ROWS, nonzeros=MAX_NONZEROS, nodes=MAX_NODES)


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def current_candidates(candidates, connection, region, normalize=lambda x: x, *, now=None):
    """Recheck concrete descriptors and expiry; never invent unknown lot bonuses.

    Owned identities were created from this request. Anonymous programmatic
    options are deliberately unsupported by the rescue path.
    """
    now = now or datetime.now(timezone.utc)
    ids = list({o.auction_lot_id for o in candidates if o.auction_lot_id is not None})
    rows = {}
    if ids:
        with connection() as conn:
            for start in range(0, len(ids), 400):
                batch = ids[start:start + 400]
                query = 'SELECT * FROM auction_lots WHERE region=? AND id IN (' + ','.join('?' for _ in batch) + ')'
                rows.update((r['id'], dict(r)) for r in conn.execute(query, [region, *batch]).fetchall())
    verified = []
    for o in candidates:
        try:
            if o.is_owned:
                if o.buyout_price == 0 and o.physical_id and math.isfinite(o.art_percent) and o.art_percent > 0:
                    verified.append(o)
                continue
            r = rows.get(o.auction_lot_id)
            if r is None:
                continue
            end = datetime.fromisoformat(r['end_time'].replace('Z', '+00:00'))
            if end.tzinfo is None or end <= now:
                continue
            bonuses = _json(r['bonus_properties'])
            if not isinstance(bonuses, list) or any(not isinstance(x, str) for x in bonuses):
                continue
            if (r['item_id'] != o.item_id or r['buyout_price'] != o.buyout_price
                    or r['buyout_price'] <= 0 or r['quality'] != o.quality or r['ptn'] != o.ptn
                    or not 0 <= o.quality <= 5 or not 0 <= o.ptn <= 15
                    or not math.isfinite(r['art_percent']) or not 0 < r['art_percent'] <= 190
                    or r['art_percent'] != o.art_percent or r['stats_random'] != o.stats_random
                    or sorted(map(normalize, bonuses)) != sorted(map(normalize, o.unlocked_bonus_ids))):
                continue
            if (o.ptn > 0 and not bonuses
                    and any(b.get('unlock_ptn', 0) <= o.ptn for b in o.bonus_stats)):
                continue
            verified.append(o)
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
    return verified


def solve(candidates, slots, budget, protection, bonuses, minimum, maximum,
          duplicates, carry, validate, *, seconds, guaranteed=False):
    """Zero objective: even HiGHS status 0 means only FEASIBLE for the user goal."""
    info = dict(state='UNKNOWN', scope='verified_current_prepared_pool', reason='time_limit',
                candidates=len(candidates), objective='feasibility')
    deadline = time.monotonic() + max(0., min(seconds, MODEL_SECONDS))
    if seconds <= 0:
        return None, info
    if guaranteed or any(k == 'stopping_protection' or k.startswith('__') for k in set(minimum) | set(maximum)):
        return None, dict(info, reason='unsupported_transform')
    if (len(candidates) > MAX_CANDIDATES or len(minimum) + len(maximum) > MAX_CONSTRAINTS
            or not 0 <= protection <= 1 or not math.isfinite(budget) or not 0 <= budget <= 2**53
            or not 1 <= slots <= 7):
        return None, dict(info, reason='model_limit')
    if len(candidates) < slots:
        return None, dict(info, state='INFEASIBLE', reason='not_enough_verified_candidates')
    try:
        import numpy as np
        from scipy.optimize import milp, Bounds, LinearConstraint
        from scipy.sparse import coo_matrix
    except ImportError:
        return None, dict(info, reason='solver_unavailable')
    from ._physical_items import availability_groups
    rr, cc, vv, lower, upper = [], [], [], [], []

    def row(values, lo, hi):
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite model')
        nonzero = np.flatnonzero(values)
        if len(vv) + len(nonzero) > MAX_NONZEROS or len(lower) >= MAX_ROWS:
            raise OverflowError('Model limit')
        rr.extend([len(lower)] * len(nonzero)); cc.extend(nonzero.tolist()); vv.extend(values[nonzero].tolist())
        lower.append(lo); upper.append(hi)

    def bound(key, value, is_min=False):
        target = value - bonuses.get(key, 0)
        if not math.isfinite(target):
            raise ValueError('Nonfinite bound')
        if key in DANGERS[:4]:
            if protection == 0:
                if not is_min and target >= 0:
                    return
                if is_min and target > 0:
                    row(np.zeros(n), 1, np.inf)
                    return
            elif target > 0:
                target /= protection
        row(np.array([o.stats.get(key, 0) for o in candidates], dtype=float),
            target if is_min else -np.inf, np.inf if is_min else target)

    try:
        n = len(candidates)
        from ._physical_items import physical_key
        info['pool_sha256'] = hashlib.sha256(json.dumps([
            [physical_key(o), o.item_id, o.buyout_price, o.stats]
            for o in candidates], sort_keys=True, allow_nan=False).encode()).hexdigest()
        info['conditions'] = dict(slots=slots, budget=budget, protection=protection,
                                 bonuses=bonuses, minimum=minimum, maximum=maximum,
                                 duplicates=duplicates, carry=carry)
        prices = np.array([o.buyout_price for o in candidates], dtype=float)
        if not np.isfinite(prices).all() or np.any(prices < 0) or np.any(prices > 2**53):
            return None, dict(info, reason='unsupported_prices')
        row(np.ones(n), slots, slots); row(prices, -np.inf, budget)
        if carry:
            for key, limit in zip(DANGERS, LIMITS):
                bound(key, limit)
        for key, value in minimum.items():
            bound(key, value, True)
        for key, value in maximum.items():
            bound(key, value)
        for group in availability_groups(candidates, duplicates):
            values = np.zeros(n); values[list(group)] = 1
            row(values, -np.inf, 1)
        remaining = min(seconds, MODEL_SECONDS, deadline - time.monotonic())
        if remaining <= 0:
            return None, info
        info.update(rows=len(lower), nonzeros=len(vv), solver_seconds=remaining, node_limit=MAX_NODES)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(lower), n)).tocsc()
        result = milp(np.zeros(n), integrality=np.ones(n), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options=dict(time_limit=remaining, node_limit=MAX_NODES, mip_rel_gap=0.))
        info['solver_status'] = int(result.status)
        info['solver_message'] = str(getattr(result, 'message', ''))[:1000]
        if result.x is not None:
            if not np.isfinite(result.x).all() or len(result.x) != n:
                return None, dict(info, reason='invalid_incumbent')
            ids = np.flatnonzero(result.x > .5).astype(np.int32)
            if (len(ids) == slots and np.max(np.abs(result.x - np.rint(result.x))) <= 1e-5 and validate(ids)):
                return ids, dict(info, state='FEASIBLE', reason='validated_incumbent',
                                 termination='time_or_node_limit' if result.status == 1 else 'solver_finished')
            return None, dict(info, reason='invalid_incumbent')
        if result.status == 2:
            return None, dict(info, state='INFEASIBLE', reason='proved_for_verified_prepared_pool')
        return None, dict(info, reason='time_or_node_limit' if result.status == 1 else 'solver_error')
    except OverflowError:
        return None, dict(info, reason='model_limit')
    except Exception:
        return None, dict(info, reason='solver_error')

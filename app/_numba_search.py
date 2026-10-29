"""
Numba-JIT accelerated combinatorial search for artifact builds.

Completed game stats feed the shared fixed objective; no auxiliary EHP score.
Stat ordering: danger stats [rad,bio,psy,therm,frost] at indices 0-4, rest after.

allow_dup is an internal switch for conflict-group checks. The public wrapper
encodes physical groups when type repeats are allowed, and item types otherwise.
min_budget flag: when True, score = -price (find cheapest builds meeting thresholds).
Numba specializes on boolean params at JIT time.
"""
import numpy as np
from numba import njit

from ._physical_items import conflict_ids

_DANGER_THRESH = np.array([10.0, 10.0, 10.0, 10.0, 20.0], dtype=np.float64)


@njit(nogil=True, cache=True)
def _finished_stats(raw, final, protection, b_idx, b_val, min_idx, min_val,
                    max_idx, max_val, require_carry, tear_idx, stop_idx):
    for s in range(len(raw)):
        final[s] = raw[s]
    for d in range(4):
        if final[d] > 0:
            final[d] *= protection
    for b in range(len(b_idx)):
        final[b_idx[b]] += b_val[b]
    if tear_idx >= 0 and stop_idx >= 0:
        denominator = 100 + final[tear_idx]
        if denominator == 0:
            return False
        final[stop_idx] = (1 - 100 / denominator * (1 - final[stop_idx] / 100)) * 100
    for s in range(len(final)):
        if not np.isfinite(final[s]):
            return False
    for m in range(len(min_idx)):
        if final[min_idx[m]] < min_val[m]:
            return False
    for m in range(len(max_idx)):
        if final[max_idx[m]] > max_val[m]:
            return False
    if require_carry:
        for d in range(5):
            if final[d] > _DANGER_THRESH[d]:
                return False
    return True


@njit(nogil=True, cache=True)
def _objective_score(coefficients, stats):
    # Explicit scalar sum keeps the CPU path independent of SciPy/BLAS.
    score = 0.0
    for s in range(len(stats)):
        score += coefficients[s] * stats[s]
    return score


@njit(nogil=True, cache=True)
def _objective_better(score, rank, old_score, old_rank, occupied, cost, old_cost):
    if not occupied:
        return True
    if len(rank):
        for i in range(len(rank)):
            if rank[i] != old_rank[i]:
                return rank[i] > old_rank[i]
        return cost < old_cost
    return score > old_score or (score == old_score and cost < old_cost)


@njit(nogil=True, cache=True)
def _search_4(raw_mx, prices, is_counter, item_ids, prot_mult, budget, max_c, n_keep,
              tear_idx, stop_idx, b_idx, b_val, m_idx, m_val,
              allow_dup, min_budget, require_carry,
              objective_coeffs, lex_idx, lex_dir, tolerance, max_idx, max_val):
    n = raw_mx.shape[0]; ns = raw_mx.shape[1]
    bs = np.full(n_keep, -1e18, dtype=np.float64)
    bc = np.full((n_keep, 4), -1, dtype=np.int32)
    best_ranks = np.zeros((n_keep, len(lex_idx)))
    rank = np.zeros(len(lex_idx))
    best_costs = np.full(n_keep, np.inf)
    cr = np.zeros(ns, dtype=np.float64)
    f = np.zeros(ns, dtype=np.float64)
    for i0 in range(n):
        p0 = prices[i0]
        if p0 > budget: break
        c0 = 1 if is_counter[i0] else 0
        if c0 > max_c: continue
        for s in range(ns): cr[s] = raw_mx[i0, s]
        for i1 in range(i0+1, n):
            if not allow_dup:
                if item_ids[i1] == item_ids[i0]: continue
            p1 = p0 + prices[i1]
            if p1 > budget: break
            c1 = c0 + (1 if is_counter[i1] else 0)
            if c1 > max_c: continue
            for s in range(ns): cr[s] += raw_mx[i1, s]
            for i2 in range(i1+1, n):
                if not allow_dup:
                    if item_ids[i2] == item_ids[i0] or item_ids[i2] == item_ids[i1]: continue
                p2 = p1 + prices[i2]
                if p2 > budget: break
                c2 = c1 + (1 if is_counter[i2] else 0)
                if c2 > max_c: continue
                for s in range(ns): cr[s] += raw_mx[i2, s]
                for i3 in range(i2+1, n):
                    if not allow_dup:
                        if item_ids[i3] == item_ids[i0] or item_ids[i3] == item_ids[i1] or item_ids[i3] == item_ids[i2]: continue
                    p3 = p2 + prices[i3]
                    if p3 > budget: break
                    c3 = c2 + (1 if is_counter[i3] else 0)
                    if c3 > max_c: continue
                    for s in range(ns): cr[s] += raw_mx[i3, s]
                    ok = _finished_stats(cr, f, prot_mult, b_idx, b_val,
                                         m_idx, m_val, max_idx, max_val, require_carry, tear_idx, stop_idx)
                    if not ok:
                        for s in range(ns): cr[s] -= raw_mx[i3, s]
                        continue
                    score = -1.0 * p3 if min_budget else _objective_score(objective_coeffs, f)
                    for l in range(len(lex_idx)):
                        rank[l] = np.floor(lex_dir[l] * f[lex_idx[l]] / tolerance + 0.5)
                    if _objective_better(score, rank, bs[-1], best_ranks[-1], bc[-1, 0] >= 0, p3, best_costs[-1]):
                        for j in range(n_keep):
                            if _objective_better(score, rank, bs[j], best_ranks[j], bc[j, 0] >= 0, p3, best_costs[j]):
                                for m in range(n_keep-1, j, -1):
                                    bs[m] = bs[m-1]
                                    best_costs[m] = best_costs[m-1]
                                    for l in range(len(lex_idx)): best_ranks[m, l] = best_ranks[m-1, l]
                                    for mi in range(4): bc[m, mi] = bc[m-1, mi]
                                for l in range(len(lex_idx)): best_ranks[j, l] = rank[l]
                                bs[j] = score; bc[j,0]=i0; bc[j,1]=i1; bc[j,2]=i2; bc[j,3]=i3
                                best_costs[j] = p3
                                break
                    for s in range(ns): cr[s] -= raw_mx[i3, s]
                for s in range(ns): cr[s] -= raw_mx[i2, s]
            for s in range(ns): cr[s] -= raw_mx[i1, s]
    return bs, bc


@njit(nogil=True, cache=True)
def _search_5(raw_mx, prices, is_counter, item_ids, prot_mult, budget, max_c, n_keep,
              tear_idx, stop_idx, b_idx, b_val, m_idx, m_val,
              allow_dup, min_budget, require_carry,
              objective_coeffs, lex_idx, lex_dir, tolerance, max_idx, max_val):
    n = raw_mx.shape[0]; ns = raw_mx.shape[1]
    bs = np.full(n_keep, -1e18, dtype=np.float64)
    bc = np.full((n_keep, 5), -1, dtype=np.int32)
    best_ranks = np.zeros((n_keep, len(lex_idx)))
    rank = np.zeros(len(lex_idx))
    best_costs = np.full(n_keep, np.inf)
    cr = np.zeros(ns, dtype=np.float64)
    f = np.zeros(ns, dtype=np.float64)
    for i0 in range(n):
        p0 = prices[i0]
        if p0 > budget: break
        c0 = 1 if is_counter[i0] else 0
        if c0 > max_c: continue
        for s in range(ns): cr[s] = raw_mx[i0, s]
        for i1 in range(i0+1, n):
            if not allow_dup:
                if item_ids[i1] == item_ids[i0]: continue
            p1 = p0 + prices[i1]
            if p1 > budget: break
            c1 = c0 + (1 if is_counter[i1] else 0)
            if c1 > max_c: continue
            for s in range(ns): cr[s] += raw_mx[i1, s]
            for i2 in range(i1+1, n):
                if not allow_dup:
                    if item_ids[i2] == item_ids[i0] or item_ids[i2] == item_ids[i1]: continue
                p2 = p1 + prices[i2]
                if p2 > budget: break
                c2 = c1 + (1 if is_counter[i2] else 0)
                if c2 > max_c: continue
                for s in range(ns): cr[s] += raw_mx[i2, s]
                for i3 in range(i2+1, n):
                    if not allow_dup:
                        if item_ids[i3] == item_ids[i0] or item_ids[i3] == item_ids[i1] or item_ids[i3] == item_ids[i2]: continue
                    p3 = p2 + prices[i3]
                    if p3 > budget: break
                    c3 = c2 + (1 if is_counter[i3] else 0)
                    if c3 > max_c: continue
                    for s in range(ns): cr[s] += raw_mx[i3, s]
                    for i4 in range(i3+1, n):
                        if not allow_dup:
                            if item_ids[i4] == item_ids[i0] or item_ids[i4] == item_ids[i1] or item_ids[i4] == item_ids[i2] or item_ids[i4] == item_ids[i3]: continue
                        p4 = p3 + prices[i4]
                        if p4 > budget: break
                        c4 = c3 + (1 if is_counter[i4] else 0)
                        if c4 > max_c: continue
                        for s in range(ns): cr[s] += raw_mx[i4, s]
                        ok = _finished_stats(cr, f, prot_mult, b_idx, b_val,
                                             m_idx, m_val, max_idx, max_val, require_carry, tear_idx, stop_idx)
                        if not ok:
                            for s in range(ns): cr[s] -= raw_mx[i4, s]
                            continue
                        score = -1.0 * p4 if min_budget else _objective_score(objective_coeffs, f)
                        for l in range(len(lex_idx)):
                            rank[l] = np.floor(lex_dir[l] * f[lex_idx[l]] / tolerance + 0.5)
                        if _objective_better(score, rank, bs[-1], best_ranks[-1], bc[-1, 0] >= 0, p4, best_costs[-1]):
                            for j in range(n_keep):
                                if _objective_better(score, rank, bs[j], best_ranks[j], bc[j, 0] >= 0, p4, best_costs[j]):
                                    for m in range(n_keep-1, j, -1):
                                        bs[m] = bs[m-1]
                                        best_costs[m] = best_costs[m-1]
                                        for l in range(len(lex_idx)): best_ranks[m, l] = best_ranks[m-1, l]
                                        for mi in range(5): bc[m, mi] = bc[m-1, mi]
                                    for l in range(len(lex_idx)): best_ranks[j, l] = rank[l]
                                    bs[j] = score; bc[j,0]=i0; bc[j,1]=i1; bc[j,2]=i2; bc[j,3]=i3; bc[j,4]=i4
                                    best_costs[j] = p4
                                    break
                        for s in range(ns): cr[s] -= raw_mx[i4, s]
                    for s in range(ns): cr[s] -= raw_mx[i3, s]
                for s in range(ns): cr[s] -= raw_mx[i2, s]
            for s in range(ns): cr[s] -= raw_mx[i1, s]
    return bs, bc

@njit(nogil=True, cache=True)
def _search_6(raw_mx, prices, is_counter, item_ids, prot_mult, budget, max_c, n_keep,
              tear_idx, stop_idx, b_idx, b_val, m_idx, m_val,
              allow_dup, min_budget, require_carry,
              objective_coeffs, lex_idx, lex_dir, tolerance, max_idx, max_val):
    n = raw_mx.shape[0]; ns = raw_mx.shape[1]
    bs = np.full(n_keep, -1e18, dtype=np.float64)
    bc = np.full((n_keep, 6), -1, dtype=np.int32)
    best_ranks = np.zeros((n_keep, len(lex_idx)))
    rank = np.zeros(len(lex_idx))
    best_costs = np.full(n_keep, np.inf)
    cr = np.zeros(ns, dtype=np.float64)
    f = np.zeros(ns, dtype=np.float64)
    for i0 in range(n):
        p0 = prices[i0]
        if p0 > budget: break
        c0 = 1 if is_counter[i0] else 0
        if c0 > max_c: continue
        for s in range(ns): cr[s] = raw_mx[i0, s]
        for i1 in range(i0+1, n):
            if not allow_dup:
                if item_ids[i1] == item_ids[i0]: continue
            p1 = p0 + prices[i1]
            if p1 > budget: break
            c1 = c0 + (1 if is_counter[i1] else 0)
            if c1 > max_c: continue
            for s in range(ns): cr[s] += raw_mx[i1, s]
            for i2 in range(i1+1, n):
                if not allow_dup:
                    if item_ids[i2] == item_ids[i0] or item_ids[i2] == item_ids[i1]: continue
                p2 = p1 + prices[i2]
                if p2 > budget: break
                c2 = c1 + (1 if is_counter[i2] else 0)
                if c2 > max_c: continue
                for s in range(ns): cr[s] += raw_mx[i2, s]
                for i3 in range(i2+1, n):
                    if not allow_dup:
                        if item_ids[i3] == item_ids[i0] or item_ids[i3] == item_ids[i1] or item_ids[i3] == item_ids[i2]: continue
                    p3 = p2 + prices[i3]
                    if p3 > budget: break
                    c3 = c2 + (1 if is_counter[i3] else 0)
                    if c3 > max_c: continue
                    for s in range(ns): cr[s] += raw_mx[i3, s]
                    for i4 in range(i3+1, n):
                        if not allow_dup:
                            if item_ids[i4] == item_ids[i0] or item_ids[i4] == item_ids[i1] or item_ids[i4] == item_ids[i2] or item_ids[i4] == item_ids[i3]: continue
                        p4 = p3 + prices[i4]
                        if p4 > budget: break
                        c4 = c3 + (1 if is_counter[i4] else 0)
                        if c4 > max_c: continue
                        for s in range(ns): cr[s] += raw_mx[i4, s]
                        for i5 in range(i4+1, n):
                            if not allow_dup:
                                if item_ids[i5] == item_ids[i0] or item_ids[i5] == item_ids[i1] or item_ids[i5] == item_ids[i2] or item_ids[i5] == item_ids[i3] or item_ids[i5] == item_ids[i4]: continue
                            p5 = p4 + prices[i5]
                            if p5 > budget: break
                            c5 = c4 + (1 if is_counter[i5] else 0)
                            if c5 > max_c: continue
                            for s in range(ns): cr[s] += raw_mx[i5, s]
                            ok = _finished_stats(cr, f, prot_mult, b_idx, b_val,
                                                 m_idx, m_val, max_idx, max_val, require_carry, tear_idx, stop_idx)
                            if not ok:
                                for s in range(ns): cr[s] -= raw_mx[i5, s]
                                continue
                            score = -1.0 * p5 if min_budget else _objective_score(objective_coeffs, f)
                            for l in range(len(lex_idx)):
                                rank[l] = np.floor(lex_dir[l] * f[lex_idx[l]] / tolerance + 0.5)
                            if _objective_better(score, rank, bs[-1], best_ranks[-1], bc[-1, 0] >= 0, p5, best_costs[-1]):
                                for j in range(n_keep):
                                    if _objective_better(score, rank, bs[j], best_ranks[j], bc[j, 0] >= 0, p5, best_costs[j]):
                                        for m in range(n_keep-1, j, -1):
                                            bs[m] = bs[m-1]
                                            best_costs[m] = best_costs[m-1]
                                            for l in range(len(lex_idx)): best_ranks[m, l] = best_ranks[m-1, l]
                                            for mi in range(6): bc[m, mi] = bc[m-1, mi]
                                        for l in range(len(lex_idx)): best_ranks[j, l] = rank[l]
                                        bs[j] = score; bc[j,0]=i0; bc[j,1]=i1; bc[j,2]=i2; bc[j,3]=i3; bc[j,4]=i4; bc[j,5]=i5
                                        best_costs[j] = p5
                                        break
                            for s in range(ns): cr[s] -= raw_mx[i5, s]
                        for s in range(ns): cr[s] -= raw_mx[i4, s]
                    for s in range(ns): cr[s] -= raw_mx[i3, s]
                for s in range(ns): cr[s] -= raw_mx[i2, s]
            for s in range(ns): cr[s] -= raw_mx[i1, s]
    return bs, bc

@njit(nogil=True, cache=True)
def _search_7(raw_mx, prices, is_counter, item_ids, prot_mult, budget, max_c, n_keep,
              tear_idx, stop_idx, b_idx, b_val, m_idx, m_val,
              allow_dup, min_budget, require_carry,
              objective_coeffs, lex_idx, lex_dir, tolerance, max_idx, max_val):
    n = raw_mx.shape[0]; ns = raw_mx.shape[1]
    bs = np.full(n_keep, -1e18, dtype=np.float64)
    bc = np.full((n_keep, 7), -1, dtype=np.int32)
    best_ranks = np.zeros((n_keep, len(lex_idx)))
    rank = np.zeros(len(lex_idx))
    best_costs = np.full(n_keep, np.inf)
    cr = np.zeros(ns, dtype=np.float64)
    f = np.zeros(ns, dtype=np.float64)
    for i0 in range(n):
        p0 = prices[i0]
        if p0 > budget: break
        c0 = 1 if is_counter[i0] else 0
        if c0 > max_c: continue
        for s in range(ns): cr[s] = raw_mx[i0, s]
        for i1 in range(i0+1, n):
            if not allow_dup:
                if item_ids[i1] == item_ids[i0]: continue
            p1 = p0 + prices[i1]
            if p1 > budget: break
            c1 = c0 + (1 if is_counter[i1] else 0)
            if c1 > max_c: continue
            for s in range(ns): cr[s] += raw_mx[i1, s]
            for i2 in range(i1+1, n):
                if not allow_dup:
                    if item_ids[i2] == item_ids[i0] or item_ids[i2] == item_ids[i1]: continue
                p2 = p1 + prices[i2]
                if p2 > budget: break
                c2 = c1 + (1 if is_counter[i2] else 0)
                if c2 > max_c: continue
                for s in range(ns): cr[s] += raw_mx[i2, s]
                for i3 in range(i2+1, n):
                    if not allow_dup:
                        if item_ids[i3] == item_ids[i0] or item_ids[i3] == item_ids[i1] or item_ids[i3] == item_ids[i2]: continue
                    p3 = p2 + prices[i3]
                    if p3 > budget: break
                    c3 = c2 + (1 if is_counter[i3] else 0)
                    if c3 > max_c: continue
                    for s in range(ns): cr[s] += raw_mx[i3, s]
                    for i4 in range(i3+1, n):
                        if not allow_dup:
                            if item_ids[i4] == item_ids[i0] or item_ids[i4] == item_ids[i1] or item_ids[i4] == item_ids[i2] or item_ids[i4] == item_ids[i3]: continue
                        p4 = p3 + prices[i4]
                        if p4 > budget: break
                        c4 = c3 + (1 if is_counter[i4] else 0)
                        if c4 > max_c: continue
                        for s in range(ns): cr[s] += raw_mx[i4, s]
                        for i5 in range(i4+1, n):
                            if not allow_dup:
                                if item_ids[i5] == item_ids[i0] or item_ids[i5] == item_ids[i1] or item_ids[i5] == item_ids[i2] or item_ids[i5] == item_ids[i3] or item_ids[i5] == item_ids[i4]: continue
                            p5 = p4 + prices[i5]
                            if p5 > budget: break
                            c5 = c4 + (1 if is_counter[i5] else 0)
                            if c5 > max_c: continue
                            for s in range(ns): cr[s] += raw_mx[i5, s]
                            for i6 in range(i5+1, n):
                                if not allow_dup:
                                    if item_ids[i6] == item_ids[i0] or item_ids[i6] == item_ids[i1] or item_ids[i6] == item_ids[i2] or item_ids[i6] == item_ids[i3] or item_ids[i6] == item_ids[i4] or item_ids[i6] == item_ids[i5]: continue
                                p6 = p5 + prices[i6]
                                if p6 > budget: break
                                c6 = c5 + (1 if is_counter[i6] else 0)
                                if c6 > max_c: continue
                                for s in range(ns): cr[s] += raw_mx[i6, s]
                                ok = _finished_stats(cr, f, prot_mult, b_idx, b_val,
                                                     m_idx, m_val, max_idx, max_val, require_carry, tear_idx, stop_idx)
                                if not ok:
                                    for s in range(ns): cr[s] -= raw_mx[i6, s]
                                    continue
                                score = -1.0 * p6 if min_budget else _objective_score(objective_coeffs, f)
                                for l in range(len(lex_idx)):
                                    rank[l] = np.floor(lex_dir[l] * f[lex_idx[l]] / tolerance + 0.5)
                                if _objective_better(score, rank, bs[-1], best_ranks[-1], bc[-1, 0] >= 0, p6, best_costs[-1]):
                                    for j in range(n_keep):
                                        if _objective_better(score, rank, bs[j], best_ranks[j], bc[j, 0] >= 0, p6, best_costs[j]):
                                            for m in range(n_keep-1, j, -1):
                                                bs[m] = bs[m-1]
                                                best_costs[m] = best_costs[m-1]
                                                for l in range(len(lex_idx)): best_ranks[m, l] = best_ranks[m-1, l]
                                                for mi in range(7): bc[m, mi] = bc[m-1, mi]
                                            for l in range(len(lex_idx)): best_ranks[j, l] = rank[l]
                                            bs[j] = score; bc[j,0]=i0; bc[j,1]=i1; bc[j,2]=i2; bc[j,3]=i3; bc[j,4]=i4; bc[j,5]=i5; bc[j,6]=i6
                                            best_costs[j] = p6
                                            break
                                for s in range(ns): cr[s] -= raw_mx[i6, s]
                            for s in range(ns): cr[s] -= raw_mx[i5, s]
                        for s in range(ns): cr[s] -= raw_mx[i4, s]
                    for s in range(ns): cr[s] -= raw_mx[i3, s]
                for s in range(ns): cr[s] -= raw_mx[i2, s]
            for s in range(ns): cr[s] -= raw_mx[i1, s]
    return bs, bc


_SEARCH_FUNCS = {4: _search_4, 5: _search_5, 6: _search_6, 7: _search_7}


def _is_counter_art(opt) -> bool:
    for pid, val in opt.stats.items():
        if pid.endswith("_accumulation") and val < 0:
            return True
    return False


def search(candidates, k, budget, max_counter, prot_mult,
           cont_bonus_dict, stat_max, priorities, min_thresholds, n_keep,
           allow_duplicates=False, min_budget=False, require_carry=True,
           max_thresholds=None, objective=None, optimization_mode="balanced",
           priority_order=None, objective_tolerance=1e-6):
    """Search all C(n, k) combinations using Numba JIT.

    Returns (scores, combos) where combos[i] = array of candidate indices.
    min_budget=True: score = -price (find cheapest builds meeting thresholds).
    """
    n = len(candidates)
    from ._objective import Objective, direction
    objective = objective or Objective.create(
        priorities, stat_max, "min_budget" if min_budget else optimization_mode,
        priority_order, objective_tolerance,
    )
    min_budget = objective.mode == "min_budget"
    if n < k:
        return np.array([]), np.zeros((0, k), dtype=np.int32)
    # The JIT loops break on exceeding budget, requiring global price order.
    # Keep a mapping so callers still receive indices into their original pool.
    original_indices = np.argsort(
        np.array([o.buyout_price for o in candidates]), kind="stable"
    )
    candidates = [candidates[int(i)] for i in original_indices]

    _DANGER_KEYS = ["radiation_accumulation", "biological_accumulation",
                    "psycho_accumulation", "thermal_accumulation", "frost_accumulation"]
    all_keys = set()
    all_keys.update(priorities.keys())
    all_keys.update(min_thresholds.keys())
    all_keys.update((max_thresholds or {}).keys())
    all_keys.update(objective.keys)
    all_keys.update(_DANGER_KEYS)
    if "stopping_protection" in all_keys:
        all_keys.add("tear_dmg_factor")
    for opt in candidates:
        all_keys.update(k for k in opt.stats.keys() if k in all_keys)
    non_danger = sorted(all_keys - set(_DANGER_KEYS))
    # Always put all 5 danger stats first — JIT functions check range(5)
    # with _DANGER_THRESH[0..4]. Missing danger stats get 0.0 (no penalty).
    # If fewer than 5 are present, range(5) overflows into non-danger stats
    # (speed, stamina) and penalizes them as danger stats.
    stat_keys = _DANGER_KEYS + non_danger
    n_stats = len(stat_keys)
    objective_coeffs = np.zeros(n_stats)
    for key, coefficient in zip(objective.keys, objective.coefficients):
        objective_coeffs[stat_keys.index(key)] = coefficient
    lex_idx = np.array([stat_keys.index(key) for key in objective.order], dtype=np.int32)
    lex_dir = np.array([direction(key) for key in objective.order])
    raw_mx = np.array([[opt.stats.get(key, 0.0) for key in stat_keys]
                       for opt in candidates], dtype=np.float64)
    tear_idx = stat_keys.index("tear_dmg_factor") if "tear_dmg_factor" in stat_keys else -1
    stop_idx = stat_keys.index("stopping_protection") if "stopping_protection" in stat_keys else -1
    prices = np.array([o.buyout_price for o in candidates], dtype=np.float64)
    is_counter = np.array([_is_counter_art(o) for o in candidates], dtype=np.bool_)

    b_idx_list, b_val_list = [], []
    for key, val in cont_bonus_dict.items():
        if key in stat_keys:
            b_idx_list.append(stat_keys.index(key))
            b_val_list.append(val)
    n_b = len(b_idx_list)
    b_idx = np.array(b_idx_list, dtype=np.int32) if n_b else np.zeros(0, dtype=np.int32)
    b_val = np.array(b_val_list, dtype=np.float64) if n_b else np.zeros(0, dtype=np.float64)

    m_idx_list, m_val_list = [], []
    for key, val in min_thresholds.items():
        if key in stat_keys:
            m_idx_list.append(stat_keys.index(key))
            m_val_list.append(val)
    n_m = len(m_idx_list)
    m_idx = np.array(m_idx_list, dtype=np.int32) if n_m else np.zeros(0, dtype=np.int32)
    m_val = np.array(m_val_list, dtype=np.float64) if n_m else np.zeros(0, dtype=np.float64)
    max_idx = np.array([stat_keys.index(key) for key in (max_thresholds or {})], dtype=np.int32)
    max_val = np.array(list((max_thresholds or {}).values()), dtype=np.float64)

    if k not in _SEARCH_FUNCS:
        bs = np.full(n_keep, -1e18, dtype=np.float64)
        bc = np.full((n_keep, k), -1, dtype=np.int32)
        return bs, bc

    # Existing JIT uniqueness checks operate on physical or type conflict groups.
    item_id_arr = conflict_ids(candidates, allow_duplicates)
    check_conflicts = len(np.unique(item_id_arr)) != n

    func = _SEARCH_FUNCS[k]
    result = func(
        raw_mx, prices, is_counter, item_id_arr, prot_mult, budget, max_counter, n_keep,
        tear_idx, stop_idx, b_idx, b_val, m_idx, m_val,
        not check_conflicts,
        min_budget,
        require_carry,
        objective_coeffs, lex_idx, lex_dir, objective.tolerance, max_idx, max_val,
    )
    if result is None:
        # Safety: JIT function returned None (missing return statement or internal error)
        bs = np.full(n_keep, -1e18, dtype=np.float64)
        bc = np.full((n_keep, k), -1, dtype=np.int32)
        return bs, bc
    scores, combos = result
    valid = combos >= 0
    combos[valid] = original_indices[combos[valid]]
    return scores, combos

"""
Beam search engine for artifact build optimization.

O(n × beam_width × k) instead of O(n^k) for brute-force Numba search.
Handles ALL auction lots (500+) in seconds.

Completed scores match builder.priority_value, or negative price in min_budget.
Prefix feasibility penalties remain heuristic; a finite beam is not exact.
"""
import os
import warnings
import gc
import logging
import traceback

# GPU backend: CuPy on NVIDIA GPUs. Auto-disabled if no CUDA (e.g. AMD machines).
# Suppress CuPy "CUDA path could not be detected" UserWarning — cosmetic on non-NVIDIA.
warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*",
                        category=UserWarning, module="cupy._environment")
_GPU_OK = False
cp = None
try:
    import cupy as cp
    import site
    for _sp in site.getsitepackages():
        _nvidia_base = os.path.join(_sp, 'nvidia')
        if os.path.exists(_nvidia_base):
            os.environ['CUDA_PATH'] = _nvidia_base
            for _s in ['cuda_nvrtc', 'cuda_runtime', 'cublas']:
                _bd = os.path.join(_nvidia_base, _s, 'bin')
                if os.path.exists(_bd):
                    os.environ['PATH'] = _bd + os.pathsep + os.environ.get('PATH', '')
            break
    cp.array([1])
    _GPU_OK = True
except Exception:
    cp = None
import numpy as np

from ._objective import Objective
from ._physical_items import conflict_ids

log = logging.getLogger(__name__)


def _is_cuda_oom(error):
    if cp is None:
        return False
    if isinstance(error, cp.cuda.memory.OutOfMemoryError):
        return True
    # CUDA Runtime cudaErrorMemoryAllocation and Driver CUDA_ERROR_OUT_OF_MEMORY.
    for api, name in (("runtime", "CUDARuntimeError"), ("driver", "CUDADriverError")):
        kind = getattr(getattr(cp.cuda, api, None), name, None)
        if kind is not None and isinstance(error, kind) and error.status == 2:
            return True
    return False


def _release_gpu_resources():
    # Failed frames have already been cleared. Collect any closure cycles before
    # releasing cached blocks; live allocations of concurrent callers are retained.
    gc.collect()
    stream = cp.cuda.get_current_stream()
    actions = (stream.synchronize,
               lambda: cp.get_default_memory_pool().free_all_blocks(stream=stream),
               lambda: cp.get_default_pinned_memory_pool().free_all_blocks())
    for action in actions:
        try:
            action()
        except Exception as error:
            if not _is_cuda_oom(error):
                raise
            log.warning("CUDA OOM during cleanup: %s", error)


def _clear_failed_frames(error):
    pending, seen = [error], set()
    while pending:
        failure = pending.pop()
        if id(failure) in seen:
            continue
        seen.add(id(failure))
        pending.extend(cause for cause in (failure.__cause__, failure.__context__)
                       if cause is not None)
        traceback.clear_frames(failure.__traceback__)
        failure.__traceback__ = None

_DANGER_KEYS = [
    "radiation_accumulation", "biological_accumulation",
    "psycho_accumulation", "thermal_accumulation", "frost_accumulation",
]
_DANGER_THRESH = np.array([10.0, 10.0, 10.0, 10.0, 20.0], dtype=np.float64)


def _is_counter_art(opt) -> bool:
    """Same as _numba_search._is_counter_art."""
    for pid, val in opt.stats.items():
        if pid.endswith("_accumulation") and val < 0:
            return True
    return False


def search(candidates, k, budget, max_counter, prot_mult,
           cont_bonus_dict, stat_max, priorities, min_thresholds, n_keep,
           allow_duplicates=False, min_budget=False, require_carry=True,
           beam_width=2000, max_thresholds=None, objective=None,
           optimization_mode="balanced", priority_order=None, objective_tolerance=1e-6,
           *, danger_lookahead=None, constraint_lookahead=False, expansion_chunk_size=None,
           joint_budget_tail=False, joint_tail_limit=256, joint_tail_frontier=False,
           joint_budget_scan=False, joint_scan_block=32, joint_scan_max_pairs=32768):
    """Beam search over C(n, k) combinations.

    Returns (scores, combos) -- same format as Numba search.
    combos[i] = array of candidate indices into the candidates list.
    """
    if danger_lookahead is not None and not isinstance(danger_lookahead, bool):
        raise ValueError("danger_lookahead must be a boolean or None")
    if not isinstance(constraint_lookahead, bool):
        raise ValueError("constraint_lookahead must be a boolean")
    if not isinstance(joint_budget_tail, bool):
        raise ValueError("joint_budget_tail must be a boolean")
    if not isinstance(joint_tail_frontier, bool):
        raise ValueError("joint_tail_frontier must be a boolean")
    if not isinstance(joint_budget_scan, bool):
        raise ValueError("joint_budget_scan must be a boolean")
    from ._min_budget_stream import positive_integer
    positive_integer(joint_scan_block, 'joint_scan_block')
    positive_integer(joint_scan_max_pairs, 'joint_scan_max_pairs')
    if isinstance(joint_tail_limit, bool) or not isinstance(joint_tail_limit, int) or joint_tail_limit < 1:
        raise ValueError("joint_tail_limit must be a positive integer")
    if expansion_chunk_size is not None and (isinstance(expansion_chunk_size, bool)
            or not isinstance(expansion_chunk_size, int) or expansion_chunk_size < 1):
        raise ValueError("expansion_chunk_size must be a positive integer or None")
    objective = objective or Objective.create(
        priorities, stat_max, "min_budget" if min_budget else optimization_mode,
        priority_order, objective_tolerance,
    )
    args = (candidates, k, budget, max_counter, prot_mult, cont_bonus_dict,
            stat_max, priorities, min_thresholds, n_keep)
    kwargs = dict(allow_duplicates=allow_duplicates, min_budget=min_budget,
                  require_carry=require_carry, beam_width=beam_width,
                  max_thresholds=max_thresholds, objective=objective,
                  optimization_mode=optimization_mode, priority_order=priority_order,
                  objective_tolerance=objective_tolerance,
                  danger_lookahead=(os.environ.get("STALZONE_DANGER_LOOKAHEAD", "0") == "1"
                                    if danger_lookahead is None else danger_lookahead),
                  constraint_lookahead=constraint_lookahead, expansion_chunk_size=expansion_chunk_size,
                  joint_budget_tail=joint_budget_tail, joint_tail_limit=joint_tail_limit,
                  joint_tail_frontier=joint_tail_frontier,
                  joint_budget_scan=joint_budget_scan, joint_scan_block=joint_scan_block,
                  joint_scan_max_pairs=joint_scan_max_pairs)
    use_gpu = _GPU_OK and cp is not None and beam_width <= 20000
    if not use_gpu:
        return _search_impl(*args, **kwargs, use_gpu=False)
    try:
        return _search_impl(*args, **kwargs, use_gpu=True)
    except Exception as error:
        if not _is_cuda_oom(error):
            raise
        log.warning("CUDA OOM; restarting full Beam search on CPU: %s", error)
        # An exception traceback holds arrays in abandoned GPU frames. Merely
        # freeing the pool inside this except block would leave them live.
        _clear_failed_frames(error)
    # Leave the exception scope before cleanup/retry. No global backend toggle,
    # no partial beam transfer, and no recursive fallback on CPU failures.
    _release_gpu_resources()
    return _search_impl(*args, **kwargs, use_gpu=False)


def _search_impl(candidates, k, budget, max_counter, prot_mult,
                 cont_bonus_dict, stat_max, priorities, min_thresholds, n_keep,
                 allow_duplicates=False, min_budget=False, require_carry=True,
                 beam_width=2000, max_thresholds=None, objective=None,
                 optimization_mode="balanced", priority_order=None, objective_tolerance=1e-6,
                 *, use_gpu=False, danger_lookahead=False, constraint_lookahead=False,
                 expansion_chunk_size=None, joint_budget_tail=False, joint_tail_limit=256, joint_tail_frontier=False,
                 joint_budget_scan=False, joint_scan_block=32, joint_scan_max_pairs=32768):
    """One complete attempt with a request-local backend and Lookahead flag."""
    objective = objective or Objective.create(
        priorities, stat_max, "min_budget" if min_budget else optimization_mode,
        priority_order, objective_tolerance,
    )
    min_budget = objective.mode == "min_budget"
    _danger_lookahead = (
        danger_lookahead
        and not min_budget
    )
    n = len(candidates)
    _relax_partial_danger = (
        not _danger_lookahead and
        not min_budget
        and (
            set(objective.keys) == {"sprint_speed_modifier"}
            or set(objective.keys) == {"stopping_protection"}
        )
    )
    if n < k or k < 1:
        return np.array([]), np.zeros((0, max(k, 1)), dtype=np.int32)

    # ---- Build matrices (same logic as Numba search) ----
    all_keys = set()
    all_keys.update(priorities.keys())
    all_keys.update(objective.keys)
    all_keys.update(min_thresholds.keys())
    all_keys.update((max_thresholds or {}).keys())
    all_keys.update(_DANGER_KEYS)
    if "stopping_protection" in all_keys:
        all_keys.add("tear_dmg_factor")
    non_danger = sorted(all_keys - set(_DANGER_KEYS))
    stat_keys = _DANGER_KEYS + non_danger
    n_stats = len(stat_keys)
    primary_idx = np.array([stat_keys.index(pid) for pid in objective.keys], dtype=np.int32)
    primary_sign = np.array(objective.coefficients)

    raw_mx = np.zeros((n, n_stats), dtype=np.float64)
    # The actual objective is evaluated from final stats, not a weighted matrix.
    # Empty placeholders retain _score_vec's private legacy call signature.
    w_mx = np.zeros((n, 0), dtype=np.float64)
    for i, opt in enumerate(candidates):
        for j, key in enumerate(stat_keys):
            val = opt.stats.get(key, 0.0)
            raw_mx[i, j] = val

    from ._danger_lookahead import build_reduction_table

    prices = np.array(
        [o.buyout_price for o in candidates],
        dtype=np.float64,
    )

    _budget_reduction_table = None
    _budget_caps = None

    if _danger_lookahead:
        _budget_reduction_table, _budget_caps = build_reduction_table(
            raw_mx, prices, budget, buckets=64
        )

    is_counter = np.array([_is_counter_art(o) for o in candidates], dtype=np.float64)
    _price_bounds = None
    if min_budget:
        from ._min_budget import suffix_bounds
        _price_bounds = suffix_bounds(raw_mx, prices, k)
    _constraint_bounds = None
    # This weighted penalty relaxation has not preserved lexicographic quality.
    # Keep that mode's established prefix order; price-first already has bounds.
    if constraint_lookahead and objective.mode == "balanced" and (min_thresholds or max_thresholds):
        from ._constraint_lookahead import suffix_extrema, optimistic_constraints
        _constraint_bounds = suffix_extrema(raw_mx)

    # Container bonus
    b_idx_list, b_val_list = [], []
    for key, val in cont_bonus_dict.items():
        if key in stat_keys:
            b_idx_list.append(stat_keys.index(key))
            b_val_list.append(val)
    n_b = len(b_idx_list)
    b_idx_arr = np.array(b_idx_list, dtype=np.int32) if n_b else np.zeros(0, dtype=np.int32)
    b_val_arr = np.array(b_val_list, dtype=np.float64) if n_b else np.zeros(0, dtype=np.float64)

    # EHP indices
    hp_idx = stat_keys.index("health_bonus") if "health_bonus" in stat_keys else -1
    bul_idx = stat_keys.index("bullet_dmg_factor") if "bullet_dmg_factor" in stat_keys else -1
    hp_w = float(priorities.get("health_bonus", 0.0))
    bul_w = float(priorities.get("bullet_dmg_factor", 0.0))
    if hp_idx >= 0 and bul_idx >= 0 and hp_w > 0 and bul_w > 0:
        ehp_w = (hp_w + bul_w) / 2.0
        max_ehp = max(
            (stat_max.get("health_bonus", 1.0) * stat_max.get("bullet_dmg_factor", 1.0)) ** 0.5,
            1.0,
        )
    else:
        ehp_w = 0.0
        max_ehp = 0.0

    # Stopping indices
    tear_idx = stat_keys.index("tear_dmg_factor") if "tear_dmg_factor" in stat_keys else -1
    stop_idx = stat_keys.index("stopping_protection") if "stopping_protection" in stat_keys else -1
    stop_w = float(priorities.get("stopping_protection", 0.0))
    # Recalculated stopping must use the same normalization as w_mx.
    stop_max = stat_max.get("stopping_protection", 1.0)
    if stop_max > 0:
        stop_w /= stop_max
    has_tear = tear_idx >= 0 and stop_idx >= 0

    # With repeats enabled, groups are physical objects; otherwise item types.
    item_id_arr = conflict_ids(candidates, allow_duplicates)
    check_conflicts = len(np.unique(item_id_arr)) != n
    _joint_tail = None
    _tail_order = None
    if min_budget and joint_budget_scan:
        from ._min_budget_stream import rank_last_slot as scan_last_slot
        _tail_order = np.lexsort((np.arange(n), prices)).astype(np.int32)
    if min_budget and joint_budget_tail and not joint_budget_scan:
        from ._min_budget_joint import tail_representatives, suffix_representatives, rank_last_slot
        representatives = suffix_representatives if joint_tail_frontier else tail_representatives
        _joint_tail = representatives(raw_mx, prices, is_counter, item_id_arr, k, joint_tail_limit)

    # Min thresholds
    m_idx_list, m_val_list = [], []
    for key, val in min_thresholds.items():
        if key in stat_keys:
            m_idx_list.append(stat_keys.index(key))
            m_val_list.append(val)
    n_m = len(m_idx_list)
    m_idx_arr = np.array(m_idx_list, dtype=np.int32) if n_m else np.zeros(0, dtype=np.int32)
    m_val_arr = np.array(m_val_list, dtype=np.float64) if n_m else np.zeros(0, dtype=np.float64)

    # Max thresholds
    mx_idx_list, mx_val_list = [], []
    for key, val in (max_thresholds or {}).items():
        if key in stat_keys:
            mx_idx_list.append(stat_keys.index(key))
            mx_val_list.append(val)
    n_mx = len(mx_idx_list)
    mx_idx_arr = np.array(mx_idx_list, dtype=np.int32) if n_mx else np.zeros(0, dtype=np.int32)
    mx_val_arr = np.array(mx_val_list, dtype=np.float64) if n_mx else np.zeros(0, dtype=np.float64)

    def complete_price(stats, costs, backend):
        return _score_vec(
            stats, backend.zeros((len(costs), 0)), costs, True,
            hp_idx, bul_idx, ehp_w, max_ehp, prot_mult, n_b, b_idx_arr, b_val_arr,
            tear_idx, stop_idx, stop_w, has_tear, m_idx_arr, m_val_arr, n_m,
            mx_idx_arr, mx_val_arr, n_mx, enforce_constraints=True,
            require_carry=require_carry, xp=backend)

    def refine_tail(scores, stats, costs, last, counters, indices, backend):
        active = backend.flatnonzero(backend.isfinite(scores))
        if len(active):
            scores = scores.copy()
            if _tail_order is not None:
                scores[active] = scan_last_slot(
                    stats[active], costs[active], last[active], counters[active], indices[active],
                    raw_mx, prices, is_counter, item_id_arr, _tail_order, budget, max_counter,
                    lambda raw, total: complete_price(raw, total, backend), backend,
                    block_size=joint_scan_block, max_pairs=joint_scan_max_pairs)
                return scores
            scores[active] = rank_last_slot(
                stats[active], costs[active], last[active], counters[active], indices[active],
                raw_mx, prices, is_counter, item_id_arr, _joint_tail, budget, max_counter,
                lambda raw, total: complete_price(raw, total, backend), backend)
        return scores

    # ---- Beam search ----

    # Level 0: single items
    beam_stats = raw_mx.copy()               # (n, n_stats)
    beam_wstats = w_mx.copy()
    beam_prices = prices.copy()
    beam_counters = is_counter.copy()
    beam_last_idx = np.arange(n, dtype=np.int32)
    beam_indices = np.arange(n, dtype=np.int32).reshape(-1, 1)
    beam_item_ids = item_id_arr.reshape(-1, 1) if check_conflicts else None

    # Filter: price <= budget, counter <= max_counter
    mask = (beam_prices <= budget) & (beam_counters <= max_counter)
    beam_stats = beam_stats[mask]
    beam_wstats = beam_wstats[mask]
    beam_prices = beam_prices[mask]
    beam_counters = beam_counters[mask]
    beam_last_idx = beam_last_idx[mask]
    beam_indices = beam_indices[mask]
    if beam_item_ids is not None:
        beam_item_ids = beam_item_ids[mask]

    # Score level 0
    beam_scores = np.zeros(len(beam_prices)) if min_budget and k > 1 else _score_vec(
        beam_stats, beam_wstats, beam_prices, min_budget,
        hp_idx, bul_idx, ehp_w, max_ehp,
        prot_mult, n_b, b_idx_arr, b_val_arr,
        tear_idx, stop_idx, stop_w, has_tear,
        m_idx_arr, m_val_arr, n_m,
        mx_idx_arr, mx_val_arr, n_mx,
        ignore_danger_penalty=_relax_partial_danger,
        enforce_constraints=k == 1,
        require_carry=require_carry,
        primary_idx=primary_idx, primary_sign=primary_sign,
        remaining_slots=k - 1 if _danger_lookahead else 0,
        max_danger_reduction=(
            _budget_reduction_table[
                beam_last_idx + 1,
                np.clip(
                    np.searchsorted(
                        _budget_caps,
                        np.maximum(budget - beam_prices, 0.0),
                        side="left",
                    ),
                    0,
                    len(_budget_caps) - 1,
                ),
            ]
            if _danger_lookahead and k > 1 else None
        ),
        constraint_stats=(optimistic_constraints(
            beam_stats, beam_last_idx, k - 1, _constraint_bounds, prot_mult,
            b_idx_arr, b_val_arr, np) if _constraint_bounds is not None and k > 1 else None),
        xp=np,
    )

    # Keep top beam_width
    if min_budget and k > 1:
        from ._min_budget import rank_prefixes
        beam_scores = rank_prefixes(
            beam_stats, beam_prices, beam_last_idx, k - 1, _price_bounds,
            budget, prot_mult, b_idx_arr, b_val_arr, m_idx_arr, m_val_arr,
            mx_idx_arr, mx_val_arr, require_carry, _DANGER_THRESH, stop_idx, np,
        )
        if k == 2 and (_joint_tail is not None or _tail_order is not None):
            beam_scores = refine_tail(beam_scores, beam_stats, beam_prices, beam_last_idx,
                                      beam_counters, beam_indices, np)
    (beam_stats, beam_wstats, beam_prices, beam_counters,
     beam_last_idx, beam_indices, beam_item_ids, beam_scores) = _trim(
        beam_stats, beam_wstats, beam_prices, beam_counters,
        beam_last_idx, beam_indices, beam_item_ids, beam_scores, beam_width,
        ranks=_lex_ranks(beam_stats, beam_scores, objective, stat_keys, prot_mult,
                         b_idx_arr, b_val_arr, tear_idx, stop_idx, k == 1, np),
        costs=beam_prices,
    )

    # Convert all arrays to GPU (CuPy) when available.
    # Auto-fallback to CPU for large beam_width to avoid VRAM OOM.
    xp = cp if use_gpu else np
    if xp is cp:
        raw_mx = cp.asarray(raw_mx)
        w_mx = cp.asarray(w_mx)
        prices = cp.asarray(prices)
        is_counter = cp.asarray(is_counter)
        beam_stats = cp.asarray(beam_stats)
        beam_wstats = cp.asarray(beam_wstats)
        beam_prices = cp.asarray(beam_prices)
        beam_counters = cp.asarray(beam_counters)
        beam_last_idx = cp.asarray(beam_last_idx)
        beam_indices = cp.asarray(beam_indices)
        beam_scores = cp.asarray(beam_scores)
        item_id_arr = cp.asarray(item_id_arr)
        b_idx_arr = cp.asarray(b_idx_arr)
        b_val_arr = cp.asarray(b_val_arr)
        m_idx_arr = cp.asarray(m_idx_arr)
        m_val_arr = cp.asarray(m_val_arr)
        mx_idx_arr = cp.asarray(mx_idx_arr)
        mx_val_arr = cp.asarray(mx_val_arr)
        primary_idx = cp.asarray(primary_idx)
        primary_sign = cp.asarray(primary_sign)
        _DANGER_THRESH_GPU = cp.asarray(_DANGER_THRESH)
        if beam_item_ids is not None:
            beam_item_ids = cp.asarray(beam_item_ids)
        _dthresh = _DANGER_THRESH_GPU
    else:
        _dthresh = _DANGER_THRESH

    if _danger_lookahead:
        _budget_reduction_table = xp.asarray(
            _budget_reduction_table
        )
        _budget_caps = xp.asarray(_budget_caps)
    if min_budget:
        _price_bounds = tuple(xp.asarray(a) for a in _price_bounds)
    if _constraint_bounds is not None:
        _constraint_bounds = tuple(xp.asarray(a) for a in _constraint_bounds)
    if _joint_tail is not None:
        _joint_tail = xp.asarray(_joint_tail)
    if _tail_order is not None:
        _tail_order = xp.asarray(_tail_order)

    # Levels 1 to k-1 — fully vectorized with tie-breaking fix
    for level in range(1, k):
        B = len(beam_scores)
        if B == 0:
            break
        if min_budget:
            from ._min_budget import expand_level
            remaining = k - level - 1

            def price_score(stats, costs, last):
                if remaining:
                    return rank_prefixes(
                        stats, costs, last, remaining, _price_bounds, budget,
                        prot_mult, b_idx_arr, b_val_arr, m_idx_arr, m_val_arr,
                        mx_idx_arr, mx_val_arr, require_carry, _dthresh, stop_idx, xp,
                    )
                return _score_vec(
                    stats, xp.zeros((len(costs), 0)), costs, True,
                    hp_idx, bul_idx, ehp_w, max_ehp, prot_mult,
                    n_b, b_idx_arr, b_val_arr, tear_idx, stop_idx, stop_w, has_tear,
                    m_idx_arr, m_val_arr, n_m, mx_idx_arr, mx_val_arr, n_mx,
                    enforce_constraints=True, require_carry=require_carry,
                    xp=xp,
                )

            def joint_price_score(stats, costs, last, counters, indices):
                return refine_tail(price_score(stats, costs, last), stats, costs, last,
                                   counters, indices, xp)

            (beam_stats, beam_prices, beam_counters, beam_last_idx,
             beam_indices, beam_item_ids, beam_scores) = expand_level(
                beam_stats, beam_prices, beam_counters, beam_last_idx,
                beam_indices, beam_item_ids, raw_mx, prices, is_counter,
                item_id_arr, budget, max_counter, beam_width, not check_conflicts,
                price_score, _top_indices, xp,
                joint_score=joint_price_score if remaining == 1 and (
                    _joint_tail is not None or _tail_order is not None) else None,
            )
            beam_wstats = xp.zeros((len(beam_scores), 0))
            continue
        # Vectorized expansion: compute all B×n combinations at once
        cand_range = xp.arange(n, dtype=np.int32)
        if allow_duplicates:
            order_mask = beam_last_idx[:, None] < cand_range[None, :]
        else:
            order_mask = beam_last_idx[:, None] < cand_range[None, :]
        budget_mask = (beam_prices[:, None] + prices[None, :]) <= budget
        counter_mask = (beam_counters[:, None] + is_counter[None, :]) <= max_counter
        if check_conflicts:
            dedup_mask = xp.all(beam_item_ids[:, None, :] != item_id_arr[None, :, None], axis=2)
            valid = order_mask & budget_mask & counter_mask & dedup_mask
        else:
            valid = order_mask & budget_mask & counter_mask
        if not xp.any(valid):
            break
        beam_idx_flat, cand_idx_flat = xp.where(valid)
        # Tie-breaking fix: sort to match for-loop order (cand outer, beam inner)
        # so argpartition breaks ties identically to the original for-loop
        _sort = xp.argsort(cand_idx_flat.astype(xp.float64) * B + beam_idx_flat.astype(xp.float64))
        beam_idx_flat = beam_idx_flat[_sort]
        cand_idx_flat = cand_idx_flat[_sort]
        num_valid = len(beam_idx_flat)
        new_prices = beam_prices[beam_idx_flat] + prices[cand_idx_flat]
        def rank_expansions(stats, wstats, costs, last):
            scores = _score_vec(
                stats, wstats, costs, min_budget,
                hp_idx, bul_idx, ehp_w, max_ehp, prot_mult, n_b, b_idx_arr, b_val_arr,
                tear_idx, stop_idx, stop_w, has_tear, m_idx_arr, m_val_arr, n_m,
                mx_idx_arr, mx_val_arr, n_mx,
                ignore_danger_penalty=_relax_partial_danger,
                enforce_constraints=level == k - 1, require_carry=require_carry,
                primary_idx=primary_idx, primary_sign=primary_sign,
                remaining_slots=(k - level - 1) if _danger_lookahead else 0,
                max_danger_reduction=(
                    _budget_reduction_table[last + 1, xp.clip(xp.searchsorted(
                        _budget_caps, xp.maximum(budget - costs, 0.0), side="left"),
                        0, len(_budget_caps) - 1)]
                    if _danger_lookahead and level < k - 1 else None),
                constraint_stats=(optimistic_constraints(
                    stats, last, k - level - 1, _constraint_bounds, prot_mult,
                    b_idx_arr, b_val_arr, xp)
                    if _constraint_bounds is not None and level < k - 1 else None), xp=xp)
            ranks = _lex_ranks(stats, scores, objective, stat_keys, prot_mult,
                               b_idx_arr, b_val_arr, tear_idx, stop_idx, level == k - 1, xp)
            return scores, ranks
        if expansion_chunk_size is None:
            new_stats = beam_stats[beam_idx_flat] + raw_mx[cand_idx_flat]
            new_wstats = beam_wstats[beam_idx_flat] + w_mx[cand_idx_flat]
            all_scores, all_ranks = rank_expansions(new_stats, new_wstats, new_prices, cand_idx_flat)
        else:
            def block(start, end):
                parent, candidate = beam_idx_flat[start:end], cand_idx_flat[start:end]
                return rank_expansions(beam_stats[parent] + raw_mx[candidate],
                                       beam_wstats[parent] + w_mx[candidate],
                                       new_prices[start:end], candidate)
            all_scores, all_ranks = _score_blocks(num_valid, expansion_chunk_size, block, xp,
                                                 len(objective.order))
        all_beam_idx = beam_idx_flat.astype(np.int32)
        all_cand_idx = cand_idx_flat.astype(np.int32)
        top = _top_indices(
            all_scores, beam_width, xp,
            ranks=all_ranks,
            costs=new_prices,
        )
        sel_beam = all_beam_idx[top]
        sel_cand = all_cand_idx[top]
        beam_stats = beam_stats[sel_beam] + raw_mx[sel_cand]
        beam_wstats = beam_wstats[sel_beam] + w_mx[sel_cand]
        beam_prices = beam_prices[sel_beam] + prices[sel_cand]
        beam_counters = beam_counters[sel_beam] + is_counter[sel_cand]
        beam_last_idx = sel_cand.astype(np.int32)
        beam_indices = xp.column_stack([beam_indices[sel_beam], sel_cand.astype(np.int32)])
        if check_conflicts:
            beam_item_ids = xp.column_stack([
                beam_item_ids[sel_beam],
                item_id_arr[sel_cand].reshape(-1, 1)
            ])
        beam_scores = all_scores[top]

    # ---- Final level: enforce min_thresholds and carryable ----
    B = len(beam_scores)
    if B == 0 or beam_indices.shape[1] != k:
        return np.array([]), np.zeros((0, k), dtype=np.int32)

    # Compute final f/fw (with container bonus + stopping recalc)
    f = _final_stats(beam_stats, prot_mult, b_idx_arr, b_val_arr, tear_idx, stop_idx, xp)
    # Carryable check: protection on RAW accumulation, then container bonus
    danger = beam_stats[:, :5].copy()  # raw, no container bonus
    for d in range(4):
        danger[:, d] = xp.where(danger[:, d] > 0, danger[:, d] * prot_mult, danger[:, d])
    for b in range(n_b):
        if b_idx_arr[b] < 5:
            danger[:, b_idx_arr[b]] += b_val_arr[b]
    # Constraints refer to final game values, including container protection.
    f[:, :5] = danger
    valid = xp.ones(B, dtype=bool)
    for m in range(n_m):
        valid &= f[:, m_idx_arr[m]] >= m_val_arr[m]
    for m in range(n_mx):
        valid &= f[:, mx_idx_arr[m]] <= mx_val_arr[m]
    carry = xp.ones(B, dtype=bool)
    for d in range(5):
        carry &= danger[:, d] <= _dthresh[d]
    if require_carry:
        valid &= carry
    # Final scores
    if min_budget:
        final_scores = -beam_prices.copy()
    else:
        final_scores = objective.matrix_score(f, stat_keys, xp)
    valid_scores = final_scores[valid]
    valid_indices = beam_indices[valid]
    if len(valid_scores) == 0:
        return np.array([]), np.zeros((0, k), dtype=np.int32)
    n_results = min(len(valid_scores), n_keep)
    top = _top_indices(valid_scores, n_results, xp,
                       ranks=objective.matrix_ranks(f[valid], stat_keys, xp)
                       if objective.mode == "lexicographic" else None, costs=beam_prices[valid])

    # Convert CuPy arrays back to NumPy for the caller
    if use_gpu:
        return cp.asnumpy(valid_scores[top]), cp.asnumpy(valid_indices[top])
    return valid_scores[top], valid_indices[top]


def _score_blocks(count, size, block, xp, rank_width=0):
    """Keep original expansion order and perform one global selection afterward.

    No block-local top-k merge: it would change equal-score boundary choices.
    Only scores/ranks span all expansions; stat temporaries are block-sized.
    """
    scores = xp.empty(count, dtype=np.float64)
    ranks = xp.empty((count, rank_width), dtype=np.float64) if rank_width else None
    for start in range(0, count, size):
        end = min(start + size, count)
        values, ranking = block(start, end)
        scores[start:end] = values
        if ranks is not None:
            ranks[start:end] = ranking
    return scores, ranks


def _score_vec(stats, wstats, prices, min_budget,
               hp_idx, bul_idx, ehp_w, max_ehp,
               prot_mult, n_b, b_idx, b_val,
               tear_idx, stop_idx, stop_w, has_tear,
               m_idx, m_val, n_m,
               mx_idx, mx_val, n_mx, ignore_danger_penalty=False,
               remaining_slots=0, max_danger_reduction=None,
               enforce_constraints=False, require_carry=False,
               primary_idx=None, primary_sign=None, xp=None, constraint_stats=None):
    if xp is None:
        xp = np if isinstance(stats, np.ndarray) or cp is None else cp.get_array_module(stats)
    _dt = cp.asarray(_DANGER_THRESH) if (xp is cp) else _DANGER_THRESH
    B = stats.shape[0]
    if B == 0:
        return xp.array([], dtype=np.float64) if xp is cp else np.array([], dtype=np.float64)
    f = _final_stats(stats, prot_mult, b_idx, b_val, tear_idx, stop_idx, xp)
    fw = wstats.copy() if primary_idx is None and not min_budget else None
    for b in range(n_b):
        if fw is not None:
            fw[:, b_idx[b]] += b_val[b]
    # Actual protected accumulation for min/max constraints. The optimistic
    # continuation below affects only the danger penalty used to rank prefixes.
    actual_danger = f[:, :5]
    if has_tear and fw is not None:
        fw[:, stop_idx] = f[:, stop_idx] * stop_w
    if min_budget:
        scores = -prices.copy()
    elif primary_idx is not None:
        scores = xp.sum(f[:, primary_idx] * primary_sign, axis=1)
    else:
        # Compatibility for callers supplying an already-weighted matrix.
        # Production search always supplies fixed objective coefficients above.
        scores = xp.sum(fw, axis=1)
    if min_budget and n_m > 0:
        _thr_mult = 100.0
        _danger_mult = 1000.0
    elif min_budget:
        _thr_mult = 1_000_000_000.0
        _danger_mult = _thr_mult
    else:
        _thr_mult = 5.0
        _danger_mult = 1000.0
    danger = stats[:, :5].copy()
    if remaining_slots > 0 and max_danger_reduction is not None:
        reduction = xp.asarray(max_danger_reduction)
        danger += remaining_slots * (
            reduction[None, :] if reduction.ndim == 1 else reduction
        )
    for d in range(4):
        danger[:, d] = xp.where(danger[:, d] > 0, danger[:, d] * prot_mult, danger[:, d])
    for b in range(n_b):
        if b_idx[b] < 5:
            danger[:, b_idx[b]] += b_val[b]
    completed_objective = enforce_constraints and primary_idx is not None
    for d in range(0 if min_budget or completed_objective else 5):
        v = danger[:, d]
        exceeding = v > _dt[d]
        if not ignore_danger_penalty:
            scores -= xp.where(exceeding, (v - _dt[d]) * _danger_mult, 0)
        soft = (v > 0) & ~exceeding
        scores -= xp.where(soft, (v / _dt[d]) * 0.15, 0)
    # Normalize constraint deficits because different stats use different scales.
    for m in range(0 if min_budget else n_m):
        scale = xp.maximum(xp.abs(m_val[m]), 1.0)
        values = (constraint_stats[1][:, m_idx[m]] if constraint_stats is not None
                  and not enforce_constraints and m_idx[m] != stop_idx else f[:, m_idx[m]])
        deficit = xp.maximum(0.0, m_val[m] - values) / scale
        scores -= deficit * _thr_mult
    for mx in range(0 if min_budget else n_mx):
        scale = xp.maximum(xp.abs(mx_val[mx]), 1.0)
        values = (constraint_stats[0][:, mx_idx[mx]] if constraint_stats is not None
                  and not enforce_constraints and mx_idx[mx] != stop_idx else f[:, mx_idx[mx]])
        excess = xp.maximum(0.0, values - mx_val[mx]) / scale
        scores -= excess * _thr_mult
    if enforce_constraints:
        # Completed invalid builds must not displace feasible builds during the
        # last beam trim. Prefixes retain soft penalties because later artifacts
        # can still repair their deficits and excesses.
        feasible = xp.ones(B, dtype=bool)
        for m in range(n_m):
            feasible &= f[:, m_idx[m]] >= m_val[m]
        for m in range(n_mx):
            feasible &= f[:, mx_idx[m]] <= mx_val[m]
        if require_carry:
            feasible &= xp.all(actual_danger <= _dt, axis=1)
        scores = xp.where(feasible, scores, -xp.inf)
    return scores

def _carry_mask(stats, protection, bonus_idx, bonus_val, xp):
    danger = stats[:, :5].copy()
    for d in range(4):
        danger[:, d] = xp.where(danger[:, d] > 0, danger[:, d] * protection, danger[:, d])
    for b in range(len(bonus_idx)):
        if bonus_idx[b] < 5:
            danger[:, bonus_idx[b]] += bonus_val[b]
    return xp.all(danger <= xp.asarray(_DANGER_THRESH), axis=1)


def _final_stats(stats, protection, bonus_idx, bonus_val, tear_idx, stop_idx, xp):
    f = stats.copy()
    for d in range(4):
        f[:, d] = xp.where(f[:, d] > 0, f[:, d] * protection, f[:, d])
    for b in range(len(bonus_idx)):
        f[:, bonus_idx[b]] += bonus_val[b]
    if tear_idx >= 0 and stop_idx >= 0:
        denominator = 100 + f[:, tear_idx]
        denominator = xp.where(denominator == 0, xp.nan, denominator)
        f[:, stop_idx] = (1 - 100 / denominator * (1 - f[:, stop_idx] / 100)) * 100
    return f


def _lex_ranks(stats, scores, objective, stat_keys, protection, bonus_idx, bonus_val,
               tear_idx, stop_idx, completed, xp):
    if objective.mode != "lexicographic":
        return None
    final = _final_stats(stats, protection, bonus_idx, bonus_val, tear_idx, stop_idx, xp)
    adjustment = None if completed else scores - objective.matrix_score(final, stat_keys, xp)
    return objective.matrix_ranks(final, stat_keys, xp, adjustment)


def _top_indices(scores, count, xp, carry=None, ranks=None, costs=None):
    """Feasible first; normal mode also places carryable builds before others."""
    def take(indices, wanted):
        wanted = min(len(indices), wanted)
        if wanted <= 0:
            return xp.zeros(0, dtype=xp.int64)
        if ranks is not None:
            keys = tuple(-ranks[indices, i] for i in reversed(range(ranks.shape[1])))
            if costs is not None:
                keys = (costs[indices],) + keys
            return indices[xp.lexsort(xp.stack(keys))[:wanted]]
        if len(indices) > wanted:
            partition = xp.argpartition(-scores[indices], wanted - 1)
            if costs is None:
                indices = indices[partition[:wanted]]
            else:
                cutoff = scores[indices[partition[wanted - 1]]]
                above = indices[scores[indices] > cutoff]
                tied = indices[scores[indices] == cutoff]
                needed = wanted - len(above)
                if len(tied) > needed:
                    tied = tied[xp.argpartition(costs[tied], needed - 1)[:needed]]
                indices = xp.concatenate((above, tied))
        if costs is not None:
            return indices[xp.lexsort(xp.stack((costs[indices], -scores[indices])))]
        return indices[xp.argsort(-scores[indices])]

    finite = xp.isfinite(scores)
    if carry is None:
        return take(xp.flatnonzero(finite), count)
    good = take(xp.flatnonzero(finite & carry), count)
    other = take(xp.flatnonzero(finite & ~carry), count - len(good))
    return xp.concatenate((good, other))


def _trim(stats, wstats, prices, counters, last_idx, indices, item_ids, scores, beam_width,
          carry=None, ranks=None, costs=None):
    """Keep top beam_width entries by score."""
    top = _top_indices(scores, beam_width, np, carry, ranks, costs)
    stats = stats[top]
    wstats = wstats[top]
    prices = prices[top]
    counters = counters[top]
    last_idx = last_idx[top]
    indices = indices[top]
    if item_ids is not None:
        item_ids = item_ids[top]
    scores = scores[top]
    return stats, wstats, prices, counters, last_idx, indices, item_ids, scores

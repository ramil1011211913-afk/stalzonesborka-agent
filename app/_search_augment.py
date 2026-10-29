"""Multi-pass beam search augmentation for constraint-heavy builds.

When min_thresholds are set the single-pass beam can miss good builds because
the candidate ordering (item_id, price) buries high-stat items.  Running extra
passes with permuted orderings covers complementary regions of the search space.

Usage in builder.py (after the regular beam call):

    if bool(min_thresholds) and _beam_search_fn is None:
        from ._search_augment import multi_pass_beam
        scores, combos = multi_pass_beam(
            scores, combos,
            _search_fn, _numba_search_candidates,
            _sargs[1:], _skwargs_run,
            _beam_width, n_keep,
        )
"""
from __future__ import annotations
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _valid_rows(scores: np.ndarray, combos: np.ndarray):
    """Return (scores, combos) rows where score is not -inf/-1e18."""
    mask = scores > -1e17
    if not np.any(mask):
        return [], []
    return scores[mask].tolist(), [combos[i] for i in np.where(mask)[0]]


def _merge(buckets: list[tuple[float, np.ndarray]], n_keep: int, k: int):
    """Deduplicate by sorted-index key, return top n_keep."""
    seen: dict[tuple, tuple[float, np.ndarray]] = {}
    for score, combo in buckets:
        key = tuple(sorted(int(x) for x in combo))
        if key not in seen or score > seen[key][0]:
            seen[key] = (score, combo)
    ranked = sorted(seen.values(), key=lambda x: -x[0])[:n_keep]
    if not ranked:
        return np.array([], dtype=np.float64), np.zeros((0, k), dtype=np.int32)
    final_scores = np.array([r[0] for r in ranked], dtype=np.float64)
    final_combos = np.array([r[1] for r in ranked], dtype=np.int32)
    return final_scores, final_combos


# ── candidate orderings ───────────────────────────────────────────────────────

def _order_by_stat(candidates, priorities: dict) -> np.ndarray:
    """Sort descending by weighted sum of priority-stat contributions."""
    if not priorities:
        return np.arange(len(candidates), dtype=np.int32)
    keys = list(priorities.keys())
    weights = np.array([priorities[k] for k in keys], dtype=np.float64)
    weights /= weights.sum() + 1e-12
    scores = np.array(
        [sum(float(c.stats.get(k, 0.0)) * w for k, w in zip(keys, weights))
         for c in candidates],
        dtype=np.float64,
    )
    return np.argsort(-scores, kind="stable").astype(np.int32)


def _order_by_constraint_fit(candidates, min_thresholds: dict) -> np.ndarray:
    """Sort by how much each candidate contributes to min_threshold stats.

    Puts the most constraint-relevant artifacts first so the beam has a better
    chance of building feasible prefixes before pruning.
    """
    if not min_thresholds:
        return np.arange(len(candidates), dtype=np.int32)
    keys = list(min_thresholds.keys())
    scores = np.array(
        [sum(max(0.0, float(c.stats.get(k, 0.0))) for k in keys)
         for c in candidates],
        dtype=np.float64,
    )
    return np.argsort(-scores, kind="stable").astype(np.int32)


def _order_random(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n).astype(np.int32)


# ── public API ────────────────────────────────────────────────────────────────

def multi_pass_beam(
    scores0: np.ndarray,
    combos0: np.ndarray,
    search_fn,
    candidates: list,
    sargs_tail: tuple,
    skwargs: dict,
    beam_width: int,
    n_keep: int,
    *,
    n_random_passes: int = 2,
    rng_base_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment a completed beam-search result with extra passes.

    Extra orderings tried:
      • sorted by descending priority-stat contribution
      • sorted by descending constraint-stat contribution
      • n_random_passes random permutations

    Returns merged, deduplicated, top-n_keep result.

    Parameters
    ----------
    scores0, combos0 : output of the initial beam call (original ordering)
    search_fn        : beam search callable (same signature as _beam_search.search)
    candidates       : original candidate list (before any reordering)
    sargs_tail       : positional args AFTER candidates (k, budget, …, n_keep)
    skwargs          : keyword args forwarded to search_fn
    beam_width       : beam width used for extra passes
    n_keep           : how many results to return
    """
    n = len(candidates)
    k = combos0.shape[1] if combos0.ndim == 2 and combos0.shape[0] else (
        sargs_tail[0] if sargs_tail else 1)

    priorities: dict = sargs_tail[6] if len(sargs_tail) > 6 else {}  # arg index in _sargs
    min_thresholds: dict = sargs_tail[7] if len(sargs_tail) > 7 else {}

    # Collect all valid results, starting with pass-0
    bucket: list[tuple[float, np.ndarray]] = []
    s0, c0 = _valid_rows(scores0, combos0)
    bucket.extend(zip(s0, c0))

    # --- Extra orderings ---
    orderings = []

    # 1. Objective-stat order (best contributors first)
    ord_stat = _order_by_stat(candidates, priorities)
    if not np.all(ord_stat == np.arange(n)):  # skip if identical to current
        orderings.append(("stat", ord_stat))

    # 2. Constraint-stat order (most helpful for min_thresholds first)
    if min_thresholds:
        ord_constr = _order_by_constraint_fit(candidates, min_thresholds)
        orderings.append(("constraint", ord_constr))

    # 3. Random passes
    for i in range(n_random_passes):
        orderings.append(("random", _order_random(n, rng_base_seed + i)))

    for _label, perm in orderings:
        shuffled = [candidates[int(i)] for i in perm]
        try:
            sp, cp = search_fn(shuffled, *sargs_tail, beam_width=beam_width, **skwargs)
        except Exception:
            continue
        sv, cv = _valid_rows(sp, cp)
        if sv:
            # Remap indices: cp[i] is index in shuffled → perm[cp[i]] is original index
            for score, combo in zip(sv, cv):
                remapped = perm[np.asarray(combo, dtype=np.int32)]
                bucket.append((score, remapped))

    return _merge(bucket, n_keep, k)

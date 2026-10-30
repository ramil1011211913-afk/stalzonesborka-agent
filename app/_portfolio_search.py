"""Explicit experimental portfolio; ordinary builder requests are unchanged.

The attempt cap applies to each prepared container/armor pool. All attempts use
the same candidates and Objective. Agreement between strategies is not a proof
of optimality. No environment mutation, automatic stopping, or time guarantee.
"""
from dataclasses import dataclass
import time
import numpy as np

from . import _beam_search as beam
from ._objective import Objective
from ._physical_items import is_available


@dataclass(frozen=True)
class Strategy:
    name: str
    width: int
    lookahead: bool


STRATEGIES = (Strategy('fast', 2000, False),
              Strategy('quality', 2000, True),
              Strategy('wide_control', 10000, True))


def _check_limit(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= len(STRATEGIES):
        raise ValueError('max_attempts must be an integer between 1 and 3')


def _completed_stats(candidates, indices, protection, bonuses, objective, minimum, maximum):
    from ._derived_stats import augment_bonus_stats, value as stat_value
    bonuses = augment_bonus_stats(dict(bonuses or {}))
    keys = set(bonuses) | set(objective.keys) | set(minimum) | set(maximum)
    for candidate in candidates:
        keys.update(candidate.stats)
    keys.update(beam._DANGER_KEYS)
    if 'stopping_protection' in keys:
        keys.add('tear_dmg_factor')
    keys = list(beam._DANGER_KEYS) + sorted(keys - set(beam._DANGER_KEYS))
    raw = np.array([[stat_value(c.stats, key) for key in keys] for c in candidates], dtype=float)
    summed = raw[indices].sum(axis=1)
    bkeys = list(bonuses)
    final = beam._final_stats(summed, protection,
                             np.array([keys.index(key) for key in bkeys], dtype=np.int32),
                             np.array([bonuses[key] for key in bkeys]),
                             keys.index('tear_dmg_factor') if 'tear_dmg_factor' in keys else -1,
                             keys.index('stopping_protection') if 'stopping_protection' in keys else -1,
                             np)
    return keys, final


def search_portfolio(candidates, k, budget, max_counter, prot_mult,
                     cont_bonus_dict, stat_max, priorities, min_thresholds, n_keep,
                     allow_duplicates=False, min_budget=False, require_carry=True,
                     max_thresholds=None, objective=None, optimization_mode='balanced',
                     priority_order=None, objective_tolerance=1e-6, *, max_attempts=3,
                     attempts=None):
    """Union valid strategy results, ranked by the complete shared objective.

    Returns up to max_attempts * n_keep distinct candidate combinations so the
    ordinary builder can perform its final validation and diversity selection.
    Failures propagate; CUDA OOM recovery remains inside each Beam attempt.
    """
    _check_limit(max_attempts)
    if k < 1 or len(candidates) < k:
        return np.zeros(0), np.zeros((0, max(k, 1)), dtype=np.int32)
    objective = objective or Objective.create(
        priorities, stat_max, 'min_budget' if min_budget else optimization_mode,
        priority_order, objective_tolerance)
    reports = attempts if attempts is not None else []
    pool_id = len({r['pool'] for r in reports})
    union = {}
    for strategy in STRATEGIES[:max_attempts]:
        start = time.perf_counter()
        scores, combos = beam.search(
            candidates, k, budget, max_counter, prot_mult, cont_bonus_dict,
            stat_max, priorities, min_thresholds, n_keep,
            allow_duplicates=allow_duplicates, min_budget=min_budget, require_carry=require_carry,
            beam_width=strategy.width, max_thresholds=max_thresholds, objective=objective,
            optimization_mode=optimization_mode, priority_order=priority_order,
            objective_tolerance=objective_tolerance, danger_lookahead=strategy.lookahead)
        elapsed = time.perf_counter() - start
        accepted = []
        if len(combos):
            if combos.shape != (len(scores), k):
                raise ValueError('Invalid Beam result dimensions')
            if not np.issubdtype(combos.dtype, np.integer) or np.any(combos < 0) or np.any(combos >= len(candidates)):
                raise ValueError('Invalid Beam candidate indices')
            keys, final = _completed_stats(candidates, combos, prot_mult, cont_bonus_dict,
                                           objective, min_thresholds, max_thresholds or {})
            for indices, stats, score in zip(combos, final, scores):
                combo = [candidates[int(index)] for index in indices]
                price = sum(c.buyout_price for c in combo)
                if (not np.isfinite(score) or not np.isfinite(stats).all() or price > budget
                        or sum(beam._is_counter_art(c) for c in combo) > max_counter
                        or not is_available(combo, allow_duplicates)):
                    continue
                values = dict(zip(keys, stats))
                if (any(values.get(key, 0) < value for key, value in min_thresholds.items())
                        or any(values.get(key, 0) > value for key, value in (max_thresholds or {}).items())
                        or (require_carry and np.any(stats[:5] > beam._DANGER_THRESH))):
                    continue
                identity = tuple(sorted(int(index) for index in indices))
                ranking = tuple(-v for v in objective.rank(values, price)) + (price,)
                entry = (ranking, objective.evaluate(values, price), identity)
                union[identity] = entry
                accepted.append(entry)
        reports.append(dict(pool=pool_id, strategy=strategy.name, beam_width=strategy.width,
                            lookahead=strategy.lookahead, seconds=elapsed, returned=len(scores),
                            valid=len(accepted), retained_union=len(union),
                            objective=objective.metadata(),
                            best_rank=min((r[0] for r in accepted), default=None)))
    ordered = sorted(union.values())
    return (np.array([entry[1] for entry in ordered], dtype=float),
            np.array([entry[2] for entry in ordered], dtype=np.int32).reshape((-1, k)))


@dataclass
class PortfolioResult:
    builds: list
    attempts: list
    max_attempts_per_pool: int
    elapsed_seconds: float


def build_portfolio_sets(budget, *, max_attempts=3, **request):
    """Explicit opt-in: run up to three strategies per container/armor pool.

    Normal build_artifact_sets() keeps its existing backend/width/Lookahead
    behavior. Small pools here deliberately use Beam for oracle comparisons.
    Each invocation has its own reports; concurrent calls do not share state.
    """
    from .builder import build_artifact_sets
    _check_limit(max_attempts)
    if '_beam_search_fn' in request:
        raise ValueError('A custom engine is not a portfolio request parameter')
    reports = []
    def search(*args, **kwargs):
        kwargs.pop('beam_width', None)
        # Portfolio strategies own their lookahead configuration.
        kwargs.pop('constraint_lookahead', None)
        return search_portfolio(*args, max_attempts=max_attempts, attempts=reports, **kwargs)
    start = time.perf_counter()
    builds = build_artifact_sets(budget, _beam_search_fn=search, **request)
    return PortfolioResult(builds, reports, max_attempts, time.perf_counter() - start)

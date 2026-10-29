"""Exact one-slot continuation in the prepared Beam model, explicitly opt-in.

Same raw stats, price and counter flag define a profile. Retain the last k
distinct conflict groups of every profile. A prefix has at most k-1 groups:
if a compatible future copy exists, at least one retained copy is compatible.
The cap skips this refinement rather than truncating candidates unsafely.
"""
import numpy as np


def suffix_representatives(raw, prices, counters, groups, slots, limit=256):
    """Price/suffix dominance for ONE remaining slot; never prune the pool.

    A later, no dearer candidate with identical raw stats/counter can replace
    this candidate if its group is the same, or if k distinct such groups
    exist: a prefix occupies at most k-1 groups. Maintain the k cheapest
    distinct groups seen in the suffix, referring only to retained candidates.
    Earlier candidates cannot make an already retained suffix unnecessary.
    If the cap is exceeded, skip refinement entirely, never truncate.
    """
    for name, value in [('slots', slots), ('joint_tail_limit', limit)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    profiles, retained = {}, []
    for index in range(len(raw) - 1, -1, -1):
        key = (np.asarray(raw[index], dtype=np.float64).tobytes(), float(counters[index]))
        future = profiles.setdefault(key, {})
        price, group = float(prices[index]), int(groups[index])
        if ((group in future and future[group] <= price)
                or sum(value <= price for value in future.values()) >= slots):
            continue
        retained.append(index)
        if len(retained) > limit:
            return None
        future[group] = min(price, future.get(group, price))
        if len(future) > slots:
            del future[max(future, key=future.get)]
    return np.array(retained[::-1], dtype=np.int32)


def tail_representatives(raw, prices, counters, groups, slots, limit=256):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('joint_tail_limit must be a positive integer')
    profiles = {}
    for index, (stats, price, counter, group) in enumerate(zip(raw, prices, counters, groups)):
        key = (np.asarray(stats, dtype=np.float64).tobytes(), float(price), float(counter))
        profiles.setdefault(key, {})[int(group)] = index
    retained = []
    for copies in profiles.values():
        retained.extend(sorted(copies.values(), reverse=True)[:slots])
        if len(retained) > limit:
            return None
    return np.array(sorted(retained), dtype=np.int32)


def rank_last_slot(stats, costs, last, counters, indices, raw, prices, candidate_counters,
                   groups, representatives, budget, max_counter, complete_score, xp,
                   max_pairs=262144):
    """Use price and all completed constraints from the same physical candidate.

    Returns -inf if no compatible tail exists. Does not materialize all prefix
    x candidate stat matrices simultaneously. The engine's completed game-stat
    transform and constraints are supplied by complete_score.
    """
    result = xp.full(len(costs), -xp.inf)
    if not len(representatives) or not len(costs):
        return result
    step = max(1, max_pairs // len(representatives))
    tail_stats = raw[representatives]
    tail_prices = prices[representatives]
    tail_counters = candidate_counters[representatives]
    tail_groups = groups[representatives]
    for start in range(0, len(costs), step):
        end = min(start + step, len(costs))
        total = costs[start:end, None] + tail_prices[None, :]
        valid = representatives[None, :] > last[start:end, None]
        valid &= total <= budget
        valid &= counters[start:end, None] + tail_counters[None, :] <= max_counter
        for slot in range(indices.shape[1]):
            valid &= groups[indices[start:end, slot, None]] != tail_groups[None, :]
        combined = stats[start:end, None, :] + tail_stats[None, :, :]
        scores = complete_score(combined.reshape((-1, stats.shape[1])), total.reshape(-1))
        scores = scores.reshape(total.shape)
        result[start:end] = xp.max(xp.where(valid, scores, -xp.inf), axis=1)
    return result


def build_joint_budget_sets(budget, *, beam_width=2000, joint_tail_limit=256,
                            joint_tail_frontier=False, **request):
    """Experimental min_budget: one search per pool, no automatic wider retry."""
    from .builder import build_artifact_sets
    from . import _beam_search as beam
    if '_beam_search_fn' in request or request.get('optimization_mode', 'min_budget') != 'min_budget':
        raise ValueError('Joint budget search only supports min_budget requests')
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError('beam_width must be a positive integer')
    if isinstance(joint_tail_limit, bool) or not isinstance(joint_tail_limit, int) or joint_tail_limit < 1:
        raise ValueError('joint_tail_limit must be a positive integer')
    if not isinstance(joint_tail_frontier, bool):
        raise ValueError('joint_tail_frontier must be a boolean')
    request = dict(request, min_budget=True, optimization_mode='min_budget')
    def search(*args, **kwargs):
        kwargs.update(beam_width=beam_width, danger_lookahead=False,
                      joint_budget_tail=True, joint_tail_limit=joint_tail_limit,
                      joint_tail_frontier=joint_tail_frontier)
        return beam.search(*args, **kwargs)
    return build_artifact_sets(budget, _beam_search_fn=search, **request)

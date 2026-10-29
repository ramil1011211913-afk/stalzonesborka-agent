"""Opt-in exact last-slot search over all candidates, with bounded pair batches.

No dominance deletion is used: a cheaper copy may belong to an occupied group
or precede the prefix's last index. Price ordering only permits stopping after
a prefix has a feasible continuation. This is not an exact full-build solver.
"""
import numpy as np


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def rank_last_slot(stats, costs, last, counters, indices, raw, prices,
                   candidate_counters, groups, price_order, budget, max_counter,
                   complete_score, xp, *, block_size=32, max_pairs=32768):
    """Find the cheapest feasible single tail without mixing candidates.

    complete_score must return -total_price for valid rows, -inf otherwise.
    Only valid physical/budget/counter pairs reach the native final scorer;
    it applies all game transforms, armor bonuses, min/max and carry checks.
    max_pairs bounds each temporary pair batch, not total process memory or
    running time. In the worst case every prefix/candidate pair is examined.
    """
    positive_integer(block_size, 'joint_scan_block')
    positive_integer(max_pairs, 'joint_scan_max_pairs')
    result = xp.full(len(costs), -xp.inf)
    width = min(block_size, max_pairs)
    rows = max(1, max_pairs // width)
    for start in range(0, len(costs), rows):
        active = xp.arange(start, min(start + rows, len(costs)))
        for offset in range(0, len(price_order), width):
            if not len(active):
                break
            tail = price_order[offset:offset + width]
            # Later candidates cannot fit these rows either (sorted prices).
            active = active[costs[active] + prices[tail[0]] <= budget]
            if not len(active):
                break
            total = costs[active, None] + prices[tail][None, :]
            valid = tail[None, :] > last[active, None]
            valid &= total <= budget
            valid &= counters[active, None] + candidate_counters[tail][None, :] <= max_counter
            for slot in range(indices.shape[1]):
                valid &= groups[indices[active, slot, None]] != groups[tail][None, :]
            row, column = xp.nonzero(valid)
            scores = xp.full(total.shape, -xp.inf)
            if len(row):
                combined = stats[active[row]] + raw[tail[column]]
                scores[row, column] = complete_score(combined, total[row, column])
            best = xp.max(scores, axis=1)
            settled = xp.isfinite(best)
            result[active[settled]] = best[settled]
            active = active[~settled]
    return result


def build_stream_budget_sets(budget, *, beam_width=2000, joint_scan_block=32,
                            joint_scan_max_pairs=32768, **request):
    """Separate experiment; ordinary builder and other experiments stay intact."""
    from .builder import build_artifact_sets
    from . import _beam_search as beam
    if '_beam_search_fn' in request or request.get('optimization_mode', 'min_budget') != 'min_budget':
        raise ValueError('Stream budget search only supports min_budget requests')
    for name, value in [('beam_width', beam_width), ('joint_scan_block', joint_scan_block),
                        ('joint_scan_max_pairs', joint_scan_max_pairs)]:
        positive_integer(value, name)
    request = dict(request, min_budget=True, optimization_mode='min_budget')
    def search(*args, **kwargs):
        kwargs.update(beam_width=beam_width, danger_lookahead=False,
                      joint_budget_scan=True, joint_scan_block=joint_scan_block,
                      joint_scan_max_pairs=joint_scan_max_pairs)
        return beam.search(*args, **kwargs)
    return build_artifact_sets(budget, _beam_search_fn=search, **request)

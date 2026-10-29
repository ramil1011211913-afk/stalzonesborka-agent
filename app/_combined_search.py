"""One explicit experimental attempt per prepared pool; no default activation.

Balanced: chunked scoring and constraint bounds. Lexicographic: chunking only.
Min-budget: native blocked expansion and exact one-slot suffix frontier.
The min-budget engine already blocks expansions independently of score chunks.
"""
from . import _beam_search as beam
from ._objective import Objective
from ._min_budget_stream import positive_integer


def search_combined(candidates, k, budget, max_counter, prot_mult,
                    cont_bonus_dict, stat_max, priorities, min_thresholds, n_keep,
                    allow_duplicates=False, min_budget=False, require_carry=True,
                    beam_width=2000, max_thresholds=None, objective=None,
                    optimization_mode='balanced', priority_order=None, objective_tolerance=1e-6,
                    *, expansion_chunk_size=262144, joint_tail_limit=256):
    for name, value in [('beam_width', beam_width), ('expansion_chunk_size', expansion_chunk_size),
                        ('joint_tail_limit', joint_tail_limit)]:
        positive_integer(value, name)
    objective = objective or Objective.create(
        priorities, stat_max, 'min_budget' if min_budget else optimization_mode,
        priority_order, objective_tolerance)
    mode = objective.mode
    return beam.search(
        candidates, k, budget, max_counter, prot_mult, cont_bonus_dict, stat_max,
        priorities, min_thresholds, n_keep, allow_duplicates=allow_duplicates,
        min_budget=mode == 'min_budget', require_carry=require_carry, beam_width=beam_width,
        max_thresholds=max_thresholds, objective=objective,
        optimization_mode=mode, priority_order=priority_order, objective_tolerance=objective_tolerance,
        danger_lookahead=mode != 'min_budget', constraint_lookahead=mode == 'balanced',
        expansion_chunk_size=expansion_chunk_size, joint_budget_tail=mode == 'min_budget',
        joint_tail_frontier=mode == 'min_budget', joint_tail_limit=joint_tail_limit,
        joint_budget_scan=False)


def build_combined_sets(budget, *, beam_width=2000, expansion_chunk_size=262144,
                        joint_tail_limit=256, **request):
    from .builder import build_artifact_sets
    if '_beam_search_fn' in request:
        raise ValueError('Custom engine is not a combined request parameter')
    for name, value in [('beam_width', beam_width), ('expansion_chunk_size', expansion_chunk_size),
                        ('joint_tail_limit', joint_tail_limit)]:
        positive_integer(value, name)
    def search(*args, **kwargs):
        kwargs.update(beam_width=beam_width, expansion_chunk_size=expansion_chunk_size,
                      joint_tail_limit=joint_tail_limit)
        return search_combined(*args, **kwargs)
    return build_artifact_sets(budget, _beam_search_fn=search, **request)

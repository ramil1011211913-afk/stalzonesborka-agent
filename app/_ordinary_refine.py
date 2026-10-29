"""Bounded linear-model refinement on the existing ordinary search pools.

The verified Beam incumbent always survives failure, timeout or weaker results.
No candidate-source changes, nonlinear model, robust transform or optimality
claim is introduced into ordinary search.
"""
import time

MAX_SECONDS = 3.0
MIN_CANDIDATES = 101


def refine_incumbent(profiles, incumbent, budget, minimum, maximum, carry,
                     duplicates, priorities, objective, region, request):
    from . import builder as b
    from . import _max_quality_model as limits, _max_quality as quality
    from ._quality_scenario import current_views
    from ._survivability import requested
    if (objective.mode not in ('balanced', 'lexicographic')
            or requested(priorities, minimum, maximum)
            or request.get('conservative') or request.get('guaranteed_carryable')):
        return None, []
    deadline = time.monotonic() + MAX_SECONDS
    best = incumbent
    reports = []
    for container, prepared in profiles:
        if time.monotonic() >= deadline - .1:
            break
        if not MIN_CANDIDATES <= len(prepared) <= limits.MAX_CANDIDATES:
            continue
        # Synthetic/anonymous providers have no verifiable market contract.
        if any(not o.is_owned and o.auction_lot_id is None for o in prepared):
            continue
        report = dict(stage='ordinary_linear_refinement', state='UNKNOWN', container_id=container.item_id,
                      armor_id=(getattr(container, '_armor', None) or {}).get('id'))
        started = time.monotonic()
        try:
            current = current_views(prepared, b.get_connection, region, b._normalize_bonus_id)
            candidates = [o for o in current if limits.descriptor_supported(o)]
            if len(candidates) < container.slots:
                continue
            model = limits.make_model(candidates, container, objective, budget,
                                      minimum or {}, maximum or {}, duplicates, carry)
            model = dict(model, model_seconds=MAX_SECONDS)
            with quality.quality_slot() as acquired:
                if acquired is not True:
                    report.update(reason='quality_worker_busy' if acquired is False else 'quality_worker_unavailable',
                                  seconds=time.monotonic() - started, accepted=0)
                    reports.append(report)
                    break
                result = quality.run_model(model, deadline - .1)
            report.update(reason=result.get('reason'), candidates=len(candidates),
                          model_optimal=result.get('optimal', False), accepted=0)
            for stage in result.get('stages', []):
                if time.monotonic() >= deadline:
                    break
                indices = stage.get('indices')
                if (not isinstance(indices, list) or len(indices) != container.slots
                        or any(type(i) is not int or not 0 <= i < len(candidates) for i in indices)
                        or len(set(indices)) != len(indices)):
                    continue
                combo = [candidates[i] for i in indices]
                if len(current_views(combo, b.get_connection, region, b._normalize_bonus_id)) != len(combo):
                    continue
                checked = []
                b._process_combo(combo, container, sum(o.buyout_price for o in combo),
                                 priorities, minimum, checked, budget, objective=objective,
                                 max_thresholds=maximum, require_carry=carry,
                                 allow_duplicates=duplicates)
                if not checked or not b._final_valid(checked[0], budget, minimum, maximum,
                                                     carry, False, duplicates):
                    continue
                candidate = checked[0]
                if objective.sort_key(candidate) < objective.sort_key(best):
                    candidate = b._decorate_armor(candidate)
                    candidate._search_request = request
                    best = candidate
                    report['accepted'] += 1
        except limits.Unsupported as error:
            report.update(reason=str(error), state='UNSUPPORTED')
        except Exception as error:
            # A secondary model cannot invalidate the already verified answer.
            report.update(reason=type(error).__name__, state='UNKNOWN')
        report['seconds'] = time.monotonic() - started
        reports.append(report)
    return (best if best is not incumbent else None), reports

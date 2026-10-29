"""Incumbent-preserving second Beam ordering for ordinary lexicographic queries.

Uses the existing optimistic danger relaxation only to rank incomplete states.
The finite Beam remains heuristic. Completed candidates are independently
recomputed by the native builder and rechecked against original market records.
"""
import os
import time


def refine_incumbent(profiles, incumbent, budget, minimum, maximum, carry,
                     duplicates, priorities, objective, region, request):
    from . import builder as b
    from . import _beam_search as beam
    from ._quality_scenario import current_views
    if os.environ.get('STALZONE_DANGER_LOOKAHEAD', '0') == '1':
        return None  # Primary search already used this ranking.
    best = incumbent
    width = min(10000, int(os.environ.get('STALZONE_BEAM_WIDTH', '10000')))
    deadline = time.monotonic() + 120.0  # Bounded secondary work, not a new user mode.
    for container, prepared in profiles:
        if time.monotonic() >= deadline:
            break
        if len(prepared) <= 100 or not any(o.auction_lot_id is not None for o in prepared):
            continue
        # Historical rows and expired/changed offers cannot crowd the second
        # beam. Preserve original candidate order and physical conflict groups.
        candidates = current_views(prepared, b.get_connection, region, b._normalize_bonus_id)
        if len(candidates) < container.slots:
            continue
        bonuses = {key: val.get('value', 0) if isinstance(val, dict) else val
                   for key, val in container.bonus_stats.items()}
        scores, combos = beam.search(candidates, container.slots, budget, container.slots,
            1 - container.inner_protection / 100, bonuses, dict(zip(objective.keys, objective.scales)),
            priorities, minimum or {}, 2000, allow_duplicates=duplicates, require_carry=carry,
            beam_width=width, max_thresholds=maximum, objective=objective,
            danger_lookahead=True, expansion_chunk_size=262144)
        for indices in combos:
            if time.monotonic() >= deadline:
                break
            chosen = [candidates[int(i)] for i in indices]
            checked = []
            b._process_combo(chosen, container, sum(o.buyout_price for o in chosen), priorities,
                minimum, checked, budget, max_thresholds=maximum, objective=objective,
                require_carry=carry, allow_duplicates=duplicates)
            if not checked or not b._final_valid(checked[0], budget, minimum, maximum, carry, False, duplicates):
                continue
            candidate = checked[0]
            if objective.sort_key(candidate) < objective.sort_key(best):
                # A lot may expire or change while the beam is running.
                if len(current_views(chosen, b.get_connection, region, b._normalize_bonus_id)) != len(chosen):
                    continue
                best = b._decorate_armor(candidate)
                best._search_request = request
    return best if best is not incumbent else None

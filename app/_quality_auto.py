"""Explicit auto-container shortlist for Max Quality.

Auto mode must never imply that every container was proved.  This module ranks
the available container profiles with a cheap optimistic upper bound, keeps a
bounded number of them, and returns enough metadata for the caller to state
exactly which containers were omitted.
"""
import math

from ._max_quality_model import DANGERS


def _bonus_value(container, key):
    value = container.bonus_stats.get(key)
    if isinstance(value, dict):
        value = value.get('value', 0)
    try:
        value = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _rank_containers(containers, candidates, priorities, minimum):
    keys = list(dict.fromkeys([*priorities, *(minimum or {})]))
    # Values are stored at effectiveness=100.  A positive effectiveness
    # multiplier preserves their order, so one sorted list per key is enough.
    top_values = {}
    for key in keys:
        values = [_finite(option.stats.get(key, 0.0)) for option in candidates]
        values.sort(reverse=True)
        top_values[key] = values

    ranked = []
    for container in containers:
        slots = max(0, int(container.slots))
        effectiveness = max(0.0, _finite(container.effectiveness) / 100.0)
        upper = {}
        for key in keys:
            values = top_values[key][:slots]
            # Accumulation stats are not scaled by container effectiveness.
            scale = 1.0 if key in DANGERS else effectiveness
            upper[key] = sum(values) * scale + _bonus_value(container, key)
        score = sum(max(0.0, _finite(weight)) * upper.get(key, 0.0)
                    for key, weight in priorities.items())
        constraints_ok = all(
            upper.get(key, 0.0) + 1e-9 >= _finite(target)
            for key, target in (minimum or {}).items()
            if _finite(target) > 0
        )
        ranked.append((container, score, constraints_ok))

    # Feasible-by-upper-bound containers first; then objective potential.
    # The final fields make ties deterministic across database orderings.
    ranked.sort(key=lambda item: (
        not item[2], -item[1], -int(item[0].slots),
        -_finite(item[0].effectiveness), item[0].item_id))
    return ranked


def select_auto_containers(containers, candidates, priorities, *,
                           minimum=None, maximum=None, requested_limit=8,
                           armor_profiles=1, candidate_limit=None,
                           profile_limit=None):
    """Return at most ``requested_limit`` containers and explicit scope data.

    The ranking is an upper bound, not a proof: it is used only to choose the
    shortlist.  The Max Quality engine still solves and validates every chosen
    profile independently.  Omitted containers are reported in the metadata.
    """
    containers = list(containers or ())
    candidates = list(candidates or ())
    priorities = {key: weight for key, weight in (priorities or {}).items()
                  if _finite(weight) > 0}
    requested_limit = max(1, int(requested_limit or 1))
    armor_profiles = max(1, int(armor_profiles or 1))
    candidate_limit = None if candidate_limit is None else max(0, int(candidate_limit))
    profile_limit = max(1, int(profile_limit or requested_limit * armor_profiles))

    effective_limit = requested_limit
    candidate_reduced = False
    if candidate_limit is not None and candidates:
        fit = candidate_limit // len(candidates)
        # Keep one profile so the existing total-candidate-limit diagnostic
        # can still report an explicit UNKNOWN instead of an empty shortlist.
        fit = max(1, fit)
        if fit < effective_limit:
            effective_limit = fit
            candidate_reduced = True
    effective_limit = min(effective_limit, len(containers))

    ranked = _rank_containers(containers, candidates, priorities, minimum)
    selected = []
    selected_ids = set()
    seen_slots = set()

    # One representative per slot count first.  This prevents a shortlist made
    # only of large containers from hiding a small-container optimum (for
    # example under a tight budget or a one-artifact constraint).
    for container, _, _ in ranked:
        if len(selected) >= effective_limit:
            break
        if container.slots in seen_slots:
            continue
        selected.append(container)
        selected_ids.add(container.item_id)
        seen_slots.add(container.slots)
    for container, _, _ in ranked:
        if len(selected) >= effective_limit:
            break
        if container.item_id in selected_ids:
            continue
        selected.append(container)
        selected_ids.add(container.item_id)

    # Execute the most promising profile first.  A deadline may cut the tail,
    # but the likely winner has already been solved and validated.
    order = {id(item[0]): index for index, item in enumerate(ranked)}
    selected.sort(key=lambda container: order[id(container)])

    selected_ids = [container.item_id for container in selected]
    omitted = [container.item_id for container, _, _ in ranked
               if container.item_id not in set(selected_ids)]
    scores = {container.item_id: score for container, score, _ in ranked}
    feasible = {container.item_id: ok for container, _, ok in ranked}
    ranking = [
        dict(container_id=container.item_id, slots=container.slots,
             effectiveness=container.effectiveness, score=round(score, 6),
             constraint_upper_bound_ok=ok)
        for container, score, ok in ranked[:12]
    ]
    metadata = dict(
        mode='auto',
        method='optimistic_upper_bound_v1',
        scope='auto_shortlist',
        certificate_scope='selected_profiles_only',
        considered_count=len(containers),
        selected_count=len(selected),
        omitted_count=len(omitted),
        selected=selected_ids,
        omitted=omitted,
        armor_profiles=armor_profiles,
        selected_profile_count=len(selected) * armor_profiles,
        profile_limit=profile_limit,
        container_limit=effective_limit,
        requested_container_limit=requested_limit,
        candidate_pool=len(candidates),
        candidate_limit=candidate_limit,
        candidate_reduced=candidate_reduced,
        ranking=ranking,
        scores={key: round(value, 6) for key, value in scores.items()},
        constraint_upper_bound_ok=feasible,
    )
    return selected, metadata

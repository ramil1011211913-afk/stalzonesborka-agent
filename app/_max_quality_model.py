"""Linear capability contract and bounded physical-candidate MILP construction."""
import hashlib
import json
import math

MAX_CANDIDATES = 5000
MAX_INPUT_CANDIDATES = 10000
MAX_PROFILES = 8
MAX_TOTAL_CANDIDATES = 20000
MAX_ROWS = 256
MAX_NONZEROS = 100000
MAX_NODES = 20000
DEFAULT_SECONDS = 45.
MODEL_SECONDS = 15.
MEMORY_BYTES = 512 * 1024**2
DANGERS = ('radiation_accumulation', 'biological_accumulation', 'psycho_accumulation',
           'thermal_accumulation', 'frost_accumulation')
LIMITS = (10., 10., 10., 10., 20.)


class Unsupported(ValueError):
    pass


def descriptor_supported(option):
    from .builder import _normalize_bonus_id
    known = {_normalize_bonus_id(x.get('prop_id', '').upper()) for x in option.bonus_stats}
    selected = {_normalize_bonus_id(x.upper()) for x in option.unlocked_bonus_ids}
    return (bool(option.base_stats) and selected <= known and len(selected) == option.ptn // 5
            and 0 <= option.quality <= 5 and 0 <= option.ptn <= 15
            and math.isfinite(option.art_percent) and 0 < option.art_percent <= 190
            and (option.is_owned or option.auction_lot_id is not None))


def make_model(candidates, container, objective, budget, minimum, maximum, duplicates, carry):
    from ._survivability import requested, make_model as derived_model
    from ._derived_stats import DERIVED_STATS, augment_bonus_stats, value as stat_value
    if requested(dict(zip(objective.keys, objective.weights)), minimum, maximum):
        try:
            return derived_model(candidates, container, objective, budget, minimum, maximum, duplicates, carry)
        except (AssertionError, OverflowError, ValueError) as error:
            if isinstance(error, Unsupported):raise
            raise Unsupported('survivability_numeric_limit') from None
    from .builder import ARMOR_ADDITIVE_KEYS
    from ._physical_items import availability_groups, physical_key
    n = len(candidates)
    if n > MAX_CANDIDATES or not 1 <= container.slots <= 7 or not 0 <= budget <= 2**53:
        raise Unsupported('model_limit')
    protection = 1 - container.inner_protection / 100
    if not math.isfinite(protection) or not 0 <= protection <= 1:
        raise Unsupported('unsupported_protection')
    keys = set(objective.keys) | set(minimum) | set(maximum)
    if len(objective.keys) > 8 or len(keys) > 128:
        raise Unsupported('priority_or_constraint_limit')
    if objective.mode == 'lexicographic' and not 1e-8 <= objective.tolerance <= 1:
        raise Unsupported('unsupported_numeric_tolerance')
    known = (set(ARMOR_ADDITIVE_KEYS) | set(DERIVED_STATS)
             | {k for o in candidates for k in o.stats})
    if any(k not in known or k == 'stopping_protection' or k.startswith('__') for k in keys):
        raise Unsupported('unsupported_or_derived_stat')
    if protection != 1 and any(k in DANGERS[:4] for k in objective.keys):
        raise Unsupported('nonlinear_protected_accumulation_objective')
    bonuses = {k: v.get('value', 0) if isinstance(v, dict) else v
               for k, v in container.bonus_stats.items()}
    augment_bonus_stats(bonuses)
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in bonuses.values()):
        raise Unsupported('nonfinite_bonus')
    prices = [o.buyout_price for o in candidates]
    if any(not isinstance(p, int) or not 0 <= p <= 2**53 for p in prices):
        raise Unsupported('unsupported_price')
    rows = []
    nonzeros = 0

    def row(values, lo=None, hi=None):
        nonlocal nonzeros
        if any(not math.isfinite(v) or abs(v) > 1e14 for v in values):
            raise Unsupported('numeric_model_limit')
        nonzeros += sum(v != 0 for v in values)
        reserve = n * (1 if objective.mode == 'balanced' else len(objective.keys))
        if len(rows) >= MAX_ROWS - len(objective.keys) - 2 or nonzeros + reserve > MAX_NONZEROS:
            raise Unsupported('matrix_limit')
        rows.append(dict(values=values, lo=lo, hi=hi))

    def bound(key, target, lower):
        target -= bonuses.get(key, 0)
        if key in DANGERS[:4]:
            if protection == 0 and target >= 0:
                if lower:
                    if target > 0:
                        row([0.] * n, 1., None)
                    else:
                        row([stat_value(o.stats, key) for o in candidates], 0., None)
                return
            if target > 0:
                target /= protection
        row([stat_value(o.stats, key) for o in candidates], target if lower else None,
            None if lower else target)

    row([1.] * n, container.slots, container.slots)
    row(prices, None, budget)
    if carry:
        for key, limit in zip(DANGERS, LIMITS):
            bound(key, limit, False)
    for key, limit in minimum.items():
        bound(key, limit, True)
    for key, limit in maximum.items():
        bound(key, limit, False)
    for group in availability_groups(candidates, duplicates):
        values = [0.] * n
        for i in group:
            values[i] = 1.
        row(values, None, 1.)
    if objective.mode == 'balanced':
        goals = [[sum(stat_value(o.stats, k) * c for k, c in zip(objective.keys, objective.coefficients))
                  for o in candidates]]
        constants = [sum(bonuses.get(k, 0.) * c for k, c in zip(objective.keys, objective.coefficients))]
    else:
        from ._objective import direction
        goals = [[direction(k) * stat_value(o.stats, k) for o in candidates] for k in objective.order]
        constants = [direction(k) * bonuses.get(k, 0.) for k in objective.order]
    if any(not math.isfinite(v) or abs(v) > 1e8 for values in goals for v in values):
        raise Unsupported('nonfinite_objective')
    if any(not math.isfinite(v) for v in constants):
        raise Unsupported('nonfinite_objective_bonus')
    digest = hashlib.sha256(json.dumps([
        [physical_key(o), o.item_id, o.buyout_price, o.quality, o.ptn, o.art_percent,
         o.unlocked_bonus_ids, o.stats] for o in candidates
    ], sort_keys=True, allow_nan=False).encode()).hexdigest()
    return dict(n=n, rows=rows, nonzeros=nonzeros, prices=prices, goals=goals,
                constants=constants, mode=objective.mode, tolerance=objective.tolerance,
                node_limit=MAX_NODES, model_seconds=MODEL_SECONDS, memory_bytes=MEMORY_BYTES,
                pool_sha256=digest)

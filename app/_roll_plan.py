"""Exact hundredth roll planning for a fixed artifact set.

The planner changes only *unknown* physical quality percentages.  Known auction
rolls and owned artifacts keep their actual percentage.  Within one quality tier
the game formulas are linear in the percentage, so a small integer MILP chooses
target rolls in hundredths of a percent (for example ``125.23``).

This is a fixed-set planner.  It does not choose a different set of artifacts,
and it deliberately rejects nonlinear derived objectives (EHP/HPS) instead of
approximating them.
"""
from dataclasses import dataclass, replace
import math

MAX_VARIABLES = 7
MAX_SECONDS = 2.0
FEAS_TOL = 1e-6
LINEAR_TOL = 1e-7
PROTECTED_DANGERS = (
    "radiation_accumulation",
    "biological_accumulation",
    "psycho_accumulation",
    "thermal_accumulation",
)


class Unsupported(ValueError):
    """The fixed set cannot be represented by the linear hundredth model."""


@dataclass(frozen=True)
class Choice:
    index: int
    item_id: str
    name_ru: str
    auction_lot_id: int | None
    owned_instance_id: str | None
    baseline_percent: float
    percent: float
    minimum: float
    maximum: float
    known: bool
    variable: bool
    changed: bool


@dataclass
class Plan:
    feasible: bool
    status: str
    reason: str
    choices: tuple
    percents: tuple
    stats: dict
    objective_value: float
    carryable: bool
    warnings: list
    baseline_stats: dict
    baseline_objective: float
    solver_status: int | None = None
    message: str = ""


def _source(option):
    from ._quality_scenario import source
    return source(option)


def _short_danger_keys():
    from .builder import DANGER_THRESHOLDS
    return tuple(key.rsplit(".", 1)[-1] for key in DANGER_THRESHOLDS)


def _quality_range(quality):
    from .builder import QUALITY_RANGES
    return QUALITY_RANGES.get(quality)


def _stats_at(option, percent, container):
    from .builder import compute_artifact_stats
    qr = _quality_range(option.quality)
    if qr is None:
        raise Unsupported("unsupported_quality_tier")
    stats, raw, names, _ = compute_artifact_stats(
        option.base_stats,
        option.bonus_stats,
        option.unlocked_bonus_ids,
        float(percent),
        option.ptn,
        qr["min"],
        qr["range"],
        option.stats_random,
        container.inner_protection if container else 0.0,
        container.effectiveness if container else 100.0,
    )
    return stats, raw, names


def _relevant_keys(objective, minimum, maximum, carry):
    keys = set(objective.keys) | set(minimum or {}) | set(maximum or {})
    if carry:
        keys.update(_short_danger_keys())
    return keys


def _is_linear(option, lo, hi, keys, container):
    """Check the relevant stat columns on a few interior points."""
    points = (lo, lo + (hi - lo) * 0.25, lo + (hi - lo) * 0.5,
              lo + (hi - lo) * 0.75, hi)
    rows = [_stats_at(option, point, container)[0] for point in points]
    for key in keys:
        values = [row.get(key, 0.0) for row in rows]
        for fraction, value in zip((0.25, 0.5, 0.75), values[1:-1]):
            expected = values[0] + (values[-1] - values[0]) * fraction
            scale = max(1.0, abs(values[0]), abs(values[-1]), abs(value))
            if abs(value - expected) > LINEAR_TOL * scale:
                return False
    return True


def _bounds(option, container, keys):
    """Return (fixed, low, high, fixed_percent, excluded_lower_point)."""
    physical = _source(option)
    qr = _quality_range(physical.quality)
    if qr is None:
        raise Unsupported("unsupported_quality_tier")
    minimum = float(qr["min"])
    maximum = float(qr["min"] + qr["range"])
    known = bool(getattr(physical, "art_percent_known", True))
    owned = bool(getattr(physical, "is_owned", False))
    if known or owned:
        # A known roll is a fact.  An owned unknown roll cannot be changed by
        # buying/rolling another copy, so keep the request's scenario point.
        percent = float(physical.art_percent if known else option.art_percent)
        return True, minimum, maximum, percent, None

    lower_point = None
    lo, hi = minimum, maximum
    if not _is_linear(physical, lo, hi, keys, container):
        # Uncommon negative stats have a deliberate discontinuity at exactly
        # 100%.  For every supported objective the continuous branch (100.01%)
        # dominates the 100.00% endpoint: it has at least as much of every
        # positive stat and at most as much danger.  Excluding the endpoint is
        # therefore conservative, and the hundredth model stays exact.
        shifted = min(hi, lo + 0.01)
        if shifted <= lo or not _is_linear(physical, shifted, hi, keys, container):
            raise Unsupported("nonlinear_percent_interval")
        lower_point = lo
        lo = shifted
    return False, lo, hi, None, lower_point


def _bonus_value(container, key):
    if container is None:
        return 0.0
    entry = (container.bonus_stats or {}).get(key)
    if isinstance(entry, dict):
        return float(entry.get("value", 0.0) or 0.0)
    return float(entry or 0.0)


def _protection(container, key):
    if container is None or key not in PROTECTED_DANGERS:
        return 1.0
    value = 1.0 - float(container.inner_protection or 0.0) / 100.0
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise Unsupported("unsupported_protection")
    return value


def _constraint_bounds(key, value, is_min, protection, bonus):
    """Linear raw-artifact bounds equivalent to the protected total bound.

    Protection applies only to positive protected accumulations.  The returned
    bounds are for the *unprotected* artifact sum; the container bonus is added
    after protection.
    """
    target = float(value) - bonus
    if key in PROTECTED_DANGERS and protection != 1.0:
        if protection == 0.0:
            if is_min:
                if target > FEAS_TOL:
                    return None  # impossible: protected value is never positive
                return (target, math.inf)
            if target < -FEAS_TOL:
                return (-math.inf, target)
            return (None, None)  # every raw value satisfies this upper bound
        if is_min:
            if target <= 0:
                return (target, math.inf)
            return (target / protection, math.inf)
        if target >= 0:
            return (-math.inf, target / protection)
        return (-math.inf, target)
    return (target, math.inf) if is_min else (-math.inf, target)


def _model(artifacts, container, objective, minimum, maximum, carry):
    from ._survivability import requested
    from .builder import DANGER_THRESHOLDS
    if objective.mode not in ("balanced", "lexicographic"):
        raise Unsupported("unsupported_objective_mode")
    if requested(dict(zip(objective.keys, objective.weights)), minimum, maximum):
        raise Unsupported("survivability_not_linear")
    keys = _relevant_keys(objective, minimum, maximum, carry)
    if "stopping_protection" in keys:
        raise Unsupported("stopping_protection_not_linear")
    for key in objective.keys:
        if key in PROTECTED_DANGERS and _protection(container, key) != 1.0:
            raise Unsupported("nonlinear_protected_accumulation_objective")

    records = []
    variables = []
    for index, option in enumerate(artifacts):
        fixed, lo, hi, percent, lower_point = _bounds(option, container, keys)
        stats_lo = _stats_at(option, lo if not fixed else percent, container)[0]
        stats_hi = _stats_at(option, hi if not fixed else percent, container)[0]
        record = dict(index=index, option=option, fixed=fixed, lo=lo, hi=hi,
                      percent=percent, lower_point=lower_point,
                      stats_lo=stats_lo, stats_hi=stats_hi)
        records.append(record)
        if not fixed:
            q_lo = int(math.ceil(lo * 100.0 - 1e-9))
            q_hi = int(math.floor(hi * 100.0 + 1e-9))
            if q_lo > q_hi:
                raise Unsupported("empty_hundredth_interval")
            record["q_lo"] = q_lo
            record["q_hi"] = q_hi
            variables.append(index)
    if len(variables) > MAX_VARIABLES:
        raise Unsupported("variable_limit")

    n = len(variables)
    position = {index: pos for pos, index in enumerate(variables)}
    const = {key: 0.0 for key in keys}
    coefs = {key: [0.0] * n for key in keys}
    for record in records:
        option = record["option"]
        if record["fixed"]:
            for key in keys:
                const[key] += record["stats_lo"].get(key, 0.0)
            continue
        q_lo, q_hi = record["q_lo"], record["q_hi"]
        span = q_hi - q_lo
        pos = position[record["index"]]
        for key in keys:
            v_lo = record["stats_lo"].get(key, 0.0)
            v_hi = record["stats_hi"].get(key, 0.0)
            slope = (v_hi - v_lo) / span if span else 0.0
            const[key] += v_lo - slope * q_lo
            coefs[key][pos] = slope

    bonus = {key: _bonus_value(container, key) for key in keys}
    protection = {key: _protection(container, key) for key in keys}

    rows, lower, upper = [], [], []

    def add_row(key, bounds):
        if bounds is None:
            raise Unsupported("infeasible_constraint")
        if bounds == (None, None):
            return
        lo, hi = bounds
        base = const[key]
        row = list(coefs[key])
        lo -= base
        hi -= base
        if lo > hi + FEAS_TOL:
            raise Unsupported("infeasible_fixed_constant")
        if not any(abs(value) > 0.0 for value in row):
            if lo > FEAS_TOL or hi < -FEAS_TOL:
                raise Unsupported("infeasible_fixed_constant")
            return
        rows.append(row)
        lower.append(lo)
        upper.append(hi)

    if carry:
        for key in _short_danger_keys():
            full = next((value for value in DANGER_THRESHOLDS
                         if value.rsplit(".", 1)[-1] == key), None)
            if full is None:
                continue
            add_row(key, _constraint_bounds(
                key, DANGER_THRESHOLDS[full], False,
                protection.get(key, 1.0), bonus.get(key, 0.0)))
    for key, value in (minimum or {}).items():
        add_row(key, _constraint_bounds(
            key, value, True, protection.get(key, 1.0), bonus.get(key, 0.0)))
    for key, value in (maximum or {}).items():
        add_row(key, _constraint_bounds(
            key, value, False, protection.get(key, 1.0), bonus.get(key, 0.0)))

    return dict(records=records, variables=variables, position=position, n=n,
                const=const, coefs=coefs, bonus=bonus, protection=protection,
                rows=rows, lower=lower, upper=upper, keys=keys)


def _expr(model, key):
    return model["const"][key] + model["bonus"].get(key, 0.0), model["coefs"][key]


def _solve_milp(cvec, model, time_limit):
    import numpy as np
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as error:
        raise Unsupported("solver_unavailable") from error
    n = model["n"]
    if n == 0:
        return 0, None, 0, "empty"
    matrix = np.asarray(model["rows"], dtype=float) if model["rows"] else None
    constraints = (LinearConstraint(matrix, np.asarray(model["lower"], dtype=float),
                                    np.asarray(model["upper"], dtype=float))
                   if matrix is not None else None)
    q_lo = np.asarray([model["records"][index]["q_lo"] for index in model["variables"]], dtype=float)
    q_hi = np.asarray([model["records"][index]["q_hi"] for index in model["variables"]], dtype=float)
    result = milp(
        c=-np.asarray(cvec, dtype=float),
        integrality=np.ones(n),
        bounds=Bounds(q_lo, q_hi),
        constraints=constraints,
        options=dict(time_limit=max(0.01, float(time_limit)), node_limit=10000,
                     mip_rel_gap=0.0),
    )
    status = int(getattr(result, "status", 4))
    if result.x is None:
        return status, None, 0.0, str(getattr(result, "message", ""))[:300]
    values = np.rint(np.asarray(result.x, dtype=float))
    return status, values, float(np.dot(np.asarray(cvec, dtype=float), values)), str(getattr(result, "message", ""))[:300]


def _add_expr_constraint(model, key, direction, best):
    const, coefs = _expr(model, key)
    row = [direction * value for value in coefs]
    if not any(abs(value) > 0.0 for value in row):
        if direction * const < best - FEAS_TOL:
            raise Unsupported("numeric_lexicographic_constant")
        return
    model["rows"].append(row)
    # ``best`` already includes the constant and is the value of
    # direction * (const + coefs @ q).  The row is only the variable part.
    model["lower"].append(best - FEAS_TOL - direction * const)
    model["upper"].append(math.inf)


def _percents(model, values):
    result = []
    for record in model["records"]:
        if record["fixed"]:
            result.append(float(record["percent"]))
        else:
            pos = model["position"][record["index"]]
            result.append(float(values[pos]) / 100.0)
    return tuple(result)


def _evaluate(artifacts, percents, container):
    from .builder import (_apply_build_container_protection, _recalculate_stopping_protection,
                          check_carryability)
    from ._derived_stats import add_names, augment
    combined_raw, names, origins = {}, {}, {}
    for option, percent in zip(artifacts, percents):
        stats, _, sn = _stats_at(option, percent, container)
        for key, value in stats.items():
            combined_raw[key] = combined_raw.get(key, 0.0) + value
            names.setdefault(key, sn.get(key, key))
            origins.setdefault(key, "artefact")
    combined = _apply_build_container_protection(
        combined_raw, origins, container.inner_protection if container else 0.0)
    if container is not None and container.bonus_stats:
        for key, entry in container.bonus_stats.items():
            value = entry.get("value", 0) if isinstance(entry, dict) else entry
            if value:
                combined[key] = combined.get(key, 0.0) + float(value)
                names[key] = entry.get("name_ru", key) if isinstance(entry, dict) else key
    _recalculate_stopping_protection(combined)
    augment(combined)
    add_names(combined, names)
    carryable, warnings = check_carryability(combined)
    return combined, names, carryable, warnings


def _objective_value(objective, stats, price=0):
    if objective.mode == "min_budget":
        return -float(price)
    return float(objective.evaluate(stats, price))


def solve(artifacts, container, objective, minimum=None, maximum=None,
          carry=True, time_limit=MAX_SECONDS):
    """Choose exact hundredth percentages for one fixed artifact set."""
    minimum = minimum or {}
    maximum = maximum or {}
    artifacts = list(artifacts)
    try:
        model = _model(artifacts, container, objective, minimum, maximum, carry)
    except Unsupported as error:
        baseline, _, _, _ = _evaluate(artifacts, [float(o.art_percent) for o in artifacts], container)
        return Plan(False, "unsupported", str(error), (), (),
                    {}, float("-inf"), False, [], baseline,
                    _objective_value(objective, baseline), message=str(error))

    baseline_percents = [float(o.art_percent) for o in artifacts]
    baseline_stats, _, baseline_carry, baseline_warnings = _evaluate(
        artifacts, baseline_percents, container)
    baseline_value = _objective_value(objective, baseline_stats)

    if model["n"] == 0:
        return _finish(model, objective, baseline_percents,
                       baseline_stats, baseline_carry, baseline_warnings,
                       0, "fixed", "no_unknown_percentages")

    deadline = time_limit
    if objective.mode == "balanced":
        cvec = [0.0] * model["n"]
        for key, coefficient in zip(objective.keys, objective.coefficients):
            _, key_coefs = _expr(model, key)
            for i, value in enumerate(key_coefs):
                cvec[i] += coefficient * value
        status, values, value, message = _solve_milp(cvec, model, deadline)
        if values is None:
            return Plan(False, "infeasible" if status == 2 else "unknown",
                        "solver_infeasible" if status == 2 else "solver_no_incumbent",
                        (), (), {}, float("-inf"), False, [], baseline_stats,
                        baseline_value, status, message)
        # Prefer higher rolls when the primary objective is tied.  This never
        # sacrifices the primary value: the second model fixes it first.
        best = value
        primary_coefs = [0.0] * model["n"]
        for key, coefficient in zip(objective.keys, objective.coefficients):
            _, key_coefs = _expr(model, key)
            for i, item in enumerate(key_coefs):
                primary_coefs[i] += coefficient * item
        if any(abs(item) > 0.0 for item in primary_coefs):
            row = list(primary_coefs)
            model["rows"].append(row)
            model["lower"].append(best - FEAS_TOL)
            model["upper"].append(math.inf)
            status2, values2, _, message2 = _solve_milp([-1.0] * model["n"], model, deadline)
            if values2 is not None:
                values, status, message = values2, status2, message2
        status_text = "optimal" if status == 0 else "feasible"
    else:
        values = None
        status = 4
        message = ""
        for key in objective.order:
            from ._objective import direction
            sign = direction(key)
            const, key_coefs = _expr(model, key)
            cvec = [sign * item for item in key_coefs]
            status, values, best, message = _solve_milp(cvec, model, deadline)
            if values is None:
                return Plan(False, "infeasible" if status == 2 else "unknown",
                            "solver_infeasible" if status == 2 else "solver_no_incumbent",
                            (), (), {}, float("-inf"), False, [], baseline_stats,
                            baseline_value, status, message)
            best_full = sign * const + best
            _add_expr_constraint(model, key, sign, best_full)
        status2, values2, _, message2 = _solve_milp([-1.0] * model["n"], model, deadline)
        if values2 is not None:
            values, status, message = values2, status2, message2
        status_text = "optimal" if status == 0 else "feasible"

    percents = _percents(model, values)
    stats, _, carryable, warnings = _evaluate(artifacts, percents, container)
    return _finish(model, objective, percents, stats,
                   carryable, warnings, status, status_text, "solved", message=message,
                   baseline_stats=baseline_stats, baseline_value=baseline_value)


def _finish(model, objective, percents, stats, carryable,
            warnings, status, status_text, reason, message="", baseline_stats=None,
            baseline_value=None):
    choices = []
    for record, percent in zip(model["records"], percents):
        option = record["option"]
        physical = _source(option)
        baseline = float(option.art_percent)
        if record["fixed"]:
            lower = upper = float(record["percent"])
        else:
            lower = float(record["lower_point"] if record["lower_point"] is not None
                          else record["lo"])
            upper = float(record["hi"])
        choices.append(Choice(
            index=record["index"], item_id=option.item_id, name_ru=option.name_ru,
            auction_lot_id=option.auction_lot_id,
            owned_instance_id=option.owned_instance_id,
            baseline_percent=baseline, percent=float(percent),
            minimum=lower, maximum=upper,
            known=bool(getattr(physical, "art_percent_known", True)),
            variable=not record["fixed"],
            changed=abs(float(percent) - baseline) > 1e-9,
        ))
    if baseline_stats is None:
        baseline_stats = stats
    if baseline_value is None:
        baseline_value = _objective_value(objective, stats)
    return Plan(True, status_text, reason, tuple(choices), tuple(percents),
                dict(stats), _objective_value(objective, stats), bool(carryable),
                list(warnings), dict(baseline_stats), float(baseline_value),
                status, message)


def _planned_option(option, percent, container, variable):
    # A known roll or an owned copy is a fact, not a target.  Leave it
    # untouched so no UI/API plan marker can imply that it can be re-rolled.
    if not variable:
        return option
    stats, raw, names = _stats_at(option, percent, container)
    physical = _source(option)
    planned = replace(option, art_percent=float(percent), stats=dict(stats),
                      stats_raw=dict(raw), stats_names=dict(names))
    planned._quality_source = physical
    planned._roll_plan_percent = float(percent)
    planned._roll_plan_source_percent = float(physical.art_percent)
    return planned


def optimize_build(build, objective, minimum=None, maximum=None, carry=True,
                   duplicates=False, time_limit=MAX_SECONDS, container=None,
                   return_plan=False):
    """Return a new build with a feasible roll plan, or ``None``.

    ``container`` overrides the build's container when the caller has already
    decorated the build and kept the effective (armor-augmented) profile.
    """
    from .builder import _final_valid, _process_combo
    effective = container if container is not None else build.container
    artifacts = list(build.artifacts)
    plan = solve(artifacts, effective, objective, minimum, maximum,
                 carry=carry, time_limit=time_limit)
    if not plan.feasible:
        return (None, plan) if return_plan else None
    planned = [_planned_option(option, percent, effective, choice.variable)
               for option, percent, choice in zip(artifacts, plan.percents, plan.choices)]
    checked = []
    _process_combo(tuple(planned), effective, build.total_price,
                   build.priorities, None, checked, 0, objective=objective,
                   max_thresholds=None, require_carry=False,
                   allow_duplicates=duplicates)
    if not checked:
        return (None, plan) if return_plan else None
    result = checked[0]
    if not _final_valid(result, math.inf, minimum, maximum, carry, False, duplicates):
        return (None, plan) if return_plan else None
    result._search_request = getattr(build, "_search_request", None)
    result.guaranteed_carryable = False
    result.roll_plan = _plan_dict(plan, result)
    return (result, plan) if return_plan else result


def _plan_dict(plan, build):
    return dict(
        enabled=True,
        exact_hundredths=True,
        status=plan.status,
        reason=plan.reason,
        solver_status=plan.solver_status,
        message=plan.message,
        objective_value=round(plan.objective_value, 6),
        baseline_objective=round(plan.baseline_objective, 6),
        carryable=bool(build.carryable),
        warnings=list(build.carryability_warnings),
        known_percent_preserved=True,
        artifacts=[
            dict(index=choice.index, item_id=choice.item_id, name_ru=choice.name_ru,
                 auction_lot_id=choice.auction_lot_id,
                 owned_instance_id=choice.owned_instance_id,
                 baseline_percent=round(choice.baseline_percent, 6),
                 percent=round(choice.percent, 2),
                 minimum=round(choice.minimum, 4),
                 maximum=round(choice.maximum, 4),
                 known=choice.known, variable=choice.variable,
                 changed=choice.changed)
            for choice in plan.choices
        ],
        note=("Целевые роллы для неизвестных процентов в сотых. "
              "Известные проценты и хранилище не меняются; это план, а не гарантия рынка."),
    )


def plan_failure(plan, build):
    """JSON-safe description of a plan that could not be applied."""
    return dict(
        enabled=True,
        exact_hundredths=True,
        status=plan.status,
        reason=plan.reason,
        solver_status=plan.solver_status,
        message=plan.message,
        carryable=bool(build.carryable),
        warnings=list(build.carryability_warnings),
        known_percent_preserved=True,
        artifacts=[],
        note="План роллов не найден для этой сборки; показан исходный максимум тира.",
    )


def optimize_pool(builds, objective, minimum=None, maximum=None, carry=True,
                  duplicates=False, limit=200, time_limit=MAX_SECONDS):
    """Optimize only builds that need a roll plan, keeping fixed-feasible ones."""
    from .builder import _final_valid
    minimum = minimum or {}
    maximum = maximum or {}
    kept = []
    needs = []
    seen = set()
    def physical_identity(option):
        # Different auction lots are different purchase choices even when their
        # item, quality and displayed percentage match.  Never collapse them:
        # the cheaper lot may be the only one that still fits the budget.
        return (getattr(option, "physical_id", None), option.auction_lot_id,
                option.owned_instance_id, option.item_id, option.quality, option.ptn,
                round(float(option.art_percent), 6),
                tuple(sorted(option.unlocked_bonus_ids or ())))

    for build in builds:
        key = (build.container.item_id if build.container else None,
               getattr(getattr(build, "_armor", None), "get", lambda *_: None)("id"),
               tuple(sorted((physical_identity(o) for o in build.artifacts), key=repr)))
        if key in seen:
            continue
        seen.add(key)
        fixed_ok = _final_valid(build, math.inf, minimum, maximum, carry, False, duplicates)
        if fixed_ok:
            kept.append(build)
        else:
            needs.append(build)
    needs.sort(key=objective.sort_key)
    attempted = 0
    optimized = 0
    failed = 0
    for build in needs[:max(0, int(limit))]:
        attempted += 1
        result = optimize_build(build, objective, minimum, maximum, carry,
                                duplicates, time_limit)
        if result is None:
            failed += 1
        else:
            optimized += 1
            kept.append(result)
    return kept, dict(enabled=True, exact_hundredths=True, attempted=attempted,
                      optimized=optimized, failed=failed,
                      skipped=max(0, len(needs) - attempted),
                      limit=int(limit), carry=bool(carry),
                      known_percent_preserved=True)

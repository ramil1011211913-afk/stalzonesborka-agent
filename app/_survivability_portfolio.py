"""Additional witnesses only: never alters the main/cost MILP or its lock."""
import time
import math

MIN_CANDIDATES = 16  # Avoid seed overhead on small, already cheap problems.


def enabled(model):
    return (model.get('incumbent_seed') and model.get('time_policy') == 'deep_remaining'
            and model['n'] >= MIN_CANDIDATES
            and model['objective_keys'] == ['healing_ehp_per_second'])


def native(model, indices):
    from ._survivability import values
    names = model.get('metric_item_ids')
    order = sorted(indices, key=lambda i: names[i]) if names else indices
    stats = {}
    for i in order:
        for key, value in model['metric_inputs'][i].items():
            stats[key] = stats.get(key, 0.) + value
    for key, value in model['bonuses'].items():
        if value:
            stats[key] = stats.get(key, 0.) + value
    stats.update(values(stats))
    score = sum(stats.get(k, 0.) * a for k, a in
                zip(model['objective_keys'], model['objective_coefficients']))
    return stats, score


def augment(model, solver, kwargs, initial, deadline, nodes, validate, publish):
    import numpy as np
    from ._max_quality_worker import _finite_number, _stop_reason
    charged = 0
    seen = set()
    current = initial
    for attempt in range(2):
        remaining = deadline-time.monotonic()
        cap = min(1700, nodes-charged)
        if remaining <= .01 or cap <= 0:
            break
        if current is None or tuple(current) in seen:
            goal = -np.array([s.get('bullet_dmg_factor', 0.) for s in model['metric_inputs']])
            proxy = 'bullet_proxy'
        else:
            stats, _ = native(model, current)
            E, R = stats['effective_health'], stats['healing_percent_per_second']
            H, B = stats.get('health_bonus', 0.), stats.get('bullet_dmg_factor', 0.)
            T, U = stats.get('artefakt_heal', 0.), stats.get('heal_efficiency', 0.)
            goal = -np.array([((100+B)*R/10000*s.get('health_bonus', 0.)
                              +(1+H/100)*R/100*s.get('bullet_dmg_factor', 0.)
                              +E/500*s.get('regeneration_bonus', 0.)
                              +E/100*(1+U/100)*s.get('artefakt_heal', 0.)
                              +E*T/10000*s.get('heal_efficiency', 0.))
                             for s in model['metric_inputs']])
            seen.add(tuple(current)); proxy = 'gradient_proxy'
        if not np.isfinite(goal).all():
            break
        start = time.monotonic()
        result = solver(goal, **dict(kwargs, options=dict(time_limit=min(5., remaining),
                         node_limit=cap, mip_rel_gap=0.)))
        count = _finite_number(getattr(result, 'mip_node_count', None))
        valid_count = count is not None and count >= 0 and count.is_integer()
        charge = int(count) if valid_count else cap
        charged += charge
        message = str(result.message)[:1000]
        reason, code = _stop_reason(result.status, message, int(count) if valid_count else None, cap)
        if result.status == 0 and code in (None,7):
            reason = 'optimal_seed_only'
        record = dict(stage=-2-attempt, kind='portfolio_witness', seed_goal=proxy,
                      solver_status=int(result.status), solver_message=message,
                      node_count=int(count) if valid_count else None, reserved_nodes=charge,
                      node_limit=cap, time_limit_seconds=min(5., remaining),
                      elapsed_seconds=time.monotonic()-start, termination_reason=reason,
                      highs_status=code, objective_upper_bound=None, cost_lower_bound=None,
                      # Proxy bounds/gaps cannot certify the true nonlinear goal.
                      bound=None, gap=None, changes_primary_model=False)
        try:
            x = np.asarray(result.x, dtype=float) if getattr(result, 'x', None) is not None else None
        except (TypeError, ValueError):
            x = None
        if (x is not None and x.shape == (model['n'],) and np.isfinite(x).all()
                and np.max(abs(x-np.rint(x)), initial=0) <= 1e-5
                and np.all((np.rint(x) >= 0) & (np.rint(x) <= 1))):
            indices = np.flatnonzero(x > .5).tolist()
            stats, score = native(model, indices)
            if (len(indices) == int(model['rows'][0]['lo']) and math.isfinite(score)
                    and validate(indices, stats, score) is not None):
                current = indices
                record.update(indices=indices, actual_primary=score,
                              derived_metrics={k:stats[k] for k in model['metric_vars']})
        publish(record, charged)
    return charged

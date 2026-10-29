"""Opt-in website-scoped survivability metrics; legacy values keep their units."""
import math
import sys
from fractions import Fraction

REGISTRY = {
    'effective_health': ('Приведённое здоровье (EHP от пуль)', 'EHP', 'ehp'),
    'healing_percent_per_second': ('Лечение в секунду (%/с)', '%/с', 'healing_percent'),
    'healing_ehp_per_second': ('Восстановление (ед./с, Wiki)', 'EHP/с', 'tooltip_absolute'),
}
CONTRACT = 'survivability-objective-v1'
FORMULA = 'wiki-bullet-ehp-healing-v1'
MAX_VARIABLES = 20000
MAX_ROWS = 50000
MAX_NONZEROS = 500000

def requested(priorities=None, minimum=None, maximum=None):
    keys = {k for k,v in (priorities or {}).items() if isinstance(v,(int,float)) and v>0}
    return bool((keys | set(minimum or {}) | set(maximum or {})) & REGISTRY.keys())

def values(stats):
    from ._survivability_math import calculate_effective_health, calculate_healing_per_second
    e = calculate_effective_health(stats); r = calculate_healing_per_second(stats)
    result = dict(effective_health=e, healing_percent_per_second=r, healing_ehp_per_second=e*r/100)
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite survivability result')
    return result

def provenance():
    return dict(formula_version=FORMULA, source_scope='STALZONE Wiki calculator; fixed equipment, no selected reactions or combat debuffs',
                units={k:v[1] for k,v in REGISTRY.items()}, legacy_healing_per_second_unit='%/с',
                absolute_interpretation='Website tooltip equivalent to EHP*rate/100; not independently verified ordinary game HP/s',
                actual_percent_confirmed=False)

def validate(priorities, minimum, maximum, *, max_quality, mode):
    if not requested(priorities, minimum, maximum): return
    if not max_quality or mode != 'balanced':
        raise ValueError('Производные показатели поддержаны только в Max Quality / Deep, Balanced; нелинейный Lexicographic и Beam не поддержаны')
    for limits in (minimum,maximum):
        if limits is not None and (not isinstance(limits,dict) or any(
            type(v) not in (int,float) or not math.isfinite(v) for v in limits.values())):
            raise ValueError('Thresholds must be finite numbers')
        if limits and any(k in REGISTRY and abs(v)>1e12 for k,v in limits.items()):
            raise ValueError('Derived thresholds exceed the supported numeric range ±1e12')

def normalized_objective(objective, profiles, overrides=None):
    """Fixed once across full profiles; legacy-only requests never call this."""
    from ._objective import Objective
    from ._survivability_circuit import Circuit
    scales=dict(zip(objective.keys, objective.scales))
    wanted={REGISTRY[k][2] for k in objective.keys if k in REGISTRY}
    if not wanted:return objective
    for key in objective.keys:
        if key in REGISTRY: scales[key]=1.
    for container,options in profiles:
        if len(options)<container.slots:continue
        bonuses={k:v.get('value',0) if isinstance(v,dict) else v for k,v in container.bonus_stats.items()}
        c=Circuit([dict(stats=o.stats,buyout_price=o.buyout_price) for o in options],container.slots,bonuses)
        # Normalization needs only interval arithmetic, not a second O(n)
        # mixed-integer matrix in the API process. Same outward bounds as Circuit.
        F=Fraction
        def outward(lo,hi):return math.nextafter(float(lo),-math.inf),math.nextafter(float(hi),math.inf)
        def product(a,b):
            v=[F(x)*F(y) for x in a for y in b];return outward(min(v),max(v))
        def add(parts):
            return outward(sum(F(a)*(F(v[0]) if a>=0 else F(v[1])) for a,v in parts),
                           sum(F(a)*(F(v[1]) if a>=0 else F(v[0])) for a,v in parts))
        h=c.interval('health_bonus',1,.01);bullet=c.interval('bullet_dmg_factor',100)
        e=product(h,bullet);g=c.interval('regeneration_bonus',.5,.2)
        has_t=any(o.stats.get('artefakt_heal',0) for o in options) or bonuses.get('artefakt_heal',0)
        if has_t:
            t=c.interval('artefakt_heal');u=c.interval('heal_efficiency',1,.01)
            rate=add([(1,g),(1,product(t,u))])
            absolute=add([(.01,product(e,g)),(.01,product(product(e,t),u))])
        else:
            rate=g;absolute=add([(.01,product(e,g))])
        bounds=dict(ehp=e,healing_percent=rate,tooltip_absolute=absolute)
        for key in objective.keys:
            if key in REGISTRY:
                lo,hi=bounds[REGISTRY[key][2]];scales[key]=max(scales[key],abs(lo),abs(hi))
    if overrides:scales.update(overrides)
    return Objective.create(dict(zip(objective.keys,objective.weights)), scales, objective.mode, None, objective.tolerance)

def make_model(candidates,container,objective,budget,minimum,maximum,duplicates,carry):
    from . import _max_quality_model as mm
    from ._objective import Objective
    from ._survivability_circuit import Circuit
    if objective.mode!='balanced': raise mm.Unsupported('unsupported_survivability_lexicographic')
    if len(objective.keys)>8 or len(set(objective.keys)|set(minimum)|set(maximum))>128:
        raise mm.Unsupported('priority_or_constraint_limit')
    if any(a==0 for a in objective.coefficients):
        raise mm.Unsupported('survivability_numeric_limit')
    weights={k:w for k,w in zip(objective.keys,objective.weights) if k not in REGISTRY}
    scales={k:s for k,s in zip(objective.keys,objective.scales) if k not in REGISTRY}
    old=mm.make_model(candidates,container,Objective.create(weights,scales),budget,
        {k:v for k,v in minimum.items() if k not in REGISTRY},
        {k:v for k,v in maximum.items() if k not in REGISTRY},duplicates,carry)
    bonus={k:v.get('value',0) if isinstance(v,dict) else v for k,v in container.bonus_stats.items()}
    options=[dict(stats=o.stats,buyout_price=o.buyout_price) for o in candidates]
    rows=[dict(values=[[i,v] for i,v in enumerate(r['values']) if v],lo=r['lo'],hi=r['hi']) for r in old['rows']]
    c=Circuit(options,container.slots,bonus,rows)
    needed={REGISTRY[k][2] for k in set(objective.keys)|set(minimum)|set(maximum) if k in REGISTRY}
    ids=c.metrics(needed)
    for key,target in minimum.items():
        if key in REGISTRY:c.row({ids[REGISTRY[key][2]]:1},target,None)
    for key,target in maximum.items():
        if key in REGISTRY:c.row({ids[REGISTRY[key][2]]:1},None,target)
    goal={};constant=0.
    for key,a in zip(objective.keys,objective.coefficients):
        if key in REGISTRY:
            j=ids[REGISTRY[key][2]];goal[j]=goal.get(j,0)+a
        else:
            constant+=bonus.get(key,0)*a
            for i,o in enumerate(candidates):goal[i]=goal.get(i,0)+o.stats.get(key,0)*a
    nonzeros=sum(len(r['values']) for r in c.rows)
    if any(not math.isfinite(v) or abs(v)>1e8 for v in goal.values()) or not math.isfinite(constant):
        raise mm.Unsupported('nonfinite_objective')
    magnitude=max((abs(v) for v in goal.values()),default=0.)
    if 0<magnitude<sys.float_info.min:
        raise mm.Unsupported('survivability_numeric_limit')
    if len(c.lb)>MAX_VARIABLES or len(c.rows)>MAX_ROWS or nonzeros>MAX_NONZEROS:
        raise mm.Unsupported('survivability_matrix_limit')
    if any(not math.isfinite(v) or abs(v)>1e14 for v in c.lb+c.ub):
        raise mm.Unsupported('survivability_numeric_limit')
    seed_goal=[o.stats.get('regeneration_bonus',0) for o in candidates]
    seed=(tuple(objective.keys)==('healing_ehp_per_second',) and any(seed_goal))
    return dict(old,survivability_contract=CONTRACT,base_rows=len(old['rows']),base_nonzeros=old['nonzeros'],rows=c.rows,nonzeros=nonzeros,
                variables=len(c.lb),variable_lower=c.lb,variable_upper=c.ub,integrality=c.integer,
                goals=[[[i,v] for i,v in goal.items() if v]],constants=[constant],
                metric_vars={k:ids[v[2]] for k,v in REGISTRY.items() if v[2] in ids},
                metric_inputs=[o.stats for o in candidates],bonuses=bonus,
                metric_item_ids=[o.item_id for o in candidates],
                objective_keys=list(objective.keys),objective_coefficients=list(objective.coefficients),
                incumbent_seed=bool(seed),incumbent_rows=rows if seed else [],incumbent_goal=seed_goal if seed else [],
                minimum_derived={k:v for k,v in minimum.items() if k in REGISTRY},
                maximum_derived={k:v for k,v in maximum.items() if k in REGISTRY},
                formula_version=FORMULA)

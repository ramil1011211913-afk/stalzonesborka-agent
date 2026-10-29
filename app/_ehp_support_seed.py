"""Bounded incumbent proposal for explicit Deep EHP-only requests.

This linear support is a heuristic witness source. Its solver status or dual
bound NEVER certifies the nonlinear/native EHP optimum.
"""
import math,time

def eligible(model):
    n=model['n'];parts=model.get('physical_descriptors') or []
    return (n>=16 and len(parts)==n and model.get('objective_keys')==['effective_health']
        and not (set(model['minimum_derived'])|set(model['maximum_derived']))-{'healing_percent_per_second'}
        and all(not p.get('is_owned') and p.get('physical_id') and p.get('auction_lot_id') is not None for p in parts)
        and len({p['physical_id'] for p in parts})==n)

def propose(model,seconds,node_limit):
    import numpy as np
    from scipy.optimize import Bounds,LinearConstraint
    from scipy.sparse import coo_matrix
    from ._survivability_circuit import Circuit
    from ._survivability_native import native_solve
    begin=time.monotonic();n=model['n'];parts=model.get('physical_descriptors') or []
    if not eligible(model):return None,dict(applied=False,reason='unsupported_seed_semantics')
    H=1+model['bonuses'].get('health_bonus',0)*.01;B=100+model['bonuses'].get('bullet_dmg_factor',0)
    if H<=0 or B<=0:return None,dict(applied=False,reason='nonpositive_empty_factors')
    # Deterministic nearby support direction; never used as a safe EHP pruning rule.
    alpha=1.1*B/H
    if not math.isfinite(alpha) or not 1e-3<=alpha<=1e6:return None,dict(applied=False,reason='support_scale')
    base=model['rows'][1:1+model['base_rows']]
    options=[dict(stats=s,buyout_price=p) for s,p in zip(model['metric_inputs'],model['prices'])]
    c=Circuit(options,int(model['rows'][0]['lo']),model['bonuses'],base);c.tight_bounds=model.get('tight_bounds',{})
    from ._ehp_profile_order import prefix_rows
    ordered_rows=prefix_rows(model);c.rows.extend(ordered_rows);cuts=len(ordered_rows)
    h=c.affine('health_bonus',1,.01);b=c.affine('bullet_dmg_factor',100)
    no_direct=False
    if model['minimum_derived'] or model['maximum_derived']:
        # A decomposition used only to propose an incumbent. The unrestricted
        # original primary model is still solved afterwards, with its full time.
        direct={i:1. for i,s in enumerate(model['metric_inputs']) if s.get('artefakt_heal',0)}
        if direct:c.row(direct,0,0);no_direct=True
        rate=c.affine('regeneration_bonus',.5,.2)
        for key,value in model['minimum_derived'].items():c.row({rate:1.},value,None)
        for key,value in model['maximum_derived'].items():c.row({rate:1.},None,value)
    rr=[];cc=[];vv=[]
    for j,row in enumerate(c.rows):
        for i,v in row['values']:rr.append(j);cc.append(i);vv.append(v)
    goal=np.zeros(len(c.lb));goal[h]=-alpha;goal[b]=-1.
    remaining=seconds-(time.monotonic()-begin)
    if remaining<=.05:return None,dict(applied=False,reason='seed_setup_deadline')
    kw=dict(integrality=np.array(c.integer),bounds=Bounds(c.lb,c.ub),
        constraints=LinearConstraint(coo_matrix((vv,(rr,cc)),shape=(len(c.rows),len(c.lb))).tocsr(),
            [-np.inf if r['lo'] is None else r['lo'] for r in c.rows],[np.inf if r['hi'] is None else r['hi'] for r in c.rows]),
        options=dict(time_limit=remaining,node_limit=node_limit,mip_rel_gap=0.))
    result=native_solve(goal,kw)
    return result,dict(applied=True,strategy='linear_ehp_support',alpha=alpha,symmetry_rows=cuts,seconds=time.monotonic()-begin,
        variables=len(c.lb),rows=len(c.rows),objective_is_auxiliary=True,
        decomposition='no_direct_heal_witness_only' if no_direct else 'unrestricted_support')

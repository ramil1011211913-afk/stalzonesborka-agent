"""Explicit Deep EHP-only contract. No objective lock or cost solver exists here."""
import json,math,os,time
from pathlib import Path
from fractions import Fraction as F

CONTRACT='ehp-primary-only-v1'

def exact_game_family(model):
    """Sufficient proof guard, never a restriction on feasible candidates.

    Integer sums and quarter-valued health factors are exact in binary64 in
    this envelope. Outside it, a canonical solver optimum stays FEASIBLE.
    """
    if (set(model['minimum_derived'])|set(model['maximum_derived']))-{'effective_health'}:return False
    k=int(model['rows'][0]['lo']);h=[];b=[]
    for stats in [*model['metric_inputs'],model['bonuses']]:
        a=F(float(stats.get('health_bonus',0)));z=F(float(stats.get('bullet_dmg_factor',0)))
        if a.denominator!=1 or a%25 or z.denominator!=1:return False
        if F(float(float(a)*.01))!=a/100:return False
        h.append(abs(a));b.append(abs(z))
    if (100+sum(sorted(h[:-1],reverse=True)[:k])+h[-1])*(100+sum(sorted(b[:-1],reverse=True)[:k])+b[-1])>=2**50:return False
    bonus=model['bonuses'].get('health_bonus',0)
    if F(float(1+bonus*.01))!=1+F(float(bonus))/100:return False
    for row in model['rows'][1:1+model['base_rows']]:
        numbers=[v for _,v in row['values']]+[v for v in [row['lo'],row['hi']] if v is not None]
        if any(not float(v).is_integer() for v in numbers) or sum(abs(F(float(v))) for v in numbers)>=2**50:return False
    return True

def exact_primal_rows(x,kwargs):
    matrix=kwargs['constraints'].A.tocsr();lo=kwargs['constraints'].lb;hi=kwargs['constraints'].ub
    for j in range(matrix.shape[0]):
        value=sum((F(float(a))*F(float(x[i])) for i,a in zip(matrix.indices[matrix.indptr[j]:matrix.indptr[j+1]],matrix.data[matrix.indptr[j]:matrix.indptr[j+1]])),F(0))
        if (math.isfinite(lo[j]) and value<F(float(lo[j]))) or (math.isfinite(hi[j]) and value>F(float(hi[j]))):return False
    return all(F(float(a))<=F(float(v))<=F(float(b)) for a,v,b in zip(kwargs['bounds'].lb,x,kwargs['bounds'].ub))

def validate_request(enabled,deep,max_quality,priorities,mode='balanced',min_budget=False):
    if not isinstance(enabled,bool):raise ValueError('primary_only must be boolean')
    if not enabled:return
    keys={k for k,v in (priorities or {}).items() if v>0}
    if not deep or not max_quality or mode!='balanced' or min_budget or keys!={'effective_health'}:
        raise ValueError('primary_only requires Deep Max Quality, balanced and only effective_health')

def annotate_status(status,has_build):
    # Preserve legacy overall OPTIMAL=primary+cost. Primary proof is explicit.
    primary=(has_build and status.get('validation_reason')=='passed' and
             bool(status.get('models')) and all(m.get('priority_optimality_proven') and not m.get('validation_failed') for m in status['models']))
    status.update(primary_only=True,optimization_contract=CONTRACT,cost_optimization_requested=False,
                  primary_optimality_proven=bool(primary),cost_optimality_proven=False,
                  requested_objectives_proven=bool(primary),optimality_proven=False)
    if has_build:
        status['state']='FEASIBLE'
        status['message']=('FEASIBLE: Максимум EHP подтверждён численным сертификатом в поддержанном пуле. ' if primary else 'FEASIBLE: Допустимая сборка; максимум EHP не доказан. ')+ 'Минимальная стоимость не оптимизировалась.'
    else:status['message']+=' Минимальная стоимость не оптимизировалась.'
    status['limits'].update(later_stage_reserve_seconds=0,time_policy='primary_only_remaining_with_validation_and_cleanup')

def solve(model,output,*,solver=None):
    import numpy as np
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import coo_matrix
    from ._max_quality_worker import _finite_number,_stop_reason
    from ._survivability_native import witness,Evaluator
    from ._survivability_portfolio import native
    validate_request(model.get('primary_only'),model.get('time_policy')=='deep_remaining',True,dict(zip(model['objective_keys'],model['objective_coefficients'])))
    if not model.get('primary_only'):raise ValueError('Explicit primary_only required')
    real=solver or milp;deadline=model.get('deadline',time.monotonic()+model['seconds'])
    state=dict(stages=[],optimal=False,infeasible=False,priority_optimality_proven=False,cost_optimality_proven=False,
               primary_only=True,optimization_contract=CONTRACT,cost_optimization_requested=False,
               pool_sha256=model['pool_sha256'],market_generation=model.get('market_generation'),process_id=os.getpid(),
               memory_limit_kind=model.get('memory_limit_kind'),reason='time_limit',survivability_contract=model['survivability_contract'])
    def save():
        tmp=str(output)+'.tmp';Path(tmp).write_text(json.dumps(state,allow_nan=False),encoding='utf8');os.replace(tmp,output)
    def matrix(m):
        rr=[];cc=[];vv=[]
        for j,row in enumerate(m['rows']):
            for i,v in row['values']:rr.append(j);cc.append(i);vv.append(v)
        return dict(integrality=np.array(m['integrality']),bounds=Bounds(m['variable_lower'],m['variable_upper']),
            constraints=LinearConstraint(coo_matrix((vv,(rr,cc)),shape=(len(m['rows']),m['variables'])).tocsr(),
                [-np.inf if r['lo'] is None else r['lo'] for r in m['rows']], [np.inf if r['hi'] is None else r['hi'] for r in m['rows']]))
    original=matrix(model)
    def accept(m,r,kind,kw,elapsed,cap):
        count=_finite_number(getattr(r,'mip_node_count',None));nodes=int(count) if count is not None and count>=0 and count.is_integer() else None
        bound=_finite_number(getattr(r,'mip_dual_bound',None));gap=_finite_number(getattr(r,'mip_gap',None))
        reason,code=_stop_reason(r.status,str(r.message),nodes,cap)
        if r.status==0 and code in (None,7):reason='optimal'
        record=dict(stage=-1 if kind=='primary_seed' else 0,kind=kind,solver_status=int(r.status),solver_message=str(r.message)[:1000],
                    node_count=nodes,node_limit=cap,bound=bound,gap=gap,termination_reason=reason,highs_status=code,elapsed_seconds=elapsed,
                    objective_upper_bound=-bound if kind=='priority' and bound is not None else None,cost_lower_bound=None)
        x=getattr(r,'x',None)
        try:x=np.asarray(x,dtype=float) if x is not None else None
        except (ValueError,TypeError):x=None
        if x is not None and x.ndim==1 and len(x)>=m['n'] and np.isfinite(x).all():
            ids=np.flatnonzero(x[:m['n']]>.5).tolist()
            if len(ids)==int(m['rows'][0]['lo']):
                _,proof=witness(m,ids,kw);_,baseproof=witness(model,ids,original)
                stats,actual=native(model,ids)
                if (not proof['exact_violations'] and not baseproof['exact_violations'] and
                    all(stats[k]>=v for k,v in model['minimum_derived'].items()) and all(stats[k]<=v for k,v in model['maximum_derived'].items())):
                    record.update(indices=ids,actual_primary=actual,derived_metrics={k:stats[k] for k in m['metric_vars']},fraction_rows_checked=proof['exact_rows_checked'],
                        original_descriptors=[model['physical_descriptors'][i] for i in ids] if model.get('physical_descriptors') else [],
                        independent_witness_valid=True)
                    if kind=='priority' and len(x)==m['variables']:
                        record['primal_objective']=-sum(float(v)*x[j] for j,v in m['goals'][0])
                        record['auxiliary_metric_residuals']={k:float(x[j])-stats[k] for k,j in m['metric_vars'].items()}
                    # Strictly concordant native/canonical EHP and numerical bound.
                    # Any rounding disagreement downgrades, never adds epsilon.
                    c=Evaluator([dict(stats=s) for s in model['metric_inputs']],len(ids),model['bonuses'],model['rows'][1:1+model['base_rows']],indices=ids)
                    metric=c.metrics({'ehp'});exact=c.point[metric['ehp']]
                    exact_goal=exact*F(float(model['objective_coefficients'][0]))
                    concordant=exact==F(float(stats['effective_health'])) and bound is not None and F(-bound)==exact_goal
                    record['native_canonical_exact_concordance']=concordant
                    singleton=model['n']==len(ids)
                    record['primary_certificate_kind']='exact_single_physical_combination' if singleton else 'strict_concordant_numerical_milp' if concordant else 'unproved_rounding'
                    if kind=='priority' and ((r.status==0 and reason=='optimal' and gap==0 and concordant and exact_game_family(model) and np.array_equal(x[:m['n']],np.rint(x[:m['n']])) and len(x)==m['variables'] and exact_primal_rows(x,kw)) or singleton):
                        state['priority_optimality_proven']=all(s.get('actual_primary',actual)<=actual for s in state['stages'])
        state['stages'].append(record);state['reason']=reason
        if any('indices' in s for s in state['stages']) and reason=='infeasible':state['reason']='incumbent_infeasibility_conflict'
        save();return nodes
    # A bounded proxy is ONLY a witness source, never a substitute EHP objective.
    available=deadline-time.monotonic();seed_cap=min(1000,model['node_limit']);charge=0
    if available>.2:
        start=time.monotonic();base=model['rows'][:1+model['base_rows']]
        seed_model=dict(model,rows=base,variables=model['n'],variable_lower=[0.]*model['n'],variable_upper=[1.]*model['n'],integrality=[1]*model['n'])
        skw=matrix(seed_model);kwargs=dict(skw,options=dict(time_limit=min(3.,available*.05),node_limit=seed_cap,mip_rel_gap=0.))
        r=real(-np.array([s.get('bullet_dmg_factor',0.) for s in model['metric_inputs']]),**kwargs)
        # Complete auxiliary variables in the ORIGINAL model before publication.
        charge=accept(model,r,'primary_seed',original,time.monotonic()-start,seed_cap)
        charge=seed_cap if charge is None else min(model['node_limit'],charge)
    state['primary_seed_node_charge']=charge;save()
    tight=model
    try:
        import scipy
        from scipy.optimize._highspy import _core as h
        compatible=scipy.__version__=='1.18.1' and (h.HIGHS_VERSION_MAJOR,h.HIGHS_VERSION_MINOR,h.HIGHS_VERSION_PATCH)==(1,12,0)
        if compatible and deadline-time.monotonic()>.2:
            from ._ehp_lp_bounds import transform
            tight=transform(model,min(deadline-.1,time.monotonic()+5.))
            assert tight['goals']==model['goals'] and tight['integrality']==model['integrality'] and tight['metric_vars']==model['metric_vars']
            state['lp_bounds']=dict(applied=bool(tight['bound_proofs']),seconds=tight['bound_seconds'],certificates=tight['bound_proofs'])
        else:state['lp_bounds']=dict(applied=False,reason='solver_version_or_remaining_time')
    except (ValueError,AssertionError,AttributeError,ImportError) as e:
        tight=model;state['lp_bounds']=dict(applied=False,reason=type(e).__name__)
    save();remaining=deadline-time.monotonic()-1.;cap=model['node_limit']-charge
    if remaining<=0 or cap<=0:return state
    state['primary_solver_allowance_seconds']=remaining
    kwargs=matrix(tight);cost=np.zeros(tight['variables'])
    for j,v in tight['goals'][0]:cost[j]=-v
    start=time.monotonic()
    try:r=real(cost,**kwargs,options=dict(time_limit=remaining,node_limit=cap,mip_rel_gap=0.))
    except RuntimeError as e:
        if 'std::bad_alloc' in str(e):raise MemoryError('std::bad_alloc') from e
        raise
    accept(tight,r,'priority',kwargs,time.monotonic()-start,cap)
    if state['priority_optimality_proven']:state['reason']='primary_only_finished'
    save();return state

"""Sparse mixed-integer counterpart, under the existing process/job/deadline."""
import json,math,os,time
from fractions import Fraction as F
from pathlib import Path

def solve(model,output,*,solver=None):
    import numpy as np
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import coo_matrix,csr_matrix,vstack
    from ._max_quality_worker import _finite_number,_stop_reason
    from ._quality_time import stage_seconds
    from ._survivability import values
    native_allowed=solver is None
    solver=solver or milp
    n=model['n'];nv=model['variables'];rr=[];cc=[];vv=[];lo=[];hi=[]
    for j,row in enumerate(model['rows']):
        lo.append(-np.inf if row['lo'] is None else row['lo']);hi.append(np.inf if row['hi'] is None else row['hi'])
        for i,v in row['values']:rr.append(j);cc.append(i);vv.append(v)
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(lo),nv)).tocsr()
    lo=np.array(lo);hi=np.array(hi)
    vector=np.zeros(nv)
    for i,v in model['goals'][0]:vector[i]=v
    magnitude=float(np.max(abs(vector),initial=0))
    objective_scale=math.ldexp(1.,-math.floor(math.log2(magnitude))) if magnitude and (magnitude<1e-6 or magnitude>1e6) else 1.
    scaled_vector=vector*objective_scale
    state=dict(stages=[],optimal=False,infeasible=False,reason='time_limit',process_id=os.getpid(),
               pool_sha256=model['pool_sha256'],memory_limit_kind=model.get('memory_limit_kind'),
               priority_optimality_proven=False,cost_optimality_proven=False,
               survivability_contract=model['survivability_contract'])
    def save():
        tmp=str(output)+'.tmp';Path(tmp).write_text(json.dumps(state,allow_nan=False),encoding='utf8');os.replace(tmp,output)
    deadline=model.get('deadline',time.monotonic()+model['seconds'])
    def selected_value(indices):
        stats={}
        for i in indices:
            for k,v in model['metric_inputs'][i].items():stats[k]=stats.get(k,0)+v
        for k,v in model['bonuses'].items():stats[k]=stats.get(k,0)+v
        stats.update(values(stats))
        score=sum(stats.get(k,0)*a for k,a in zip(model['objective_keys'],model['objective_coefficients']))
        return stats,score
    def certified_seed_floor(indices,stats,actual):
        selected=set(indices)
        for row in model['incumbent_rows']:
            total=sum((F(float(v)) for i,v in row['values'] if i in selected),F(0))
            if row['lo'] is not None and total<F(float(row['lo'])):return None
            if row['hi'] is not None and total>F(float(row['hi'])):return None
        def affine(key,constant=0.,scale=1.):
            return F(float(constant+model['bonuses'].get(key,0)*scale))+sum((F(float(model['metric_inputs'][i].get(key,0)*scale)) for i in indices),F(0))
        e=affine('health_bonus',1,.01)*affine('bullet_dmg_factor',100)
        g=affine('regeneration_bonus',.5,.2);periodic=affine('artefakt_heal');eff=affine('heal_efficiency',1,.01)
        rate=g+periodic*eff;absolute=F(.01)*(e*g+e*periodic*eff)
        exact=dict(effective_health=e,healing_percent_per_second=rate,healing_ehp_per_second=absolute)
        for k,target in model['minimum_derived'].items():
            if stats[k]<target or exact[k]<F(float(target)):return None
        for k,target in model['maximum_derived'].items():
            if stats[k]>target or exact[k]>F(float(target)):return None
        exact_score=absolute*F(float(model['objective_coefficients'][0]))
        # Directed conversion of a proved lower bound, not a gameplay margin.
        floor=math.nextafter(float(min(F(float(actual)),exact_score)),-math.inf)
        assert F(floor)<=exact_score and floor<=actual
        return floor
    first_allowance=stage_seconds(model,0,2,deadline-time.monotonic())
    priority_deadline=time.monotonic()+max(0,first_allowance)
    seed_floor=None;seed_charge=0
    if model.get('incumbent_seed') and first_allowance>.15:
        seed_seconds=min(15. if model.get('time_policy')=='deep_remaining' else 5.,first_allowance/3)
        sr=[];sc=[];sv=[];sl=[];su=[]
        for j,row in enumerate(model['incumbent_rows']):
            sl.append(-np.inf if row['lo'] is None else row['lo']);su.append(np.inf if row['hi'] is None else row['hi'])
            for i,v in row['values']:sr.append(j);sc.append(i);sv.append(v)
        seed_cap=min(5000,model['node_limit']);start=time.monotonic()
        s=solver(-np.array(model['incumbent_goal']),integrality=np.ones(n),bounds=Bounds(0,1),
            constraints=LinearConstraint(coo_matrix((sv,(sr,sc)),shape=(len(sl),n)).tocsr(),sl,su),
            options=dict(time_limit=seed_seconds,node_limit=seed_cap,mip_rel_gap=0.))
        count=_finite_number(getattr(s,'mip_node_count',None))
        valid_count=count is not None and count>=0 and count.is_integer()
        seed_charge=min(model['node_limit'],int(count)) if valid_count else seed_cap
        seed=dict(stage=-1,kind='incumbent_seed',seed_goal='regeneration_bonus',solver_status=int(s.status),solver_message=str(s.message)[:1000],
                  time_limit_seconds=seed_seconds,elapsed_seconds=time.monotonic()-start,node_limit=seed_cap,
                  node_count=int(count) if valid_count else None,reserved_nodes=seed_charge,
                  bound=_finite_number(getattr(s,'mip_dual_bound',None)),gap=_finite_number(getattr(s,'mip_gap',None)),
                  objective_upper_bound=None,cost_lower_bound=None)
        try:sx=np.asarray(s.x,dtype=float) if getattr(s,'x',None) is not None else None
        except (TypeError,ValueError):sx=None
        if sx is not None and sx.shape==(n,) and np.isfinite(sx).all() and np.max(abs(sx-np.rint(sx)),initial=0)<=1e-5 and np.all((np.rint(sx)>=0)&(np.rint(sx)<=1)):
            indices=np.flatnonzero(sx>.5).tolist();stats,score=selected_value(indices)
            seed_floor=certified_seed_floor(indices,stats,score)
            if seed_floor is not None:seed.update(indices=indices,actual_primary=score,derived_metrics={k:stats[k] for k in model['metric_vars']},objective_floor=seed_floor)
        reason,native=_stop_reason(s.status,seed['solver_message'],seed['node_count'],seed_cap)
        if s.status==0 and native in (None,7):reason='optimal_seed_only'
        seed.update(termination_reason=reason,highs_status=native)
        state['stages'].append(seed);state['primary_seed_node_charge']=seed_charge
        if seed_floor is not None:
            matrix=vstack([matrix,csr_matrix(scaled_vector.reshape(1,-1))],format='csr');lo=np.append(lo,(seed_floor-model['constants'][0])*objective_scale);hi=np.append(hi,np.inf)
        save()
        # Extra witnesses use only the unused part of the ORIGINAL seed budget.
        # The G-seed cut above and both main/cost matrices remain unchanged.
        from . import _survivability_portfolio as portfolio
        if portfolio.enabled(model):
            initial_charge=seed_charge
            def publish(record,charged):
                state['stages'].append(record)
                state['primary_seed_node_charge']=initial_charge+charged
                save()
            extra=portfolio.augment(model,solver,
                dict(integrality=np.ones(n),bounds=Bounds(0,1),
                     constraints=LinearConstraint(coo_matrix((sv,(sr,sc)),shape=(len(sl),n)).tocsr(),sl,su)),
                seed.get('indices'),start+seed_seconds,seed_cap-initial_charge,
                certified_seed_floor,publish)
            seed_charge+=extra
    for stage in range(2):
        remaining=stage_seconds(model,stage,2,deadline-time.monotonic())
        if stage==0:remaining=min(remaining,priority_deadline-time.monotonic())
        if remaining<=0:save();return state
        cost=-scaled_vector if stage==0 else np.array(model['prices']+[0.]*(nv-n))/1e6
        start=time.monotonic()
        node_limit=model['node_limit']-seed_charge if stage==0 else model['node_limit']
        kwargs=dict(integrality=np.array(model['integrality']),bounds=Bounds(model['variable_lower'],model['variable_upper']),
            constraints=LinearConstraint(matrix,lo,hi),options=dict(time_limit=remaining,node_limit=node_limit,mip_rel_gap=0.))
        r=None;native_attempted=False
        if stage==0 and native_allowed:
            from . import _survivability_portfolio as portfolio
            if portfolio.enabled(model):
                native_attempted=True
                from . import _survivability_native as native_start
                r=native_start.attempt(model,cost,kwargs,state,save,start+remaining)
        if r is None:
            # Include any compatibility/witness/setup time in this stage budget.
            if native_attempted:
                kwargs['options']['time_limit']=remaining-(time.monotonic()-start)
                if kwargs['options']['time_limit']<=0:save();return state
            r=solver(cost,**kwargs)
        bound=_finite_number(getattr(r,'mip_dual_bound',None));gap=_finite_number(getattr(r,'mip_gap',None))
        nodes=_finite_number(getattr(r,'mip_node_count',None));nodes=int(nodes) if nodes is not None and nodes>=0 and nodes.is_integer() else None
        record=dict(stage=stage,kind='priority' if stage==0 else 'cost',solver_status=int(r.status),solver_message=str(r.message)[:1000],
                    bound=bound,gap=gap,node_count=nodes,node_limit=node_limit,time_limit_seconds=remaining,elapsed_seconds=time.monotonic()-start,
                    objective_scale=objective_scale if stage==0 else 1.,
                    objective_upper_bound=max(-bound/objective_scale+model['constants'][0],seed_floor) if stage==0 and bound is not None and seed_floor is not None else -bound/objective_scale+model['constants'][0] if stage==0 and bound is not None else None,
                    cost_lower_bound=bound*1e6 if stage==1 and bound is not None else None)
        try:x=np.asarray(r.x,dtype=float) if getattr(r,'x',None) is not None else None
        except (TypeError,ValueError):x=None
        if (x is not None and x.shape==(nv,) and np.isfinite(x).all() and
            np.max(abs(x[:n]-np.rint(x[:n])),initial=0)<=1e-5 and
            np.all((np.rint(x[:n])>=0)&(np.rint(x[:n])<=1))):
            indices=np.flatnonzero(x[:n]>.5).tolist();stats,actual=selected_value(indices)
            record.update(indices=indices,primal_objective=float(cost@x),actual_primary=actual,
                          derived_metrics={k:stats[k] for k in model['metric_vars']},
                          auxiliary_metric_residuals={k:float(x[i])-stats[k] for k,i in model['metric_vars'].items()})
        reason,native=_stop_reason(r.status,record['solver_message'],nodes,node_limit)
        if r.status==0 and native in (None,7):reason='optimal'
        record.update(termination_reason=reason,highs_status=native);state['stages'].append(record)
        if 'indices' not in record or r.status!=0 or reason!='optimal':
            state.update(reason='secondary_infeasible' if reason=='infeasible' and stage else 'restricted_subproblem_infeasible' if reason=='infeasible' and seed_floor is not None else reason,
                         infeasible=reason=='infeasible' and stage==0 and seed_floor is None and 'indices' not in record)
            if state['infeasible'] and any('indices' in s for s in state['stages'] if s.get('kind')=='portfolio_witness'):
                state.update(infeasible=False,reason='incumbent_infeasibility_conflict')
            save();return state
        if stage==0:
            actual=record['actual_primary'];target=actual-model['constants'][0]
            # A stronger independently retained native witness contradicts a
            # claimed primary maximum. Never minimize price at the weaker goal.
            witnesses=[s for s in state['stages'] if s.get('kind')=='portfolio_witness' and 'indices' in s]
            if witnesses:
                from ._survivability_portfolio import native
                _,main_native=native(model,record['indices'])
                if any(s['actual_primary']>main_native for s in witnesses):
                    state['reason']='portfolio_primary_not_proved';save();return state
            # Same numerical primary certificate contract as legacy Balanced,
            # compared with actual selected metrics, never auxiliary variables.
            if bound is None or abs((record['objective_upper_bound']-actual)*objective_scale)>1e-7*max(1.,abs(target*objective_scale)):
                state['reason']='unproved_primary';save();return state
            state['priority_optimality_proven']=True
            matrix=vstack([matrix,csr_matrix((scaled_vector*1e6).reshape(1,-1))],format='csr')
            lo=np.append(lo,(target*objective_scale)*1e6);hi=np.append(hi,np.inf)
        else:
            paid=sum(model['prices'][i] for i in record['indices'])
            state['cost_optimality_proven']=bound is not None and math.ceil(bound*1e6-1e-6)>=paid
            state['optimal']=state['priority_optimality_proven'] and state['cost_optimality_proven']
            state['reason']='finished' if state['optimal'] else 'unproved_cost_tie'
        save()
    return state

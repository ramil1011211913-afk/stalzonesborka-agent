"""Version-pinned native primary MIP-start; unchanged physical model and V1 cost."""
import math,time,json
from fractions import Fraction as F
import numpy as np
from scipy.optimize import milp,OptimizeResult,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from ._survivability_circuit import Circuit
from ._survivability import REGISTRY
from ._survivability_portfolio import native

class Evaluator(Circuit):
 def __init__(self,*args,indices=(),**kw):
  super().__init__(*args,**kw);self.point={i:F(int(i in indices)) for i in range(self.n)}
 def affine(self,key,constant=0,scale=1):
  j=super().affine(key,constant,scale)
  self.point[j]=F(float(constant+self.bonus.get(key,0)*scale))+sum((F(float(o['stats'].get(key,0)*scale))*self.point[i] for i,o in enumerate(self.options)),F(0));return j
 def product(self,p,key,constant=0,scale=1):
  j=super().product(p,key,constant,scale);a=F(float(constant+self.bonus.get(key,0)*scale));cursor=j+1
  for i,o in enumerate(self.options):
   coef=float(o['stats'].get(key,0)*scale)
   if coef:self.point[cursor]=self.point[p]*self.point[i];cursor+=1;a+=F(coef)*self.point[i]
  self.point[j]=self.point[p]*a;return j
 def add(self,parts):
  j=super().add(parts);self.point[j]=sum((F(float(a))*self.point[v] for v,a in parts),F(0));return j

def witness(m,ids,kw):
 opts=[dict(stats=s) for s in m['metric_inputs']]
 c=Evaluator(opts,int(m['rows'][0]['lo']),m['bonuses'],m['rows'][1:1+m['base_rows']],indices=ids)
 c.tight_bounds=m.get('tight_bounds',{})
 c.metrics({REGISTRY[k][2] for k in m['metric_vars']})
 assert len(c.point)==m['variables']
 point=[c.point[i] for i in range(m['variables'])];bad=[]
 A=kw['constraints'].A.tocsr();lower=np.broadcast_to(kw['constraints'].lb,A.shape[0]);upper=np.broadcast_to(kw['constraints'].ub,A.shape[0])
 for j in range(A.shape[0]):
  val=sum((F(float(a))*point[i] for i,a in zip(A.indices[A.indptr[j]:A.indptr[j+1]],A.data[A.indptr[j]:A.indptr[j+1]])),F(0))
  if (math.isfinite(lower[j]) and val<F(float(lower[j]))) or (math.isfinite(upper[j]) and val>F(float(upper[j]))):bad.append(j)
 for i,(a,b) in enumerate(zip(kw['bounds'].lb,kw['bounds'].ub)):
  if not F(float(a))<=point[i]<=F(float(b)):bad.append('bound:'+str(i))
 x=np.array([float(v) for v in point]);ax=A@x
 residual=max(float(np.max(np.maximum(lower-ax,0))),float(np.max(np.maximum(ax-upper,0))))
 return x,dict(indices=ids,exact_rows_checked=A.shape[0],exact_violations=bad,binary64_max_row_residual=residual,
  auxiliary_complete=len(x)==m['variables'],native_metrics=native(m,ids)[0])

def native_solve(cost,kw,start=None):
 from scipy.optimize._milp import _milp_iv,_highs_to_scipy_status_message
 from scipy.optimize._highspy import _core as h
 try:
  # Use the installed SciPy validation and option translation, same defaults.
  c,integer,lb,ub,ptr,idx,val,lo,hi,options=_milp_iv(cost,kw['integrality'],kw['bounds'],kw['constraints'],dict(kw['options']))
  lp=h.HighsLp();lp.num_col_=len(c);lp.num_row_=len(lo)
  lp.col_cost_=c;lp.col_lower_=lb;lp.col_upper_=ub;lp.row_lower_=lo;lp.row_upper_=hi
  lp.a_matrix_.num_col_=len(c);lp.a_matrix_.num_row_=len(lo);lp.a_matrix_.format_=h.MatrixFormat.kColwise
  lp.a_matrix_.start_=ptr;lp.a_matrix_.index_=idx;lp.a_matrix_.value_=val
  lp.integrality_=[h.HighsVarType(int(i)) for i in integer]
  solver=h._Highs();settings=h.HighsOptions()
  for k,v in options.items():
   if v is not None:setattr(settings,k,('on' if v else 'off') if k=='presolve' and isinstance(v,bool) else v)
  if solver.passOptions(settings)==h.HighsStatus.kError:raise NativeUnavailable("pass_options")
  if solver.passModel(lp)==h.HighsStatus.kError:raise NativeUnavailable("pass_model")
  injected=None
  if start is not None:
   solution=h.HighsSolution();solution.col_value=start;solution.value_valid=True
   injected=int(solver.setSolution(solution))
   if injected!=int(h.HighsStatus.kOk):raise NativeUnavailable("set_solution")
 except (AttributeError,TypeError) as error:
  raise NativeUnavailable(type(error).__name__) from error
 run_status=solver.run()
 if run_status==h.HighsStatus.kError:raise RuntimeError("native_run_error")
 status=solver.getModelStatus();info=solver.getInfo();solution=solver.getSolution()
 mapped,message=_highs_to_scipy_status_message(status,solver.modelStatusToString(status))
 valid=bool(solution.value_valid and info.primal_solution_status==2 and math.isfinite(info.objective_function_value))
 return OptimizeResult(status=mapped,message=message,success=mapped==0,x=np.array(solution.col_value) if valid else None,
  fun=info.objective_function_value if valid else None,mip_node_count=info.mip_node_count,mip_dual_bound=info.mip_dual_bound,mip_gap=info.mip_gap,
  native_status=int(status),injection_status=injected,primal_solution_status=int(info.primal_solution_status))


class NativeUnavailable(Exception):
 """A failure BEFORE run: safe to use the original SciPy route."""

CORE_SHA256 = 'e6beb8321a4905c9bd0ba0eec2dfa7e27d76a01ab7d0abeab81fcce9ae6732e1'

def compatibility():
 try:
  import scipy,inspect,hashlib
  from pathlib import Path
  if scipy.__version__!='1.18.1':return False,'scipy_version'
  from scipy.optimize._highspy import _core as h
  from scipy.optimize._milp import _milp_iv,_highs_to_scipy_status_message
  if (h.HIGHS_VERSION_MAJOR,h.HIGHS_VERSION_MINOR,h.HIGHS_VERSION_PATCH)!=(1,12,0):return False,'highs_version'
  if hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()!=CORE_SHA256:return False,'highs_binary'
  if str(inspect.signature(_milp_iv))!='(c, integrality, bounds, constraints, options)':return False,'scipy_api'
  for name in ('setSolution','passModel','passOptions','run','getInfo','getSolution','getModelStatus'):
   if not callable(getattr(h._Highs,name,None)):return False,'native_api'
  if 'HighsSolution' not in (h._Highs.setSolution.__doc__ or ''):return False,'solution_api'
  return True,'pinned_scipy_1.18.1_highs_1.12.0'
 except (ImportError,AttributeError,TypeError,ValueError,OSError):return False,'private_api_unavailable'

def attempt(model,cost,kwargs,state,save,deadline):
 """No extra solve/seeds/nodes; all witnesses were published by V2.2 first."""
 from ._survivability_portfolio import enabled,native
 if not enabled(model):return None
 ok,why=compatibility()
 info=dict(route='native_mip_start_v1',compatible=ok,compatibility_reason=why,
           used=False,primary_only=True,additional_seed_nodes=0)
 state['native_start']=info
 if not ok:save();return None
 if any(model.get(k) for k in ('tight_bounds','bound_proofs','symmetry_rows')):
  info['fallback']='unsupported_transformed_model';save();return None
 choices=[]
 for stage in state['stages']:
  if stage.get('kind') not in ('incumbent_seed','portfolio_witness') or 'indices' not in stage:continue
  ids=stage['indices']
  try:
   x,proof=witness(model,ids,kwargs)
  except (KeyError,IndexError,ValueError,AssertionError,OverflowError) as error:
   info['rejected_witness']=type(error).__name__;continue
  # Match the existing native default, never expand a game/model threshold.
  from scipy.optimize._highspy import _core as h
  residual=max(proof['binary64_max_row_residual'],float(np.max(np.maximum(kwargs['bounds'].lb-x,0))),float(np.max(np.maximum(x-kwargs['bounds'].ub,0))))
  if proof['exact_violations'] or residual>h.HighsOptions().mip_feasibility_tolerance:continue
  _,score=native(model,ids)
  choices.append((score,x,proof))
 if not choices:info['fallback']='no_complete_certified_start';save();return None
 score,start,proof=max(choices,key=lambda a:a[0])
 info.update(indices=proof['indices'],actual_primary=score,variables=len(start),
             exact_rows_checked=proof['exact_rows_checked'],exact_violations=[],
             binary64_max_row_residual=proof['binary64_max_row_residual'],
             phase='ready',checkpoint_published=True,node_limit=kwargs['options']['node_limit'])
 save() # atomic standard checkpoint BEFORE the native call or any native allocation
 kwargs=dict(kwargs,options=dict(kwargs['options'],time_limit=deadline-time.monotonic()))
 if kwargs['options']['time_limit']<=0:
  info.update(phase='fallback',fallback='primary_time_exhausted');save();return None
 info['time_limit_seconds']=kwargs['options']['time_limit']
 save()
 began=time.monotonic()
 try:
  result=native_solve(cost,kwargs,start)
 except NativeUnavailable as error:
  info.update(phase='fallback',fallback=str(error));save();return None
 except RuntimeError as error:
  if 'std::bad_alloc' in str(error):raise MemoryError('native std::bad_alloc') from error
  raise
 info.update(used=True,phase='returned',native_status=result.native_status,
             injection_status=result.injection_status,primal_solution_status=result.primal_solution_status,
             elapsed_seconds=time.monotonic()-began)
 save()
 return result

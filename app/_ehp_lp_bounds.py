"""Any LP dual vector yields a rationally checked box bound for 0<=x<=1.
No claim that the LP solver itself returned an exact dual certificate.
"""
from fractions import Fraction as F
import math,time
def tighten(options,rows,keys,deadline=None):
 import numpy as np
 from scipy.optimize import linprog
 from scipy.sparse import coo_matrix
 n=len(options);upper=[];equal=[]
 for row in rows:
  vals=row['values'];lo=row['lo'];hi=row['hi']
  if lo is not None and hi is not None and lo==hi:equal.append((vals,lo))
  else:
   if hi is not None:upper.append((vals,hi))
   if lo is not None:upper.append(([[i,-v] for i,v in vals],-lo))
 def matrix(rs):
  rr=[];cc=[];vv=[]
  for j,(vals,_) in enumerate(rs):
   for i,v in vals:rr.append(j);cc.append(i);vv.append(v)
  return coo_matrix((vv,(rr,cc)),shape=(len(rs),n)).tocsr()
 aub=matrix(upper);aeq=matrix(equal);bub=[b for _,b in upper];beq=[b for _,b in equal]
 out={};proofs=[]
 def solve(c):
  available=3. if deadline is None else min(3.,deadline-time.monotonic())
  if available<=0:return None,None
  r=linprog(-np.array(c),A_ub=aub,b_ub=bub,A_eq=aeq,b_eq=beq,bounds=(0,1),method='highs',options={'time_limit':available})
  if r.status!=0:return None,None
  lam=[max(F(0),-F(float(v))) for v in r.ineqlin.marginals];mu=[-F(float(v)) for v in r.eqlin.marginals]
  residual=[F(float(v)) for v in c];constant=F(0)
  for (vals,b),y in zip(upper,lam):
   constant+=F(float(b))*y
   for i,v in vals:residual[i]-=F(float(v))*y
  for (vals,b),y in zip(equal,mu):
   constant+=F(float(b))*y
   for i,v in vals:residual[i]-=F(float(v))*y
  bound=constant+sum((max(F(0),v) for v in residual),F(0))
  # Equation: c*x = lambda*A*x + mu*E*x + residual*x
  # <= lambda*b + mu*d + sum(max(0,residual)), lambda>=0.
  return bound,dict(inequality_duals=[str(v) for v in lam],equality_duals=[str(v) for v in mu],constant=str(constant),positive_residual_correction=str(sum((max(F(0),v) for v in residual),F(0))),bound=str(bound),solver_status=int(r.status))
 for key in keys:
  c=[o['stats'].get(key,0) for o in options]
  high,ph=solve(c);negative,pl=solve([-v for v in c])
  if high is None or negative is None:continue
  low=-negative
  fl=math.nextafter(float(low),-math.inf);fu=math.nextafter(float(high),math.inf)
  assert F(fl)<=low and F(fu)>=high
  out[key]=[fl,fu];proofs.append(dict(key=key,lower_exact=str(low),upper_exact=str(high),outward_binary64=[fl,fu],upper_certificate=ph,negative_lower_certificate=pl))
 return out,proofs

def transform(model,deadline):
 from ._survivability_circuit import Circuit
 from ._survivability import REGISTRY
 start=time.monotonic()
 base=model['rows'][1:1+model['base_rows']]
 opts=[dict(stats=s,buyout_price=p) for s,p in zip(model['metric_inputs'],model['prices'])]
 bounds,proof=tighten(opts,base,['health_bonus','bullet_dmg_factor','regeneration_bonus','artefakt_heal','heal_efficiency'],deadline=deadline)
 from ._ehp_profile_order import prefix_rows
 symmetry=prefix_rows(model)
 slots=int(model['rows'][0]['lo']);c=Circuit(opts,slots,model['bonuses'],base);c.tight_bounds=bounds
 required={REGISTRY[k][2] for k in model['metric_vars']};ids=c.metrics(required)
 for key,val in model['minimum_derived'].items():c.row({ids[REGISTRY[key][2]]:1},val,None)
 for key,val in model['maximum_derived'].items():c.row({ids[REGISTRY[key][2]]:1},None,val)
 c.rows.extend(symmetry)
 goal={};constant=0.
 for key,a in zip(model['objective_keys'],model['objective_coefficients']):
  if key in REGISTRY:
   j=ids[REGISTRY[key][2]];goal[j]=goal.get(j,0)+a
  else:
   constant+=model['bonuses'].get(key,0)*a
   for i,s in enumerate(model['metric_inputs']):goal[i]=goal.get(i,0)+s.get(key,0)*a
 new=dict(model,rows=c.rows,variables=len(c.lb),variable_lower=c.lb,variable_upper=c.ub,integrality=c.integer,
  goals=[[[i,v] for i,v in goal.items() if v]],constants=[constant],metric_vars={k:ids[REGISTRY[k][2]] for k in model['metric_vars']},nonzeros=sum(len(r['values']) for r in c.rows))
 new['bound_proofs']=proof;new['bound_seconds']=time.monotonic()-start;new['tight_bounds']=bounds
 if symmetry:
  new['incumbent_rows']=base if model.get('incumbent_seed') else []
 new['symmetry_rows']=symmetry
 return new

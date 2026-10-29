"""Research-only exact binary/continuous product formulation, sparse O(d*n).
No production imports or changes. Bounds are explicit, finite interval bounds.
"""
import math
from fractions import Fraction as F
class Circuit:
 def __init__(self,options,slots,bonus=None,base_rows=()):
  self.options=options;self.n=len(options);self.slots=slots;self.bonus=bonus or {}
  self.lb=[0.]*self.n;self.ub=[1.]*self.n;self.integer=[1]*self.n;self.rows=[];self.tight_bounds={}
  self.row({i:1. for i in range(self.n)},slots,slots)
  for r in base_rows:self.row(dict(r['values']),r['lo'],r['hi'])
 def row(self,c,lo=None,hi=None):self.rows.append(dict(values=[[i,float(v)] for i,v in c.items() if v],lo=lo,hi=hi))
 def var(self,lo,hi):
  assert math.isfinite(lo) and math.isfinite(hi) and lo<=hi
  j=len(self.lb);self.lb.append(float(lo));self.ub.append(float(hi));self.integer.append(0);return j
 def interval(self,key,constant=0,scale=1):
  vs=sorted(F(float(o['stats'].get(key,0)*scale)) for o in self.options)
  b=F(float(constant+float(self.bonus.get(key,0))*scale))
  low=b+sum(vs[:self.slots]);high=b+sum(vs[-self.slots:])
  if key in self.tight_bounds:
   a,z=map(F,self.tight_bounds[key]);a*=F(float(scale));z*=F(float(scale))
   errors=sorted((abs(F(float(o['stats'].get(key,0)*scale))-F(float(o['stats'].get(key,0)))*F(float(scale))) for o in self.options),reverse=True)
   error=sum(errors[:self.slots]);low=max(low,b+min(a,z)-error);high=min(high,b+max(a,z)+error)
  return math.nextafter(float(low),-math.inf),math.nextafter(float(high),math.inf)
 def affine(self,key,constant=0,scale=1):
  lo,hi=self.interval(key,constant,scale);v=self.var(lo,hi)
  c={i:-float(o['stats'].get(key,0))*scale for i,o in enumerate(self.options)};c[v]=1
  rhs=constant+float(self.bonus.get(key,0))*scale;self.row(c,rhs,rhs);return v
 def product(self,p,key,constant=0,scale=1):
  L,U=self.lb[p],self.ub[p];alo,ahi=self.interval(key,constant,scale)
  bounds=[F(L)*F(alo),F(L)*F(ahi),F(U)*F(alo),F(U)*F(ahi)];z=self.var(math.nextafter(float(min(bounds)),-math.inf),math.nextafter(float(max(bounds)),math.inf))
  c={z:1,p:-(constant+self.bonus.get(key,0)*scale)}
  for i,o in enumerate(self.options):
   a=o['stats'].get(key,0)*scale
   if not a:continue
   y=self.var(min(0,L),max(0,U))
   self.row({y:1,i:-L},0,None);self.row({y:1,i:-U},None,0)
   self.row({y:1,p:-1,i:-U},-U,None);self.row({y:1,p:-1,i:-L},None,-L)
   c[y]=-a
  self.row(c,0,0);return z
 def add(self,parts):
  lo=sum(F(self.lb[v] if a>=0 else self.ub[v])*F(float(a)) for v,a in parts)
  hi=sum(F(self.ub[v] if a>=0 else self.lb[v])*F(float(a)) for v,a in parts)
  z=self.var(math.nextafter(float(lo),-math.inf),math.nextafter(float(hi),math.inf));c={z:1}
  for v,a in parts:c[v]=c.get(v,0)-a
  self.row(c,0,0);return z
 def metrics(self,required):
  out={};has_t=any(o['stats'].get('artefakt_heal',0) for o in self.options) or self.bonus.get('artefakt_heal',0)
  if required & {'ehp','tooltip_absolute'}:
   h=self.affine('health_bonus',1,.01);e=self.product(h,'bullet_dmg_factor',100);out['ehp']=e
  if 'healing_percent' in required:
   g=self.affine('regeneration_bonus',.5,.2)
   if has_t:
    t=self.affine('artefakt_heal');tu=self.product(t,'heal_efficiency',1,.01);r=self.add([(g,1),(tu,1)])
   else:r=g
   out['healing_percent']=r
  if 'tooltip_absolute' in required:
   eg=self.product(e,'regeneration_bonus',.5,.2)
   if has_t:
    et=self.product(e,'artefakt_heal');etu=self.product(et,'heal_efficiency',1,.01);a=self.add([(eg,.01),(etu,.01)])
   else:a=self.add([(eg,.01)])
   out['tooltip_absolute']=a
  return out
 def payload(self,goal,minimum=None,maximum=None,weights=None):
  required=({'ehp','healing_percent','tooltip_absolute'} if goal=='balanced' else {goal}) | set(minimum or {}) | set(maximum or {})
  ms=self.metrics(required-{'bullet'})
  for k,v in (minimum or {}).items():self.row({ms[k]:1},v,None)
  for k,v in (maximum or {}).items():self.row({ms[k]:1},None,v)
  if goal=='balanced':
   co={ms[k]:w/max(1,abs(self.lb[ms[k]]),abs(self.ub[ms[k]])) for k,w in weights.items()}
  elif goal=='bullet':co={i:o['stats'].get('bullet_dmg_factor',0) for i,o in enumerate(self.options)}
  else:co={ms[goal]:1}
  return dict(n=self.n,variables=len(self.lb),rows=self.rows,lb=self.lb,ub=self.ub,integer=self.integer,goal=[[k,v] for k,v in co.items() if v],prices=[o['buyout_price'] for o in self.options],metric_vars=ms,constant=self.bonus.get('bullet_dmg_factor',0) if goal=='bullet' else 0)

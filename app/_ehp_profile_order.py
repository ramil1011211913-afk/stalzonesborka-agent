"""Canonical no-dearer physical copies; only exact, adjacent point profiles."""
import math
from collections import defaultdict

def prefix_rows(model):
    parts=model.get('physical_descriptors') or [];n=model['n']
    required={'item_id','quality','ptn','unlocked_bonus_ids'}
    if (len(parts)!=n or len(model['metric_inputs'])!=n
        or any(not required<=p.keys() or p.get('is_owned') or p.get('auction_lot_id') is None or not p.get('physical_id') for p in parts)
        or len({p['physical_id'] for p in parts})!=n):return []
    groups=defaultdict(list)
    for i,(p,s) in enumerate(zip(parts,model['metric_inputs'])):
        if any(not math.isfinite(float(v)) for v in s.values()):return []
        key=(p['item_id'],p['quality'],p['ptn'],tuple(sorted(p['unlocked_bonus_ids'])),tuple(sorted((k,float(v).hex()) for k,v in s.items())))
        groups[key].append(i)
    price=dict(enumerate(model['prices']))
    guards=[dict(r['values']) for r in model['rows'][1:1+model['base_rows']] if not(r['lo'] is None and dict(r['values'])==price)]
    out=[]
    for g in groups.values():
        # No other coefficient profile can be crossed by replacement. This
        # preserves native summation order as well as all exact model totals.
        if len(g)<2 or g[-1]-g[0]+1!=len(g):continue
        if any(any(row.get(i,0)!=row.get(g[0],0) for i in g) for row in guards):continue
        if any(not isinstance(model['prices'][i],int) or model['prices'][i]<0 for i in g):continue
        ordered=sorted(g,key=lambda i:(model['prices'][i],i))
        out.extend(dict(values=[[a,1.],[b,-1.]],lo=0.,hi=None) for a,b in zip(ordered,ordered[1:]))
    return out

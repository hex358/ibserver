"""Подгонка поправки орбиты θ к согласованному между полосами пятну из close_search (THEORY §23)."""
import sys, json, glob, numpy as np
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s2sat import passgeom as PG
from scipy.optimize import least_squares
x=json.load(open('results/events_rb/SL-8_20261120T030857.json')); e,r=x['event'],x['result']
orbs=PG.Orbits(until='20261124T0047'); P=PG.Pass(e,r,orbs); P.orb=PG.pick_orbit(orbs,e); P0=P.positions([0.,0.,0.])
cs=json.load(open('results/close_search/SL-8_20261120T030857.json')); grp=cs['consistent'][0][1]
tgt={b:(P0[b][0]+dx,P0[b][1]+dy) for b,(z,dx,dy) in zip(['blue','green','red'],grp)}
def res(th):
    Q=P.positions(th,tgt); return np.array(sum(([Q[b][0]-tgt[b][0],Q[b][1]-tgt[b][1]] if b in Q else [5e3,5e3] for b in tgt),[]))
best=None
for a0 in (-10,-6,-2,2):
  for c0 in (-3,-1,1):
    for h0 in (-3,0,3):
      try: s=least_squares(res,[a0,c0,h0],x_scale=[2,0.5,1],diff_step=1e-4,max_nfev=150)
      except Exception: continue
      c=float(np.sqrt(np.mean(s.fun**2)))
      if best is None or c<best[0]: best=(c,s.x); print('  ', np.round(s.x,2), round(c), flush=True)
print('ИТОГ θ', np.round(best[1],2), 'остаток', round(best[0]), 'м')

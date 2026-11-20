"""Чувствительность: вставка синтетического объекта в реальный снимок события и поиск его тем же конвейером."""
import argparse, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import datetime as dt
from s2sat import orbits as O, crossings as C, pipeline as P

ap = argparse.ArgumentParser()
ap.add_argument("--obj", default="STARLINK-32519")
ap.add_argument("--start", default="2026-11-19T02:30"); ap.add_argument("--end", default="2026-11-19T03:30")
ap.add_argument("--amps", default="100,200,400,800")
ap.add_argument("--len-px", type=int, default=3); ap.add_argument("--wid-px", type=int, default=2)
ap.add_argument("--method", default="glrt")
ap.add_argument("--rho", type=float, default=None, help="DN объекта (модель заслонения); amps тогда = доли f в процентах")
ap.add_argument("--orbits", default="gp_starlink.tle")
ap.add_argument("--dm", type=float, default=10.0); ap.add_argument("--rm", type=float, default=1500.0)
a = ap.parse_args()
snap = sorted((ROOT / "data/tle_snapshots").iterdir())[-1]
sents = O.parse_tle((snap / "sentinel2.tle").read_text())
objs = [o for o in O.parse_tle((snap / a.orbits).read_text()) if o.name.split()[0] == a.obj]
t0 = dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc); t1 = dt.datetime.fromisoformat(a.end).replace(tzinfo=dt.timezone.utc)
e = C.find_crossings(sents, objs, O.jd_from_dt(t0), O.jd_from_dt(t1))[0]
sen = {s.name: s for s in sents}[e["sent"]]
for amp in [0] + [float(x) for x in a.amps.split(",")]:
    inj = (dict(amp=amp, shift=(600.0, -400.0), len_px=a.len_px, wid_px=a.wid_px,
                **({"rho_dn": a.rho, "f": amp / 100.0} if a.rho else {})) if amp else None)
    r = P.run_event(sen, objs[0], e, Rm=a.rm, minb=3, dM_max=a.dm, inject=inj, method=a.method)
    print(f"INJ {a.method} amp {amp:5.0f} DN: score {r['score']:.2f} null max {r['null_max']:.2f}  dM {r['dM']:+.1f} "
          f"shift ({r['sx']:.0f}, {r['sy']:.0f})  >4σ {r['hit']}/{r['cnt']}", flush=True)

"""E1. Ошибки орбит: старый TLE против более нового (того же объекта) и GP против SupGP.

Для пары (A — старше, B — новее) оба TLE распространяются на эпоху B, разность раскладывается
в осях RTN (радиально, вдоль, поперёк) орбиты B. «Вдоль» — то, что определяет фазу dM.
Вывод: CSV + сводка медиан/квантилей по типам объектов и возрасту.
"""
import sys, warnings, csv, math
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); warnings.filterwarnings("ignore")
import numpy as np
from s2sat import orbits as O

S = ROOT / "data/tle_snapshots"
old = {}; new = {}
for f in ["gp_active.tle", "gp_starlink.tle"]:
    for o in O.parse_tle((S / "20261120T0320" / f).read_text()): old.setdefault(o.catnr, o)
for f in ["gp_active.tle", "gp_starlink.tle", "gp_visual.tle", "gp_stations.tle"] + [p.name for p in (S / "20261121T0627").glob("debris_*.tle")]:
    p = S / "20261121T0627" / f
    if p.exists():
        for o in O.parse_tle(p.read_text()): new.setdefault(o.catnr, o)
sup_old = {o.catnr: o for o in O.parse_tle((S / "20261120T0320/supgp_starlink.tle").read_text())}

def kind(name):
    n = name.upper()
    if n.startswith("STARLINK"): return "starlink"
    if "R/B" in n: return "rocket_body"
    if "DEB" in n: return "debris"
    if n.startswith("ONEWEB"): return "oneweb"
    return "other_payload"

def rtn_diff(oa, ob, jd):
    ra, va = oa.teme([jd]); rb, vb = ob.teme([jd])
    ra, va, rb, vb = ra[0], va[0], rb[0], vb[0]
    if not (np.all(np.isfinite(ra)) and np.all(np.isfinite(rb))): return None
    R = rb / np.linalg.norm(rb); N = np.cross(rb, vb); N /= np.linalg.norm(N); T = np.cross(N, R)
    d = ra - rb
    return float(d @ R), float(d @ T), float(d @ N)

rows = []
for cat, b in new.items():
    a = old.get(cat)
    if a is None or b.epoch_jd - a.epoch_jd < 0.05 or b.perigee_km > 2000: continue
    r = rtn_diff(a, b, b.epoch_jd)
    if r: rows.append(dict(cat=cat, name=b.name, kind=kind(b.name), cmp="old_vs_new", age_d=b.epoch_jd - a.epoch_jd,
                           radial_km=r[0], along_km=r[1], cross_km=r[2]))
gp_old = {o.catnr: o for o in O.parse_tle((S / "20261120T0320/gp_starlink.tle").read_text())}
for cat, s in sup_old.items():
    g = gp_old.get(cat)
    if g is None: continue
    for dt_d in (0.0, -0.5, -1.0):   # в эпоху SupGP и раньше (наши события в прошлом относительно снимка)
        r = rtn_diff(g, s, s.epoch_jd + dt_d)
        if r: rows.append(dict(cat=cat, name=s.name, kind="starlink", cmp=f"gp_vs_supgp_{dt_d:+.1f}d", age_d=s.epoch_jd + dt_d - g.epoch_jd,
                               radial_km=r[0], along_km=r[1], cross_km=r[2]))
out = ROOT / "results/lab/orbit_errors.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("rows", len(rows), "->", out)
import collections
grp = collections.defaultdict(list)
for r in rows:
    key = (r["cmp"] if r["cmp"] != "old_vs_new" else "old_vs_new", r["kind"],
           "age<0.75" if r["age_d"] < 0.75 else ("age<1.5" if r["age_d"] < 1.5 else "age>=1.5"))
    grp[key].append(r)
print(f"{'comparison':22s} {'kind':14s} {'age':9s} {'n':>5s} | |along| km: p50   p90   p99 | |cross| p50 p90 | |radial| p50 p90")
for k in sorted(grp):
    v = grp[k]
    if len(v) < 5: continue
    A = np.abs([x["along_km"] for x in v]); C = np.abs([x["cross_km"] for x in v]); R = np.abs([x["radial_km"] for x in v])
    q = lambda X, p: np.percentile(X, p)
    print(f"{k[0]:22s} {k[1]:14s} {k[2]:9s} {len(v):5d} | {q(A,50):6.2f} {q(A,90):6.2f} {q(A,99):7.2f} | {q(C,50):5.2f} {q(C,90):5.2f} | {q(R,50):5.2f} {q(R,90):5.2f}")

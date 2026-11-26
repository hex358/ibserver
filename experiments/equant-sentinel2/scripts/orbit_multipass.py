"""Орбита по нескольким пролётам одного объекта: уточнение элементов публичного TLE по положениям со снимков.

  python scripts/orbit_multipass.py STARLINK-3358 [--out results/orbits]
Положения — results/orbits/<объект>_<время>.json (орбита Sentinel по GPS, углы полос в 3D). Основа — последний
публичный TLE с эпохой ДО последнего пролёта (то, что было бы доступно). Подгоняются поправки элементов SGP4:
средняя аномалия M₀, среднее движение n (торможение копится как ошибка n), при ≥ 3 пролётах — ещё долгота узла и
наклонение (плоскость). Невязки — вдоль и поперёк орбиты с погрешностями снимка.
Проверка прогнозом: эталоны — более поздние публичные TLE в СВОЮ эпоху; сравниваются
  было (основа как есть) / один пролёт (только фаза по последнему) / все пролёты (подгонка элементов).
"""
import argparse, glob, json, math, sys
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, WGS72

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import geometry as G, orbits as O, passgeom as PG

ap = argparse.ArgumentParser(); ap.add_argument("obj"); ap.add_argument("--out", default=str(ROOT / "results/orbits"))
a = ap.parse_args()


def make(base, dM=0.0, dn=0.0, dnode=0.0, dinc=0.0):
    s = base.satrec; r = Satrec()
    r.sgp4init(WGS72, "i", s.satnum, s.jdsatepoch + s.jdsatepochF - 2433281.5, s.bstar, s.ndot, s.nddot, s.ecco, s.argpo,
               s.inclo + dinc, s.mo + dM, s.no_kozai + dn, s.nodeo + dnode)
    return O.Orbit(base.name, r, base.line1, base.line2)


def axes(r, v):
    uv = G.unit(v); ur = G.unit(r); uc = G.unit(np.cross(ur, uv)); return np.stack([uv, uc, ur])


passes = []
for f in sorted(glob.glob(str(Path(a.out) / f"{a.obj}_*.json"))):
    d = json.loads(Path(f).read_text())
    if "position_ecef_km" in d:
        passes.append(d)
passes.sort(key=lambda d: d["jd"])
if len(passes) < 2:
    sys.exit(f"{a.obj}: пролётов с положением {len(passes)} — нужно ≥ 2")
cat = str(passes[0]["norad"])
orbs = PG.Orbits(until="99999999")
pool = orbs.GP.get(cat, []) + orbs.SG.get(cat, []) or orbs.GP.get(cat.zfill(5), []) + orbs.SG.get(cat.zfill(5), [])
t_last = passes[-1]["jd"]
before = [o for o in pool if o.epoch_jd < t_last]
base = max(before, key=lambda o: o.epoch_jd)
print(f"{a.obj}: пролётов {len(passes)} ({', '.join(O.dt_from_jd(p['jd']).strftime('%d.%m %H:%M') for p in passes)}), основа — TLE "
      f"{O.dt_from_jd(base.epoch_jd).strftime('%d.%m %H:%M')} ({t_last - base.epoch_jd:.2f} сут до последнего пролёта)")


def resid(orb, ps):
    out = []
    for p in ps:
        r, v = orb.ecef([p["jd"]]); A = axes(r[0], v[0]); d = A @ (r[0] - np.asarray(p["position_ecef_km"])) * 1e3   # м
        s = np.asarray(p["sigma_m"])
        out += [d[0] / max(s[0], 20.0), d[1] / max(s[1], 20.0)]
    return np.array(out)


npar = 4 if len(passes) >= 3 else 2
x = np.zeros(npar); steps = np.array([1e-5, 1e-9, 1e-6, 1e-6])[:npar]
for _ in range(8):                                     # Гаусс–Ньютон с численным якобианом
    f0 = resid(make(base, *x), passes)
    J = np.stack([(resid(make(base, *(x + np.eye(npar)[k] * steps[k])), passes) - f0) / steps[k] for k in range(npar)], 1)
    dx, *_ = np.linalg.lstsq(J, -f0, rcond=None); x = x + dx
    if np.all(np.abs(dx) < steps * 1e-2):
        break
fit = make(base, *x)
rb = resid(base, passes); rf = resid(fit, passes)
print(f"  невязки в пролётах (в σ): основа {np.round(rb, 1).tolist()} → подгонка {np.round(rf, 2).tolist()}")
print(f"  поправки: ΔM₀ {math.degrees(x[0]) * 1e3:+.3f} мград, Δn {x[1] * 1440 / (2 * math.pi) * 1e6:+.2f}·10⁻⁶ об/сут"
      + (f", Δузел {math.degrees(x[2]) * 1e3:+.2f} мград, Δнакл. {math.degrees(x[3]) * 1e3:+.2f} мград" if npar == 4 else ""))
# один пролёт: только фаза по последнему
pl = passes[-1]; r, v = base.ecef([pl["jd"]]); A = axes(r[0], v[0]); d0 = A @ (r[0] - np.asarray(pl["position_ecef_km"]))
dM1 = -float(d0[0]) / float(np.linalg.norm(r[0])) if False else None
x1 = np.zeros(2)
for _ in range(6):
    f0 = resid(make(base, x1[0]), [pl])[:1]; J1 = (resid(make(base, x1[0] + 1e-5), [pl])[:1] - f0) / 1e-5
    x1[0] -= float(f0[0] / J1[0])
one = make(base, x1[0])
refs = sorted([o for o in pool if o.epoch_jd > t_last + 0.05], key=lambda o: o.epoch_jd)
rows = []
for o in refs:
    t2 = o.epoch_jd; r_ref, v_ref = o.ecef([t2]); A2 = axes(r_ref[0], v_ref[0])
    e = {k: A2 @ (m.ecef([t2])[0][0] - r_ref[0]) for k, m in (("было", base), ("один пролёт", one), ("все пролёты", fit))}
    rows.append(dict(ref_epoch_days=t2 - t_last, **{k: float(np.linalg.norm(v)) for k, v in e.items()},
                     **{f"{k} вдоль": float(v[0]) for k, v in e.items()}))
    print(f"  +{t2 - t_last:5.2f} сут: ошибка прогноза — было {np.linalg.norm(e['было']):7.2f} км, один пролёт {np.linalg.norm(e['один пролёт']):7.2f} км, "
          f"все пролёты {np.linalg.norm(e['все пролёты']):7.2f} км")
(Path(a.out) / f"multipass_{a.obj}.json").write_text(json.dumps(dict(obj=a.obj, passes=[p["name"] for p in passes],
    base_epoch=base.epoch_jd, corrections=x.tolist(), residuals_base=rb.tolist(), residuals_fit=rf.tolist(), predictions=rows),
    ensure_ascii=False, indent=1))

"""Прогноз орбиты: точка со снимка + история публичных орбит (THEORY §25).

  python scripts/orbit_fuse.py [--days 3]
Основа — последний публичный TLE до снимка. Наблюдения для подгонки поправок его элементов (ΔM₀, Δn, Δṅ):
  * публичные TLE за --days суток до снимка — каждое в СВОЮ эпоху (там подгонка публичной орбиты точнее всего),
    σ вдоль 0,5 км (GP) / 0,3 км (SupGP), поперёк 0,2 км — по сверке с снимками (orbit_check);
  * положение со снимка, σ — из orbit_check (десятки метров).
История задаёт снос фазы (среднее движение и торможение), снимок — фазу в момент съёмки. Проверка — прогноз на эпохи
более поздних публичных орбит: было (основа) / только фаза по снимку / снимок + история.
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, WGS72

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import geometry as G, orbits as O, passgeom as PG

ap = argparse.ArgumentParser(); ap.add_argument("--days", type=float, default=3.0); ap.add_argument("--out", default=str(ROOT / "results/orbits"))
a = ap.parse_args()
SIG = {"GP": (500.0, 200.0), "SupGP": (300.0, 200.0)}


def make(base, dM=0.0, dn=0.0, dnd=0.0):
    s = base.satrec; r = Satrec()
    r.sgp4init(WGS72, "i", s.satnum, s.jdsatepoch + s.jdsatepochF - 2433281.5, s.bstar, s.ndot + dnd, s.nddot, s.ecco, s.argpo,
               s.inclo, s.mo + dM, s.no_kozai + dn, s.nodeo)
    return O.Orbit(base.name, r, base.line1, base.line2)


def axes(r, v):
    uv = G.unit(v); ur = G.unit(r); uc = G.unit(np.cross(ur, uv)); return np.stack([uv, uc, ur])


def resid(orb, obs):
    out = []
    for jd, rr, sa, sc in obs:
        r, v = orb.ecef([jd]); A = axes(r[0], v[0]); d = A @ (r[0] - rr) * 1e3
        out += [d[0] / sa, d[1] / sc]
    return np.array(out)


def fit(base, obs, npar):
    x = np.zeros(npar); st = np.array([1e-5, 1e-9, 1e-12])[:npar]
    for _ in range(10):
        f0 = resid(make(base, *x), obs)
        J = np.stack([(resid(make(base, *(x + np.eye(npar)[k] * st[k])), obs) - f0) / st[k] for k in range(npar)], 1)
        dx, *_ = np.linalg.lstsq(J, -f0, rcond=None); x = x + dx
        if np.all(np.abs(dx) < st * 1e-2):
            break
    return make(base, *x)


orbs = PG.Orbits(until="99999999"); rows = []
for f in sorted(Path(a.out).glob("*_2026*.json")):
    d = json.loads(f.read_text())
    if "position_ecef_km" not in d:
        continue
    cat = str(d["norad"]); jd = d["jd"]; rm = np.asarray(d["position_ecef_km"]); sg = np.maximum(np.asarray(d["sigma_m"])[:2], 10.0)
    pool = [("GP", o) for o in orbs.GP.get(cat, [])] + [("SupGP", o) for o in orbs.SG.get(cat, [])]
    hist = [(k, o) for k, o in pool if 0 < jd - o.epoch_jd <= a.days]
    refs = [(k, o) for k, o in pool if 0.05 <= o.epoch_jd - jd <= 3]
    if len(hist) < 2 or not refs:
        continue
    kb, base = max(hist, key=lambda q: q[1].epoch_jd)
    img = [(jd, rm, float(sg[0]), float(sg[1]))]
    hobs = [(o.epoch_jd, o.ecef([o.epoch_jd])[0][0], *SIG[k]) for k, o in hist]
    phase = fit(base, img, 1)                       # только фаза по снимку
    fused = fit(base, img + hobs, 3 if len(hist) >= 3 else 2)
    for kr, o in refs:
        t2 = o.epoch_jd; rr, vr = o.ecef([t2]); A2 = axes(rr[0], vr[0])
        e = {k: float(np.linalg.norm(A2 @ (m.ecef([t2])[0][0] - rr[0]))) for k, m in (("было", base), ("фаза", phase), ("снимок+история", fused))}
        rows.append(dict(name=d["name"], horizon=t2 - jd, hist=len(hist), **e))
out = Path(a.out) / "fuse.json"; out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
print(f"прогнозов {len(rows)}, объектов {len({r['name'] for r in rows})}")
for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 3.0)):
    m = [r for r in rows if lo <= r["horizon"] < hi]
    if m:
        print(f"  горизонт {lo}–{hi} сут (n={len(m)}): медиана ошибки — было {np.median([r['было'] for r in m]):.2f} км, "
              f"фаза по снимку {np.median([r['фаза'] for r in m]):.2f} км, снимок + история {np.median([r['снимок+история'] for r in m]):.2f} км; "
              f"снимок + история лучше «было» в {np.mean([r['снимок+история'] < r['было'] for r in m]) * 100:.0f}%")

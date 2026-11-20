"""Поиск пересечений «веер Sentinel-2 × объект» перебором (прокси Equant, HANDOFF 6.1).

Отбор на сетке в TEME (векторизовано по объектам через SatrecArray), уточнение в ECEF.
"""
from __future__ import annotations

import math

import numpy as np
from sgp4.api import SatrecArray

from .geometry import fan_normal, ray_ellipsoid, ecef_to_lonlat, unit
from .orbits import Orbit, sun_ecef

RE = 6378.137


def _teme_many(orbs, jd):
    whole = np.floor(jd - 0.5) + 0.5
    e, r, v = SatrecArray([o.satrec for o in orbs]).sgp4(whole, jd - whole)
    r[e != 0] = np.nan
    return r, v


def find_crossings(sents: list[Orbit], objs: list[Orbit], jd0: float, jd1: float, step_s=30.0,
                   half_fov_deg=10.5, max_rng=1500.0, chunk=400):
    grid = np.arange(jd0, jd1, step_s / 86400.0)
    hf = math.radians(half_fov_deg)
    S = []
    for s in sents:
        rs, vs = s.teme(grid)
        S.append((s, rs, fan_normal(rs, vs)))
    cands = []
    for k in range(0, len(objs), chunk):
        part = objs[k:k + chunk]
        ro, _ = _teme_many(part, grid)  # (n, t, 3)
        for s, rs, nn in S:
            d = ro - rs[None]
            f = np.einsum("ntk,tk->nt", d, nn)
            dist = np.linalg.norm(d, axis=-1)
            with np.errstate(invalid="ignore"):
                hit = (np.sign(f[:, :-1]) != np.sign(f[:, 1:])) & (dist[:, :-1] < max_rng)
            # грубый фильтр по углу от надира
            for n, i in zip(*np.where(hit)):
                w = f[n, i] / (f[n, i] - f[n, i + 1])  # линейная интерполяция к пересечению
                dl = unit(d[n, i] + w * (d[n, i + 1] - d[n, i]))
                nad = -unit(rs[i] + w * (rs[i + 1] - rs[i]))
                if math.acos(np.clip(dl @ nad, -1, 1)) <= hf + math.radians(3):
                    cands.append((s, part[n], grid[i], grid[i + 1]))
    ev = [e for e in (_refine(s, o, ta, tb, hf) for s, o, ta, tb in cands) if e is not None]
    ev.sort(key=lambda e: e["jd"])
    return ev


def _F(s, o, t):
    rS, vS = s.ecef(t); rO, _ = o.ecef(t)
    return np.einsum("ak,ak->a", rO - rS, fan_normal(rS, vS))


def _refine(s, o, ta, tb, hf):
    t = np.array([ta, tb]); fa, fb = _F(s, o, t)
    if not (np.isfinite(fa) and np.isfinite(fb)) or np.sign(fa) == np.sign(fb):
        return None
    for _ in range(50):
        tm = 0.5 * (ta + tb); fm = _F(s, o, [tm])[0]
        if np.sign(fm) == np.sign(fa): ta, fa = tm, fm
        else: tb = tm
        if (tb - ta) * 86400 < 1e-4: break
    return event(s, o, tm, hf)


def event(s: Orbit, o: Orbit, tm: float, hf=math.radians(10.5)):
    rS, vS = (a[0] for a in s.ecef([tm])); rO, vO = (a[0] for a in o.ecef([tm]))
    dv = rO - rS; rng = float(np.linalg.norm(dv)); nad = -unit(rS)
    if math.acos(np.clip(dv @ nad / rng, -1, 1)) > hf:
        return None
    lam = float(ray_ellipsoid(rS, dv / rng))
    if not np.isfinite(lam) or lam < rng:
        return None
    gp = rS + lam * dv / rng
    lon, lat = ecef_to_lonlat(gp)
    su = sun_ecef([tm])[0]
    sun_el = math.degrees(math.asin(float(su @ unit(gp))))
    proj = float(rO @ su); lit = proj > 0 or np.linalg.norm(rO - proj * su) > RE
    n = fan_normal(rS, vS)
    rSi, vSi = (a[0] for a in s.teme([tm]))
    return dict(sent=s.name, obj=o.name, cat=o.catnr, jd=float(tm), lat=float(lat[0]), lon=float(lon[0]),
                rng=rng, lam=lam, alt=float(np.linalg.norm(rO) - RE), sun_el=sun_el, lit=bool(lit),
                desc=bool(vSi[2] < 0), v_al=float((vO - vS) @ n), pix_obj=10.0 * rng / lam,
                age=float(tm - o.epoch_jd))

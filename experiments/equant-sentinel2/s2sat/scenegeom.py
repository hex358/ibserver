"""Геометрия кадра для любого объекта (THEORY §23–24): орбиты на момент снимка, орбитальные оси, Солнце, визир.

Орбитальные оси (LVLH) — по ИНЕРЦИАЛЬНОЙ скорости: в них определены законы ориентации всех аппаратов. Якобианы полос в
реконструкции (results/shape/*.pkl) записаны в осях по скорости относительно вращающейся Земли; переход между ними —
поворот вокруг радиуса на 2–4° (lvlh_to_fit).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from s2sat import geometry as G, orbits as O, passgeom as PG

ROOT = Path(__file__).resolve().parents[1]
OMEGA = 7.2921159e-5


def ground_utm(rS, p, epsg):
    d = G.unit(p - rS); lam = G.ray_ellipsoid(rS, d)
    X, Y = G.ecef_to_utm((rS + lam * d)[None], epsg)
    return np.array([float(np.ravel(X)[0]), float(np.ravel(Y)[0])])


def event_of(name):
    """Событие обнаружения (event, result) по имени реконструкции."""
    f = sorted(glob.glob(str(ROOT / f"results/found*/{name}.json"))) + sorted(glob.glob(str(ROOT / f"results/events_*/{name}.json")))
    x = json.loads(Path(f[0]).read_text())
    return x["event"], x["result"]


def choose_orbits(e, rc, until=None):
    """TLE Sentinel-2 с эпохой, ближайшей к снимку. Орбита объекта — та же, что в детекции и восстановлении формы
    (источник и возраст записаны в событии: PG.pick_orbit находит её точно), иначе — из TLE объекта в ±6 сут та, что
    воспроизводит дальность реконструкции. Поправки θ (shape_fit) отсчитаны от неё — карточки и кадры должны брать её."""
    orbs = PG.Orbits(until=until or "99999999")
    jd = e["jd"]
    S2 = []
    for sd in (ROOT / "data/tle_snapshots").iterdir():
        f = sd / "sentinel2.tle"
        if f.exists():
            S2 += [o for o in O.parse_tle(f.read_text()) if o.name == e["sent"]]
    sen = min(S2, key=lambda o: abs(o.epoch_jd - jd))
    if e.get("orbit_src") and e.get("orbit_age") is not None:
        o = PG.pick_orbit(orbs, e); age = float(e["orbit_age"])
        if min(abs(o.epoch_jd - (jd - age)), abs(o.epoch_jd - (jd + age))) < 1e-5:
            return sen, o
    pool = [o for o in orbs.SG.get(e["cat"], []) + orbs.GP.get(e["cat"], []) if abs(o.epoch_jd - jd) < 6]
    rS = sen.ecef([jd])[0][0]
    rng = lambda o: float(np.linalg.norm(o.ecef([jd])[0][0] - rS))
    orb = min(pool, key=lambda o: abs(rng(o) - rc["rng"]))
    return sen, orb


def axes(orb, jd, inertial=True):
    """Орбитальные оси (вдоль, поперёк, радиус) в ECEF: строки матрицы."""
    rO, vO = orb.ecef([jd]); rO, vO = rO[0], vO[0]
    ur = G.unit(rO)
    v = vO + (np.cross([0, 0, OMEGA], rO) if inertial else 0.0)
    uv = G.unit(v - (v @ ur) * ur); uc = G.unit(np.cross(ur, uv))
    return np.stack([uv, uc, ur])


def frame_geometry(sen, orb, jd, epsg, inertial=True):
    rO, vO = orb.ecef([jd]); rO = rO[0]
    rS = sen.ecef([jd])[0][0]
    A = axes(orb, jd, inertial)
    su = O.sun_ecef([jd])[0]; su = su / np.linalg.norm(su)
    los = G.unit(rS - rO)
    g0 = ground_utm(rS, rO, epsg)
    h = 0.05
    Jf = np.stack([(ground_utm(rS, rO + A[k] * h, epsg) - g0) / (h * 1000.0) for k in range(3)], 1)
    dt = 0.05 / 86400.0
    gp = ground_utm(sen.ecef([jd + dt])[0][0], orb.ecef([jd + dt])[0][0], epsg)
    gm = ground_utm(sen.ecef([jd - dt])[0][0], orb.ecef([jd - dt])[0][0], epsg)
    vG = (gp - gm) / 0.1
    return dict(A=A, sun=A @ su, los=A @ los, Jf=Jf, vG=vG, g0=g0, rng=float(np.linalg.norm(rS - rO)),
                alt=float(np.linalg.norm(rO) - 6371.0), rS=rS, rO=rO)


def lvlh_to_fit(name, rc):
    """Поворот C: координаты в инерциальных орбитальных осях → в осях реконструкции (скорость относительно Земли)."""
    e, _ = event_of(name)
    sen, orb = choose_orbits(e, rc)
    jd = float(np.median([d["jd"] for d in rc["bands"].values()]))
    return axes(orb, jd, False) @ axes(orb, jd, True).T

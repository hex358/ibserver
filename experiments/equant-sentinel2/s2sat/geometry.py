"""Геометрия push-broom Sentinel-2 в ECEF (HANDOFF 4.1).

Скорость Sentinel в ECEF уже относительно вращающейся Земли, поэтому нормаль веера
строится прямо из неё (в пилоте для этого вычитали ω×r из инерциальной скорости).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from pyproj import Transformer

A84 = 6378.137
B84 = 6356.752314245
R_LOC = 6371.0  # радиус для перевода зенитного угла на земле в надирный на спутнике


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def fan_normal(rS, vS):
    """Единичная проекция скорости на плоскость, перпендикулярную r̂ (направление вдоль трассы)."""
    rh = unit(rS)
    vh = vS - np.sum(vS * rh, axis=-1, keepdims=True) * rh
    return unit(vh)


def band_plane(rS, vS, alpha_rad):
    """Нормаль плоскости полосы с углом α вдоль трассы. rS, vS: (..., 3); alpha: (...)."""
    rh = unit(rS)
    n = fan_normal(rS, vS)
    ct = np.cross(rh, n)
    a = np.asarray(alpha_rad)[..., None]
    look = -rh * np.cos(a) + n * np.sin(a)
    return np.cross(look, ct)


def ray_ellipsoid(p, d):
    """Первое пересечение луча p + λd с WGS84; λ (км), NaN если мимо. p, d: (..., 3)."""
    a2, b2 = A84 ** 2, B84 ** 2
    A = (d[..., 0] ** 2 + d[..., 1] ** 2) / a2 + d[..., 2] ** 2 / b2
    B = 2 * ((p[..., 0] * d[..., 0] + p[..., 1] * d[..., 1]) / a2 + p[..., 2] * d[..., 2] / b2)
    C = (p[..., 0] ** 2 + p[..., 1] ** 2) / a2 + p[..., 2] ** 2 / b2 - 1
    disc = B * B - 4 * A * C
    with np.errstate(invalid="ignore"):
        lam = (-B - np.sqrt(disc)) / (2 * A)
    return np.where(disc >= 0, lam, np.nan)


@lru_cache(maxsize=None)
def _tr(dst: int):
    return Transformer.from_crs(4978, dst, always_xy=True)


def ecef_to_lonlat(r_km):
    r = np.atleast_2d(r_km) * 1000.0
    lon, lat, _ = _tr(4326).transform(r[:, 0], r[:, 1], r[:, 2])
    return np.asarray(lon), np.asarray(lat)


def ecef_to_utm(r_km, epsg: int):
    """Точка на эллипсоиде (ECEF, км) → (x, y) в UTM, м."""
    r = np.atleast_2d(r_km) * 1000.0
    x, y, _ = _tr(epsg).transform(r[:, 0], r[:, 1], r[:, 2])
    return np.asarray(x), np.asarray(y)


@lru_cache(maxsize=None)
def _utm2ll(epsg: int):
    return Transformer.from_crs(epsg, 4326, always_xy=True)


def utm_to_lonlat(x, y, epsg: int):
    return _utm2ll(epsg).transform(x, y)


def track_azimuth(sen, jd):
    """Азимут подспутниковой трассы ψ (рад) по двум положениям с шагом 1 с."""
    r, _ = sen.ecef([jd, jd + 1 / 86400])
    lon, lat = ecef_to_lonlat(r)
    dlon = ((lon[1] - lon[0] + 540) % 360) - 180
    return math.atan2(dlon * math.cos(math.radians(lat[0])), lat[1] - lat[0])


def apparent_track(sen, obj, jd0, alphas_deg, dpos_km=None, half_window_s=120.0, step_s=0.25, iters=40):
    """Видимый трек объекта на ортоснимке.

    Для каждого угла α (градусы вдоль трассы) ищем момент t, когда объект лежит
    в плоскости полосы α, и пересекаем луч Sentinel→объект с WGS84.
    dpos_km: необязательный сдвиг объекта вдоль его скорости, КМ (см. баг-ловушку в HANDOFF 6).
    Возвращает dict: alpha, jd, ground (ECEF км), rng, lam; NaN там, где корня нет.
    """
    al = np.radians(np.asarray(alphas_deg, float))
    grid = jd0 + np.arange(-half_window_s, half_window_s + 1e-9, step_s) / 86400.0

    def state(t):
        rS, vS = sen.ecef(t)
        rO, vO = obj.ecef(t)
        if dpos_km:
            rO = rO + dpos_km * unit(vO)
        return rS, vS, rO

    rS, vS, rO = state(grid)
    F = np.einsum("tk,atk->at", rO - rS, band_plane(rS[None], vS[None], al[:, None]))
    sgn = np.sign(F)
    ch = sgn[:, :-1] != sgn[:, 1:]
    mid = len(grid) // 2
    # ближайшая к jd0 смена знака
    idx = np.where(ch, np.abs(np.arange(len(grid) - 1) - mid), 10 ** 9).argmin(axis=1)
    ok = ch[np.arange(len(al)), idx]
    ta, tb = grid[idx], grid[idx + 1]
    fa = F[np.arange(len(al)), idx]
    for _ in range(iters):
        tm = 0.5 * (ta + tb)
        rS, vS, rO = state(tm)
        fm = np.einsum("ak,ak->a", rO - rS, band_plane(rS, vS, al))
        same = np.sign(fm) == np.sign(fa)
        ta = np.where(same, tm, ta); fa = np.where(same, fm, fa)
        tb = np.where(same, tb, tm)
    tm = 0.5 * (ta + tb)
    rS, vS, rO = state(tm)
    d = rO - rS
    rng = np.linalg.norm(d, axis=-1)
    lam = ray_ellipsoid(rS, d / rng[:, None])
    ground = rS + lam[:, None] * d / rng[:, None]
    bad = ~ok | ~np.isfinite(lam) | (lam < rng)
    ground[bad] = np.nan
    return dict(alpha=np.degrees(al), jd=np.where(bad, np.nan, tm), ground=ground, rng=rng, lam=lam)

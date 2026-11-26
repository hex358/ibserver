"""Углы обзора полос и детекторов из MTD_TL.xml; корни «α трека = α полосы» (HANDOFF 4.1, 6.6)."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from .geometry import R_LOC

# bandId в MTD_TL.xml → имя ассета Earth Search
BID = {"blue": "1", "green": "2", "red": "3", "rededge1": "4", "rededge2": "5", "rededge3": "6",
       "nir": "7", "nir08": "8", "swir16": "11", "swir22": "12"}
BRES = {"blue": 10, "green": 10, "red": 10, "nir": 10, "rededge1": 20, "rededge2": 20, "rededge3": 20,
        "nir08": 20, "swir16": 20, "swir22": 20}
# полосы 60 м: B01 (443 нм) и B09 (945 нм) — в тех же данных L2A; детектор их не берёт, а восстановление формы и вращения —
# берёт: каждая полоса снята в свой момент, это лишние точки во времени (THEORY §25)
BID60 = {"coastal": "0", "nir09": "9"}
BID_ALL = {**BID, **BID60}
BRES.update({"coastal": 60, "nir09": 60})
STEP = 5000.0  # шаг сетки углов, м


def alpha_3d(Zg, Ag, ulx, uly, epsg, sen, jd0):
    """Угол вдоль трассы α (°) для узлов сетки углов в трёхмерной геометрии (THEORY §25).

    Узел — точка P на эллипсоиде; зенит Z и азимут Az (от географического севера в самой точке P, по часовой) задают луч
    от P к спутнику в местной системе ENU точки P. Момент съёмки узла — когда Sentinel (орбита sen, лучше GPS) ближе
    всего к этому лучу; α — угол вдоль трассы того же луча в системе спутника в этот момент (как в band_plane).
    Прежняя формула α = atan(−tan η·cos(Az − ψ)) брала одно направление трассы ψ для всех узлов, а азимут отсчитан от
    местного севера: разница (сходимость меридианов) до ~1° на 100 км от трассы → ошибка α до ~0,1°.
    """
    from pyproj import Transformer
    from .geometry import unit, fan_normal
    nr, nc = Zg.shape
    X = ulx + (np.arange(nc) * STEP); Y = uly - (np.arange(nr) * STEP)
    Xg, Yg = np.meshgrid(X, Y)
    lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(Xg, Yg)
    x, y, z = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True).transform(lon, lat, np.zeros_like(lon))
    P = np.stack([x, y, z], -1) / 1000.0
    la, lo = np.radians(lat), np.radians(lon)
    uE = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], -1)
    uN = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], -1)
    uU = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    Zr, Ar = np.radians(Zg), np.radians(Ag)
    d = (np.sin(Zr) * np.sin(Ar))[..., None] * uE + (np.sin(Zr) * np.cos(Ar))[..., None] * uN + np.cos(Zr)[..., None] * uU
    ok = np.isfinite(d).all(-1)
    out = np.full(Zg.shape, np.nan)
    if not ok.any():
        return out
    Pp, dd = P[ok], d[ok]
    t = np.full(len(Pp), float(jd0))
    for _ in range(6):                      # Ньютон по g(t) = s·v − (s·d)(v·d), s = S(t) − P
        rS, vS = sen.ecef(t); s_ = rS - Pp
        g = np.einsum("ij,ij->i", s_, vS) - np.einsum("ij,ij->i", s_, dd) * np.einsum("ij,ij->i", vS, dd)
        dg = np.einsum("ij,ij->i", vS, vS) - np.einsum("ij,ij->i", vS, dd) ** 2
        t = t - g / dg / 86400.0
    rS, vS = sen.ecef(t)
    rh = unit(rS); n = fan_normal(rS, vS); look = -dd
    out[ok] = np.degrees(np.arctan2(np.einsum("ij,ij->i", look, n), -np.einsum("ij,ij->i", look, rh)))
    return out


def parse_grids(xml_text: str, psi_rad: float, H_km: float, item_id: str = "", sen=None, epsg=None, jd0=None, all60=False):
    """→ {band: [dict(det, g=α° (23×23), ulx, uly, item)]}.

    С sen/epsg/jd0 — α в трёхмерной геометрии (alpha_3d). Без них — прежнее приближение: надирный угол на спутнике
    η = asin(R sin Z / (R+H)), α = atan(−tan η · cos(Az − ψ)).
    """
    root = ET.fromstring(xml_text)
    ulx = uly = None
    for g in root.iter("Geoposition"):
        if g.get("resolution") == "10":
            ulx = float(g.find("ULX").text); uly = float(g.find("ULY").text)
    out = {}
    for grid in root.iter("Viewing_Incidence_Angles_Grids"):
        names = [k for k, v in (BID_ALL if all60 else BID).items() if v == grid.get("bandId")]
        if not names:
            continue

        def rd(tag):
            return np.array([[float(v) for v in row.text.split()]
                             for row in grid.find(tag).find("Values_List").findall("VALUES")])

        Zg, Ag = rd("Zenith"), rd("Azimuth")
        if sen is not None and epsg is not None:
            al = alpha_3d(Zg, Ag, ulx, uly, epsg, sen, jd0)
        else:
            eta = np.arcsin(R_LOC * np.sin(np.radians(Zg)) / (R_LOC + H_km))
            al = np.degrees(np.arctan(-np.tan(eta) * np.cos(np.radians(Ag) - psi_rad)))
        out.setdefault(names[0], []).append(dict(det=int(grid.get("detectorId")), g=al, ulx=ulx, uly=uly, item=item_id))
    return out


def gval(dd, X, Y):
    """Билинейная интерполяция α по сетке 5 км; NaN вне детектора."""
    g = dd["g"]; nr, nc = g.shape
    fr = (dd["uly"] - Y) / STEP; fc = (X - dd["ulx"]) / STEP
    r0 = np.floor(fr).astype(int); c0 = np.floor(fc).astype(int); wr = fr - r0; wc = fc - c0
    ok = (r0 >= 0) & (c0 >= 0) & (r0 < nr - 1) & (c0 < nc - 1)
    acc = np.zeros(np.shape(X)); wac = np.zeros(np.shape(X))
    for dr, dc, w in ((0, 0, (1 - wr) * (1 - wc)), (0, 1, (1 - wr) * wc), (1, 0, wr * (1 - wc)), (1, 1, wr * wc)):
        v = g[np.clip(r0 + dr, 0, nr - 1), np.clip(c0 + dc, 0, nc - 1)]
        m = ok & np.isfinite(v)
        acc[m] += w[m] * v[m]; wac[m] += w[m]
    out = np.full(np.shape(X), np.nan); m = wac > 0.5; out[m] = acc[m] / wac[m]
    return out


_TR = {}


def _to_epsg(X, Y, src, dst):
    if src == dst or not src or not dst:
        return X, Y
    from pyproj import Transformer
    k = (src, dst)
    if k not in _TR:
        _TR[k] = Transformer.from_crs(src, dst, always_xy=True)
    return _TR[k].transform(X, Y)


def band_roots(A, X, Y, DG, bands, min_sep_m=200.0, epsg_main=None):
    """Точки трека (A=α°, X, Y в UTM главной гранулы), где α трека = α сетки полосы. → {band: [(det, α, x, y)]}.
    Сетки гранул другой зоны UTM вычисляются в своих координатах (точки трека пересчитываются), корни — в главной."""
    R = {}
    _cache = {}
    for b in bands:
        lst = []
        for dd in DG.get(b, []):
            ep = dd.get("epsg")
            if epsg_main and ep and ep != epsg_main:
                if ep not in _cache:
                    _cache[ep] = _to_epsg(np.asarray(X), np.asarray(Y), epsg_main, ep)
                Xg, Yg = _cache[ep]
            else:
                Xg, Yg = X, Y
            f = A - gval(dd, np.asarray(Xg), np.asarray(Yg))
            ok = np.isfinite(f[:-1]) & np.isfinite(f[1:]) & (np.sign(f[:-1]) != np.sign(f[1:]))
            for i in np.where(ok)[0]:
                w = f[i] / (f[i] - f[i + 1])
                x = X[i] + w * (X[i + 1] - X[i]); y = Y[i] + w * (Y[i + 1] - Y[i])
                if all(math.hypot(x - q[2], y - q[3]) > min_sep_m for q in lst):
                    lst.append((dd["det"], float(A[i] + w * (A[i + 1] - A[i])), float(x), float(y)))
        if lst:
            R[b] = lst
    return R

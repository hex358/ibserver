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
STEP = 5000.0  # шаг сетки углов, м


def parse_grids(xml_text: str, psi_rad: float, H_km: float, item_id: str = ""):
    """→ {band: [dict(det, g=α° (23×23), ulx, uly, item)]}.

    Zenith в метаданных измерен на земле; надирный угол на спутнике η = asin(R sin Z / (R+H)),
    угол вдоль трассы α = atan(−tan η · cos(Az − ψ)).
    """
    root = ET.fromstring(xml_text)
    ulx = uly = None
    for g in root.iter("Geoposition"):
        if g.get("resolution") == "10":
            ulx = float(g.find("ULX").text); uly = float(g.find("ULY").text)
    out = {}
    for grid in root.iter("Viewing_Incidence_Angles_Grids"):
        names = [k for k, v in BID.items() if v == grid.get("bandId")]
        if not names:
            continue

        def rd(tag):
            return np.array([[float(v) for v in row.text.split()]
                             for row in grid.find(tag).find("Values_List").findall("VALUES")])

        Zg, Ag = rd("Zenith"), rd("Azimuth")
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


def band_roots(A, X, Y, DG, bands, min_sep_m=200.0):
    """Точки трека (A=α°, X, Y в UTM), где α трека = α сетки полосы. → {band: [(det, α, x, y)]}."""
    R = {}
    for b in bands:
        lst = []
        for dd in DG.get(b, []):
            f = A - gval(dd, X, Y)
            ok = np.isfinite(f[:-1]) & np.isfinite(f[1:]) & (np.sign(f[:-1]) != np.sign(f[1:]))
            for i in np.where(ok)[0]:
                w = f[i] / (f[i] - f[i + 1])
                x = X[i] + w * (X[i + 1] - X[i]); y = Y[i] + w * (Y[i + 1] - Y[i])
                if all(math.hypot(x - q[2], y - q[3]) > min_sep_m for q in lst):
                    lst.append((dd["det"], float(A[i] + w * (A[i + 1] - A[i])), float(x), float(y)))
        if lst:
            R[b] = lst
    return R

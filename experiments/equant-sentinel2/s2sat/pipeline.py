"""Полный прогон одного события: гранулы → углы → корни → карты → сдвиг → нуль-тест."""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from . import detect, geometry as G, imagery as I, s2meta as M
from .orbits import Orbit, dt_from_jd

log = logging.getLogger("s2sat")
CELL, PAD = 5000.0, 500.0  # нормировка z по ячейкам 5 км с полями 500 м, как в пилоте


def run_event(sen: Orbit, obj: Orbit, e: dict, Rm=4000.0, minb=3, bands=None, n_null=10,
              ref_ids: dict | None = None, alpha_span=2.6, n_alpha=2000):
    """e: событие из crossings.event(). Возвращает dict с результатом и диагностикой."""
    t0 = time.time()
    out = dict(obj=e["obj"], sent=e["sent"], jd=e["jd"], v_al=e["v_al"], reasons=[])
    t = dt_from_jd(e["jd"])
    its = I.items_around(e["lon"], e["lat"], t, e["sent"][-1])
    if not its:
        out["reasons"].append("no granules"); return out
    inside = [i for i in its if i.bbox[0] <= e["lon"] <= i.bbox[2] and i.bbox[1] <= e["lat"] <= i.bbox[3]] or its
    main = min(inside, key=lambda i: i.properties.get("eo:cloud_cover", 100))
    epsg = I.epsg_of(main)
    out.update(item=main.id, epsg=epsg, cloud=main.properties.get("eo:cloud_cover"), granules=[i.id for i in its])

    # углы полос
    psi = G.track_azimuth(sen, e["jd"])
    H = float(np.linalg.norm(sen.ecef([e["jd"]])[0][0])) - G.R_LOC
    DG = {}
    for it in its:
        if I.epsg_of(it) != epsg:
            continue
        try:
            g = M.parse_grids(I.get_text(it.assets["granule_metadata"].href), psi, H, it.id)
        except Exception as ex:
            out["reasons"].append(f"meta {it.id}: {ex!r}"[:120]); continue
        for b, lst in g.items():
            DG.setdefault(b, []).extend(lst)
    bands = [b for b in (bands or M.BID) if b in DG and b in main.assets]

    # видимый трек и корни
    tr = G.apparent_track(sen, obj, e["jd"], np.linspace(-alpha_span, alpha_span, n_alpha))
    ok = np.isfinite(tr["ground"][:, 0])
    X, Y = G.ecef_to_utm(tr["ground"][ok], epsg)
    A = tr["alpha"][ok]
    roots = M.band_roots(A, X, Y, DG, bands)
    bands = [b for b in bands if b in roots]
    out.update(roots=roots, bands=bands, psi_deg=math.degrees(psi), H=H)
    if len(bands) < minb:
        out["reasons"].append(f"too few bands with roots: {list(roots)}"); return out
    Arep = {b: float(np.median([q[1] for q in roots[b]])) for b in bands}
    kmdeg = float(np.median(np.hypot(np.diff(X), np.diff(Y)) / np.diff(A))) / 1000
    thr = max(0.03, 0.6 / kmdeg)
    out.update(kmdeg=kmdeg, Arep=Arep)

    # область: ячейки 5 км, покрывающие корни ± Rm
    xs = [q[2] for b in bands for q in roots[b]]; ys = [q[3] for b in bands for q in roots[b]]
    i0 = int(math.floor((min(xs) - Rm - 400) / CELL)); i1 = int(math.floor((max(xs) + Rm + 400) / CELL))
    j0 = int(math.floor((min(ys) - Rm - 400) / CELL)); j1 = int(math.floor((max(ys) + Rm + 400) / CELL))
    bounds = (i0 * CELL - PAD, j0 * CELL - PAD, (i1 + 1) * CELL + PAD, (j1 + 1) * CELL + PAD)
    W = int(round((bounds[2] - bounds[0]) / detect.RES)); Hh = int(round((bounds[3] - bounds[1]) / detect.RES))
    cells = [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
    cpx, ppx = int(CELL / detect.RES), int(PAD / detect.RES)

    def cslice(c):
        i, j = c
        r0 = (j1 - j) * cpx; c0 = (i - i0) * cpx
        return slice(r0, r0 + cpx + 2 * ppx), slice(c0, c0 + cpx + 2 * ppx)

    def cinner(c):
        i, j = c
        r0 = (j1 - j) * cpx + ppx; c0 = (i - i0) * cpx + ppx
        return slice(r0, r0 + cpx), slice(c0, c0 + cpx)

    # выбор гранулы для каждой ячейки по покрытию синего канала
    cov = {}
    for it in its:
        try:
            a = I.read_window(it.assets["blue"].href, epsg, bounds)
        except Exception as ex:
            out["reasons"].append(f"read blue {it.id}: {ex!r}"[:120]); continue
        cov[it.id] = {c: float((a[cslice(c)] > 0).mean()) for c in cells}
    byid = {it.id: it for it in its}
    pick = {}
    for c in cells:
        best = max(cov, key=lambda k: (cov[k][c], k == main.id), default=None)
        if best is None or cov[best][c] < 0.05:
            out["reasons"].append(f"cell {c}: nodata"); continue
        pick[c] = best
    used = sorted(set(pick.values()))
    refs = {}
    for k in used:
        rid = (ref_ids or {}).get(k)
        refs[k] = (I.search(ids=[rid])[0] if rid else I.ref_same_orbit(byid[k]))
    out["refs"] = {k: (r.id if r is not None else None) for k, r in refs.items()}

    Z = {b: np.full((Hh, W), np.nan, np.float32) for b in bands}
    RGB = np.zeros((Hh, W, 3), np.float32)
    modes = {}
    for b in bands:
        for k in used:
            ev = I.read_window(byid[k].assets[b].href, epsg, bounds)
            rf = refs[k]
            rv = I.read_window(rf.assets[b].href, epsg, bounds) if (rf is not None and b in rf.assets) else None
            for c, kk in pick.items():
                if kk != k:
                    continue
                z, m = detect.zmap(ev[cslice(c)], None if rv is None else rv[cslice(c)])
                modes.setdefault(b, []).append(m)
                if z is not None:
                    Z[b][cinner(c)] = z[ppx:-ppx, ppx:-ppx]
            if b in ("red", "green", "blue"):
                ch = ("red", "green", "blue").index(b)
                for c, kk in pick.items():
                    if kk == k:
                        RGB[cinner(c) + (ch,)] = ev[cslice(c)][ppx:-ppx, ppx:-ppx]
    out["modes"] = {b: sorted(set(v)) for b, v in modes.items()}
    E = detect.exclusive_maps(Z, Arep, thr)
    mos = detect.Raster(E, bounds)
    res = detect.shift_search(roots, mos, bands, Rm, minb=minb)
    nulls = detect.null_test(roots, mos, bands, Rm, n=n_null, minb=minb)
    per = {b: max(float(np.nan_to_num(mos.get(b, np.array([x + res["sx"]]), np.array([y + res["sy"]]))[0], nan=-1))
                  for (_d, _a, x, y) in roots[b]) for b in bands}
    out.update(res, nulls=nulls, null_max=max(nulls), null_med=float(np.median(nulls)), per_band=per,
               bounds=bounds, seconds=time.time() - t0)
    out["_maps"] = dict(Z=Z, E=E, RGB=RGB, track=(A, X, Y))
    return out

"""Полный прогон одного события: гранулы → углы → корни → карты по ячейкам → поиск → нуль-тест.

Поиск двухуровневый:
  * фаза вдоль орбиты dM (км, ±dM_max) — ручной аналог фазовых интервалов Equant:
    ошибка TLE в основном вдоль орбиты, а на снимке она растягивается в λ/rng ≈ 2–2.5 раза;
  * для каждой фазы — общий 2D-сдвиг на снимке в пределах ±Rm (поперечная/радиальная ошибка).
Данные читаются только в ячейках 5 км вдоль полосы возможных положений.
"""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import maximum_filter

from . import detect, geometry as G, imagery as I, s2meta as M
from .orbits import Orbit, dt_from_jd

log = logging.getLogger("s2sat")
CELL, PAD = 5000.0, 500.0
NIN, NPAD = int(CELL / detect.RES), int(PAD / detect.RES)


class CellRaster:
    """Карты E по ячейкам 5 км (внутренняя часть 250×250 при 20 м)."""

    def __init__(self, cells: dict, bands):
        ks = list(cells)
        ci = [k[0] for k in ks]; cj = [k[1] for k in ks]
        self.i0, self.j0 = min(ci), min(cj)
        self.M = -np.ones((max(ci) - self.i0 + 1, max(cj) - self.j0 + 1), int)
        for n, k in enumerate(ks):
            self.M[k[0] - self.i0, k[1] - self.j0] = n
        self.st = {b: np.stack([cells[k]["E"][b] for k in ks]) for b in bands}
        self.cells = cells; self.keys = ks

    def get(self, b, X, Y):
        I_ = np.floor(X / CELL).astype(int); J_ = np.floor(Y / CELL).astype(int)
        i = I_ - self.i0; j = J_ - self.j0
        ok = (i >= 0) & (i < self.M.shape[0]) & (j >= 0) & (j < self.M.shape[1])
        t = np.full(np.shape(X), -1); t[ok] = self.M[i[ok], j[ok]]; ok &= t >= 0
        col = np.floor((X - I_ * CELL) / detect.RES).astype(int)
        row = np.floor(((J_ + 1) * CELL - Y) / detect.RES).astype(int)
        col = np.clip(col, 0, NIN - 1); row = np.clip(row, 0, NIN - 1)
        V = np.full(np.shape(X), np.nan, np.float32)
        V[ok] = self.st[b][t[ok], row[ok], col[ok]]
        return V


def _roots_vs_phase(sen, obj, e, DG, bands, epsg, dms, alpha_span, n_alpha):
    """Корни по полосам/детекторам для набора фаз dM (км). → {(b, det): (n_dm, 3) [α, x, y]}."""
    R = {}
    track0 = None
    for k, dm in enumerate(dms):
        tr = G.apparent_track(sen, obj, e["jd"], np.linspace(-alpha_span, alpha_span, n_alpha), dpos_km=float(dm))
        ok = np.isfinite(tr["ground"][:, 0])
        if ok.sum() < 10:
            continue
        X, Y = G.ecef_to_utm(tr["ground"][ok], epsg); A = tr["alpha"][ok]
        if abs(dm) < 1e-9 or track0 is None:
            track0 = (A, X, Y)
        for b, lst in M.band_roots(A, X, Y, DG, bands).items():
            for det, al, x, y in lst:
                key = (b, det)
                if key not in R:
                    R[key] = np.full((len(dms), 3), np.nan)
                if np.isnan(R[key][k, 0]):
                    R[key][k] = (al, x, y)
    return R, track0


def _interp(R, dms, fine):
    out = {}
    for key, arr in R.items():
        ok = np.isfinite(arr[:, 1])
        if ok.sum() == 0:
            continue
        if ok.sum() == 1:
            v = np.full((len(fine), 3), np.nan)
            v[np.isclose(fine, dms[ok][0])] = arr[ok][0]
            out[key] = v; continue
        v = np.stack([np.interp(fine, dms[ok], arr[ok, c], left=np.nan, right=np.nan) for c in range(3)], 1)
        out[key] = v
    return out


def _roots_at(RF, n):
    roots = {}
    for (b, det), v in RF.items():
        if np.isfinite(v[n, 1]):
            roots.setdefault(b, []).append((det, float(v[n, 0]), float(v[n, 1]), float(v[n, 2])))
    return roots


def run_event(sen: Orbit, obj: Orbit, e: dict, Rm=4000.0, minb=3, bands=None, n_null=10,
              ref_ids: dict | None = None, alpha_span=2.6, n_alpha=2000, max_cloud=None,
              dM_max=0.0, dM_step=0.5, workers=8):
    """e: событие из crossings.event(). dM_max (км): диапазон перебора фазы вдоль орбиты."""
    t0 = time.time()
    out = dict(obj=e["obj"], sent=e["sent"], jd=e["jd"], v_al=e["v_al"], reasons=[])
    t = dt_from_jd(e["jd"])
    its = I.items_around(e["lon"], e["lat"], t, e["sent"][-1], d=0.6 + 0.012 * dM_max)
    if not its:
        out["reasons"].append("no granules"); return out
    inside = [i for i in its if i.bbox[0] <= e["lon"] <= i.bbox[2] and i.bbox[1] <= e["lat"] <= i.bbox[3]] or its
    main = min(inside, key=lambda i: i.properties.get("eo:cloud_cover", 100))
    epsg = I.epsg_of(main)
    out.update(item=main.id, epsg=epsg, cloud=main.properties.get("eo:cloud_cover"), granules=[i.id for i in its])
    if max_cloud is not None and (out["cloud"] or 0) > max_cloud:
        out["reasons"].append(f"cloud {out['cloud']:.0f}% > {max_cloud}"); return out

    # углы полос (гранулы той же зоны UTM)
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

    # корни как функция фазы
    n_coarse = max(1, int(round(2 * dM_max / 4.0)) + 1) if dM_max > 0 else 1
    dms = np.linspace(-dM_max, dM_max, n_coarse) if dM_max > 0 else np.array([0.0])
    fine = np.arange(-dM_max, dM_max + 1e-9, dM_step) if dM_max > 0 else np.array([0.0])
    R, track0 = _roots_vs_phase(sen, obj, e, DG, bands, epsg, dms, alpha_span, n_alpha)
    RF = _interp(R, dms, fine)
    k0 = int(np.argmin(np.abs(fine)))
    roots0 = _roots_at(RF, k0)
    bands_r = sorted({b for (b, _d) in RF}, key=list(M.BID).index)
    bands = [b for b in bands if b in bands_r]
    out.update(roots=roots0, bands=bands, psi_deg=math.degrees(psi), H=H, dM_grid=(float(fine[0]), float(fine[-1]), dM_step))
    if len(bands) < minb:
        out["reasons"].append(f"too few bands with roots: {bands}"); return out
    allr = [v for v in RF.values()]
    Arep = {b: float(np.nanmedian(np.concatenate([RF[k][:, 0] for k in RF if k[0] == b]))) for b in bands}
    if track0 is not None:
        A, X, Y = track0
        kmdeg = float(np.median(np.hypot(np.diff(X), np.diff(Y)) / np.diff(A))) / 1000
    else:
        kmdeg = 10.0
    thr = max(0.03, 0.6 / kmdeg)
    out.update(kmdeg=kmdeg, Arep=Arep)

    # ячейки вдоль полосы возможных положений
    cells = set()
    rad = Rm + 400
    for v in allr:
        pts = v[np.isfinite(v[:, 1])]
        for _a, x, y in pts:
            for xx in np.arange(x - rad, x + rad + 1, 2500):
                for yy in np.arange(y - rad, y + rad + 1, 2500):
                    cells.add((int(math.floor(xx / CELL)), int(math.floor(yy / CELL))))
    cells = sorted(cells)
    out["n_cells"] = len(cells)

    byid = {it.id: it for it in its}
    refcache = {}

    def ref_for(it):
        if it.id not in refcache:
            rid = (ref_ids or {}).get(it.id)
            refcache[it.id] = I.search(ids=[rid])[0] if rid else I.ref_same_orbit(it)
        return refcache[it.id]

    def process(c):
        i, j = c
        bb = (i * CELL - PAD, j * CELL - PAD, (i + 1) * CELL + PAD, (j + 1) * CELL + PAD)
        lon, lat = G.utm_to_lonlat((i + 0.5) * CELL, (j + 0.5) * CELL, epsg)
        cands = [it for it in its if it.bbox[0] - 0.05 <= lon <= it.bbox[2] + 0.05 and it.bbox[1] - 0.05 <= lat <= it.bbox[3] + 0.05]
        cands.sort(key=lambda it: (it.id != main.id, math.hypot(lon - (it.bbox[0] + it.bbox[2]) / 2, lat - (it.bbox[1] + it.bbox[3]) / 2)))
        best = None
        for it in cands[:3]:
            try:
                a = I.read_window(it.assets["blue"].href, epsg, bb)
            except Exception as ex:
                continue
            cov = float((a > 0).mean())
            if best is None or cov > best[1]:
                best = (it, cov, a)
            if cov > 0.97:
                break
        if best is None:
            return c, None, f"nofetch ({len(cands)} cands)"
        if best[1] < 0.05:
            return c, None, "nodata"
        it, cov, a_blue = best
        rf = ref_for(it)
        Z = {}
        for b in bands:
            ev = a_blue if b == "blue" else I.read_window(it.assets[b].href, epsg, bb)
            rv = I.read_window(rf.assets[b].href, epsg, bb) if (rf is not None and b in rf.assets) else None
            z, _m = detect.zmap(ev, rv)
            Z[b] = z if z is not None else np.full(ev.shape, np.nan, np.float32)
            if b == "red": red = ev
            if b == "green": green = ev
        E = detect.exclusive_maps(Z, Arep, thr)
        sl = (slice(NPAD, NPAD + NIN), slice(NPAD, NPAD + NIN))
        rgb = np.stack([red[sl], green[sl], a_blue[sl]], -1) if "red" in bands and "green" in bands else None
        return c, dict(E={b: E[b][sl].astype(np.float16) for b in bands},
                       Z={b: Z[b][sl].astype(np.float16) for b in ("blue", "green", "red", "nir") if b in Z},
                       RGB=rgb, item=it.id, ref=None if rf is None else rf.id, cov=cov), "ok"

    tiles, fails = {}, []
    with ThreadPoolExecutor(workers) as ex:
        for c, T, why in ex.map(lambda c: _safe(process, c), cells):
            if T is None:
                fails.append((c, why))
            else:
                tiles[c] = T
    out["cells_ok"] = len(tiles)
    out["cell_fail"] = fails[:10]
    out["refs"] = sorted({T["ref"] for T in tiles.values()} - {None})
    if len(tiles) < 1:
        out["reasons"].append("no cells"); return out
    mos = CellRaster(tiles, bands)

    def scan(disp=None):
        best = None
        for n in range(len(fine)):
            roots = _roots_at(RF, n)
            if len([b for b in bands if b in roots]) < minb:
                continue
            r = detect.shift_search(roots, mos, bands, Rm, disp=disp, minb=minb)
            if best is None or r["score"] > best["score"]:
                best = dict(r, dM=float(fine[n]), n=n)
        return best

    res = scan()
    rng = np.random.default_rng(5); nulls = []
    for _ in range(n_null):
        disp = {}
        for b in bands:
            r = rng.uniform(1000, 4000); th = rng.uniform(0, 2 * math.pi); disp[b] = (r * math.cos(th), r * math.sin(th))
        nr = scan(disp)
        nulls.append(nr["score"] if nr else -1e9)
    rootsb = _roots_at(RF, res["n"])
    per = {b: max(float(np.nan_to_num(mos.get(b, np.array([x + res["sx"]]), np.array([y + res["sy"]]))[0], nan=-1))
                  for (_d, _a, x, y) in rootsb[b]) for b in bands if b in rootsb}
    out.update(res, roots=rootsb, nulls=nulls, null_max=max(nulls), null_med=float(np.median(nulls)), per_band=per,
               seconds=time.time() - t0)
    out["_maps"] = dict(mos=mos, tiles=tiles)
    return out


def _safe(f, c):
    try:
        return f(c)
    except Exception as ex:
        return c, None, "exc " + repr(ex)[:100]

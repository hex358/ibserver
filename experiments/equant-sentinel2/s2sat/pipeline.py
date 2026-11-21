"""Полный прогон одного события: гранулы → углы → корни → карты по ячейкам → поиск → нуль-тест.

Поиск двухуровневый:
  * фаза вдоль орбиты dM (км, ±dM_max) — ручной аналог фазовых интервалов Equant:
    ошибка TLE в основном вдоль орбиты, а на снимке она растягивается в λ/rng ≈ 2–2.5 раза;
  * для каждой фазы — общий 2D-сдвиг на снимке в пределах ±Rm (поперечная/радиальная ошибка).
Данные читаются только в ячейках 5 км вдоль полосы возможных положений.
"""
from __future__ import annotations

import logging
import os
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import maximum_filter

from . import bgsub, detect, geometry as G, imagery as I, s2meta as M
from .orbits import Orbit, dt_from_jd

log = logging.getLogger("s2sat")
CELL, PAD = 5000.0, 500.0
NIN, NPAD = int(CELL / detect.RES), int(PAD / detect.RES)


class CellRaster:
    """Карты E по ячейкам 5 км (внутренняя часть 250×250 при 20 м)."""

    def __init__(self, cells: dict, bands, key="E"):
        ks = list(cells)
        ci = [k[0] for k in ks]; cj = [k[1] for k in ks]
        self.i0, self.j0 = min(ci), min(cj)
        self.M = -np.ones((max(ci) - self.i0 + 1, max(cj) - self.j0 + 1), int)
        for n, k in enumerate(ks):
            self.M[k[0] - self.i0, k[1] - self.j0] = n
        self.st = {b: np.stack([cells[k][key][b] for k in ks]) for b in bands}
        self.res = {b: CELL / self.st[b].shape[1] for b in bands}
        self.cells = cells; self.keys = ks

    def get(self, b, X, Y):
        I_ = np.floor(X / CELL).astype(int); J_ = np.floor(Y / CELL).astype(int)
        i = I_ - self.i0; j = J_ - self.j0
        ok = (i >= 0) & (i < self.M.shape[0]) & (j >= 0) & (j < self.M.shape[1])
        t = np.full(np.shape(X), -1); t[ok] = self.M[i[ok], j[ok]]; ok &= t >= 0
        res = self.res[b]; n = self.st[b].shape[1]
        col = np.floor((X - I_ * CELL) / res).astype(int)
        row = np.floor(((J_ + 1) * CELL - Y) / res).astype(int)
        col = np.clip(col, 0, n - 1); row = np.clip(row, 0, n - 1)
        V = np.full(np.shape(X), np.nan, np.float32)
        V[ok] = self.st[b][t[ok], row[ok], col[ok]]
        return V


def run_event(sen: Orbit, obj: Orbit, e: dict, Rm=4000.0, minb=3, bands=None, n_null=10,
              ref_ids: dict | None = None, alpha_span=2.6, n_alpha=2000, max_cloud=None,
              dM_max=0.0, dM_step=0.5, workers=4, inject=None, n_refs=3, native=True, method="glrt", step=None, use_ref=False, tol_m=30.0, model="occlusion", area_m2=None, early_thr=6.0):
    """e: событие из crossings.event(). dM_max (км): диапазон перебора фазы вдоль орбиты."""
    t0 = time.time()
    out = dict(obj=e["obj"], sent=e["sent"], jd=e["jd"], v_al=e["v_al"], reasons=[])
    t = dt_from_jd(e["jd"])
    its = I.items_around(e["lon"], e["lat"], t, e["sent"][-1], d=0.8 + 0.012 * dM_max)
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

    # корни как функция фазы. Для медленных попутных пролётов сдвиг на снимке на 1 км ошибки
    # вдоль орбиты в разы больше, чем для встречных, поэтому шаг подбирается по фактическому
    # «усилению» (м на снимке / км фазы), чтобы сдвиг за шаг был не больше Rm.
    def roots_at_dm(dm):
        tr = G.apparent_track(sen, obj, e["jd"], np.linspace(-alpha_span, alpha_span, n_alpha), dpos_km=float(dm))
        ok = np.isfinite(tr["ground"][:, 0])
        if ok.sum() < 10:
            return {}, None
        X, Y = G.ecef_to_utm(tr["ground"][ok], epsg); A = tr["alpha"][ok]
        return M.band_roots(A, X, Y, DG, bands), (A, X, Y)

    roots0, track0 = roots_at_dm(0.0)
    gain = None
    if dM_max > 0 and track0 is not None:
        g0 = G.apparent_track(sen, obj, e["jd"], [0.0])["ground"]
        g1 = G.apparent_track(sen, obj, e["jd"], [0.0], dpos_km=1.0)["ground"]
        gain = float(np.linalg.norm(g1 - g0)) * 1000.0  # м на снимке на 1 км фазы
        dm_step = min(dM_step, max(0.05, 0.9 * Rm / max(gain, 1.0)))
        fine = np.arange(-dM_max, dM_max + 1e-9, dm_step)
        fine = np.unique(np.r_[fine, 0.0])
    else:
        fine = np.array([0.0])
    ROOTS = [roots0 if abs(dm) < 1e-12 else roots_at_dm(dm)[0] for dm in fine]
    k0 = int(np.argmin(np.abs(fine)))
    RF = ROOTS  # для _inject
    k_inj = int(np.argmin(np.abs(fine - (inject or {}).get("dm", 0.0))))
    bands_r = {b for rr in ROOTS for b in rr}
    bands = [b for b in bands if b in bands_r]
    out.update(roots0=roots0, roots=roots0, bands=bands, psi_deg=math.degrees(psi), H=H, gain_m_per_km=gain,
               dM_grid=(float(fine[0]), float(fine[-1]), len(fine)))
    if len(bands) < minb:
        out["reasons"].append(f"too few bands with roots: {bands}"); return out
    Arep = {b: float(np.median([q[1] for rr in ROOTS for q in rr.get(b, [])])) for b in bands}
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
    for rr in ROOTS:
        pts = [(q[1], q[2], q[3]) for lst in rr.values() for q in lst]
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
            refcache[it.id] = [I.search(ids=[rid])[0]] if rid else (I.ref_candidates(it, n=n_refs) or [])
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
        rfs = ref_for(it)
        Z, AZ20, RGB = {}, {}, {}
        for b in bands:
            res = M.BRES[b] if native else 20
            if res == 20 and b == "blue":
                ev = a_blue
            else:
                ev = I.read_window(it.assets[b].href, epsg, bb, res=res)
            if inject:
                ev = _inject(ev, b, bb, inject, RF, k_inj, res)
            rv = _ref_median(ev, [I.read_window(r.assets[b].href, epsg, bb, res=res) for r in rfs if b in r.assets])
            z, _m = detect.zmap(ev, rv, k=13 if res == 10 else 7)
            Z[b] = z if z is not None else np.full(ev.shape, np.nan, np.float32)
            a = np.abs(np.nan_to_num(Z[b]))
            AZ20[b] = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).max(axis=(1, 3)) if res == 10 else a
            if b in ("red", "green", "blue"):
                RGB[b] = ev
        E = {}
        for b in bands:
            far = [c for c in bands if abs(Arep[c] - Arep[b]) > thr]
            F = maximum_filter(np.max(np.stack([AZ20[c] for c in far]), 0), size=7) if far else np.zeros_like(AZ20[b])
            if Z[b].shape[0] != F.shape[0]:
                F = np.kron(F, np.ones((2, 2), np.float32))
            ex = maximum_filter(np.maximum(0.0, np.abs(np.nan_to_num(Z[b])) - F), size=5 if Z[b].shape[0] != AZ20[b].shape[0] else 3).astype(np.float32)
            ex[np.isnan(Z[b])] = np.nan
            E[b] = ex

        def inner(A):
            k = int(round(A.shape[0] * PAD / (CELL + 2 * PAD)))
            return A[k:A.shape[0] - k, k:A.shape[1] - k]
        rgb = np.stack([inner(RGB[c]) for c in ("red", "green", "blue")], -1) if len(RGB) == 3 else None
        return c, dict(E={b: inner(E[b]).astype(np.float16) for b in bands},
                       Z={b: inner(Z[b]).astype(np.float16) for b in ("blue", "green", "red", "nir") if b in Z},
                       RGB=rgb, item=it.id, ref=",".join(r.id for r in rfs) or None, cov=cov), "ok"

    tol_px = int(round(tol_m / 10.0))
    # банк ядер: круглое + отрезки поперёк трассы (сжатое push-broom изображение, THEORY.md §3)
    if track0 is not None:
        _A, _X, _Y = track0; _k = len(_X) // 2
        ang_track = math.atan2(_Y[min(_k + 5, len(_Y) - 1)] - _Y[_k], _X[min(_k + 5, len(_X) - 1)] - _X[_k])
    else:
        ang_track = 0.0
    # форма изображения (THEORY.md §3, калибровка E3b): поперёк — отрезки; вдоль — штрих длины L_вдоль
    geo = image_geometry(e, area_m2)
    La = geo["along_px"]
    KBANK = [bgsub.KERNEL] + [bgsub.line_kernel(L, ang_track + math.pi / 2) for L in (3, 5, 7)]
    for L in sorted({max(3, int(round(La / 2))), max(3, int(round(La)))}):
        if L > 3:
            KBANK.append(bgsub.line_kernel(min(L, 25), ang_track))
    # допуск: вдоль трассы — половина штриха (+20 м), поперёк — ±tol_m
    # вдоль трассы ошибка угла полосы δα ≈ 0.002–0.003° (калибровка: 31550 ±20 м при 12 км/°, 30580 ±100 м при 33 км/°)
    # переходит в сдвиг kmdeg·δα — у медленных пролётов в разы больше
    tol_along_px = max(tol_px, min(La, 25) / 2 + 2, DALPHA_DEG * kmdeg * 1000.0 / 10.0)
    FP = bgsub.rect_footprint(tol_along_px, tol_px, ang_track)
    out["tol_along_m"] = tol_along_px * 10.0
    out["image_geometry"] = geo

    def process_glrt(c):
        """Все полосы на общей сетке 10 м; фон — регрессией по другим полосам того же снимка."""
        i, j = c
        bb = (i * CELL - PAD, j * CELL - PAD, (i + 1) * CELL + PAD, (j + 1) * CELL + PAD)
        lon, lat = G.utm_to_lonlat((i + 0.5) * CELL, (j + 0.5) * CELL, epsg)
        cands = [it for it in its if it.bbox[0] - 0.05 <= lon <= it.bbox[2] + 0.05 and it.bbox[1] - 0.05 <= lat <= it.bbox[3] + 0.05]
        cands.sort(key=lambda it: (it.id != main.id, math.hypot(lon - (it.bbox[0] + it.bbox[2]) / 2, lat - (it.bbox[1] + it.bbox[3]) / 2)))
        best = None
        for it in cands[:3]:
            try:
                a = I.read_window(it.assets["blue"].href, epsg, bb, res=10)
            except Exception:
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
        rfs = ref_for(it) if use_ref else []
        IM = {}
        for b in bands:
            ev = a_blue if b == "blue" else I.read_window(it.assets[b].href, epsg, bb, res=10)
            if inject:
                ev = _inject(ev, b, bb, inject, RF, k_inj, 10)
            IM[b] = ev
        valid = np.all(np.stack([IM[b] > 0 for b in bands]), 0)
        if "scl" in it.assets:  # маска облаков/теней/насыщения из L2A, с запасом на параллакс облаков
            from scipy.ndimage import binary_dilation
            scl = I.read_window(it.assets["scl"].href, epsg, bb, res=10, resampling="nearest")
            bad = np.isin(scl.astype(int), (1, 3, 8, 9, 10))
            valid &= ~binary_dilation(bad, iterations=20)
        Y, S = {}, {}
        for b in bands:
            near = {c2 for c2 in bands if abs(Arep[c2] - Arep[b]) * kmdeg < 0.3}
            regs = [IM[c2] for c2 in bands if c2 not in near]
            if rfs:
                rv = _ref_median(IM[b], [I.read_window(r.assets[b].href, epsg, bb, res=10) for r in rfs if b in r.assets])
                if rv is not None:
                    regs.append(rv)
            R = bgsub.predict_band(IM[b], regs, valid)
            Y[b], S[b] = bgsub.matched_bank(R, KBANK)

        Uc = bgsub.remove_common_mode({b: Y[b] / S[b] for b in bands})
        # яркость фона в каждой полосе (для модели заслонения: отклик = a − f·G_b)
        from scipy.ndimage import median_filter as _mf
        Gb = {}
        for b in bands:  # яркость фона: медиана по блокам 100 м (векторно), без медианного фильтра
            gm, _gs, gshp = bgsub.block_median_mad(np.where(IM[b] > 0, IM[b], np.nan).astype(np.float32), 10, min_n=20)
            Gb[b] = bgsub._unblocks(gm, gshp, 10)

        def inner(A):
            k = int(round(A.shape[0] * PAD / (CELL + 2 * PAD)))
            return A[k:A.shape[0] - k, k:A.shape[1] - k]
        rgb = np.stack([inner(IM[c2]) for c2 in ("red", "green", "blue")], -1).astype(np.float16) if all(c2 in IM for c2 in ("red", "green", "blue")) else None
        def blk(A, n):  # константа блока: храним n×n значений вместо полного разрешения
            I_ = inner(A); f = I_.shape[0] // n
            return I_[f // 2::f, f // 2::f][:n, :n].astype(np.float32)
        return c, dict(S={b: blk(S[b], 10) for b in bands},
                       G={b: blk(Gb[b], 50) for b in bands},
                       Z={b: inner(Y[b] / S[b]).astype(np.float16) for b in bands},
                       U={b: inner(bgsub.standardize(bgsub.local_extreme_fp(Uc[b], FP))).astype(np.float16) for b in bands},
                       RGB=rgb, item=it.id, ref=",".join(r.id for r in rfs) or None, cov=cov), "ok"

    tiles, fails = {}, []
    fn = process_glrt if method == "glrt" else process

    def run_cells(cs):
        global _FN
        _FN = fn
        if method == "glrt" and len(cs) > 2:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(min(workers, os.cpu_count() or 4)) as pool:
                it = pool.imap_unordered(_call, cs)
                for n_done, (c, T, why) in enumerate(it, 1):
                    if n_done % 20 == 0 or n_done == len(cs):
                        print(f"  cells {len(tiles) + len(fails) + 1}/{len(cells)} ({time.time() - t0:.0f}s)", flush=True)
                    (fails.append((c, why)) if T is None else tiles.__setitem__(c, T))
        else:
            with ThreadPoolExecutor(workers) as ex:
                for c, T, why in ex.map(lambda c: _safe(fn, c), cs):
                    (fails.append((c, why)) if T is None else tiles.__setitem__(c, T))

    # сначала ячейки в точке номинальной орбиты: ранний прогноз SNR (THEORY.md §6)
    probe = sorted({(int(math.floor(q[2] / CELL)), int(math.floor(q[3] / CELL))) for lst in roots0.values() for q in lst} & set(cells))
    run_cells(probe)
    if method == "glrt" and area_m2 and tiles:
        _mS, _mG = CellRaster(tiles, bands, "S"), CellRaster(tiles, bands, "G")
        fc0 = forecast(e, roots0, bands, _mS, _mG, area_m2)
        out["forecast"] = fc0
        if fc0 is not None and fc0["T_pred"] < early_thr:
            out["early_exit"] = True
            out["reasons"].append(f"too faint: forecast T {fc0['T_pred']:.1f} < {early_thr}")
            return out
    run_cells([c for c in cells if c not in set(probe)])
    out["cells_ok"] = len(tiles)
    out["cell_fail"] = fails[:10]
    out["refs"] = sorted({T["ref"] for T in tiles.values()} - {None})
    out["refs"] = out["refs"][:5]
    if len(tiles) < 1:
        out["reasons"].append("no cells"); return out
    mos = CellRaster(tiles, bands, "U" if method == "glrt" else "E")
    if method == "glrt":
        mosS = CellRaster(tiles, bands, "S")
        mosU = CellRaster(tiles, bands, "U")
        mosG = CellRaster(tiles, bands, "G")

    def scan(disp=None, prof=None):
        best = None
        need = minb   # частичная видимость: искать при ≥ minb полос (нуль-тест с тем же правилом)
        for n in range(len(fine)):
            roots = ROOTS[n]
            if len([b for b in bands if b in roots]) < need:
                continue
            if method == "glrt":
                if model == "occlusion":
                    r = detect.occlusion_search(roots, mosU, mosS, mosG, bands, Rm, step=step or 20.0, disp=disp, minb=need)
                else:
                    r = detect.coherent_search_u(roots, mosU, bands, Rm, step=step or 20.0, disp=disp, minb=need)
            else:
                r = detect.shift_search(roots, mos, bands, Rm, disp=disp, minb=need)
            if prof is not None:
                prof[n] = r["score"]
            if best is None or r["score"] > best["score"]:
                best = dict(r, dM=float(fine[n]), n=n)
        return best

    profile = np.full(len(fine), np.nan)
    res = scan(prof=profile)
    null_profiles = []
    rng = np.random.default_rng(5); nulls = []
    for _ in range(n_null):
        disp = {}
        for b in bands:
            r = rng.uniform(1000, 4000); th = rng.uniform(0, 2 * math.pi); disp[b] = (r * math.cos(th), r * math.sin(th))
        npf = np.full(len(fine), np.nan); nr = scan(disp, prof=npf); null_profiles.append(npf.tolist())
        nulls.append(nr["score"] if nr else -1e9)
    rootsb = ROOTS[res["n"]]
    if method == "glrt":
        mosZ = CellRaster(tiles, bands, "Z")
        per = {b: max((float(np.nan_to_num(mosZ.get(b, np.array([x + res["sx"]]), np.array([y + res["sy"]]))[0], nan=0))
                       for (_d, _a, x, y) in rootsb[b]), key=abs) for b in bands if b in rootsb}
        per_dn = {b: round(per[b] * float(np.nan_to_num(mosS.get(b, np.array([rootsb[b][0][2] + res["sx"]]), np.array([rootsb[b][0][3] + res["sy"]]))[0], nan=0)), 1)
                  for b in per}
        out["per_band_dn"] = per_dn
    else:
        per = {b: max(float(np.nan_to_num(mos.get(b, np.array([x + res["sx"]]), np.array([y + res["sy"]]))[0], nan=-1))
                      for (_d, _a, x, y) in rootsb[b]) for b in bands if b in rootsb}
    out.update(profile_dm=[float(x) for x in fine], profile=profile.tolist(), null_profiles=null_profiles)
    out.update(res, roots=rootsb, nulls=nulls, null_max=max(nulls), null_med=float(np.median(nulls)), per_band=per,
               seconds=time.time() - t0)
    if method == "glrt":
        out["forecast"] = forecast(e, ROOTS[k0], bands, mosS, mosG, area_m2)
    out["roots_all"] = [{b: [list(q) for q in lst] for b, lst in rr.items()} for rr in ROOTS]
    out["_maps"] = dict(mos=mos, tiles=tiles)
    return out


def _ref_median(ev, refs):
    """Медиана опорных снимков, каждый приведён к яркости события своим коэффициентом."""
    v = ev > 0; out = []
    for rv in refs:
        m = v & (rv > 0)
        if m.mean() < 0.6:
            continue
        out.append(np.where(rv > 0, rv * np.median(ev[m] / rv[m]), np.nan))
    if not out:
        return None
    med = np.nanmedian(np.stack(out), 0)
    return np.nan_to_num(med, nan=0.0).astype(np.float32)


def _inject(ev, b, bb, inj, RF, k0, res=20):
    """Синтетика: прямоугольник inj['len_px']×inj['wid_px'] (20 м) с добавкой inj['amp'] DN
    в корне полосы b (фаза dM=0) плюс сдвиг inj['shift'] (м)."""
    ev = ev.copy()
    sx, sy = inj.get("shift", (0.0, 0.0))
    for (_det, _a, x0, y0) in RF[k0].get(b, []):
        x, y = x0 + sx, y0 + sy
        col = int((x - bb[0]) // res); row = int((bb[3] - y) // res)
        f = 20 // res
        L, W = inj.get("len_px", 3) * f, inj.get("wid_px", 2) * f
        if res == 10 and inj.get("len10"):  # размеры прямо в пикселях 10 м
            L, W = inj["len10"], inj["wid10"]
        r0, c0 = row - W // 2, col - L // 2
        if 0 <= r0 and r0 + W <= ev.shape[0] and 0 <= c0 and c0 + L <= ev.shape[1]:
            m = ev[r0:r0 + W, c0:c0 + L] > 0
            if inj.get("rho_dn") is not None:  # модель заслонения: объект закрывает долю f пикселя
                f = inj.get("f", 1.0)
                ev[r0:r0 + W, c0:c0 + L][m] = (1 - f) * ev[r0:r0 + W, c0:c0 + L][m] + f * inj["rho_dn"]
            else:
                ev[r0:r0 + W, c0:c0 + L][m] += inj["amp"]
    return ev


_FN = None


def _call(c):
    """Для пула процессов (fork): вызывает функцию обработки ячейки, заданную в родителе."""
    return _safe(_FN, c)


def _safe(f, c):
    try:
        return f(c)
    except Exception as ex:
        return c, None, "exc " + repr(ex)[:100]


V_G = 6.63  # км/с, скорость следа строки по земле


DALPHA_DEG = 0.004      # допуск на ошибку угла полосы вдоль трассы (данные E3/E3b)
RHO_OBJ = 0.28          # серое отражение Starlink по калибровке E3b (30245: 0.26, 30580: 0.30)
F_HI, F_LO = 0.7, 0.25  # доля закрытого пикселя: измерено 0.25–0.71
DIL20 = 0.5             # ослабление малого объекта в полосах 20 м


def image_geometry(e, area_m2):
    """Размер изображения (в пикселях 10 м): вдоль трассы — штрих push-broom, поперёк — проекция объекта."""
    k = e["lam"] / e["rng"]
    A = area_m2 or 4.0
    L = math.sqrt(A * 7.5)                       # длинный размер (Starlink 120 м² → 30 м)
    Wd = A / L                                   # ширина
    along = (L * k + 10.0) * V_G / max(abs(e["v_al"]) * k, 0.3) / 10.0
    across = max(1.0, Wd * k / 10.0)
    return dict(k=k, L=L, along_px=float(np.clip(along, 1, 40)), across_px=float(across),
                n_px=float(np.clip(along, 1, 25) * min(across, 4)))


def forecast(e, roots, bands, mosS, mosG, area_m2, eff=0.59):
    """Прогноз по калиброванной модели (THEORY.md, E3b): контраст пикселя c_b = f·(ρ_obj − ρ_фона,b)·ослабление_b,
    отклик фильтра ≈ eff·c_b·√N, u_b = отклик/σ_b. T при f = 0.7 и f = 0.25 в шкале статистики (вклад ≤ 8)."""
    if not area_m2:
        return None
    geo = image_geometry(e, area_m2); sqN = math.sqrt(geo["n_px"])
    fcap = min(1.0, area_m2 * geo["k"] ** 2 / 100.0)   # объект мельче пикселя не закроет больше своей площади
    us = {}
    for b in bands:
        if b not in roots:
            continue
        _d, _a, x, y = roots[b][0]
        sb = float(np.nan_to_num(mosS.get(b, np.array([x]), np.array([y]))[0], nan=np.nan))
        gb = float(np.nan_to_num(mosG.get(b, np.array([x]), np.array([y]))[0], nan=np.nan))
        if not (np.isfinite(sb) and np.isfinite(gb) and sb > 0):
            continue
        rho_bg = gb * 1e-4 - 0.1
        dil = 1.0 if M.BRES[b] == 10 else DIL20
        us[b] = eff * (RHO_OBJ - rho_bg) * dil * 1e4 * sqN / sb          # при f = 1
    fh, fl = min(F_HI, fcap), min(F_LO, fcap)
    clipT = lambda f: float(np.sqrt(sum(min(abs(f * v), 8.0) ** 2 for v in us.values())))
    return dict(T_pred=clipT(fh), T_low=clipT(fl), T_raw=float(np.sqrt(sum((fh * v) ** 2 for v in us.values()))),
                u={b: round(fh * v, 1) for b, v in us.items()}, geometry=geo, fcap=fcap)
"""Физический детектор: обобщённый критерий отношения правдоподобия по сырым пикселям (THEORY §19).

Отличия от основного конвейера (§13–14):
  * сырые остатки loglocal в родном разрешении полосы, без «локального экстремума» с допуском;
  * шаблон в каждой полосе — точное изображение V2 mini (ось крыльев вдоль полёта ± ψ, §18): тело через камеру J_b,
    смаз push-broom вдоль видимой трассы, ящик пикселя, ФРТ MSI;
  * положения полос связаны общей ошибкой орбиты θ = (вдоль, поперёк, радиус) по точной геометрии
    (вдоль — узлы через 0.5 км, поперёк и радиус — якобиан); априорный штраф Σ(θ/σ)²;
  * счёт T² = Σ_b z_b² − штраф (z — нормированный отклик согласованного фильтра, два знака: тёмный или блик);
  * порог — нулевые прогоны со случайными сдвигами карт полос (как в §14), правило threshold.py.

  python scripts/physdet.py results/events_v11/STARLINK-36251_*.json [...] [--psis 0,15] [--nnull 10]
"""
import argparse, glob, json, math, sys, time
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter, binary_erosion, binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import passgeom as PG, imagery as I, bgsub, s2meta as M
from s2sat.orbits import dt_from_jd
from s2sat.threshold import threshold

SIG_C, SIG_R = 1.1, 1.9
CAP = 3
NEAR_M = 300.0
V_SCAN = 6600.0
PSF = {10: 4.2, 20: 8.4}


def template(J, res, row, vapp, psi_deg, Lw=12.8, Ww=4.1, Lb=4.1, Wb=2.7, half_px=None):
    """Изображение V2 mini (доля закрытия пикселя) на сетке полосы вокруг центра. J — 2×3 м/км."""
    Jm = np.asarray(J, float) / 1000.0
    psi = math.radians(psi_deg)
    u = np.array([math.cos(psi), math.sin(psi), 0.0]); w = np.array([-math.sin(psi), math.cos(psi), 0.0])
    pts, area = [], []
    for c0, L, Wd, ns in (((Lb / 2 + Lw / 2), Lw, Ww, 40), (-(Lb / 2 + Lw / 2), Lw, Ww, 40), (0.0, Lb, Wb, 8)):
        s = (np.arange(ns) + 0.5) / ns - 0.5; t = (np.arange(3) + 0.5) / 3 - 0.5
        S, T = np.meshgrid(s, t, indexing="ij")
        P3 = c0 * u[None] + (L * S.ravel())[:, None] * u[None] + (Wd * T.ravel())[:, None] * w[None]
        j1, j2 = Jm @ u, Jm @ w
        dA = abs(j1[0] * j2[1] - j1[1] * j2[0]) * L * Wd / S.size
        pts.append(P3 @ Jm.T); area.append(np.full(S.size, dA))
    P2 = np.vstack(pts); A = np.concatenate(area)
    t_ = np.asarray(row, float); vg = np.asarray(vapp, float)
    sv = vg * res / max(abs(V_SCAN - vg @ t_), 0.05 * V_SCAN)
    C = (PSF[int(res)] ** 2 + res ** 2 / 12.0) * np.eye(2) + np.outer(sv, sv) / 12.0
    ext = np.abs(P2).max(0) + 3 * math.sqrt(C.max())
    h = half_px or int(math.ceil(max(ext) / res)) + 1
    g = np.arange(-h, h + 1) * res
    X, Y = np.meshgrid(g, -g)
    iC = np.linalg.inv(C); det = np.linalg.det(C)
    dx = X[..., None] - P2[:, 0]; dy = Y[..., None] - P2[:, 1]
    q = iC[0, 0] * dx ** 2 + 2 * iC[0, 1] * dx * dy + iC[1, 1] * dy ** 2
    occ = (A / (2 * math.pi * math.sqrt(det)) * np.exp(-0.5 * q)).sum(-1)
    return occ


def read_band_region(items, epsg, bounds, bands, res):
    """Все полосы на окне bounds (сетка res), мозаика по гранулам: пропуски заполняет следующая гранула."""
    IM = {}
    for b in bands + ["scl"]:
        acc = None
        for it in items:
            if b not in it.assets:
                continue
            try:
                a = I.read_window(it.assets[b].href, epsg, bounds, res=res, resampling="nearest" if b == "scl" else "average")
            except Exception:
                continue
            acc = a.astype(np.float32) if acc is None else np.where(acc > 0, acc, a)
            if (acc > 0).mean() > 0.99:
                break
        if acc is not None:
            IM[b] = acc
    return IM


def residual(IM, b, far, res):
    """loglocal (THEORY §13) по плиткам: прогноз полосы по полосам, далёким по времени съёмки."""
    valid = np.all(np.stack([IM[c] > 0 for c in [b] + far]), 0)
    if "scl" in IM:
        scl = IM["scl"].astype(int)
        cl = np.isin(scl, (8, 9, 10)); edge = cl & ~binary_erosion(cl, iterations=1)
        valid &= ~binary_dilation(edge, iterations=max(1, int(200 / res))) & ~binary_dilation(scl == 1, iterations=2)
    LG = {c: np.log(np.maximum(IM[c], 1.0)).astype(np.float32) for c in [b] + far}
    H, W = IM[b].shape; blk = int(100 / res); win = 3 * blk; T = 40 * blk; mg = win
    Rl = np.full((H, W), np.nan, np.float32)
    for i0 in range(0, H, T):
        for j0 in range(0, W, T):
            a0, a1 = max(0, i0 - mg), min(H, i0 + T + mg); b0, b1 = max(0, j0 - mg), min(W, j0 + T + mg)
            v = valid[a0:a1, b0:b1]
            if v.mean() < 0.05:
                continue
            Rt = bgsub.predict_band_fast(LG[b][a0:a1, b0:b1], [LG[c][a0:a1, b0:b1] for c in far], v, block=blk, win=win)
            Rl[i0:min(H, i0 + T), j0:min(W, j0 + T)] = Rt[i0 - a0:i0 - a0 + min(T, H - i0), j0 - b0:j0 - b0 + min(T, W - j0)]
    R = IM[b] * (1.0 - np.exp(-np.nan_to_num(Rl)))
    R[~np.isfinite(Rl)] = np.nan
    return R


def zmaps(R, G, t):
    """Отклик согласованного фильтра Y = R⋆t и регрессоры физической модели в тех же единицах (z = Y/S):
    отражение X1 = 1000·Σt²/S (яркость объекта ∝ спектр), заслонение X2 = −G·Σt²/S (G — земля, DN − 1000).
    Модель: z_b ≈ a·s_b·X1_b + f·X2_b, a, f ≥ 0 (THEORY §19). S — шум Y по блокам (MAD)."""
    ok = np.isfinite(R)
    Y = fftconvolve(np.where(ok, R, 0.0), t[::-1, ::-1], mode="same").astype(np.float32)
    sup = (t > 0.02 * t.max()).astype(np.float32)
    cov = fftconvolve(ok.astype(np.float32), sup[::-1, ::-1], mode="same")
    Y[cov < 0.8 * sup.sum()] = np.nan
    m, sd, shp = bgsub.block_median_mad(Y, 50)
    Mb = bgsub._unblocks(m, shp, 50); Sb = bgsub._unblocks(sd + 1e-6, shp, 50)
    z = np.clip((Y - Mb) / Sb, -8.0, 8.0).astype(np.float32)
    tt = float((t ** 2).sum())
    X1 = (1000.0 * tt / Sb).astype(np.float32)
    X2 = (-np.nan_to_num(G) * tt / Sb).astype(np.float32)
    bad = ~np.isfinite(z) | ~np.isfinite(X1)
    z[bad] = 0.0; X1[bad] = 0.0; X2[bad] = 0.0
    return z, X1, X2


_NULLFN = None


def _call_null(sh):
    return _NULLFN(sh)


def run(path, psis, nnull, rng):
    x = json.loads(Path(path).read_text()); e, r = x["event"], x["result"]
    t0 = time.time()
    orbs = run.orbs
    P = PG.Pass(e, r, orbs); P.orb = PG.pick_orbit(orbs, e)
    if P.orb is None:
        return dict(name=Path(path).stem, status="no_orbit")
    D = e.get("orbit_disagree_km") or 0.0
    sa = float(e.get("sigma_along") or math.sqrt(0.85 ** 2 + (0.78 * D) ** 2))
    sc, sr = SIG_C, SIG_R
    # узлы вдоль: точные положения полос
    nodes = np.arange(-3 * sa, 3 * sa + 1e-9, min(0.5, sa))
    # полоса видит объект не при всех θ (края детекторов и полосы съёмки): положения по узлам там, где есть корень,
    # остальное — линейное продолжение по узлам; карта полосы сама даёт «нет данных» вне снимка
    NPd, Jd = {}, {}
    last = None
    for k, a in enumerate(nodes):
        Pa = P.positions([a, 0.0, 0.0], last)
        last = {b: v[:2] for b, v in Pa.items()} if Pa else last
        for b, v in Pa.items():
            NPd.setdefault(b, {})[k] = v
    P0, J = {}, {}
    for b, d in NPd.items():
        if len(d) < 2:
            continue
        ks = np.array(sorted(d)); xy = np.array([d[k][:2] for k in ks])
        cx = np.polyfit(nodes[ks], xy[:, 0], 1 if len(ks) < 3 else 2); cy = np.polyfit(nodes[ks], xy[:, 1], 1 if len(ks) < 3 else 2)
        NPd[b] = np.c_[np.polyval(cx, nodes), np.polyval(cy, nodes)]
        k0 = int(ks[np.argmin(np.abs(nodes[ks]))])
        Pj, Jj = P.jacobian([float(nodes[k0]), 0.0, 0.0])
        if b in Jj:
            J[b] = Jj[b]; P0[b] = (float(np.polyval(cx, 0.0)), float(np.polyval(cy, 0.0)), d[k0][2])
    bands = [b for b in P0 if b in J]
    NP = {b: NPd[b] for b in bands}
    # видимая скорость (для смаза) и направление строк
    Pm = np.array([P0[b][:2] for b in bands]); Tm = np.array([P0[b][2] for b in bands]) * 86400
    A_ = np.c_[Tm - Tm.mean(), np.ones_like(Tm)]
    vapp = np.array([np.linalg.lstsq(A_, Pm[:, 0], rcond=None)[0][0], np.linalg.lstsq(A_, Pm[:, 1], rcond=None)[0][0]])
    ps = math.radians(r["psi_deg"]); row = np.array([math.sin(ps), math.cos(ps)])
    its = I.items_around(e["lon"], e["lat"], dt_from_jd(e["jd"]), e["sent"][-1], d=1.0)
    its.sort(key=lambda it: it.id != r.get("item"))
    epsg = P.epsg
    maps = {}
    for b in bands:
        res = float(M.BRES.get(b, 20))
        if res > 20:
            continue
        # область полосы: все положения в коробке θ ±3σ + запас на шаблон
        pts = []
        for k in range(len(nodes)):
            if not np.isfinite(NP[b][k, 0]):
                continue
            for c in (-3 * sc, 3 * sc):
                for rr in (-3 * sr, 3 * sr):
                    pts.append(NP[b][k] + J[b][:, 1] * c + J[b][:, 2] * rr)
        if not pts:
            print(f"  {b}: нет корней на узлах", flush=True); continue
        pts = np.array(pts); mg = 600.0
        x0 = math.floor((pts[:, 0].min() - mg) / 20) * 20; x1 = math.ceil((pts[:, 0].max() + mg) / 20) * 20
        y0 = math.floor((pts[:, 1].min() - mg) / 20) * 20; y1 = math.ceil((pts[:, 1].max() + mg) / 20) * 20
        if (x1 - x0) * (y1 - y0) / res ** 2 > 2.5e7:
            print(f"  {b}: область {(x1 - x0) / 1000:.0f}×{(y1 - y0) / 1000:.0f} км — слишком велика, пропуск", flush=True)
            continue
        near = [c for c in bands if math.hypot(P0[c][0] - P0[b][0], P0[c][1] - P0[b][1]) < NEAR_M]
        far = [c for c in bands if c not in near]
        IM = read_band_region(its, epsg, (x0, y0, x1, y1), [b] + far, res)
        if b not in IM or (IM[b] > 0).mean() < 0.05 or len([c for c in far if c in IM]) < 2:
            print(f"  {b}: нет данных (покрытие {0 if b not in IM else (IM[b] > 0).mean():.2f}, далёких полос {len([c for c in far if c in IM])}, ближних {near})", flush=True)
            continue
        far = [c for c in far if c in IM]
        R = residual(IM, b, far, res)
        Gr = IM[b] - R - 1000.0
        gm, _gs, gshp = bgsub.block_median_mad(np.where(IM[b] > 0, Gr, np.nan).astype(np.float32), int(100 / res), min_n=10)
        Gb = bgsub._unblocks(gm, gshp, int(100 / res))
        Zs = {}
        for psi in psis:
            t = template(J[b], res, row, vapp, psi)
            z, X1, X2 = zmaps(R, Gb, t)
            if res == 10:          # сетка гипотез 15 м: пик не теряется (максимум |z| в окрестности 3×3 с его знаком)
                zp = maximum_filter(z, size=3); zn = -maximum_filter(-z, size=3); z = np.where(zp >= -zn, zp, zn)
            Zs[psi] = (z, X1, X2)
        maps[b] = dict(Zs=Zs, x0=x0, y1=y1, res=res, shape=R.shape)
        del IM, R
    if len(maps) < 2:
        return dict(name=Path(path).stem, status="unobservable", bands=list(maps))
    # сетка гипотез (a, c) и радиус r. Радиус сдвигает изображение полосы почти на постоянный вектор: в координатах
    # (a, c) полосы b это сдвиг r·m_b, m_b = M_b⁻¹·J_r,b, M_b = [∂P_b/∂a, J_c,b]. Общая часть K = среднее m_b
    # поглощается сдвигом сетки (a' = a + K_a r, c' = c + K_c r), остаток δ_b = m_b − K мал (параллакс полос).
    # Тогда T²(a', c', r) = Σ_b W_b(a' + r·δ_b,a, c' + r·δ_b,c), W_b — карта z² полосы, перенесённая на сетку (a, c)
    # один раз; перебор по r — сложение сдвинутых массивов.
    bl = list(maps)
    mb = {}
    for b in bl:
        Mb = J[b][:, :2]
        mb[b] = np.linalg.lstsq(Mb, J[b][:, 2], rcond=None)[0]
    K = np.mean([mb[b] for b in bl], 0)
    dlt = {b: mb[b] - K for b in bl}
    Jac = np.mean([J[b][:, :2] for b in bl], 0)
    step_img = 15.0
    da = step_img / max(np.hypot(*Jac[:, 0]), 1.0); dc = step_img / max(np.hypot(*np.mean([J[b][:, 1] for b in bl], 0)), 1.0)
    A1 = np.arange(-3 * sa - 3 * sr * abs(K[0]), 3 * sa + 3 * sr * abs(K[0]) + 1e-9, da)
    C1 = np.arange(-3 * sc - 3 * sr * abs(K[1]), 3 * sc + 3 * sr * abs(K[1]) + 1e-9, dc)
    if len(A1) * len(C1) > 6e6:
        f = math.sqrt(len(A1) * len(C1) / 6e6); da *= f; dc *= f
        A1 = np.arange(A1[0], A1[-1] + 1e-9, da); C1 = np.arange(C1[0], C1[-1] + 1e-9, dc)
    smax = max(max(abs(dlt[b][0]) / da, abs(dlt[b][1]) / dc) for b in bl)
    # шаг по радиусу: разностный сдвиг полос ≤ 1.5 клетки (z уже взят максимумом по 3×3)
    R1 = np.linspace(-3 * sr, 3 * sr, int(max(3, min(81, math.ceil(6 * sr * smax / 1.5) + 1))))
    print(f"  {Path(path).stem}: полос {len(bl)}, гипотез {len(A1)}×{len(C1)}×{len(R1)}, σ {sa:.2f}/{sc}/{sr} км, "
          f"чтение+фильтры {time.time() - t0:.0f} с", flush=True)

    from s2sat.detect import SPECTRA
    SP = list(SPECTRA.values())

    def warp(b, psi, shift):
        m = maps[b]; z, X1, X2 = m["Zs"][psi]
        pa = np.c_[np.interp(A1, nodes, NP[b][:, 0]), np.interp(A1, nodes, NP[b][:, 1])].astype(np.float64)
        px = pa[:, 0][:, None] + J[b][0, 1] * C1[None, :] + shift[0]
        py = pa[:, 1][:, None] + J[b][1, 1] * C1[None, :] + shift[1]
        col = np.floor((px - m["x0"]) / m["res"]).astype(np.int32); rowi = np.floor((m["y1"] - py) / m["res"]).astype(np.int32)
        ok = (col >= 0) & (col < m["shape"][1]) & (rowi >= 0) & (rowi < m["shape"][0])
        out = []
        for Mp in (z, X1, X2):
            W = np.zeros(px.shape, np.float32); W[ok] = Mp[rowi[ok], col[ok]]; out.append(W)
        zW, x1, x2 = out
        return dict(z=zW, x1=x1, x2=x2, n=(x1 > 0).astype(np.float32))

    def shifted(W, di, dj):
        out = np.zeros_like(W); H_, W_ = W.shape
        a0, a1 = max(0, -di), min(H_, H_ - di); b0, b1 = max(0, -dj), min(W_, W_ - dj)
        if a1 > a0 and b1 > b0:
            out[a0:a1, b0:b1] = W[a0 + di:a1 + di, b0 + dj:b1 + dj]
        return out

    def glrt(A11, A12, A22, B1, B2):
        """max по a, f ≥ 0 от 2(a·B1 + f·B2) − (a, f)A(a, f)ᵀ (= T²)."""
        det = A11 * A22 - A12 ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            ah = (A22 * B1 - A12 * B2) / det; fh = (A11 * B2 - A12 * B1) / det
            both = np.where((ah >= 0) & (fh >= 0) & (det > 1e-9 * (A11 * A22 + 1e-30)), ah * B1 + fh * B2, 0.0)
            oa = np.where((B1 > 0) & (A11 > 0), B1 * B1 / np.where(A11 > 0, A11, 1), 0.0)
            of = np.where((B2 > 0) & (A22 > 0), B2 * B2 / np.where(A22 > 0, A22, 1), 0.0)
        return np.nan_to_num(np.maximum(both, np.maximum(oa, of)))

    def score(psi, shifts=None):
        Ws = {b: warp(b, psi, (0.0, 0.0) if shifts is None else shifts[b]) for b in bl}
        best = (-1e9, None)
        for rr in R1:
            Sh = []
            for b in bl:
                di, dj = int(round(rr * dlt[b][0] / da)), int(round(rr * dlt[b][1] / dc))
                Sh.append({k: shifted(v, di, dj) for k, v in Ws[b].items()})
            Z = np.stack([w["z"] for w in Sh]); Nn = sum(w["n"] for w in Sh)
            if CAP and len(Sh) >= CAP:
                # ограничение по рангу (как CAP_RANK в §12): |z| каждой полосы ≤ CAP-го по величине |z| среди полос —
                # одна-две выбросные полосы не создают сигнал, нужно согласие ≥ CAP полос
                capv = -np.partition(-np.abs(Z), CAP - 1, axis=0)[CAP - 1]
                Z = np.clip(Z, -capv, capv)
            A22 = sum(w["x2"] ** 2 for w in Sh); B2 = sum(w["x2"] * Zb for w, Zb in zip(Sh, Z))
            T2 = np.zeros_like(A22)
            for sp in SP:
                sv = [sp.get(b, 1.0) for b in bl]
                A11 = sum((s_ * w["x1"]) ** 2 for s_, w in zip(sv, Sh)); A12 = sum(s_ * w["x1"] * w["x2"] for s_, w in zip(sv, Sh))
                B1 = sum(s_ * w["x1"] * Zb for s_, w, Zb in zip(sv, Sh, Z))
                T2 = np.maximum(T2, glrt(A11, A12, A22, B1, B2))
            acc = {"N": Nn}
            a = A1 - K[0] * rr; c = C1 - K[1] * rr
            pen = (a[:, None] / sa) ** 2 + (c[None, :] / sc) ** 2 + (rr / sr) ** 2
            T2 = np.where(acc["N"] >= CAP, T2 - pen, -1e9)
            k = int(np.argmax(T2))
            if T2.flat[k] > best[0]:
                i_, j_ = np.unravel_index(k, T2.shape)
                best = (float(T2.flat[k]), (float(a[i_]), float(c[j_]), float(rr)))
        return best

    out = dict(name=Path(path).stem, bands=bl, sigma=(sa, sc, sr))
    res_ = {}
    for psi in psis:
        res_[psi] = score(psi)
    psi_b = max(res_, key=lambda p: res_[p][0])
    T2b, th = res_[psi_b]
    out.update(score=math.sqrt(max(T2b, 0)), theta=th, psi=psi_b)
    # z по полосам в лучшей гипотезе
    zb = {}
    for b in bl:
        m = maps[b]; a, c, rr = th
        px = np.interp(a, nodes, NP[b][:, 0]) + J[b][0, 1] * c + J[b][0, 2] * rr
        py = np.interp(a, nodes, NP[b][:, 1]) + J[b][1, 1] * c + J[b][1, 2] * rr
        col = int((px - m["x0"]) // m["res"]); rw = int((m["y1"] - py) // m["res"])
        if 0 <= rw < m["shape"][0] and 0 <= col < m["shape"][1]:
            zb[b] = round(float(m["Zs"][psi_b][0][rw, col]), 1)
    out["z"] = zb
    # нули: случайные сдвиги карт полос (1–6 км), максимум по ψ
    shs = []
    for k in range(nnull):
        shs.append({b: tuple(rng.uniform(1000, 6000) * np.array([math.cos(q), math.sin(q)]) for q in [rng.uniform(0, 2 * math.pi)])[0] for b in bl})
    global _NULLFN
    _NULLFN = lambda sh: math.sqrt(max(max(score(psi, sh)[0] for psi in psis), 0))
    import multiprocessing as mp
    with mp.get_context("fork").Pool(int(__import__("os").environ.get("PHYSDET_WORKERS", "2"))) as pool:
        nulls = pool.map(_call_null, shs, chunksize=1)
    out["nulls"] = nulls; out["null_max"] = max(nulls)
    thr = threshold(out)
    out["thr"] = thr
    out["status"] = "detected" if out["score"] >= thr else "not_detected"
    out["seconds"] = round(time.time() - t0)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+"); ap.add_argument("--psis", default="0,15"); ap.add_argument("--nnull", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results/physdet"))
    a = ap.parse_args()
    run.orbs = PG.Orbits(until="20261123T0607")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    psis = [float(v) for v in a.psis.split(",")]
    for pth in a.paths:
        try:
            o = run(pth, psis, a.nnull, rng)
        except Exception as ex:
            import traceback; traceback.print_exc(); o = dict(name=Path(pth).stem, status=f"error {ex}")
        Path(a.out, o["name"] + ".json").write_text(json.dumps(o, default=float, indent=1))
        if "score" in o:
            print(f"PHYSDET {o['name']}: {o['status']}  счёт {o['score']:.1f} порог {o['thr']:.1f} (нули медиана {np.median(o['nulls']):.1f}, "
                  f"макс {o['null_max']:.1f})  θ {np.round(o['theta'], 2).tolist()} км, ψ {o['psi']:g}°, z {o['z']}  {o['seconds']} с", flush=True)
        else:
            print(f"PHYSDET {o['name']}: {o['status']}", flush=True)

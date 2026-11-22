"""Детектор: карты изменений, карта «эксклюзивности», совместный поиск сдвига, нуль-тест (HANDOFF 6)."""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import maximum_filter, median_filter

RES = 20.0
CLIPE = 8.0


def zmap(ev, rf, k=7, min_valid=300):
    """z-карта изменений: R = ev − g·ref, минус медиана k×k, нормировка на MAD.

    ev, rf: float, 0 = нет данных. Возвращает (Z с NaN вне данных, режим 'ref'|'self'|'empty').
    """
    v = ev > 0
    mode, Rr = "self", ev
    if rf is not None:
        vr = v & (rf > 0)
        if vr.mean() > 0.6:
            g = np.median(ev[vr] / rf[vr]); Rr = ev - g * rf; v = vr; mode = "ref"
    Rr = Rr - median_filter(Rr, k)
    if v.sum() < min_valid:
        return None, "empty"
    md = np.median(Rr[v]); sg = 1.4826 * np.median(np.abs(Rr[v] - md)) + 1e-6
    Z = ((Rr - md) / sg).astype(np.float32); Z[~v] = np.nan
    return Z, mode


def exclusive_maps(Z: dict, Arep: dict, thr: float):
    """E_b = maxfilter3(max(0, |z_b| − maxfilter7(max_{c далеко} |z_c|)))."""
    bands = list(Z)
    AZ = {b: np.abs(np.nan_to_num(Z[b])) for b in bands}
    E = {}
    for b in bands:
        far = [c for c in bands if abs(Arep[c] - Arep[b]) > thr]
        F = maximum_filter(np.max(np.stack([AZ[c] for c in far]), 0), size=7) if far else 0.0
        ex = maximum_filter(np.maximum(0.0, AZ[b] - F), size=3).astype(np.float32)
        ex[np.isnan(Z[b])] = np.nan
        E[b] = ex
    return E


class Raster:
    """Набор карт на общей сетке (minx, maxy, шаг RES)."""

    def __init__(self, maps: dict, bounds):
        self.maps = maps; self.minx, self.miny, self.maxx, self.maxy = bounds

    def get(self, b, X, Y):
        A = self.maps[b]; H, W = A.shape
        col = np.floor((X - self.minx) / RES).astype(int); row = np.floor((self.maxy - Y) / RES).astype(int)
        ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        V = np.full(np.shape(X), np.nan, np.float32)
        V[ok] = A[row[ok], col[ok]]
        return V


def shift_search(roots, mos: Raster, bands, Rm, step=20.0, disp=None, minb=3):
    """Общий сдвиг (sx, sy) для всех полос; счёт = Σ min(E, 8)/√n."""
    S = np.arange(-Rm, Rm + 1e-6, step); SX, SY = np.meshgrid(S, S)
    tot = np.zeros(SX.shape, np.float32); cnt = np.zeros_like(tot); hit = np.zeros_like(tot)
    for b in bands:
        if b not in roots:
            continue
        dx, dy = disp[b] if disp else (0.0, 0.0)
        Vb = np.full(SX.shape, np.nan, np.float32)
        for (_det, _a, x, y) in roots[b]:
            Vb = np.fmax(Vb, mos.get(b, x + dx + SX, y + dy + SY))
        ok = ~np.isnan(Vb)
        tot[ok] += np.minimum(Vb[ok], CLIPE); cnt[ok] += 1; hit[ok] += Vb[ok] > 4
    need = max(minb, 0.6 * len([b for b in bands if b in roots]))
    sc = np.where(cnt >= need, tot / np.sqrt(np.maximum(cnt, 1)), -1e9)
    i, j = np.unravel_index(np.argmax(sc), sc.shape)
    return dict(score=float(sc[i, j]), sx=float(SX[i, j]), sy=float(SY[i, j]), hit=int(hit[i, j]), cnt=int(cnt[i, j]))


def null_test(roots, mos, bands, Rm, n=10, seed=5, rmin=1000.0, rmax=4000.0, minb=3):
    """Каждой полосе — случайное смещение 1–4 км; возвращает список счётов."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        disp = {}
        for b in bands:
            r = rng.uniform(rmin, rmax); th = rng.uniform(0, 2 * math.pi)
            disp[b] = (r * math.cos(th), r * math.sin(th))
        out.append(shift_search(roots, mos, bands, Rm, disp=disp, minb=minb)["score"])
    return out


HUBER_C = 3.0


def coherent_search(roots, mosY, mosS, bands, Rm, step=10.0, disp=None, minb=3, c=HUBER_C):
    """Когерентный тест: одна общая яркость объекта a во всех полосах (знак любой).

    y_b = a + n_b, шум с тяжёлыми хвостами (края облаков, блики). Робастная (Хьюбер) score-статистика:
    u_b = y_b/s_b, w_b = 1/s_b,  T = |Σ w_b ψ_c(u_b)| / sqrt(Σ w_b² E[ψ_c²]),  ψ_c(u) = clip(u, −c, c).
    Вклад одной полосы ограничен c, поэтому выброс в одной полосе не перевешивает согласованный сигнал.
    Для полосы с двумя корнями (два детектора) берётся тот, где |y/s| больше.
    """
    S = np.arange(-Rm, Rm + 1e-6, step); SX, SY = np.meshgrid(S, S)
    num = np.zeros(SX.shape, np.float64); den = np.zeros_like(num); cnt = np.zeros_like(num); hit = np.zeros_like(num)
    amp_num = np.zeros_like(num)
    for b in bands:
        if b not in roots:
            continue
        dx, dy = disp[b] if disp else (0.0, 0.0)
        by = None
        for (_det, _a, x, y) in roots[b]:
            Yv = mosY.get(b, x + dx + SX, y + dy + SY).astype(np.float64)
            Sv = mosS.get(b, x + dx + SX, y + dy + SY).astype(np.float64)
            u = np.abs(Yv / Sv)
            if by is None:
                by = (Yv, Sv, u)
            else:
                take = np.nan_to_num(u, nan=-1) > np.nan_to_num(by[2], nan=-1)
                by = (np.where(take, Yv, by[0]), np.where(take, Sv, by[1]), np.where(take, u, by[2]))
        Yv, Sv, u = by
        ok = np.isfinite(Yv) & np.isfinite(Sv) & (Sv > 0)
        uu = np.clip(Yv[ok] / Sv[ok], -c, c)
        num[ok] += uu / Sv[ok]; den[ok] += 1.0 / Sv[ok] ** 2; cnt[ok] += 1; hit[ok] += u[ok] > 4
        amp_num[ok] += Yv[ok] / Sv[ok] ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        e_psi2 = _huber_var(c)
        snr = np.where((cnt >= minb) & (den > 0), np.abs(num) / np.sqrt(den * e_psi2), -1e9)
    i, j = np.unravel_index(np.argmax(snr), snr.shape)
    amp = amp_num[i, j] / den[i, j] if den[i, j] > 0 else float("nan")
    return dict(score=float(snr[i, j]), sx=float(SX[i, j]), sy=float(SY[i, j]), hit=int(hit[i, j]), cnt=int(cnt[i, j]),
                amp_dn=float(amp))


def _huber_var(c):
    """E[clip(u, −c, c)²] для u ~ N(0, 1)."""
    from math import erf, exp, pi, sqrt
    Phi = 0.5 * (1 + erf(c / sqrt(2))); phi = exp(-c * c / 2) / sqrt(2 * pi)
    return (2 * Phi - 1) - 2 * c * phi + 2 * c * c * (1 - Phi)


def coherent_search_u(roots, mosU, bands, Rm, step=20.0, disp=None, minb=3, c=6.0):
    """Робастный когерентный тест по нормированным откликам u_b (с допуском положения).

    T = |Σ_b ψ_c(u_b)| / sqrt(n · E[ψ_c²]); знак общий для всех полос (объект во всех
    полосах светлее фона или во всех темнее), вклад полосы ограничен c."""
    S = np.arange(-Rm, Rm + 1e-6, step); SX, SY = np.meshgrid(S, S)
    num = np.zeros(SX.shape, np.float64); cnt = np.zeros_like(num); hit = np.zeros_like(num)
    for b in bands:
        if b not in roots:
            continue
        dx, dy = disp[b] if disp else (0.0, 0.0)
        best = None
        for (_det, _a, x, y) in roots[b]:
            U = mosU.get(b, x + dx + SX, y + dy + SY).astype(np.float64)
            best = U if best is None else np.where(np.abs(np.nan_to_num(U)) > np.abs(np.nan_to_num(best)), U, best)
        ok = np.isfinite(best)
        num[ok] += best[ok]; cnt[ok] += 1; hit[ok] += np.abs(best[ok]) > 3
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where(cnt >= minb, np.abs(num) / np.sqrt(np.maximum(cnt, 1)), -1e9)
    i, j = np.unravel_index(np.argmax(T), T.shape)
    return dict(score=float(T[i, j]), sx=float(SX[i, j]), sy=float(SY[i, j]), hit=int(hit[i, j]), cnt=int(cnt[i, j]),
                sign=int(np.sign(num[i, j])))


SPECTRA = {
    # форма спектра отражения объекта ρ_b ∝ s_b (нормировка не важна: амплитуда свободна)
    "gray": {b: 1.0 for b in ("blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22")},
    # Starlink по детекциям 31550/32702: ~0.1–0.2 в видимых, ~0.3 в NIR (E3, THEORY.md §3)
    "red": {"blue": 0.4, "green": 0.6, "red": 0.7, "rededge1": 0.8, "rededge2": 0.9, "rededge3": 1.0,
            "nir": 1.0, "nir08": 1.0, "swir16": 0.9, "swir22": 0.8},
}


CAP_RANK = int(__import__("os").environ.get("S2_CAPRANK", "0"))


def occlusion_search(roots, mosU, mosS, mosG, bands, Rm, step=20.0, disp=None, minb=3, spectra=None, cap_rank=None):
    """GLRT модели заслонения: объект закрывает долю f пикселя, отклик полосы b
        y_b = 1e4·f·(ρ_b − ρ_фона,b),  ρ_b = A·s_b  (форма спектра s задана, A и f ≥ 0 свободны),
    ρ_фона,b = G_b·1e−4 − 0.1 (L2A). В единицах u_b = y_b/σ_b: u = p1·x1 + p2·x2 + шум,
    x1 = s_b/σ_b, x2 = −1e4·ρ_фона,b/σ_b, p1 = 1e4·f·A, p2 = f. T² = uᵀX(XᵀX)⁻¹Xᵀu (2 ст. свободы);
    при f̂ < 0 — модель с одним параметром. Максимум по набору спектров."""
    spectra = spectra or SPECTRA
    S = np.arange(-Rm, Rm + 1e-6, step); SX, SY = np.meshgrid(S, S)
    sh = SX.shape
    per = []
    cnt = np.zeros(sh); hit = np.zeros(sh)
    for b in bands:
        if b not in roots:
            continue
        dx, dy = disp[b] if disp else (0.0, 0.0)
        bu = bs = bg = None
        for (_det, _a, x, y) in roots[b]:
            X_, Y_ = x + dx + SX, y + dy + SY
            u = mosU.get(b, X_, Y_).astype(np.float64)
            sv = mosS.get(b, X_, Y_).astype(np.float64); gv = mosG.get(b, X_, Y_).astype(np.float64)
            if bu is None:
                bu, bs, bg = u, sv, gv
            else:
                take = np.abs(np.nan_to_num(u)) > np.abs(np.nan_to_num(bu))
                bu, bs, bg = np.where(take, u, bu), np.where(take, sv, bs), np.where(take, gv, bg)
        ok = np.isfinite(bu) & np.isfinite(bs) & (bs > 0) & np.isfinite(bg)
        inv = np.where(ok, 1.0 / np.where(ok, bs, 1), 0)
        uu = np.where(ok, bu, 0)
        x2 = -inv * (np.where(ok, bg, 1000.0) - 1000.0)   # −1e4·ρ_фона/σ (DN)
        per.append((b, inv, x2, uu))
        cnt += ok; hit += ok & (np.abs(uu) > 3)
    cap_rank = CAP_RANK if cap_rank is None else cap_rank
    if cap_rank and len(per) >= cap_rank:
        # многополосная согласованность: вклад полосы не больше m-й по силе полосы в этой точке,
        # так одиночное яркое пятно в одной полосе (нуль) не даёт счёта, а объект в ≥ m полосах — даёт
        Au = np.sort(np.stack([np.abs(p_[3]) for p_ in per]), axis=0)[-cap_rank]
        per = [(b, inv, x2, np.clip(uu, -Au, Au)) for b, inv, x2, uu in per]
    best_T = np.full(sh, -1e9); best_spec = np.full(sh, "", object)
    out_a = out_f = None
    for name, sp in spectra.items():
        A11 = np.zeros(sh); A12 = np.zeros(sh); A22 = np.zeros(sh); B1 = np.zeros(sh); B2 = np.zeros(sh)
        for b, inv, x2, uu in per:
            x1 = sp.get(b, 1.0) * 1e4 * 0.1 * inv   # масштаб ~ «ρ=0.1» для обусловленности
            A11 += x1 * x1; A12 += x1 * x2; A22 += x2 * x2; B1 += x1 * uu; B2 += x2 * uu
        det = A11 * A22 - A12 * A12
        with np.errstate(invalid="ignore", divide="ignore"):
            a_hat = (A22 * B1 - A12 * B2) / (det + 1e-12 * (A11 * A22 + 1e-30))
            f_hat = (A11 * B2 - A12 * B1) / (det + 1e-12 * (A11 * A22 + 1e-30))
            T2 = np.where(f_hat >= 0, a_hat * B1 + f_hat * B2, B1 * B1 / A11)
            T = np.where(cnt >= minb, np.sqrt(np.maximum(np.nan_to_num(T2, nan=0.0), 0)), -1e9)
        better = T > best_T
        best_T = np.where(better, T, best_T); best_spec = np.where(better, name, best_spec)
        if out_a is None:
            out_a, out_f = a_hat, f_hat
        else:
            out_a = np.where(better, a_hat, out_a); out_f = np.where(better, f_hat, out_f)
    i, j = np.unravel_index(np.argmax(best_T), sh)
    return dict(score=float(best_T[i, j]), sx=float(SX[i, j]), sy=float(SY[i, j]), hit=int(hit[i, j]), cnt=int(cnt[i, j]),
                a_hat=float(out_a[i, j]), f_hat=float(out_f[i, j]), spectrum=str(best_spec[i, j]))
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

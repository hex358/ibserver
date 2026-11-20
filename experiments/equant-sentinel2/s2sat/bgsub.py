"""Удаление фона по самому снимку и согласованный фильтр.

Неподвижная сцена во всех полосах одного снимка совмещена (полосы снимаются за ~2 с),
а движущийся объект в каждой полосе стоит в своём месте. Поэтому каждая полоса b
предсказывается линейной комбинацией других полос в тех же пикселях:

    I_b(x) ≈ β0 + Σ_c β_c I_c(x) [+ β_ref · Ref_b(x)]

Коэффициенты подбираются робастно по блокам 500 м. Остаток R_b = I_b − предсказание
содержит объект (его нет в других полосах в этом месте) и шум; фон уходит без
снимка другой даты, поэтому сезонные изменения, облака другой даты и т.п. не мешают.

Полосы, в которых объект может оказаться в тех же пикселях (близкие по времени съёмки),
из регрессоров исключаются: это определяется геометрией (разница углов × км/градус).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve


def _gauss_kernel(sigma_px=0.8, r=2):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = np.exp(-(x * x + y * y) / (2 * sigma_px ** 2))
    return (k / np.sqrt((k * k).sum())).astype(np.float32)  # ||k||₂ = 1: белый шум → дисперсия 1·σ²


KERNEL = _gauss_kernel()


def predict_band(target, regs, valid, block=50, win=100, iters=2):
    """Робастная блочная регрессия target ~ [1, regs...]. Возвращает остаток (NaN вне valid)."""
    H, W = target.shape
    X = np.stack([np.ones_like(target)] + regs, -1)  # (H, W, p)
    R = np.full((H, W), np.nan, np.float32)
    half = (win - block) // 2
    for r0 in range(0, H, block):
        for c0 in range(0, W, block):
            ra, rb = max(0, r0 - half), min(H, r0 + block + half)
            ca, cb = max(0, c0 - half), min(W, c0 + block + half)
            m = valid[ra:rb, ca:cb]
            if m.sum() < 200:
                continue
            y = target[ra:rb, ca:cb][m].astype(np.float64)
            A = X[ra:rb, ca:cb][m].astype(np.float64)
            w = np.ones_like(y, bool)
            beta = None
            for _ in range(iters + 1):
                beta, *_ = np.linalg.lstsq(A[w], y[w], rcond=None)
                res = y - A @ beta
                s = 1.4826 * np.median(np.abs(res[w] - np.median(res[w]))) + 1e-6
                w = np.abs(res) < 4 * s
            ia, ib = r0, min(H, r0 + block); ja, jb = c0, min(W, c0 + block)
            Xi = X[ia:ib, ja:jb].astype(np.float64)
            pred = Xi @ beta
            out = (target[ia:ib, ja:jb] - pred).astype(np.float32)
            out[~valid[ia:ib, ja:jb]] = np.nan
            R[ia:ib, ja:jb] = out
    return R


def _win_sums(F, block, half):
    """Суммы F по окнам [r0−half, r0+block+half) × [c0−half, c0+block+half) для всех блоков (интегральное изображение)."""
    H, W = F.shape[:2]
    S = np.zeros((H + 1, W + 1) + F.shape[2:], np.float64)
    S[1:, 1:] = F.cumsum(0).cumsum(1)
    r0 = np.arange(0, H, block); c0 = np.arange(0, W, block)
    ra = np.clip(r0 - half, 0, H); rb = np.clip(r0 + block + half, 0, H)
    ca = np.clip(c0 - half, 0, W); cb = np.clip(c0 + block + half, 0, W)
    return (S[rb][:, cb] - S[ra][:, cb] - S[rb][:, ca] + S[ra][:, ca])  # (nbr, nbc, ...)


def predict_band_fast(target, regs, valid, block=50, win=100, iters=1, ridge=1e-6):
    """То же, что predict_band, но через нормальные уравнения по окнам (интегральные изображения):
    XᵀWX и XᵀWy для всех блоков сразу, затем малые системы p×p. Робастность: повторная подгонка
    без пикселей с |остаток| > 4σ (σ по блоку)."""
    H, W = target.shape
    X = np.stack([np.ones_like(target)] + regs, -1).astype(np.float64)
    X /= np.maximum(np.nanmax(np.abs(X), axis=(0, 1)), 1.0)  # масштаб для устойчивости
    y = target.astype(np.float64)
    half = (win - block) // 2
    w = valid.astype(np.float64)
    nbr, nbc = len(range(0, H, block)), len(range(0, W, block))
    bi = np.minimum(np.arange(H) // block, nbr - 1); bj = np.minimum(np.arange(W) // block, nbc - 1)
    p = X.shape[-1]
    R = None
    for it in range(iters + 1):
        Xw = X * w[..., None]
        XtX = _win_sums(Xw[..., :, None] * X[..., None, :], block, half)  # (nbr, nbc, p, p)
        Xty = _win_sums(Xw * y[..., None], block, half)                   # (nbr, nbc, p)
        n = _win_sums(w, block, half)
        XtX = XtX + ridge * np.eye(p) * np.maximum(n, 1)[..., None, None]
        beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]               # (nbr, nbc, p)
        B = beta[bi][:, bj]                                                # (H, W, p)
        R = y - np.einsum("hwp,hwp->hw", X, B)
        # σ по блокам (по валидным пикселям, MAD)
        sig = np.ones((nbr, nbc))
        for a in range(nbr):
            for b in range(nbc):
                v = R[a * block:(a + 1) * block, b * block:(b + 1) * block][valid[a * block:(a + 1) * block, b * block:(b + 1) * block]]
                if v.size > 50:
                    sig[a, b] = 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-6
        w = (valid & (np.abs(R) < 4 * sig[bi][:, bj])).astype(np.float64)
    R = R.astype(np.float32); R[~valid] = np.nan
    return R


def matched(R, block=50):
    """Согласованный фильтр по остатку и нормировка на шум блока.

    Возвращает (Y — отклик в DN, S — шум отклика в DN, по блокам)."""
    Rf = np.nan_to_num(R)
    Y = convolve(Rf, KERNEL, mode="nearest").astype(np.float32)
    Y[np.isnan(R)] = np.nan
    H, W = R.shape
    S = np.full((H, W), np.nan, np.float32)
    for r0 in range(0, H, block):
        for c0 in range(0, W, block):
            v = Y[r0:r0 + block, c0:c0 + block]
            f = v[np.isfinite(v)]
            if f.size > 100:
                S[r0:r0 + block, c0:c0 + block] = 1.4826 * np.median(np.abs(f - np.median(f))) + 1e-3
    return Y, S


def local_extreme(u, radius_px=5):
    """Знаковое значение с максимальным |u| в окне ±radius_px.

    Неизвестный малый сдвиг полосы (ошибка модели углов/задержек ~десятки м) как
    параметр-помеха GLRT: берём лучшее положение в пределах допуска."""
    from scipy.ndimage import maximum_filter, minimum_filter
    f = np.nan_to_num(u, nan=0.0)
    hi = maximum_filter(f, size=2 * radius_px + 1); lo = minimum_filter(f, size=2 * radius_px + 1)
    out = np.where(hi >= -lo, hi, lo).astype(np.float32)
    out[np.isnan(u)] = np.nan
    return out


def remove_common_mode(Z: dict):
    """Убрать синфазную составляющую: структура, видная во всех полосах в одной точке, — сцена.

    Движущийся объект в данной точке есть лишь в 1–2 полосах из ~10, поэтому медиана |z| по
    полосам (с допуском 1 пиксель) — робастная оценка «общего» уровня аномалии сцены.
    u_b ← sign(z_b)·max(0, |z_b| − A)."""
    from scipy.ndimage import maximum_filter
    st = np.stack([np.abs(np.nan_to_num(z)) for z in Z.values()])
    A = maximum_filter(np.median(st, 0), size=3)
    out = {}
    for b, z in Z.items():
        u = np.sign(z) * np.maximum(0.0, np.abs(z) - A)
        out[b] = u.astype(np.float32)
    return out


def standardize(u, block=50, c=8.0):
    """Пере-калибровка по блокам: (u − медиана)/MAD. После выбора экстремума в окне распределение
    смещено (тяжёлый положительный хвост сцены); так под нулём отклик снова центрирован и единичен."""
    out = np.full(u.shape, np.nan, np.float32)
    H, W = u.shape
    for r0 in range(0, H, block):
        for c0 in range(0, W, block):
            v = u[r0:r0 + block, c0:c0 + block]
            f = v[np.isfinite(v)]
            if f.size > 100:
                m = np.median(f); s = 1.4826 * np.median(np.abs(f - m)) + 1e-6
                out[r0:r0 + block, c0:c0 + block] = np.clip((v - m) / s, -c, c)  # вклад полосы ≤ c
    return out

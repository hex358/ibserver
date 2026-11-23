"""Форма объекта по снимкам Sentinel-2: дифференцируемый рендер «анализ через синтез».

Методы из литературы:
  * мягкий растеризатор (SoftRas, Liu et al., arXiv:1904.01786) и сплаттинг гауссианами (3D Gaussian Splatting,
    Kerbl et al., arXiv:2308.04079; EWA, Zwicker et al. 2001): поверхность — облако точек-гауссиан, проекция
    аффинной камерой даёт гауссиану на снимке в замкнутой форме, всё дифференцируемо;
  * переменная проекция (Golub & Pereyra) для линейных параметров (яркость объекта по полосам);
  * лапласовская оценка апостериорной неопределённости (гессиан через автодифференцирование).

Камера: для полосы b точка тела со смещением δ (м, оси «вдоль, поперёк, радиус» орбиты объекта) видна в
p_b + J_b·δ/1000, где J_b — якобиан ошибки орбиты (м снимка на км; passgeom.Pass.jacobian). Это точная
локальная модель съёмки push-broom с движением объекта; разные полосы — разные моменты и ракурсы.

Пиксель: наблюдение = (1 − occ)·земля + occ·объект ⇒ остаток r = occ·(c_b − g), где occ — доля пикселя,
закрытая объектом (свёртка силуэта с ФРТ и ящиком пикселя), g — яркость земли (DN − 1000), c_b ≥ 0 —
яркость объекта в полосе (линейный параметр, решается точно на каждом шаге).
"""
from __future__ import annotations

import math

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

ER = jnp.array([0.0, 0.0, 1.0])


def _grid(ns, nt):
    s = (np.arange(ns) + 0.5) / ns - 0.5; t = (np.arange(nt) + 0.5) / nt - 0.5
    S, T = np.meshgrid(s, t, indexing="ij")
    return jnp.asarray(S.ravel()), jnp.asarray(T.ravel())


S_W, T_W = _grid(32, 3)       # крыло (шаг на снимке ≤ σ ФРТ)
S_B, T_B = _grid(4, 3)        # шина


def axes(psi, phi):
    """u — ось крыльев (горизонтальна: шина смотрит в надир), n — нормаль крыла (поворот вокруг u на φ от зенита)."""
    u = jnp.array([jnp.cos(psi), jnp.sin(psi), 0.0])
    w = jnp.cross(ER, u)
    n = jnp.cos(phi) * ER + jnp.sin(phi) * w
    v = jnp.cross(n, u)
    return u, w, n, v


def starlink_plates(shape, psi, phi):
    """Пластины Starlink V2 mini: шина Lb×Wb (горизонтальна) + два крыла Lw×Ww на концах шины по оси u."""
    Lw, Ww, Lb, Wb = shape
    u, w, n, v = axes(psi, phi)
    off = (Lb / 2 + Lw / 2) * u
    return [(off, u, v, Lw, Ww, S_W, T_W), (-off, u, v, Lw, Ww, S_W, T_W), (jnp.zeros(3), u, w, Lb, Wb, S_B, T_B)]


def compact_plates(area, psi):
    """Точечная (неразрешённая) альтернатива той же площади: горизонтальный квадрат."""
    a = jnp.sqrt(area)
    u, w, n, v = axes(psi, 0.0)
    return [(jnp.zeros(3), u, w, a, a, S_B, T_B)]


def render_occ(plates, Jm, p, X, Y, C):
    """Доля пикселя, закрытая объектом. Jm: (2,3) м/м; p: (2,) центр; X,Y: (H,W) центры пикселей;
    C: (2,2) ковариация ядра точки на снимке = ФРТ + ящик пикселя поперёк строки + смаз push-broom (kernel_cov)."""
    occ = jnp.zeros(X.shape)
    det = C[0, 0] * C[1, 1] - C[0, 1] ** 2
    i00, i11, i01 = C[1, 1] / det, C[0, 0] / det, -C[0, 1] / det
    for c, a1, a2, L, W, S, T in plates:
        P3 = c[None, :] + (L * S)[:, None] * a1[None, :] + (W * T)[:, None] * a2[None, :]     # (K,3) м
        P2 = p[None, :] + P3 @ Jm.T                                                              # (K,2) м
        j1 = Jm @ a1; j2 = Jm @ a2
        dA = jnp.abs(j1[0] * j2[1] - j1[1] * j2[0]) * L * W / S.shape[0]                         # площадь точки на снимке
        dx = X[..., None] - P2[:, 0]; dy = Y[..., None] - P2[:, 1]
        q = i00 * dx ** 2 + 2 * i01 * dx * dy + i11 * dy ** 2
        occ = occ + (dA / (2 * jnp.pi * jnp.sqrt(det))) * jnp.exp(-0.5 * q).sum(-1)
    return occ


def kernel_cov(d, sig_psf):
    """Ядро точки на снимке push-broom (THEORY §18). Строка снимка — момент времени; полоса обзора строки бежит по
    земле со скоростью v_скан·t̂, а проекция точки объекта на землю — со скоростью v_G (видимая трасса: положения
    объекта по полосам во времени). Точка остаётся в полосе Δt = res/|v_скан − v_G·t̂| и записывается отрезком
    s = v_G·Δt. Плюс ящик пикселя (res×res) и ФРТ оптики. Для неподвижной земли s = 0 — обычный пиксель."""
    res = d["res"]; t = d["row"]; vg = d["vapp"]
    sv = vg * res / jnp.maximum(jnp.abs(V_SCAN - vg @ t), 0.05 * V_SCAN)
    return (sig_psf ** 2 + res ** 2 / 12.0) * jnp.eye(2) + jnp.outer(sv, sv) / 12.0


V_SCAN = 6600.0     # скорость сканирования земли Sentinel-2, м/с


def band_loss(occ, r, s, g, m, hub=3.0):
    """Переменная проекция по c_b ≥ 0 и псевдо-Хьюбер (устойчивость к остаткам фона)."""
    w = m / s ** 2
    c = jnp.sum(w * occ * (r + occ * g)) / (jnp.sum(w * occ ** 2) + 1e-12)
    c = jnp.maximum(c, 0.0)
    model = occ * (c - g)
    z = jnp.where(m > 0, (r - model) / s, 0.0)
    return jnp.sum(m * 2 * hub ** 2 * (jnp.sqrt(1 + (z / hub) ** 2) - 1)), c, model


def pass_loss(shape, sig, pp, D, kind="starlink", phi=None):
    """pp = (dθ_x1000 (3,), ψ, φ) — dθ в метрах; D — данные пролёта (список полос). sig = (σ10, σ20[, σ_смаз]).
    phi: если задан — угол крыла фиксирован (плашмя / по Солнцу), pp[4] не используется."""
    dth, psi = pp[:3] / 1000.0, pp[3]
    phi = pp[4] if phi is None else phi
    plates = starlink_plates(shape, psi, phi) if kind == "starlink" else compact_plates(shape[0], psi)
    tot = 0.0
    for d in D:
        p = d["p0"] + d["J"] @ dth
        C = kernel_cov(d, sig[0] if d["res"] <= 10 else sig[1])
        occ = render_occ(plates, d["J"] / 1000.0, p, d["X"], d["Y"], C)
        l, _, _ = band_loss(occ, d["r"], d["s"], d["g"], d["m"])
        tot = tot + l
    return tot


def to_jax(bands, row):
    """row: единичный вектор направления строк снимка (полёта Sentinel) в UTM: (sin ψ, cos ψ), ψ — азимут трассы.
    vapp: видимая скорость объекта на снимке (м/с) — по положениям объекта в полосах и моментам их съёмки (геометрия)."""
    P = np.array([d["p0"] for d in bands.values()]); T = np.array([d["jd"] for d in bands.values()]) * 86400.0
    if len(P) >= 2 and np.ptp(T) > 0:
        A = np.c_[T - T.mean(), np.ones_like(T)]
        vapp = np.array([np.linalg.lstsq(A, P[:, 0], rcond=None)[0][0], np.linalg.lstsq(A, P[:, 1], rcond=None)[0][0]])
    else:
        vapp = np.array(row, float) * V_SCAN
    out = []
    for b, d in bands.items():
        m = np.isfinite(d["r"]).astype(float)
        out.append(dict(b=b, res=float(d["res"]), p0=jnp.asarray(d["p0"], float), J=jnp.asarray(d["J"], float), row=jnp.asarray(row, float), vapp=jnp.asarray(vapp, float),
                        X=jnp.asarray(d["X"], float), Y=jnp.asarray(d["Y"], float), r=jnp.asarray(np.nan_to_num(d["r"]), float),
                        s=float(d["s"]), g=jnp.asarray(np.nan_to_num(d["g"]), float), m=jnp.asarray(m)))
    return out


def sun_phi(sun, psi):
    """Угол поворота крыла, при котором нормаль смотрит на Солнце (слежение за Солнцем)."""
    u = np.array([math.cos(psi), math.sin(psi), 0.0]); w = np.cross([0, 0, 1.0], u)
    return math.atan2(float(sun @ w), float(sun[2]))


# ---------------------------------------------------------------------------------------------------------------
# Непараметрическая форма для объектов без модели (засекреченные): смесь K трёхмерных гауссиан (3D Gaussian
# Splatting, arXiv:2308.04079; EWA-сплаттинг). Аффинная камера J_b переводит гауссиану N(μ, Σ) тела в гауссиану
# N(p + Jμ, JΣJᵀ) на снимке — в замкнутой форме, без растеризации. Покрытие пикселя: occ = 1 − exp(−Σ_k a_k·N_k),
# a_k — площадь компоненты на снимке. Форма задана в осях орбиты (вдоль, поперёк, радиус): у аппаратов с
# ориентацией на Землю она одна и та же во всех пролётах — так кадры разных пролётов складываются в одну модель.
# ---------------------------------------------------------------------------------------------------------------

def gauss_params(theta, K):
    """theta: (K, 10) = μ (3, м), log-масштабы (3), углы (3: рысканье, тангаж, крен), log a (1)."""
    T = theta.reshape(K, 10)
    mu = T[:, :3]; sc = jnp.exp(T[:, 3:6]); ang = T[:, 6:9]; la = T[:, 9]
    cy, sy = jnp.cos(ang[:, 0]), jnp.sin(ang[:, 0]); cp, sp = jnp.cos(ang[:, 1]), jnp.sin(ang[:, 1]); cr, sr = jnp.cos(ang[:, 2]), jnp.sin(ang[:, 2])
    o, z = jnp.ones_like(cy), jnp.zeros_like(cy)
    Rz = jnp.stack([jnp.stack([cy, -sy, z], -1), jnp.stack([sy, cy, z], -1), jnp.stack([z, z, o], -1)], -2)
    Ry = jnp.stack([jnp.stack([cp, z, sp], -1), jnp.stack([z, o, z], -1), jnp.stack([-sp, z, cp], -1)], -2)
    Rx = jnp.stack([jnp.stack([o, z, z], -1), jnp.stack([z, cr, -sr], -1), jnp.stack([z, sr, cr], -1)], -2)
    R = Rz @ Ry @ Rx
    Sig = R @ (sc[:, :, None] ** 2 * jnp.swapaxes(R, -1, -2))
    return mu, Sig, jnp.exp(la)


def render_gauss(theta, K, Jm, p, X, Y, C):
    mu, Sig, a = gauss_params(theta, K)
    m2 = p[None, :] + mu @ Jm.T                                         # (K,2)
    S2 = Jm[None] @ Sig @ Jm.T[None] + C[None]                         # (K,2,2): EWA — ковариации складываются
    det = S2[:, 0, 0] * S2[:, 1, 1] - S2[:, 0, 1] ** 2
    i00, i11, i01 = S2[:, 1, 1] / det, S2[:, 0, 0] / det, -S2[:, 0, 1] / det
    dx = X[..., None] - m2[:, 0]; dy = Y[..., None] - m2[:, 1]
    q = i00 * dx ** 2 + 2 * i01 * dx * dy + i11 * dy ** 2
    dens = (a / (2 * jnp.pi * jnp.sqrt(det))) * jnp.exp(-0.5 * q)
    return 1.0 - jnp.exp(-dens.sum(-1))


def pass_loss_gauss(theta, K, sig, dth_m, D):
    dth = dth_m / 1000.0
    tot = 0.0
    for d in D:
        p = d["p0"] + d["J"] @ dth
        occ = render_gauss(theta, K, d["J"] / 1000.0, p, d["X"], d["Y"], kernel_cov(d, sig[0] if d["res"] <= 10 else sig[1]))
        l, _, _ = band_loss(occ, d["r"], d["s"], d["g"], d["m"])
        tot = tot + l
    return tot

"""Вращение объекта внутри одного пролёта (THEORY §24, PLAN_characterization §2.3 п. 4).

Полосы Sentinel-2 сняты в разные моменты t_b (разнос до ~2 с), поэтому вращающееся тело попадает в каждую полосу в
своей позе: R(t) = exp([ω]× (t − t₀)) · R₀ (ось и скорость постоянны за ~2 с; поворот осей орбиты за это время ~0,1°).
На пикселе 10 м поворот почти не меняет форму пятна, но меняет его яркость (площадь сечения, освещённость граней).
Поэтому яркость материалов связывается между полосами допущением «серого тела»: отражательная способность группы
материалов одна во всех полосах (краска, алюминий, ткань ступеней и обломков почти серые). Тогда изменение яркости
между полосами — это вращение, а не свободный параметр.

Параметры: R₀ — вектор поворота (3), ω — угловая скорость в орбитальных осях (3, рад/с), сдвиг положения (2).
Сравнение «вращается / не вращается» — та же функция потерь при ω = 0; погрешность ω — по кривизне невязки.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize, nnls

from s2sat import autopose as AP


def band_times(rc):
    """Моменты съёмки полос относительно средней, с."""
    jd = {b: float(d["jd"]) for b, d in rc["bands"].items()}
    j0 = float(np.median(list(jd.values())))
    return {b: (v - j0) * 86400.0 for b, v in jd.items()}


class Spin:
    def __init__(self, rc, spec, C=None, kappa=1.0):
        self.P = AP.Problem(rc, spec, C); self.P.kappa = kappa
        self.spec, self.rc = spec, rc
        self.t = band_times(rc)
        self.bands = self.P.bands

    def pose(self, x, t):
        """Вершины (оси реконструкции) в момент t: R(t) = exp(ω t) R₀ M2B."""
        R = AP.rotvec(np.asarray(x[3:6]) * t) @ AP.rotvec(x[0:3]) @ self.spec["M2B"]
        return (self.spec["V"] @ R.T) @ self.P.C.T, R

    def loss(self, x, bands=None, prior=True):
        """Совместная по полосам невязка при яркостях групп, общих для всех полос («серое тело», НМНК)."""
        bands = bands or self.bands
        rows, ys, ss = [], [], []
        for b in bands:
            d = self.rc["bands"][b]
            Vb, _ = self.pose(x, self.t[b])
            occ = AP.regress(d, Vb, self.spec["F"], self.spec["G"], self.P.ng, self.P.los, np.r_[x[6], x[7], 0.0])
            ok = np.isfinite(d["r"]) & np.isfinite(d["g"])
            O = np.stack([q[ok] for q in occ], 1); s = float(d["s"])
            gm = float(np.nanmedian(d["g"]))
            rows.append(O * gm / s)                       # яркость группы — в долях медианы фона полосы
            ys.append((d["r"][ok] + d["g"][ok] * O.sum(1)) / s); ss.append(np.full(ok.sum(), 1.0))
        A = np.vstack(rows); y = np.concatenate(ys)
        a = nnls(A, y)[0] if A.max() > 1e-9 else np.zeros(A.shape[1])
        z = y - A @ a
        L = float((2 * 9.0 * (np.sqrt(1 + (z / 3.0) ** 2) - 1)).sum()) / self.P.kappa
        if prior:
            L += float(x[6] ** 2 + x[7] ** 2) / AP.SIG_POS ** 2
        return L, a

    def fit(self, x0, idx, maxfev=400, bands=None):
        x0 = np.asarray(x0, float)

        def f(u):
            x = x0.copy(); x[idx] = u
            return self.loss(x, bands)[0]
        r = minimize(f, x0[idx], method="Powell", options=dict(maxfev=maxfev, xtol=1e-3, ftol=1e-5))
        x = x0.copy(); x[idx] = r.x
        return x, r.fun

    def starts(self, n_dir=40):
        """Стартовые R₀: направления оси тела по сфере (лучшие по невязке при ω = 0); ω: 0 и перебор
        скоростей 15–120 °/с вокруг осей, перпендикулярных продольной (плоское кувыркание) и вдоль неё."""
        k = np.arange(n_dir) + 0.5; el = np.arcsin(k / n_dir); az = math.pi * (1 + 5 ** 0.5) * k
        cand = []
        for a_, e_ in zip(az, el):
            d = np.array([math.cos(e_) * math.cos(a_), math.cos(e_) * math.sin(a_), math.sin(e_)])
            Rm = AP._align(self.spec["M2B"] @ np.array([1.0, 0, 0]), d)
            from scipy.spatial.transform import Rotation as Ro
            cand.append(np.r_[Ro.from_matrix(Rm).as_rotvec(), 0, 0, 0, 0, 0])
        sc = [self.loss(c)[0] for c in cand]
        return [cand[i] for i in np.argsort(sc)[:3]]

    def solve(self, maxfev=400, log=print):
        """Поза без вращения и с вращением; значимость вращения (Δ эфф. χ²), ω и погрешность."""
        best0 = None
        for s in self.starts():
            x, L = self.fit(s, [0, 1, 2, 6, 7], maxfev)
            if best0 is None or L < best0[1]:
                best0 = (x, L)
        x0, L0 = best0
        log(f"   без вращения: невязка {L0:.1f}")
        best = (x0, L0)
        long_axis = AP.rotvec(x0[0:3]) @ self.spec["M2B"] @ np.array([1.0, 0, 0])
        perp = np.cross(long_axis, [0.0, 0, 1]); perp = perp / (np.linalg.norm(perp) + 1e-12); perp2 = np.cross(long_axis, perp)
        for axis in (perp, perp2, long_axis):
            for rate in (15, 40, 90):
                for sg in (1, -1):
                    s = x0.copy(); s[3:6] = axis * math.radians(rate) * sg
                    x, L = self.fit(s, [0, 1, 2, 3, 4, 5, 6, 7], maxfev)
                    if L < best[1]:
                        best = (x, L)
        x1, L1 = best
        w = x1[3:6]; rate = math.degrees(np.linalg.norm(w))
        # погрешность компонент ω по кривизне (Δχ² = 1)
        sig = []
        h = math.radians(2.0)
        for k in range(3):
            xp, xm = x1.copy(), x1.copy(); xp[3 + k] += h; xm[3 + k] -= h
            H = (self.loss(xp)[0] - 2 * L1 + self.loss(xm)[0]) / h ** 2
            sig.append(math.degrees(math.sqrt(2 / H)) if H > 1e-9 else float("inf"))
        sig_rate = float(np.sqrt(np.sum((np.asarray(sig) * (w / (np.linalg.norm(w) + 1e-12))) ** 2))) if rate > 0 else float("inf")
        axis = (w / np.linalg.norm(w)).tolist() if rate > 1e-6 else None
        out = dict(L_static=L0, L_spin=L1, gain=L0 - L1, rate_deg_s=rate, rate_sigma_deg_s=sig_rate, axis_lvlh=axis,
                   omega_sigma_deg_s=sig, x=x1.tolist(), x_static=x0.tolist(),
                   spinning=bool(L0 - L1 > 10.0 and rate > 2 * sig_rate),
                   span_s=float(max(self.t.values()) - min(self.t.values())))
        log(f"   с вращением: невязка {L1:.1f} (выигрыш {L0 - L1:.1f}), скорость {rate:.1f} ± {sig_rate:.1f} °/с, "
            f"съёмка полос за {out['span_s']:.2f} с")
        return out

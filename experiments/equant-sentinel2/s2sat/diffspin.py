"""Вращение внутри пролёта градиентным спуском по дифференцируемой модели (s2sat/diffmodel.py).

R(t) = exp([ω]× t) · R₀ (оси орбиты), модель в осях реконструкции: Q(t) = C · R(t) · M2B; параметры R₀ (3), ω (3),
сдвиг (2, м объекта). Пакет из B стартов (случайные R₀ и ω, скорости 3–180 °/с, плюс ω = 0) — Adam, затем дожим
L-BFGS лучших. Невязка — «серое тело» (яркость групп общая для полос), как в s2sat/spin.py.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from s2sat import autopose as AP
from s2sat.diffmodel import Scene, rotvec_to_mat


class DiffSpin:
    def __init__(self, rc, spec, C, kappa=1.0, seed=0):
        self.sc = Scene(rc, spec, C); self.kappa = kappa; self.rng = np.random.default_rng(seed)
        self.t = self.sc.t

    def loss_batch(self, x, bands=None, prior=True):
        R0 = rotvec_to_mat(x[:, 0:3]); w = x[:, 3:6]
        C = self.sc.C

        def Rfun(b):
            return C @ rotvec_to_mat(w * self.t[b]) @ R0
        dpos = torch.cat([x[:, 6:8], torch.zeros(x.shape[0], 1, dtype=x.dtype)], 1)
        L = self.sc.loss(Rfun, dpos, self.kappa, grey=True, bands=bands)
        if prior:
            L = L + (x[:, 6] ** 2 + x[:, 7] ** 2) / AP.SIG_POS ** 2
        return L

    def starts(self, B=64):
        from scipy.spatial.transform import Rotation as Ro
        R0 = Ro.random(B, random_state=int(self.rng.integers(1 << 30))).as_rotvec()
        d = self.rng.normal(size=(B, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        rate = np.exp(self.rng.uniform(math.log(3), math.log(180), B)); rate[: B // 8] = 0.0
        return np.c_[R0, d * np.radians(rate)[:, None], np.zeros((B, 2))]

    def fit(self, x0, iters=250, lr=0.03, bands=None, log=None):
        scale = torch.tensor([0.3, 0.3, 0.3, 0.15, 0.15, 0.15, 20.0, 20.0])
        u = (torch.tensor(np.asarray(x0, float), dtype=torch.float32) / scale).requires_grad_(True)
        opt = torch.optim.Adam([u], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for k in range(iters):
            opt.zero_grad(); L = self.loss_batch(u * scale, bands); L.sum().backward(); opt.step(); sched.step()
            if log and k % 50 == 0:
                log(f"      шаг {k}: лучшая невязка {float(L.min()):.1f}")
        with torch.no_grad():
            L = self.loss_batch(u * scale, bands)
        return (u * scale).detach().numpy(), L.detach().numpy()

    def polish(self, x, iters=60, bands=None):
        scale = torch.tensor([0.3, 0.3, 0.3, 0.15, 0.15, 0.15, 20.0, 20.0])
        u = (torch.tensor(np.asarray(x, float)[None], dtype=torch.float32) / scale).requires_grad_(True)
        opt = torch.optim.LBFGS([u], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

        def cl():
            opt.zero_grad(); L = self.loss_batch(u * scale, bands).sum(); L.backward(); return L
        opt.step(cl)
        with torch.no_grad():
            L = float(self.loss_batch(u * scale, bands)[0])
        return (u * scale).detach().numpy()[0], L

    def solve(self, B=64, iters=250, log=print, jackknife=True):
        xs, Ls = self.fit(self.starts(B), iters, log=log)
        order = np.argsort(Ls)
        sols = [self.polish(xs[i]) for i in order[:6]]
        sols.sort(key=lambda q: q[1]); x1, L1 = sols[0]
        # без вращения: тот же спуск с ω = 0 (лучшие R₀ из пакета)
        x0s = xs[order[:8]].copy(); x0s[:, 3:6] = 0.0
        st = self.fit(x0s, iters // 2, log=None)
        xst = st[0][int(np.argmin(st[1]))]
        with torch.no_grad():
            L0 = float(self.loss_batch(torch.tensor(xst[None], dtype=torch.float32))[0])
        w = x1[3:6]; rate = math.degrees(np.linalg.norm(w))
        fam = [math.degrees(np.linalg.norm(x[3:6])) for x, L in sols if L - L1 < 4.0]
        jk = []
        if jackknife and len(self.sc.bands) >= 4:
            for b in self.sc.bands:
                rest = [c for c in self.sc.bands if c != b]
                xj, _ = self.polish(x1, iters=30, bands=rest)
                jk.append(math.degrees(np.linalg.norm(xj[3:6])))
        n = len(jk)
        sig_jk = math.sqrt((n - 1) / n * float(np.sum((np.array(jk) - np.mean(jk)) ** 2))) if n >= 3 else 0.0
        sig = max(sig_jk, (max(fam) - min(fam)) / 2 if len(fam) > 1 else 0.0)
        out = dict(L_static=L0, L_spin=L1, gain=L0 - L1, rate_deg_s=rate, rate_sigma_deg_s=sig, family_rates_deg_s=sorted(fam),
                   axis_lvlh=(w / np.linalg.norm(w)).tolist() if rate > 1e-6 else None, x=x1.tolist(),
                   spinning=bool(L0 - L1 > 10.0 and rate > 2 * max(sig, 0.5)),
                   span_s=float(max(self.t.values()) - min(self.t.values())))
        log(f"   градиентный спуск: без вращения {L0:.1f}, с вращением {L1:.1f} (выигрыш {L0 - L1:.1f}), "
            f"скорость {rate:.1f} ± {sig:.1f} °/с, семейство {[round(v) for v in sorted(fam)]}")
        return out

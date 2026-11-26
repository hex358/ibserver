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

    def loss_batch(self, x, bands=None, prior=True, t0=0.0):
        """R(t) = exp([ω]× (t − t0)) · R₀."""
        R0 = rotvec_to_mat(x[:, 0:3]); w = x[:, 3:6]
        C = self.sc.C

        def Rfun(b):
            return C @ rotvec_to_mat(w * (self.t[b] - t0)) @ R0
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

    def fit(self, x0, iters=250, lr=0.03, bands=None, log=None, t0=0.0, mask=None):
        scale = torch.tensor([0.3, 0.3, 0.3, 0.15, 0.15, 0.15, 20.0, 20.0])
        u = (torch.tensor(np.asarray(x0, float), dtype=torch.float32) / scale).requires_grad_(True)
        m = torch.ones(8) if mask is None else torch.tensor(mask, dtype=torch.float32)
        opt = torch.optim.Adam([u], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for k in range(iters):
            opt.zero_grad(); L = self.loss_batch(u * scale, bands, t0=t0); L.sum().backward(); u.grad *= m; opt.step(); sched.step()
            if log and k % 50 == 0:
                log(f"      шаг {k}: лучшая невязка {float(L.min()):.1f}")
        with torch.no_grad():
            L = self.loss_batch(u * scale, bands, t0=t0)
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

    def groups(self, gap=1.0):
        bs = sorted(self.sc.bands, key=lambda b: self.t[b]); out = [[bs[0]]]
        for b in bs[1:]:
            (out[-1].append(b) if self.t[b] - self.t[out[-1][-1]] <= gap else out.append([b]))
        return out

    def group_poses(self, bands, t0, B=24, iters=120, keep=4):
        """Поза в момент группы (полосы сняты за доли секунды — поза однозначна, вращение за это время мало):
        градиентный спуск из B стартов по сфере, ω = 0; различные минимумы (> 15°)."""
        from scipy.spatial.transform import Rotation as Ro
        x0 = np.c_[Ro.random(B, random_state=int(self.rng.integers(1 << 30))).as_rotvec(), np.zeros((B, 5))]
        xs, Ls = self.fit(x0, iters, bands=bands, t0=t0, mask=[1, 1, 1, 0, 0, 0, 1, 1])
        out = []
        for i in np.argsort(Ls):
            R = Ro.from_rotvec(xs[i, 0:3])
            if all((R * Ro.from_rotvec(y[0:3]).inv()).magnitude() > math.radians(15) for y, _ in out):
                out.append((xs[i], float(Ls[i])))
            if len(out) >= keep:
                break
        return out

    def alias_candidates(self, log=print, kmax=2):
        """Раскрытие фазы вращения: поза первой и последней группы → поворот между ними за Δt → ω с точностью до
        целого числа оборотов: ω_k = û (θ + 2πk)/Δt; плюс симметрии тела (развороты на 180° вокруг осей тела —
        у ступеней и панелей поза по силуэту определена с точностью до них)."""
        from scipy.spatial.transform import Rotation as Ro
        grp = self.groups()
        ga, gb = grp[0], grp[-1]
        ta = float(np.mean([self.t[b] for b in ga])); tb = float(np.mean([self.t[b] for b in gb])); dt = tb - ta
        A = self.group_poses(ga, ta); Bp = self.group_poses(gb, tb)
        sym = [Ro.identity()] + [Ro.from_rotvec(math.pi * e) for e in np.eye(3)]
        cands = []
        for xa, _ in A:
            Ra = Ro.from_rotvec(xa[0:3])
            for xb, _ in Bp:
                for S in sym:
                    Rb = Ro.from_rotvec(xb[0:3]) * S
                    phi = (Rb * Ra.inv()).as_rotvec(); th = np.linalg.norm(phi); u = phi / th if th > 1e-9 else np.array([0.0, 0, 1])
                    for k in range(-kmax, kmax + 1):
                        w = u * (th + 2 * math.pi * k) / dt
                        if np.degrees(np.linalg.norm(w)) > 200:
                            continue
                        R0 = Ro.from_rotvec(-w * ta) * Ra                        # R(t) = exp(ω t) R0, R(ta) = Ra
                        cands.append(np.r_[R0.as_rotvec(), w, 0.5 * (xa[6:8] + xb[6:8])])
        log(f"   групп полос {len(grp)}, разрыв {dt:.2f} с; поз: {len(A)} и {len(Bp)}; кандидатов ω {len(cands)}")
        return np.array(cands), dt

    def hessian_sigma(self, x):
        """Ковариация параметров по матрице вторых производных невязки (Δχ² = 1 ↔ невязка/2)."""
        xt = torch.tensor(np.asarray(x, float), dtype=torch.float32)
        H = torch.autograd.functional.hessian(lambda z: self.loss_batch(z[None])[0], xt).detach().numpy().astype(float)
        try:
            return np.linalg.inv(0.5 * (H + H.T) / 2.0)
        except np.linalg.LinAlgError:
            return np.full((8, 8), np.nan)

    def solve(self, B=24, iters=150, log=print, jackknife=False):
        """Вращение: раскрытие фазы между группами полос → лучшие кандидаты → градиентный дожим по всем полосам;
        погрешность — по матрице вторых производных; семейство — кандидаты в пределах Δχ² < 4."""
        if len(self.groups()) < 2:
            return self.solve_starts(B=max(B, 32), iters=iters, log=log, jackknife=jackknife)
        cands, dt = self.alias_candidates(log=log)
        with torch.no_grad():
            Lc = np.concatenate([self.loss_batch(torch.tensor(cands[i:i + 64], dtype=torch.float32)).numpy()
                                 for i in range(0, len(cands), 64)])
        top = cands[np.argsort(Lc)[:8]]
        xs, Ls = self.fit(top, iters // 2)
        sols = [self.polish(xs[i]) for i in np.argsort(Ls)[:4]]
        sols.sort(key=lambda q: q[1]); x1, L1 = sols[0]
        x0s = top.copy(); x0s[:, 3:6] = 0.0
        st = self.fit(x0s, iters // 2, mask=[1, 1, 1, 0, 0, 0, 1, 1]); L0 = float(np.min(st[1]))
        Cv = self.hessian_sigma(x1); w = x1[3:6]; rate = math.degrees(np.linalg.norm(w))
        g = w / (np.linalg.norm(w) + 1e-12)
        sig_h = math.degrees(math.sqrt(max(float(g @ Cv[3:6, 3:6] @ g), 0.0))) if np.all(np.isfinite(Cv)) else float("nan")
        fam = sorted(math.degrees(np.linalg.norm(x[3:6])) for x, L in sols if L - L1 < 4.0)
        sig = max(sig_h if math.isfinite(sig_h) else 0.0, (fam[-1] - fam[0]) / 2 if len(fam) > 1 else 0.0)
        out = dict(L_static=L0, L_spin=L1, gain=L0 - L1, rate_deg_s=rate, rate_sigma_deg_s=sig, rate_sigma_hessian=sig_h,
                   family_rates_deg_s=fam, axis_lvlh=(g).tolist() if rate > 1e-6 else None, x=x1.tolist(),
                   omega_cov_deg2=(np.degrees(np.degrees(Cv[3:6, 3:6]))).tolist() if np.all(np.isfinite(Cv)) else None,
                   spinning=bool(L0 - L1 > 10.0 and rate > 2 * max(sig, 0.5)), span_s=float(max(self.t.values()) - min(self.t.values())),
                   method="раскрытие фазы между группами полос + градиент")
        log(f"   раскрытие фазы: без вращения {L0:.1f}, с вращением {L1:.1f} (выигрыш {L0 - L1:.1f}), скорость {rate:.1f} ± {sig:.1f} °/с, "
            f"семейство {[round(v) for v in fam]}")
        return out

    def solve_starts(self, B=64, iters=250, log=print, jackknife=True):
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

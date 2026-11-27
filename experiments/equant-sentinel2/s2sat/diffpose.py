"""Поза объекта по пикселям — без законов ориентации и без выбора из готовых вариантов (THEORY §24, пересмотр).

Параметры: поворот корпуса R (оси тела → орбитальные оси; вся SO(3)), углы ВСЕХ приводов панелей модели (каждое
крыло — своё), сдвиг объекта (2, м). Невязка — дифференцируемая модель снимка (s2sat/diffmodel.py) по всем полосам,
опоры полос — геометрические положения по точной орбите Sentinel (GPS + 3D-углы визирования, scripts/orbit_check.py).
Поиск:
  1. сетка по всей SO(3) (super-Fibonacci) × углы приводов (шаг 60° на [0°, 180°)) — невязка пакетом;
  2. лучшие ячейки — градиентный спуск по всем параметрам сразу (Adam), затем L-BFGS;
  3. различные минимумы (поворот > 20° или привод > 20°) — с невязкой и апостериорной вероятностью.
Откуда известно, что поза правильная (scripts/pose_pix.py):
  * σ по Гессиану в локальных осях поворота (с поправкой κ на несоответствие модели);
  * перекрёстная проверка: поза только по одной части полос предсказывает пиксели остальных (яркость — тоже по
    обучающим полосам), сравнение с «объекта нет»;
  * подстановка: снимок, нарисованный моделью в известной позе с шумом снимка, — восстанавливается ли поза.
Законы ориентации — только для описания найденного (угол до ближайшего «на Землю» с любым рысканьем), не на входе.
Вращение — тот же объект, поза R(t) = exp([ω]× (t − t₀)) R₀ (s2sat/diffspin.py).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from s2sat import autopose as AP
from s2sat.diffmodel import Scene, rotvec_to_mat


def so3_grid(n):
    """Почти равномерная сетка поворотов (super-Fibonacci, Alexa 2022) → векторы поворота."""
    from scipy.spatial.transform import Rotation as Ro
    phi, psi = math.sqrt(2.0), 1.533751168755204288118041
    s = np.arange(n) + 0.5; r = np.sqrt(s / n); R = np.sqrt(1.0 - s / n)
    al, be = 2 * math.pi * s / phi, 2 * math.pi * s / psi
    q = np.c_[r * np.sin(al), r * np.cos(al), R * np.sin(be), R * np.cos(be)]
    return Ro.from_quat(q).as_rotvec()


def face_parts(spec):
    """Подвижная часть каждой грани (−1 — корпус) и цепочка приводов части [(ось, точка, номер привода)]: сначала
    второй привод (вокруг длины крыла), затем первый — как autopose.apply_drives."""
    F = np.asarray(spec["F"]); L = AP.drive_layout(spec); part = -np.ones(len(F), int); chains = []
    for i, ar in enumerate(spec["arrays"]):
        inw = np.asarray(ar["mask"], bool)[F].all(1)
        prim = (np.asarray(ar["axis"], float), np.asarray(ar["point"], float), L["prim"][i])
        for k, sw in zip(L["sub"][i], ar.get("sub") or []):
            ms = np.asarray(sw["mask"], bool)[F].all(1) & inw & (part < 0)
            part[ms] = len(chains); chains.append([(np.asarray(sw["axis"], float), np.asarray(sw["point"], float), k), prim])
        rest = inw & (part < 0)
        part[rest] = len(chains); chains.append([prim])
    return part, chains


def rot_angle(Ra, Rb):
    return math.degrees(math.acos(float(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))


class DiffPose:
    """x = [вектор поворота корпуса (оси тела → орбитальные оси) 3, углы приводов n, сдвиг 2]."""

    def __init__(self, rc, spec, C, shade=False, kappa=1.0, lmax=None, coarse=False, band_off=None):
        """coarse — для глобального поиска: окно по размаху объекта, пятна ≤ 3 м, сетка σ/1 (diffmodel)."""
        part, chains = face_parts(spec)
        # грубая модель — те же группы материалов, что и точная: при одной группе разные по знаку контрасты (светлый
        # корпус, тёмные панели) гасили друг друга, и глобальный перебор не видел правильную позу (истина лучше на 484)
        import os as _os
        ng_c = 1 if (coarse and _os.environ.get("S2_COARSE_NG", "1") == "1") else None
        self.sc = sc = Scene(rc, spec, C, parts=part, lmax=3.0 if coarse else lmax, ng=ng_c, coarse=coarse)
        self.spec, self.shade, self.kappa = spec, shade, kappa
        self.nd = AP.n_drives(spec); self.P = 3 + self.nd + 2
        self.M2B = torch.tensor(np.asarray(spec["M2B"], float), dtype=torch.float32)
        self.cm = torch.tensor(sc.c_model, dtype=torch.float32); self.Sm = torch.tensor(sc.S_model, dtype=torch.float32)
        p = sc.part; self.nb = int(np.sum(p < 0))
        self.sl = [(int(np.searchsorted(p, j, "left")), int(np.searchsorted(p, j, "right"))) for j in range(len(chains))]
        self.chains = [[(torch.tensor(a / np.linalg.norm(a), dtype=torch.float32), torch.tensor(pt, dtype=torch.float32), k)
                        for a, pt, k in ch] for ch in chains]
        self.npix = int(sum(len(sc.B[b].r) for b in sc.bands))
        self.scale = torch.tensor([0.3] * 3 + [0.3] * self.nd + [3.0, 3.0])
        self.Rbase = torch.eye(3)
        # band_off = σ, м: опора каждой полосы — свободный сдвиг на снимке с априорной ошибкой σ (разброс положений
        # полос вокруг точной орбиты ~24 м, scripts/orbit_check.py); параметры — в конце x (2 на полосу)
        self.band_off = band_off; self.nbd = len(sc.bands)
        if band_off:
            self.P += 2 * self.nbd; self.scale = torch.cat([self.scale, torch.full((2 * self.nbd,), 5.0)])
        self.o0 = 3 + self.nd + 2
        self.grey = True          # False — яркости материалов свои в каждой полосе (спектр не серый)

    # --- модель ---
    def spots(self, x, Rt=None):
        """→ центры и векторные площади пятен в осях реконструкции (B, m, 3). Rt (B, 3, 3) — поворот за время с t₀
        (вращение), в орбитальных осях, слева от позы корпуса."""
        B = x.shape[0]
        R = rotvec_to_mat(x[:, 0:3]) @ self.Rbase @ self.M2B
        R = self.sc.C @ (R if Rt is None else Rt @ R)
        cs, Ss = [self.cm[:self.nb].expand(B, -1, -1)], [self.Sm[:self.nb].expand(B, -1, -1)]
        for (lo, hi), ch in zip(self.sl, self.chains):
            c = self.cm[lo:hi].expand(B, -1, -1); S = self.Sm[lo:hi].expand(B, -1, -1)
            for ax, pt, k in ch:
                Rd = rotvec_to_mat(ax[None, :] * x[:, 3 + k:4 + k])
                c = torch.einsum("bij,bmj->bmi", Rd, c - pt) + pt; S = torch.einsum("bij,bmj->bmi", Rd, S)
            cs.append(c); Ss.append(S)
        c = torch.cat(cs, 1); S = torch.cat(Ss, 1)
        return torch.einsum("bij,bmj->bmi", R, c), torch.einsum("bij,bmj->bmi", R, S)

    def dpos(self, x):
        return torch.cat([x[:, self.o0 - 2:self.o0], torch.zeros(x.shape[0], 1, dtype=x.dtype)], 1)

    def off(self, x, b):
        if not self.band_off or x.shape[1] < self.o0 + 2 * self.nbd:
            return None
        k = self.sc.bands.index(b); return x[:, self.o0 + 2 * k:self.o0 + 2 * k + 2]

    def loss(self, x, bands=None, prior=True, shade=None):
        shade = self.shade if shade is None else shade
        cw, Sw = self.spots(x); dp = self.dpos(x)
        L = self.sc.loss(None, dp, self.kappa, grey=self.grey, bands=bands, shade=shade,
                         occfun=lambda b, sh: self.sc.band_occ_cs(b, cw, Sw, dp, sh, off=self.off(x, b)))
        if prior:
            L = L + (x[:, self.o0 - 2] ** 2 + x[:, self.o0 - 1] ** 2) / AP.SIG_POS ** 2
            if self.band_off and x.shape[1] >= self.o0 + 2 * self.nbd:
                L = L + (x[:, self.o0:self.o0 + 2 * self.nbd] ** 2).sum(1) / self.band_off ** 2
        return L

    def L(self, x, bands=None, prior=True):
        with torch.no_grad():
            return self.loss(torch.tensor(np.atleast_2d(x), dtype=torch.float32), bands, prior).numpy()

    def null(self, bands=None):
        bands = bands or self.sc.bands; z = torch.cat([self.sc.B[b].r / self.sc.B[b].s for b in bands])
        return float((2 * 9.0 * (torch.sqrt(1 + (z / 3.0) ** 2) - 1)).sum())

    # --- подгонка ---
    def fit(self, x0, iters=150, lr=0.04, bands=None, chunk=64):
        out_x, out_L = [], []
        for i in range(0, len(x0), chunk):
            u = (torch.tensor(np.asarray(x0[i:i + chunk], float), dtype=torch.float32) / self.scale).requires_grad_(True)
            opt = torch.optim.Adam([u], lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
            for _ in range(iters):
                opt.zero_grad(); self.loss(u * self.scale, bands).sum().backward(); opt.step(); sch.step()
            xs = (u * self.scale).detach().numpy(); out_x.append(xs); out_L.append(self.L(xs, bands))
        return np.concatenate(out_x), np.concatenate(out_L)

    def polish(self, x, iters=80, bands=None):
        u = (torch.tensor(np.asarray(x, float)[None], dtype=torch.float32) / self.scale).requires_grad_(True)
        opt = torch.optim.LBFGS([u], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

        def cl():
            opt.zero_grad(); L = self.loss(u * self.scale, bands).sum(); L.backward(); return L
        opt.step(cl)
        xs = (u * self.scale).detach().numpy()[0]
        return xs, float(self.L(xs, bands)[0])

    def grid(self, n_rot=3000, step_deg=60.0):
        D = np.radians(np.arange(0.0, 180.0, step_deg))
        if self.nd and len(D) ** self.nd > 27:              # много приводов (МКС): 9 случайных наборов углов на поворот
            Dd = np.random.default_rng(1).uniform(0, math.pi, (9, self.nd))
        else:
            Dd = np.array(list(itertools.product(D, repeat=self.nd))) if self.nd else np.zeros((1, 0))
        R = so3_grid(n_rot)
        return np.c_[np.repeat(R, len(Dd), 0), np.tile(Dd, (len(R), 1)), np.zeros((len(R) * len(Dd), self.P - 3 - self.nd))]

    def search(self, bands=None, n_rot=3000, step_deg=60.0, top=256, iters=150, n_polish=16, log=print, fine=None, twins=False):
        """Глобально: вся SO(3) × приводы → градиент по всем параметрам → различные минимумы. Сетка и спуск — на этой
        (грубой) модели, дожим — на fine (мелкие пятна), если задана.
        Сначала — сдвиг объекта: остаток орбиты даёт до ~10 м в осях объекта (×4–6 на снимке — несколько пикселей),
        и перебор поз при нулевом сдвиге ищет позу смещённого объекта. Сдвиг — сетка ±16 м (шаг 4 м) × 100 поз по
        SO(3), лучший — исходный для всей сетки поз (градиент его доводит)."""
        X = self.grid(n_rot, step_deg); Ls = []
        Xp = self.grid(100, step_deg)[::max(1, len(self.grid(1, step_deg)))]           # 100 поворотов, одни приводы
        sh = np.arange(-16.0, 16.01, 4.0); best = (None, 0.0, 0.0)
        for dx in sh:
            for dy in sh:
                Xs = Xp.copy(); Xs[:, self.o0 - 2] = dx; Xs[:, self.o0 - 1] = dy
                Lm_ = float(self.L(Xs, bands).min())
                if best[0] is None or Lm_ < best[0]:
                    best = (Lm_, dx, dy)
        # перебор поз — и при найденном сдвиге, и при нулевом: у слабых наборов полос (1–2 полосы) минимум по сетке
        # сдвигов может подогнаться под шум; лучшие позы отбираются из обоих
        if abs(best[1]) + abs(best[2]) > 0:
            X0 = X.copy(); X[:, self.o0 - 2] = best[1]; X[:, self.o0 - 1] = best[2]; X = np.vstack([X0, X])
        log(f"   сдвиг объекта до перебора поз: ({best[1]:+.0f}, {best[2]:+.0f}) м (перебор при нём и при нулевом)")
        for i in range(0, len(X), 512):
            Ls.append(self.L(X[i:i + 512], bands))
        Ls = np.concatenate(Ls); o = np.argsort(Ls)[:top]
        log(f"   сетка: {len(X)} поз (SO(3) {n_rot} × приводы {len(X) // n_rot}), лучшая невязка {Ls[o[0]]:.0f}, "
            f"без объекта {self.null(bands):.0f}")
        xs, Lf = self.fit(X[o], iters=iters, bands=bands)
        o2 = np.argsort(Lf)[:n_polish]; F = fine or self; res = [F.polish(xs[k], bands=bands) for k in o2]
        res.sort(key=lambda q: q[1])
        md = F.modes(res)
        if not twins:
            return md, Ls
        tw = [F.polish(t, bands=bands) for m in md[:4] for t in F.necker_twins(m["x"])]   # двойники Неккера
        return F.modes(sorted(res + tw, key=lambda q: q[1])), Ls

    def necker_twins(self, x):
        """Двойники глубины ортографической проекции: тело, отражённое относительно плоскости ⊥ визиру (M = I − 2llᵀ),
        с отражением одной из осей тела D_k (чтобы поворот остался собственным): R' = M R D_k — силуэт тот же (точно для
        плоских частей, ⊥ оси k), глубина обратная. Углы приводов при отражении меняют знак."""
        from scipy.spatial.transform import Rotation as Ro
        l = self.sc.C.numpy().T @ (self.sc.los_t.numpy()); l = l / np.linalg.norm(l); M = np.eye(3) - 2 * np.outer(l, l)
        R = self.R_of(x); out = []
        for k in range(3):
            D = np.eye(3); D[k, k] = -1.0
            Rt = M @ R @ D
            out.append(np.r_[Ro.from_matrix(Rt).as_rotvec(), -np.asarray(x[3:3 + self.nd]), np.asarray(x[3 + self.nd:])])
        return out

    def R_of(self, x):
        return AP.rotvec(np.asarray(x[:3], float)) @ self.Rbase.numpy()

    def modes(self, res, rot_deg=20.0, drv_deg=20.0):
        out = []
        for x, L in res:
            R = self.R_of(x); d = np.asarray(x[3:3 + self.nd])
            dup = False
            for m in out:
                dd = np.abs((np.degrees(d - m["drives"]) + 90) % 180 - 90) if self.nd else np.zeros(0)
                if rot_angle(R, m["R"]) < rot_deg and (dd < drv_deg).all():
                    dup = True; break
            if not dup:
                out.append(dict(x=np.asarray(x), L=float(L), R=R, drives=d))
        return out

    # --- проверки ---
    def hessian_sigma(self, x, bands=None):
        """σ в локальных осях: R = exp([δ]×) · R̂ — δ по орбитальным осям (вдоль, поперёк, радиус), рад; приводы; сдвиг.
        Гессиан — центральными разностями по невязке (шаг ~1°, 0,5 м: выше шума float32), одним пакетом. Отрицательные
        собственные значения — найденная точка не минимум по этому направлению (сообщается, σ = ∞)."""
        Rb = self.Rbase; self.Rbase = torch.tensor(self.R_of(x), dtype=torch.float32)
        u0 = np.r_[np.zeros(3), np.asarray(x[3:], float)]; P = len(u0)
        h = np.r_[[0.02] * 3, [0.02] * self.nd, [0.5, 0.5], [2.0] * (P - 5 - self.nd)][:P]
        pts = [u0]; idx = {}
        for i in range(P):
            for j in range(i, P):
                for si in (1, -1):
                    for sj in (1, -1):
                        if i == j and sj == -1:
                            continue
                        u = u0.copy(); u[i] += si * h[i]
                        if i != j:
                            u[j] += sj * h[j]
                        else:
                            u[i] += si * h[i] * 0.0
                        idx[(i, j, si, sj)] = len(pts); pts.append(u)
        try:
            Lp = np.concatenate([self.L(np.array(pts[k:k + 128]), bands) for k in range(0, len(pts), 128)]).astype(float)
        finally:
            self.Rbase = Rb
        f0 = Lp[0]; H = np.zeros((P, P))
        for i in range(P):
            H[i, i] = (Lp[idx[(i, i, 1, 1)]] - 2 * f0 + Lp[idx[(i, i, -1, 1)]]) / h[i] ** 2
            for j in range(i + 1, P):
                H[i, j] = H[j, i] = (Lp[idx[(i, j, 1, 1)]] - Lp[idx[(i, j, 1, -1)]] - Lp[idx[(i, j, -1, 1)]] + Lp[idx[(i, j, -1, -1)]]) / (4 * h[i] * h[j])
        kap = max(1.0, f0 / max(1, self.npix - self.P))
        w, V = np.linalg.eigh(H)
        self.neg_dirs = [V[:, k] for k in range(P) if w[k] <= 0]
        wi = np.where(w > 0, 1.0 / np.maximum(w, 1e-12), np.inf)
        Cov = 2.0 * kap * (V * wi) @ V.T if np.isfinite(wi).all() else np.full((P, P), np.inf)
        sd = np.sqrt(np.abs(np.diag(Cov))) if np.isfinite(wi).all() else np.array([np.sqrt(2 * kap / H[k, k]) if H[k, k] > 0 else np.inf for k in range(P)])
        return sd, Cov, kap

    def columns(self, x, b, shade=None):
        shade = self.shade if shade is None else shade
        with torch.no_grad():
            xt = torch.tensor(np.atleast_2d(x), dtype=torch.float32); cw, Sw = self.spots(xt)
            res = self.sc.band_occ_cs(b, cw, Sw, self.dpos(xt), shade, off=self.off(xt, b))
        if shade:
            occ, lit, sp = res; cols = torch.cat([lit, sp], 1)
        else:
            occ = res; cols = occ
        return occ[0].numpy(), cols[0].numpy()

    def predict(self, x, train, test):
        """Яркость групп (НК) только по train; предсказание пикселей test. → по полосе: χ² без объекта / с моделью и
        доля объяснённого сигнала 1 − Σ(r − m)² / Σ r²."""
        A, y = [], []
        for b in train:
            bd = self.sc.B[b]; occ, cols = self.columns(x, b)
            A.append(cols.T * bd.gm / bd.s); y.append((bd.r.numpy() + bd.g.numpy() * occ.sum(0)) / bd.s)
        A = np.concatenate(A); y = np.concatenate(y)
        from scipy.optimize import nnls
        rho = nnls(A, y)[0]                                 # точные неотрицательные НК (как в невязке)
        out = {}
        for b in test:
            bd = self.sc.B[b]; occ, cols = self.columns(x, b); r = bd.r.numpy(); g = bd.g.numpy()
            m = cols.T @ rho * bd.gm - g * occ.sum(0)
            out[b] = dict(chi_null=float(np.sum((r / bd.s) ** 2)), chi_model=float(np.sum(((r - m) / bd.s) ** 2)),
                          explained=float(1 - np.sum((r - m) ** 2) / max(1e-9, np.sum(r ** 2))))
        return out, rho

    def synth(self, x, rho, noise, rng):
        """Снимок, нарисованный моделью в позе x (яркости rho, шум noise·s): заменяет r полос. → прежние r."""
        old = {}
        for b in self.sc.bands:
            bd = self.sc.B[b]; occ, cols = self.columns(x, b)
            m = cols.T @ rho * bd.gm - bd.g.numpy() * occ.sum(0)
            old[b] = bd.r; bd.r = torch.tensor(m + noise * bd.s * rng.standard_normal(len(m)), dtype=torch.float32)
        return old

    def restore(self, old):
        for b, r in old.items():
            self.sc.B[b].r = r

    # --- описание найденного (законы — только здесь, после подгонки) ---
    def nearest_nadir(self, R):
        """Ближайшая ориентация «на Землю» (тело z — зенит) с ЛЮБЫМ рысканьем: угол и рысканье."""
        best = (1e9, 0.0)
        for yaw in np.arange(0.0, 360.0, 0.5):
            d = rot_angle(R, AP.att(math.radians(yaw), 0.0, 0.0))
            if d < best[0]:
                best = (d, yaw)
        return best

    def drive_report(self, x, sun_l):
        """Нормали крыльев против Солнца: угол между нормалью и направлением на Солнце (0° — элементы к Солнцу)."""
        R = self.R_of(x) @ np.asarray(self.spec["M2B"], float); L = AP.drive_layout(self.spec); out = []
        ang = np.asarray(x[3:3 + self.nd], float)
        for i, ar in enumerate(self.spec["arrays"]):
            n = AP.rot_axis(ar["axis"], ang[L["prim"][i]]) @ np.asarray(ar["normal0"], float)
            c = abs(float((R @ n) @ (np.asarray(sun_l) / np.linalg.norm(sun_l))))
            out.append(math.degrees(math.acos(min(1.0, c))))
        return out


class DiffSpinPose(DiffPose):
    """Вращение в пролёте поверх позы по пикселям: R(t) = exp([ω]× (t − t₀)) · R₀, крылья — постоянные углы.
    x = [R₀ 3, приводы n, сдвиг 2, ω 3 (рад/с, орбитальные оси)].
    Поиск (без перебора скоростей по сетке):
      1. поза каждой группы полос (группы — по времени съёмки) — та же глобальная поза по пикселям, все её минимумы;
      2. пара минимумов групп A, B даёт поворот ΔR = R_B R_Aᵀ за Δt — значит ω = n (θ + 2πk) / Δt (n, θ — ось и угол
         ΔR; k — целое: неоднозначность фазы, как у фазовых измерений ГНСС). Симметрии тела не задаются руками: они уже
         среди минимумов поз групп;
      3. каждый кандидат — градиентом по всем параметрам на всех полосах; целое k выбирает разброс времён ВНУТРИ групп
         (короткая база); итог — минимум, семейство кандидатов с близкой невязкой и вероятности."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.P = 3 + self.nd + 2 + 3
        self.scale = torch.cat([self.scale, torch.tensor([0.3, 0.3, 0.3])])
        self.t0 = 0.0

    def loss(self, x, bands=None, prior=True, shade=None):
        shade = self.shade if shade is None else shade
        xs, w = x[:, :-3], x[:, -3:]; dp = self.dpos(xs)

        def occ(b, sh):
            Rt = rotvec_to_mat(w * (self.sc.t[b] - self.t0))
            cw, Sw = self.spots(xs, Rt)
            return self.sc.band_occ_cs(b, cw, Sw, dp, sh)
        L = self.sc.loss(None, dp, self.kappa, grey=self.grey, bands=bands, shade=shade, occfun=occ)
        if prior:
            L = L + (xs[:, -2] ** 2 + xs[:, -1] ** 2) / AP.SIG_POS ** 2
        return L

    def dpos(self, x):
        xs = x[:, :-3] if x.shape[1] == self.P else x
        return torch.cat([xs[:, -2:], torch.zeros(x.shape[0], 1, dtype=x.dtype)], 1)

    def columns(self, x, b, shade=None):
        shade = self.shade if shade is None else shade
        with torch.no_grad():
            xt = torch.tensor(np.atleast_2d(x), dtype=torch.float32); xs, w = xt[:, :-3], xt[:, -3:]
            cw, Sw = self.spots(xs, rotvec_to_mat(w * (self.sc.t[b] - self.t0)))
            res = self.sc.band_occ_cs(b, cw, Sw, self.dpos(xs), shade)
        if shade:
            occ, lit, sp = res; cols = torch.cat([lit, sp], 1)
        else:
            occ = res; cols = occ
        return occ[0].numpy(), cols[0].numpy()

    def groups(self, gap=1.0):
        bs = sorted(self.sc.bands, key=lambda b: self.sc.t[b]); out = [[bs[0]]]
        for b in bs[1:]:
            (out[-1].append(b) if self.sc.t[b] - self.sc.t[out[-1][-1]] <= gap else out.append([b]))
        return out

    def clusters(self, span=0.2):
        """Подгруппы полос с разбросом времён ≤ span: внутри поворот мал даже при быстром вращении (120 °/с × 0,2 с =
        24°), поэтому статическая поза подгруппы верна. Близкие подгруппы (Δt ~0,1–0,4 с) дают ω без неоднозначности
        2πk (поворот < 180° до ~500 °/с), далёкие (Δt ~2 с) — точность; неоднозначность далёких снимает близкая база."""
        bs = sorted(self.sc.bands, key=lambda b: self.sc.t[b]); out = [[bs[0]]]
        for b in bs[1:]:
            (out[-1].append(b) if self.sc.t[b] - self.sc.t[out[-1][0]] <= span else out.append([b]))
        return out

    def candidates(self, gmodes, gt, w_max_deg=200.0, k_modes=4):
        """Кандидаты ω из пар минимумов поз групп (см. класс). gmodes — по группам списки минимумов (x статической
        позы), gt — моменты групп. → список x (опора t₀ — момент самой большой группы)."""
        from scipy.spatial.transform import Rotation as Ro
        out = []; wmax = math.radians(w_max_deg)
        ib = int(np.argmax([len(m) and 1 for m in gmodes]))
        for i in range(len(gmodes)):
            for j in range(i + 1, len(gmodes)):
                dt = gt[j] - gt[i]
                for ma in gmodes[i][:k_modes]:
                    for mb in gmodes[j][:k_modes]:
                        dR = self.R_of(mb) @ self.R_of(ma).T; rv = Ro.from_matrix(dR).as_rotvec(); th = float(np.linalg.norm(rv))
                        nax = rv / th if th > 1e-9 else np.array([0.0, 0, 1])
                        kmax = int(wmax * abs(dt) / (2 * math.pi)) + 1
                        for k in range(-kmax, kmax + 1):
                            ang = th + 2 * math.pi * k
                            if abs(ang / dt) > wmax:
                                continue
                            w = nax * ang / dt
                            base = mb if abs(gt[j] - self.t0) < abs(gt[i] - self.t0) else ma
                            tb = gt[j] if base is mb else gt[i]
                            R0 = AP.rotvec(-w * (tb - self.t0)) @ self.R_of(base)            # поза в t₀
                            out.append(np.r_[Ro.from_matrix(R0).as_rotvec(), base[3:3 + self.nd], base[3 + self.nd:3 + self.nd + 2], w])
        return out

    def mirror_twins(self, x):
        """Зеркальный двойник движения (ортогональная проекция: снимок не различает отражение относительно плоскости ⊥
        визиру): R(t) = exp([ω]× t)·R₀ → M·R(t)·D_k = exp([−Mω]× t)·(M R₀ D_k), M = I − 2llᵀ, D_k — отражение оси тела k.
        Скорость та же, составляющая ω вдоль визира та же, в плоскости снимка — обратная. Различает их только свет."""
        from scipy.spatial.transform import Rotation as Ro
        l = self.sc.C.numpy().T @ self.sc.los_t.numpy(); l = l / np.linalg.norm(l); M = np.eye(3) - 2 * np.outer(l, l)
        R0 = self.R_of(x); w = np.asarray(x[-3:], float); out = []
        for k in range(3):
            D = np.eye(3); D[k, k] = -1.0
            out.append(np.r_[Ro.from_matrix(M @ R0 @ D).as_rotvec(), -np.asarray(x[3:3 + self.nd]), np.asarray(x[3 + self.nd:-3]), -M @ w])
        return out

    def static_x(self, xs):
        return np.r_[xs, 0.0, 0.0, 0.0]


class DiffSpinCoM(DiffSpinPose):
    """Вращение + ЦЕНТР МАСС по фото. Тело без внешних моментов вращается вокруг центра масс, а орбита ведёт именно
    его; поэтому точка тела, которая при вращении стоит на орбите, — центр масс, измеренный по пикселям (без модели
    масс и плотностей; у сломанного объекта — так же). x = [поза…, сдвиг 2, c 3 (м, оси модели), ω 3]:
        p(t) = C·(Rt·R·(q − c) + R·c) = C·Rt·R·q − C·(Rt − I)·R·c,
    при t = t₀ поправка нулевая (не путается со сдвигом) — c видно только по повороту между полосами. Априори c — в
    пределах размера объекта (σ = половина размаха модели)."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.P += 3
        self.scale = torch.cat([self.scale[:-3], torch.tensor([2.0, 2.0, 2.0]), self.scale[-3:]])
        V = np.asarray(self.spec["V"], float); self.sig_com = 0.5 * float(np.ptp(V, 0).max())

    def _split(self, x):
        return x[:, :-6], x[:, -6:-3], x[:, -3:]

    def body_R(self, xs):
        return rotvec_to_mat(xs[:, 0:3]) @ self.Rbase @ self.M2B

    def com_shift(self, xs, c, Rt):
        I3 = torch.eye(3, dtype=xs.dtype)[None]
        return -torch.einsum("ij,bjk,bk->bi", self.sc.C, (Rt - I3) @ self.body_R(xs), c)

    def loss(self, x, bands=None, prior=True, shade=None):
        shade = self.shade if shade is None else shade
        xs, c, w = self._split(x); dp = self.dpos(x)

        def occ(b, sh):
            Rt = rotvec_to_mat(w * (self.sc.t[b] - self.t0))
            cw, Sw = self.spots(xs, Rt); cw = cw + self.com_shift(xs, c, Rt)[:, None, :]
            return self.sc.band_occ_cs(b, cw, Sw, dp, sh)
        L = self.sc.loss(None, dp, self.kappa, grey=self.grey, bands=bands, shade=shade, occfun=occ)
        if prior:
            L = L + (xs[:, -2] ** 2 + xs[:, -1] ** 2) / AP.SIG_POS ** 2 + (c ** 2).sum(1) / self.sig_com ** 2
        if getattr(self, "_cfix", None) is not None:
            L = L + self._cfix[1] * ((c - self._cfix[0][None]) ** 2).sum(1)
        return L

    def dpos(self, x):
        xs = x[:, :-6] if x.shape[1] == self.P else x
        return torch.cat([xs[:, -2:], torch.zeros(x.shape[0], 1, dtype=x.dtype)], 1)

    def columns(self, x, b, shade=None):
        shade = self.shade if shade is None else shade
        with torch.no_grad():
            xt = torch.tensor(np.atleast_2d(x), dtype=torch.float32); xs, c, w = self._split(xt)
            Rt = rotvec_to_mat(w * (self.sc.t[b] - self.t0)); cw, Sw = self.spots(xs, Rt)
            res = self.sc.band_occ_cs(b, cw + self.com_shift(xs, c, Rt)[:, None, :], Sw, self.dpos(xt), shade)
        if shade:
            occ, lit, sp = res; cols = torch.cat([lit, sp], 1)
        else:
            occ = res; cols = occ
        return occ[0].numpy(), cols[0].numpy()

    def from_spin(self, xw, c=(0.0, 0.0, 0.0)):
        return np.r_[np.asarray(xw[:-3], float), np.asarray(c, float), np.asarray(xw[-3:], float)]

    def full_cov(self, x, bands=None, kap=1.0):
        """Ковариация ВСЕХ параметров по полному гессиану невязки (центральные разности одним пакетом; шаги: поворот
        и приводы 0,02 рад, сдвиг 0,5 м, центр масс 2 м, ω 0,005 рад/с — выше шума float32). σ центра масс —
        маргинальная (с подстройкой позы, сдвига, ω), а не при закреплённых остальных (та была занижена в 2–3 раза)."""
        x = np.asarray(x, float); P = len(x); k0 = P - 6
        h = np.r_[[0.02] * 3, [0.02] * self.nd, [0.5, 0.5], [0.0] * (k0 - 5 - self.nd), [2.0] * 3, [0.005] * 3]
        h[h == 0] = 0.5
        pts = [x]; idx = {}
        for i in range(P):
            for j in range(i, P):
                for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    if i == j and (si, sj) in ((1, -1), (-1, 1)):
                        continue
                    u = x.copy(); u[i] += si * h[i]; u[j] += sj * h[j]; idx[(i, j, si, sj)] = len(pts); pts.append(u)
        Lp = np.concatenate([self.L(np.array(pts[k:k + 128]), bands) for k in range(0, len(pts), 128)]).astype(float)
        H = np.zeros((P, P))
        for i in range(P):
            H[i, i] = (Lp[idx[(i, i, 1, 1)]] - 2 * Lp[0] + Lp[idx[(i, i, -1, -1)]]) / (4 * h[i] ** 2)
            for j in range(i + 1, P):
                a_, b_, c_, d_ = (Lp[idx[(i, j, s1, s2)]] for s1, s2 in ((1, 1), (1, -1), (-1, 1), (-1, -1)))
                H[i, j] = H[j, i] = (a_ - b_ - c_ + d_) / (4 * h[i] * h[j])
        ev = np.linalg.eigvalsh(0.5 * H)
        if ev.min() <= 0:
            return None
        return kap * np.linalg.inv(0.5 * H)

    def com_sigma_marginal(self, x, bands=None, kap=1.0):
        """σ центра масс поперёк оси — маргинальная по полному гессиану. → (σ 2, базис 3×2, ковариация 2×2)."""
        nm = self.axis_model(x); e1 = np.cross(nm, [1.0, 0, 0] if abs(nm[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
        E = np.c_[e1, np.cross(nm, e1)]; cov = self.full_cov(x, bands, kap)
        if cov is None:
            return np.full(2, np.inf), E, None
        k0 = len(x) - 6; Cc = E.T @ cov[k0:k0 + 3, k0:k0 + 3] @ E
        return np.sqrt(np.diag(Cc)), E, Cc

    def com_sigma_profile(self, x, bands=None, kap=1.0, d=2.0):
        """σ центра масс поперёк оси по ПРОФИЛЮ невязки: c = ĉ ± d·e_i (e_i ⊥ оси), все остальные параметры дожимаются
        заново (поза, приводы, сдвиг, ω), σ_i² = κ·d²/ΔL_i (ΔL — средний рост по двум знакам). Маргинальная оценка
        без гессиана (у приводов бывают плоские направления — гессиан не положительно определён).
        → (σ 2, базис 3×2)."""
        x = np.asarray(x, float); k0 = len(x) - 6; nm = self.axis_model(x)
        e1 = np.cross(nm, [1.0, 0, 0] if abs(nm[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1); E = np.c_[e1, np.cross(nm, e1)]
        L0 = float(self.L(x, bands)[0]); sig = []
        for i in range(2):
            dl = []
            for sg in (1.0, -1.0):
                cf = x[k0:k0 + 3] + sg * d * E[:, i]; u = x.copy(); u[k0:k0 + 3] = cf
                self._cfix = (torch.tensor(cf, dtype=torch.float32), 1e4)
                try:
                    u, _ = self.polish(u, iters=100, bands=bands)
                finally:
                    self._cfix = None
                u[k0:k0 + 3] = cf; dl.append(float(self.L(u, bands)[0]) - L0)
            dL = float(np.mean(dl)); sig.append(math.sqrt(kap * d * d / dL) if dL > 0 else float("inf"))
        return np.array(sig), E

    def fit_com(self, xw, bands=None, kap=1.0):
        """Центр масс от найденного вращения xw (вокруг начала модели): дожим всех параметров из 7 стартов c = 0 и
        ±σ/3 по осям модели (один старт может остановиться в начале — градиент мал), до сходимости; лучший по невязке.
        → (x, L, L при c = 0, σ(c), ковариация)."""
        x0 = self.from_spin(xw); L0 = float(self.L(x0, bands)[0]); d = self.sig_com / 3.0
        starts = [np.zeros(3)] + [s_ * d * np.eye(3)[k] for k in range(3) for s_ in (1, -1)]
        best = None
        for c0 in starts:
            x, L = self.polish(self.from_spin(xw, c0), iters=150, bands=bands)
            for _ in range(3):
                x2, L2 = self.polish(x, iters=150, bands=bands)
                if L2 >= L - 0.5:
                    break
                x, L = x2, L2
            if best is None or L < best[1]:
                best = (x, L)
        sig, E = self.com_sigma_profile(best[0], bands, kap=kap)
        return best[0], best[1], L0, sig, E, None

    def axis_model(self, x):
        """Ось вращения в осях модели (единичная)."""
        w = np.asarray(x[-3:], float); R = AP.rotvec(np.asarray(x[:3], float)) @ self.Rbase.numpy() @ self.M2B.numpy()
        return R.T @ (w / max(float(np.linalg.norm(w)), 1e-12))

    def com_sigma(self, x, bands=None, h=2.0, kap=1.0):
        """σ центра масс ПОПЕРЁК оси вращения (м). Вдоль оси центр масс по вращению не измеряется в принципе: точки
        оси неподвижны — снимок даёт ось вращения как прямую в теле, проходящую через центр масс. Гессиан невязки по
        двум составляющим c ⊥ оси — центральными разностями (шаг h м; меньший шаг тонет в шуме float32), остальные
        параметры — в найденной точке. → (σ по двум направлениям ⊥ оси, базис 3×2, ковариация 2×2)."""
        x = np.asarray(x, float); k0 = len(x) - 6; nm = self.axis_model(x)
        e1 = np.cross(nm, [1.0, 0, 0] if abs(nm[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1); E = np.c_[e1, np.cross(nm, e1)]
        pts = [x]
        for i in range(2):
            for j in range(i, 2):
                for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    u = x.copy(); u[k0:k0 + 3] += si * h * E[:, i] + sj * h * E[:, j]; pts.append(u)
        Lp = self.L(np.array(pts), bands).astype(float); H = np.zeros((2, 2)); q = 1
        for i in range(2):
            for j in range(i, 2):
                a, b_, c_, d_ = Lp[q:q + 4]; q += 4
                H[i, j] = H[j, i] = (a - b_ - c_ + d_) / (4 * h * h)
        H = 0.5 * H                                   # невязка = −2 ln L → информация = H/2
        if np.linalg.eigvalsh(H).min() <= 0:
            return np.full(2, np.inf), E, None
        cov = kap * np.linalg.inv(H); return np.sqrt(np.diag(cov)), E, cov


class DiffTumble(DiffSpinPose):
    """Кувыркание без внешних моментов (уравнения Эйлера) — ТЕНЗОР ИНЕРЦИИ по фото. Равномерное вращение — частный
    случай (ω вдоль главной оси или все моменты равны); при вращении не вокруг главной оси ω в теле прецессирует, и
    поворот за пролёт отходит от равномерного — по пикселям видно отношения моментов и главные оси, без модели масс.
    x = [R₀ 3, приводы, сдвиг 2, ω₀ 3 (рад/с, орбит. оси, в t₀), q 3 (главные оси в осях модели, вектор поворота),
         λ 2 (ln I₁/I₃, ln I₂/I₃)].
    Интегрирование: ω̇_p = I⁻¹((I ω_p) × ω_p), Ȧ = A [ω_p]× (A — главные оси → орбитальные), РК4 по ω, поворот за шаг —
    экспонента от ω в середине шага; Rt(t) = A(t)·A(t₀)ᵀ. Шаг ≤ 0,02 с (ω·dt ≤ 4° при 200 °/с)."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.P += 5
        self.scale = torch.cat([self.scale, torch.tensor([0.3, 0.3, 0.3, 0.2, 0.2])])
        self.dt_max = 0.02

    def _split(self, x):
        return x[:, :-11], x[:, -11:-8], x[:, -8:-5], x[:, -5:-2], x[:, -2:]

    def from_spin(self, xw, q=(0.0, 0.0, 0.0), lam=(0.0, 0.0)):
        """Равномерное вращение xw → x кувыркания (q — главные оси, λ — логарифмы отношений)."""
        return np.r_[np.asarray(xw, float), np.asarray(q, float), np.asarray(lam, float)]

    def _Rt_all(self, xs, w0, q, lam, times):
        """Повороты за время t − t₀ для списка моментов (орбитальные оси) → dict t → (B, 3, 3)."""
        B = xs.shape[0]; Rb = rotvec_to_mat(xs[:, 0:3]) @ self.Rbase @ self.M2B
        A0 = Rb @ rotvec_to_mat(q); Iv = torch.cat([torch.exp(lam), torch.ones(B, 1, dtype=xs.dtype)], 1)   # I₁, I₂, I₃ = 1
        wp0 = torch.einsum("bji,bj->bi", A0, w0)

        def f(w):
            return torch.cross(Iv * w, w, dim=1) / Iv
        out = {}
        for sgn in (1.0, -1.0):
            ts = sorted([t for t in times if (t - self.t0) * sgn > 0], key=lambda t: abs(t - self.t0))
            A, w, tc = A0, wp0, self.t0
            for t in ts:
                T = abs(t - tc); nst = max(1, int(math.ceil(T / self.dt_max))); h = sgn * T / nst
                for _ in range(nst):
                    k1 = f(w); k2 = f(w + 0.5 * h * k1); k3 = f(w + 0.5 * h * k2); k4 = f(w + h * k3)
                    wn = w + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                    A = A @ rotvec_to_mat(0.5 * (w + wn) * h); w = wn
                tc = t; out[t] = A @ A0.transpose(1, 2)
        for t in times:
            if t not in out:
                out[t] = torch.eye(3, dtype=xs.dtype).expand(B, 3, 3)
        return out

    def loss(self, x, bands=None, prior=True, shade=None):
        shade = self.shade if shade is None else shade
        xs, w0, q, lam = x[:, :-8], x[:, -8:-5], x[:, -5:-2], x[:, -2:]; dp = self.dpos(x)
        bl = bands or self.sc.bands; Rts = self._Rt_all(xs, w0, q, lam, sorted({self.sc.t[b] for b in bl}))

        def occ(b, sh):
            cw, Sw = self.spots(xs, Rts[self.sc.t[b]])
            return self.sc.band_occ_cs(b, cw, Sw, dp, sh)
        L = self.sc.loss(None, dp, self.kappa, grey=self.grey, bands=bands, shade=shade, occfun=occ)
        if prior:
            L = L + (xs[:, -2] ** 2 + xs[:, -1] ** 2) / AP.SIG_POS ** 2 + (lam ** 2).sum(1) / 3.0 ** 2
        return L

    def dpos(self, x):
        xs = x[:, :-8] if x.shape[1] == self.P else x
        return torch.cat([xs[:, -2:], torch.zeros(x.shape[0], 1, dtype=x.dtype)], 1)

    def columns(self, x, b, shade=None):
        shade = self.shade if shade is None else shade
        with torch.no_grad():
            xt = torch.tensor(np.atleast_2d(x), dtype=torch.float32)
            xs, w0, q, lam = xt[:, :-8], xt[:, -8:-5], xt[:, -5:-2], xt[:, -2:]
            Rt = self._Rt_all(xs, w0, q, lam, [self.sc.t[b]])[self.sc.t[b]]
            cw, Sw = self.spots(xs, Rt); res = self.sc.band_occ_cs(b, cw, Sw, self.dpos(xt), shade)
        if shade:
            occ, lit, sp = res; cols = torch.cat([lit, sp], 1)
        else:
            occ = res; cols = occ
        return occ[0].numpy(), cols[0].numpy()

    def inertia_model(self, x):
        """→ (моменты I₁:I₂:I₃ с I₃ = 1, главные оси столбцами в осях модели)."""
        q = np.asarray(x[-5:-2], float); lam = np.asarray(x[-2:], float)
        return np.r_[np.exp(lam), 1.0], AP.rotvec(q)


def anchor_to_geometry(rc, orbit_json):
    """Опоры полос — геометрические положения по точной орбите Sentinel (scripts/orbit_check.py: gps3d / gps) вместо
    ярких пиков в каждой полосе. → описание (или None, если орбиты нет)."""
    import json
    from pathlib import Path
    p = Path(orbit_json)
    if not p.exists():
        return None
    q = json.loads(p.read_text()); f = q["fits"].get("gps3d") or q["fits"].get("gps") or {}
    bp = f.get("band_positions")
    if not bp or not all(b in bp for b in rc["bands"]):
        return None
    sh = {}
    for b, d in rc["bands"].items():
        sh[b] = float(np.hypot(*(np.asarray(d.get("meas") or d["p0"], float)[:2] - bp[b])))
        d["p0"] = np.asarray(bp[b], float); d.pop("meas", None)
    return f"геометрия по орбите Sentinel ({'GPS + 3D-углы' if 'gps3d' in q['fits'] else 'GPS'}); прежние опоры смещены на " + \
        ", ".join(f"{b} {v:.0f} м" for b, v in sh.items())


def solve_spin(Wf, Wc, Sf, Sc, bands=None, n_rot=1500, kap=1.0, log=print, cache=None):
    """Вращение по пикселям на наборе полос bands (все — по умолчанию); то же, что проверено подстановками
    (scripts/spin_pix_test.py): 1) позы подгрупп ≤ 0,2 с; 2) кандидаты ω из пар минимумов (+2πk) и «не вращается»;
    3) градиент по всем параметрам; 4) профиль по оси (вся сфера) × величине × позам (найденная, её двойники тела,
    минимумы подгруппы у t₀), градиент для каждого направления; 5) зеркальный двойник (направление).
    → dict(x, L, L_static, x_static, mirror_dL, mirror_axis_deg, p_direction)."""
    from scipy.spatial.transform import Rotation as Ro
    bands = list(bands or Wf.sc.bands)
    bs = sorted(bands, key=lambda b: Wf.sc.t[b]); CL = [[bs[0]]]
    for b in bs[1:]:
        (CL[-1].append(b) if Wf.sc.t[b] - Wf.sc.t[CL[-1][0]] <= 0.2 else CL.append([b]))
    ct = [float(np.median([Wf.sc.t[b] for b in c])) for c in CL]
    cache = {} if cache is None else cache          # статические позы набора полос не зависят от других полос

    def srch(bb, npol):
        k_ = (tuple(sorted(bb)), npol)
        if k_ not in cache:
            md, _ = Sc.search(bands=list(bb), n_rot=n_rot, top=128, fine=Sf, n_polish=npol, log=lambda s: None)
            cache[k_] = [m["x"] for m in md]
        return cache[k_]
    gm = [srch(c, 8) for c in CL]
    ib = int(np.argmin([abs(t - Wf.t0) for t in ct]))
    cand = Wf.candidates(gm, ct, k_modes=3) + [Wf.static_x(m) for m in gm[ib][:4]]
    # группы полос (разрыв > 1 с): поза группы надёжнее позы мелкой подгруппы; поворот между позами групп и все целые
    # k дают точный набор кандидатных ω (разрешение целочисленной неоднозначности — дальше, по невязке)
    G1 = []
    for b in bs:
        (G1[-1].append(b) if G1 and Wf.sc.t[b] - Wf.sc.t[G1[-1][-1]] <= 1.0 else G1.append([b]))
    if len(G1) >= 2:
        gt1 = [float(np.median([Wf.sc.t[b] for b in g])) for g in G1]; gm1 = [srch(g, 10) for g in G1]
        cand += Wf.candidates(gm1, gt1, k_modes=5)
        # опора на САМУЮ ИНФОРМАТИВНУЮ группу (наибольший выигрыш над «объекта нет»): её позы + полный перебор ω —
        # ось (120 направлений) × скорость 1…200 °/с шагом 2 °/с (на базе Δt шаг поворота 2°/с·Δt ≲ 10°); поза в t₀ —
        # перенос позы группы на t₀ при этой ω. Слабые группы позу не определяют (много равных минимумов), а ω при
        # надёжной позе сильной группы выбирается невязкой всех полос, включая поворот внутри групп
        gain = [Sf.null(g) - float(Sf.L(gm1[i_][0], bands=g)[0]) for i_, g in enumerate(G1)]
        ia = int(np.argmax(gain)); kq = np.arange(120) + 0.5; zq = 1 - 2 * kq / 120; pq = math.pi * (1 + 5 ** 0.5) * kq
        dirs3 = np.c_[np.sqrt(1 - zq ** 2) * np.cos(pq), np.sqrt(1 - zq ** 2) * np.sin(pq), zq]
        spds = np.radians(np.arange(1.0, 200.1, 2.0)); from scipy.spatial.transform import Rotation as Ro_
        for m in gm1[ia][:8]:
            Ra = Wf.R_of(m); rest = m[3:]
            W3 = np.array([d_ * s_ for d_ in dirs3 for s_ in spds])
            R0s = Ro_.from_rotvec(-W3 * (gt1[ia] - Wf.t0)) * Ro_.from_matrix(Ra)
            X3 = np.c_[R0s.as_rotvec(), np.tile(rest, (len(W3), 1)), W3]
            L3 = np.concatenate([Wc.L(X3[i_:i_ + 256], bands) for i_ in range(0, len(X3), 256)]).reshape(len(dirs3), len(spds))
            cand += [X3[d_ * len(spds) + int(np.argmin(L3[d_]))] for d_ in np.argsort(L3.min(1))[:6]]
    xc, Lc = Wc.fit(np.array(cand), iters=120, bands=bands)
    o = np.argsort(Lc)[:12]; xf, Lf = Wf.fit(xc[o], iters=60, lr=0.01, bands=bands)
    xe, Le = sorted([Wf.polish(xf[k], bands=bands) for k in np.argsort(Lf)[:4]], key=lambda q: q[1])[0]
    w0 = xe[-3:]; r0 = max(float(np.linalg.norm(w0)), math.radians(1.0)); n0 = w0 / max(float(np.linalg.norm(w0)), 1e-9)
    kf = np.arange(200) + 0.5; zf = 1 - 2 * kf / 200; phf = math.pi * (1 + 5 ** 0.5) * kf
    dirs = np.vstack([n0, np.c_[np.sqrt(1 - zf ** 2) * np.cos(phf), np.sqrt(1 - zf ** 2) * np.sin(phf), zf]])
    mags = np.arange(0.80, 1.26, 0.03); W = [d_ * r0 * m for d_ in dirs for m in mags]
    Rh = Wf.R_of(xe); dr_ = xe[3:3 + Wf.nd]; rest_ = xe[3 + Wf.nd:-3]; bases = []
    for k_ in (None, 0, 1, 2):
        Rk = Rh if k_ is None else Rh @ AP.rot_axis(np.eye(3)[k_], math.pi)
        for dd_ in (dr_, dr_ + math.pi):
            bases.append((Rk, np.r_[dd_, rest_], False))
    bases += [(Wf.R_of(m), m[3:], True) for m in gm[ib][:4]]
    dtc = ct[ib] - Wf.t0; Xw = []
    for Rb_, rest, shift in bases:
        for w in W:
            R0 = AP.rotvec(-np.asarray(w) * dtc) @ Rb_ if shift else Rb_
            Xw.append(np.r_[Ro.from_matrix(R0).as_rotvec(), rest, w])
    Xw = np.array(Xw); Lw = np.concatenate([Wc.L(Xw[i:i + 256], bands) for i in range(0, len(Xw), 256)])
    Lr = Lw.reshape(len(bases), len(dirs), len(mags))
    per_dir = [np.unravel_index(np.argmin(Lr[:, d_, :]), (len(bases), len(mags))) for d_ in range(len(dirs))]
    Xd = np.array([Xw[(bi * len(dirs) + d_) * len(mags) + mi] for d_, (bi, mi) in enumerate(per_dir)])
    xd, Ld = Wc.fit(Xd, iters=40, lr=0.02, bands=bands)
    ow = np.argsort(Ld)[:12]; xw, Lw2 = Wf.fit(xd[ow], iters=60, lr=0.01, bands=bands)
    xe, Le = sorted([Wf.polish(xw[k], bands=bands) for k in np.argsort(Lw2)[:3]] + [(xe, Le)], key=lambda q: q[1])[0]
    # 4б. полный перебор поза × ось при каждой кандидатной скорости: найденной и её целочисленных соседях
    #     |ω| + k·2π/Δt (k = −2…2; Δt — наибольший промежуток между группами полос): поворот между группами у них
    #     одинаков, различает их только поворот внутри групп — и только при полной свободе позы и оси (с неверной
    #     осью соседняя яма не достигается). Поза в t₀ — сетка 600 по SO(3), ось — 120 направлений по сфере; для
    #     каждой скорости и направления лучшая поза → градиент → точная модель; итог — лучший по невязке
    spd = float(np.linalg.norm(xe[-3:]))
    if spd > math.radians(1.0):
        speeds = [spd]
        grp = [g for g in Wf.groups(1.0) if any(b in bands for b in g)]
        if len(grp) >= 2:
            tg = sorted(float(np.median([Wf.sc.t[b] for b in g if b in bands])) for g in grp)
            stp = 2 * math.pi / max(tg[i_ + 1] - tg[i_] for i_ in range(len(tg) - 1))
            speeds += [abs(spd + k * stp) for k in (-2, -1, 1, 2) if math.radians(1.0) < abs(spd + k * stp) <= math.radians(200.0)]
        R0g = so3_grid(600); kq = np.arange(120) + 0.5; zq = 1 - 2 * kq / 120; pq = math.pi * (1 + 5 ** 0.5) * kq
        dirs2 = np.c_[np.sqrt(1 - zq ** 2) * np.cos(pq), np.sqrt(1 - zq ** 2) * np.sin(pq), zq]
        rest2 = xe[3:-3]; cands = []
        for sp_ in speeds:
            Xg = np.array([np.r_[rv, rest2, d_ * sp_] for d_ in dirs2 for rv in R0g])
            Lg = np.concatenate([Wc.L(Xg[i_:i_ + 256], bands) for i_ in range(0, len(Xg), 256)]).reshape(len(dirs2), len(R0g))
            Xd2 = np.array([Xg[d_ * len(R0g) + int(np.argmin(Lg[d_]))] for d_ in range(len(dirs2))])
            xd2, Ld2 = Wc.fit(Xd2, iters=40, lr=0.02, bands=bands)
            cands += [xd2[k] for k in np.argsort(Ld2)[:6]]
        xw2, Lw3 = Wf.fit(np.array(cands), iters=60, lr=0.01, bands=bands)
        xe, Le = sorted([Wf.polish(xw2[k], bands=bands) for k in np.argsort(Lw3)[:4]] + [(xe, Le)], key=lambda q: q[1])[0]
    xm, Lm = sorted([Wf.polish(t, bands=bands) for t in Wf.mirror_twins(xe)], key=lambda q: q[1])[0]
    if Lm < Le:
        (xe, Le), (xm, Lm) = (xm, Lm), (xe, Le)
    ang_m = math.degrees(math.acos(float(np.clip(xe[-3:] @ xm[-3:] / (np.linalg.norm(xe[-3:]) * np.linalg.norm(xm[-3:]) + 1e-12), -1, 1))))
    xs_, Ls = min([Sf.polish(xe[:-3], bands=bands)] + [Sf.polish(m, bands=bands) for m in gm[ib][:4]], key=lambda q: q[1])
    return dict(x=xe, L=Le, L_static=Ls, x_static=xs_, mirror_dL=Lm - Le, mirror_axis_deg=ang_m,
                p_direction=1.0 / (1.0 + math.exp(-0.5 * (Lm - Le) / kap)), bands=bands)

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
        """coarse — для глобального поиска: силуэт одной группой, окно по размаху объекта, пятна ≤ 3 м (diffmodel)."""
        part, chains = face_parts(spec)
        self.sc = sc = Scene(rc, spec, C, parts=part, lmax=3.0 if coarse else lmax, ng=1 if coarse else None, coarse=coarse)
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
        (грубой) модели, дожим — на fine (мелкие пятна), если задана."""
        X = self.grid(n_rot, step_deg); Ls = []
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


def solve_spin(Wf, Wc, Sf, Sc, bands=None, n_rot=1500, kap=1.0, log=print):
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
    gm = []
    for c in CL:
        md, _ = Sc.search(bands=c, n_rot=n_rot, top=128, fine=Sf, n_polish=8, log=lambda s: None)
        gm.append([m["x"] for m in md])
    ib = int(np.argmin([abs(t - Wf.t0) for t in ct]))
    cand = Wf.candidates(gm, ct, k_modes=3) + [Wf.static_x(m) for m in gm[ib][:4]]
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
    xm, Lm = sorted([Wf.polish(t, bands=bands) for t in Wf.mirror_twins(xe)], key=lambda q: q[1])[0]
    if Lm < Le:
        (xe, Le), (xm, Lm) = (xm, Lm), (xe, Le)
    ang_m = math.degrees(math.acos(float(np.clip(xe[-3:] @ xm[-3:] / (np.linalg.norm(xe[-3:]) * np.linalg.norm(xm[-3:]) + 1e-12), -1, 1))))
    xs_, Ls = min([Sf.polish(xe[:-3], bands=bands)] + [Sf.polish(m, bands=bands) for m in gm[ib][:4]], key=lambda q: q[1])
    return dict(x=xe, L=Le, L_static=Ls, x_static=xs_, mirror_dL=Lm - Le, mirror_axis_deg=ang_m,
                p_direction=1.0 / (1.0 + math.exp(-0.5 * (Lm - Le) / kap)), bands=bands)

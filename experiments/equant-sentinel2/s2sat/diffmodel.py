"""Дифференцируемая модель снимка объекта (PyTorch): градиенты невязки по позе, вращению, положению, углам приводов.

Прежняя модель (autopose.regress) рисует грани жёсткими масками на сетке 1 м — невязка кусочно-постоянна по позе, и
подгонка возможна только перебором/Powell. Здесь:
  * сетка модели дробится до треугольников меньше ширины ядра полосы (в пикселях снимка);
  * каждый треугольник — «пятно» в центре со своей проекцией площади на снимок: a = (Q·S)·q, где S — векторная площадь
    грани, Q — поворот модели, q = m₁ × m₂ (строки якобиана полосы J/1000: метры объекта → метры снимка); учитываются
    только грани, обращённые к Sentinel (заслонение фона — силуэт);
  * пятна раскладываются билинейно на мелкую сетку полосы, свёртка с тем же гауссовым ядром, что в autopose
    (функция рассеяния + пиксель + смаз push-broom: model_recon.kernel), значения — в центрах пикселей;
  * яркости групп материалов — НК внутри (закрытая форма, дифференцируема), невязка — та же псевдо-Хьюбер.
Всё пакетом по B стартам сразу (B независимых подгонок): градиентный спуск вместо перебора.
"""
from __future__ import annotations

import math

import numpy as np
import torch

torch.set_default_dtype(torch.float32)
# мелкие тензоры: много потоков на занятых ядрах тормозят в 10–100 раз (проверено: 512 поз 9,1 с при 4 потоках, 0,7 с при 1)
torch.set_num_threads(2)
HUB = 3.0


def closed_weight(F, V=None):
    """Вес проекции грани: 0,5 — грань замкнутой части (все рёбра общие с другими гранями: силуэт = ½ Σ|проекций| по
    обеим сторонам), 1 — открытая пластина (как в растровой модели: рисуется с обеих сторон). Не зависит от обхода.
    Совпадающие вершины сначала склеиваются (у сеток каталога вершины граней часто не общие)."""
    F = np.asarray(F)
    if V is not None:
        V = np.asarray(V, float); tol = 1e-4 * float(np.ptp(V, 0).max() or 1.0)
        _, uid = np.unique(np.round(V / tol).astype(np.int64), axis=0, return_inverse=True)
        F = np.asarray(uid).reshape(-1)[F]
    e = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1), axis=2).reshape(-1, 2)
    key = e[:, 0].astype(np.int64) * (int(F.max()) + 1) + e[:, 1]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    shared = (cnt[inv] >= 2).reshape(-1, 3).all(1)
    return np.where(shared, 0.5, 1.0)


def subdivide(V, F, G, lmax, cap=60000, W=None):
    """Делит треугольники на 4 (середины рёбер), пока длиннейшее ребро > lmax. Возвращает центры, векторные площади,
    группы, веса."""
    T = np.asarray(V, float)[np.asarray(F)]; Gg = np.asarray(G); Ww = np.ones(len(T)) if W is None else np.asarray(W, float)
    while len(T) < cap:
        e = np.stack([np.linalg.norm(T[:, 1] - T[:, 0], axis=1), np.linalg.norm(T[:, 2] - T[:, 1], axis=1),
                      np.linalg.norm(T[:, 0] - T[:, 2], axis=1)], 1).max(1)
        big = e > lmax
        if not big.any():
            break
        Bt = T[big]; a, b, c = Bt[:, 0], Bt[:, 1], Bt[:, 2]; ab = (a + b) / 2; bc = (b + c) / 2; ca = (c + a) / 2
        new = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1), np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)])
        T = np.concatenate([T[~big], new]); Gg = np.concatenate([Gg[~big], np.tile(Gg[big], 4)])
        Ww = np.concatenate([Ww[~big], np.tile(Ww[big], 4)])
    return T.mean(1), 0.5 * np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), Gg, Ww


def rotvec_to_mat(v):
    """Родриг для пакета: v (..., 3) → (..., 3, 3). R = I + A·[v]× + B·[v]×², A = sin θ/θ, B = (1 − cos θ)/θ² — с рядами
    у нуля (гладко, вторые производные конечны: Гессиан в точке поворота)."""
    t2 = (v * v).sum(-1, keepdim=True)[..., None]; small = t2 < 1e-6
    th = torch.sqrt(torch.where(small, torch.ones_like(t2), t2))
    A = torch.where(small, 1 - t2 / 6 + t2 * t2 / 120, torch.sin(th) / th)
    Bc = torch.where(small, 0.5 - t2 / 24 + t2 * t2 / 720, (1 - torch.cos(th)) / (th * th))
    K = torch.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -v[..., 2], v[..., 1], v[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -v[..., 0], -v[..., 1], v[..., 0]
    I = torch.eye(3, dtype=v.dtype).expand(K.shape)
    return I + A * K + Bc * (K @ K)


class Band:
    """Одна полоса: окно пикселей, мелкая сетка, ядро в Фурье, индексы выборки."""

    def __init__(self, d, ng, margin=30.0, crop=None, hfac=1.6):
        """crop — радиус (м снимка) вокруг опоры: пиксели дальше, куда модель не может попасть, не берутся (их вклад в
        невязку постоянен — для глобального поиска); hfac — шаг мелкой сетки σ_ядра / hfac."""
        self.res = float(d["res"]); self.s = float(d["s"])
        ok = np.isfinite(d["r"]) & np.isfinite(d["g"])
        if crop is not None:
            c0 = np.asarray(d["meas"][:2], float) if d.get("meas") else np.asarray(d["p0"], float)
            ok &= (np.abs(d["X"] - c0[0]) <= crop) & (np.abs(d["Y"] - c0[1]) <= crop)
        self.idx = np.where(ok)[0]                                 # номера пикселей в исходном окне полосы
        d = dict(d, X=d["X"][ok], Y=d["Y"][ok], r=d["r"][ok], g=d["g"][ok]); ok = np.ones(len(d["X"]), bool)
        self.X, self.Y = d["X"][ok], d["Y"][ok]
        self.r = torch.tensor(d["r"][ok], dtype=torch.float32); self.g = torch.tensor(d["g"][ok], dtype=torch.float32)
        self.gm = float(np.nanmedian(d["g"]))
        self.base = np.asarray(d["meas"][:2], float) if d.get("meas") else np.asarray(d["p0"], float)
        self.M = np.asarray(d["J"], float) / 1000.0                # м снимка на м объекта
        self.q = np.cross(self.M[0], self.M[1])                    # проекция площади: a = S·q
        # ядро: то же, что model_recon.kernel
        k = np.asarray(d["_k"]); C = self._cov(k)
        self.h = max(1.0, min(self.res / 3.0 * 1.6 / hfac, math.sqrt(np.linalg.eigvalsh(C).min()) / hfac))
        self.x0 = d["X"].min() - self.res - margin; self.y1 = d["Y"].max() + self.res + margin
        self.W = int((d["X"].max() - d["X"].min() + 2 * self.res + 2 * margin) / self.h) + 2
        self.H = int((d["Y"].max() - d["Y"].min() + 2 * self.res + 2 * margin) / self.h) + 2
        gx = (np.arange(self.W) - self.W // 2) * self.h; gy = (np.arange(self.H) - self.H // 2) * self.h
        Xg, Yg = np.meshgrid(gx, -gy); iC = np.linalg.inv(C)
        kern = np.exp(-0.5 * (iC[0, 0] * Xg ** 2 + 2 * iC[0, 1] * Xg * Yg + iC[1, 1] * Yg ** 2)); kern /= kern.sum()
        kern = np.fft.ifftshift(kern)
        self.Kf = torch.fft.rfft2(torch.tensor(kern, dtype=torch.float32)); self.sig_min = math.sqrt(np.linalg.eigvalsh(C).min())
        # выборка в центрах пикселей (билинейно по мелкой сетке)
        u = (self.X - self.x0) / self.h; v = (self.y1 - self.Y) / self.h
        i0 = np.floor(u).astype(int); j0 = np.floor(v).astype(int); fu = u - i0; fv = v - j0
        self.samp = [(torch.tensor(j * self.W + i), torch.tensor(w, dtype=torch.float32)) for i, j, w in
                     [(i0, j0, (1 - fu) * (1 - fv)), (i0 + 1, j0, fu * (1 - fv)), (i0, j0 + 1, (1 - fu) * fv), (i0 + 1, j0 + 1, fu * fv)]]
        self.ng = ng

    @staticmethod
    def _cov(k):
        n = k.shape[0]; h = (n - 1) // 2; g = np.arange(-h, h + 1, dtype=float)
        Xg, Yg = np.meshgrid(g, -g); w = k / k.sum()
        return np.array([[np.sum(w * Xg ** 2), np.sum(w * Xg * Yg)], [np.sum(w * Xg * Yg), np.sum(w * Yg ** 2)]])   # м² (FINE = 1 м)

    def occupancy(self, P, A, grp):
        """P (B, m, 2) — положения пятен на снимке (м UTM), A (B, m) — площади на снимке (м², только лицевые), grp (m) —
        группы. → занятость пикселей (B, ng, npix)."""
        B = P.shape[0]; HW = self.H * self.W
        u = ((P[..., 0] - self.x0) / self.h).clamp(0.0, self.W - 2.001); v = ((self.y1 - P[..., 1]) / self.h).clamp(0.0, self.H - 2.001)
        i0 = torch.floor(u.detach()); j0 = torch.floor(v.detach()); fu = u - i0; fv = v - j0
        base = (j0 * self.W + i0).long() + (grp * HW)[None, :]            # пятна за краем сетки — на край (поля 30 м)
        dens = A / self.h ** 2; a = dens * (1 - fv); c = dens * fv
        grid = torch.zeros(B, self.ng * HW, dtype=P.dtype)
        grid.scatter_add_(1, base, a * (1 - fu)); grid.scatter_add_(1, base + 1, a * fu)
        grid.scatter_add_(1, base + self.W, c * (1 - fu)); grid.scatter_add_(1, base + self.W + 1, c * fu)
        img = torch.fft.irfft2(torch.fft.rfft2(grid.view(B, self.ng, self.H, self.W)) * self.Kf, s=(self.H, self.W)).view(B, self.ng, -1)
        out = 0
        for idx, w in self.samp:
            out = out + img[..., idx] * w
        return out


class Scene:
    """Объект (модель каталога) и полосы снимка. Поворот модели в оси реконструкции Q = C · R_lvlh · M2B."""

    def __init__(self, rc, spec, C, ng=None, lmax=None, t=None, parts=None, coarse=False):
        """parts (по граням) — номер подвижной части (−1 — корпус); пятна сортируются по частям: self.part.
        coarse — грубая модель для глобального поиска: окно вокруг опоры не шире, чем объект может занять (размах
        модели на снимке + 3σ ядра + 40 м), сетка σ/1."""
        self.ng = int(ng or (int(np.max(spec["G"])) + 1))
        Ms = [np.asarray(d["J"], float) / 1000.0 for d in rc["bands"].values()]
        k = float(np.median([np.linalg.norm(M, 2) for M in Ms]))          # м снимка на м объекта
        self.bands = [b for b in rc["bands"]]
        crop = None
        if coarse:
            Vm = np.asarray(spec["V"], float); rad = float(np.max(np.linalg.norm(Vm - Vm.mean(0), axis=1)))
            crop = rad * max(np.linalg.norm(M, 2) for M in Ms) + 3 * 8.0 + 40.0
        self.B = {b: Band(rc["bands"][b], self.ng, crop=crop, hfac=1.0 if coarse else 1.6) for b in self.bands}
        smin = min(bd.sig_min for bd in self.B.values())
        lmax = lmax or max(0.2, 0.8 * smin / k)                            # пятно ≤ 0,8 σ ядра на снимке
        G0 = np.asarray(spec["G"]) % self.ng
        Gp = G0 if parts is None else G0 + self.ng * (np.asarray(parts) + 1)
        c, S, g, wt = subdivide(spec["V"], spec["F"], Gp, lmax, W=closed_weight(spec["F"], spec["V"]))
        part = g // self.ng - (0 if parts is None else 1); o = np.argsort(part, kind="stable")
        c, S, g, wt, part = c[o], S[o], g[o], wt[o], part[o]
        self.part = part; self.c_model = c; self.S_model = S * wt[:, None]
        M2B = np.asarray(spec["M2B"], float)
        self.c = torch.tensor(c @ M2B.T, dtype=torch.float32); self.S = torch.tensor((S * wt[:, None]) @ M2B.T, dtype=torch.float32)
        self.grp = torch.tensor(np.asarray(g) % self.ng)
        self.C = torch.tensor(np.asarray(C, float), dtype=torch.float32); self.los = np.asarray(rc["los"], float)
        sun = np.asarray(rc["sun"], float); los_ = self.los / np.linalg.norm(self.los); hv = sun + los_
        self.sun_t = torch.tensor(sun / np.linalg.norm(sun), dtype=torch.float32)
        self.los_t = torch.tensor(los_, dtype=torch.float32); self.half_t = torch.tensor(hv / np.linalg.norm(hv), dtype=torch.float32)
        jd = {b: float(rc["bands"][b]["jd"]) for b in self.bands}; j0 = float(np.median(list(jd.values())))
        self.t = t or {b: (v - j0) * 86400.0 for b, v in jd.items()}
        for b in self.bands:
            bd = self.B[b]; bd.Mt = torch.tensor(bd.M, dtype=torch.float32); bd.qt = torch.tensor(bd.q, dtype=torch.float32)
            bd.bt = torch.tensor(bd.base, dtype=torch.float32)
        print(f"   дифф. модель: {len(c)} пятен (≤ {lmax:.2f} м), {len(self.bands)} полос", flush=True)

    def band_occ(self, b, Rfit, dpos, shade=False):
        """Rfit (B, 3, 3): поворот модели → оси реконструкции в момент полосы; dpos (B, 3) — сдвиг, м объекта.
        shade: кроме заслонения фона (занятость групп) — освещённость Солнцем: по Ламберту cos(n·s)⁺ стороны грани,
        обращённой к Sentinel, и блик стекла панелей (n·h)^p, h — биссектриса Солнце/визир. → (occ, lit, spec)."""
        cw = torch.einsum("bij,mj->bmi", Rfit, self.c); Sw = torch.einsum("bij,mj->bmi", Rfit, self.S)
        return self.band_occ_cs(b, cw, Sw, dpos, shade)

    def band_occ_cs(self, b, cw, Sw, dpos, shade=False):
        """То же по уже повёрнутым центрам пятен cw (B, m, 3) и векторным площадям Sw (B, m, 3) — для моделей с
        подвижными частями (крылья на приводах), где у частей свой поворот."""
        bd = self.B[b]
        P = bd.bt + torch.einsum("ij,bmj->bmi", bd.Mt, cw + dpos[:, None, :])
        A = torch.abs(torch.einsum("bmj,j->bm", Sw, bd.qt))                   # вес 0,5/1 — в S
        occ = bd.occupancy(P, A, self.grp)
        if not shade:
            return occ
        n = Sw / torch.linalg.norm(Sw, dim=-1, keepdim=True).clamp_min(1e-9)
        nl = (n * self.los_t).sum(-1); nv = n * torch.sign(nl)[..., None]          # сторона к камере
        cosi = torch.relu((nv * self.sun_t).sum(-1))
        spec = torch.relu((nv * self.half_t).sum(-1)) ** 60
        lit = bd.occupancy(P, A * cosi, self.grp)
        sp = bd.occupancy(P, A * spec * (self.grp == 0), self.grp).sum(1, keepdim=True)   # блик — панели (группа 0)
        return occ, lit, sp

    def loss(self, Rfun, dpos, kappa=1.0, grey=True, bands=None, shade=False, occfun=None):
        """Rfun(b) → (B,3,3). Серое тело (grey): яркости групп общие для полос (в долях медианы фона), иначе — свои.
        shade: яркость объекта = Σ ρ_g · освещённость_g + блик (физически: Солнце, визир), а не свободная яркость групп.
        occfun(b, shade) — вместо Rfun: своя занятость (модель с подвижными частями)."""
        bands = bands or self.bands
        rows, ys = [], []
        L = 0.0
        for b in bands:
            bd = self.B[b]
            res = occfun(b, shade) if occfun is not None else self.band_occ(b, Rfun(b), dpos, shade=shade)
            if shade:
                occ, lit, sp = res
                cols = torch.cat([lit, sp], 1)
            else:
                occ = res; cols = occ
            A = cols.transpose(1, 2) * (bd.gm / bd.s if grey else 1.0 / bd.s)         # (B, n, k)
            y = (bd.r + bd.g * occ.sum(1)) / bd.s                                      # (B, n)
            if grey:
                rows.append(A); ys.append(y)
            else:
                L = L + self._robust(A, y)
        if grey:
            L = self._robust(torch.cat(rows, 1), torch.cat(ys, 1))
        return L / kappa

    @staticmethod
    def _robust(A, y):
        """Яркости групп ≥ 0 — ТОЧНЫЕ неотрицательные НК (перебор активных множеств, групп ≤ 4–5: 2^k решений пакетом),
        а не «НК, затем обрезать отрицательные» (обрезка давала модель хуже пустой: объект рисовался чёрным силуэтом)."""
        Bn, n, k = A.shape
        AtA = A.transpose(1, 2) @ A + 1e-9 * torch.eye(k); Aty = A.transpose(1, 2) @ y[..., None]
        best_a = torch.zeros(Bn, k, 1, dtype=A.dtype); best_sse = (y * y).sum(1)
        for mask in range(1, 2 ** k):
            idx = [j for j in range(k) if (mask >> j) & 1]
            a_s = torch.linalg.solve(AtA[:, idx][:, :, idx], Aty[:, idx])
            a_f = torch.zeros(Bn, k, 1, dtype=A.dtype); a_f[:, idx] = a_s
            z = y - (A @ a_f)[..., 0]; sse = (z * z).sum(1)
            take = (a_s[..., 0] >= 0).all(1) & (sse < best_sse)
            best_a = torch.where(take[:, None, None], a_f, best_a); best_sse = torch.where(take, sse, best_sse)
        z = y - (A @ best_a)[..., 0]
        return (2 * HUB ** 2 * (torch.sqrt(1 + (z / HUB) ** 2) - 1)).sum(1)


def transfer_r(src, dst):
    """Остатки снимка r из сцены src в сцену dst (другое окно/сетка тех же пикселей) — для подстановок."""
    for b in dst.bands:
        S, D = src.B[b], dst.B[b]; pos = np.searchsorted(S.idx, D.idx)
        D.r = S.r[torch.tensor(pos)].clone()

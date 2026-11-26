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
    """Родриг для пакета: v (..., 3) → (..., 3, 3)."""
    th = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-12)
    k = v / th; K = torch.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]; I = torch.eye(3, dtype=v.dtype).expand(K.shape)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


class Band:
    """Одна полоса: окно пикселей, мелкая сетка, ядро в Фурье, индексы выборки."""

    def __init__(self, d, ng, margin=30.0):
        self.res = float(d["res"]); self.s = float(d["s"])
        ok = np.isfinite(d["r"]) & np.isfinite(d["g"])
        self.X, self.Y = d["X"][ok], d["Y"][ok]
        self.r = torch.tensor(d["r"][ok], dtype=torch.float32); self.g = torch.tensor(d["g"][ok], dtype=torch.float32)
        self.gm = float(np.nanmedian(d["g"]))
        self.base = np.asarray(d["meas"][:2], float) if d.get("meas") else np.asarray(d["p0"], float)
        self.M = np.asarray(d["J"], float) / 1000.0                # м снимка на м объекта
        self.q = np.cross(self.M[0], self.M[1])                    # проекция площади: a = S·q
        # ядро: то же, что model_recon.kernel
        k = np.asarray(d["_k"]); C = self._cov(k)
        self.h = max(1.0, min(self.res / 3.0, math.sqrt(np.linalg.eigvalsh(C).min()) / 1.6))
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
        B = P.shape[0]; u = (P[..., 0] - self.x0) / self.h; v = (self.y1 - P[..., 1]) / self.h
        i0 = torch.floor(u).detach(); j0 = torch.floor(v).detach(); fu = u - i0; fv = v - j0
        i0 = i0.long(); j0 = j0.long()
        grid = torch.zeros(B, self.ng, self.H * self.W, dtype=P.dtype)
        dens = A / self.h ** 2
        for di, dj, w in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)), (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
            ii = (i0 + di).clamp(0, self.W - 1); jj = (j0 + dj).clamp(0, self.H - 1)
            inside = ((i0 + di) >= 0) & ((i0 + di) < self.W) & ((j0 + dj) >= 0) & ((j0 + dj) < self.H)
            idx = (jj * self.W + ii) + (grp[None, :] * self.H * self.W)
            grid.view(B, -1).scatter_add_(1, idx, dens * w * inside)
        img = torch.fft.irfft2(torch.fft.rfft2(grid.view(B, self.ng, self.H, self.W)) * self.Kf, s=(self.H, self.W)).view(B, self.ng, -1)
        out = 0
        for idx, w in self.samp:
            out = out + img[..., idx] * w
        return out


class Scene:
    """Объект (модель каталога) и полосы снимка. Поворот модели в оси реконструкции Q = C · R_lvlh · M2B."""

    def __init__(self, rc, spec, C, ng=None, lmax=None, t=None):
        self.ng = int(ng or (int(np.max(spec["G"])) + 1))
        Ms = [np.asarray(d["J"], float) / 1000.0 for d in rc["bands"].values()]
        k = float(np.median([np.linalg.norm(M, 2) for M in Ms]))          # м снимка на м объекта
        self.bands = [b for b in rc["bands"]]
        self.B = {b: Band(rc["bands"][b], self.ng) for b in self.bands}
        smin = min(bd.sig_min for bd in self.B.values())
        lmax = lmax or max(0.2, 0.8 * smin / k)                            # пятно ≤ 0,8 σ ядра на снимке
        c, S, g, wt = subdivide(spec["V"], spec["F"], spec["G"], lmax, W=closed_weight(spec["F"], spec["V"]))
        M2B = np.asarray(spec["M2B"], float)
        self.c = torch.tensor(c @ M2B.T, dtype=torch.float32); self.S = torch.tensor((S * wt[:, None]) @ M2B.T, dtype=torch.float32)
        self.grp = torch.tensor(np.asarray(g) % self.ng)
        self.C = torch.tensor(np.asarray(C, float), dtype=torch.float32); self.los = np.asarray(rc["los"], float)
        jd = {b: float(rc["bands"][b]["jd"]) for b in self.bands}; j0 = float(np.median(list(jd.values())))
        self.t = t or {b: (v - j0) * 86400.0 for b, v in jd.items()}
        for b in self.bands:
            bd = self.B[b]; bd.Mt = torch.tensor(bd.M, dtype=torch.float32); bd.qt = torch.tensor(bd.q, dtype=torch.float32)
            bd.bt = torch.tensor(bd.base, dtype=torch.float32)
        print(f"   дифф. модель: {len(c)} пятен (≤ {lmax:.2f} м), {len(self.bands)} полос", flush=True)

    def band_occ(self, b, Rfit, dpos):
        """Rfit (B, 3, 3): поворот модели → оси реконструкции в момент полосы; dpos (B, 3) — сдвиг, м объекта."""
        bd = self.B[b]
        cw = torch.einsum("bij,mj->bmi", Rfit, self.c) + dpos[:, None, :]
        P = bd.bt + torch.einsum("ij,bmj->bmi", bd.Mt, cw)
        A = torch.abs(torch.einsum("bij,mj,i->bm", Rfit, self.S, bd.qt))     # вес 0,5/1 — в S
        return bd.occupancy(P, A, self.grp)

    def loss(self, Rfun, dpos, kappa=1.0, grey=True, bands=None):
        """Rfun(b) → (B,3,3). Серое тело (grey): яркости групп общие для полос (в долях медианы фона), иначе — свои."""
        bands = bands or self.bands
        rows, ys = [], []
        L = 0.0
        for b in bands:
            bd = self.B[b]; occ = self.band_occ(b, Rfun(b), dpos)                       # (B, ng, n)
            A = occ.transpose(1, 2) * (bd.gm / bd.s if grey else 1.0 / bd.s)          # (B, n, ng)
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
        AtA = A.transpose(1, 2) @ A + 1e-9 * torch.eye(A.shape[2]); Aty = (A.transpose(1, 2) @ y[..., None])
        a = torch.linalg.solve(AtA, Aty).clamp_min(0.0)
        z = y - (A @ a)[..., 0]
        return (2 * HUB ** 2 * (torch.sqrt(1 + (z / HUB) ** 2) - 1)).sum(1)

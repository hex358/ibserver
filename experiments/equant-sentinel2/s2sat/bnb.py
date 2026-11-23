"""Перебор гипотез ветвями и границами по схеме Equant (THEORY §14).

Гипотеза h = (n — фаза, e — растяжение, a — сдвиг вдоль видимой трассы, c — поперёк) на той же сетке, что и полный
перебор конвейера. Блок гипотез → для каждой полосы прямоугольник, куда попадают её точки → M_b = max|U_b| по
пирамиде максимумов → UB = √Σ min(M_b, M_(m))² (m — ранг отсечки; GLRT — проекция, T ≤ ‖u‖). Обход «сначала лучший»;
блоки с UB ≤ max(лучший, уровень) отбрасываются. Возвращает точный максимум (или доказательство «≤ уровня»).
"""
from __future__ import annotations
import os

import heapq
import math

import numpy as np

from . import detect


class Search:
    def __init__(self, ROOTS, fine, stretch, kmdeg, tdir, R_al, Rc, step, mosU, mosS, mosG, tiles, bands,
                 minb=3, cap_rank=None, cell=5000.0, margin=4500.0, max_px=float(__import__("os").environ.get("S2_BNB_MAXPX", "6e8"))):
        self.fine = np.asarray(fine, float); self.N = len(fine)
        self.ST = np.asarray(stretch if len(stretch) else [0.0], float)
        self.KST = kmdeg * 1000.0
        self.t = np.array(tdir, float); self.nrm = np.array([-self.t[1], self.t[0]])
        self.Aa = np.arange(-R_al, R_al + 1e-6, step); self.Cc = np.arange(-Rc, Rc + 1e-6, step) if Rc else np.array([0.0])
        self.mosU, self.mosS, self.mosG = mosU, mosS, mosG
        self.bands = list(bands); self.minb = minb
        self.cap = detect.CAP_RANK if cap_rank is None else cap_rank
        self.cell = cell
        # корни: RX/RY[n, b, k], VAL, DA[n, b] = α первого детектора − α_ref (как в конвейере)
        self.roots = [{b: [(q[1], q[2], q[3]) for q in R[b]] for b in R if b in self.bands} for R in ROOTS]
        self.aref = np.array([min([v[0][0] for v in R.values()]) if R else 0.0 for R in self.roots])
        K = max([len(v) for R in self.roots for v in R.values()] or [1]); NB = len(self.bands); self.K = K
        self.BI = {b: i for i, b in enumerate(self.bands)}
        self.RX = np.zeros((self.N, NB, K)); self.RY = np.zeros((self.N, NB, K)); self.VAL = np.zeros((self.N, NB, K), bool)
        self.DA = np.zeros((self.N, NB))
        for n, R in enumerate(self.roots):
            for b, lst in R.items():
                i = self.BI[b]; self.DA[n, i] = lst[0][0] - self.aref[n]
                for k, q in enumerate(lst):
                    self.RX[n, i, k] = q[1]; self.RY[n, i, k] = q[2]; self.VAL[n, i, k] = True
        # плотная пирамида max|U| (uint8, ×10) по области всех гипотез (+ запас под смещения нуля)
        v = self.VAL
        if not v.any():
            self.ok = False; self.reason = "no roots"; return
        ext = abs(self.ST).max() * self.KST * float(np.abs(self.DA).max()) + max(R_al, 0) + (Rc or 0) + margin
        T0 = next(iter(tiles.values()))["U"][self.bands[0]]; npx = T0.shape[0]; self.res = cell / npx
        ci = [c[0] for c in tiles]; cj = [c[1] for c in tiles]
        self.PYR, self.ORG = {}, {}; tot = 0
        for bi, b in enumerate(self.bands):          # своя пирамида на полосу — только по её области гипотез
            vb = self.VAL[:, bi, :]
            if not vb.any():
                continue
            xs, ys = self.RX[:, bi, :][vb], self.RY[:, bi, :][vb]
            i0 = max(min(ci), int(math.floor((xs.min() - ext) / cell))); i1 = min(max(ci), int(math.floor((xs.max() + ext) / cell)))
            j0 = max(min(cj), int(math.floor((ys.min() - ext) / cell))); j1 = min(max(cj), int(math.floor((ys.max() + ext) / cell)))
            if i1 < i0 or j1 < j0:
                continue
            H, W = (j1 - j0 + 1) * npx, (i1 - i0 + 1) * npx; tot += H * W
            if tot > max_px:
                self.ok = False; self.reason = f"pyramids {tot:.2e} px"; return
            D = np.zeros((H, W), np.uint8)
            for (ii, jj), Tt in tiles.items():
                if i0 <= ii <= i1 and j0 <= jj <= j1 and b in Tt["U"]:
                    A = np.nan_to_num(np.abs(Tt["U"][b].astype(np.float32)))
                    D[(j1 - jj) * npx:(j1 - jj + 1) * npx, (ii - i0) * npx:(ii - i0 + 1) * npx] = np.clip(np.ceil(A * 10), 0, 255).astype(np.uint8)
            lv = [D]
            while max(lv[-1].shape) > 1:
                P = lv[-1]; h2, w2 = (P.shape[0] + 1) // 2, (P.shape[1] + 1) // 2
                Q = np.zeros((h2 * 2, w2 * 2), np.uint8); Q[:P.shape[0], :P.shape[1]] = P
                lv.append(Q.reshape(h2, 2, w2, 2).max(axis=(1, 3)))
            self.PYR[b] = lv; self.ORG[b] = (i0 * cell, (j1 + 1) * cell, H, W)
        self.reason = f"pyramids {tot:.2e} px"
        self.ok = True
        self.gstep = float(np.median(np.abs(np.diff(self.fine)))) if self.N > 1 else 0.0

    # --- оценки ---
    def _rect_max(self, b, x0, x1, y0, y1):
        if b not in self.PYR:
            return np.zeros(len(x0))
        res = self.res; X0, Y1, H, W = self.ORG[b]
        c0 = np.clip(np.floor((x0 - X0) / res).astype(int), 0, W - 1); c1 = np.clip(np.floor((x1 - X0) / res).astype(int), 0, W - 1)
        r0 = np.clip(np.floor((Y1 - y1) / res).astype(int), 0, H - 1); r1 = np.clip(np.floor((Y1 - y0) / res).astype(int), 0, H - 1)
        L = np.ceil(np.log2(np.maximum(np.maximum(c1 - c0, r1 - r0) + 1, 1))).astype(int)
        out = np.zeros(len(x0)); lv = self.PYR[b]
        for l in np.unique(L):
            m = L == l; l_ = min(int(l), len(lv) - 1); P = lv[l_]
            rr0, rr1 = r0[m] >> l_, np.minimum(r1[m] >> l_, P.shape[0] - 1); cc0, cc1 = c0[m] >> l_, np.minimum(c1[m] >> l_, P.shape[1] - 1)
            out[m] = np.maximum(np.maximum(P[rr0, cc0], P[rr0, cc1]), np.maximum(P[rr1, cc0], P[rr1, cc1])) / 10.0
        return out

    def _bound(self, boxes, DXY):
        B = np.array(boxes); nb = len(B); NB = len(self.bands); K = self.K
        n0, n1, e0, e1, a0, a1, c0, c1 = B.T
        alo, ahi, clo, chi = self.Aa[a0], self.Aa[a1 - 1], self.Cc[c0], self.Cc[c1 - 1]
        slo, shi = self.ST[e0], self.ST[e1 - 1]
        span = int((n1 - n0).max())
        INF = np.inf
        xmin = np.full((nb, NB, K), INF); xmax = np.full((nb, NB, K), -INF); ymin = np.full((nb, NB, K), INF); ymax = np.full((nb, NB, K), -INF)
        anyv = np.zeros((nb, NB, K), bool); nav = np.zeros((nb, span), int)
        t, nr = self.t, self.nrm
        for j in range(span):
            n = n0 + j; okn = n < n1; n = np.minimum(n, self.N - 1)
            da = self.DA[n]
            o1 = slo[:, None] * self.KST * da; o2 = shi[:, None] * self.KST * da
            al_lo = np.minimum(o1, o2) + alo[:, None]; al_hi = np.maximum(o1, o2) + ahi[:, None]
            def rng(cf, lo, hi):
                v1, v2 = lo * cf, hi * cf; return np.minimum(v1, v2), np.maximum(v1, v2)
            txl, txh = rng(t[0], al_lo, al_hi); tyl, tyh = rng(t[1], al_lo, al_hi)
            nxl, nxh = rng(nr[0], clo[:, None], chi[:, None]); nyl, nyh = rng(nr[1], clo[:, None], chi[:, None])
            v = self.VAL[n] & okn[:, None, None]
            X = self.RX[n] + DXY[None, :, 0:1]; Y = self.RY[n] + DXY[None, :, 1:2]
            xmin = np.where(v, np.minimum(xmin, X + (txl + nxl)[..., None]), xmin); xmax = np.where(v, np.maximum(xmax, X + (txh + nxh)[..., None]), xmax)
            ymin = np.where(v, np.minimum(ymin, Y + (tyl + nyl)[..., None]), ymin); ymax = np.where(v, np.maximum(ymax, Y + (tyh + nyh)[..., None]), ymax)
            anyv |= v; nav[:, j] = np.where(okn, v.any(-1).sum(-1), 0)
        M = np.zeros((nb, NB))
        for bi, b in enumerate(self.bands):
            for k in range(K):
                m = anyv[:, bi, k]
                if m.any():
                    M[m, bi] = np.maximum(M[m, bi], self._rect_max(b, xmin[m, bi, k], xmax[m, bi, k], ymin[m, bi, k], ymax[m, bi, k]))
        Ms = M.T
        if self.cap and NB >= self.cap:
            capv = np.sort(Ms, axis=0)[-self.cap]; Ms = np.minimum(Ms, capv)
        ub = np.sqrt((Ms ** 2).sum(0))
        return np.where(nav.max(1) >= self.minb, ub, 0.0)

    # --- точный счёт (повтор detect.occlusion_search для набора сдвигов) ---
    def _score(self, n, e, AA, CC, disp):
        R = self.roots[n]
        if len(R) < self.minb:
            return np.full(len(AA), -1e9)
        t, nr = self.t, self.nrm
        SX = AA * t[0] + CC * nr[0]; SY = AA * t[1] + CC * nr[1]
        sig = self.ST[e] * self.KST
        per = []; cnt = np.zeros(len(AA))
        for b, lst in R.items():
            dx, dy = disp.get(b, (0.0, 0.0)) if disp else (0.0, 0.0)
            off = sig * (lst[0][0] - self.aref[n])
            bu = bs = bg = None
            for (al, x, y) in lst:
                X = x + dx + off * t[0] + SX; Y = y + dy + off * t[1] + SY
                u = self.mosU.get(b, X, Y).astype(np.float64); sv = self.mosS.get(b, X, Y).astype(np.float64); gv = self.mosG.get(b, X, Y).astype(np.float64)
                if bu is None:
                    bu, bs, bg = u, sv, gv
                else:
                    tk = np.abs(np.nan_to_num(u)) > np.abs(np.nan_to_num(bu)); bu, bs, bg = np.where(tk, u, bu), np.where(tk, sv, bs), np.where(tk, gv, bg)
            okb = np.isfinite(bu) & np.isfinite(bs) & (bs > 0) & np.isfinite(bg)
            inv = np.where(okb, 1.0 / np.where(okb, bs, 1), 0); uu = np.where(okb, bu, 0); x2 = -inv * (np.where(okb, bg, 1000.0) - 1000.0)
            per.append((b, inv, x2, uu)); cnt += okb
        if self.cap and len(per) >= self.cap:
            Au = np.sort(np.stack([np.abs(p[3]) for p in per]), axis=0)[-self.cap]
            per = [(b, inv, x2, np.clip(uu, -Au, Au)) for b, inv, x2, uu in per]
        best = np.full(len(AA), -1e9)
        for sp in detect.SPECTRA.values():
            A11 = A12 = A22 = B1 = B2 = 0.0
            for b, inv, x2, uu in per:
                x1 = sp.get(b, 1.0) * 1e3 * inv
                A11 = A11 + x1 * x1; A12 = A12 + x1 * x2; A22 = A22 + x2 * x2; B1 = B1 + x1 * uu; B2 = B2 + x2 * uu
            det = A11 * A22 - A12 * A12
            with np.errstate(invalid="ignore", divide="ignore"):
                ah = (A22 * B1 - A12 * B2) / (det + 1e-12 * (A11 * A22 + 1e-30)); fh = (A11 * B2 - A12 * B1) / (det + 1e-12 * (A11 * A22 + 1e-30))
                T2 = np.where(fh >= 0, ah * B1 + fh * B2, B1 * B1 / A11)
                Tm = np.where(cnt >= self.minb, np.sqrt(np.maximum(np.nan_to_num(T2, nan=0.0), 0)), -1e9)
            best = np.maximum(best, Tm)
        return best

    def _split(self, bx):
        n0, n1, e0, e1, a0, a1, c0, c1 = bx
        nm, em, am, cm = (n0 + n1) // 2, (e0 + e1) // 2, (a0 + a1) // 2, (c0 + c1) // 2
        st = float(self.Aa[1] - self.Aa[0]) if len(self.Aa) > 1 else 30.0
        ext = {"n": (n1 - n0 - 1) * self.gstep * 1000.0, "e": (e1 - e0 - 1) * 0.004 * self.KST * float(np.abs(self.DA).max()),
               "a": (a1 - a0 - 1) * st, "c": (c1 - c0 - 1) * st}
        for d in sorted(ext, key=ext.get, reverse=True):
            if d == "n" and n1 - n0 > 1: return [(n0, nm, e0, e1, a0, a1, c0, c1), (nm, n1, e0, e1, a0, a1, c0, c1)]
            if d == "e" and e1 - e0 > 1: return [(n0, n1, e0, em, a0, a1, c0, c1), (n0, n1, em, e1, a0, a1, c0, c1)]
            if d == "a" and a1 - a0 > 1: return [(n0, n1, e0, e1, a0, am, c0, c1), (n0, n1, e0, e1, am, a1, c0, c1)]
            if d == "c" and c1 - c0 > 1: return [(n0, n1, e0, e1, a0, a1, c0, cm), (n0, n1, e0, e1, a0, a1, cm, c1)]
        return [bx]

    def run(self, disp=None, floor=None, budget=None, chunk=8):
        """→ (best_score или None если ≤ floor, гипотеза (n, e, a, c), оставшаяся верхняя граница, число точных счетов)."""
        DXY = np.array([disp.get(b, (0.0, 0.0)) if disp else (0.0, 0.0) for b in self.bands])
        NE, NA, NC = len(self.ST), len(self.Aa), len(self.Cc)
        init = [(n, min(n + chunk, self.N), 0, NE, 0, NA, 0, NC) for n in range(0, self.N, chunk)]
        ub0 = self._bound(init, DXY)
        lim = -1e9 if floor is None else floor
        heap = [(-u, bx) for u, bx in zip(ub0, init) if u > lim]; heapq.heapify(heap)
        best, best_h, n_eval = -1e9, None, 0
        while heap and -heap[0][0] > max(best, lim):
            if budget and n_eval > budget:
                break
            batch = []
            while heap and len(batch) < 256 and -heap[0][0] > max(best, lim):
                batch.append(heapq.heappop(heap)[1])
            kids = []
            for bx in batch:
                n0, n1, e0, e1, a0, a1, c0, c1 = bx
                if (n1 - n0) * (e1 - e0) * (a1 - a0) * (c1 - c0) <= 64:
                    for n in range(n0, n1):
                        for ee in range(e0, e1):
                            AA, CC = np.meshgrid(self.Aa[a0:a1], self.Cc[c0:c1]); s = self._score(n, ee, AA.ravel(), CC.ravel(), disp); n_eval += s.size
                            j = int(np.argmax(s))
                            if s[j] > best:
                                best = float(s[j]); best_h = (n, ee, float(AA.ravel()[j]), float(CC.ravel()[j]))
                else:
                    kids += self._split(bx)
            if kids:
                uk = self._bound(kids, DXY)
                for u, bx in zip(uk, kids):
                    if u > max(best, lim):
                        heapq.heappush(heap, (-u, bx))
        ub_left = float(-heap[0][0]) if heap else (best if best > -1e9 else lim)
        return (best if best > lim else None), best_h, ub_left, n_eval

    def result_at(self, h, disp=None):
        """Поля результата конвейера (a_hat, f_hat, spectrum, hit, cnt, sx, sy, stretch, dM, n) в гипотезе h."""
        n, e, a, c = h
        R = {b: [(0, al, x, y) for (al, x, y) in lst] for b, lst in self.roots[n].items()}
        t, nr = self.t, self.nrm
        sig = self.ST[e] * self.KST
        d2 = {}
        for b, lst in self.roots[n].items():
            dx, dy = disp.get(b, (0.0, 0.0)) if disp else (0.0, 0.0)
            off = sig * (lst[0][0] - self.aref[n])
            d2[b] = (dx + off * t[0] + a * t[0] + c * nr[0], dy + off * t[1] + a * t[1] + c * nr[1])
        rr = detect.occlusion_search(R, self.mosU, self.mosS, self.mosG, self.bands, 0.0, step=30.0, disp=d2, minb=self.minb)
        sx, sy = a * t[0] + c * nr[0], a * t[1] + c * nr[1]
        return dict(rr, sx=float(sx), sy=float(sy), stretch=float(self.ST[e]), dM=float(self.fine[n]), n=int(n))


class SearchPhys(Search):
    """Физическая модель (THEORY §17): гипотеза θ = (n — фаза вдоль на сетке, da — доля шага фазы, dc — поперёк,
    dr — по радиусу). Положения полос линейны по (da, dc, dr): p_b = root_b(n) + Σ_d θ_d·J_{b,d}(n), где J — производные
    точной геометрии (конечные разности). Априорное распределение ошибки орбиты — штраф в счёте:
    T²_апост = T² − (a/σ_a)² − (dc/σ_c)² − (dr/σ_r)², a = фаза(n) + da. Отсечение по схеме Equant без изменений."""

    def __init__(self, ROOTS, fine, J, sig, grids, mosU, mosS, mosG, tiles, bands, minb=3, cap_rank=None,
                 cell=5000.0, margin=4500.0, max_px=float(__import__("os").environ.get("S2_BNB_MAXPX", "6e8"))):
        # J: (N, NB, D, 2) м на единицу θ_d; grids: список массивов значений θ_d; sig: (σ_a, σ_c, σ_r) км
        self.J = np.asarray(J, float); self.G = [np.asarray(g, float) for g in grids]; self.sig = sig
        self.Dn = len(self.G)
        super().__init__(ROOTS, fine, [0.0], 12.0, (1.0, 0.0), 0.0, 0.0, 30.0, mosU, mosS, mosG, tiles, bands,
                         minb=minb, cap_rank=cap_rank, cell=cell, margin=margin + float(np.abs(self.J).max(initial=0) * max(np.abs(g).max() for g in self.G)), max_px=max_px)

    def _pen(self, n, coefs):
        a = self.fine[n] + coefs[0]
        return (a / self.sig[0]) ** 2 + sum((coefs[d] / self.sig[d]) ** 2 for d in range(1, self.Dn))

    def _pen_min(self, n0, n1, lo, hi):
        # минимум штрафа по блоку (по каждой оси отдельно; оси независимы)
        a_lo = self.fine[n0] + lo[0]; a_hi = self.fine[n1 - 1] + hi[0]
        pa = 0.0 if a_lo <= 0 <= a_hi else min(a_lo ** 2, a_hi ** 2) / self.sig[0] ** 2
        p = pa
        for d in range(1, self.Dn):
            p += 0.0 if lo[d] <= 0 <= hi[d] else min(lo[d] ** 2, hi[d] ** 2) / self.sig[d] ** 2
        return p

    def _bound_phys(self, boxes, DXY):
        """Векторно по блокам: прямоугольники полос по θ-блоку → max|U| по пирамиде → UB с минимальным штрафом."""
        B = np.array(boxes); nb = len(B); NB = len(self.bands); K = self.K; D = self.Dn
        n0, n1 = B[:, 0], B[:, 1]
        lo = np.stack([self.G[d][B[:, 2 + 2 * d]] for d in range(D)], 1); hi = np.stack([self.G[d][B[:, 3 + 2 * d] - 1] for d in range(D)], 1)
        a_lo = self.fine[n0] + lo[:, 0]; a_hi = self.fine[n1 - 1] + hi[:, 0]
        pmin = np.where((a_lo <= 0) & (a_hi >= 0), 0.0, np.minimum(a_lo ** 2, a_hi ** 2)) / self.sig[0] ** 2
        for d in range(1, D):
            pmin = pmin + np.where((lo[:, d] <= 0) & (hi[:, d] >= 0), 0.0, np.minimum(lo[:, d] ** 2, hi[:, d] ** 2)) / self.sig[d] ** 2
        span = int((n1 - n0).max())
        INF = np.inf
        xmin = np.full((nb, NB, K), INF); xmax = np.full((nb, NB, K), -INF); ymin = np.full((nb, NB, K), INF); ymax = np.full((nb, NB, K), -INF)
        anyv = np.zeros((nb, NB, K), bool); nav = np.zeros((nb, span), int)
        for jj in range(span):
            n = n0 + jj; okn = n < n1; n = np.minimum(n, self.N - 1)
            Jn = self.J[n]                                            # (nb, NB, D, 2)
            l = lo[:, None, :]; h = hi[:, None, :]
            ax = l * Jn[..., 0]; bx_ = h * Jn[..., 0]; ay = l * Jn[..., 1]; by = h * Jn[..., 1]
            dxl = np.minimum(ax, bx_).sum(-1); dxh = np.maximum(ax, bx_).sum(-1); dyl = np.minimum(ay, by).sum(-1); dyh = np.maximum(ay, by).sum(-1)
            v = self.VAL[n] & okn[:, None, None]
            X = self.RX[n] + DXY[None, :, 0:1]; Y = self.RY[n] + DXY[None, :, 1:2]
            xmin = np.where(v, np.minimum(xmin, X + dxl[..., None]), xmin); xmax = np.where(v, np.maximum(xmax, X + dxh[..., None]), xmax)
            ymin = np.where(v, np.minimum(ymin, Y + dyl[..., None]), ymin); ymax = np.where(v, np.maximum(ymax, Y + dyh[..., None]), ymax)
            anyv |= v; nav[:, jj] = np.where(okn, v.any(-1).sum(-1), 0)
        M = np.zeros((nb, NB))
        for bi, b in enumerate(self.bands):
            for k in range(K):
                m = anyv[:, bi, k]
                if m.any():
                    M[m, bi] = np.maximum(M[m, bi], self._rect_max(b, xmin[m, bi, k], xmax[m, bi, k], ymin[m, bi, k], ymax[m, bi, k]))
        Ms = M.T
        if self.cap and NB >= self.cap:
            Ms = np.minimum(Ms, np.sort(Ms, axis=0)[-self.cap])
        ub2 = (Ms ** 2).sum(0) - pmin
        return np.where(nav.max(1) >= self.minb, np.sqrt(np.maximum(ub2, 0.0)), 0.0)

    def _score_phys(self, n, coef_sets, disp):
        """Точный счёт с штрафом для набора θ (coef_sets: (M, D)) при фазе n."""
        R = self.roots[n]
        if len(R) < self.minb:
            return np.full(len(coef_sets), -1e9)
        C = np.asarray(coef_sets, float)
        per = []; cnt = np.zeros(len(C))
        for b, lst in R.items():
            bi = self.BI[b]; dx, dy = disp.get(b, (0.0, 0.0)) if disp else (0.0, 0.0)
            SX = C @ self.J[n, bi, :, 0]; SY = C @ self.J[n, bi, :, 1]
            bu = bs = bg = None
            for (al, x, y) in lst:
                X = x + dx + SX; Y = y + dy + SY
                u = self.mosU.get(b, X, Y).astype(np.float64); sv = self.mosS.get(b, X, Y).astype(np.float64); gv = self.mosG.get(b, X, Y).astype(np.float64)
                if bu is None:
                    bu, bs, bg = u, sv, gv
                else:
                    tk = np.abs(np.nan_to_num(u)) > np.abs(np.nan_to_num(bu)); bu, bs, bg = np.where(tk, u, bu), np.where(tk, sv, bs), np.where(tk, gv, bg)
            okb = np.isfinite(bu) & np.isfinite(bs) & (bs > 0) & np.isfinite(bg)
            inv = np.where(okb, 1.0 / np.where(okb, bs, 1), 0); uu = np.where(okb, bu, 0); x2 = -inv * (np.where(okb, bg, 1000.0) - 1000.0)
            per.append((b, inv, x2, uu)); cnt += okb
        if self.cap and len(per) >= self.cap:
            Au = np.sort(np.stack([np.abs(p[3]) for p in per]), axis=0)[-self.cap]
            per = [(b, inv, x2, np.clip(uu, -Au, Au)) for b, inv, x2, uu in per]
        best = np.full(len(C), -1e9)
        for sp in detect.SPECTRA.values():
            A11 = A12 = A22 = B1 = B2 = 0.0
            for b, inv, x2, uu in per:
                x1 = sp.get(b, 1.0) * 1e3 * inv
                A11 = A11 + x1 * x1; A12 = A12 + x1 * x2; A22 = A22 + x2 * x2; B1 = B1 + x1 * uu; B2 = B2 + x2 * uu
            det = A11 * A22 - A12 * A12
            with np.errstate(invalid="ignore", divide="ignore"):
                ah = (A22 * B1 - A12 * B2) / (det + 1e-12 * (A11 * A22 + 1e-30)); fh = (A11 * B2 - A12 * B1) / (det + 1e-12 * (A11 * A22 + 1e-30))
                T2 = np.where(fh >= 0, ah * B1 + fh * B2, B1 * B1 / A11)
            pen = ((self.fine[n] + C[:, 0]) / self.sig[0]) ** 2 + sum((C[:, d] / self.sig[d]) ** 2 for d in range(1, self.Dn))
            Tm = np.where(cnt >= self.minb, np.sqrt(np.maximum(np.nan_to_num(T2, nan=0.0) - pen, 0)), -1e9)
            best = np.maximum(best, Tm)
        return best

    def run(self, disp=None, floor=None, budget=None, chunk=4):
        DXY = np.array([disp.get(b, (0.0, 0.0)) if disp else (0.0, 0.0) for b in self.bands])
        full = tuple(x for d in range(self.Dn) for x in (0, len(self.G[d])))
        init = [(n, min(n + chunk, self.N)) + full for n in range(0, self.N, chunk)]
        ub0 = self._bound_phys(init, DXY)
        lim = -1e9 if floor is None else floor
        heap = [(-u, bx) for u, bx in zip(ub0, init) if u > lim]; heapq.heapify(heap)
        best, best_h, n_eval = -1e9, None, 0; _it = 0; _dbg = os.environ.get("S2_BNB_DEBUG") == "1" and disp is None
        while heap and -heap[0][0] > max(best, lim):
            if budget and n_eval > budget:
                break
            _it += 1
            if _dbg and _it % 200 == 0:
                print(f"    bnb it {_it} eval {n_eval} heap {len(heap)} top {-heap[0][0]:.2f} best {best:.2f} lim {lim:.2f}", flush=True)
            batch = []
            while heap and len(batch) < 256 and -heap[0][0] > max(best, lim):
                batch.append(heapq.heappop(heap)[1])
            kids = []
            for bx in batch:
                sizes = [bx[1] - bx[0]] + [bx[3 + 2 * d] - bx[2 + 2 * d] for d in range(self.Dn)]
                if int(np.prod(sizes)) <= 64:
                    grids = [self.G[d][bx[2 + 2 * d]:bx[3 + 2 * d]] for d in range(self.Dn)]
                    C = np.array(np.meshgrid(*grids, indexing="ij")).reshape(self.Dn, -1).T
                    for n in range(bx[0], bx[1]):
                        s = self._score_phys(n, C, disp); n_eval += s.size
                        j = int(np.argmax(s))
                        if s[j] > best:
                            best = float(s[j]); best_h = (n, tuple(C[j]))
                    continue
                # делим по оси с наибольшим размахом на снимке
                ext = [(bx[1] - bx[0] - 1) * 1.0 * self.gstep * 1000.0 * 0 + (bx[1] - bx[0] - 1) * float(np.abs(self.J[:, :, 0, :]).max(initial=1.0)) * (self.G[0][1] - self.G[0][0] if len(self.G[0]) > 1 else 1.0)]
                for d in range(self.Dn):
                    lo, hi = bx[2 + 2 * d], bx[3 + 2 * d]
                    step = (self.G[d][1] - self.G[d][0]) if len(self.G[d]) > 1 else 0.0
                    ext.append((hi - lo - 1) * step * float(np.abs(self.J[:, :, d, :]).max(initial=0)))
                dd = int(np.argmax(ext))
                if dd == 0 and bx[1] - bx[0] > 1:
                    m = (bx[0] + bx[1]) // 2; kids += [(bx[0], m) + bx[2:], (m, bx[1]) + bx[2:]]
                else:
                    if dd == 0:
                        dd = 1 + int(np.argmax(ext[1:]))
                    d = dd - 1; lo, hi = bx[2 + 2 * d], bx[3 + 2 * d]
                    if hi - lo <= 1:
                        cand = [i for i in range(self.Dn) if bx[3 + 2 * i] - bx[2 + 2 * i] > 1]
                        if not cand:
                            m = (bx[0] + bx[1]) // 2; kids += [(bx[0], m) + bx[2:], (m, bx[1]) + bx[2:]]; continue
                        d = cand[0]; lo, hi = bx[2 + 2 * d], bx[3 + 2 * d]
                    mid = (lo + hi) // 2; l = list(bx)
                    l1 = l.copy(); l1[3 + 2 * d] = mid; l2 = l.copy(); l2[2 + 2 * d] = mid
                    kids += [tuple(l1), tuple(l2)]
            if kids:
                uk = self._bound_phys(kids, DXY)
                for u, bx in zip(uk, kids):
                    if u > max(best, lim):
                        heapq.heappush(heap, (-u, bx))
        ub_left = float(-heap[0][0]) if heap else (best if best > -1e9 else lim)
        return (best if best > lim else None), best_h, ub_left, n_eval

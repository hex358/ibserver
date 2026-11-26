"""Геометрия одного пролёта для работы с формой: орбиты (правило fit-span), корни полос как функция
ошибки орбиты θ = (вдоль, поперёк, радиус) км, якобиан J (м снимка на км), Солнце и Sentinel в осях объекта.

Точка тела объекта со смещением δ (м) от центра видна там же, где был бы весь объект при ошибке орбиты δ:
поэтому якобиан ошибки орбиты J_b (2×3, м/км) — это и есть камера Sentinel-2 для формы (аффинная,
отдельная на каждую полосу; полосы снимают в разные моменты → разные ракурсы и смаз)."""
from __future__ import annotations

import glob
import math
import pickle
from pathlib import Path

import numpy as np

from . import orbits as O, geometry as G, imagery as I, s2meta as M
from .orbits import dt_from_jd

ROOT = Path(__file__).resolve().parents[1]
CELL = 5000.0


class Orbits:
    def __init__(self, until="20261122T2348"):
        snaps = sorted(p for p in (ROOT / "data/tle_snapshots").iterdir() if p.is_dir() and p.name <= until)
        self.SG, self.GP, self.S2 = self._load(snaps, "supgp_*.tle"), self._load(snaps, "gp_*.tle"), {}
        for sd in snaps:
            f = sd / "sentinel2.tle"
            if f.exists():
                for o in O.parse_tle(f.read_text()):
                    if o.name not in self.S2 or o.epoch_jd > self.S2[o.name].epoch_jd:
                        self.S2[o.name] = o

    @staticmethod
    def _load(snaps, pat):
        allo = {}
        for sd in snaps:
            for f in sd.glob(pat):
                for o in O.parse_tle(f.read_text()):
                    allo.setdefault(o.catnr, {})[round(o.epoch_jd, 6)] = o
        return {k: list(v.values()) for k, v in allo.items()}

    def orbit_for(self, e):
        """Правило fit-span (THEORY §14): SupGP с эпохой до снимка, иначе GP с эпохой после."""
        sg = [o for o in self.SG.get(e["cat"], []) if o.epoch_jd <= e["jd"] and e["jd"] - o.epoch_jd <= 2.0]
        if sg:
            return max(sg, key=lambda o: o.epoch_jd)
        gp = [o for o in self.GP.get(e["cat"], []) if o.epoch_jd > e["jd"] and o.epoch_jd - e["jd"] <= 2.0]
        if gp:
            return min(gp, key=lambda o: o.epoch_jd)
        gp = [o for o in self.GP.get(e["cat"], []) if abs(o.epoch_jd - e["jd"]) <= 5.0]
        return min(gp, key=lambda o: abs(o.epoch_jd - e["jd"])) if gp else None


class Pass:
    def __init__(self, e, r, orbs: Orbits):
        self.e, self.r = e, r
        self.sen = orbs.S2[e["sent"]]; self.orb = orbs.orbit_for(e); self.epsg = int(r["epsg"])
        its = I.items_around(e["lon"], e["lat"], dt_from_jd(e["jd"]), e["sent"][-1], d=1.0)
        psi = G.track_azimuth(self.sen, e["jd"]); H = float(np.linalg.norm(self.sen.ecef([e["jd"]])[0][0])) - G.R_LOC
        self.DG = {}
        for it in its:
            try:
                gg = M.parse_grids(I.get_text(it.assets["granule_metadata"].href), psi, H, it.id)
            except Exception:
                continue
            for b, lst in gg.items():
                for dd in lst:
                    dd["epsg"] = I.epsg_of(it)
                self.DG.setdefault(b, []).extend(lst)
        self.bands = list(self.DG)

    def regrid(self, sen=None, all60=False):
        """Пересчитать углы полос в трёхмерной геометрии с орбитой sen (обычно GPS) — s2meta.alpha_3d."""
        if sen is not None:
            self.sen = sen
        psi = G.track_azimuth(self.sen, self.e["jd"]); H = float(np.linalg.norm(self.sen.ecef([self.e["jd"]])[0][0])) - G.R_LOC
        its = I.items_around(self.e["lon"], self.e["lat"], dt_from_jd(self.e["jd"]), self.e["sent"][-1], d=1.0)
        self.DG = {}
        for it in its:
            try:
                gg = M.parse_grids(I.get_text(it.assets["granule_metadata"].href), psi, H, it.id, sen=self.sen,
                                   epsg=I.epsg_of(it), jd0=self.e["jd"], all60=all60)
            except Exception:
                continue
            for b, lst in gg.items():
                for dd in lst:
                    dd["epsg"] = I.epsg_of(it)
                self.DG.setdefault(b, []).extend(lst)
        self.bands = list(self.DG)

    def roots(self, theta):
        tr = G.apparent_track(self.sen, self.orb, self.e["jd"], np.linspace(-2.6, 2.6, 2000), dpos_km=np.asarray(theta, float))
        ok = np.isfinite(tr["ground"][:, 0])
        X, Y = G.ecef_to_utm(tr["ground"][ok], self.epsg)
        R = M.band_roots(tr["alpha"][ok], X, Y, self.DG, self.bands, epsg_main=self.epsg)
        jd = {}
        for b, lst in R.items():                     # момент съёмки полосы: по α корня
            jd[b] = [float(np.interp(q[1], tr["alpha"][ok], tr["jd"][ok])) for q in lst]
        return R, jd

    def positions(self, theta, near=None):
        """Положение объекта в каждой полосе (ближайший к near корень) и момент съёмки."""
        R, jd = self.roots(theta)
        out = {}
        for b, lst in R.items():
            k = 0 if near is None or b not in near else int(np.argmin([math.hypot(q[2] - near[b][0], q[3] - near[b][1]) for q in lst]))
            out[b] = (lst[k][2], lst[k][3], jd[b][k])
        return out

    def fit_theta(self, target, x0=None):
        """θ, при котором положения полос ближе всего к target (м)."""
        from scipy.optimize import least_squares
        x0 = np.array([self.r.get("dM", 0.0), 0.0, 0.0] if x0 is None else x0, float)

        def res(th):
            P = self.positions(th, target); o = []
            for b, (tx, ty) in target.items():
                o += [P[b][0] - tx, P[b][1] - ty] if b in P else [3000.0, 3000.0]
            return np.array(o)
        best = None
        for c0 in (-1.0, 0.0, 1.0):
            for h0 in (-1.0, 0.0, 1.0):
                try:
                    s_ = least_squares(res, x0 + [0, c0, h0], x_scale=[1.0, 0.5, 0.5], diff_step=1e-3, max_nfev=60)
                except Exception:
                    continue
                c = float(np.sqrt(np.mean(s_.fun ** 2)))
                if best is None or c < best[0]:
                    best = (c, s_.x)
        return best

    def jacobian(self, theta, dlt=0.05):
        P0 = self.positions(theta)
        J = {}
        for d in range(3):
            th = np.array(theta, float); th[d] += dlt
            P = self.positions(th, {b: v[:2] for b, v in P0.items()})
            for b in P0:
                if b in P:
                    J.setdefault(b, np.zeros((2, 3)))[:, d] = (np.array(P[b][:2]) - np.array(P0[b][:2])) / dlt
        return P0, J

    def frame(self, jd):
        """Оси объекта (вдоль, поперёк, радиус) в ECEF, направления на Солнце и на Sentinel в этих осях."""
        rO, vO = self.orb.ecef([jd]); rO, vO = rO[0], vO[0]
        uv = G.unit(vO); ur = G.unit(rO); uc = G.unit(np.cross(ur, uv))
        A = np.stack([uv, uc, ur])
        su = O.sun_ecef([jd])[0]
        rS = self.sen.ecef([jd])[0][0]
        los = G.unit(rS - rO)
        return A, A @ su, A @ los, float(np.linalg.norm(rS - rO))


def tile_dir(e, r):
    key = f"{e['obj'].split()[0]}_{e['sent'][-2:]}_{round(e['jd'] * 86400)}_glrt_z*"
    ds = sorted(glob.glob(str(I.CACHE / "tiles" / key)))
    ds = [d for d in ds if d.endswith(f"_e{r['epsg']}_v2")] or ds
    return ds


def patches(e, r, centers, half=15, keys=("Z", "S", "G")):
    """Окна (2·half+1)² пикселей 10 м вокруг centers[b] из кэша ячеек: r = Z·S (DN), σ = S, фон g = G − 1000."""
    out = {}
    for d in tile_dir(e, r)[::-1]:
        need = {(int((cx + dx) // CELL), int((cy + dy) // CELL)) for cx, cy in centers.values() for dx in (-400, 0, 400) for dy in (-400, 0, 400)}
        tiles = {}
        for (ci, cj) in need:
            p = Path(d) / f"{ci}_{cj}.pkl"
            if p.exists():
                tiles[(ci, cj)] = pickle.loads(p.read_bytes())
        if not tiles:
            continue
        for b, (cx, cy) in centers.items():
            if b in out:
                continue
            x0 = math.floor(cx / 10.0) * 10.0 + 5.0; y0 = math.floor(cy / 10.0) * 10.0 + 5.0
            g = np.arange(-half, half + 1) * 10.0
            X, Y = np.meshgrid(x0 + g, y0 - g)          # строки — на север→юг, как на снимке
            vals = {}
            for k in keys:
                arrs = [T[k].get(b) for T in tiles.values()]
                if any(a is None for a in arrs):
                    break
                vals[k] = _sample(tiles, k, b, X, Y)
            if len(vals) == len(keys) and np.isfinite(vals["Z"]).mean() > 0.9:
                out[b] = dict(X=X, Y=Y, r=vals["Z"] * vals["S"], s=vals["S"], g=vals["G"] - 1000.0, z=vals["Z"])
    return out


def _sample(tiles, k, b, X, Y):
    V = np.full(X.shape, np.nan)
    for (ci, cj), T in tiles.items():
        A = np.asarray(T[k][b], float); n = A.shape[0]; res = CELL / n
        m = (np.floor(X / CELL) == ci) & (np.floor(Y / CELL) == cj)
        if not m.any():
            continue
        col = np.clip(np.floor((X[m] - ci * CELL) / res).astype(int), 0, n - 1)
        row = np.clip(np.floor(((cj + 1) * CELL - Y[m]) / res).astype(int), 0, n - 1)
        V[m] = A[row, col]
    return V


def pick_orbit(orbs: Orbits, e):
    """Та же орбита, что в прогоне детекции (по источнику и возрасту из события), иначе правило fit-span."""
    src, age = e.get("orbit_src", ""), e.get("orbit_age")
    pool = orbs.SG.get(e["cat"], []) if src.startswith("supgp") else orbs.GP.get(e["cat"], [])
    if age is not None and pool:
        best = min(pool, key=lambda o: min(abs(o.epoch_jd - (e["jd"] - age)), abs(o.epoch_jd - (e["jd"] + age))))
        return best
    return orbs.orbit_for(e)


def gauss_newton_theta(P, target, theta0, iters=4):
    """θ (км), при котором положения полос ближе всего к target: Гаусс–Ньютон с численным якобианом."""
    th = np.array(theta0, float); P0 = J = None
    for _ in range(iters):
        P0, J = P.jacobian(th)
        bs = [b for b in target if b in P0 and b in J]
        A = np.concatenate([J[b] for b in bs]); rv = np.concatenate([np.array(target[b]) - np.array(P0[b][:2]) for b in bs])
        # априорное распределение ошибки орбиты (THEORY §17) как псевдонаблюдения: поперёк 1.1 км, радиус 1.9 км;
        # погрешность положения в полосе ~30 м. Без него радиус (почти вырожден на снимке) уходит на сотни км.
        sp = 30.0; pr = np.array([[0, 1 / 1.1, 0], [0, 0, 1 / 1.9]]) * sp
        A2 = np.vstack([A, pr]); r2 = np.concatenate([rv, -pr @ th])
        dth, *_ = np.linalg.lstsq(A2, r2, rcond=None)
        dth = np.clip(dth, -5.0, 5.0)
        th = th + dth
        if np.abs(dth).max() < 1e-3:
            break
    P0, J = P.jacobian(th)
    bs = [b for b in target if b in P0]
    rms = float(np.sqrt(np.mean([(P0[b][0] - target[b][0]) ** 2 + (P0[b][1] - target[b][1]) ** 2 for b in bs]) / 2))
    return th, P0, J, rms


def window_residuals(P, pos, half_m=600.0, mask_m=60.0):
    """Сырые остатки (не согласованный фильтр) в окне вокруг положения объекта в каждой полосе.
    Фон — loglocal (THEORY §13) по полосам, далёким по времени съёмки; пиксели вокруг положений объекта во ВСЕХ
    полосах исключены из подгонки фона, чтобы объект не «съедался» регрессией. Разрешение — родное у полосы."""
    from . import bgsub
    e, r = P.e, P.r
    its = I.items_around(e["lon"], e["lat"], dt_from_jd(e["jd"]), e["sent"][-1], d=1.0)
    main = [it for it in its if it.id == r.get("item")] or its
    out = {}
    for b, (px, py) in pos.items():
        res = float(M.BRES.get(b, 20))
        if res > 60:
            continue
        x0 = math.floor((px - half_m) / 20.0) * 20.0; y0 = math.floor((py - half_m) / 20.0) * 20.0
        bb = (x0, y0, x0 + 2 * half_m + 40.0, y0 + 2 * half_m + 40.0)
        IM = None
        for it in main[:3]:
            try:
                IM = {c: I.read_window(it.assets[c].href, P.epsg, bb, res=res) for c in pos}
            except Exception:
                IM = None
                continue
            if (IM[b] > 0).mean() > 0.9:
                break
        if IM is None:
            continue
        H_, W_ = IM[b].shape
        X = bb[0] + (np.arange(W_) + 0.5) * res; Y = bb[3] - (np.arange(H_) + 0.5) * res
        XX, YY = np.meshgrid(X, Y)
        valid = np.all(np.stack([IM[c] > 0 for c in IM]), 0)
        fitmask = valid.copy()
        for c, (qx, qy) in pos.items():
            fitmask &= np.hypot(XX - qx, YY - qy) > mask_m
        near = {c for c, (qx, qy) in pos.items() if math.hypot(qx - px, qy - py) < 300.0}
        LG = {c: np.log(np.maximum(IM[c], 1.0)).astype(np.float32) for c in IM}
        regs = [LG[c] for c in IM if c not in near]
        if len(regs) < 2:
            continue
        blk = max(int(100 / res), 2); win = max(int(300 / res), 3 * blk)
        Rl = bgsub.predict_band(LG[b], regs, fitmask, block=blk, win=win, out_mask=valid, min_n=40)
        pred = IM[b] * np.exp(-np.nan_to_num(Rl))
        R = IM[b] - pred
        R[~valid | ~np.isfinite(Rl)] = np.nan
        far = np.isfinite(R) & (np.hypot(XX - px, YY - py) > 200.0)
        med = float(np.median(R[far])); sig = float(1.4826 * np.median(np.abs(R[far] - med)))
        out[b] = dict(X=XX, Y=YY, r=R - med, s=sig, g=pred - 1000.0, res=res, jd=None)
    return out

"""Тензор инерции по фото (s2sat/diffpose.DiffTumble) — проверка подстановкой, одинаково для всех объектов.

  python scripts/tumble_test.py <имя> [--rate 60] [--from-truth] [--threads 1]
Подставной объект: поза по пикселям, кувыркание без внешних моментов — моменты инерции и главные оси тонкостенной
оболочки модели (s2sat.debris.shell_mass_props; только как «истина» подстановки), ω₀ = rate °/с не вдоль главной оси
(та же случайная ось, что в spin_real), яркости — НК по настоящему снимку, шум ×√κ.
Решение: равномерное вращение (старт: истинное ω₀ при --from-truth) → кувыркание из нескольких стартов главных осей
→ отношения моментов, главные оси, выигрыш над равномерным (порог BIC: κ·5·ln N за 5 лишних параметров).
"""
import argparse, json, math, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import modelcat as MC, scenegeom as SG, debris as DB
from s2sat.diffpose import DiffPose, DiffSpinPose, DiffTumble
from s2sat import autopose as AP
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation as Ro

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--rate", type=float, default=60.0)
ap.add_argument("--from-truth", action="store_true"); ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args(); torch.set_num_threads(a.threads); n = a.name; t0_ = time.time()
rc = pickle.loads((ROOT / "results/shape" / f"{n}.pkl").read_bytes()); MR.prepare(rc)
spec = MC.spec_for(int(rc["cat"]), n); C = SG.lvlh_to_fit(n, rc)
Wt = DiffTumble(rc, spec, C); Wf = DiffSpinPose(rc, spec, C)
for D in (Wt, Wf):
    D.grey = False
G = Wf.groups(); gt = [float(np.median([Wf.sc.t[b] for b in g])) for g in G]; Wf.t0 = Wt.t0 = gt[int(np.argmax([len(g) for g in G]))]
pp = json.loads((ROOT / "results/pose_pix_off" / f"{n}.json").read_text()); kap = float(pp["kappa"])
Do = DiffPose(rc, spec, C, band_off=25.0); xs0 = np.asarray(pp["x"], float)[:Do.o0]
# истина подстановки: моменты и главные оси оболочки модели (I₃ — наименьший → λ = ln I₁/I₃, ln I₂/I₃ ≥ 0)
_, I, ev, axm = DB.shell_mass_props(np.asarray(spec["V"], float), np.asarray(spec["F"]), 1.0)
order = np.argsort(ev)[::-1]; ev = ev[order]; axm = axm[:, order]
if np.linalg.det(axm) < 0:
    axm[:, 2] *= -1
q_t = Ro.from_matrix(axm).as_rotvec(); lam_t = np.log(ev[:2] / ev[2])
rng = np.random.default_rng(7); ax_t = rng.normal(size=3); ax_t /= np.linalg.norm(ax_t)
xt = Wt.from_spin(np.r_[xs0, ax_t * math.radians(a.rate)], q_t, lam_t)
for b in Wt.sc.bands:
    bd = Wt.sc.B[b]; occ, cols = Wt.columns(xt, b); A_ = cols.T * bd.gm / bd.s; y_ = (bd.r.numpy() + bd.g.numpy() * occ.sum(0)) / bd.s
    rb = nnls(A_, y_)[0]; m_ = cols.T @ rb * bd.gm - bd.g.numpy() * occ.sum(0)
    bd.r = torch.tensor(m_ + math.sqrt(kap) * bd.s * rng.standard_normal(len(m_)), dtype=torch.float32)
from s2sat.diffmodel import transfer_r
transfer_r(Wt.sc, Wf.sc)
nm_t = (AP.rotvec(xs0[:3]) @ Wt.Rbase.numpy() @ Wt.M2B.numpy() @ axm).T @ ax_t    # ω₀ в главных осях
print(f"{n}: полос {len(Wt.sc.bands)}, разброс {np.ptp(list(Wt.sc.t.values())):.2f} с, κ {kap:.2f}; подстановка {a.rate:.0f} °/с, "
      f"моменты I₁:I₂:I₃ = {np.round(ev / ev[2], 2).tolist()}, ω₀ в главных осях {np.round(nm_t, 2).tolist()}", flush=True)
Lt = float(Wt.L(Wt.polish(xt, iters=150)[0])[0])
xw = np.r_[xs0, ax_t * math.radians(a.rate)]; xw, Lw = Wf.polish(xw, iters=150)
for _ in range(3):
    x2, L2 = Wf.polish(xw, iters=150)
    if L2 >= Lw - 0.5:
        break
    xw, Lw = x2, L2
print(f"   равномерное вращение: {math.degrees(np.linalg.norm(xw[-3:])):.1f} °/с, невязка {Lw:.0f}; истина (кувыркание, дожатая) {Lt:.0f}", flush=True)
best = None; rs = np.random.default_rng(3)
starts = [np.zeros(3)] + [Ro.random(random_state=int(rs.integers(1 << 30))).as_rotvec() for _ in range(11)]
for q0 in starts:
    for lam0 in ((1.0, 0.5), (0.5, 1.0)):
        x, L = Wt.polish(Wt.from_spin(xw, q0, lam0), iters=150)
        for _ in range(3):
            x2, L2 = Wt.polish(x, iters=150)
            if L2 >= L - 0.5:
                break
            x, L = x2, L2
        if best is None or L < best[1]:
            best = (x, L)
xb, Lb = best; Im, Ax = Wt.inertia_model(xb)
o = np.argsort(Im)[::-1]; Im = Im[o] / Im[o][2]; Ax = Ax[:, o]
ang = [math.degrees(math.acos(min(1.0, abs(float(Ax[:, k] @ axm[:, k]))))) for k in range(3)]
thr = kap * 5 * math.log(Wt.npix)
print(f"   кувыркание: моменты {np.round(Im, 2).tolist()} (истина {np.round(ev / ev[2], 2).tolist()}); главные оси от истины "
      f"{np.round(ang, 1).tolist()}°; невязка {Lb:.0f}, выигрыш над равномерным {Lw - Lb:.0f} (порог {thr:.0f})", flush=True)
out = dict(name=n, rate=a.rate, I_true=(ev / ev[2]).tolist(), I_found=Im.tolist(), axes_err_deg=ang, L_uniform=Lw, L_tumble=Lb,
           L_truth=Lt, threshold=thr, from_truth=a.from_truth, seconds=time.time() - t0_)
(ROOT / "results/tumble_test").mkdir(exist_ok=True)
(ROOT / "results/tumble_test" / f"{n}_r{a.rate:.0f}{'_ft' if a.from_truth else ''}.json").write_text(json.dumps(out, indent=1))

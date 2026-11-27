"""Центр масс по фото (s2sat/diffpose.DiffSpinCoM) — проверка подстановкой, одинаково для всех объектов.

  python scripts/com_test.py <имя> [--rate 60] [--com 3] [--from-truth] [--threads 1]
Подставной объект: поза по пикселям (results/pose_pix_off), вращение rate °/с (ось — та же, что в spin_real),
центр масс смещён от начала модели на |c| = com м (случайное направление), яркости — НК по настоящему снимку в каждой
полосе, шум ×√κ. Решение: вращение (solve_spin, вокруг начала модели) → дожим с центром масс как свободным параметром
→ c и σ(c) по гессиану. --com 0 — проверка, что смещение не выдумывается. --from-truth — старт с истинного вращения
(быстрая проверка измеримости c, без поиска вращения).
"""
import argparse, json, math, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import modelcat as MC, scenegeom as SG
from s2sat.diffpose import DiffPose, DiffSpinPose, DiffSpinCoM, solve_spin
from s2sat.diffmodel import transfer_r
from scipy.optimize import nnls

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--rate", type=float, default=60.0)
ap.add_argument("--com", type=float, default=3.0); ap.add_argument("--from-truth", action="store_true")
ap.add_argument("--threads", type=int, default=1); ap.add_argument("--n-rot", type=int, default=1500)
a = ap.parse_args(); torch.set_num_threads(a.threads); n = a.name; t0_ = time.time()
rc = pickle.loads((ROOT / "results/shape" / f"{n}.pkl").read_bytes()); MR.prepare(rc)
spec = MC.spec_for(int(rc["cat"]), n); C = SG.lvlh_to_fit(n, rc)
Wm = DiffSpinCoM(rc, spec, C); Wf = DiffSpinPose(rc, spec, C)
for D in (Wm, Wf):
    D.grey = False
G = Wf.groups(); gt = [float(np.median([Wf.sc.t[b] for b in g])) for g in G]; Wf.t0 = Wm.t0 = gt[int(np.argmax([len(g) for g in G]))]
pp = json.loads((ROOT / "results/pose_pix_off" / f"{n}.json").read_text()); kap = float(pp["kappa"])
Do = DiffPose(rc, spec, C, band_off=25.0); xs0 = np.asarray(pp["x"], float)[:Do.o0]
rng = np.random.default_rng(7); ax_t = rng.normal(size=3); ax_t /= np.linalg.norm(ax_t)
rng2 = np.random.default_rng(11); c_dir = rng2.normal(size=3); c_t = c_dir / np.linalg.norm(c_dir) * a.com
xt = Wm.from_spin(np.r_[xs0, ax_t * math.radians(a.rate)], c_t)
for b in Wm.sc.bands:      # яркости по полосам — НК по настоящему снимку в той же позе
    bd = Wm.sc.B[b]; occ, cols = Wm.columns(xt, b); A_ = cols.T * bd.gm / bd.s; y_ = (bd.r.numpy() + bd.g.numpy() * occ.sum(0)) / bd.s
    rb = nnls(A_, y_)[0]; m_ = cols.T @ rb * bd.gm - bd.g.numpy() * occ.sum(0)
    bd.r = torch.tensor(m_ + math.sqrt(kap) * bd.s * rng.standard_normal(len(m_)), dtype=torch.float32)
transfer_r(Wm.sc, Wf.sc)
print(f"{n}: полос {len(Wm.sc.bands)}, разброс времён {np.ptp(list(Wm.sc.t.values())):.2f} с, κ {kap:.2f}; подстановка "
      f"{a.rate:.0f} °/с, центр масс {np.round(c_t, 2).tolist()} м (|c| {a.com:.1f} м), априори σ {Wm.sig_com:.1f} м", flush=True)
if a.from_truth:
    xw = np.r_[xs0, ax_t * math.radians(a.rate)]; xw, Lw = Wf.polish(xw)
else:
    Sf, Sc = DiffPose(rc, spec, C), DiffPose(rc, spec, C, coarse=True); Wc = DiffSpinPose(rc, spec, C, coarse=True)
    for D in (Sf, Sc, Wc):
        D.grey = False; transfer_r(Wm.sc, D.sc)
    Wc.t0 = Wf.t0
    r = solve_spin(Wf, Wc, Sf, Sc, n_rot=a.n_rot, kap=kap, log=lambda s: None); xw, Lw = r["x"], r["L"]
rate = math.degrees(np.linalg.norm(xw[-3:]))
print(f"   вращение (вокруг начала модели): {rate:.1f} °/с, невязка {Lw:.0f}", flush=True)
xc, Lc, L0, sig, E, cov = Wm.fit_com(xw, kap=kap); c = xc[-6:-3]; nm = Wm.axis_model(xc)
Lt = float(Wm.L(Wm.polish(xt, iters=150)[0])[0])
d_ = c - c_t; e_par = float(d_ @ nm); e_perp = float(np.linalg.norm(d_ - e_par * nm)); sp = float(np.linalg.norm(sig))
# центр масс поперёк оси: проекция на плоскость ⊥ оси через начало модели
c_perp = c - (c @ nm) * nm; ct_perp = c_t - (c_t @ nm) * nm
print(f"   ось вращения в осях модели {np.round(nm, 2).tolist()}; центр масс — на этой оси через точку {np.round(c_perp, 2).tolist()} м "
      f"(истина {np.round(ct_perp, 2).tolist()}); σ поперёк оси {np.round(sig, 2).tolist()} м", flush=True)
print(f"   ошибка поперёк оси {e_perp:.2f} м ({e_perp / max(sp, 1e-9):.1f} σ), вдоль оси {e_par:+.2f} м (по вращению не измеряется)", flush=True)
print(f"   невязка: c = 0 → {L0:.0f}, с центром масс {Lc:.0f} (выигрыш {L0 - Lc:.0f}), истина (дожатая) {Lt:.0f}", flush=True)
err = e_perp
out = dict(name=n, rate_inj=a.rate, com_true=c_t.tolist(), com=c.tolist(), com_sigma_perp=sig.tolist(), com_err_perp_m=e_perp, com_err_par_m=e_par, axis_model=nm.tolist(),
           rate_found=rate, L_c0=L0, L=Lc, L_truth=Lt, from_truth=a.from_truth, seconds=time.time() - t0_)
(ROOT / "results/com_test").mkdir(exist_ok=True)
(ROOT / "results/com_test" / f"{n}_r{a.rate:.0f}_c{a.com:.0f}{'_ft' if a.from_truth else ''}.json").write_text(json.dumps(out, indent=1))

"""Даёт ли освещение Солнцем (s2sat/diffmodel.py, shade) однозначное вращение? Подстановка с затенением — две подгонки:
без затенения (яркость групп свободна) и с ним (яркость — по Солнцу и визиру). Тот же розыгрыш, что diffspin_test.py.

  python scripts/diffspin_shade_test.py STARLINK-30580_20261120T111642 --rates 60,120
"""
import argparse, copy, math, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import autopose as AP, modelcat as MC, scenegeom as SG
from s2sat.diffmodel import rotvec_to_mat
from s2sat.diffspin import DiffSpin

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("--norad", type=int, default=22220); ap.add_argument("--rates", default="60,120")
ap.add_argument("--seed", type=int, default=3); ap.add_argument("--noise", type=float, default=1.5); ap.add_argument("--iters", type=int, default=150)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
rc = pickle.loads((ROOT / "results/shape" / f"{a.base}.pkl").read_bytes()); MR.prepare(rc)
spec = MC.spec_for(a.norad, ""); C = SG.lvlh_to_fit(a.base, rc); want = {float(v) for v in a.rates.split(",")}
Vb_ = spec["V"] @ np.asarray(spec["M2B"]).T; e_long = np.eye(3)[int(np.argmax(np.ptp(Vb_, 0)))]


def visible(R0v, w_deg):
    L = AP.rotvec(R0v) @ e_long; return w_deg - (w_deg @ L) * L


for rate in [0.0, 20.0, 60.0, 120.0]:
    w0 = rng.normal(size=3); w0 /= np.linalg.norm(w0); R0 = w0 * rng.uniform(0.3, 2.5)
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    xt = np.r_[R0, ax * math.radians(rate), rng.normal(0, 3.0, 2)]
    if rate not in want:
        continue
    rc2 = copy.deepcopy(rc); G = DiffSpin(rc2, spec, C, seed=a.seed, shade=True); sc = G.sc
    rho = np.array([0.05, 0.6, 0.3])[:sc.ng]; spec_amp = 2.0                   # ячейки, светлое, тёмное; сила блика
    with torch.no_grad():
        x_t = torch.tensor(xt[None], dtype=torch.float32); R0m = rotvec_to_mat(x_t[:, 0:3]); w = x_t[:, 3:6]
        dpos = torch.tensor([[xt[6], xt[7], 0.0]], dtype=torch.float32)
        for b in sc.bands:
            bd = sc.B[b]; Rf = sc.C @ rotvec_to_mat(w * G.t[b]) @ R0m
            occ, lit, sp = sc.band_occ(b, Rf, dpos, shade=True)
            sig = (torch.tensor(rho, dtype=torch.float32)[None, :, None] * lit).sum(1) * bd.gm + spec_amp * sp[:, 0] * bd.gm - bd.g * occ.sum(1)
            noise = a.noise * bd.s * torch.tensor(rng.standard_normal(sig.shape[1]), dtype=torch.float32)
            bd.r = (sig[0] + noise)
    wt_v = visible(R0, ax * rate)
    for shade in (False, True):
        t0 = time.time(); D = DiffSpin(rc2, spec, C, seed=a.seed, shade=shade)
        for b in sc.bands:
            D.sc.B[b].r = sc.B[b].r.clone()
        res = D.solve(iters=a.iters, log=lambda s: None)
        xe = np.asarray(res["x"]); we_v = visible(xe[0:3], np.degrees(xe[3:6]))
        xt_d, Lt_d = D.polish(xt, iters=60)
        print(f"истина {rate:.0f} °/с (видимая {np.linalg.norm(wt_v):.1f}), {'С ЗАТЕНЕНИЕМ' if shade else 'без затенения'}: "
              f"найдено {res['rate_deg_s']:.1f} (видимая {np.linalg.norm(we_v):.1f}) ± {res['rate_sigma_deg_s']:.1f}, семейство "
              f"{[round(v) for v in res['family_rates_deg_s']]}; невязка найдено {res['L_spin']:.1f} / истина {Lt_d:.1f}; "
              f"{time.time() - t0:.0f} с", flush=True)

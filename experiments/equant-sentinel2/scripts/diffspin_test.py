"""Проверка вращения градиентным спуском (s2sat/diffspin.py) на той же подстановке, что spin_diag.py / inject_spin.py.

  python scripts/diffspin_test.py STARLINK-30580_20261120T111642 --rates 20,60,120 [--B 16 --iters 150]
"""
import argparse, copy, json, math, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import autopose as AP, modelcat as MC, scenegeom as SG, spin as SP
from s2sat.diffspin import DiffSpin

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("--norad", type=int, default=22220); ap.add_argument("--rates", default="20,60,120")
ap.add_argument("--seed", type=int, default=3); ap.add_argument("--noise", type=float, default=1.5)
ap.add_argument("--B", type=int, default=16); ap.add_argument("--iters", type=int, default=150)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
rc = pickle.loads((ROOT / "results/shape" / f"{a.base}.pkl").read_bytes()); MR.prepare(rc)
spec = MC.spec_for(a.norad, ""); C = SG.lvlh_to_fit(a.base, rc)
want = {float(v) for v in a.rates.split(",")}; rows = []
for rate in [0.0, 20.0, 60.0, 120.0]:                       # тот же розыгрыш, что в inject_spin.py / spin_diag.py
    w0 = rng.normal(size=3); w0 /= np.linalg.norm(w0); R0 = w0 * rng.uniform(0.3, 2.5)
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    xt = np.r_[R0, ax * math.radians(rate), rng.normal(0, 3.0, 2)]
    rc2 = copy.deepcopy(rc); S = SP.Spin(rc2, spec, C)
    abright = np.array([0.6, 1.4, 2.2])[:S.P.ng] * (1 + 0.1 * rng.standard_normal(S.P.ng))
    for b in S.bands:
        d = rc2["bands"][b]; Vb, _ = S.pose(xt, S.t[b])
        occ = AP.regress(d, Vb, spec["F"], spec["G"], S.P.ng, S.P.los, np.r_[xt[6], xt[7], 0.0])
        O = np.stack(occ, -1); gm = float(np.nanmedian(d["g"]))
        d["r"] = (O * abright * gm).sum(-1) - d["g"] * O.sum(-1) + a.noise * float(d["s"]) * rng.standard_normal(d["r"].shape)
    if rate not in want:
        continue
    t0 = time.time(); D = DiffSpin(rc2, spec, C, seed=a.seed)
    res = D.solve(B=a.B, iters=a.iters, log=lambda s: print(s, flush=True))
    w = np.asarray(res["x"][3:6]); err_ax = None
    if rate > 0 and np.linalg.norm(w) > 1e-6:
        err_ax = math.degrees(math.acos(min(1.0, abs(float(w @ ax)) / np.linalg.norm(w))))
    print(f"истина {rate:.0f} °/с → градиент: {res['rate_deg_s']:.1f} ± {res['rate_sigma_deg_s']:.1f} °/с, ось ±"
          f"{'—' if err_ax is None else f'{err_ax:.0f}°'}, вращение {'да' if res['spinning'] else 'нет'} (выигрыш {res['gain']:.1f}), "
          f"{time.time() - t0:.0f} с", flush=True)
    rows.append(dict(true_rate=rate, est_rate=res["rate_deg_s"], sigma=res["rate_sigma_deg_s"], axis_err_deg=err_ax, gain=res["gain"],
                     spinning=res["spinning"], family=res["family_rates_deg_s"]))
(ROOT / "results/auto" / f"inject_diffspin_{a.norad}_{a.base}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

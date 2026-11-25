"""Проверка: восстанавливается ли вращение кувыркающейся ступени по одному пролёту Sentinel-2 (s2sat/spin.py).

  python scripts/inject_spin.py STARLINK-30580_20261120T111642 --norad 22220 --rates 0,20,60,120
Объект — модель каталога для NORAD (по умолчанию 22220 — вторая ступень «Зенита», 11 × 3,9 м) с известными R₀ и ω,
отрисованный в настоящих окнах полос снимка (их моменты съёмки и геометрия), «серое тело», шум σ = 1,5·s.
"""
import argparse, copy, json, math, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import autopose as AP, modelcat as MC, scenegeom as SG, spin as SP

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("--norad", type=int, default=22220); ap.add_argument("--rates", default="0,20,60,120")
ap.add_argument("--tag", default=""); ap.add_argument("--seed", type=int, default=3); ap.add_argument("--noise", type=float, default=1.5); ap.add_argument("--maxfev", type=int, default=300)
a = ap.parse_args()


def main():
    rng = np.random.default_rng(a.seed)
    rc = pickle.loads((ROOT / "results/shape" / f"{a.base}.pkl").read_bytes()); MR.prepare(rc)
    spec = MC.spec_for(a.norad, "")
    C = SG.lvlh_to_fit(a.base, rc)
    print(f"объект: {spec['source']} ({spec['kind']}), окна снимка {a.base}", flush=True)
    rows = []
    for rate in [float(v) for v in a.rates.split(",")]:
        t0 = time.time()
        w0 = rng.normal(size=3); w0 /= np.linalg.norm(w0); R0 = w0 * rng.uniform(0.3, 2.5)
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        xt = np.r_[R0, ax * math.radians(rate), rng.normal(0, 3.0, 2)]
        rc2 = copy.deepcopy(rc)
        S = SP.Spin(rc2, spec, C)
        abright = np.array([0.6, 1.4, 2.2])[:S.P.ng] * (1 + 0.1 * rng.standard_normal(S.P.ng))
        for b in S.bands:
            d = rc2["bands"][b]; Vb, _ = S.pose(xt, S.t[b])
            occ = AP.regress(d, Vb, spec["F"], spec["G"], S.P.ng, S.P.los, np.r_[xt[6], xt[7], 0.0])
            O = np.stack(occ, -1); gm = float(np.nanmedian(d["g"]))
            d["r"] = (O * abright * gm).sum(-1) - d["g"] * O.sum(-1) + a.noise * float(d["s"]) * rng.standard_normal(d["r"].shape)
        S = SP.Spin(rc2, spec, C)
        res = S.solve(maxfev=a.maxfev, log=lambda *_: None)
        w_est = np.asarray(res["x"][3:6]); ang_ax = None
        if rate > 0 and np.linalg.norm(w_est) > 1e-6:
            ang_ax = math.degrees(math.acos(min(1.0, abs(float(w_est @ ax) / np.linalg.norm(w_est)))))
        row = dict(true_rate=rate, est_rate=res["rate_deg_s"], sigma=res["rate_sigma_deg_s"], axis_err_deg=ang_ax,
                   sigma_curv=res.get("rate_sigma_curv_deg_s"), sigma_jk=res.get("rate_sigma_jackknife_deg_s"),
                   axis_sigma_jk=res.get("axis_sigma_jackknife_deg"), family=res.get("family_rates_deg_s"), ambiguous=res.get("ambiguous"),
                   spinning=res["spinning"], gain=res["gain"], span_s=res["span_s"], seconds=round(time.time() - t0))
        rows.append(row)
        print(f"истина {rate:5.0f} °/с → найдено {res['rate_deg_s']:6.1f} ± {res['rate_sigma_deg_s']:5.1f} °/с, ось ±"
              f"{'—' if ang_ax is None else f'{ang_ax:.0f}°'}, вращение {'да' if res['spinning'] else 'нет'} "
              f"(выигрыш {res['gain']:.1f}), семейство {res.get('family_rates_deg_s')}, полосы за {res['span_s']:.2f} с, {row['seconds']} с", flush=True)
    (ROOT / "results/auto").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/auto" / f"inject_spin_{a.norad}_{a.base}{a.tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

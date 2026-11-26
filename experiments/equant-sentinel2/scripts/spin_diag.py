"""Почему вращение восстанавливается неверно: поиск или информация? (к inject_spin.py)

  python scripts/spin_diag.py STARLINK-30580_20261120T111642 --rates 20,60 --seed 3
Повторяет подстановку inject_spin.py (тот же генератор случайных чисел), затем сравнивает невязку:
  * в истинных параметрах и после локальной подгонки из истины (дно «истинной» ямы);
  * лучшего решения обычного поиска (Spin.solve).
Если дно истинной ямы ниже найденного — виноват поиск (мало стартов); если не ниже — снимок не различает скорости
(наложение: полосы сняты группами с разрывом в секунды, тело симметрично), и выдавать надо семейство.
"""
import argparse, copy, math, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import autopose as AP, modelcat as MC, scenegeom as SG, spin as SP

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("--norad", type=int, default=22220); ap.add_argument("--rates", default="20,60")
ap.add_argument("--seed", type=int, default=3); ap.add_argument("--noise", type=float, default=1.5); ap.add_argument("--maxfev", type=int, default=300)
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
rc = pickle.loads((ROOT / "results/shape" / f"{a.base}.pkl").read_bytes()); MR.prepare(rc)
spec = MC.spec_for(a.norad, ""); C = SG.lvlh_to_fit(a.base, rc)
want = {float(v) for v in a.rates.split(",")}
for rate in [0.0, 20.0, 60.0, 120.0]:                       # тот же порядок розыгрыша, что в inject_spin.py
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
    S = SP.Spin(rc2, spec, C); t0 = time.time()
    L_true = S.loss(xt)[0]
    xb, L_basin = S.fit(xt, [0, 1, 2, 3, 4, 5, 6, 7], a.maxfev)
    res = S.solve(maxfev=a.maxfev, log=lambda *_: None, jackknife=False)
    L_found = res["L_spin"]
    verdict = "ПОИСК (истинная яма глубже — нужны старты/перебор)" if L_basin < L_found - 1.0 else \
              "ИНФОРМАЦИЯ (снимок не различает — выдавать семейство)"
    print(f"истина {rate:.0f} °/с: невязка в истине {L_true:.1f}, дно истинной ямы {L_basin:.1f} "
          f"(скорость {math.degrees(np.linalg.norm(xb[3:6])):.1f}); найдено {L_found:.1f} (скорость {res['rate_deg_s']:.1f}, "
          f"семейство {res['family_rates_deg_s']}) → {verdict}; {time.time() - t0:.0f} с", flush=True)

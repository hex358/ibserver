"""Вращение объекта по настоящему снимку (s2sat/diffpose.solve_spin) с проверкой согласованности — одинаково для всех.

  python scripts/spin_real.py STARLINK-32851_20261013T055346 [--nongrey] [--shade]
Несоответствие модели снимку (спектр, ядро, опоры полос) «вращение» может изображать: каждая полоса снята в свой
момент, и поворот подстраивается под особенности полосы. Поэтому вращение признаётся, только если:
  1) выигрыш над лучшей статической позой больше порога (BIC: κ·3·ln N);
  2) вращение, найденное НЕЗАВИСИМО по полосам 10 м и по полосам 20 м, одно и то же (скорость ±25 %, ось ±30°).
Пишет results/spin_real/<имя>.json.
"""
import argparse, json, math, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import modelcat as MC, scenegeom as SG
from s2sat.diffpose import DiffPose, DiffSpinPose, solve_spin

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--nongrey", action="store_true"); ap.add_argument("--shade", action="store_true")
ap.add_argument("--n-rot", type=int, default=1500); ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args(); torch.set_num_threads(a.threads); n = a.name; t0_ = time.time()
rc = pickle.loads((ROOT / "results/shape" / f"{n}.pkl").read_bytes()); MR.prepare(rc)
spec = MC.spec_for(int(rc["cat"]), n); C = SG.lvlh_to_fit(n, rc)
Sf, Sc = DiffPose(rc, spec, C, shade=a.shade), DiffPose(rc, spec, C, shade=a.shade, coarse=True)
Wf, Wc = DiffSpinPose(rc, spec, C, shade=a.shade), DiffSpinPose(rc, spec, C, shade=a.shade, coarse=True)
for D in (Sf, Sc, Wf, Wc):
    D.grey = not a.nongrey
G = Wf.groups(); gt = [float(np.median([Wf.sc.t[b] for b in g])) for g in G]; Wf.t0 = Wc.t0 = gt[int(np.argmax([len(g) for g in G]))]
pp = json.loads((ROOT / "results/pose_pix_off" / f"{n}.json").read_text()); kap = float(pp["kappa"])
print(f"{n}: полос {len(Wf.sc.bands)}, разброс времён {np.ptp(list(Wf.sc.t.values())):.2f} с, κ {kap:.2f}, "
      f"яркости {'по полосам' if a.nongrey else 'серые'}{', освещение' if a.shade else ''}", flush=True)


def summ(r):
    w = r["x"][-3:]; rate = math.degrees(np.linalg.norm(w)); ax = w / max(np.linalg.norm(w), 1e-12)
    return rate, ax


out = {}
r = solve_spin(Wf, Wc, Sf, Sc, n_rot=a.n_rot, kap=kap); rate, ax = summ(r)
thr = kap * 3 * math.log(Wf.npix); dL = r["L_static"] - r["L"]
print(f"   все полосы: {rate:.1f} °/с, ось {np.round(ax, 2).tolist()}; выигрыш над статикой {dL:.0f} (порог {thr:.0f}); "
      f"зеркальный двойник хуже на {r['mirror_dL']:.0f}", flush=True)
out["all"] = dict(rate=rate, axis=ax.tolist(), dL=dL, thr=thr, mirror_dL=r["mirror_dL"], p_direction=r["p_direction"])
sub = {"10 м": [b for b in Wf.sc.bands if Wf.sc.B[b].res <= 10], "20 м": [b for b in Wf.sc.bands if Wf.sc.B[b].res > 10]}
res_sub = {}
for k, bs in sub.items():
    if len(bs) >= 2 and np.ptp([Wf.sc.t[b] for b in bs]) >= 0.3:
        rs = solve_spin(Wf, Wc, Sf, Sc, bands=bs, n_rot=a.n_rot, kap=kap); rk, axk = summ(rs)
        res_sub[k] = (rk, axk, rs["L_static"] - rs["L"])
        print(f"   полосы {k} {bs}: {rk:.1f} °/с, ось {np.round(axk, 2).tolist()}, выигрыш {rs['L_static'] - rs['L']:.0f}", flush=True)
        out[k] = dict(bands=bs, rate=rk, axis=axk.tolist(), dL=rs["L_static"] - rs["L"])
cons = None
if len(res_sub) == 2:
    (r1, a1, _), (r2, a2, _) = res_sub.values()
    ang = math.degrees(math.acos(float(np.clip(a1 @ a2, -1, 1)))); rel = abs(r1 - r2) / max(r1, r2, 1e-9)
    cons = bool(ang <= 30.0 and rel <= 0.25)
    print(f"   согласованность 10 м / 20 м: скорости {r1:.1f} / {r2:.1f} °/с (разница {rel * 100:.0f}%), оси расходятся на {ang:.0f}° → "
          f"{'СОГЛАСОВАНЫ' if cons else 'НЕ согласованы'}", flush=True)
    out["consistency"] = dict(axis_angle_deg=ang, rate_rel=rel, consistent=cons)
spinning = dL > thr and (cons is not False)
verdict = ("ВРАЩАЕТСЯ" if spinning else "НЕ вращается") + ("" if cons is not None else " (проверки 10/20 м нет — мало полос)")
print(f"   ИТОГ: {verdict}" + (f": {rate:.1f} °/с, ось {np.round(ax, 2).tolist()}" if spinning else ""), flush=True)
out.update(name=n, spinning=bool(spinning), verdict=verdict, seconds=time.time() - t0_)
(ROOT / "results/spin_real").mkdir(exist_ok=True)
(ROOT / "results/spin_real" / f"{n}{'_nongrey' if a.nongrey else ''}{'_shade' if a.shade else ''}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

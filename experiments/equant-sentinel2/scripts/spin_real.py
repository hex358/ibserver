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
ap.add_argument("--inject", type=float, default=None, help="проверка правила: подставить вращение (°/с) с известной осью")
a = ap.parse_args(); torch.set_num_threads(a.threads); n = a.name; t0_ = time.time()
rc = pickle.loads((ROOT / "results/shape" / f"{n}.pkl").read_bytes()); MR.prepare(rc)
spec = MC.spec_for(int(rc["cat"]), n); C = SG.lvlh_to_fit(n, rc)
Sf, Sc = DiffPose(rc, spec, C, shade=a.shade), DiffPose(rc, spec, C, shade=a.shade, coarse=True)
Wf, Wc = DiffSpinPose(rc, spec, C, shade=a.shade), DiffSpinPose(rc, spec, C, shade=a.shade, coarse=True)
for D in (Sf, Sc, Wf, Wc):
    D.grey = not a.nongrey
G = Wf.groups(); gt = [float(np.median([Wf.sc.t[b] for b in g])) for g in G]; Wf.t0 = Wc.t0 = gt[int(np.argmax([len(g) for g in G]))]
pp = json.loads((ROOT / "results/pose_pix_off" / f"{n}.json").read_text()); kap = float(pp["kappa"])
truth = None
if a.inject is not None:      # подстановка: объект в позе по пикселям, вращение a.inject °/с, яркости по снимку, шум ×√κ
    from s2sat.diffmodel import transfer_r
    Do = DiffPose(rc, spec, C, band_off=25.0); Do.grey = not a.nongrey
    xo = np.asarray(pp["x"], float); xs0 = xo[:Do.o0]; xo_nb = np.r_[xs0, np.zeros(Do.P - Do.o0)]
    _, rho = Do.predict(xo_nb, Do.sc.bands, Do.sc.bands[:1]) if Do.grey else (None, None)
    rng = np.random.default_rng(7); ax_t = rng.normal(size=3); ax_t /= np.linalg.norm(ax_t)
    xt = np.r_[xs0, ax_t * math.radians(a.inject)]
    if Wf.grey:
        Wf.synth(xt, rho, math.sqrt(kap), rng)
    else:                     # яркости по полосам: для каждой полосы свои (НК по настоящему снимку в той же позе)
        for b in Wf.sc.bands:
            bd = Wf.sc.B[b]; occ, cols = Wf.columns(xt, b); A_ = cols.T * bd.gm / bd.s; y_ = (bd.r.numpy() + bd.g.numpy() * occ.sum(0)) / bd.s
            from scipy.optimize import nnls
            rb = nnls(A_, y_)[0]; m_ = cols.T @ rb * bd.gm - bd.g.numpy() * occ.sum(0)
            bd.r = torch.tensor(m_ + math.sqrt(kap) * bd.s * rng.standard_normal(len(m_)), dtype=torch.float32)
    for D in (Sf, Sc, Wc):
        transfer_r(Wf.sc, D.sc)
    truth = (a.inject, ax_t)
    print(f"   ПОДСТАНОВКА: {a.inject:.0f} °/с, ось {np.round(ax_t, 2).tolist()}; сигнал: без объекта {Wf.null():.0f} → истина {float(Wf.L(xt)[0]):.0f}", flush=True)
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
# согласованность: каждая независимая оценка (10 м, 20 м) против оценки по всем полосам; ось — с точностью до
# зеркального двойника нет: направление сравнивается как есть (скорость ±25 %, ось ±30°)
cons, checks = None, []
l_ = Wf.sc.C.numpy().T @ Wf.sc.los_t.numpy(); l_ /= np.linalg.norm(l_); ax_m = -(ax - 2 * (ax @ l_) * l_)   # зеркальная ось
for k, (rk, axk, _) in res_sub.items():
    ang = min(math.degrees(math.acos(float(np.clip(axk @ ax, -1, 1)))), math.degrees(math.acos(float(np.clip(axk @ ax_m, -1, 1)))))
    rel = abs(rk - rate) / max(rk, rate, 1e-9)
    ok = bool(ang <= 30.0 and rel <= 0.25); checks.append(ok)
    print(f"   согласованность {k} с общей: {rk:.1f} / {rate:.1f} °/с (разница {rel * 100:.0f}%), оси расходятся на {ang:.0f}° → "
          f"{'согласована' if ok else 'НЕ согласована'}", flush=True)
    out.setdefault("consistency", {})[k] = dict(axis_angle_deg=ang, rate_rel=rel, consistent=ok)
if checks:
    cons = all(checks)
if dL <= thr:
    spinning, verdict = False, "НЕ вращается (вращение не улучшает снимок)"
elif cons is None:
    spinning, verdict = None, "НЕ ПРОВЕРЕНО: вращение улучшает снимок, но независимых наборов полос для проверки нет"
elif cons:
    spinning, verdict = True, "ВРАЩАЕТСЯ (независимые наборы полос согласованы)"
else:
    spinning, verdict = False, "НЕ вращается: «вращение» подстраивается под полосы (независимые наборы не согласованы)"
print(f"   ИТОГ: {verdict}" + (f": {rate:.1f} °/с, ось {np.round(ax, 2).tolist()}" if spinning else ""), flush=True)
if truth:
    ang_t = math.degrees(math.acos(float(np.clip(ax @ truth[1], -1, 1))))
    print(f"   сравнение с подстановкой: скорость {rate:.1f} / {truth[0]:.0f} °/с, ось (со знаком) {ang_t:.0f}°", flush=True)
    out["inject"] = dict(rate=truth[0], axis=truth[1].tolist(), axis_err_deg=ang_t)
out.update(name=n, spinning=spinning, verdict=verdict, seconds=time.time() - t0_)
(ROOT / "results/spin_real").mkdir(exist_ok=True)
(ROOT / "results/spin_real" / f"{n}{'_inj%d' % a.inject if a.inject is not None else ''}{'_nongrey' if a.nongrey else ''}{'_shade' if a.shade else ''}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

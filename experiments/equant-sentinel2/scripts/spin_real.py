"""Вращение объекта по настоящему снимку (s2sat/diffpose.solve_spin) с проверкой согласованности — одинаково для всех.

  python scripts/spin_real.py STARLINK-32851_20261013T055346 [--nongrey] [--shade]
Несоответствие модели снимку (спектр, ядро, опоры полос) «вращение» может изображать: каждая полоса снята в свой
момент, и поворот подстраивается под особенности полосы. Поэтому вращение признаётся, только если:
  1) выигрыш над лучшей статической позой больше порога (BIC: κ·3·ln N);
  2) вращение предсказывает ОТЛОЖЕННЫЕ полосы лучше статики (каждая полоса по очереди: подгонка по остальным,
     предсказание отложенной; сумма > 2κ на полосу и лучше в ≥ 2/3 полос).
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
l_ = Wf.sc.C.numpy().T @ Wf.sc.los_t.numpy(); l_ /= np.linalg.norm(l_); ax_mir = -(ax - 2 * (ax @ l_) * l_)   # ось зеркального двойника
dir_ok = r["mirror_dL"] > 2 * kap * 3                       # направление различимо (двойник заметно хуже)
print(f"   направление: {'определено' if dir_ok else 'НЕ различимо по снимку — две оси'}; ось {np.round(ax, 2).tolist()}"
      + ("" if dir_ok else f" или {np.round(ax_mir, 2).tolist()}"), flush=True)
out["all"] = dict(rate=rate, axis=ax.tolist(), axis_mirror=ax_mir.tolist(), direction_resolved=bool(dir_ok), dL=dL, thr=thr,
                  mirror_dL=r["mirror_dL"], p_direction=r["p_direction"])
# проверка предсказанием отложенной полосы: по очереди каждая полоса откладывается, вращение и статика дожимаются
# (от найденных решений) по остальным, и обе модели предсказывают отложенную полосу (яркости — по ней самой, свобода
# одинаковая). Настоящее вращение предсказывает форму отложенной полосы лучше статики; «вращение», подогнанное под
# особенности полос, — нет (её оно не видело)
cv = []
for b in sorted(Wf.sc.bands, key=lambda q: Wf.sc.t[q]):
    tr = [c for c in Wf.sc.bands if c != b]
    xs_tr, _ = Sf.polish(r["x_static"], iters=60, bands=tr); xw_tr, _ = Wf.polish(r["x"], iters=60, bands=tr)
    d_ = float(Sf.L(xs_tr, bands=[b], prior=False)[0]) - float(Wf.L(xw_tr, bands=[b], prior=False)[0])
    cv.append((b, d_)); print(f"   отложена {b:9s}: предсказание вращением лучше статики на {d_:+.0f}", flush=True)
dcv = sum(d for _, d in cv); nb = sum(d > 0 for _, d in cv); thr_cv = 2 * kap * len(cv)
cons = bool(dcv > thr_cv and nb >= math.ceil(2 * len(cv) / 3))
print(f"   отложенные полосы: в сумме {dcv:+.0f} (порог {thr_cv:.0f}), лучше в {nb} из {len(cv)} → {'вращение ПОДТВЕРЖДЕНО' if cons else 'вращение НЕ подтверждено'}", flush=True)
out["cv"] = dict(per_band={b: d for b, d in cv}, total=dcv, threshold=thr_cv, n_better=nb, confirmed=cons)
if dL <= thr:
    spinning, verdict = False, "НЕ вращается (вращение не улучшает снимок)"
elif cons:
    spinning, verdict = True, "ВРАЩАЕТСЯ (вращение предсказывает отложенные полосы)"
else:
    spinning, verdict = False, "НЕ вращается: «вращение» подстраивается под полосы и не предсказывает отложенные"
print(f"   ИТОГ: {verdict}" + (f": {rate:.1f} °/с, ось {np.round(ax, 2).tolist()}" if spinning else ""), flush=True)
if truth:
    xtp, Ltp = Wf.polish(xt)
    print(f"   невязка: найдено {r['L']:.0f}, истина (дожатая) {Ltp:.0f} → {'поиск не нашёл истину' if Ltp < r['L'] - 1 else 'снимок предпочитает найденное'}", flush=True)
    ang_t = math.degrees(math.acos(float(np.clip(ax @ truth[1], -1, 1)))) if truth[0] > 0 else float("nan")
    ang_tm = math.degrees(math.acos(float(np.clip(ax_mir @ truth[1], -1, 1)))) if truth[0] > 0 else float("nan")
    print(f"   сравнение с подстановкой: скорость {rate:.1f} / {truth[0]:.0f} °/с, ось (со знаком) {ang_t:.0f}°, "
          f"с учётом зеркального двойника {min(ang_t, ang_tm):.0f}°", flush=True)
    out["inject"] = dict(rate=truth[0], axis=truth[1].tolist(), axis_err_deg=ang_t, axis_err_mod_mirror_deg=min(ang_t, ang_tm))
out.update(name=n, spinning=spinning, verdict=verdict, seconds=time.time() - t0_)
(ROOT / "results/spin_real").mkdir(exist_ok=True)
(ROOT / "results/spin_real" / f"{n}{'_inj%d' % a.inject if a.inject is not None else ''}{'_nongrey' if a.nongrey else ''}{'_shade' if a.shade else ''}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

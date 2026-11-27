"""Вращение объекта по настоящему снимку (s2sat/diffpose.solve_spin) с проверкой согласованности — одинаково для всех.

  python scripts/spin_real.py STARLINK-32851_20261013T055346 [--nongrey] [--shade]
Несоответствие модели снимку (спектр, ядро, опоры полос) «вращение» может изображать: каждая полоса снята в свой
момент, и поворот подстраивается под особенности полосы. Поэтому вращение признаётся, только если:
  1) выигрыш над лучшей статической позой больше порога (BIC: κ·3·ln N);
  2) вращение предсказывает ОТЛОЖЕННЫЕ полосы лучше статики (каждая полоса по очереди: подгонка по остальным,
     предсказание отложенной; сумма > 2κ на полосу и лучше в ≥ 2/3 полос).
Пишет results/spin_real/<имя>.json.
"""
import argparse, json, math, os, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import modelcat as MC, scenegeom as SG
from s2sat.diffpose import DiffPose, DiffSpinPose, DiffSpinCoM, solve_spin

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--nongrey", action="store_true"); ap.add_argument("--shade", action="store_true")
ap.add_argument("--n-rot", type=int, default=1500); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--inject", type=float, default=None, help="проверка правила: подставить вращение (°/с) с известной осью")
ap.add_argument("--mismatch", action="store_true", help="подстановка рисуется ДРУГОЙ моделью отражения (не Ламберт cos^0.7, "
                "разброс отражения по граням ±30 %%, блик шире), решатель — своей (проверка устойчивости направления)")
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
from s2sat.diffmodel import transfer_r
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
        colf = Wf.columns
        if a.mismatch:
            assert a.shade, "--mismatch — только с --shade (сравнение моделей освещения)"
            from s2sat.diffmodel import rotvec_to_mat
            tex = torch.tensor(np.random.default_rng(5).uniform(0.7, 1.3, len(Wf.sc.grp)), dtype=torch.float32)

            def colf(x, b):
                with torch.no_grad():
                    xt_ = torch.tensor(np.atleast_2d(x), dtype=torch.float32); xs_, w_ = xt_[:, :-3], xt_[:, -3:]
                    Rt = rotvec_to_mat(w_ * (Wf.sc.t[b] - Wf.t0)); cw, Sw = Wf.spots(xs_, Rt); dp = Wf.dpos(xs_); sc = Wf.sc; bd = sc.B[b]
                    P = bd.bt + torch.einsum("ij,bmj->bmi", bd.Mt, cw + dp[:, None, :])
                    A = torch.abs(torch.einsum("bmj,j->bm", Sw, bd.qt)); occ = bd.occupancy(P, A, sc.grp)
                    n_ = Sw / torch.linalg.norm(Sw, dim=-1, keepdim=True).clamp_min(1e-9)
                    nv = n_ * torch.sign((n_ * sc.los_t).sum(-1))[..., None]
                    ci = torch.relu((nv * sc.sun_t).sum(-1)) ** 0.7 * tex[None]
                    sp = torch.relu((nv * sc.half_t).sum(-1)) ** 15
                    lit = bd.occupancy(P, A * ci, sc.grp); spo = bd.occupancy(P, A * sp * (sc.grp == 0), sc.grp).sum(1, keepdim=True)
                    return occ[0].numpy(), torch.cat([lit, spo], 1)[0].numpy()
        for b in Wf.sc.bands:
            bd = Wf.sc.B[b]; occ, cols = colf(xt, b); A_ = cols.T * bd.gm / bd.s; y_ = (bd.r.numpy() + bd.g.numpy() * occ.sum(0)) / bd.s
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
# проверка предсказанием отложенного МОМЕНТА съёмки: по очереди откладывается целая подгруппа полос (снятых в пределах
# 0,2 с — у них одна поза), и вращение ищется ЗАНОВО по остальным подгруппам (тот же решатель, без единого параметра,
# подогнанного по отложенной); статика — так же. Обе модели предсказывают отложенную подгруппу (яркости — по ней самой,
# свобода одинаковая). Настоящее вращение предсказывает позу в неувиденный момент; «вращение», подогнанное под
# особенности полос, — нет. Дожим от решения по всем полосам здесь недопустим: быстрая «скорость» держит позу каждой
# полосы в своей яме, и отложенная полоса продолжает «помнить» себя (так 33558 и 33700 получали ложное «вращается»).
# Вращение добавляет 3 параметра — ровно одну лишнюю позу; при < 3 моментах съёмки его не отличить от несовпадения
# полос, и проверка невозможна → вращение по этому пролёту не определяется.
CL = Wf.clusters(0.2); ct = [float(np.median([Wf.sc.t[b] for b in c])) for c in CL]
cv, cache = [], {}
identifiable = len(CL) >= 3
if dL > thr and identifiable:
    for c in CL:
        tr = [b for b in Wf.sc.bands if b not in c]
        rr = solve_spin(Wf, Wc, Sf, Sc, bands=tr, n_rot=a.n_rot, kap=kap, cache=cache, log=lambda s: None)
        d_ = float(Sf.L(rr["x_static"], bands=c, prior=False)[0]) - float(Wf.L(rr["x"], bands=c, prior=False)[0])
        cv.append(("+".join(c), d_, math.degrees(np.linalg.norm(rr["x"][-3:]))))
        print(f"   отложено {'+'.join(c):24s}: вращение по остальным {cv[-1][2]:6.1f} °/с, предсказание лучше статики на {d_:+.0f}", flush=True)
dcv = sum(d for _, d, _ in cv); nb = sum(d > 0 for _, d, _ in cv); thr_cv = 2 * kap * max(1, len(cv))
cons = bool(cv) and bool(dcv > thr_cv and nb >= math.ceil(2 * len(cv) / 3))
if cv:
    print(f"   отложенные моменты: в сумме {dcv:+.0f} (порог {thr_cv:.0f}), лучше в {nb} из {len(cv)} → {'вращение ПОДТВЕРЖДЕНО' if cons else 'вращение НЕ подтверждено'}", flush=True)
out["cv"] = dict(per_cluster={k: d for k, d, _ in cv}, rate_per_fold={k: r_ for k, _, r_ in cv}, total=dcv, threshold=thr_cv,
                 n_better=nb, confirmed=cons, n_clusters=len(CL))
if not identifiable:
    spinning, verdict = False, f"вращение по пролёту не определяется: {len(CL)} момент(а) съёмки (нужно ≥ 3)"
elif dL <= thr:
    spinning, verdict = False, "НЕ вращается (вращение не улучшает снимок)"
elif cons:
    spinning, verdict = True, "ВРАЩАЕТСЯ (вращение предсказывает отложенные моменты съёмки)"
else:
    spinning, verdict = False, "НЕ вращается: «вращение» подстраивается под полосы и не предсказывает отложенные моменты"
print(f"   ИТОГ: {verdict}" + (f": {rate:.1f} °/с, ось {np.round(ax, 2).tolist()}" if spinning else ""), flush=True)
if spinning:   # центр масс по вращению: прямая в теле (ось вращения через центр масс); вдоль оси не измеряется
    Wm = DiffSpinCoM(rc, spec, C, shade=a.shade); Wm.grey = not a.nongrey; Wm.t0 = Wf.t0; transfer_r(Wf.sc, Wm.sc)
    xc, Lc, L0c, sig, E, _ = Wm.fit_com(r["x"], kap=kap); nm = Wm.axis_model(xc); c = xc[-6:-3]; cp = c - (c @ nm) * nm
    print(f"   центр масс: на оси вращения {np.round(nm, 2).tolist()} (оси модели) через точку {np.round(cp, 2).tolist()} м, "
          f"σ поперёк оси {np.round(sig, 2).tolist()} м; выигрыш {L0c - Lc:.0f}", flush=True)
    out["com"] = dict(axis_model=nm.tolist(), point_perp_m=cp.tolist(), sigma_perp_m=sig.tolist(), dL=L0c - Lc,
                      note="центр масс лежит на оси вращения через эту точку; положение вдоль оси по вращению не измеряется")
if truth:
    xtp, Ltp = Wf.polish(xt)
    print(f"   невязка: найдено {r['L']:.0f}, истина (дожатая) {Ltp:.0f} → {'поиск не нашёл истину' if Ltp < r['L'] - 1 else 'снимок предпочитает найденное'}", flush=True)
    ang_t = math.degrees(math.acos(float(np.clip(ax @ truth[1], -1, 1)))) if truth[0] > 0 else float("nan")
    ang_tm = math.degrees(math.acos(float(np.clip(ax_mir @ truth[1], -1, 1)))) if truth[0] > 0 else float("nan")
    print(f"   сравнение с подстановкой: скорость {rate:.1f} / {truth[0]:.0f} °/с, ось (со знаком) {ang_t:.0f}°, "
          f"с учётом зеркального двойника {min(ang_t, ang_tm):.0f}°", flush=True)
    out["inject"] = dict(rate=truth[0], axis=truth[1].tolist(), axis_err_deg=ang_t, axis_err_mod_mirror_deg=min(ang_t, ang_tm))
out.update(name=n, spinning=spinning, verdict=verdict, identifiable=identifiable, seconds=time.time() - t0_)
(ROOT / "results/spin_real").mkdir(exist_ok=True)
(ROOT / "results/spin_real" / f"{n}{'_inj%d' % a.inject if a.inject is not None else ''}{'_nongrey' if a.nongrey else ''}{'_shade' if a.shade else ''}{'_mm' if a.mismatch else ''}{os.environ.get('SPIN_TAG', '')}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

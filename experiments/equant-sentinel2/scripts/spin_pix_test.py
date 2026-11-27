"""Вращение по пикселям (s2sat/diffpose.DiffSpinPose) на подстановке: настоящая геометрия пролёта, модель в позе,
найденной по пикселям (results/pose_pix), вращение с известной скоростью, шум снимка × √κ — восстановление.

  python scripts/spin_pix_test.py STARLINK-30245_20261120T110322 --rates 20,60,120 [--shade]
Без перебора скоростей: позы групп полос → кандидаты ω из пар минимумов (ось и угол поворота между группами + 2πk) →
градиент по всем параметрам на всех полосах (s2sat/diffpose.py).
"""
import argparse, json, math, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import autopose as AP, modelcat as MC, scenegeom as SG
from s2sat.diffmodel import transfer_r
from s2sat.diffpose import DiffPose, DiffSpinPose, anchor_to_geometry, rot_angle

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--rates", default="20,60,120"); ap.add_argument("--shade", action="store_true")
ap.add_argument("--real", action="store_true", help="настоящий снимок (без подстановки): вращается ли объект и как")
ap.add_argument("--nongrey", action="store_true", help="яркости материалов свои в каждой полосе")
ap.add_argument("--seed", type=int, default=5); ap.add_argument("--n-rot", type=int, default=1500); ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args(); torch.set_num_threads(a.threads); rng = np.random.default_rng(a.seed); n = a.name


def log(*s):
    print(*s, flush=True)


rc = pickle.loads((ROOT / "results/shape" / f"{n}.pkl").read_bytes())
if a.real:      # настоящий снимок: опоры как в позе по пикселям (геометрия + подогнанные сдвиги полос, results/shape)
    log(f"{n}: НАСТОЯЩИЙ снимок; опоры — геометрия + сдвиги полос из позы по пикселям")
else:
    log(f"{n}: опоры — {anchor_to_geometry(rc, ROOT / 'results/orbits' / f'{n}.json')}")
MR.prepare(rc); spec = MC.spec_for(int(rc["cat"]), n); C = SG.lvlh_to_fit(n, rc)
Sf, Sc = DiffPose(rc, spec, C, shade=a.shade), DiffPose(rc, spec, C, shade=a.shade, coarse=True)
Wf, Wc = DiffSpinPose(rc, spec, C, shade=a.shade), DiffSpinPose(rc, spec, C, shade=a.shade, coarse=True)
for D_ in (Sf, Sc, Wf, Wc):
    D_.grey = not a.nongrey
G = Wf.groups(); gt = [float(np.median([Wf.sc.t[b] for b in g])) for g in G]
big = int(np.argmax([len(g) for g in G])); Wf.t0 = Wc.t0 = gt[big]
log(f"   подгруппы ≤ 0,2 с: {[c for c in Wf.clusters(0.2)]}"); log(f"   группы полос: " + "; ".join(f"{'+'.join(g)} (t {t:+.2f} с, разброс {np.ptp([Wf.sc.t[b] for b in g]):.2f} с)" for g, t in zip(G, gt)))
# яркость подставного объекта — как у НАСТОЯЩЕГО: из подгонки с подогнанными опорами полос (results/pose_pix_off);
# без сдвигов опор модель почти не объясняет объект, и подставной выходил в ~20 раз слабее настоящего
pp = json.loads((ROOT / "results/pose_pix_off" / f"{n}.json").read_text()); xo = np.asarray(pp["x"], float); kap = float(pp["kappa"])
Do = DiffPose(rc, spec, C, band_off=25.0, shade=a.shade); _, rho = Do.predict(xo, Do.sc.bands, Do.sc.bands[:1])
if a.shade:      # с освещением — физические классы материалов (в долях яркости фона): ячейки, светлое, тёмное, блик стекла;
    # подгонка блика по настоящему снимку при позе без освещения неустойчива (давала коэффициент ~10⁵)
    rho = np.r_[np.array([0.05, 0.6, 0.3])[:Do.sc.ng], 2.0]
log(f"   яркости подставного ({'освещение Солнцем + блик' if a.shade else 'силуэт'}): {np.round(rho, 3).tolist()}")
xs0 = xo[:Do.o0]
print(f"   объект на снимке: невязка без объекта {pp['null']:.0f} → в позе {pp['L']:.0f} (сигнал {pp['null'] - pp['L']:.0f})", flush=True)
rows = []
for rate in ([None] if a.real else [float(v) for v in a.rates.split(",")]):
    t_ = time.time()
    if a.real:
        w = np.zeros(3); xt = None
    else:
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax); w = ax * math.radians(rate)
        xt = np.r_[xs0, w]                                              # поза в t₀ — найденная по снимку
        orig = Wf.synth(xt, rho, math.sqrt(kap), rng)
        log(f"   сигнал подставного: без объекта {Wf.null():.0f} → истина {float(Wf.L(xt)[0]):.0f} (разница {Wf.null() - float(Wf.L(xt)[0]):.0f})")
        for D in (Sf, Sc, Wc):
            transfer_r(Wf.sc, D.sc)
    CL = Wf.clusters(0.2); ct = [float(np.median([Wf.sc.t[b] for b in c])) for c in CL]
    gm = []
    for c in CL:                                                    # 1. поза каждой подгруппы (≤ 0,2 с)
        md, _ = Sc.search(bands=c, n_rot=a.n_rot, top=128, fine=Sf, n_polish=8, log=lambda s: None)
        gm.append([m["x"] for m in md])
    ib = int(np.argmin([abs(t - Wf.t0) for t in ct]))
    for c_, tc_, ms in zip(CL, ct, gm):
        if a.real:
            continue
        Rtc = AP.rotvec(w * (tc_ - Wf.t0)) @ Wf.R_of(xt)
        log(f"   подгруппа {'+'.join(c_)} (t {tc_ - Wf.t0:+.2f} с): минимумов {len(ms)}, ближайший к истинной позе {min(rot_angle(Wf.R_of(m), Rtc) for m in ms):.0f}°"
            f" (лучший по невязке {rot_angle(Wf.R_of(ms[0]), Rtc):.0f}°)")
    cand = Wf.candidates(gm, ct, k_modes=3) + [Wf.static_x(m) for m in gm[ib][:4]]   # 2. кандидаты ω (+ «не вращается»)
    C_ = np.array(cand); dw = np.degrees(np.linalg.norm(C_[:, -3:] - w, axis=1))
    if not a.real:
        dr = np.array([rot_angle(Wf.R_of(c), Wf.R_of(xt)) for c in C_]); kk = int(np.argmin(dw + dr / 3))
        log(f"   диагностика: ближайший к истине кандидат — ω ошибка {dw[kk]:.1f} °/с, поза в t₀ {dr[kk]:.0f}°; "
            f"по ω лучший {dw.min():.1f} °/с, по позе лучший {dr.min():.0f}°")
    xc, Lc = Wc.fit(np.array(cand), iters=120)                     # 3. градиент по всем параметрам, все полосы
    dwf = np.degrees(np.linalg.norm(xc[:, -3:] - w, axis=1)); kf = int(np.argmin(dwf))
    if not a.real:
        log(f"   после спуска: ближайший по ω {dwf[kf]:.1f} °/с, его невязка {Lc[kf]:.0f}, лучшая {Lc.min():.0f}")
    o = np.argsort(Lc)[:12]
    xf, Lf = Wf.fit(xc[o], iters=60, lr=0.01)                       # дожим на точной модели (Adam), затем L-BFGS
    res = sorted([Wf.polish(xf[k]) for k in np.argsort(Lf)[:4]], key=lambda q: q[1])
    xe, Le = res[0]
    # 4. профиль правдоподобия по ω: поза в t₀, крылья, сдвиг — из лучшего; ω — сетка по величине (×0,5…1,6) и оси
    #    (конус до 20°); невязка пакетом → лучшие точки → градиент. Дальняя база (≈2 с) даёт узкие ямы по ω (~5 °/с),
    #    ближняя — широкое априорное окно; сетка закрывает его целиком.
    w0 = xe[-3:]; r0 = float(np.linalg.norm(w0)); n0 = w0 / max(r0, 1e-9)
    # ось — по всей сфере (Фибоначчи, 200 направлений + найденная), величина ×0,80…1,25: дальняя база фиксирует |ω|
    # точнее, чем направление; ограничение конусом 20° держало ось на краю конуса
    kf = np.arange(200) + 0.5; zf = 1 - 2 * kf / 200; phf = math.pi * (1 + 5 ** 0.5) * kf
    dirs = np.c_[np.sqrt(1 - zf ** 2) * np.cos(phf), np.sqrt(1 - zf ** 2) * np.sin(phf), zf]; dirs = np.vstack([n0, dirs])
    W = [d_ * r0 * m for d_ in dirs for m in np.arange(0.80, 1.26, 0.03)]
    # позы в t₀: найденная и её двойники тела (поворот на 180° вокруг каждой оси тела при той же ω: у почти
    # симметричного тела силуэт тот же, а истина могла оказаться именно двойником), с углами крыльев как есть и +180°;
    # плюс статические минимумы ближайшей к t₀ подгруппы
    from scipy.spatial.transform import Rotation as Ro
    Rh = Wf.R_of(xe); dr_ = xe[3:3 + Wf.nd]; rest_ = xe[3 + Wf.nd:-3]
    bases = []
    for k_ in (None, 0, 1, 2):
        Rk = Rh if k_ is None else Rh @ AP.rot_axis(np.eye(3)[k_], math.pi)
        for dd_ in (dr_, dr_ + math.pi):
            bases.append((Rk, np.r_[dd_, rest_], False))
    bases += [(Wf.R_of(m), m[3:], True) for m in gm[ib][:4]]
    dtc = ct[ib] - Wf.t0; Xw = []
    for Rb_, rest, shift in bases:
        for w in W:
            R0 = AP.rotvec(-np.asarray(w) * dtc) @ Rb_ if shift else Rb_
            Xw.append(np.r_[Ro.from_matrix(R0).as_rotvec(), rest, w])
    Xw = np.array(Xw); Lw = np.concatenate([Wc.L(Xw[i:i + 256]) for i in range(0, len(Xw), 256)])
    # профиль по направлению оси: для каждого направления — лучшая точка сетки (по позам и величинам), затем градиент
    # по всем параметрам при этом направлении (грубая модель) → лучшие 12 → точная модель
    nb_, nm_ = len(bases), len(np.arange(0.80, 1.26, 0.03)); Lr = Lw.reshape(nb_, len(dirs), nm_)
    per_dir = [np.unravel_index(np.argmin(Lr[:, d_, :]), (nb_, nm_)) for d_ in range(len(dirs))]
    Xd = np.array([Xw[(bi * len(dirs) + d_) * nm_ + mi] for d_, (bi, mi) in enumerate(per_dir)])
    xd, Ld = Wc.fit(Xd, iters=40, lr=0.02)
    ow = np.argsort(Ld)[:12]
    xw, Lw2 = Wf.fit(xd[ow], iters=60, lr=0.01)
    res2 = sorted([Wf.polish(xw[k]) for k in np.argsort(Lw2)[:3]] + [(xe, Le)], key=lambda q: q[1])
    xe, Le = res2[0]
    # 5. зеркальный двойник (направление): дожать все три варианта отражения, сравнить с лучшим
    tw = sorted([Wf.polish(t) for t in Wf.mirror_twins(xe)], key=lambda q: q[1]); xm, Lm = tw[0]
    if Lm < Le:
        (xe, Le), (xm, Lm) = (xm, Lm), (xe, Le)
    ang_m = math.degrees(math.acos(float(np.clip(xe[-3:] @ xm[-3:] / (np.linalg.norm(xe[-3:]) * np.linalg.norm(xm[-3:]) + 1e-12), -1, 1))))
    p_dir = 1.0 / (1.0 + math.exp(-0.5 * (Lm - Le) / kap))
    log(f"   направление: зеркальный двойник хуже на {Lm - Le:.0f} (ось отличается на {ang_m:.0f}°) → вероятность найденного направления {p_dir:.3f}")
    if a.real:
        # не вращается ли: лучшая статическая поза (ω = 0, все остальные параметры свободны) против найденного вращения;
        # 3 лишних параметра ω — порог по BIC: ΔL > κ·3·ln N
        xs_best = min([Sf.polish(xe[:-3])] + [Sf.polish(m) for m in gm[ib][:4]], key=lambda q: q[1])
        Ls = xs_best[1]; thr_ = kap * 3 * math.log(Wf.npix); spinning = (Ls - Le) > thr_
        re = math.degrees(np.linalg.norm(xe[-3:])); nax = xe[-3:] / max(np.linalg.norm(xe[-3:]), 1e-12)
        r_ = dict(real=True, rate_found=re, axis_lvlh=nax.tolist(), L_spin=Le, L_static=Ls, dL=Ls - Le, threshold=thr_, spinning=bool(spinning),
                  mirror_dL=Lm - Le, mirror_axis_deg=ang_m, p_direction=p_dir, seconds=time.time() - t_)
        rows.append(r_)
        log(f"   НАСТОЯЩИЙ: лучшее вращение {re:.1f} °/с (ось в орбитальных осях {np.round(nax, 2).tolist()}), невязка {Le:.0f}; "
            f"без вращения {Ls:.0f}; выигрыш {Ls - Le:.0f} при пороге {thr_:.0f} → {'ВРАЩАЕТСЯ' if spinning else 'НЕ вращается (вращение не нужно снимку)'}")
        continue
    xt_p, Lt_p = Wf.polish(xt)
    re = math.degrees(np.linalg.norm(xe[-3:])); fam = [round(math.degrees(np.linalg.norm(x[-3:])), 1) for x, L in res2 if L - Le < 2 * kap * 3]
    axe = math.degrees(math.acos(float(np.clip(float(xe[-3:] @ w) / (np.linalg.norm(xe[-3:]) * np.linalg.norm(w) + 1e-12), -1, 1)))) if rate > 0 else float("nan")   # со знаком: обратное вращение — 180°
    r_ = dict(rate_true=rate, rate_found=re, axis_err_deg=axe, mirror_dL=Lm - Le, mirror_axis_deg=ang_m, p_direction=p_dir, family=fam, L_found=Le, L_truth=Lt_p, n_cand=len(cand),
              pose_err_deg=rot_angle(Wf.R_of(xe), Wf.R_of(xt)), seconds=time.time() - t_)
    rows.append(r_)
    log(f"   истина {rate:.0f} °/с → найдено {re:.1f} °/с, ось (со знаком) {axe:.0f}°, поза в t₀ ±{r_['pose_err_deg']:.0f}°; семейство {fam}; "
        f"невязка найдено {Le:.0f} / истина {Lt_p:.0f}; кандидатов {len(cand)}; {r_['seconds']:.0f} с")
    Wf.restore(orig)
    for D in (Sf, Sc, Wc):
        transfer_r(Wf.sc, D.sc)
out = ROOT / "results/spin_pix"; out.mkdir(exist_ok=True)
(out / f"{n}{'_real' if a.real else ''}{'_nongrey' if a.nongrey else ''}{'_shade' if a.shade else ''}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

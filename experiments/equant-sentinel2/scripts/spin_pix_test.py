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
from s2sat.diffpose import DiffPose, DiffSpinPose, anchor_to_geometry, rot_angle, solve_spin

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
    r = solve_spin(Wf, Wc, Sf, Sc, n_rot=a.n_rot, kap=kap)           # тот же решатель, что для настоящих снимков
    xe, Le = r["x"], r["L"]; Lm, ang_m, p_dir = r["L"] + r["mirror_dL"], r["mirror_axis_deg"], r["p_direction"]
    res2 = [(xe, Le)]; gm = [[r["x_static"]]]; ib = 0; cand = []
    log(f"   направление: зеркальный двойник хуже на {Lm - Le:.0f} (ось отличается на {ang_m:.0f}°) → вероятность найденного направления {p_dir:.3f}")
    if a.real:
        # не вращается ли: лучшая статическая поза (ω = 0, все остальные параметры свободны) против найденного вращения;
        # 3 лишних параметра ω — порог по BIC: ΔL > κ·3·ln N
        Ls = r["L_static"]; thr_ = kap * 3 * math.log(Wf.npix); spinning = (Ls - Le) > thr_
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

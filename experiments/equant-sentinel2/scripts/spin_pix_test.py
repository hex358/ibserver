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
from s2sat.diffpose import DiffPose, DiffSpinPose, anchor_to_geometry, rot_angle

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--rates", default="20,60,120"); ap.add_argument("--shade", action="store_true")
ap.add_argument("--seed", type=int, default=5); ap.add_argument("--n-rot", type=int, default=1500); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args(); torch.set_num_threads(a.threads); rng = np.random.default_rng(a.seed); n = a.name


def log(*s):
    print(*s, flush=True)


rc = pickle.loads((ROOT / "results/shape" / f"{n}.pkl").read_bytes())
log(f"{n}: опоры — {anchor_to_geometry(rc, ROOT / 'results/orbits' / f'{n}.json')}")
MR.prepare(rc); spec = MC.spec_for(int(rc["cat"]), n); C = SG.lvlh_to_fit(n, rc)
Sf, Sc = DiffPose(rc, spec, C, shade=a.shade), DiffPose(rc, spec, C, shade=a.shade, lmax=3.0)
Wf, Wc = DiffSpinPose(rc, spec, C, shade=a.shade), DiffSpinPose(rc, spec, C, shade=a.shade, lmax=3.0)
G = Wf.groups(); gt = [float(np.median([Wf.sc.t[b] for b in g])) for g in G]
big = int(np.argmax([len(g) for g in G])); Wf.t0 = Wc.t0 = gt[big]
log(f"   группы полос: " + "; ".join(f"{'+'.join(g)} (t {t:+.2f} с, разброс {np.ptp([Wf.sc.t[b] for b in g]):.2f} с)" for g, t in zip(G, gt)))
pp = json.loads((ROOT / "results/pose_pix" / f"{n}.json").read_text())
xs0 = np.asarray(pp["x"], float); kap = float(pp["kappa"])
_, rho = Sf.predict(xs0, Sf.sc.bands, Sf.sc.bands[:1])
rows = []
for rate in [float(v) for v in a.rates.split(",")]:
    t_ = time.time()
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax); w = ax * math.radians(rate)
    xt = np.r_[xs0, w]                                              # поза в t₀ — найденная по снимку
    orig = Wf.synth(xt, rho, math.sqrt(kap), rng)
    for D in (Sf, Sc, Wc):
        for b in D.sc.bands:
            D.sc.B[b].r = Wf.sc.B[b].r.clone()
    gm = []
    for g in G:                                                     # 1. поза каждой группы
        md, _ = Sc.search(bands=g, n_rot=a.n_rot, top=64, fine=Sf, n_polish=6, log=lambda s: None)
        gm.append([m["x"] for m in md])
    cand = Wf.candidates(gm, gt) + [Wf.static_x(m) for m in gm[big][:4]]     # 2. кандидаты ω (+ «не вращается»)
    xc, Lc = Wc.fit(np.array(cand), iters=120)                     # 3. градиент по всем параметрам, все полосы
    o = np.argsort(Lc)[:6]; res = sorted([Wf.polish(xc[k]) for k in o], key=lambda q: q[1])
    xe, Le = res[0]; xt_p, Lt_p = Wf.polish(xt)
    re = math.degrees(np.linalg.norm(xe[-3:])); fam = [round(math.degrees(np.linalg.norm(x[-3:])), 1) for x, L in res if L - Le < 2 * kap * 3]
    axe = math.degrees(math.acos(min(1.0, abs(float(xe[-3:] @ w) / (np.linalg.norm(xe[-3:]) * np.linalg.norm(w) + 1e-12))))) if rate > 0 else float("nan")
    r_ = dict(rate_true=rate, rate_found=re, axis_err_deg=axe, family=fam, L_found=Le, L_truth=Lt_p, n_cand=len(cand),
              pose_err_deg=rot_angle(Wf.R_of(xe), Wf.R_of(xt)), seconds=time.time() - t_)
    rows.append(r_)
    log(f"   истина {rate:.0f} °/с → найдено {re:.1f} °/с, ось ±{axe:.0f}°, поза в t₀ ±{r_['pose_err_deg']:.0f}°; семейство {fam}; "
        f"невязка найдено {Le:.0f} / истина {Lt_p:.0f}; кандидатов {len(cand)}; {r_['seconds']:.0f} с")
    Wf.restore(orig)
    for D in (Sf, Sc, Wc):
        for b in D.sc.bands:
            D.sc.B[b].r = Wf.sc.B[b].r.clone()
out = ROOT / "results/spin_pix"; out.mkdir(exist_ok=True)
(out / f"{n}{'_shade' if a.shade else ''}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

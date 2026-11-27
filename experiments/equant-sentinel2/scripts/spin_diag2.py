"""Почему вращение находится неверно: на подстановке (как spin_pix_test.py, та же затравка) для каждой подгруппы полос
(≤ 0,2 с) — невязка в ИСТИННОЙ позе (повёрнутой на момент подгруппы, дожатой по сдвигу/приводам) против невязки
лучшего найденного минимума. Истина хуже найденного — подгруппа позу не определяет (информация); истина лучше —
ошибается поиск.

  python scripts/spin_diag2.py STARLINK-30245_20261120T110322 --rate 60
"""
import argparse, json, math, pickle, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import autopose as AP, modelcat as MC, scenegeom as SG
from s2sat.diffmodel import transfer_r
from s2sat.diffpose import DiffPose, DiffSpinPose, anchor_to_geometry, rot_angle

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--rate", type=float, default=60.0); ap.add_argument("--seed", type=int, default=5)
a = ap.parse_args(); torch.set_num_threads(2); rng = np.random.default_rng(a.seed); n = a.name
rc = pickle.loads((ROOT / "results/shape" / f"{n}.pkl").read_bytes()); anchor_to_geometry(rc, ROOT / "results/orbits" / f"{n}.json")
MR.prepare(rc); spec = MC.spec_for(int(rc["cat"]), n); C = SG.lvlh_to_fit(n, rc)
Sf, Sc = DiffPose(rc, spec, C), DiffPose(rc, spec, C, coarse=True); Wf = DiffSpinPose(rc, spec, C)
G = Wf.groups(); gt = [float(np.median([Wf.sc.t[b] for b in g])) for g in G]; Wf.t0 = gt[int(np.argmax([len(g) for g in G]))]
# яркость подставного объекта — как у НАСТОЯЩЕГО: из подгонки с подогнанными опорами полос (results/pose_pix_off);
# без сдвигов опор модель почти не объясняет объект, и подставной выходил в ~20 раз слабее настоящего
pp = json.loads((ROOT / "results/pose_pix_off" / f"{n}.json").read_text()); xo = np.asarray(pp["x"], float); kap = float(pp["kappa"])
Do = DiffPose(rc, spec, C, band_off=25.0); _, rho = Do.predict(xo, Do.sc.bands, Do.sc.bands[:1])
xs0 = xo[:Do.o0]
print(f"   объект на снимке: невязка без объекта {pp['null']:.0f} → в позе {pp['L']:.0f} (сигнал {pp['null'] - pp['L']:.0f})", flush=True)
ax = rng.normal(size=3); ax /= np.linalg.norm(ax); w = ax * math.radians(a.rate)   # как первая скорость в spin_pix_test
xt = np.r_[xs0, w]; Wf.synth(xt, rho, math.sqrt(kap), rng); transfer_r(Wf.sc, Sf.sc); transfer_r(Wf.sc, Sc.sc)
print(f"{n}: подстановка {a.rate:.0f} °/с, ось {np.round(ax, 2).tolist()}, шум ×√κ = {math.sqrt(kap):.2f}; "
      f"сигнал подставного: без объекта {Wf.null():.0f} → истина {float(Wf.L(xt)[0]):.0f} (разница {Wf.null() - float(Wf.L(xt)[0]):.0f})")
from scipy.spatial.transform import Rotation as Ro
for c in Wf.clusters(0.2):
    tc = float(np.median([Wf.sc.t[b] for b in c])); Rtc = AP.rotvec(w * (tc - Wf.t0)) @ Wf.R_of(xt)
    xtrue = np.r_[Ro.from_matrix(Rtc).as_rotvec(), xs0[3:]]
    Lt0 = float(Sf.L(xtrue, bands=c)[0]); xtp, Ltp = Sf.polish(xtrue, bands=c)
    md, _ = Sc.search(bands=c, n_rot=1500, top=128, fine=Sf, n_polish=8, log=lambda s: None)
    Lb = md[0]["L"]; near = min(rot_angle(Sf.R_of(m["x"]), Rtc) for m in md)
    dpol = rot_angle(Sf.R_of(xtp), Rtc)
    print(f"  {'+'.join(c):28s} t {tc - Wf.t0:+.2f} с: без объекта {Sf.null(c):6.0f} | истина {Lt0:6.0f}, дожатая {Ltp:6.0f} "
          f"(ушла на {dpol:.0f}°) | лучший найденный {Lb:6.0f} ({rot_angle(Sf.R_of(md[0]['x']), Rtc):.0f}° от истины), "
          f"ближайший из {len(md)} минимумов {near:.0f}° → {'ИНФОРМАЦИЯ: снимок подгруппы не различает' if Lb <= Ltp + 1 else 'ПОИСК: истина лучше найденного'}", flush=True)

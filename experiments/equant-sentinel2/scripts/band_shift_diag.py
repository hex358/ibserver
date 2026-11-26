"""Проверка опор полос: в найденной позе (results/pose_pix/<имя>.json) для каждой полосы — сдвиг модели на снимке
(±48 м, шаг 4 м), при котором невязка полосы минимальна (яркость групп — своя для полосы). Большие и устойчивые сдвиги
означают, что опора полосы (момент/положение) неточна и поза по всем полосам сразу будет вписываться плохо.

  python scripts/band_shift_diag.py STARLINK-30245_20261120T110322 [--shade]
"""
import argparse, json, pickle, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import modelcat as MC, scenegeom as SG
from s2sat.diffpose import DiffPose, anchor_to_geometry

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--shade", action="store_true"); ap.add_argument("--half", type=float, default=48.0)
a = ap.parse_args(); n = a.name
rc = pickle.loads((ROOT / "results/shape" / f"{n}.pkl").read_bytes())
meas = {b: (np.asarray(d["meas"][:2], float) if d.get("meas") else None) for b, d in rc["bands"].items()}
print(f"{n}: опоры — {anchor_to_geometry(rc, ROOT / 'results/orbits' / f'{n}.json')}")
MR.prepare(rc); spec = MC.spec_for(int(rc["cat"]), n); C = SG.lvlh_to_fit(n, rc)
pp = json.loads((ROOT / ("results/pose_pix_shade" if a.shade else "results/pose_pix") / f"{n}.json").read_text())
x = np.asarray(pp["x"], float)
D = DiffPose(rc, spec, C, shade=a.shade)
g = np.arange(-a.half, a.half + 1e-6, 4.0); rows = {}
for b in D.sc.bands:
    bd = D.sc.B[b]; b0 = bd.bt.clone(); best = (None, 0, 0)
    L0 = float(D.L(x, bands=[b], prior=False)[0]); Ln = D.null([b])
    for dx in g:
        for dy in g:
            bd.bt = b0 + torch.tensor([dx, dy], dtype=torch.float32)
            L = float(D.L(x, bands=[b], prior=False)[0])
            if best[0] is None or L < best[0]:
                best = (L, dx, dy)
    bd.bt = b0
    mo = (meas[b] - b0.numpy()) if meas[b] is not None else None
    rows[b] = dict(L0=L0, L_best=best[0], null=Ln, shift=[best[1], best[2]], peak_minus_geom=None if mo is None else mo.tolist())
    print(f"  {b:9s} без объекта {Ln:7.0f} | в позе {L0:7.0f} | со сдвигом {best[0]:7.0f} при ({best[1]:+.0f}, {best[2]:+.0f}) м"
          + ("" if mo is None else f" | пик яркости − геометрия ({mo[0]:+.0f}, {mo[1]:+.0f}) м"), flush=True)
(ROOT / "results/pose_pix" / f"{n}_band_shifts{'_shade' if a.shade else ''}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

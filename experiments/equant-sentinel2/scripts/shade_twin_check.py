"""Различает ли ОСВЕЩЕНИЕ позу и её зеркального двойника на НАСТОЯЩИХ снимках — проверка с физической истиной.

Без освещения снимок не различает позу R и двойника M·R·D_k (M — отражение относительно плоскости ⊥ визиру): это та же
неоднозначность, что у направления вращения. Истина для работающих Starlink — их крылья развёрнуты на Солнце (|n·s| ≈ 1),
а у двойника нормаль крыла n′ = M·n с Солнцем расходится на десятки градусов (|n·M s|). Значит, у пары «поза /
двойник» по крыльям видно, какая настоящая, а освещение (модель s2sat.diffmodel shade) должно выбрать ту же.
  python scripts/shade_twin_check.py [имена…]   (по умолчанию — все results/pose_pix_off/*.json)
"""
import json, math, pickle, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import modelcat as MC, scenegeom as SG
from s2sat.diffpose import DiffPose
from scipy.spatial.transform import Rotation as Ro

torch.set_num_threads(1)
names = sys.argv[1:] or sorted(p.stem for p in (ROOT / "results/pose_pix_off").glob("*.json"))
rows = []
for n in names:
    rc = pickle.loads((ROOT / "results/shape" / f"{n}.pkl").read_bytes()); MR.prepare(rc)
    spec = MC.spec_for(int(rc["cat"]), n); C = SG.lvlh_to_fit(n, rc)
    pp = json.loads((ROOT / "results/pose_pix_off" / f"{n}.json").read_text()); x = np.asarray(pp["x"], float)
    D0, D1 = DiffPose(rc, spec, C, band_off=25.0), DiffPose(rc, spec, C, band_off=25.0, shade=True)
    D0.grey = D1.grey = False
    sun_l = C.T @ np.asarray(rc["sun"], float) if False else None
    s_ = D0.sc.C.numpy().T @ D0.sc.sun_t.numpy(); s_ /= np.linalg.norm(s_)          # Солнце в орбитальных осях
    l_ = D0.sc.C.numpy().T @ D0.sc.los_t.numpy(); l_ /= np.linalg.norm(l_); M = np.eye(3) - 2 * np.outer(l_, l_)
    cands = [("поза", x)]
    for k in range(3):
        Dk = np.eye(3); Dk[k, k] = -1.0
        xt = x.copy(); xt[:3] = Ro.from_matrix(M @ D0.R_of(x) @ Dk).as_rotvec(); xt[3:3 + D0.nd] = -x[3:3 + D0.nd]
        cands.append((f"двойник D{k}", xt))
    out = []
    for lab, xc in cands:
        ang = D0.drive_report(xc, s_)
        x0p, L0 = D0.polish(xc, iters=100); x1p, L1 = D1.polish(xc, iters=100)
        out.append((lab, float(np.mean(ang)), L0, L1))
    print(f"\n{n}: Солнце·визир {float(s_ @ l_):+.2f}", flush=True)
    for lab, a_, L0, L1 in out:
        print(f"   {lab:12s}: крылья от Солнца {a_:5.1f}°  невязка без освещения {L0:8.0f}  с освещением {L1:8.0f}", flush=True)
    i_true = int(np.argmin([o[1] for o in out])); i_sh = int(np.argmin([o[3] for o in out])); i_ns = int(np.argmin([o[2] for o in out]))
    rows.append(dict(name=n, cands=[dict(label=o[0], wing_sun_deg=o[1], L_noshade=o[2], L_shade=o[3]) for o in out],
                     true_by_wings=out[i_true][0], pick_shade=out[i_sh][0], pick_noshade=out[i_ns][0],
                     shade_margin=sorted(o[3] for o in out)[1] - out[i_sh][3]))
    print(f"   по крыльям настоящая: {out[i_true][0]} ({out[i_true][1]:.0f}°); освещение выбирает: {out[i_sh][0]} "
          f"(запас {rows[-1]['shade_margin']:.0f}); без освещения: {out[i_ns][0]}", flush=True)
(ROOT / "results/shade_twin.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

"""Записать позу по пикселям (results/pose_pix*/<имя>.json) для листов и рендеров: results/auto/<имя>.json (прежний —
в results/auto_laws/) и опоры полос в results/shape/<имя>.pkl = геометрия по точной орбите + подогнанные сдвиги опор
(прежний — в results/shape_peaks/).

  python scripts/pose_write.py STARLINK-30245_20261120T110322 --src results/pose_pix_off
"""
import argparse, copy, json, pickle, shutil, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import modelcat as MC
from s2sat.diffpose import anchor_to_geometry

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--src", default="results/pose_pix_off")
a = ap.parse_args(); n = a.name
pp = json.loads((ROOT / a.src / f"{n}.json").read_text())
shp = ROOT / "results/shape" / f"{n}.pkl"; peaks = ROOT / "results/shape_peaks" / f"{n}.pkl"
(ROOT / "results/shape_peaks").mkdir(exist_ok=True); (ROOT / "results/auto_laws").mkdir(exist_ok=True)
if not peaks.exists():
    shutil.copy(shp, peaks)
rc = pickle.loads(peaks.read_bytes())
anchor_to_geometry(rc, ROOT / "results/orbits" / f"{n}.json")
off = pp.get("band_offsets_m") or {}
for b, d in rc["bands"].items():
    d["p0"] = np.asarray(d["p0"], float) + np.asarray(off.get(b, [0.0, 0.0]), float)
shp.write_bytes(pickle.dumps(rc))
spec = MC.spec_for(int(rc["cat"]), n); nd = len(pp["drives_deg"])
x = np.asarray(pp["x"], float); o0 = 3 + nd + 2
xa = np.r_[x[:3], 0.0, x[o0 - 2], x[o0 - 1], 0.0, x[3:3 + nd]]
fa = ROOT / "results/auto" / f"{n}.json"; src = ROOT / "results/auto_laws" / f"{n}.json"
if fa.exists() and not src.exists():
    shutil.copy(fa, src)
sd = pp["sigma"]["rot_deg"]; nn = pp["nearest_nadir"]
desc = f"по пикселям: до «на Землю» {nn['angle_deg']:.0f}° (рысканье {nn['yaw_deg']:.0f}°), σ поворота {', '.join(f'{v:.1f}' for v in sd)}°"
fa.write_text(json.dumps(dict(kind=spec.get("kind"), source=spec.get("source"), name=n, law=dict(name="поза по пикселям", pix=True),
                              x=xa.tolist(), chosen=desc, anomalous=False, margin_eff=float("nan"), hypotheses=[],
                              C_lvlh_to_fit=pp["C_lvlh_to_fit"], sun_lvlh=pp["sun_lvlh"],
                              R=(np.asarray(pp["R_body_to_lvlh"]) @ np.asarray(spec["M2B"], float)).tolist(),
                              array_angles_deg=pp["drives_deg"], pose_pix=str(ROOT / a.src / f"{n}.json")), ensure_ascii=False, indent=1))
print(f"{n}: записано {fa} и опоры {shp} (сдвиги опор: {', '.join(f'{b} {np.hypot(*v):.0f} м' for b, v in off.items()) or 'нет'})")

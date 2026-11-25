"""Пересчитать решение «штатно / нештатно» для готовых результатов (results/auto/*.json) без повторной подгонки:
по сохранённым невязкам и параметрам гипотез (autopose.decide)."""
import json, pickle, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from s2sat import autopose as AP, modelcat as MC

for f in sorted((ROOT / "results/auto").glob("*.json")):
    if f.name.startswith("inject_") or not (ROOT / "results/shape" / f"{f.stem}.pkl").exists():
        continue
    out = json.loads(f.read_text())
    if not out.get("free"):
        continue
    rc = pickle.loads((ROOT / "results/shape" / f"{f.stem}.pkl").read_bytes())
    spec = MC.spec_for(rc.get("cat", 0), f.stem)
    byname = {h["name"]: h for h in spec["laws"]}
    res = [dict(r, law=byname[r["name"]]) for r in out["hypotheses"] if r["name"] in byname]
    free = dict(out["free"], law=dict(name="свободный поворот (оценка)", free3=True))
    sun_l = np.asarray(out["sun_lvlh"], float)
    out["chosen"] = min(res, key=lambda r: r["cv"])["name"]
    AP.decide(out, spec, sun_l, res, free, out["eq_rms_m"])
    ch = free if out["anomalous"] else next(r for r in res if r["name"] == out["chosen"])
    Vb, R, angs = AP.pose_body(spec, ch["law"], np.asarray(ch["x"]), sun_l)
    out["R"] = R.tolist(); out["x"] = ch["x"]; out["law"] = ch["law"]; out["array_angles_deg"] = [float(np.degrees(a)) for a in angs]
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    d = out.get("deviation_deg")
    print(f"{f.stem:34s} {out['chosen'][:60]:60s} {'НЕШТАТНО' if out['anomalous'] else ''} "
          f"выигрыш своб. {out['free_gain_eff']:7.1f} (порог {out['T_anom']:.1f}) ближайший закон {out['nearest_law'][:40]} "
          f"{'' if d is None else f'отклонение {d:.0f}°'}", flush=True)

"""Перерисовать кадры событий по кэшу ячеек с учётом растяжения узора полос (без пересчёта снимков).

  python scripts/replot.py results/events_v9 [--bg loglocal]

Точки полос: корень полосы + сдвиг + σ·(α_b − α_ref)·t (t — направление видимой трассы по корням полос).
Пишет <событие>.png рядом с JSON и добавляет в JSON pos_best и per_band (значение карты Z/U в точке).
"""
import argparse, glob, json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import imagery as I, viz
from s2sat.pipeline import CellRaster, CELL

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--bg", default="loglocal")
a = ap.parse_args()
for d in a.dirs:
    for f in sorted(Path(d).glob("*.json")):
        x = json.loads(f.read_text())
        if not isinstance(x, dict) or "event" not in x:
            continue
        e, r = x["event"], x["result"]
        if "roots_all" not in r or "n" not in r:
            continue
        roots = r["roots_all"][r["n"]]
        if r.get("pos_best"):
            pos = r["pos_best"]
        else:
            bands = list(roots); A = np.array([roots[b][0][1] for b in bands]); P = np.array([roots[b][0][2:4] for b in bands], float)
            c = np.polyfit(A, P, 1)[0] if len(A) >= 2 else np.array([1.0, 0.0]); t = c / (np.linalg.norm(c) + 1e-12)
            sig = (r.get("stretch") or 0.0) * r["kmdeg"] * 1000.0; aref = A.min()
            pos = {b: [P[i, 0] + r["sx"] + sig * (A[i] - aref) * t[0], P[i, 1] + r["sy"] + sig * (A[i] - aref) * t[1]] for i, b in enumerate(bands)}
        key = f"{e['obj'].split()[0]}_{e['sent'][-2:]}_{round(e['jd'] * 86400)}_glrt_z*{a.bg}*"
        ds = sorted(glob.glob(str(I.CACHE / "tiles" / key)))
        if not ds:
            print("no cache", f.stem); continue
        need = {(int((px + dx) // CELL), int((py + dy) // CELL)) for px, py in pos.values() for dx in (-600, 0, 600) for dy in (-600, 0, 600)}
        tiles = {}
        for p in Path(ds[-1]).glob("*.pkl"):
            ci, cj = map(int, p.stem.split("_"))
            if (ci, cj) in need:
                tiles[(ci, cj)] = pickle.loads(p.read_bytes())
        if not tiles:
            print("no cells", f.stem); continue
        per = {}
        for b, (px, py) in pos.items():
            T = [tt for (ci, cj), tt in tiles.items() if (ci, cj) == (int(px // CELL), int(py // CELL))]
            key_ = "Z" if T and b in T[0].get("Z", {}) else "U"
            try:
                per[b] = float(np.nan_to_num(CellRaster(tiles, [b], key_).get(b, np.array([px]), np.array([py]))[0]))
            except Exception:
                per[b] = float("nan")
        rr = dict(r, roots={b: roots[b] for b in roots}, pos_best=pos, per_band=per)
        viz.plot_event(rr, {"tiles": tiles}, str(f.with_suffix(".png")))
        r["pos_best"] = pos; r["per_band"] = per; f.write_text(json.dumps(x, default=float))
        print(f.stem, {b: round(v, 1) for b, v in per.items()}, flush=True)

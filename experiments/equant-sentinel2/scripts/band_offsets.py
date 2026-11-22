"""Смещения объекта по полосам относительно предсказанной картины (по кэшу обработанных ячеек, без пересчёта).

  python scripts/band_offsets.py results/events_uni/STARLINK-33558_20261120T144023.json [--r 200]

В лучшей гипотезе (фаза dM, растяжение, сдвиг) для каждой полосы ищется экстремум |U| в радиусе r вокруг
предсказанной точки; смещение раскладывается вдоль/поперёк видимой трассы и сопоставляется с моментом съёмки
полосы (α_b − α_ref). Линейная зависимость = одна величина (скорость сдвига картины), а не допуск на полосу.
"""
import argparse, glob, json, math, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import imagery as I
from s2sat.pipeline import CellRaster

ap = argparse.ArgumentParser()
ap.add_argument("json")
ap.add_argument("--r", type=float, default=200.0)
a = ap.parse_args()
x = json.loads(Path(a.json).read_text()); e, r = x["event"], x["result"]
n = r["n"]; roots = r["roots_all"][n]
key = f"{e['obj'].split()[0]}_{e['sent'][-2:]}_{round(e['jd'] * 86400)}_glrt_z*"
dirs = sorted(glob.glob(str(I.CACHE / "tiles" / key)))
if not dirs:
    sys.exit(f"no tile cache for {key}")
tiles = {}
for p in Path(dirs[-1]).glob("*.pkl"):
    ci, cj = map(int, p.stem.split("_"))
    tiles[(ci, cj)] = pickle.loads(p.read_bytes())
bands = [b for b in roots if all(b in T["U"] for T in tiles.values())]
U = CellRaster(tiles, bands, "U")
A = np.array([roots[b][0][1] for b in bands]); P = np.array([roots[b][0][2:4] for b in bands])
c = np.polyfit(A, P, 1)[0]; tdir = c / np.linalg.norm(c); ndir = np.array([-tdir[1], tdir[0]])
aref = A.min(); sig = (r.get("stretch") or 0.0) * r["kmdeg"] * 1000.0
g = np.arange(-a.r, a.r + 1, 10.0); GX, GY = np.meshgrid(g, g)
print(f"{Path(a.json).stem}: dM {r['dM']:+.2f} stretch {r.get('stretch')} shift ({r['sx']:.0f},{r['sy']:.0f})  score {r['score']:.1f}")
rows = []
for b, al, (x0, y0) in zip(bands, A, P):
    px = x0 + sig * (al - aref) * tdir[0] + r["sx"]; py = y0 + sig * (al - aref) * tdir[1] + r["sy"]
    v = U.get(b, px + GX, py + GY).astype(float)
    v0 = float(np.nan_to_num(U.get(b, np.array([px]), np.array([py]))[0]))
    k = np.nanargmax(np.abs(np.nan_to_num(v)))
    dx, dy = GX.flat[k], GY.flat[k]
    rows.append((b, al - aref, float(np.dot([dx, dy], tdir)), float(np.dot([dx, dy], ndir)), float(np.nan_to_num(v.flat[k])), v0))
for b, da, al_, cr_, vk, v0 in sorted(rows, key=lambda t: t[1]):
    print(f"  {b:9s} Δα {da:7.4f}°  peak {vk:6.1f}σ (at point {v0:5.1f}σ)  offset along {al_:6.0f} m  across {cr_:6.0f} m")
strong = [t for t in rows if abs(t[4]) >= 4]
if len(strong) >= 3:
    da = np.array([t[1] for t in strong]); oa = np.array([t[2] for t in strong]); oc = np.array([t[3] for t in strong])
    ka, ba = np.polyfit(da, oa, 1); kc, bc = np.polyfit(da, oc, 1)
    ra = oa - (ka * da + ba); rc = oc - (kc * da + bc)
    print(f"  linear fit on {len(strong)} strong bands: along {ka:.0f} m/° (+{ba:.0f}), rms {np.sqrt(np.mean(ra**2)):.0f} m;"
          f" across {kc:.0f} m/° (+{bc:.0f}), rms {np.sqrt(np.mean(rc**2)):.0f} m")

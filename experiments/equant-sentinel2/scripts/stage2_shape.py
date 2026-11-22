"""Вторая ступень: совпадение формы изображения между полосами (по кэшу ячеек, без пересчёта снимков).

  python scripts/stage2_shape.py results/events_v9 [--half 5]

Настоящий объект в каждой полосе даёт одно и то же пятно (та же PSF, то же смазывание движением);
текстура (рельеф, дюны) даёт в разных полосах пятна разной формы и ориентации. Для кандидата берутся
участки карты Z (отклик фильтра / шум, 10-м полосы) вокруг точек полос, знак выравнивается по центру,
C — средняя попарная корреляция Пирсона. Считается для лучшего кандидата и для лучших кандидатов нуля
(параметры нуля сохраняются в JSON с v9: null_best, null_disp) — распределения сравниваются по данным.
"""
import argparse, glob, json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import imagery as I
from s2sat.pipeline import CellRaster
from s2sat.threshold import threshold

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--half", type=int, default=5)
ap.add_argument("--align", type=int, default=5)
a = ap.parse_args()
ALIGN = a.align


def load_tiles(e):
    key = f"{e['obj'].split()[0]}_{e['sent'][-2:]}_{round(e['jd'] * 86400)}_glrt_z*loglocal*"
    ds = sorted(glob.glob(str(I.CACHE / "tiles" / key)))
    if not ds:
        return None
    T = {}
    for p in Path(ds[-1]).glob("*.pkl"):
        ci, cj = map(int, p.stem.split("_")); T[(ci, cj)] = {"Z": pickle.loads(p.read_bytes())["Z"]}   # только Z (память)
    return T


def coherence(MZ, zb, roots, kmdeg, stretch, sx, sy, disp=None, half=5):
    bs = [b for b in zb if b in roots]
    if len(bs) < 2:
        return np.nan, 0
    A = np.array([roots[b][0][1] for b in roots]); Pp = np.array([roots[b][0][2:4] for b in roots], float)
    c = np.polyfit(A, Pp, 1)[0] if len(A) >= 2 else np.array([1.0, 0.0]); tdir = c / (np.linalg.norm(c) + 1e-12)
    aref = A.min(); sig = (stretch or 0.0) * kmdeg * 1000.0
    g = np.arange(-half, half + 1) * 10.0; GX, GY = np.meshgrid(g, -g)
    patches = []
    for b in bs:
        x0, y0 = roots[b][0][2], roots[b][0][3]; al = roots[b][0][1]
        dx, dy = (disp or {}).get(b, (0.0, 0.0))
        px = x0 + sig * (al - aref) * tdir[0] + sx + dx; py = y0 + sig * (al - aref) * tdir[1] + sy + dy
        # выравнивание по пику |Z| в радиусе ALIGN пикселей (реальные смещения полос 20–50 м)
        gs = np.arange(-ALIGN, ALIGN + 1) * 10.0; SX_, SY_ = np.meshgrid(gs, -gs)
        zc = np.abs(np.nan_to_num(MZ.get(b, px + SX_, py + SY_).astype(float)))
        k = int(np.argmax(zc)); px, py = px + SX_.flat[k], py + SY_.flat[k]
        z = MZ.get(b, px + GX, py + GY).astype(float)
        if not np.isfinite(z).all():
            continue
        z = z * np.sign(z[half, half] or 1.0)
        z = z - z.mean(); n = np.linalg.norm(z)
        if n > 0:
            patches.append(z / n)
    if len(patches) < 2:
        return np.nan, len(patches)
    M = np.array([p.ravel() for p in patches]); Cm = M @ M.T
    iu = np.triu_indices(len(patches), 1)
    return float(Cm[iu].mean()), len(patches)


rows = []
for d in a.dirs:
    for f in sorted(Path(d).glob("*.json")):
        x = json.loads(f.read_text())
        if not isinstance(x, dict) or "event" not in x:
            continue
        e, r = x["event"], x["result"]
        if "roots_all" not in r or "n" not in r:
            continue
        T = load_tiles(e)
        if not T:
            continue
        zb = [b for b in ("blue", "green", "red", "nir") if all(b in t.get("Z", {}) for t in T.values())]
        if len(zb) < 2:
            continue
        MZ = CellRaster(T, zb, "Z")
        C, nb = coherence(MZ, zb, r["roots_all"][r["n"]], r["kmdeg"], r.get("stretch"), r["sx"], r["sy"], half=a.half)
        Cn = []
        for nb_, dsp in zip(r.get("null_best") or [], r.get("null_disp") or []):
            if nb_ and nb_.get("n") is not None:
                cc, _ = coherence(MZ, zb, r["roots_all"][int(nb_["n"])], r["kmdeg"], nb_.get("stretch"), nb_["sx"], nb_["sy"], disp=dsp, half=a.half)
                Cn.append(cc)
        m = r["score"] - threshold(r)
        rows.append((f.stem[:30], round(m, 1), round(C, 2), nb, [round(c, 2) for c in Cn]))
        print(f"{f.stem[:30]:30s} margin {m:5.1f}  C_real {C:5.2f} ({nb} bands)  C_null {[round(c, 2) for c in Cn]}", flush=True)
det = [r_[2] for r_ in rows if r_[1] >= 0 and np.isfinite(r_[2])]
nul = [c for r_ in rows for c in r_[4] if np.isfinite(c)]
if det:
    print(f"\nC of detected (real best): median {np.median(det):.2f}  min {min(det):.2f}  n {len(det)}")
if nul:
    print(f"C of null best candidates: median {np.median(nul):.2f}  90% {np.quantile(nul, 0.9):.2f}  max {max(nul):.2f}  n {len(nul)}")

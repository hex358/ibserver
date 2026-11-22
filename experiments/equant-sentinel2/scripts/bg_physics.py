"""Модели фона по физике (THEORY §13): сигнал объекта и шум остатка для разных моделей на одном снимке.

  python scripts/bg_physics.py results/events_uni/STARLINK-33558_20261120T144023.json

A  — текущая: линейный прогноз полосы по другим полосам, блок 500 м / окно 1 км (сложение).
B  — логарифмы яркости (тень рельефа = общее слагаемое), тот же блок 500 м.
C  — логарифмы + локально: блок 100 м / окно 300 м.
Полосы, где объект в момент съёмки рядом (< 300 м), из прогноза исключаются (как в конвейере).
Сигнал — максимум |отклика фильтра| в 30 м от предсказанной точки; шум — MAD отклика по окну без 150 м вокруг объектов.
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import imagery as I, bgsub

x = json.loads(Path(sys.argv[1]).read_text()); e, r = x["event"], x["result"]
roots = r["roots_all"][r["n"]]
bands = [b for b in roots]
A = np.array([roots[b][0][1] for b in bands]); P = np.array([roots[b][0][2:4] for b in bands], float)
c = np.polyfit(A, P, 1)[0]; tdir = c / np.linalg.norm(c); aref = A.min()
sig = (r.get("stretch") or 0.0) * r["kmdeg"] * 1000.0
pos = {b: (P[i, 0] + sig * (A[i] - aref) * tdir[0] + r["sx"], P[i, 1] + sig * (A[i] - aref) * tdir[1] + r["sy"]) for i, b in enumerate(bands)}
xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
bb = (min(xs) - 1500, min(ys) - 1500, max(xs) + 1500, max(ys) + 1500)
it = next(iter(I.client().search(collections=[I.COLL], ids=[r["item"]]).items()))
IM = {b: I.read_window(it.assets[b].href, r["epsg"], bb, res=10).astype(np.float32) for b in bands}
H, W = next(iter(IM.values())).shape
valid = np.all(np.stack([IM[b] > 0 for b in bands]), 0)
col = lambda X: int((X - bb[0]) / 10); row = lambda Y: int((bb[3] - Y) / 10)
mask_obj = np.zeros((H, W), bool)
for (X, Y) in pos.values():
    i, j = row(Y), col(X); mask_obj[max(0, i - 15):i + 16, max(0, j - 15):j + 16] = True
kmdeg = r["kmdeg"]; Arep = dict(zip(bands, A))


def resid(model):
    out = {}
    for b in bands:
        near = {c2 for c2 in bands if abs(Arep[c2] - Arep[b]) * kmdeg < 0.3}
        regs = [c2 for c2 in bands if c2 not in near]
        if model == "A":
            T, R_ = IM[b], [IM[c2] for c2 in regs]; blk, win = 50, 100
        else:
            T, R_ = np.log(np.maximum(IM[b], 1)), [np.log(np.maximum(IM[c2], 1)) for c2 in regs]
            blk, win = (50, 100) if model == "B" else (10, 30)
        out[b] = bgsub.predict_band(T, R_, valid, block=blk, win=win)
    return out


print(f"{Path(sys.argv[1]).stem}: window {W}x{H} px, bands {bands}")
res = {}
for m in ("A", "B", "C"):
    R = resid(m); res[m] = {}
    for b in bands:
        Y, _S = bgsub.matched(R[b])
        Y = np.where(valid, Y, np.nan)
        bg = Y[valid & ~mask_obj]; s = 1.4826 * np.nanmedian(np.abs(bg - np.nanmedian(bg))) + 1e-12
        i, j = row(pos[b][1]), col(pos[b][0])
        w = Y[max(0, i - 3):i + 4, max(0, j - 3):j + 4]
        amp = float(np.nanmax(np.abs(w - np.nanmedian(bg)))) if np.isfinite(w).any() else float("nan")
        # хвост фона: доля пикселей фона выше сигнала объекта (сколько «двойников» у объекта в этом окне)
        tail = float(np.mean(np.abs(bg - np.nanmedian(bg)) >= amp)) if np.isfinite(amp) else float("nan")
        res[m][b] = (amp / s, tail)
print(f"{'band':9s} " + "  ".join(f"{m}: SNR  twins/km²" for m in ("A", "B", "C")))
for b in bands:
    print(f"{b:9s} " + "  ".join(f"{res[m][b][0]:6.1f} {res[m][b][1] * 1e4:8.2f}" for m in ("A", "B", "C")))
for m in ("A", "B", "C"):
    v = np.array([res[m][b][0] for b in bands]); print(f"{m}: sum SNR² over bands = {np.nansum(v ** 2):.0f}  (√ = {math.sqrt(np.nansum(v ** 2)):.1f})")

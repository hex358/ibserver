"""Прямой поиск близкого объекта по сырым пикселям каждой полосы в квадрате ±W км вокруг расчётного положения
(THEORY §22). Для объектов в десятках км под Sentinel-2 ошибка орбиты поперёк трассы и по радиусу переходит на снимок
с множителем λ/R ~ 10–25 и вдоль своих (перекошенных) направлений — окно должно быть квадратным и широким, а
допуск по положению — нулевым (иначе шум размазывается и поднимает порог).

  python scripts/close_search.py results/events_rb/SL-8_20261120T030857.json 20261124T0047 --half 25
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import label, maximum_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from s2sat import passgeom as PG, imagery as I, bgsub
from s2sat.orbits import dt_from_jd
import physdet as PD

ap = argparse.ArgumentParser()
ap.add_argument("event"); ap.add_argument("until"); ap.add_argument("--half", type=float, default=25.0)
ap.add_argument("--bands", default="blue,green,red"); ap.add_argument("--res", type=float, default=10.0)
ap.add_argument("--out", default=str(ROOT / "results/close_search"))
a = ap.parse_args()
x = json.loads(Path(a.event).read_text()); e, r = x["event"], x["result"]
orbs = PG.Orbits(until=a.until); P = PG.Pass(e, r, orbs); P.orb = PG.pick_orbit(orbs, e)
P0 = P.positions([0.0, 0.0, 0.0]); pos = {b: v[:2] for b, v in P0.items()}
bands = [b for b in a.bands.split(",") if b in pos]
its = I.items_around(e["lon"], e["lat"], dt_from_jd(e["jd"]), e["sent"][-1], d=1.0)
its.sort(key=lambda it: it.id != r.get("item"))
k = e["lam"] / e["rng"]
# шаблон: тёмное пятно размера проекции ступени (поперёк ×λ/R) — гауссиана σ ≈ половина диаметра проекции
sig_m = max(10.0, 2.4 * k / 2.0)
g = np.arange(-int(3 * sig_m / a.res), int(3 * sig_m / a.res) + 1) * a.res
Xk, Yk = np.meshgrid(g, g); tmpl = np.exp(-0.5 * (Xk ** 2 + Yk ** 2) / sig_m ** 2)
print(f"{Path(a.event).stem}: дальность {e['rng']:.0f} км, λ/R {k:.1f}, шаблон σ {sig_m:.0f} м, окно ±{a.half} км, полосы {bands}", flush=True)
out = {}
Path(a.out).mkdir(parents=True, exist_ok=True)
for b in bands:
    t0 = time.time()
    px, py = pos[b]; H = a.half * 1000.0
    bb = (math.floor((px - H) / 20) * 20, math.floor((py - H) / 20) * 20, math.ceil((px + H) / 20) * 20, math.ceil((py + H) / 20) * 20)
    far = [c for c in ("blue", "green", "red", "nir", "rededge1", "swir16") if c != b and math.hypot(pos[c][0] - px, pos[c][1] - py) > 300.0][:4]
    IM = PD.read_band_region(its, P.epsg, bb, [b] + far, a.res)
    far = [c for c in far if c in IM]
    R = PD.residual(IM, b, far, a.res)
    ok = np.isfinite(R)
    Y = fftconvolve(np.where(ok, R, 0.0), tmpl[::-1, ::-1], mode="same").astype(np.float32); Y[~ok] = np.nan
    m, sd, shp = bgsub.block_median_mad(Y, 50); Z = (Y - bgsub._unblocks(m, shp, 50)) / bgsub._unblocks(sd + 1e-6, shp, 50)
    Z = np.nan_to_num(Z.astype(np.float32))
    # пики: локальные минимумы (тёмное) и максимумы (блик), разнесённые ≥ 500 м
    nb = int(500 / a.res)
    peaks = []
    for sgn in (-1, 1):
        S = sgn * Z; loc = (S == maximum_filter(S, size=nb)) & (S > 6)
        for i, j in zip(*np.where(loc)):
            peaks.append((float(sgn * S[i, j]), float(bb[0] + (j + 0.5) * a.res - px), float(bb[3] - (i + 0.5) * a.res - py)))
    peaks.sort(key=lambda t: -abs(t[0]))
    out[b] = dict(peaks=peaks[:15], n6=len(peaks), sigma_tail=float(np.percentile(np.abs(Z[ok]), 99.99)) if ok.any() else None)
    print(f"  {b}: {time.time() - t0:.0f} с, пиков |z|>6: {len(peaks)}; 99.99% |z| = {out[b]['sigma_tail']:.1f}; сильнейшие: " +
          "; ".join(f"{p[0]:+.1f}σ в ({p[1] / 1000:+.1f}, {p[2] / 1000:+.1f}) км" for p in peaks[:6]), flush=True)
    np.save(Path(a.out) / f"{Path(a.event).stem}_{b}_z.npy", Z[::2, ::2].astype(np.float16))
# согласованность: пик в каждой полосе со сдвигом от своего расчётного положения, близким к сдвигу в других полосах
cand = []
B = list(out)
if len(B) >= 2:
    for p0 in out[B[0]]["peaks"]:
        grp = [p0]; ok_ = True
        for b in B[1:]:
            best = min(out[b]["peaks"], key=lambda q: math.hypot(q[1] - p0[1], q[2] - p0[2]), default=None)
            if best is None or math.hypot(best[1] - p0[1], best[2] - p0[2]) > 2000 or np.sign(best[0]) != np.sign(p0[0]):
                ok_ = False; break
            grp.append(best)
        if ok_:
            cand.append((sum(abs(q[0]) for q in grp), [(round(q[0], 1), round(q[1]), round(q[2])) for q in grp]))
cand.sort(key=lambda t: -t[0])
print("согласованные между полосами (сдвиг от расчёта ≤ 2 км друг от друга):", cand[:5], flush=True)
json.dump(dict(event=Path(a.event).stem, bands=out, consistent=cand[:10]), open(Path(a.out) / f"{Path(a.event).stem}.json", "w"), indent=1)

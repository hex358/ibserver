"""Шаг 1 физической модели (THEORY §17): объясняет ли ошибка орбиты по трём осям положения объекта в полосах
без подобранных «растяжения» и двумерного сдвига.

  python scripts/fit3d.py results/events_v11/STARLINK-30245_20261120T110322.json [...]

Измерено: центр пятна в каждой полосе (кэш ячеек, около найденной гипотезы). Модель A (нынешняя): фаза dM +
сдвиг (sx, sy) + растяжение — точки pos_best. Модель B (физическая): θ = (вдоль, поперёк, по радиусу), положения —
точные уравнения съёмки (apparent_track + band_roots). Сравнение — СКО невязки по полосам.
"""
import glob, json, math, pickle, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import orbits as O, geometry as G, imagery as I, s2meta as M
from s2sat.pipeline import CellRaster, CELL
from s2sat.orbits import dt_from_jd

UNTIL = "20261122T0319"
snaps = sorted(p for p in (ROOT / "data/tle_snapshots").iterdir() if p.is_dir() and p.name <= UNTIL)


def load_all(pat):
    allo = {}
    for sd in snaps:
        for f in sd.glob(pat):
            for o in O.parse_tle(f.read_text()):
                allo.setdefault(o.catnr, {})[round(o.epoch_jd, 6)] = o
    return {k: list(v.values()) for k, v in allo.items()}


SG, GP, S2 = load_all("supgp_*.tle"), load_all("gp_*.tle"), {}
for sd in snaps:
    for o in O.parse_tle((sd / "sentinel2.tle").read_text()):
        if o.name not in S2 or o.epoch_jd > S2[o.name].epoch_jd:
            S2[o.name] = o


def orbit_for(e):
    sg = [o for o in SG.get(e["cat"], []) if o.epoch_jd <= e["jd"] and e["jd"] - o.epoch_jd <= 2.0]
    if sg:
        return max(sg, key=lambda o: o.epoch_jd)
    gp = [o for o in GP.get(e["cat"], []) if o.epoch_jd > e["jd"] and o.epoch_jd - e["jd"] <= 2.0]
    return min(gp, key=lambda o: o.epoch_jd) if gp else None


def measure(e, r, radius=250.0):
    pos = r["pos_best"]
    key = f"{e['obj'].split()[0]}_{e['sent'][-2:]}_{round(e['jd'] * 86400)}_glrt_z*"
    ds = sorted(glob.glob(str(I.CACHE / "tiles" / key)))
    if not ds:
        return None
    need = {(int((px + dx) // CELL), int((py + dy) // CELL)) for px, py in pos.values() for dx in (-500, 0, 500) for dy in (-500, 0, 500)}
    tiles = {}
    for p in Path(ds[-1]).glob("*.pkl"):
        ci, cj = map(int, p.stem.split("_"))
        if (ci, cj) in need:
            tiles[(ci, cj)] = {"U": pickle.loads(p.read_bytes())["U"]}
    g = np.arange(-radius, radius + 1, 10.0); GX, GY = np.meshgrid(g, g); out = {}
    for b, (px, py) in pos.items():
        try:
            v = np.abs(np.nan_to_num(CellRaster(tiles, [b], "U").get(b, px + GX, py + GY).astype(float)))
        except Exception:
            continue
        k = int(np.argmax(v)); pk = v.flat[k]
        if pk < 5:
            continue
        lab, _ = label(v >= 0.7 * pk); m = lab == lab.flat[k]; w = v * m
        out[b] = (px + float((GX * w).sum() / w.sum()), py + float((GY * w).sum() / w.sum()))
    return out


def roots_fn(e, r):
    sen = S2[e["sent"]]; orb = orbit_for(e); epsg = r["epsg"]
    its = I.items_around(e["lon"], e["lat"], dt_from_jd(e["jd"]), e["sent"][-1], d=1.0)
    psi = G.track_azimuth(sen, e["jd"]); H = float(np.linalg.norm(sen.ecef([e["jd"]])[0][0])) - G.R_LOC
    DG = {}
    for it in its:
        try:
            gg = M.parse_grids(I.get_text(it.assets["granule_metadata"].href), psi, H, it.id)
        except Exception:
            continue
        for b, lst in gg.items():
            for dd in lst:
                dd["epsg"] = I.epsg_of(it)
            DG.setdefault(b, []).extend(lst)
    bands = list(DG)

    def f(theta):
        tr = G.apparent_track(sen, orb, e["jd"], np.linspace(-2.6, 2.6, 2000), dpos_km=np.asarray(theta, float))
        ok = np.isfinite(tr["ground"][:, 0])
        X, Y = G.ecef_to_utm(tr["ground"][ok], epsg)
        return M.band_roots(tr["alpha"][ok], X, Y, DG, bands, epsg_main=epsg)
    return f


for path in sys.argv[1:]:
    x = json.loads(Path(path).read_text()); e, r = x["event"], x["result"]
    if "pos_best" not in r:
        print(Path(path).stem, "no pos_best"); continue
    meas = measure(e, r)
    if not meas or len(meas) < 4:
        print(Path(path).stem, "too few measured bands", None if not meas else len(meas)); continue
    resA = np.array([[meas[b][0] - r["pos_best"][b][0], meas[b][1] - r["pos_best"][b][1]] for b in meas])
    rf = roots_fn(e, r)
    sx, sy = r["sx"], r["sy"]

    def res(theta):
        R = rf(theta); out = []
        for b, (mx, my) in meas.items():
            if b not in R:
                out += [3000.0, 3000.0]; continue
            q = min(R[b], key=lambda q: math.hypot(q[2] - mx, q[3] - my))
            out += [q[2] - mx, q[3] - my]
        return np.array(out)
    best = None
    for d0 in (r["dM"],):
        for c0 in (-1.0, 0.0, 1.0):
            for h0 in (-1.0, 0.0, 1.0):
                try:
                    s_ = least_squares(res, [d0, c0, h0], x_scale=[1.0, 0.5, 0.5], diff_step=1e-3, max_nfev=60)
                except Exception:
                    continue
                c = float(np.sqrt(np.mean(s_.fun ** 2)))
                if best is None or c < best[0]:
                    best = (c, s_.x)
    rmsA = float(np.sqrt(np.mean(resA ** 2)))
    print(f"{Path(path).stem}: bands {len(meas)} | model A (dM+shift+stretch, 4 params): rms {rmsA:.0f} m | "
          f"model B (3 orbit errors): rms {best[0]:.0f} m at along {best[1][0]:+.2f} km, cross {best[1][1]:+.2f} km, radial {best[1][2]:+.2f} km"
          f"  (stretch was {r.get('stretch')}, shift {sx:.0f},{sy:.0f})", flush=True)

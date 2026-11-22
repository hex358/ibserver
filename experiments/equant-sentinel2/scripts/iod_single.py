"""Орбита по одному снимку (THEORY §14): состояние объекта только из положений его изображения в полосах.

  python scripts/iod_single.py results/events_v9/STARLINK-31550_20261119T181632.json [...]

Полосы снимаются в разные моменты (до ~2.6 с), Sentinel-2 за это время смещается на ~20 км — база для
триангуляции. Для каждой полосы b известна плоскость визирования (угол α_b, MTD) и измеренная точка на земле X_b
(центр пятна на карте U в кэше ячеек). Неизвестно: положение и скорость объекта r0, v0 в момент t0 (6 чисел).
Модель: r(t) = r0 + v0·t + ½·g(r0)·t²; t_b — пересечение объекта с плоскостью полосы b; G_b — луч Sentinel→объект
∩ WGS84. Минимизируется Σ|G_b − X_b|². Начальные приближения — сетка высот и направлений; каталог НЕ используется,
он только для сравнения результата.
"""
import glob, json, math, pickle, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import orbits as O, geometry as G, imagery as I
from s2sat.pipeline import CellRaster, CELL

MU = 398600.4418
OMEGA = np.array([0.0, 0.0, 7.2921159e-5])   # вращение Земли, рад/с (состояние в ECEF)
snaps = sorted(p for p in (ROOT / "data/tle_snapshots").iterdir() if p.is_dir() and p.name <= "20261121T1217")


def load(pat):
    best = {}
    for sd in snaps:
        for f in sd.glob(pat):
            for o in O.parse_tle(f.read_text()):
                if o.catnr not in best or o.epoch_jd > best[o.catnr].epoch_jd:
                    best[o.catnr] = o
    return best


S2 = {o.name: o for o in load("sentinel2.tle").values()}
GP, SG = load("gp_*.tle"), load("supgp_*.tle")
A_WGS, F_WGS = 6378.137, 1 / 298.257223563
E2 = F_WGS * (2 - F_WGS)


def lla_to_ecef(lon, lat, h=0.0):
    lo, la = np.radians(lon), np.radians(lat)
    N = A_WGS / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.array([(N + h) * np.cos(la) * np.cos(lo), (N + h) * np.cos(la) * np.sin(lo), (N * (1 - E2) + h) * np.sin(la)])


def measure_positions(e, r, radius=200.0):
    """Центр пятна на карте U в каждой полосе около точки лучшей гипотезы (кэш ячеек)."""
    roots = r["roots_all"][r["n"]]
    pos = r.get("pos_best")
    if not pos:
        bands = list(roots); A = np.array([roots[b][0][1] for b in bands]); P = np.array([roots[b][0][2:4] for b in bands], float)
        c = np.polyfit(A, P, 1)[0]; t = c / np.linalg.norm(c); sig = (r.get("stretch") or 0.0) * r["kmdeg"] * 1000.0; aref = A.min()
        pos = {b: [P[i, 0] + r["sx"] + sig * (A[i] - aref) * t[0], P[i, 1] + r["sy"] + sig * (A[i] - aref) * t[1]] for i, b in enumerate(bands)}
    key = f"{e['obj'].split()[0]}_{e['sent'][-2:]}_{round(e['jd'] * 86400)}_glrt_z*loglocal*"
    ds = sorted(glob.glob(str(I.CACHE / "tiles" / key)))
    if not ds:
        return None
    need = {(int((px + dx) // CELL), int((py + dy) // CELL)) for px, py in pos.values() for dx in (-400, 0, 400) for dy in (-400, 0, 400)}
    tiles = {}
    for p in Path(ds[-1]).glob("*.pkl"):
        ci, cj = map(int, p.stem.split("_"))
        if (ci, cj) in need:
            tiles[(ci, cj)] = {"U": pickle.loads(p.read_bytes())["U"]}
    out = {}
    g = np.arange(-radius, radius + 1, 10.0); GX, GY = np.meshgrid(g, g)
    for b, (px, py) in pos.items():
        try:
            v = np.abs(np.nan_to_num(CellRaster(tiles, [b], "U").get(b, px + GX, py + GY).astype(float)))
        except Exception:
            continue
        k = int(np.argmax(v)); pk = v.flat[k]
        if pk < 4:
            continue                                   # полоса без уверенного пятна — не используем
        lab, _ = label(v >= 0.7 * pk); m = (lab == lab.flat[k]); w = v * m
        out[b] = (px + float((GX * w).sum() / w.sum()), py + float((GY * w).sum() / w.sum()), float(pk), roots[b][0][1])
    return out


def predict(state, jd0, sen, alphas):
    """Точки изображения в полосах (ECEF, км) для состояния объекта (r0, v0) в момент jd0."""
    r0, v0 = state[:3], state[3:]
    g0 = -MU * r0 / np.linalg.norm(r0) ** 3
    ts = np.arange(-6.0, 6.0001, 0.02)
    rS, vS = sen.ecef(jd0 + ts / 86400.0)
    rO = r0[None] + v0[None] * ts[:, None] + 0.5 * g0[None] * ts[:, None] ** 2
    out = []
    for al in alphas:
        F = np.einsum("tk,tk->t", rO - rS, G.band_plane(rS, vS, math.radians(al)))
        s = np.where(np.sign(F[:-1]) != np.sign(F[1:]))[0]
        if len(s) == 0:
            out.append(np.full(3, np.nan)); continue
        i = s[np.argmin(np.abs(s - len(ts) // 2))]
        f = F[i] / (F[i] - F[i + 1]); t = ts[i] + f * (ts[i + 1] - ts[i])
        rs = rS[i] + f * (rS[i + 1] - rS[i]); ro = r0 + v0 * t + 0.5 * g0 * t * t
        d = ro - rs; dn = d / np.linalg.norm(d)
        lam = G.ray_ellipsoid(rs[None], dn[None])[0]
        out.append(rs + lam * dn)
    return np.array(out)


def solve(e, r, meas):
    sen = S2[e["sent"]]; jd0 = e["jd"]
    bands = list(meas); alphas = [meas[b][3] for b in bands]
    X = np.array([lla_to_ecef(*G.utm_to_lonlat(meas[b][0], meas[b][1], r["epsg"])) for b in bands]).reshape(len(bands), 3)
    Xc = X.mean(0); up = Xc / np.linalg.norm(Xc)
    rS0, vS0 = sen.ecef(np.array([jd0]))
    rS0 = rS0[0]
    best = None
    for h in (350.0, 500.0, 650.0):
        # начальная точка: на луче Sentinel → центр изображения, на высоте h
        d = Xc - rS0; d /= np.linalg.norm(d)
        lo, hi = 0.0, np.linalg.norm(Xc - rS0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.linalg.norm(rS0 + mid * d) - 6371.0 > h: lo = mid
            else: hi = mid
        p0 = rS0 + lo * d; up0 = p0 / np.linalg.norm(p0)
        east = np.cross([0, 0, 1.0], up0); east /= np.linalg.norm(east); north = np.cross(up0, east)
        vc = math.sqrt(MU / np.linalg.norm(p0))
        for az in np.radians(np.arange(0, 360, 30)):
            v0 = vc * (math.cos(az) * north + math.sin(az) * east) - np.cross(OMEGA, p0)   # инерциальная → ECEF
            x0 = np.r_[p0, v0]
            altS = np.linalg.norm(rS0) - 6371.0

            def res(x):
                P = predict(x, jd0, sen, alphas)
                rr = np.nan_to_num(((P - X) * 1000.0).ravel(), nan=1e5)
                rn = np.linalg.norm(x[:3]); alt_ = rn - 6371.0
                # физика: объект между Sentinel-2 и Землёй; орбита почти круговая (|v| ≈ √(μ/r), допуск 20 м/с)
                pen = [max(0.0, alt_ - (altS - 50.0)) * 1e4, max(0.0, 150.0 - alt_) * 1e4,
                       (np.linalg.norm(x[3:] + np.cross(OMEGA, x[:3])) - math.sqrt(MU / rn)) / 0.02 * 30.0]   # инерциальная скорость
                return np.r_[rr, pen]
            try:
                sol = least_squares(res, x0, x_scale=np.r_[[50.0] * 3, [0.5] * 3], max_nfev=200)
            except Exception:
                continue
            cost = float(np.sqrt(np.mean(sol.fun[:-3] ** 2))) + float(np.abs(sol.fun[-3:]).sum())
            if best is None or cost < best[0]:
                best = (cost, sol.x)
    return best, X


if __name__ == "__main__":
    for path in sys.argv[1:]:
        x = json.loads(Path(path).read_text()); e, r = x["event"], x["result"]
        meas = measure_positions(e, r)
        if not meas or len(meas) < 4:
            print(Path(path).stem, "too few measured bands", None if not meas else len(meas)); continue
        best, X = solve(e, r, meas)
        if best is None:
            print(Path(path).stem, "no solution"); continue
        cost, st = best
        rr, vv = st[:3], st[3:]
        alt = np.linalg.norm(rr) - 6371.0; spd = np.linalg.norm(vv)
        orb = SG.get(e["cat"]) if e.get("supgp") and e["cat"] in SG else GP.get(e["cat"])
        rc, vc = orb.ecef(np.array([e["jd"]]))
        rc, vc = rc[0], vc[0]
        alt_c = np.linalg.norm(rc) - 6371.0
        ang = math.degrees(math.acos(np.clip(np.dot(vv, vc) / (np.linalg.norm(vv) * np.linalg.norm(vc)), -1, 1)))
        # расстояние до каталожной орбиты по нормали к её скорости (вдоль — это и есть искомая ошибка фазы)
        dv = rr - rc; along = float(np.dot(dv, vc / np.linalg.norm(vc))); perp = float(np.linalg.norm(dv - along * vc / np.linalg.norm(vc)))
        print(f"{Path(path).stem}: bands {len(meas)}, fit rms {cost:.0f} m | single-image: alt {alt:.1f} km, speed {spd:.3f} km/s"
              f" | catalogue: alt {alt_c:.1f} km, speed {np.linalg.norm(vc):.3f} | direction diff {ang:.2f}°, "
              f"position diff along {along:+.1f} km, across {perp:.2f} km  (detector dM {r.get('dM'):+.1f} km)", flush=True)

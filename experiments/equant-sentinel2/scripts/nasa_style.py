"""Кадр «как фото облёта NASA»: виртуальная камера в сотнях метров от объекта (как с отходящего корабля), перспектива;
тот же момент, Солнце и поза, что в realistic_view.py (THEORY §23). Внизу — Земля из ТОЙ ЖЕ съёмки Sentinel-2 (все
гранулы пролёта, ~100 м/пкс, облака того момента) на эллипсоиде WGS84; к ней добавлены рэлеевская дымка и ослабление по
реальной геометрии луча камера → земля. Земля — протяжённый источник света: подсвечивает объект снизу своей настоящей
яркостью; за краем снимка — средняя яркость сцены ниже горизонта. Потом — фотоаппарат: экспозиция, ореол бликов,
оптика, хроматизм, виньетирование, шум сенсора, JPEG. Это рендер, не фотография (так и записано в EXIF).

  python scripts/nasa_style.py ISS_20261118T222808 --el 60
"""
import argparse, glob, json, math, pickle, sys, time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--el", type=float, default=60.0, help="камера над горизонтом объекта, °")
ap.add_argument("--az", type=float, default=None, help="азимут камеры от объекта (от севера), °; по умолчанию — где Земля за объектом покрыта снимком и объект освещён спереди")
ap.add_argument("--dist", type=float, default=None, help="м; по умолчанию 4 размера объекта")
ap.add_argument("--fov", type=float, default=None, help="горизонтальный угол поля зрения, °")
ap.add_argument("--W", type=int, default=1920); ap.add_argument("--H", type=int, default=1080); ap.add_argument("--spp", type=int, default=512)
ap.add_argument("--work", default="/tmp/realistic_work"); ap.add_argument("--out", default=str(ROOT / "results/realistic"))
ap.add_argument("--tag", default="nasa")
a = ap.parse_args()
sys.argv = [sys.argv[0], a.name]
import realistic_view as RV
import hd_models as HDM
from s2sat import geometry as G, orbits as O, imagery as I
from s2sat.orbits import dt_from_jd

WA, WF = 6378.137, 1 / 298.257223563
WE2 = WF * (2 - WF)
TAU_R = np.array([0.0465, 0.0940, 0.1550])          # рэлеевская толщина, полосы R, G, B
TAU_A = 0.05                                         # аэрозоль (серый), чистый воздух


def lla_to_ecef(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    N = WA / np.sqrt(1 - WE2 * np.sin(la) ** 2)
    return np.stack([N * np.cos(la) * np.cos(lo), N * np.cos(la) * np.sin(lo), N * (1 - WE2) * np.sin(la)], -1)


def normal_ecef(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def cam_rays(cam, tgt, upv, fovx, W, H, nu=48, nv=27):
    f = tgt - cam; f /= np.linalg.norm(f)
    r = np.cross(f, upv); r /= np.linalg.norm(r); u = np.cross(r, f)
    tx = math.tan(math.radians(fovx) / 2); ty = tx * H / W
    U, Vv = np.meshgrid(np.linspace(-1, 1, nu), np.linspace(-1, 1, nv))
    d = f[None, None] + U[..., None] * tx * r + Vv[..., None] * ty * u
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def main():
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); work = Path(a.work) / f"{a.name}_{a.tag}"; work.mkdir(parents=True, exist_ok=True)
    evf = sorted(glob.glob(str(ROOT / f"results/found*/{a.name}.json"))) + sorted(glob.glob(str(ROOT / f"results/events_*/{a.name}.json")))
    x = json.loads(Path(evf[0]).read_text()); e, r = x["event"], x["result"]
    rc = pickle.loads((ROOT / "results/shape" / f"{a.name}.pkl").read_bytes())
    epsg = int(r["epsg"]); sen, orb = RV.choose_orbits(e, rc, None)
    band = next((b for b in ("green", "red", "blue", "nir") if b in rc["bands"]), next(iter(rc["bands"])))
    db = rc["bands"][band]; jd = float(db["jd"]); anchor = np.asarray(db["meas"][:2] if db.get("meas") else db["p0"], float)
    is_iss = a.name.startswith("ISS")
    geo = RV.frame_geometry(sen, orb, jd, epsg, inertial=is_iss)
    A, sun_b = geo["A"], geo["sun"]
    sun_e = O.sun_ecef([jd])[0]; sun_e = sun_e / np.linalg.norm(sun_e)
    # истинное положение объекта: на луче Sentinel → измеренная точка снимка, на дальности из орбиты
    lon_m, lat_m = G.utm_to_lonlat(anchor[0], anchor[1], epsg)
    Gm = lla_to_ecef(float(np.ravel(lon_m)[0]), float(np.ravel(lat_m)[0]))
    rS = geo["rS"]; rO = rS + geo["rng"] * (Gm - rS) / np.linalg.norm(Gm - rS)
    # модель и поза
    if is_iss:
        parts, angs = HDM.iss_parts(sun_b); note = f"МКС, штатная ориентация, крылья на Солнце (SARJ {angs['S']:+.0f}°)"
    else:
        rows = {}
        for f in sorted(ROOT.glob("results/recon*/summary.json")):
            for rr in json.loads(f.read_text()):
                rows[rr["name"]] = rr
        psi, phi = rows[a.name]["psi_deg"], rows[a.name]["phi_deg"]
        psi, phi, pn = HDM.starlink_pose(a.name, psi, phi)
        parts = HDM.starlink_parts(math.radians(psi), math.radians(phi)); note = f"Starlink V2 mini, {pn}"
    Vall = HDM.all_vertices(parts); size = float(np.ptp(Vall, 0).max())
    dist = a.dist or 4.0 * size
    fov = a.fov or math.degrees(2 * math.atan(0.5 * size / 0.62 / dist))
    # ось «вверх» в точке объекта, восток, север (ECEF)
    up_e = rO / np.linalg.norm(rO); east = np.cross([0, 0, 1.0], up_e); east /= np.linalg.norm(east); north = np.cross(up_e, east)
    up_b = A @ up_e
    items = I.items_around(e["lon"], e["lat"], dt_from_jd(e["jd"]), e["sent"][-1], d=4.0, minutes=8)
    from matplotlib.path import Path as MPath
    polys = []
    for it in items:                                          # контуры данных гранул (не прямоугольники тайлов)
        gm = it.geometry or {}
        rings = [gm["coordinates"][0]] if gm.get("type") == "Polygon" else [p_[0] for p_ in gm.get("coordinates", [])]
        polys += [MPath(np.asarray(rg)[:, :2]) for rg in rings]

    def footprint(az, el):
        h = math.cos(math.radians(az)) * north + math.sin(math.radians(az)) * east
        off_e = (math.cos(math.radians(el)) * h + math.sin(math.radians(el)) * up_e) * dist / 1000.0
        cam_b = A @ off_e * 1000.0
        d_b = cam_rays(cam_b, np.zeros(3), up_b, fov, a.W, a.H)
        d_e = d_b @ A                                       # тело → ECEF
        rC = rO + off_e
        lam = G.ray_ellipsoid(np.broadcast_to(rC, d_e.shape), d_e)
        P = rC + lam[..., None] * d_e
        ll = np.stack(G.ecef_to_lonlat(P.reshape(-1, 3)), -1) if np.isfinite(lam).all() else None
        return cam_b, rC, off_e, lam, P, ll

    best = None
    for el, az in [(el_, az_) for el_ in ((a.el,) if a.az is not None else (a.el, a.el + 6, a.el + 12))
                   for az_ in ([a.az] if a.az is not None else range(0, 360, 20))]:
        cam_b, rC, off_e, lam, P, ll = footprint(az, el)
        if ll is None:
            continue
        lo, la = ll[:, 0], ll[:, 1]
        pts = np.c_[lo, la]; inside = np.zeros(len(pts), bool)
        for pg in polys:
            inside |= pg.contains_points(pts)
        cov = float(inside.mean())
        front = float(off_e @ sun_e / np.linalg.norm(off_e))        # >0 — камера со стороны Солнца
        # полное покрытие важнее всего; затем освещённость спереди и наклон ближе к заданному
        score = 3.0 * min(cov, 0.995) + 0.3 * front - 0.01 * (el - a.el)
        if best is None or score > best[0]:
            best = (score, az, cov, front, el)
    _, az, cov, front, el = best
    cam_b, rC, off_e, lam, P, ll = footprint(az, el)
    print(f"{a.name}: {note}; камера {dist:.0f} м, азимут {az:.0f}°, {el:.0f}° над горизонтом, поле зрения {fov:.1f}°; "
          f"Земля в кадре покрыта снимком на {cov * 100:.0f}%, фаза {math.degrees(math.acos(np.clip(front, -1, 1))):.0f}°", flush=True)
    # снимок Земли: окно UTM по следу кадра, ~кадр/1.3 по разрешению
    X, Y = G.ecef_to_utm(P.reshape(-1, 3), epsg); X, Y = np.ravel(X), np.ravel(Y)
    span = max(X.max() - X.min(), Y.max() - Y.min())
    res = float(max(20.0, round(span / (a.W * 1.3) / 10) * 10))
    mg = 0.08 * span
    bb = (math.floor((X.min() - mg) / res) * res, math.floor((Y.min() - mg) / res) * res, math.ceil((X.max() + mg) / res) * res, math.ceil((Y.max() + mg) / res) * res)
    lon0, lat0 = G.utm_to_lonlat(np.array([bb[0], bb[2]]), np.array([bb[1], bb[3]]), epsg)
    use = [it for it in items if not (it.bbox[2] < min(lon0) or it.bbox[0] > max(lon0) or it.bbox[3] < min(lat0) or it.bbox[1] > max(lat0))]
    use.sort(key=lambda it: it.properties.get("eo:cloud_cover", 50))
    t1 = time.time(); RGB = []
    from concurrent.futures import ThreadPoolExecutor

    from pyproj import Transformer
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    Hm, Wm = int(round((bb[3] - bb[1]) / res)), int(round((bb[2] - bb[0]) / res))

    def rd(job):
        """Только пересечение гранулы с окном кадра, без дискового кэша (окна кадров облёта — сотни км)."""
        it, b = job
        lo_, la_ = np.meshgrid(np.linspace(it.bbox[0], it.bbox[2], 5), np.linspace(it.bbox[1], it.bbox[3], 5))
        ux, uy = to_utm.transform(lo_.ravel(), la_.ravel())
        x0, x1 = max(bb[0], min(ux)), min(bb[2], max(ux)); y0, y1 = max(bb[1], min(uy)), min(bb[3], max(uy))
        if x1 <= x0 or y1 <= y0:
            return None
        c0, c1 = int((x0 - bb[0]) // res), int(math.ceil((x1 - bb[0]) / res)); r0, r1 = int((bb[3] - y1) // res), int(math.ceil((bb[3] - y0) / res))
        tr = from_origin(bb[0] + c0 * res, bb[3] - r0 * res, res, res)
        for k in range(3):
            try:
                with rasterio.open(it.assets[b].href) as ds:
                    with WarpedVRT(ds, crs=f"EPSG:{epsg}", transform=tr, width=c1 - c0, height=r1 - r0,
                                   resampling=Resampling.average, nodata=0, src_nodata=0) as vrt:
                        return (r0, c0, vrt.read(1).astype(np.float32))
            except Exception:
                time.sleep(2 * (k + 1))
        return None
    jobs = [(it, b) for b in ("red", "green", "blue") for it in use]
    with ThreadPoolExecutor(8) as ex:
        wins = list(ex.map(rd, jobs))
    for b in ("red", "green", "blue"):
        acc = np.zeros((Hm, Wm), np.float32)
        for (it, bb_), w in zip(jobs, wins):
            if bb_ != b or w is None:
                continue
            r0, c0, arr = w; h_, w_ = arr.shape
            sub = acc[r0:r0 + h_, c0:c0 + w_]
            sub[...] = np.maximum(sub, arr[:sub.shape[0], :sub.shape[1]])   # края тайлов частично пусты — берём полный
        RGB.append(acc)
    valid = np.all(np.stack([c > 0 for c in RGB]), 0)
    rho = np.stack(RGB, -1) / 10000.0 - 0.1
    print(f"  Земля: {len(use)} гранул, {rho.shape[1]}×{rho.shape[0]} пкс по {res:.0f} м, покрыто {valid.mean() * 100:.0f}%, {time.time() - t1:.0f} с", flush=True)
    # вне полосы съёмки: продолжение соседней местности (нормированная свёртка), вдали от данных — открытый океан
    wv = valid.astype(np.float32)
    fill = np.zeros_like(rho); have = np.zeros(valid.shape, bool)
    for sg in (40, 160, 640):                                 # многомасштабно: ближе — детальнее, дальше — среднее
        den = gaussian_filter(wv, sg, mode="nearest")
        ok_ = (den > 0.02) & ~have
        for c in range(3):
            num = gaussian_filter(np.where(valid, rho[..., c], 0).astype(np.float32), sg, mode="nearest")
            fill[..., c] = np.where(ok_, num / np.maximum(den, 1e-6), fill[..., c])
        have |= ok_
    fill[~have] = [0.012, 0.018, 0.035]
    rho = np.where(valid[..., None], rho, fill)
    rho = np.clip(rho, 0, 1.2)
    # геометрия по текселям (грубая сетка → интерполяция): μs, μv, угол рассеяния
    Hh, Ww = rho.shape[:2]; gy, gx = np.linspace(0, Hh - 1, 48), np.linspace(0, Ww - 1, 48)
    GX, GY = np.meshgrid(bb[0] + (gx + 0.5) * res, bb[3] - (gy + 0.5) * res)
    lo, la = G.utm_to_lonlat(GX.ravel(), GY.ravel(), epsg)
    Pg = lla_to_ecef(np.asarray(lo), np.asarray(la)); ng = normal_ecef(np.asarray(lo), np.asarray(la))
    vdir = rC - Pg; vdir /= np.linalg.norm(vdir, axis=1, keepdims=True)
    mus = np.clip(ng @ sun_e, 0.02, 1); muv = np.clip(np.sum(ng * vdir, 1), 0.05, 1); cosT = -(vdir @ sun_e)
    up_ = lambda q: zoom(q.reshape(48, 48), (Hh / 48, Ww / 48), order=1)[:Hh, :Ww]
    mus, muv, cosT = up_(mus), up_(muv), up_(cosT)
    PR = 0.75 * (1 + cosT ** 2)
    path = (TAU_R[None, None] * PR[..., None] + TAU_A * 1.5) / (4 * mus[..., None] * muv[..., None]) * (1 - np.exp(-(TAU_R + TAU_A) * (1 / mus[..., None] + 1 / muv[..., None]))) / ((TAU_R + TAU_A) * (1 / mus[..., None] + 1 / muv[..., None]))
    trans = np.exp(-(TAU_R + TAU_A)[None, None] * (1 / mus[..., None] + 1 / muv[..., None]) * 0.55)
    L = (rho * trans + path) * mus[..., None]                  # яркость земли в единицах объекта (Солнце = π)
    nbad = int((~np.isfinite(L)).sum())
    if nbad:
        print(f"  ВНИМАНИЕ: {nbad} нечисловых значений в яркости Земли — заменены", flush=True)
        L = np.nan_to_num(L, nan=float(np.nanmedian(L)), posinf=1.0, neginf=0.0)
    import mitsuba as mi
    mi.set_variant("llvm_ad_rgb")
    mi.Bitmap(L.astype(np.float32)).write(str(work / "earth.exr"))
    # сетка Земли на эллипсоиде, в осях тела (м) относительно объекта; нормали — вверх
    n = 220
    ex = np.linspace(bb[0], bb[2], n); ey = np.linspace(bb[3], bb[1], n)
    EX, EY = np.meshgrid(ex, ey)
    lo, la = G.utm_to_lonlat(EX.ravel(), EY.ravel(), epsg)
    Pm = lla_to_ecef(np.asarray(lo), np.asarray(la))
    Vg = ((Pm - rO) @ A.T) * 1000.0
    UV = np.c_[(EX.ravel() - bb[0]) / (bb[2] - bb[0]), (EY.ravel() - bb[1]) / (bb[3] - bb[1])]
    idx = np.arange(n * n).reshape(n, n)
    q = np.c_[idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()]
    Fg = np.r_[q[:, [0, 1, 2]], q[:, [0, 2, 3]]]
    nz = np.cross(Vg[Fg[:, 1]] - Vg[Fg[:, 0]], Vg[Fg[:, 2]] - Vg[Fg[:, 0]])
    if np.mean(nz @ up_b) < 0:
        Fg = Fg[:, [0, 2, 1]]
    with open(work / "earth.obj", "w") as fh:
        fh.write("".join(f"v {p[0]:.2f} {p[1]:.2f} {p[2]:.2f}\n" for p in Vg))
        fh.write("".join(f"vt {u:.6f} {v:.6f}\n" for u, v in UV))
        fh.write("".join(f"f {i + 1}/{i + 1} {j + 1}/{j + 1} {k + 1}/{k + 1}\n" for i, j, k in Fg))
    T = mi.ScalarTransform4f
    shapes = HDM.mitsuba_shapes(parts, work / "obj", smooth=is_iss)
    Hh2, Wh2 = 128, 256
    elv = 90.0 - (np.arange(Hh2) + 0.5) * 180.0 / Hh2
    dip = math.degrees(math.acos(6371.0 / np.linalg.norm(rO)))
    env = np.zeros((Hh2, Wh2, 3), np.float32); env[elv < -dip] = L[valid].mean(0) if valid.any() else L.reshape(-1, 3).mean(0)
    # +Y карты окружения → «вверх» объекта (в осях тела)
    yb = up_b / np.linalg.norm(up_b); xb = np.cross(yb, [0, 0, 1.0] if abs(yb[2]) < 0.9 else [1.0, 0, 0]); xb /= np.linalg.norm(xb); zb = np.cross(xb, yb)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2] = xb, yb, zb
    scene = {"type": "scene", "integrator": {"type": "path", "max_depth": 7},
             "sensor": {"type": "perspective", "fov": fov, "fov_axis": "x", "near_clip": 0.5, "far_clip": 3.0e6,
                        "to_world": T().look_at(origin=list(cam_b), target=[0, 0, 0], up=list(up_b)),
                        "film": {"type": "hdrfilm", "width": a.W, "height": a.H, "rfilter": {"type": "gaussian"}},
                        "sampler": {"type": "multijitter", "sample_count": a.spp}},
             "sun": {"type": "directional", "direction": list(-sun_b), "irradiance": {"type": "rgb", "value": math.pi}},
             "sky": {"type": "envmap", "bitmap": mi.Bitmap(env), "to_world": T(M.tolist())},
             "earth": {"type": "obj", "filename": str(work / "earth.obj"), "face_normals": False,
                       "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.0}},
                       "emitter": {"type": "area", "radiance": {"type": "bitmap", "filename": str(work / "earth.exr"), "raw": True}}}}
    scene.update(shapes)
    t1 = time.time()
    img = np.array(mi.render(mi.load_dict(scene)), dtype=np.float32)[..., :3]
    nbad = int((~np.isfinite(img)).sum())
    if nbad:
        print(f"  ВНИМАНИЕ: {nbad} нечисловых пикселей рендера — заменены нулём", flush=True)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  рендер {a.W}×{a.H}, {a.spp} выборок: {time.time() - t1:.0f} с", flush=True)
    np.save(work / "linear.npy", img)
    jpg = camera(img, L[valid] if valid.any() else L.reshape(-1, 3))
    fn = out / f"{a.name}_{a.tag}.jpg"
    from PIL import Image
    im = Image.fromarray(jpg)
    ex = im.getexif(); ex[0x010E] = "Ocellus render, virtual camera (not a photograph): Sentinel-2 Earth, NASA/satvis 3D model"
    ex[0x0131] = "Mitsuba 3"
    im.save(fn, quality=92, exif=ex)
    meta = dict(name=a.name, note=note, dist_m=dist, fov_deg=fov, az_deg=az, el_deg=el, coverage=cov, earth_res_m=res,
                lens_mm_fullframe=round(36 / (2 * math.tan(math.radians(fov) / 2))), file=fn.name)
    (out / f"{a.name}_{a.tag}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"  → {fn.name}  (≈ {meta['lens_mm_fullframe']} мм на полном кадре), всего {time.time() - t0:.0f} с", flush=True)
    import shutil
    shutil.rmtree(work, ignore_errors=True)                     # текстура Земли и сетки — сотни МБ на объект


def camera(lin, Lground, seed=1):
    """Цифровая зеркальная камера: экспозиция по Земле, ореол ярких бликов, оптика, хроматизм, виньетирование,
    шум сенсора (фотонный + считывания), плёночная кривая, лёгкая резкость камеры, sRGB."""
    rng = np.random.default_rng(seed)
    lum = lambda x: x @ np.array([0.2126, 0.7152, 0.0722])
    k = 0.22 / max(float(np.median(lum(Lground))), 1e-4)
    x = lin * k
    # ореол: только то, что ярче белого
    hi = np.clip(x - 1.0, 0, None)
    glow = sum(w * np.stack([gaussian_filter(hi[..., c], s) for c in range(3)], -1) for s, w in ((2.0, 0.10), (8.0, 0.05), (32.0, 0.03)))
    x = x + glow
    # оптика: размытие ~0.7 пкс, поперечный хроматизм ±0.08 %, виньетирование 15 % в углах
    H, W = x.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); cy, cx = (H - 1) / 2, (W - 1) / 2
    for c, s in ((0, 1.0008), (2, 0.9992)):
        x[..., c] = map_coordinates(x[..., c], [cy + (yy - cy) / s, cx + (xx - cx) / s], order=1, mode="nearest")
    x = np.stack([gaussian_filter(x[..., c], 0.7) for c in range(3)], -1)
    r2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / (cy ** 2 + cx ** 2)
    x *= (1 - 0.15 * r2)[..., None]
    # сенсор: ~6000 фотоэлектронов на «белое», шум считывания 0.0015
    x = np.clip(x, 0, None)
    x = x + rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(x / 6000.0 + 0.0015 ** 2)
    y = RV.srgb(RV.aces(np.clip(x, 0, None)))
    # резкость камеры (нерезкая маска) и лёгкая насыщенность
    bl = np.stack([gaussian_filter(y[..., c], 1.2) for c in range(3)], -1)
    y = np.clip(y + 0.35 * (y - bl), 0, 1)
    g = y.mean(-1, keepdims=True); y = np.clip(g + 1.06 * (y - g), 0, 1)
    return (y * 255 + 0.5).astype(np.uint8)


if __name__ == "__main__":
    main()

"""Реалистичный кадр «как с кадровой камеры на борту Sentinel-2»: объект (сетка KeepTrack) в физически
правдоподобной позе над своей настоящей местностью из той же сцены (THEORY §23).

Физика кадра:
  * геометрия — центральная проекция из точки, где был Sentinel-2 в момент съёмки полосы (без растяжения
    push-broom): точка тела δ видна на земле в  J_f·δ,  J_f (2×3) — численно, луч Sentinel → точка → эллипсоид;
    масштаб ≈ λ/R (дальность до земли / дальность до объекта);
  * свет — Солнце (направленный источник, освещённость π: ламбертова поверхность альбедо ρ даёт яркость ρ·cos i,
    в тех же единицах, что и отражательная способность земли × cos θs) + свечение Земли снизу (полусфера ниже
    горизонта из точки объекта, яркость — средняя по сцене);
  * рендер — Mitsuba 3, трассировка путей (самозатенение, переотражения, блики);
  * земля — L2A той же сцены, собственные пятна объекта в RGB заменены прогнозом фона из других полос; объект над
    атмосферой, земля под ней → к земле добавлено рэлеевское рассеяние (дымка) и ослабление;
  * «широкий» кадр стабилизирован по земле (объект чуть смазан за выдержку), «теле» сопровождает объект
    (земля смазана вдоль видимого движения: |v_G|·τ);
  * земля 10 м → 2.5 м: EDSR ×4 (OpenCV dnn_superres, веса — raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x4.pb
    в data/models/sr/; нужен opencv-contrib-python-headless<5), дальше бикубика; рендер — pip install mitsuba.

  python scripts/realistic_view.py ISS_20261118T222808
  python scripts/realistic_view.py STARLINK-30580_20261120T111642 --shots wide:1.0:ground,tele:0.15:track
"""
import argparse, glob, json, math, pickle, sys, time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from s2sat import passgeom as PG, geometry as G, orbits as O, imagery as I
from s2sat.orbits import dt_from_jd
from s2sat.scenegeom import ground_utm, choose_orbits, frame_geometry   # общая геометрия кадра (оси орбиты — инерциальные)

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--until", default=None, help="снимок каталога; по умолчанию — последний не позже даты события + 1 сут")
ap.add_argument("--band", default="green", help="момент съёмки этой полосы = момент кадра")
ap.add_argument("--shots", default="auto", help="имя:м/пкс на земле:ground|track[:выдержка], либо auto")
ap.add_argument("--W", type=int, default=1920); ap.add_argument("--H", type=int, default=1080)
ap.add_argument("--tau-ground", type=float, default=1 / 4000); ap.add_argument("--tau-track", type=float, default=1 / 1000)
ap.add_argument("--spp", type=int, default=256)
ap.add_argument("--psi", type=float, default=None); ap.add_argument("--phi", type=float, default=None)
ap.add_argument("--out", default=str(ROOT / "results/realistic"))
ap.add_argument("--work", default="/tmp/realistic_work")
a = ap.parse_args()

OMEGA = 7.2921159e-5
SR_MODEL = ROOT / "data/models/sr/EDSR_x4.pb"


# ---------------------------------------------------------------- сетка и поза
def load_obj(path):
    V, F, MAT, OBJ = [], [], [], []
    o = mat = None
    for ln in open(path):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("o "):
            o = ln.split()[1]
        elif ln.startswith("usemtl"):
            mat = ln.split()[1]
        elif ln.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
            for k in range(1, len(idx) - 1):
                F.append([idx[0], idx[k], idx[k + 1]]); MAT.append(mat); OBJ.append(o)
    return np.array(V), np.array(F), np.array(MAT), np.array(OBJ)


def camera_basis(Jf, los):
    """Базис плоскости кадра (вправо, вверх) ⟂ линии визирования, повёрнутый так, чтобы карта кадр → земля (В, С)
    была симметричной (без поворота): A2' = W Σ Wᵀ."""
    t = np.array([0.0, 0.0, 1.0]) - los[2] * los
    if np.linalg.norm(t) < 1e-6:
        t = np.array([1.0, 0, 0]) - los[0] * los
    u2 = t / np.linalg.norm(t); u1 = np.cross(-los, u2)
    A2 = Jf @ np.c_[u1, u2]
    Wm, S, Vt = np.linalg.svd(A2)
    M = Vt.T @ Wm.T
    B = np.c_[u1, u2] @ M
    right, up = B[:, 0], B[:, 1]
    assert np.allclose(np.cross(-los, up), right, atol=1e-6), "кадр зеркальный"
    return right, up, Jf @ B, S


# ---------------------------------------------------------------- рендер объекта (Mitsuba 3)
def render_object(Vb, F, MAT, sun, los, right, up, pix, npx, earth_rgb, alt_km, spp, work, shapes=None, env=None):
    import mitsuba as mi
    try:
        mi.set_variant("llvm_ad_rgb")
    except Exception:
        mi.set_variant("scalar_rgb")
    T = mi.ScalarTransform4f
    work.mkdir(parents=True, exist_ok=True)
    pre = shapes
    shapes = dict(pre) if pre is not None else {}
    for i, m in enumerate([] if pre is not None else sorted(set(MAT))):
        sel = F[MAT == m]; u, inv = np.unique(sel, return_inverse=True)
        p = work / f"m{i}.obj"
        with open(p, "w") as fh:
            fh.write("".join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in Vb[u]))
            fh.write("".join(f"f {i0 + 1} {i1 + 1} {i2 + 1}\n" for i0, i1, i2 in inv.reshape(-1, 3)))
        base, met, rough = MATS.get(m, ((0.5, 0.5, 0.5), 0.2, 0.5))
        shapes[f"s{i}"] = {"type": "obj", "filename": str(p), "face_normals": True,
                           "bsdf": {"type": "twosided", "bsdf": {"type": "principled", "base_color": {"type": "rgb", "value": list(base)},
                                                                  "metallic": met, "roughness": rough}}}
    if env is None:
        env = earth_env(sun, alt_km, earth_rgb)
    D = 5000.0
    scene = {"type": "scene", "integrator": {"type": "path", "max_depth": 8, "hide_emitters": True},
             "sensor": {"type": "orthographic",
                        "to_world": T().look_at(origin=list(los * D), target=[0, 0, 0], up=list(up)) @ T().scale([npx * pix / 2, npx * pix / 2, 1]),
                        "near_clip": 1.0, "far_clip": 2 * D,
                        "film": {"type": "hdrfilm", "width": npx, "height": npx, "pixel_format": "rgba", "rfilter": {"type": "gaussian"}},
                        "sampler": {"type": "multijitter", "sample_count": spp}},
             "sun": {"type": "directional", "direction": list(-np.asarray(sun)), "irradiance": {"type": "rgb", "value": math.pi}},
             "earth": {"type": "envmap", "bitmap": mi.Bitmap(env), "to_world": T().rotate([1, 0, 0], 90)}}
    scene.update(shapes)
    sc = mi.load_dict(scene)
    # проверка ориентации камеры: «вправо» в кадре = right
    img = np.array(mi.render(sc), dtype=np.float32)
    return img                                  # RGB предумножен на покрытие, A — покрытие


# ---------------------------------------------------------------- земля
def earth_env_s2(geo, refl, bb, res, epsg, H=256, W=512):
    """Свет Земли для объекта из снимка Sentinel-2 этого же пролёта (оси тела: z — радиус). Каждое направление ниже
    горизонта — точка на эллипсоиде (луч из положения объекта); яркость = μs·(ρ·T + дымка) с косинусом зенитного угла
    Солнца μs в этой точке и наклоном луча μv (рэлеевское рассеяние, пропускание по двум путям). ρ — отражательная
    способность L2A полосы снимка (refl, сетка bb с шагом res в UTM epsg); за краем полосы съёмки — средняя ρ полосы.
    Раскладка — как у envmap Mitsuba при повороте [1,0,0] на 90° (строка — угол от зенита, столбец — азимут, u=0 → +y)."""
    A, rO = geo["A"], geo["rO"]
    s_e = A.T @ (np.asarray(geo["sun"], float) / np.linalg.norm(geo["sun"]))
    v = (np.arange(H) + 0.5) / H; u = (np.arange(W) + 0.5) / W
    th = math.pi * v[:, None]; ph = 2 * math.pi * u[None, :]
    Db = np.stack([np.sin(th) * np.sin(ph), np.sin(th) * np.cos(ph), np.cos(th) * np.ones_like(ph)], -1)
    De = Db @ A                                                       # направления в ECEF
    lam = G.ray_ellipsoid(np.broadcast_to(rO, De.shape), De)
    hit = np.isfinite(lam)
    Pg = rO + np.where(hit, lam, 0.0)[..., None] * De
    n = G.unit(Pg)
    mus = np.clip(n @ s_e, 0.0, 1.0); muv = np.clip(-(De * n).sum(-1), 0.02, 1.0)
    # ρ в точке: из снимка, где он есть; иначе — средняя по полосе
    good = np.isfinite(refl).all(-1) & (refl.max(-1) > 0)
    rho_mean = np.nanmean(np.where(good[..., None], refl, np.nan).reshape(-1, 3), 0)
    rho = np.tile(rho_mean, (H, W, 1))
    idx = np.where(hit.ravel())[0]
    if len(idx):
        X, Y = G.ecef_to_utm(Pg.reshape(-1, 3)[idx], epsg)
        col = ((X - bb[0]) / res - 0.5); row = ((bb[3] - Y) / res - 0.5)
        ins = (col >= 0) & (col < refl.shape[1] - 1) & (row >= 0) & (row < refl.shape[0] - 1)
        ci, ri = np.round(col[ins]).astype(int), np.round(row[ins]).astype(int)
        ok = good[ri, ci]
        flat = rho.reshape(-1, 3); sel = idx[ins][ok]
        flat[sel] = refl[ri[ok], ci[ok]]
        rho = flat.reshape(H, W, 3)
    tau = np.array([0.0465, 0.0940, 0.1550]) + 0.05 * (np.array([0.665, 0.56, 0.49]) / 0.55) ** -1.3   # Рэлей + аэрозоль
    am = 1 / np.clip(mus, 0.05, 1)[..., None] + 1 / muv[..., None]
    trans = np.exp(-tau * am)
    cosT = np.clip((Db * (np.asarray(geo["sun"], float) / np.linalg.norm(geo["sun"]))).sum(-1), -1, 1)
    P_ray = 0.75 * (1 + cosT ** 2)
    haze = np.array([0.0465, 0.0940, 0.1550]) * P_ray[..., None] / (4 * muv[..., None])
    haze = np.minimum(haze, 0.35)
    L = mus[..., None] * (np.clip(rho, 0, 1.5) * trans + haze)
    return np.where(hit[..., None], L, 0.0).astype(np.float32)


def earth_env(sun_b, alt_km, earth_rgb_local, local_km=60.0, H=192, W=384):
    """Карта окружения: свет Земли, как его видит объект на высоте alt_km (оси тела: z — радиус). Каждое направление ниже
    горизонта — точка на шаре Земли; яркость = отражательная способность × освещённость Солнцем в этой точке (косинус
    зенитного угла Солнца там — терминатор, ночная сторона тёмная) + рэлеевская дымка с учётом наклона луча.
    Под объектом (радиус local_km) — отражательная способность из снимка Sentinel-2 (earth_rgb_local — яркость на
    надире); дальше — средняя Земля с облаками (альбедо в верхней атмосфере ≈ 0.30). Единицы: облучённость Солнца = π.
    Раскладка — как у envmap Mitsuba при повороте [1,0,0] на 90°: строка — угол от зенита, столбец — азимут
    (u = 0 → +y, u = 1/4 → +x)."""
    R = 6371.0; h = float(alt_km); P = np.array([0.0, 0.0, R + h])
    s = np.asarray(sun_b, float); s = s / np.linalg.norm(s)
    v = (np.arange(H) + 0.5) / H; u = (np.arange(W) + 0.5) / W
    th = math.pi * v[:, None]; ph = 2 * math.pi * u[None, :]
    D = np.stack([np.sin(th) * np.sin(ph), np.sin(th) * np.cos(ph), np.cos(th) * np.ones_like(ph)], -1)
    b = D @ P; c = P @ P - R * R; disc = b * b - c
    hit = (disc > 0) & (b < 0)
    t = np.where(hit, -b - np.sqrt(np.clip(disc, 0, None)), 0.0)
    G = P + t[..., None] * D; n = G / R
    mus = np.clip(n @ s, 0.0, 1.0)                                   # освещённость точки Солнцем
    muv = np.clip(-(D * n).sum(-1), 0.02, 1.0)                       # наклон луча к поверхности (у лимба мал)
    tau = np.array([0.0465, 0.0940, 0.1550])
    trans = np.exp(-0.5 * tau[None, None, :] * (1 / np.clip(mus, 0.05, 1)[..., None] + 1 / muv[..., None]))
    cosT = np.clip((D * s).sum(-1), -1, 1)                           # рассеяние назад/вперёд
    haze = tau[None, None, :] * (0.75 * (1 + cosT ** 2))[..., None] / (4 * muv[..., None])
    haze = np.minimum(haze, 0.35)                                    # у лимба — насыщение (однократное рассеяние)
    dist = R * np.arccos(np.clip(n[..., 2], -1, 1))                  # расстояние по поверхности от надира
    mu0 = max(float(s[2]), 0.05)
    rho_loc = np.asarray(earth_rgb_local, float) / mu0              # отражательная способность под объектом (с дымкой)
    rho_glob = np.array([0.30, 0.31, 0.34])                          # средняя Земля с облаками (альбедо ВГА)
    w = np.clip((local_km * 1.5 - dist) / (local_km * 0.5), 0, 1)[..., None]   # плавный переход
    surf = w * rho_loc + (1 - w) * rho_glob
    L = mus[..., None] * (surf * np.where(w > 0, 1.0, trans) + haze * (1 - w))
    env = np.where(hit[..., None], L, 0.0).astype(np.float32)
    return env


def rayleigh(sun_el_deg, vza_deg=5.0):
    """Однократное рэлеевское рассеяние: дымка (в ед. отражательной способности) и пропускание, полосы R, G, B."""
    tau = np.array([0.0465, 0.0940, 0.1550])
    ms, mv = math.sin(math.radians(sun_el_deg)), math.cos(math.radians(vza_deg))
    cosT = -ms * mv + math.cos(math.radians(sun_el_deg)) * math.sin(math.radians(vza_deg))
    P = 0.75 * (1 + cosT ** 2)
    path = tau * P / (4 * ms * mv)
    trans = np.exp(-0.5 * tau * (1 / ms + 1 / mv))
    return path, trans


def super_res(rgb_lin, scale_ref):
    """EDSR ×4 на гамма-кодированной картинке (так модель обучена), обратно в линейные единицы."""
    import cv2
    sr = cv2.dnn_superres.DnnSuperResImpl_create(); sr.readModel(str(SR_MODEL)); sr.setModel("edsr", 4)
    enc = np.clip(rgb_lin / scale_ref, 0, 1) ** (1 / 2.2)
    bgr = (enc[..., ::-1] * 255 + 0.5).astype(np.uint8)
    up = sr.upsample(bgr).astype(np.float32) / 255.0
    return (up[..., ::-1] ** 2.2) * scale_ref


def line_kernel(v_px, n_sub=8):
    L = float(np.hypot(*v_px))
    if L < 0.5:
        return np.ones((1, 1), np.float32)
    R = int(math.ceil(L / 2)) + 1
    k = np.zeros((2 * R + 1, 2 * R + 1), np.float32)
    for t in np.linspace(-0.5, 0.5, max(2, int(L * n_sub))):
        x, y = t * v_px[0] + R, -t * v_px[1] + R        # v_px: (восток, север) в пикселях; строки вниз
        i0, j0 = int(math.floor(y)), int(math.floor(x)); fy, fx = y - i0, x - j0
        for di, wy in ((0, 1 - fy), (1, fy)):
            for dj, wx in ((0, 1 - fx), (1, fx)):
                k[i0 + di, j0 + dj] += wy * wx
    return k / k.sum()


def blur(img, k):
    if k.size == 1:
        return img
    py, px = k.shape[0] // 2 + 1, k.shape[1] // 2 + 1
    pad = np.pad(img, ((py, py), (px, px), (0, 0)), mode="edge")
    return np.stack([fftconvolve(pad[..., c], k, mode="same") for c in range(img.shape[-1])], -1)[py:-py, px:-px]


def aces(x):
    return np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


# ---------------------------------------------------------------- main
def main():
    import physdet as PD
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    evf = sorted(glob.glob(str(ROOT / f"results/found*/{a.name}.json"))) + sorted(glob.glob(str(ROOT / f"results/events_*/{a.name}.json")))
    x = json.loads(Path(evf[0]).read_text()); e, r = x["event"], x["result"]
    rc = pickle.loads((ROOT / "results/shape" / f"{a.name}.pkl").read_bytes())
    epsg = int(r["epsg"])
    sen, orb = choose_orbits(e, rc, a.until)
    band = next((b for b in (a.band, "green", "red", "blue", "nir") if b in rc["bands"]), next(iter(rc["bands"])))
    db = rc["bands"][band]
    jd = float(db["jd"]); anchor = np.asarray(db["meas"][:2] if db.get("meas") else db["p0"], float)
    geo = frame_geometry(sen, orb, jd, epsg)
    sun, los = geo["sun"], geo["los"]
    import hd_models as HDM
    parts, pose_note, smooth = HDM.parts_for(a.name, sun, int(e.get("cat", 0)))   # модель из каталога, поза — автоматически
    Vb = HDM.all_vertices(parts); F = MAT = None
    hd_shapes = HDM.mitsuba_shapes(parts, Path(a.work) / a.name / "hd", smooth=smooth)
    right, up, A2, S = camera_basis(geo["Jf"], los)
    k_scale = float(S.mean())
    print(f"{a.name}: дальность {geo['rng']:.0f} км, λ/R≈{k_scale:.2f} (σ {S.round(3)}), видимая скорость на земле "
          f"{np.linalg.norm(geo['vG']) / 1000:.2f} км/с, {pose_note}; якорь − орбита = {np.round(anchor - geo['g0'])} м", flush=True)
    vsun = math.degrees(math.asin(max(-1, min(1, sun[2]))))

    # земля: L2A RGB, окно под самый широкий кадр
    proj = (Vb - np.outer(Vb @ los, los))
    ext = float(np.max(np.abs(np.c_[proj @ right, proj @ up]))) + 2.0
    if a.shots == "auto":                         # общий план 2.5 м; средний — объект на 15% ширины; теле — на 45%
        span = 2 * ext * k_scale
        shots = [["wide", "2.5", "ground", str(a.tau_ground)],
                 ["mid", f"{max(1.0, round(span / (0.15 * a.W), 2))}", "track", str(1 / 4000)],
                 ["tele", f"{max(0.03, round(span / (0.45 * a.W), 3))}", "track", str(a.tau_track)]]
    else:
        shots = [s.split(":") for s in a.shots.split(",")]
    half = max(float(s_[1]) * a.W / 2 for s_ in shots) + 300.0; halfy = max(float(s_[1]) * a.H / 2 for s_ in shots) + 300.0
    bb = (math.floor((anchor[0] - half) / 10) * 10, math.floor((anchor[1] - halfy) / 10) * 10,
          math.ceil((anchor[0] + half) / 10) * 10, math.ceil((anchor[1] + halfy) / 10) * 10)
    its = I.items_around(e["lon"], e["lat"], dt_from_jd(e["jd"]), e["sent"][-1], d=1.0)
    its.sort(key=lambda it: it.id != r.get("item"))
    IM = PD.read_band_region(its, epsg, bb, ["red", "green", "blue"], 10.0)
    # убрать собственные пятна объекта (в каждой полосе — своё место): прогноз фона из реконструкции
    for b in ("red", "green", "blue"):
        d = rc["bands"].get(b)
        if d is None:
            continue
        msk = binary_dilation(np.abs(d["r"]) > 4 * np.asarray(d["s"]), iterations=3)
        cols = np.round((d["X"] - bb[0] - 5) / 10).astype(int); rows_ = np.round((bb[3] - 5 - d["Y"]) / 10).astype(int)
        okk = msk & (cols >= 0) & (cols < IM[b].shape[1]) & (rows_ >= 0) & (rows_ < IM[b].shape[0])
        IM[b][rows_[okk], cols[okk]] = d["g"][okk] + 1000.0   # g — без смещения BOA_ADD_OFFSET
        print(f"  {b}: заменено {okk.sum()} пкс пятна объекта", flush=True)
    refl = np.stack([IM["red"], IM["green"], IM["blue"]], -1) / 10000.0 - 0.1   # Collection-1 L2A: ρ = DN·1e-4 − 0.1
    path, trans = rayleigh(e["sun_el"])
    mus = math.sin(math.radians(e["sun_el"]))
    Lg = (np.clip(refl, 0, 1.5) * trans + path) * mus           # яркость земли на верхней границе атмосферы, ед. объекта
    earth = Lg.reshape(-1, 3).mean(0)
    ref = float(np.percentile(Lg, 99.9) * 1.25)
    t1 = time.time()
    cache = Path(a.work) / f"sr_{a.name}_{'_'.join(str(int(v)) for v in bb)}.npy"
    if cache.exists():
        Lsr = np.load(cache)
    else:
        Lsr = super_res(Lg, ref); cache.parent.mkdir(parents=True, exist_ok=True); np.save(cache, Lsr)
    print(f"  EDSR ×4: {Lg.shape[:2]} → {Lsr.shape[:2]}, {time.time() - t1:.0f} с", flush=True)
    lum = Lg @ np.array([0.2126, 0.7152, 0.0722])
    expo = float(np.clip(0.18 / np.median(lum), 1.2, 8.0))
    # свет Земли: вся полоса съёмки этого пролёта вокруг (±200 км, 500 м), та же отражательная способность L2A
    t1 = time.time(); W_KM = 200.0; RES_W = 500.0
    bbw = (anchor[0] - W_KM * 1e3, anchor[1] - W_KM * 1e3, anchor[0] + W_KM * 1e3, anchor[1] + W_KM * 1e3)
    itw = I.items_around(e["lon"], e["lat"], dt_from_jd(e["jd"]), e["sent"][-1], d=2.2)
    IMW = PD.read_band_region(itw, epsg, bbw, ["red", "green", "blue"], RES_W)
    rw = np.stack([IMW[b] for b in ("red", "green", "blue")], -1)
    rw = np.where((rw > 0).all(-1, keepdims=True), rw / 10000.0 - 0.1, np.nan)
    env = earth_env_s2(geo, rw, bbw, RES_W, epsg)
    print(f"  свет Земли: полоса {rw.shape[:2]} по {RES_W:.0f} м, покрыто снимком {np.isfinite(rw).all(-1).mean() * 100:.0f}%, "
          f"{len(itw)} гранул, {time.time() - t1:.0f} с", flush=True)

    meta = dict(name=a.name, pose=pose_note, rng_km=geo["rng"], k=k_scale, vG_ms=geo["vG"].tolist(), sun_body=sun.tolist(),
                los_body=los.tolist(), earth_rgb=earth.tolist(), earth_light="полоса Sentinel-2 этого пролёта (earth_env_s2)", haze=path.tolist(), trans=trans.tolist(), exposure_gain=expo, shots=[])
    for sh in shots:
        nm, rg, mode = sh[0], float(sh[1]), sh[2]; t1 = time.time()
        tau_sh = float(sh[3]) if len(sh) > 3 else (a.tau_track if mode == "track" else a.tau_ground)
        pix = rg / k_scale                                     # м на объекте на пиксель кадра
        npx = int(math.ceil(2 * ext / pix)) + 8
        obj = render_object(Vb, F, MAT, sun, los, right, up, pix, npx, earth, geo["alt"], a.spp, Path(a.work) / a.name, shapes=hd_shapes, env=env)
        phf = Path(a.out) / f"photometry_{a.name}.json"          # яркость объекта, измеренная Sentinel-2 (photometry_check)
        if phf.exists():
            ch = json.loads(phf.read_text())["channels"]
            obj[..., :3] *= np.array([ch.get(c, {}).get("scale", 1.0) for c in "RGB"], np.float32)
        # кадр: центр — якорь, север вверх
        jj, ii = np.meshgrid(np.arange(a.W), np.arange(a.H))
        gE = (jj + 0.5 - a.W / 2) * rg; gN = (a.H / 2 - ii - 0.5) * rg
        q = np.linalg.solve(A2, np.stack([gE.ravel(), gN.ravel()]))
        cc = q[0] / pix + npx / 2 - 0.5; rr_ = npx / 2 - q[1] / pix - 0.5
        layer = np.stack([map_coordinates(obj[..., c], [rr_, cc], order=1, mode="constant").reshape(a.H, a.W) for c in range(4)], -1)
        # земля в этом кадре: из сетки 2.5 м (EDSR), бикубически
        sres = 2.5
        X0 = bb[0]; Y1 = bb[3]
        gx = (anchor[0] + gE - X0) / sres - 0.5; gy = (Y1 - anchor[1] - gN) / sres - 0.5
        ground = np.stack([map_coordinates(Lsr[..., c], [gy, gx], order=3, mode="nearest") for c in range(3)], -1)
        vpx = geo["vG"] / rg
        if mode == "track":
            ground = blur(ground, line_kernel(vpx * tau_sh)); tau = tau_sh
        else:
            layer = blur(layer, line_kernel(vpx * tau_sh)); tau = tau_sh
        comp = layer[..., :3] + ground * (1 - layer[..., 3:4])
        img = srgb(aces(comp * expo))
        from PIL import Image
        fn = out / f"{a.name}_{nm}.png"
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(fn)
        meta["shots"].append(dict(file=fn.name, m_per_px_ground=rg, m_per_px_object=pix, mode=mode, exposure_s=tau,
                                  smear_m=float(np.linalg.norm(geo["vG"]) * tau)))
        print(f"  {nm}: {rg} м/пкс на земле = {pix * 100:.0f} см на объекте, {mode}, выдержка 1/{1 / tau:.0f} с "
              f"(смаз {np.linalg.norm(geo['vG']) * tau:.1f} м), {time.time() - t1:.0f} с → {fn.name}", flush=True)
    (out / f"{a.name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"готово за {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()

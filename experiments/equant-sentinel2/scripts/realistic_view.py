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


def iss_body(V, F, OBJ, sun_b):
    """Штатная ориентация +XVV (модули вдоль скорости, ферма поперёк, ферма над модулями) — сетка: Z вдоль, X поперёк,
    Y в зенит. Панели (SARJ) повёрнуты вокруг оси фермы на Солнце: нормаль панелей = проекция Солнца на плоскость
    (вдоль, радиус). BGA (β ≈ 2°) не учитываем."""
    W = V.copy()
    rot = np.zeros(len(V), bool)
    rot[np.unique(F[np.isin(OBJ, ["ITSSolarPanels", "ITSPanelBrackets"])])] = True
    pan = np.unique(F[OBJ == "ITSSolarPanels"])
    y0, z0 = V[pan, 1].mean(), V[pan, 2].mean()
    alpha = math.atan2(-sun_b[2], sun_b[0])          # нормаль (Y=0, Z=1) → (−sin α, cos α) ∥ (s_рад, s_вдоль)
    c, s = math.cos(alpha), math.sin(alpha)
    y, z = V[rot, 1] - y0, V[rot, 2] - z0
    W[rot, 1] = y0 + c * y - s * z; W[rot, 2] = z0 + s * y + c * z
    return np.c_[W[:, 2], W[:, 0], W[:, 1]], math.degrees(alpha)


MATS = {  # материал → principled BSDF (база, металл, шероховатость); по смыслу, альбедо из справочных значений
    "SolarPanels": ((0.040, 0.040, 0.075), 0.0, 0.12), "solar_cell": ((0.035, 0.045, 0.10), 0.0, 0.12),
    "WhiteMetal": ((0.80, 0.80, 0.78), 0.0, 0.6), "white": ((0.82, 0.82, 0.84), 0.0, 0.6),
    "LightGrayMetal": ((0.60, 0.60, 0.62), 0.6, 0.45), "LightGrayMetal.001": ((0.60, 0.60, 0.62), 0.6, 0.45),
    "DarkGrayMetal": ((0.10, 0.10, 0.11), 0.3, 0.5), "dark_detail": ((0.10, 0.10, 0.11), 0.3, 0.5),
    "BrownMetal": ((0.75, 0.55, 0.25), 0.85, 0.35), "RadiatorWhite": ((0.86, 0.86, 0.85), 0.0, 0.5),
    "panel_back": ((0.70, 0.71, 0.73), 0.0, 0.7), "silver": ((0.80, 0.81, 0.84), 0.9, 0.25),
    "bus_gray": ((0.30, 0.31, 0.33), 0.5, 0.45), "phased_array": ((0.05, 0.07, 0.12), 0.2, 0.3),
}


# ---------------------------------------------------------------- геометрия кадра
def ground_utm(rS, p, epsg):
    d = G.unit(p - rS); lam = G.ray_ellipsoid(rS, d)
    X, Y = G.ecef_to_utm((rS + lam * d)[None], epsg)
    return np.array([float(np.ravel(X)[0]), float(np.ravel(Y)[0])])


def choose_orbits(e, rc, until=None):
    """TLE Sentinel-2 с эпохой, ближайшей к снимку (архивные события — не из будущего каталога); орбита объекта —
    из TLE этого объекта в ±6 сут та, что воспроизводит дальность, с которой делалась реконструкция."""
    orbs = PG.Orbits(until=until or "99999999")
    jd = e["jd"]
    S2 = []
    for sd in (ROOT / "data/tle_snapshots").iterdir():
        f = sd / "sentinel2.tle"
        if f.exists():
            S2 += [o for o in O.parse_tle(f.read_text()) if o.name == e["sent"]]
    sen = min(S2, key=lambda o: abs(o.epoch_jd - jd))
    pool = [o for o in orbs.SG.get(e["cat"], []) + orbs.GP.get(e["cat"], []) if abs(o.epoch_jd - jd) < 6]
    rS = sen.ecef([jd])[0][0]
    rng = lambda o: float(np.linalg.norm(o.ecef([jd])[0][0] - rS))
    orb = min(pool, key=lambda o: abs(rng(o) - rc["rng"]))
    return sen, orb


def frame_geometry(sen, orb, jd, epsg, inertial):
    rO, vO = orb.ecef([jd]); rO, vO = rO[0], vO[0]
    rS = sen.ecef([jd])[0][0]
    ur = G.unit(rO)
    v = vO + (np.cross([0, 0, OMEGA], rO) if inertial else 0.0)   # LVLH МКС — по инерциальной скорости
    uv = G.unit(v - (v @ ur) * ur); uc = G.unit(np.cross(ur, uv))
    A = np.stack([uv, uc, ur])
    su = O.sun_ecef([jd])[0]; su = su / np.linalg.norm(su)
    los = G.unit(rS - rO)
    g0 = ground_utm(rS, rO, epsg)
    h = 0.05
    Jf = np.stack([(ground_utm(rS, rO + A[k] * h, epsg) - g0) / (h * 1000.0) for k in range(3)], 1)
    dt = 0.05 / 86400.0
    gp = ground_utm(sen.ecef([jd + dt])[0][0], orb.ecef([jd + dt])[0][0], epsg)
    gm = ground_utm(sen.ecef([jd - dt])[0][0], orb.ecef([jd - dt])[0][0], epsg)
    vG = (gp - gm) / 0.1
    return dict(A=A, sun=A @ su, los=A @ los, Jf=Jf, vG=vG, g0=g0, rng=float(np.linalg.norm(rS - rO)),
                alt=float(np.linalg.norm(rO) - 6371.0), rS=rS, rO=rO)


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
def render_object(Vb, F, MAT, sun, los, right, up, pix, npx, earth_rgb, alt_km, spp, work):
    import mitsuba as mi
    try:
        mi.set_variant("llvm_ad_rgb")
    except Exception:
        mi.set_variant("scalar_rgb")
    T = mi.ScalarTransform4f
    work.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for i, m in enumerate(sorted(set(MAT))):
        sel = F[MAT == m]; u, inv = np.unique(sel, return_inverse=True)
        p = work / f"m{i}.obj"
        with open(p, "w") as fh:
            fh.write("".join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in Vb[u]))
            fh.write("".join(f"f {i0 + 1} {i1 + 1} {i2 + 1}\n" for i0, i1, i2 in inv.reshape(-1, 3)))
        base, met, rough = MATS.get(m, ((0.5, 0.5, 0.5), 0.2, 0.5))
        shapes[f"s{i}"] = {"type": "obj", "filename": str(p), "face_normals": True,
                           "bsdf": {"type": "twosided", "bsdf": {"type": "principled", "base_color": {"type": "rgb", "value": list(base)},
                                                                  "metallic": met, "roughness": rough}}}
    # свечение Земли: полусфера ниже горизонта (провал горизонта acos(R/(R+h))), локальный +Y карты → +радиус тела
    Hh, Wh = 128, 256
    el = 90.0 - (np.arange(Hh) + 0.5) * 180.0 / Hh
    dip = math.degrees(math.acos(6371.0 / (6371.0 + alt_km)))
    env = np.zeros((Hh, Wh, 3), np.float32); env[el < -dip] = np.asarray(earth_rgb, np.float32)
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
    is_iss = a.name.startswith("ISS")
    geo = frame_geometry(sen, orb, jd, epsg, inertial=is_iss)
    sun, los = geo["sun"], geo["los"]
    if is_iss:
        V, F, MAT, OBJ = load_obj(ROOT / "data/models/iss.obj")
        Vb, alpha = iss_body(V, F, OBJ, sun)
        MAT = np.where(np.char.startswith(OBJ.astype(str), "Radiator") | (OBJ == "ITSRadiators"), "RadiatorWhite", MAT)
        pose_note = f"штатная ориентация +XVV, панели на Солнце (SARJ {alpha:+.0f}°)"
    else:
        import model_recon as MR
        rows = {}
        for f in sorted(ROOT.glob("results/recon*/summary.json")):
            for rr in json.loads(f.read_text()):
                rows[rr["name"]] = rr
        psi = a.psi if a.psi is not None else rows[a.name]["psi_deg"]; phi = a.phi if a.phi is not None else rows[a.name]["phi_deg"]
        Vb = MR.body_vertices(math.radians(psi), math.radians(phi)); F = MR.F0
        MAT = []; mat = None
        for ln in open(MR.MESH):
            if ln.startswith("usemtl"):
                mat = ln.split()[1]
            elif ln.startswith("f "):
                MAT += [mat] * (len(ln.split()) - 3)
        MAT = np.array(MAT)
        pose_note = f"поза из реконструкции: рысканье {psi:+.0f}°, крылья {phi:+.0f}°"
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

    meta = dict(name=a.name, pose=pose_note, rng_km=geo["rng"], k=k_scale, vG_ms=geo["vG"].tolist(), sun_body=sun.tolist(),
                los_body=los.tolist(), earth_rgb=earth.tolist(), haze=path.tolist(), trans=trans.tolist(), exposure_gain=expo, shots=[])
    for sh in shots:
        nm, rg, mode = sh[0], float(sh[1]), sh[2]; t1 = time.time()
        tau_sh = float(sh[3]) if len(sh) > 3 else (a.tau_track if mode == "track" else a.tau_ground)
        pix = rg / k_scale                                     # м на объекте на пиксель кадра
        npx = int(math.ceil(2 * ext / pix)) + 8
        obj = render_object(Vb, F, MAT, sun, los, right, up, pix, npx, earth, geo["alt"], a.spp, Path(a.work) / a.name)
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

"""Видео пролёта в нескольких вариантах — от самого достоверного к самому понятному (THEORY §23):

  1 «сырые данные»       — настоящие пиксели Sentinel-2 (10 м), без модели; каждая полоса сканирует кадр своей линией
                            в свой момент и ловит объект в своём месте (разноцветные пятна на снимке);
  2 «геометрия сканера»  — то же, но вместо пикселей 10 м — модель, снятая каждой полосой так, как её снял бы сканер
                            Sentinel-2 с пикселем ~0.6 м: сканер для быстро летящего объекта = ортографическая камера,
                            смотрящая вдоль нуль-вектора якобиана полосы J_b (а не по линии визирования), и отображение
                            J_b на землю — отсюда растяжение, перекос и разнос цветов;
  3 «кадровая камера»    — flyby_video.py: камера в точке Sentinel-2 без искажений, реальный масштаб и поворот;
  4 «сопровождение»      — та же точка, камера ведёт объект, земля уносится вдоль следа (смаз выдержки);
  5 «облёт NASA»         — камера в сотнях метров от объекта летит вместе с ним, Земля той же съёмки внизу (реальное время).

  python scripts/flyby_variants.py ISS_20261118T222808 --only 1,2,4,5
"""
import argparse, glob, json, math, pickle, subprocess, sys, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--only", default="1,2,4,5"); ap.add_argument("--W", type=int, default=1920); ap.add_argument("--H", type=int, default=1080)
ap.add_argument("--fps", type=int, default=30); ap.add_argument("--spp", type=int, default=192)
ap.add_argument("--work", default="/tmp/realistic_work"); ap.add_argument("--out", default=str(ROOT / "results/flyby"))
a = ap.parse_args()
sys.argv = [sys.argv[0], a.name]
import flyby_video as FV            # make_sprite, paste, ground_layer, tone, BANDC, BANDN, nice, L_fmt
RV = FV.RV; HDM = FV.HDM
from s2sat import geometry as G, imagery as I
from s2sat.orbits import dt_from_jd

FONT, FONTB = FV.FONT, FV.FONTB
V_SCAN = 6600.0


class Enc:
    def __init__(self, path, W, H, fps):
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", str(path)], stdin=subprocess.PIPE)
        self.n = 0

    def put(self, img):
        self.p.stdin.write(np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img, np.uint8).tobytes()); self.n += 1

    def close(self):
        self.p.stdin.close(); self.p.wait()


def header(img, num, title, sub, right=None):
    dr = ImageDraw.Draw(img, "RGBA")
    f1, f2 = ImageFont.truetype(FONTB, 32), ImageFont.truetype(FONT, 22)
    dr.rectangle([0, 0, img.width, 92], fill=(0, 0, 0, 120))
    dr.text((26, 12), f"{num}/5 · {title}", font=f1, fill=(255, 255, 255, 240))
    dr.text((26, 56), sub, font=f2, fill=(215, 222, 228, 225))
    if right:
        dr.text((img.width - 26 - dr.textlength(right, font=f1), 12), right, font=f1, fill=(255, 255, 255, 235))
    return dr


def scalebar(dr, x, y, px, label, col=(255, 255, 255)):
    f3 = ImageFont.truetype(FONT, 21)
    dr.rectangle([x - 8, y - 30, x + max(px, dr.textlength(label, font=f3)) + 8, y + 12], fill=(0, 0, 0, 120))
    dr.line([x, y, x + px, y], fill=col + (255,), width=5)
    for xx in (x, x + px):
        dr.line([xx, y - 8, xx, y + 8], fill=col + (255,), width=3)
    dr.text((x, y - 30), label, font=f3, fill=col + (255,))


def north(dr, W, H):
    f2 = ImageFont.truetype(FONT, 22); cx, cy = W - 56, H - 86
    dr.polygon([(cx, cy - 34), (cx - 12, cy), (cx + 12, cy)], fill=(255, 255, 255, 230)); dr.text((cx - 8, cy + 4), "С", font=f2, fill=(255, 255, 255, 230))


# ------------------------------------------------------------------ общая подготовка
def setup():
    e_f = sorted(glob.glob(str(ROOT / f"results/found*/{a.name}.json"))) + sorted(glob.glob(str(ROOT / f"results/events_*/{a.name}.json")))
    x = json.loads(Path(e_f[0]).read_text()); e, r = x["event"], x["result"]
    rc = pickle.loads((ROOT / "results/shape" / f"{a.name}.pkl").read_bytes()); epsg = int(r["epsg"])
    sen, orb = RV.choose_orbits(e, rc, None)
    b0 = next((b for b in ("green", "red", "blue", "nir") if b in rc["bands"]), next(iter(rc["bands"])))
    db = rc["bands"][b0]; jd0 = float(db["jd"]); anchor = np.asarray(db["meas"][:2] if db.get("meas") else db["p0"], float)
    is_iss = a.name.startswith("ISS")
    geo = RV.frame_geometry(sen, orb, jd0, epsg, inertial=is_iss)
    geo_e = RV.frame_geometry(sen, orb, jd0, epsg, inertial=False)        # оси, в которых записаны J полос
    parts, _, smooth = HDM.parts_for(a.name, geo["sun"], int(e.get("cat", 0)))
    title = "МКС" if is_iss else a.name.split("_")[0]
    work = Path(a.work) / f"{a.name}_var"; work.mkdir(parents=True, exist_ok=True)
    Vb = HDM.all_vertices(parts); shapes = HDM.mitsuba_shapes(parts, work / "hd", smooth=smooth)
    right, up, A2, S = RV.camera_basis(geo["Jf"], geo["los"]); k = float(S.mean())
    proj = Vb - np.outer(Vb @ geo["los"], geo["los"]); ext = float(np.max(np.abs(np.c_[proj @ right, proj @ up]))) + 2.0
    bands = sorted([(b, (float(d["jd"]) - jd0) * 86400.0, np.asarray(d["meas"][:2] if d.get("meas") else d["p0"], float), d) for b, d in rc["bands"].items()],
                   key=lambda t: t[1])
    its = I.items_around(e["lon"], e["lat"], dt_from_jd(e["jd"]), e["sent"][-1], d=1.0); its.sort(key=lambda it: it.id != r.get("item"))
    path, trans = RV.rayleigh(e["sun_el"]); mus = math.sin(math.radians(e["sun_el"]))
    toL = lambda rho: (np.clip(rho, 0, 1.5) * trans + path) * mus
    psi_tr = G.track_azimuth(sen, jd0)
    a_hat = np.array([math.sin(psi_tr), math.cos(psi_tr)])                 # направление сканирования на земле (В, С)
    R_ie = geo_e["A"] @ geo["A"].T                                          # тело (инерциальные оси) → оси J
    return dict(e=e, r=r, rc=rc, epsg=epsg, sen=sen, orb=orb, jd0=jd0, anchor=anchor, geo=geo, parts=parts, shapes=shapes, Vb=Vb, title=title,
                right=right, up=up, A2=A2, k=k, ext=ext, span_g=2 * ext * k, bands=bands, its=its, toL=toL, a_hat=a_hat, R_ie=R_ie, work=work,
                is_iss=is_iss)


def raw_rgb(c, bb):
    """Сырые полосы R, G, B на 10 м (без очистки — с настоящими пятнами объекта)."""
    import physdet as PD
    IM = PD.read_band_region(c["its"], c["epsg"], bb, ["red", "green", "blue"], 10.0)
    return np.stack([IM["red"], IM["green"], IM["blue"]], -1) / 10000.0 - 0.1


def scan_region(c, s=2):
    """Окно для вариантов 1–2: все пятна R, G, B в кадре; s экранных пикселей на пиксель 10 м."""
    P = np.array([p for b, _, p, _ in c["bands"] if b in ("red", "green", "blue")])
    cen = (P.min(0) + P.max(0)) / 2
    w, h = a.W // s * 10, a.H // s * 10
    bb = (int(math.floor((cen[0] - w / 2) / 10) * 10), int(math.floor((cen[1] - h / 2) / 10) * 10))
    bb = (bb[0], bb[1], bb[0] + w, bb[1] + h)
    return bb


def fronts(c, bb, res):
    """Вдоль-трассовая координата s каждого пикселя и линия сканирования полосы: s_b(t) = s_blob + V·(t − t_b)."""
    Hn, Wn = int((bb[3] - bb[1]) / res), int((bb[2] - bb[0]) / res)
    jj, ii = np.meshgrid(np.arange(Wn), np.arange(Hn))
    E = bb[0] + (jj + 0.5) * res; N = bb[3] - (ii + 0.5) * res
    s_pix = E * c["a_hat"][0] + N * c["a_hat"][1]
    fb = {b: (float(p @ c["a_hat"]), tb) for b, tb, p, _ in c["bands"]}
    return s_pix, fb


def tl_of(fb, b, t):
    s0, tb = fb[b]; return s0 + V_SCAN * (t - tb)


# ------------------------------------------------------------------ 1. сырые данные
def v1(c):
    s = 2; bb = scan_region(c, s)
    rho = raw_rgb(c, bb)
    L = c["toL"](rho); expo = float(np.clip(0.18 / np.median(L @ np.array([0.2126, 0.7152, 0.0722])), 1.2, 8.0)); c["expo"] = expo
    T = (FV.tone(L, expo) * 255 + 0.5).astype(np.uint8)
    s_pix, fb = fronts(c, bb, 10.0)
    BG = {"red": 0, "green": 1, "blue": 2}
    tb = [fb[b] for b in BG]
    t_lo = min(( s_pix.min() - s0) / V_SCAN + t_ for s0, t_ in tb) - 0.05
    t_hi = max(( s_pix.max() - s0) / V_SCAN + t_ for s0, t_ in tb) + 0.05
    dur = 14.0; n = int(dur * a.fps); slow = dur / (t_hi - t_lo)
    enc = Enc(Path(a.out) / f"{a.name}_v1_raw.mp4", a.W, a.H, a.fps)
    f3 = ImageFont.truetype(FONT, 22)
    up = lambda arr: np.repeat(np.repeat(arr, s, 0), s, 1)
    def px(p):
        return ((p[0] - bb[0]) / 10 * s, (bb[3] - p[1]) / 10 * s)
    for i in range(n):
        t = t_lo + (t_hi - t_lo) * i / (n - 1)
        fr = np.zeros_like(T)
        for b, ch in BG.items():
            m = s_pix <= tl_of(fb, b, t)
            fr[..., ch] = np.where(m, T[..., ch], 0)
        img = Image.fromarray(up(fr)); dr = ImageDraw.Draw(img, "RGBA")
        # линии сканирования полос
        for b, ch in BG.items():
            sl = tl_of(fb, b, t)
            # точки линии: s_pix == sl, отрезок поперёк трассы через кадр
            nrm = np.array([c["a_hat"][1], -c["a_hat"][0]]); cen = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
            p0 = cen + c["a_hat"] * (sl - cen @ c["a_hat"]); pts = [p0 + nrm * d for d in (-3e4, 3e4)]
            q = [px(p) for p in pts]
            dr.line(q, fill=FV.BANDC[b] + (200,), width=3)
        for b, tb_, p, _ in c["bands"]:
            if b in BG and t >= tb_:
                x, y = px(p); col = FV.BANDC[b]
                dr.ellipse([x - 34, y - 34, x + 34, y + 34], outline=col + (220,), width=3)
                dr.text((x + 40, y - 14), f"{FV.BANDN[b]}: {c['title']} здесь, t = {tb_ * 1000:+.0f} мс", font=f3, fill=col + (235,))
        header(img, 1, "Сырые данные Sentinel-2 — как видел спутник",
               f"настоящие пиксели 10 м, без модели · каждая полоса сканирует своей линией в свой момент · замедлено ×{slow:.0f}",
               f"t = {t * 1000:+.0f} мс")
        scalebar(dr, 30, a.H - 40, 1000 / 10 * s, "1 км на земле"); north(dr, a.W, a.H)
        enc.put(img)
    enc.close(); return dict(slow=slow, real_s=t_hi - t_lo, bb=bb)


# ------------------------------------------------------------------ 2. модель в геометрии сканера
def band_sprite(c, b, rg, spp):
    """Рендер модели, как её записал бы сканер полосы b: ортографическая камера вдоль нуль-вектора J_b."""
    d = c["rc"]["bands"][b]
    J = d["J"] / 1000.0 @ c["R_ie"]                                       # м на земле на м тела (оси тела — как у модели)
    n = np.linalg.svd(J)[2][2]; n = n if n @ c["geo"]["los"] > 0 else -n
    right, up, A2b, Sb = RV.camera_basis(J, n)
    Vb = c["Vb"]; proj = Vb - np.outer(Vb @ n, n); ext = float(np.max(np.abs(np.c_[proj @ right, proj @ up]))) + 2.0
    pix = rg / float(Sb.max()) / 2; npx = int(math.ceil(2 * ext / pix)) + 8
    obj = RV.render_object(Vb, None, None, c["geo"]["sun"], n, right, up, pix, npx, np.array([0.08, 0.08, 0.09]), c["geo"]["alt"], spp,
                           c["work"] / f"pb_{b}", shapes=c["shapes"])
    vG = np.asarray(d.get("vapp", c["geo"]["vG"]), float)
    return FV.make_sprite(obj, npx, pix, A2b, rg, np.zeros(2), 0.0), Sb, math.degrees(math.acos(np.clip(n @ c["geo"]["los"], -1, 1)))


def v2(c):
    s = 2; bb = scan_region(c, s); rg = 10.0 / s
    rho = FV.ground_layer(c["its"], c["epsg"], c["rc"], (bb[0] - 300, bb[1] - 300, bb[2] + 300, bb[3] + 300), 10.0)
    L = c["toL"](rho); expo = c.get("expo") or float(np.clip(0.18 / np.median(L @ np.array([0.2126, 0.7152, 0.0722])), 1.2, 8.0))
    Lsr = RV.super_res(L, float(np.percentile(L, 99.9) * 1.25))           # 2.5 м
    jj, ii = np.meshgrid(np.arange(a.W), np.arange(a.H))
    gx = (bb[0] + (jj + 0.5) * rg - (bb[0] - 300)) / 2.5 - 0.5; gy = ((bb[3] + 300) - (bb[3] - (ii + 0.5) * rg)) / 2.5 - 0.5
    Gl = np.stack([map_coordinates(Lsr[..., ch], [gy, gx], order=3, mode="nearest") for ch in range(3)], -1)
    BG = {"red": 0, "green": 1, "blue": 2}
    pos = {b: p for b, _, p, _ in c["bands"]}; tbs = {b: tb for b, tb, _, _ in c["bands"]}
    comp = Gl.copy(); info = {}
    ins_rg = 0.6
    insets = {}
    for b, ch in BG.items():
        sp, Sb, ang = band_sprite(c, b, rg, a.spp); info[b] = dict(stretch=[float(v) for v in Sb], view_vs_los_deg=ang)
        x = (pos[b][0] - bb[0]) / rg; y = (bb[3] - pos[b][1]) / rg
        res = FV.paste(comp[..., ch:ch + 1].repeat(3, -1), sp, x, y)
        if res:
            a0, a1, b0, b1, win = res; comp[a0:a1, b0:b1, ch] = win[..., ch]
        # вставка: что записала эта полоса, 0.6 м/пкс, в цвете полосы
        spi, _, _ = band_sprite(c, b, ins_rg, a.spp)
        iw, ih = 600, 300
        jj2, ii2 = np.meshgrid(np.arange(iw), np.arange(ih))
        gx2 = (pos[b][0] + (jj2 + 0.5 - iw / 2) * ins_rg - (bb[0] - 300)) / 2.5 - 0.5; gy2 = ((bb[3] + 300) - (pos[b][1] + (ih / 2 - ii2 - 0.5) * ins_rg)) / 2.5 - 0.5
        gi = map_coordinates(Lsr[..., ch], [gy2, gx2], order=3, mode="nearest")
        lay = np.repeat(gi[..., None], 3, -1)
        res = FV.paste(lay, spi, iw / 2, ih / 2)
        if res:
            a0, a1, b0, b1, win = res; lay[a0:a1, b0:b1] = np.repeat(win[..., ch:ch + 1], 3, -1)
        mono = FV.tone(lay[..., 0], expo)
        tint = np.array(FV.BANDC[b]) / 255.0
        insets[b] = (np.clip(mono[..., None] * (0.35 + 0.65 * tint)[None, None] / (0.35 + 0.65 * tint).max(), 0, 1) * 255).astype(np.uint8)
    T = (FV.tone(comp, expo) * 255 + 0.5).astype(np.uint8)
    s_pix, fb = fronts(c, bb, 10.0 / s)
    tb = [fb[b] for b in BG]
    t_lo = min((s_pix.min() - s0) / V_SCAN + t_ for s0, t_ in tb) - 0.05
    t_hi = max((s_pix.max() - s0) / V_SCAN + t_ for s0, t_ in tb) + 0.05
    dur = 14.0; n = int(dur * a.fps); slow = dur / (t_hi - t_lo)
    enc = Enc(Path(a.out) / f"{a.name}_v2_scanner.mp4", a.W, a.H, a.fps)
    f3 = ImageFont.truetype(FONT, 21)
    px = lambda p: ((p[0] - bb[0]) / rg, (bb[3] - p[1]) / rg)
    for i in range(n):
        t = t_lo + (t_hi - t_lo) * i / (n - 1)
        fr = np.zeros_like(T)
        for b, ch in BG.items():
            fr[..., ch] = np.where(s_pix <= tl_of(fb, b, t), T[..., ch], 0)
        img = Image.fromarray(fr); dr = ImageDraw.Draw(img, "RGBA")
        nrm = np.array([c["a_hat"][1], -c["a_hat"][0]]); cen = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
        for b in BG:
            sl = tl_of(fb, b, t); p0 = cen + c["a_hat"] * (sl - cen @ c["a_hat"])
            dr.line([px(p0 + nrm * dd) for dd in (-3e4, 3e4)], fill=FV.BANDC[b] + (200,), width=3)
        k_ = 0
        for b in ("red", "green", "blue"):
            if t >= tbs[b]:
                x, y = px(pos[b]); col = FV.BANDC[b]
                dr.ellipse([x - 40, y - 40, x + 40, y + 40], outline=col + (220,), width=3)
                X0 = 26 + k_ * 620; Y0 = a.H - 300 - 60
                img.paste(Image.fromarray(insets[b]), (X0, Y0)); dr = ImageDraw.Draw(img, "RGBA")
                dr.rectangle([X0, Y0, X0 + 600, Y0 + 300], outline=col + (255,), width=3)
                st = info[b]["stretch"]
                dr.text((X0 + 10, Y0 + 8), f"{FV.BANDN[b]}, 0.6 м/пкс: растяжение {st[0]:.2f}×{st[1]:.2f}", font=f3, fill=(255, 255, 255, 240))
            k_ += 1
        header(img, 2, "Модель в геометрии сканера Sentinel-2",
               f"как сняла бы каждая полоса с пикселем 0.6 м: растяжение, перекос и разнос цветов — от сканирования · замедлено ×{slow:.0f}",
               f"t = {t * 1000:+.0f} мс")
        north(dr, a.W, a.H)
        enc.put(img)
    enc.close(); return dict(slow=slow, real_s=t_hi - t_lo, bands=info)


# ------------------------------------------------------------------ 4. сопровождение
def v4(c):
    geo = c["geo"]; vG = geo["vG"]; anchor = c["anchor"]; k = c["k"]
    rg = float(max(0.5, round(c["span_g"] / (0.3 * a.W) * 4) / 4)); dur = 8.0
    T_real = 0.20 if c["is_iss"] else 0.12
    trav = np.abs(vG) * T_real
    hx, hy = a.W * rg / 2 + trav[0] / 2 + 200, a.H * rg / 2 + trav[1] / 2 + 200
    bb = (int(math.floor((anchor[0] - hx) / 10) * 10), int(math.floor((anchor[1] - hy) / 10) * 10), int(math.ceil((anchor[0] + hx) / 10) * 10), int(math.ceil((anchor[1] + hy) / 10) * 10))
    rho = FV.ground_layer(c["its"], c["epsg"], c["rc"], bb, 10.0); L = c["toL"](rho)
    expo = c.get("expo") or float(np.clip(0.18 / np.median(L @ np.array([0.2126, 0.7152, 0.0722])), 1.2, 8.0))
    Lsr = RV.super_res(L, float(np.percentile(L, 99.9) * 1.25))
    pix = rg / k; npx = int(math.ceil(2 * c["ext"] / pix)) + 8
    obj = RV.render_object(c["Vb"], None, None, geo["sun"], geo["los"], c["right"], c["up"], pix, npx, L.reshape(-1, 3).mean(0), geo["alt"], a.spp, c["work"] / "v4", shapes=c["shapes"])
    sp = FV.make_sprite(obj, npx, pix, c["A2"], rg, np.zeros(2), 0.0)
    tau = 1 / 1000; Lsr = RV.blur(Lsr, RV.line_kernel(vG / 2.5 * tau))    # смаз выдержки в осях земли — один раз
    n = int(dur * a.fps); slow = dur / T_real
    jj, ii = np.meshgrid(np.arange(a.W), np.arange(a.H))
    enc = Enc(Path(a.out) / f"{a.name}_v4_tracking.mp4", a.W, a.H, a.fps)
    for i in range(n):
        t = -T_real / 2 + T_real * i / (n - 1)
        c0 = anchor + vG * t
        gx = (c0[0] + (jj + 0.5 - a.W / 2) * rg - bb[0]) / 2.5 - 0.5; gy = (bb[3] - c0[1] - (a.H / 2 - ii - 0.5) * rg) / 2.5 - 0.5
        Gl = np.stack([map_coordinates(Lsr[..., ch], [gy, gx], order=1, mode="nearest") for ch in range(3)], -1)
        res = FV.paste(Gl, sp, a.W / 2, a.H / 2)
        if res:
            a0, a1, b0, b1, win = res; Gl[a0:a1, b0:b1] = win
        img = Image.fromarray((FV.tone(Gl, expo) * 255 + 0.5).astype(np.uint8))
        dr = header(img, 4, "Сопровождение с точки Sentinel-2",
                    f"камера ведёт объект, земля уносится со скоростью следа {np.linalg.norm(vG) / 1000:.1f} км/с · выдержка 1/1000 с · {rg:g} м/пкс на земле · замедлено ×{slow:.0f}",
                    f"t = {t * 1000:+.0f} мс")
        scalebar(dr, 30, a.H - 40, nice_px(200, rg), f"{FV.L_fmt(200)} на земле")
        scalebar(dr, 30, a.H - 98, 50 * k / rg, f"50 м на объекте (×{k:.2f}: он ближе к камере)", (255, 220, 120)); north(dr, a.W, a.H)
        enc.put(img)
    enc.close(); return dict(slow=slow, m_per_px=rg)


def nice_px(m, rg):
    return m / rg


# ------------------------------------------------------------------ 5. облёт NASA (реальное время)
def v5(c, dur=12.0):
    """Облёт: камера в 4 размерах объекта летит вместе с ним (неподвижна в орбитальных осях), Земля той же съёмки
    Sentinel-2 внизу движется с реальной скоростью; дымка — по геометрии каждого пикселя каждого кадра."""
    import nasa_style as NS
    import mitsuba as mi
    from concurrent.futures import ThreadPoolExecutor
    from matplotlib.path import Path as MPath
    from s2sat import orbits as O
    geo = c["geo"]; e = c["e"]; jd0 = c["jd0"]; orb = c["orb"]; sen = c["sen"]; is_iss = c["is_iss"]; epsg = c["epsg"]
    W, H = a.W, a.H
    lon_m, lat_m = G.utm_to_lonlat(c["anchor"][0], c["anchor"][1], epsg)
    Gm = NS.lla_to_ecef(float(np.ravel(lon_m)[0]), float(np.ravel(lat_m)[0]))
    rS = geo["rS"]; rO0 = rS + geo["rng"] * (Gm - rS) / np.linalg.norm(Gm - rS)
    dO = rO0 - orb.ecef([jd0])[0][0]                                   # поправка орбиты (истинное − TLE)
    def frame_at(t):
        jd = jd0 + t / 86400.0
        rO, vO = orb.ecef([jd]); rO, vO = rO[0] + dO, vO[0]
        ur = rO / np.linalg.norm(rO); v = vO + (np.cross([0, 0, 7.2921159e-5], rO) if is_iss else 0.0)
        uv = v - (v @ ur) * ur; uv /= np.linalg.norm(uv); uc = np.cross(ur, uv)
        return rO, np.stack([uv, uc, ur])
    size = float(np.ptp(c["Vb"], 0).max()); dist = 4.0 * size; fov = math.degrees(2 * math.atan(0.5 * size / 0.62 / dist)); el = 60.0
    sun_e = O.sun_ecef([jd0])[0]; sun_e /= np.linalg.norm(sun_e)
    items = I.items_around(e["lon"], e["lat"], dt_from_jd(e["jd"]), e["sent"][-1], d=4.0, minutes=8)
    polys = []
    for it in items:
        gm = it.geometry or {}
        rings = [gm["coordinates"][0]] if gm.get("type") == "Polygon" else [q[0] for q in gm.get("coordinates", [])]
        polys += [MPath(np.asarray(rg_)[:, :2]) for rg_ in rings]
    def ground_pts(t, cam_b, nu, nv):
        rO, A = frame_at(t); up_b = np.array([0, 0, 1.0])
        d_b = NS.cam_rays(cam_b, np.zeros(3), up_b, fov, W, H, nu, nv); d_e = d_b @ A
        rC = rO + (A.T @ cam_b) / 1000.0
        lam = G.ray_ellipsoid(np.broadcast_to(rC, d_e.shape), d_e)
        return rC, d_e, lam, rC + lam[..., None] * d_e
    best = None
    for az in range(0, 360, 20):
        rO, A = frame_at(0.0)
        up_e = rO / np.linalg.norm(rO); east = np.cross([0, 0, 1.0], up_e); east /= np.linalg.norm(east); north = np.cross(up_e, east)
        h = math.cos(math.radians(az)) * north + math.sin(math.radians(az)) * east
        off_e = (math.cos(math.radians(el)) * h + math.sin(math.radians(el)) * up_e) * dist
        cam_b = A @ off_e
        covs = []
        for t in (0.0, dur):
            _, _, lam, P = ground_pts(t, cam_b, 16, 9)
            if not np.isfinite(lam).all():
                covs.append(0.0); continue
            lo, la = G.ecef_to_lonlat(P.reshape(-1, 3)); pts = np.c_[lo, la]; ins = np.zeros(len(pts), bool)
            for pg in polys:
                ins |= pg.contains_points(pts)
            covs.append(ins.mean())
        front = float(off_e @ sun_e / dist); score = 3 * min(min(covs), 0.995) + 0.3 * front
        if best is None or score > best[0]:
            best = (score, az, cam_b, min(covs))
    _, az, cam_b, cov = best
    # мозаика Земли по объединению следов кадра в начале и в конце
    XY = []
    for t in (0.0, dur / 2, dur):
        _, _, _, P = ground_pts(t, cam_b, 16, 9); X, Y = G.ecef_to_utm(P.reshape(-1, 3), epsg); XY.append(np.c_[np.ravel(X), np.ravel(Y)])
    XY = np.vstack(XY); span = np.ptp(XY, 0).max()
    res = float(max(30.0, round(span / (W * 1.6) / 10) * 10)); mg = 0.06 * span
    bb = (math.floor((XY[:, 0].min() - mg) / res) * res, math.floor((XY[:, 1].min() - mg) / res) * res, math.ceil((XY[:, 0].max() + mg) / res) * res, math.ceil((XY[:, 1].max() + mg) / res) * res)
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from pyproj import Transformer
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    Hm, Wm = int(round((bb[3] - bb[1]) / res)), int(round((bb[2] - bb[0]) / res))
    def rd(job):
        it, b = job
        lo_, la_ = np.meshgrid(np.linspace(it.bbox[0], it.bbox[2], 5), np.linspace(it.bbox[1], it.bbox[3], 5))
        ux, uy = to_utm.transform(lo_.ravel(), la_.ravel())
        x0, x1 = max(bb[0], min(ux)), min(bb[2], max(ux)); y0, y1 = max(bb[1], min(uy)), min(bb[3], max(uy))
        if x1 <= x0 or y1 <= y0:
            return None
        c0, c1 = int((x0 - bb[0]) // res), int(math.ceil((x1 - bb[0]) / res)); r0, r1 = int((bb[3] - y1) // res), int(math.ceil((bb[3] - y0) / res))
        for kk in range(3):
            try:
                with rasterio.open(it.assets[b].href) as ds:
                    with WarpedVRT(ds, crs=f"EPSG:{epsg}", transform=from_origin(bb[0] + c0 * res, bb[3] - r0 * res, res, res), width=c1 - c0, height=r1 - r0,
                                   resampling=Resampling.average, nodata=0, src_nodata=0) as vrt:
                        return (r0, c0, vrt.read(1).astype(np.float32))
            except Exception:
                time.sleep(2 * (kk + 1))
        return None
    jobs = [(it, b) for b in ("red", "green", "blue") for it in items]
    with ThreadPoolExecutor(8) as ex:
        wins = list(ex.map(rd, jobs))
    RGB = []
    for b in ("red", "green", "blue"):
        acc = np.zeros((Hm, Wm), np.float32)
        for (it, b_), w in zip(jobs, wins):
            if b_ != b or w is None:
                continue
            r0, c0, arr = w; sub = acc[r0:r0 + arr.shape[0], c0:c0 + arr.shape[1]]; sub[...] = np.maximum(sub, arr[:sub.shape[0], :sub.shape[1]])
        RGB.append(acc)
    valid = np.all(np.stack([q > 0 for q in RGB]), 0); rho = np.stack(RGB, -1) / 10000.0 - 0.1
    wv = valid.astype(np.float32); fill = np.zeros_like(rho); have = np.zeros(valid.shape, bool)
    for sg in (20, 80, 320):
        den = gaussian_filter(wv, sg, mode="nearest"); ok_ = (den > 0.02) & ~have
        for ch in range(3):
            num = gaussian_filter(np.where(valid, rho[..., ch], 0).astype(np.float32), sg, mode="nearest")
            fill[..., ch] = np.where(ok_, num / np.maximum(den, 1e-6), fill[..., ch])
        have |= ok_
    fill[~have] = [0.012, 0.018, 0.035]; rho = np.clip(np.where(valid[..., None], rho, fill), 0, 1.2)
    print(f"  облёт: азимут камеры {az}°, {dist:.0f} м, поле {fov:.1f}°, покрытие {cov * 100:.0f}%, Земля {Wm}×{Hm} по {res:.0f} м ({valid.mean() * 100:.0f}% снимком)", flush=True)
    # объект: рендер один раз (камера неподвижна в орбитальных осях)
    mi.set_variant("llvm_ad_rgb"); T = mi.ScalarTransform4f
    Hh2, Wh2 = 128, 256; elv = 90.0 - (np.arange(Hh2) + 0.5) * 180.0 / Hh2
    rO, A = frame_at(0.0); dip = math.degrees(math.acos(6371.0 / np.linalg.norm(rO)))
    Lmean = (rho[valid] if valid.any() else rho.reshape(-1, 3)).mean(0) * 0.9 * math.sin(math.radians(e["sun_el"]))
    env = np.zeros((Hh2, Wh2, 3), np.float32); env[elv < -dip] = Lmean
    scene = {"type": "scene", "integrator": {"type": "path", "max_depth": 7, "hide_emitters": True},
             "sensor": {"type": "perspective", "fov": fov, "fov_axis": "x", "near_clip": 0.5, "far_clip": 1e5,
                        "to_world": T().look_at(origin=list(cam_b), target=[0, 0, 0], up=[0, 0, 1]),
                        "film": {"type": "hdrfilm", "width": W, "height": H, "pixel_format": "rgba", "rfilter": {"type": "gaussian"}},
                        "sampler": {"type": "multijitter", "sample_count": max(256, a.spp)}},
             "sun": {"type": "directional", "direction": list(-geo["sun"]), "irradiance": {"type": "rgb", "value": math.pi}},
             "sky": {"type": "envmap", "bitmap": mi.Bitmap(env), "to_world": T().rotate([1, 0, 0], 90)}}
    scene.update(c["shapes"])
    objl = np.array(mi.render(mi.load_dict(scene)), dtype=np.float32)
    # кадры
    n = int(dur * a.fps); enc = Enc(Path(a.out) / f"{a.name}_v5_flyaround.mp4", W, H, a.fps)
    nu, nv = W // 8, H // 8; Lref = None
    for i in range(n):
        t = i / a.fps
        rC, d_e, lam, P = ground_pts(t, cam_b, nu, nv)
        ok = np.isfinite(lam); P = np.where(ok[..., None], P, rC)
        X, Y = G.ecef_to_utm(P.reshape(-1, 3), epsg); X = np.ravel(X).reshape(nv, nu); Y = np.ravel(Y).reshape(nv, nu)
        lo, la = G.ecef_to_lonlat(P.reshape(-1, 3)); ng = NS.normal_ecef(np.asarray(lo), np.asarray(la)).reshape(nv, nu, 3)
        mus = np.clip(ng @ sun_e, 0.02, 1); muv = np.clip(np.sum(ng * -d_e, -1), 0.05, 1); cosT = -(np.sum(-d_e * sun_e, -1))
        upz = lambda q: zoom(q, (H / nv, W / nu), order=1)[:H, :W]
        Xf, Yf, musf, muvf, cosTf, okf = upz(X), upz(Y), upz(mus), upz(muv), upz(cosT), upz(ok.astype(np.float32)) > 0.5
        cc = (Xf - bb[0]) / res - 0.5; rr = (bb[3] - Yf) / res - 0.5
        rh = np.stack([map_coordinates(rho[..., ch], [rr, cc], order=1, mode="nearest") for ch in range(3)], -1)
        tau = NS.TAU_R[None, None] + NS.TAU_A; m_ = 1 / musf[..., None] + 1 / muvf[..., None]
        PR = 0.75 * (1 + cosTf ** 2)
        path = (NS.TAU_R[None, None] * PR[..., None] + NS.TAU_A * 1.5) / (4 * musf[..., None] * muvf[..., None]) * (1 - np.exp(-tau * m_)) / (tau * m_)
        Lg = (rh * np.exp(-tau * m_ * 0.55) + path) * musf[..., None]
        Lg[~okf] = 0.0
        comp = objl[..., :3] + Lg * (1 - objl[..., 3:4])
        if Lref is None:
            Lref = Lg[okf].reshape(-1, 3)
        img = Image.fromarray(NS.camera(comp, Lref, seed=i))
        dr = header(img, 5, "Облёт в стиле фото NASA — самое понятное",
                    f"виртуальная камера в {dist:.0f} м летит вместе с объектом · Земля и облака — та же съёмка Sentinel-2 · реальное время · рендер",
                    f"{t:4.1f} с")
        enc.put(img)
    enc.close()
    return dict(dist_m=dist, fov=fov, az=az, coverage=cov, earth_res=res, frames=n)


def main():
    t0 = time.time(); Path(a.out).mkdir(parents=True, exist_ok=True)
    c = setup(); meta = {}
    only = a.only.split(",")
    for key, fn in (("1", v1), ("2", v2), ("4", v4), ("5", v5)):
        if key in only:
            t1 = time.time(); meta[key] = fn(c)
            print(f"  вариант {key}: {time.time() - t1:.0f} с {meta[key]}", flush=True)
    (Path(a.out) / f"{a.name}_variants.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    import shutil
    shutil.rmtree(c["work"], ignore_errors=True)
    print(f"готово за {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()

"""Видео пролёта и фото «в реальном масштабе и повороте» (THEORY §23): кадровая камера в точке, где был Sentinel-2,
неподвижная относительно земли, север вверх, без растяжения push-broom. Объект — детальная модель в той же позе и при
том же Солнце, что в realistic_view.py; земля — та же сцена Sentinel-2 (пятна объекта убраны).

Часть 1 — «как Sentinel-2 снимал пролёт»: общий план, объект летит по своему следу на земле (скорость |v_G|), в момент
съёмки каждой полосы в точке, где эта полоса его поймала, вспыхивает метка цвета полосы — отсюда разноцветные пятна
на снимке. Часть 2 — «пролёт вблизи»: та же неподвижная камера крупнее. Обе части замедлены (коэффициент — на кадре).
Масштаб: объект ближе к камере, чем земля, поэтому на земле он выглядит в λ/R раз крупнее своего размера — на кадре
две линейки: метры на земле и метры на объекте.

  python scripts/flyby_video.py ISS_20261118T222808
"""
import argparse, glob, json, math, pickle, subprocess, sys, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, map_coordinates, shift as nd_shift, zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--W", type=int, default=1920); ap.add_argument("--H", type=int, default=1080)
ap.add_argument("--fps", type=int, default=30); ap.add_argument("--dur1", type=float, default=10.0); ap.add_argument("--dur2", type=float, default=8.0)
ap.add_argument("--spp", type=int, default=256); ap.add_argument("--tau", type=float, default=1 / 4000)
ap.add_argument("--work", default="/tmp/realistic_work"); ap.add_argument("--out", default=str(ROOT / "results/flyby"))
a = ap.parse_args()
sys.argv = [sys.argv[0], a.name]
import realistic_view as RV
import hd_models as HDM
from s2sat import imagery as I
from s2sat.orbits import dt_from_jd

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BANDC = {"blue": (77, 163, 255), "green": (61, 220, 132), "red": (255, 90, 77), "rededge1": (255, 140, 60), "rededge2": (255, 170, 70),
         "rededge3": (255, 200, 80), "nir": (255, 216, 77), "nir08": (240, 230, 120), "swir16": (200, 120, 255), "swir22": (170, 90, 230)}
BANDN = {"blue": "B02 синяя", "green": "B03 зелёная", "red": "B04 красная", "rededge1": "B05", "rededge2": "B06", "rededge3": "B07",
         "nir": "B08 ИК", "nir08": "B8A", "swir16": "B11", "swir22": "B12"}


def ground_layer(its, epsg, rc, bb, res):
    """Земля L2A на сетке res (≥10 м) с убранными пятнами объекта; яркость в единицах объекта."""
    import physdet as PD
    IM = PD.read_band_region(its, epsg, bb, ["red", "green", "blue"], 10.0)
    for b in ("red", "green", "blue"):
        d = rc["bands"].get(b)
        if d is None or b not in IM:
            continue
        msk = binary_dilation(np.abs(d["r"]) > 4 * np.asarray(d["s"]), iterations=3)
        cols = np.round((d["X"] - bb[0] - 5) / 10).astype(int); rows_ = np.round((bb[3] - 5 - d["Y"]) / 10).astype(int)
        ok = msk & (cols >= 0) & (cols < IM[b].shape[1]) & (rows_ >= 0) & (rows_ < IM[b].shape[0])
        IM[b][rows_[ok], cols[ok]] = d["g"][ok] + 1000.0
    refl = np.stack([IM["red"], IM["green"], IM["blue"]], -1) / 10000.0 - 0.1
    if res > 10:
        f = int(round(res / 10)); h, w = refl.shape[0] // f * f, refl.shape[1] // f * f
        refl = refl[:h, :w].reshape(h // f, f, w // f, f, 3).mean((1, 3))
    return refl


def tone(x, expo):
    return RV.srgb(RV.aces(np.clip(x * expo, 0, None)))


def make_sprite(obj, npx, pix, A2, rg, vG, tau):
    """Рендер объекта (квадрат npx, pix м на объекте) → спрайт в пикселях кадра (север вверх, rg м на земле), со смазом."""
    ext_g = npx * pix * np.abs(A2).sum(1).max() / 2
    n = int(math.ceil(2 * ext_g / rg)) + 6
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    gE = (jj + 0.5 - n / 2) * rg; gN = (n / 2 - ii - 0.5) * rg
    q = np.linalg.solve(A2, np.stack([gE.ravel(), gN.ravel()]))
    cc = q[0] / pix + npx / 2 - 0.5; rr = npx / 2 - q[1] / pix - 0.5
    if pix < rg / 2.5:                                       # уменьшение: сгладить до выборки
        from scipy.ndimage import gaussian_filter
        s = 0.5 * rg / pix / np.mean(np.linalg.svd(A2)[1])
        obj = np.stack([gaussian_filter(obj[..., c], s) for c in range(4)], -1)
    sp = np.stack([map_coordinates(obj[..., c], [rr, cc], order=1, mode="constant").reshape(n, n) for c in range(4)], -1)
    return RV.blur(sp, RV.line_kernel(vG / rg * tau))


def paste(frame_lin, sp, cx, cy):
    """Наложить спрайт (предумноженный RGB + альфа) центром в (cx, cy) с субпиксельным сдвигом; вернуть окно."""
    n = sp.shape[0]; x0 = cx - n / 2; y0 = cy - n / 2
    ix, iy = int(math.floor(x0)), int(math.floor(y0)); fx, fy = x0 - ix, y0 - iy
    s = np.stack([nd_shift(sp[..., c], (fy, fx), order=1, mode="constant") for c in range(4)], -1)
    H, W = frame_lin.shape[:2]
    a0, b0 = max(0, iy), max(0, ix); a1, b1 = min(H, iy + n), min(W, ix + n)
    if a1 <= a0 or b1 <= b0:
        return None
    sub = s[a0 - iy:a1 - iy, b0 - ix:b1 - ix]
    win = frame_lin[a0:a1, b0:b1]
    return (a0, a1, b0, b1, sub[..., :3] + win * (1 - sub[..., 3:4]))


def overlay(img, title, sub, t_ms, slow, rg, k, scale_g, scale_o):
    dr = ImageDraw.Draw(img, "RGBA")
    f1, f2, f3 = ImageFont.truetype(FONTB, 34), ImageFont.truetype(FONT, 24), ImageFont.truetype(FONT, 22)
    dr.rectangle([0, 0, img.width, 96], fill=(0, 0, 0, 110))
    dr.text((28, 14), title, font=f1, fill=(255, 255, 255, 235)); dr.text((28, 58), sub, font=f2, fill=(220, 225, 230, 220))
    tc = f"t = {t_ms:+.1f} мс · ×{slow:.0f}"
    dr.text((img.width - 28 - dr.textlength(tc, font=f1), 14), tc, font=f1, fill=(255, 255, 255, 235))
    # линейки: земля и объект (объект в λ/R раз крупнее на земле)
    y = img.height - 70; x = 30
    for L, lab, col in ((scale_g, f"{L_fmt(scale_g)} на земле", (255, 255, 255)), (scale_o, f"{L_fmt(scale_o)} на объекте (×{k:.2f} — он ближе к камере)", (255, 220, 120))):
        px = L / rg * (k if col != (255, 255, 255) else 1.0)
        dr.rectangle([x - 8, y - 30, x + max(px, dr.textlength(lab, font=f3)) + 8, y + 12], fill=(0, 0, 0, 120))
        dr.line([x, y, x + px, y], fill=col + (255,), width=5); dr.line([x, y - 8, x, y + 8], fill=col + (255,), width=3)
        dr.line([x + px, y - 8, x + px, y + 8], fill=col + (255,), width=3)
        dr.text((x, y - 30), lab, font=f3, fill=col + (255,))
        y -= 58
    # север
    cx, cy = img.width - 60, img.height - 90
    dr.polygon([(cx, cy - 34), (cx - 12, cy), (cx + 12, cy)], fill=(255, 255, 255, 230)); dr.text((cx - 8, cy + 4), "С", font=f2, fill=(255, 255, 255, 230))
    return img


def L_fmt(m):
    return f"{m / 1000:g} км" if m >= 1000 else f"{m:g} м"


def nice(x):
    for v in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000):
        if v >= x:
            return v
    return x


def main():
    t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    work = Path(a.work) / f"{a.name}_flyby"; work.mkdir(parents=True, exist_ok=True)
    evf = sorted(glob.glob(str(ROOT / f"results/found*/{a.name}.json"))) + sorted(glob.glob(str(ROOT / f"results/events_*/{a.name}.json")))
    x = json.loads(Path(evf[0]).read_text()); e, r = x["event"], x["result"]
    rc = pickle.loads((ROOT / "results/shape" / f"{a.name}.pkl").read_bytes()); epsg = int(r["epsg"])
    sen, orb = RV.choose_orbits(e, rc, None)
    band0 = next((b for b in ("green", "red", "blue", "nir") if b in rc["bands"]), next(iter(rc["bands"])))
    db = rc["bands"][band0]; jd0 = float(db["jd"]); anchor = np.asarray(db["meas"][:2] if db.get("meas") else db["p0"], float)
    geo = RV.frame_geometry(sen, orb, jd0, epsg); sun, los = geo["sun"], geo["los"]; vG = geo["vG"]
    parts, note, smooth = HDM.parts_for(a.name, sun, int(e.get("cat", 0)))
    title = e.get("obj") or a.name.split("_")[0]
    Vb = HDM.all_vertices(parts); shapes = HDM.mitsuba_shapes(parts, work / "hd", smooth=smooth)
    right, up, A2, S = RV.camera_basis(geo["Jf"], los); k = float(S.mean())
    proj = Vb - np.outer(Vb @ los, los); ext = float(np.max(np.abs(np.c_[proj @ right, proj @ up]))) + 2.0
    span_g = 2 * ext * k                                       # размер объекта на земле, м
    # моменты и места съёмки полос
    bands = []
    for b, d in rc["bands"].items():
        p = np.asarray(d["meas"][:2] if d.get("meas") else d["p0"], float)
        bands.append((b, (float(d["jd"]) - jd0) * 86400.0, p))
    bands.sort(key=lambda t: t[1])
    pred = [np.linalg.norm(p - (anchor + vG * tb)) for _, tb, p in bands]
    print(f"{a.name}: λ/R {k:.2f}, след {np.linalg.norm(vG) / 1000:.1f} км/с, азимут следа {math.degrees(math.atan2(vG[0], vG[1])) % 360:.0f}°, "
          f"полос {len(bands)}; расхождение точек полос с прямой: медиана {np.median(pred):.0f} м", flush=True)
    its = I.items_around(e["lon"], e["lat"], dt_from_jd(e["jd"]), e["sent"][-1], d=1.0); its.sort(key=lambda it: it.id != r.get("item"))
    path, trans = RV.rayleigh(e["sun_el"]); mus = math.sin(math.radians(e["sun_el"]))
    toL = lambda refl: (np.clip(refl, 0, 1.5) * trans + path) * mus
    import mitsuba as mi
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.W}x{a.H}", "-r", str(a.fps),
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", str(out / f"{a.name}_flyby.mp4")],
                          stdin=subprocess.PIPE)
    # ---------- часть 1: общий план, все полосы
    P = np.array([p for _, _, p in bands]); cen = (P.min(0) + P.max(0)) / 2
    rg1 = float(nice(max(np.ptp(P[:, 0]) / (0.72 * a.W), np.ptp(P[:, 1]) / (0.62 * a.H), span_g / (0.012 * a.W), 5.0)))
    hx, hy = a.W * rg1 / 2, a.H * rg1 / 2
    bb1 = tuple(int(v) for v in (math.floor((cen[0] - hx - 200) / 20) * 20, math.floor((cen[1] - hy - 200) / 20) * 20,
                                 math.ceil((cen[0] + hx + 200) / 20) * 20, math.ceil((cen[1] + hy + 200) / 20) * 20))
    rho1 = ground_layer(its, epsg, rc, bb1, 10.0); L1 = toL(rho1)
    expo = float(np.clip(0.18 / np.median(L1 @ np.array([0.2126, 0.7152, 0.0722])), 1.2, 8.0))
    jj, ii = np.meshgrid(np.arange(a.W), np.arange(a.H))
    gx = (cen[0] + (jj + 0.5 - a.W / 2) * rg1 - bb1[0]) / 10 - 0.5; gy = (bb1[3] - cen[1] - (a.H / 2 - ii - 0.5) * rg1) / 10 - 0.5
    from scipy.ndimage import gaussian_filter
    sm = max(0.0, 0.45 * rg1 / 10)
    G1 = np.stack([map_coordinates(gaussian_filter(L1[..., c], sm) if sm > 0.3 else L1[..., c], [gy, gx], order=1, mode="nearest") for c in range(3)], -1)
    G1t = (tone(G1, expo) * 255 + 0.5).astype(np.uint8)
    pix1 = rg1 / k / 3; npx1 = int(math.ceil(2 * ext / pix1)) + 8
    obj1 = RV.render_object(Vb, None, None, sun, los, right, up, pix1, npx1, toL(rho1).reshape(-1, 3).mean(0), geo["alt"], 128, work / "r1", shapes=shapes)
    sp1 = make_sprite(obj1, npx1, pix1, A2, rg1, vG, a.tau)
    ta, tb_ = bands[0][1] - 0.15, bands[-1][1] + 0.15; n1 = int(a.dur1 * a.fps); slow1 = a.dur1 / (tb_ - ta)
    to_px = lambda p: ((p[0] - cen[0]) / rg1 + a.W / 2, a.H / 2 - (p[1] - cen[1]) / rg1)
    sg1 = nice(a.W * rg1 / 8); so1 = nice(span_g / k / 2)
    f3 = ImageFont.truetype(FONT, 22)
    for i in range(n1):
        t = ta + (tb_ - ta) * i / (n1 - 1)
        frame = G1t.copy(); pos = anchor + vG * t; cx, cy = to_px(pos)
        res = paste(G1, sp1, cx, cy)
        if res:
            a0, a1, b0, b1, win = res; frame[a0:a1, b0:b1] = (tone(win, expo) * 255 + 0.5).astype(np.uint8)
        img = Image.fromarray(frame); dr = ImageDraw.Draw(img, "RGBA")
        dr.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], outline=(255, 255, 255, 230), width=3)
        dr.text((cx + 34, cy + 18), title, font=f3, fill=(255, 255, 255, 235))
        p0, p1 = to_px(anchor + vG * ta), to_px(anchor + vG * tb_)
        dr.line([p0, p1], fill=(255, 255, 255, 50), width=2)
        for b, tb, p in bands:
            if t >= tb:
                bx, by = to_px(p); c = BANDC.get(b, (255, 255, 255)); age = (t - tb) * slow1
                rad = 14 + 40 * max(0.0, 1 - age / 0.6)
                dr.ellipse([bx - rad, by - rad, bx + rad, by + rad], outline=c + (int(255 * max(0.35, 1 - age / 0.6)),), width=4)
                dr.text((bx + 18, by - 12), BANDN.get(b, b), font=f3, fill=c + (230,))
        img = overlay(img, f"3/5 · Кадровая камера Sentinel-2 без искажений", "реальный масштаб и поворот, север вверху · кольца — где каждая полоса поймала объект",
                      (t) * 1000, slow1, rg1, k, sg1, so1)
        ff.stdin.write(np.asarray(img.convert("RGB")).tobytes())
    print(f"  часть 1: {rg1:g} м/пкс, {tb_ - ta:.2f} с за {a.dur1:.0f} с (×{slow1:.0f}), {time.time() - t0:.0f} с", flush=True)
    # ---------- часть 2: пролёт вблизи (объект ~25 % ширины)
    rg2 = float(min(2.5, max(0.5, round(span_g / (0.25 * a.W) * 4) / 4)))
    hx, hy = a.W * rg2 / 2, a.H * rg2 / 2
    bb2 = tuple(int(v) for v in (math.floor((anchor[0] - hx - 300) / 10) * 10, math.floor((anchor[1] - hy - 300) / 10) * 10,
                                 math.ceil((anchor[0] + hx + 300) / 10) * 10, math.ceil((anchor[1] + hy + 300) / 10) * 10))
    rho2 = ground_layer(its, epsg, rc, bb2, 10.0); L2 = toL(rho2)
    Lsr = RV.super_res(L2, float(np.percentile(L2, 99.9) * 1.25))
    gx = (anchor[0] + (jj + 0.5 - a.W / 2) * rg2 - bb2[0]) / 2.5 - 0.5; gy = (bb2[3] - anchor[1] - (a.H / 2 - ii - 0.5) * rg2) / 2.5 - 0.5
    G2 = np.stack([map_coordinates(Lsr[..., c], [gy, gx], order=3, mode="nearest") for c in range(3)], -1)
    G2t = (tone(G2, expo) * 255 + 0.5).astype(np.uint8)
    pix2 = rg2 / k; npx2 = int(math.ceil(2 * ext / pix2)) + 8
    obj2 = RV.render_object(Vb, None, None, sun, los, right, up, pix2, npx2, L2.reshape(-1, 3).mean(0), geo["alt"], a.spp, work / "r2", shapes=shapes)
    sp2 = make_sprite(obj2, npx2, pix2, A2, rg2, vG, a.tau)
    T2 = (a.W * rg2 + span_g) / np.linalg.norm(vG) * 1.05; n2 = int(a.dur2 * a.fps); slow2 = a.dur2 / T2
    sg2 = nice(a.W * rg2 / 8); so2 = nice(span_g / k / 3)
    to_px2 = lambda p: ((p[0] - anchor[0]) / rg2 + a.W / 2, a.H / 2 - (p[1] - anchor[1]) / rg2)
    for i in range(n2):
        t = -T2 / 2 + T2 * i / (n2 - 1)
        frame = G2t.copy(); cx, cy = to_px2(anchor + vG * t)
        res = paste(G2, sp2, cx, cy)
        if res:
            a0, a1, b0, b1, win = res; frame[a0:a1, b0:b1] = (tone(win, expo) * 255 + 0.5).astype(np.uint8)
        img = overlay(Image.fromarray(frame), f"3/5 · Та же камера крупнее", f"{rg2:g} м/пкс на земле · {note}",
                      t * 1000, slow2, rg2, k, sg2, so2)
        ff.stdin.write(np.asarray(img.convert("RGB")).tobytes())
        if abs(t) <= T2 / (n2 - 1) / 2 + 1e-9 or i == n2 // 2:
            img.save(out / f"{a.name}_photo_annot.jpg", quality=93)
            Image.fromarray(frame).save(out / f"{a.name}_photo.jpg", quality=93)
    ff.stdin.close(); ff.wait()
    meta = dict(name=a.name, k=k, vG_kms=float(np.linalg.norm(vG) / 1000), part1=dict(m_per_px=rg1, real_s=tb_ - ta, slow=slow1),
                part2=dict(m_per_px=rg2, real_s=T2, slow=slow2), bands=[dict(band=b, t_s=tb, E=float(p[0]), N=float(p[1])) for b, tb, p in bands],
                band_line_resid_m=[float(v) for v in pred], note=note)
    (out / f"{a.name}_flyby.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    print(f"  часть 2: {rg2:g} м/пкс, {T2 * 1000:.0f} мс за {a.dur2:.0f} с (×{slow2:.0f}); видео и фото — {out}, всего {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()

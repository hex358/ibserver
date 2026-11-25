"""Косой ракурс той же модели в той же позе и при том же Солнце, что в realistic_view.py, — чтобы был виден объём и
собственные тени (с Sentinel-2 объект виден почти строго сверху, 1–9° от зенита, и выглядит плоским).

  python scripts/oblique_view.py STARLINK-31550_20261119T181632 --zen 50
"""
import argparse, glob, json, math, pickle, sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--zen", type=float, default=50.0, help="угол камеры от зенита объекта")
ap.add_argument("--az", type=float, default=90.0, help="азимут камеры от азимута Солнца, °")
ap.add_argument("--W", type=int, default=1920); ap.add_argument("--H", type=int, default=1080); ap.add_argument("--spp", type=int, default=256)
ap.add_argument("--work", default="/tmp/realistic_work"); ap.add_argument("--out", default=str(ROOT / "results/realistic"))
a = ap.parse_args()
sys.argv = [sys.argv[0], a.name]
import realistic_view as RV


def main():
    rc = pickle.loads((ROOT / "results/shape" / f"{a.name}.pkl").read_bytes())
    meta = json.loads((ROOT / "results/realistic" / f"{a.name}.json").read_text())
    sun = np.asarray(meta["sun_body"]); expo = meta["exposure_gain"]; earth = meta["earth_rgb"]
    import hd_models as HDM
    parts, _, smooth = HDM.parts_for(a.name, sun, int(rc.get("cat", 0)))
    Vb = HDM.all_vertices(parts); F = MAT = None; alt = 786.0 - rc["rng"]
    shapes = HDM.mitsuba_shapes(parts, Path(a.work) / (a.name + "_obl") / "hd", smooth=smooth)
    # камера: от зенита на zen, по азимуту (Солнце + az) в горизонтальной плоскости (вдоль, поперёк)
    saz = math.atan2(sun[1], sun[0]) + math.radians(a.az)
    h = np.array([math.cos(saz), math.sin(saz), 0.0]); zr = np.array([0.0, 0.0, 1.0])
    c = math.cos(math.radians(a.zen)) * zr + math.sin(math.radians(a.zen)) * h
    up = zr - (zr @ c) * c; up /= np.linalg.norm(up); right = np.cross(-c, up)
    proj = Vb - np.outer(Vb @ c, c)
    ext = float(np.max(np.abs(np.c_[proj @ right, proj @ up]))) * 1.25 + 1.0
    pix = 2 * ext / a.H
    import mitsuba as mi
    obj = RV.render_object(Vb, F, MAT, sun, c, right, up, pix, a.W, earth, alt, a.spp, Path(a.work) / (a.name + "_obl"), shapes=shapes)
    # render_object делает квадратный кадр W×W — берём центральную полосу H
    i0 = (a.W - a.H) // 2; obj = obj[i0:i0 + a.H]
    cache = sorted(glob.glob(str(Path(a.work) / f"sr_{a.name}_*.npy")))
    if cache:
        G = np.load(cache[0]); hh, ww = G.shape[:2]
        crop = G[hh // 2 - 270:hh // 2 + 270, ww // 2 - 480:ww // 2 + 480]
        bg = zoom(crop, (a.H / crop.shape[0], a.W / crop.shape[1], 1), order=1)
        bg = np.stack([gaussian_filter(bg[..., k], 6) for k in range(3)], -1)
    else:                                                  # снимка Земли нет — чёрный космос (а не ровная «земля»)
        bg = np.zeros((a.H, a.W, 3))
    comp = obj[..., :3] + bg * (1 - obj[..., 3:4])
    img = RV.srgb(RV.aces(comp * expo))
    from PIL import Image
    fn = Path(a.out) / f"{a.name}_oblique.png"
    Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(fn)
    print("OK", fn, f"камера {a.zen}° от зенита, {pix * 100:.1f} см/пкс")


if __name__ == "__main__":
    main()

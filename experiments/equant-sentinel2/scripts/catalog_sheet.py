"""Контактный лист каталога 3D-моделей (THEORY §24): каждая модель — в штатной позе своего класса (на Землю, панели на
Солнце; ступень — наклонена), Солнце 50° над горизонтом, вид сбоку-сверху, ортографическая камера, Mitsuba.

  python scripts/catalog_sheet.py [--only KEY,KEY] [--px 512] [--spp 64]
Пишет results/catalog/<ключ>.png и results/catalog/catalog_sheet.json (источник, треугольники, размеры, крылья, кому
назначена модель: номера NORAD, шины GCAT).
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
ap = argparse.ArgumentParser()
ap.add_argument("--only", default=""); ap.add_argument("--skip", default="ISS-(ZARYA)"); ap.add_argument("--px", type=int, default=512); ap.add_argument("--spp", type=int, default=64)
ap.add_argument("--work", default="/tmp/catalog_work"); ap.add_argument("--out", default=str(ROOT / "results/catalog"))
a = ap.parse_args()
sys.argv = [sys.argv[0], "CATALOG"]
import hd_models as HDM
import realistic_view as RV
from s2sat import autopose as AP, modelcat as MC


def main():
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    js = out / "catalog_sheet.json"
    rows = json.loads(js.read_text()) if js.exists() else {}
    only = set(a.only.split(",")) if a.only else None
    sun = np.array([math.cos(math.radians(50)) * math.cos(math.radians(30)), math.cos(math.radians(50)) * math.sin(math.radians(30)),
                    math.sin(math.radians(50))])
    G = MC._gcat()
    for e in MC._manifest():
        k = e["key"]
        if (only and k not in only) or e.get("error") or (not only and k in a.skip.split(",")):
            continue
        if not only and (out / f"{k}.png").exists() and k in rows:
            continue
        t0 = time.time()
        g = None
        for n in e["match"].get("norad") or []:
            g = G.get(int(n))
            if g:
                break
        if g is None and e["match"].get("bus"):
            bl = [b.lower() for b in e["match"]["bus"]]
            g = next((r for r in G.values() if (r.get("Bus") or "").lower() in bl), None)
        try:
            M = MC.load_catalog_model(e, g, fit=False)
        except Exception as ex:                                     # noqa: BLE001
            print("ОШИБКА", k, ex); continue
        spec = dict(V=M["V"], F=M["F"], G=M["G"], M2B=M["M2B"], arrays=M["arrays"])
        if e["kind"] == "rocketbody":
            law, x = dict(name="t", tumble=True), np.array([0.5, 0.35, 0, 0, 0, 0, 0])
        else:
            law, x = dict(name="t", yaw=0, arrays="sun"), np.zeros(7)
        Vb, _, _ = AP.pose_body(spec, law, x, sun)
        parts = []
        for i, p in enumerate(M["parts"]):
            u, off = M["gidx"][i]
            V = np.array(p["V"], float); V[u] = Vb[off:off + len(u)]
            parts.append(dict(p, V=V))
        # камера: 55° от зенита, азимут Солнца + 70°
        saz = math.atan2(sun[1], sun[0]) + math.radians(70)
        h = np.array([math.cos(saz), math.sin(saz), 0.0]); zr = np.array([0.0, 0, 1])
        c = math.cos(math.radians(55)) * zr + math.sin(math.radians(55)) * h
        up = zr - (zr @ c) * c; up /= np.linalg.norm(up); right = np.cross(-c, up)
        allV = np.vstack([p["V"][np.unique(p["F"])] for p in parts]); cen = 0.5 * (allV.min(0) + allV.max(0))
        for p in parts:
            p["V"] = p["V"] - cen
        allV = allV - cen
        proj = allV - np.outer(allV @ c, c)
        ext = float(np.max(np.abs(np.c_[proj @ right, proj @ up]))) * 1.12 + 1e-3
        pix = 2 * ext / a.px
        work = Path(a.work) / k
        shapes = HDM.mitsuba_shapes(parts, work / "hd", smooth=e["frame"] != "satvis")
        img = RV.render_object(None, None, None, sun, c, right, up, pix, a.px, (0.05, 0.06, 0.08), 600.0, a.spp, work, shapes=shapes)
        bg = np.array([0.006, 0.007, 0.010])
        lum = img[..., :3].mean(-1)[img[..., 3] > 0.5]
        expo = min(10.0, 0.55 / max(1e-4, float(np.percentile(lum, 90)))) if lum.size else 1.0
        rgb = img[..., :3] * expo + bg * (1 - img[..., 3:4])
        from PIL import Image
        Image.fromarray((RV.srgb(RV.aces(rgb)) * 255 + 0.5).astype(np.uint8)).save(out / f"{k}.png")
        nb = sum(1 for r in G.values() if (r.get("Bus") or "").lower() in [b.lower() for b in e["match"].get("bus") or []])
        rows[k] = dict(key=k, source=e["source"], license=e["license"], frame=e["frame"], kind=e["kind"], tri=M["tri"], dims_m=M["dims_m"],
                       wings=M["n_arrays"], fixed_panels=M["n_fixed"], norad=e["match"].get("norad") or [], bus=e["match"].get("bus") or [],
                       name_re=e["match"].get("name"), gcat_bus_objects=nb, scale_m_per_px=round(pix, 4), seconds=round(time.time() - t0, 1))
        js.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"{k:28s} {M['tri']:9d} тр.  крыльев {M['n_arrays']}  {pix * 100:.1f} см/пкс  {time.time() - t0:.0f} с", flush=True)
        import shutil
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

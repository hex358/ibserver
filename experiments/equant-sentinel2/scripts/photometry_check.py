"""Реализм рендера числом: интегральный контраст объекта на снимке Sentinel-2 — рендер против измерения.

  python scripts/photometry_check.py ISS_20261118T222808 [--spp 64]
Рендер (тот же, что в realistic_view: Солнце с облучённостью π — яркость в единицах отражательной способности, свет
Земли, материалы по классам, поза из подгонки) — ортогональная камера вдоль визира Sentinel-2. Интегральный контраст
в канале c: ΔI = Σ (L_c − α·ρ_фон,c) · (площадь пикселя на земле), м² — не зависит от функции рассеяния и смаза.
Измерение: ΔI = Σ r_b · (10 м)² по окну объекта в полосах B04/B03/B02 (r — остаток снимка без фона, отражение).
Отношение рендер/измерение ≈ 1 — свет и материалы реалистичны.
"""
import json, math, pickle, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
name = sys.argv[1]; spp = int(sys.argv[sys.argv.index("--spp") + 1]) if "--spp" in sys.argv else 64
sys.argv = [sys.argv[0], name]
import realistic_view as RV                      # noqa: E402  (разбор аргументов внутри модуля)
import hd_models as HDM                          # noqa: E402
from s2sat import scenegeom as SG                # noqa: E402

e, r = SG.event_of(name)
rc = pickle.loads((ROOT / "results/shape" / f"{name}.pkl").read_bytes())
sen, orb = SG.choose_orbits(e, rc)
band = "green"; db = rc["bands"][band]; jd = float(db["jd"])
geo = SG.frame_geometry(sen, orb, jd, int(r["epsg"]))
sun, los = geo["sun"], geo["los"]
parts, note, smooth = HDM.parts_for(name, sun, int(e.get("cat", 0)))
work = Path("/tmp/photcheck") / name; shapes = HDM.mitsuba_shapes(parts, work / "hd", smooth=smooth)
Vb = HDM.all_vertices(parts)
right, up, A2, S = RV.camera_basis(geo["Jf"], los)
proj = Vb - np.outer(Vb @ los, los); ext = float(np.max(np.abs(np.c_[proj @ right, proj @ up]))) + 1.0
pix = max(0.05, 2 * ext / 640); npx = int(math.ceil(2 * ext / pix))
bg = {c: float(np.nanmedian(rc["bands"][b]["g"])) / 1e4 for c, b in (("R", "red"), ("G", "green"), ("B", "blue")) if b in rc["bands"]}
earth = np.array([bg.get("R", 0.2), bg.get("G", 0.2), bg.get("B", 0.2)])
img = RV.render_object(Vb, None, None, sun, los, right, up, pix, npx, earth, geo["alt"], spp, work, shapes=shapes)
rgb, alpha = img[..., :3], img[..., 3]
area_g = pix * pix * abs(np.linalg.det(A2))                     # м² на земле на пиксель кадра
X = {c: float(np.sum(rgb[..., i]) * area_g) for i, c in enumerate("RGB")}          # яркость объекта
Y = {c: float(np.sum(alpha) * earth[i] * area_g) for i, c in enumerate("RGB")}        # заслонённый фон
pred = {c: X[c] - Y[c] for c in "RGB"}
obs = {}
for c, b in (("R", "red"), ("G", "green"), ("B", "blue")):
    if b in rc["bands"]:
        d = rc["bands"][b]; ok = np.isfinite(d["r"])
        obs[c] = float(np.sum(d["r"][ok]) / 1e4 * d["res"] ** 2)
print(f"{name}: {note}")
print(f"  фон (отражение) R {earth[0]:.3f} G {earth[1]:.3f} B {earth[2]:.3f}; кадр {npx}×{npx}, {pix:.2f} м/пкс, на земле ×{abs(np.linalg.det(A2)):.2f}")
rows = {}
for c in "RGB":
    if c in obs:
        ratio = pred[c] / obs[c] if abs(obs[c]) > 1e-9 else float("nan")
        scale = float(np.clip((obs[c] + Y[c]) / X[c], 0.1, 10.0)) if X[c] > 1e-9 else 1.0   # измеренная поправка яркости
        rows[c] = dict(render_m2=round(pred[c], 1), measured_m2=round(obs[c], 1), ratio=round(ratio, 2), scale=round(scale, 3))
        print(f"  {c}: интегральный контраст — рендер {pred[c]:+8.1f} м², снимок {obs[c]:+8.1f} м², отношение {ratio:+.2f}; "
              f"поправка яркости объекта ×{rows[c]['scale']:.2f}")
out = ROOT / "results/realistic" / f"photometry_{name}.json"
out.write_text(json.dumps(dict(name=name, note=note, background=earth.tolist(), channels=rows), ensure_ascii=False, indent=1))

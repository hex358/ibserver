"""Наложение модели на снимок: сетка в найденной позе проецируется точной камерой каждой полосы (J_b) — в тех же
координатах и с тем же поворотом/растяжением push-broom, что и снимок; контур силуэта рисуется поверх пикселей
Sentinel-2. Плюс вид модели с севером вверх без искажений push-broom (поворот из полярного разложения J).

  python scripts/overlay_mesh.py --mesh data/models/iss.obj --name ISS_20261118T222808 --recon results/recon_iss2 \
      --axes Z,X,Y --panel-axis X --reg-meas --out results/overlay
"""
import argparse, json, math, pickle, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True); ap.add_argument("--name", required=True); ap.add_argument("--recon", required=True)
ap.add_argument("--axes", default="Z,X,Y"); ap.add_argument("--panel-axis", default="X"); ap.add_argument("--reg-meas", action="store_true")
ap.add_argument("--half", type=float, default=220.0); ap.add_argument("--out", default=str(ROOT / "results/overlay"))
a = ap.parse_args()
sys.argv = [sys.argv[0], "--mesh", a.mesh, "--name", a.name, "--axes", a.axes, "--panel-axis", a.panel_axis] + (["--reg-meas"] if a.reg_meas else [])
import model_recon_mesh as G
import model_recon as MR


def shaded_image(Vb, los, sun, J, p, x0, y1, W, H, res, tint):
    """Рендер сетки в координатах снимка полосы (UTM, север вверх) с освещением; фон прозрачный (альфа)."""
    P2 = p[None, :] + Vb @ (J / 1000.0).T
    col = (P2[:, 0] - x0) / res; row = (y1 - P2[:, 1]) / res
    depth = -(Vb @ los)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(img)
    for t in np.argsort(-depth[G.F].mean(1)):
        if not MR.facing(Vb, t, los):
            continue
        i, j, k = G.F[t]
        n = np.cross(Vb[j] - Vb[i], Vb[k] - Vb[i]); n /= (np.linalg.norm(n) + 1e-12)
        lam = max(0.0, float(n @ sun)); h = sun + los; h /= np.linalg.norm(h)
        spec = (max(0.0, float(n @ h)) ** 60) * 0.5
        c = G.KD.get(G.MAT[t], np.array([0.5, 0.5, 0.5])) * (0.15 + 0.85 * lam) + spec
        c = np.clip(c * tint, 0, 1) ** (1 / 2.2) * 255
        dr.polygon([(col[i], row[i]), (col[j], row[j]), (col[k], row[k])], fill=tuple(int(v) for v in c) + (255,))
    return np.asarray(img)


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rc = pickle.loads((ROOT / "results/shape" / f"{a.name}.pkl").read_bytes()); MR.prepare(rc)
    rec = json.loads((ROOT / a.recon / f"{a.name}.json").read_text())
    pose = np.r_[np.radians(rec["pose_deg"]), rec["dtheta_m"]]
    los = np.asarray(rc["los"], float); sun = np.asarray(rc["sun"], float)
    Vb = G.body(pose)
    bands = [b for b in ("blue", "green", "red", "nir") if b in rc["bands"]]
    cols = {"blue": "#4da3ff", "green": "#3ddc84", "red": "#ff5a4d", "nir": "#ffd84d"}
    fig, ax = plt.subplots(2, len(bands) + 1, figsize=(4.2 * (len(bands) + 1), 8.6))
    tint = np.ones(3)
    for i, b in enumerate(bands):
        d = rc["bands"][b]
        base = np.asarray(d["meas"][:2]) if (a.reg_meas and d.get("meas")) else d["p0"]
        p = base + d["J"] @ np.asarray(pose[4:7]) / 1000.0
        # окно ±half вокруг центра пятна; пиксели полосы 10 м
        X, Y = d["X"], d["Y"]; img = d["g"] + d["r"]
        msk = (np.abs(X - p[0]) <= a.half) & (np.abs(Y - p[1]) <= a.half)
        ii = np.where(msk.any(1))[0]; jj = np.where(msk.any(0))[0]
        crop = img[ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1]
        ext = [X[0, jj[0]] - 5 - p[0], X[0, jj[-1]] + 5 - p[0], Y[ii[-1], 0] - 5 - p[1], Y[ii[0], 0] + 5 - p[1]]
        lo, hi = np.nanpercentile(crop, 2), np.nanpercentile(crop, 99.8)
        # силуэт модели на сетке 1 м (координаты относительно центра)
        res = 1.0; x0 = p[0] - a.half; y1 = p[1] + a.half; Wp = Hp = int(2 * a.half / res)
        masks = MR.raster(Vb, los, d["J"], p, x0, y1, Wp, Hp)
        sil = sum(masks) > 0
        sh = shaded_image(Vb, los, sun, d["J"], p, x0, y1, Wp, Hp, res, tint)
        ax[0, i].imshow(crop, cmap="gray", vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
        ax[0, i].contour(np.linspace(-a.half, a.half, Wp), np.linspace(a.half, -a.half, Hp), sil.astype(float), levels=[0.5], colors=[cols[b]], linewidths=1.2)
        ax[0, i].set_title(f"{b}: снимок 10 м + контур модели")
        # снимок, увеличенный до 1 м (ближайший сосед), и модель поверх с прозрачностью
        gx = np.arange(Wp) * res + x0 + res / 2; gy = y1 - np.arange(Hp) * res - res / 2
        GX, GY = np.meshgrid(gx, gy)
        ci = np.clip(np.floor((GX - X[0, 0] + 5) / 10).astype(int), 0, X.shape[1] - 1); ri = np.clip(np.floor((Y[0, 0] + 5 - GY) / 10).astype(int), 0, X.shape[0] - 1)
        up = np.clip((img[ri, ci] - lo) / (hi - lo + 1e-9), 0, 1)
        base_rgb = np.stack([up] * 3, -1)
        al = sh[..., 3:4] / 255.0 * 0.75
        comp = base_rgb * (1 - al) + (sh[..., :3] / 255.0) * al
        ax[1, i].imshow(comp, extent=[-a.half, a.half, -a.half, a.half], interpolation="nearest")
        ax[1, i].set_title(f"{b}: модель (1 м) поверх снимка")
        for r_ in (0, 1):
            ax[r_, i].set_xlim(-a.half, a.half); ax[r_, i].set_ylim(-a.half, a.half)
            ax[r_, i].set_xlabel("восток, м"); ax[r_, i].set_ylabel("север, м")
    # без искажений: север вверх, поворот — из полярного разложения J (горизонтальные оси тела)
    d = rc["bands"]["red" if "red" in rc["bands"] else bands[0]]
    Jh = d["J"][:, :2] / 1000.0; U, S, Vt = np.linalg.svd(Jh); Rt = U @ Vt
    P2 = Vb[:, :2] @ Rt.T; s = 0.25; ext2 = np.abs(P2).max() + 3; Wv = int(2 * ext2 / s)
    colv = (P2[:, 0] + ext2) / s; rowv = (ext2 - P2[:, 1]) / s
    depth = -(Vb @ los)
    im = Image.new("RGB", (Wv, Wv), (11, 13, 18)); dr = ImageDraw.Draw(im)
    for t in np.argsort(-depth[G.F].mean(1)):
        if not MR.facing(Vb, t, los):
            continue
        i_, j_, k_ = G.F[t]
        n = np.cross(Vb[j_] - Vb[i_], Vb[k_] - Vb[i_]); n /= (np.linalg.norm(n) + 1e-12)
        c = G.KD.get(G.MAT[t], np.array([0.5, 0.5, 0.5])) * (0.15 + 0.85 * max(0.0, float(n @ sun)))
        c = np.clip(c, 0, 1) ** (1 / 2.2) * 255
        dr.polygon([(colv[i_], rowv[i_]), (colv[j_], rowv[j_]), (colv[k_], rowv[k_])], fill=tuple(int(v) for v in c))
    ax[0, -1].imshow(np.asarray(im), extent=[-ext2, ext2, -ext2, ext2]); ax[0, -1].set_title("модель без искажений push-broom,\nсевер вверх, 0.25 м")
    ax[0, -1].set_xlabel("восток, м"); ax[0, -1].set_ylabel("север, м")
    ax[1, -1].axis("off")
    ax[1, -1].text(0, 0.95, f"{a.name}\nпоза: {np.round(rec['pose_deg']).tolist()}°\n"
                   f"Δχ² {rec['dchi2_all']:.0f}, на отложенных полосах {rec['dchi2_holdout']:.0f}\n\n"
                   "Верхний ряд — пиксели полосы и контур модели,\nспроецированной камерой этой полосы.\n"
                   "Нижний ряд — та же модель в той же геометрии,\nс разрешением 1 м.\n"
                   "Справа — модель как она есть (без растяжения\nпо времени съёмки строк), север вверх.\n"
                   f"Растяжение снимка вдоль/поперёк: {S[0]:.2f}, {S[1]:.2f} м/м.",
                   va="top", fontsize=10, transform=ax[1, -1].transAxes)
    fig.tight_layout(); fig.savefig(out / f"{a.name}_overlay.png", dpi=90); plt.close(fig)
    print("OK", out / f"{a.name}_overlay.png", "растяжение", S)


if __name__ == "__main__":
    main()

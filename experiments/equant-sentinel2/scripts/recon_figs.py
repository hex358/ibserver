"""Картинки реконструкции по найденной позе (results/recon*/summary.json): исходный снимок 10 м, предсказание модели
в тех же пикселях, остаток, и вид спутника со стороны Sentinel-2 с разрешением 0.25 м — сетка KeepTrack в найденной
позе, освещение — направление на Солнце в момент съёмки (Ламберт по альбедо материалов из .mtl), цвет материалов
подстроен под измеренную яркость в полосах red/green/blue.

  python scripts/recon_figs.py results/recon results/recon_b results/recon_c --out results/recon_figs
"""
import argparse, glob, json, math, pickle, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR

KD = {}
for ln in open(ROOT / "data/models/starlink-v2mini.mtl"):
    if ln.startswith("newmtl"):
        cur = ln.split()[1]
    elif ln.strip().startswith("Kd"):
        KD[cur] = np.array([float(v) for v in ln.split()[1:4]])
# материал каждой грани (как в load_obj)
MAT = []
mat = None
for ln in open(MR.MESH):
    if ln.startswith("usemtl"):
        mat = ln.split()[1]
    elif ln.startswith("f "):
        n = len(ln.split()) - 1
        MAT += [mat] * (n - 2)
MAT = np.array(MAT)


def shaded_view(Vb, los, sun, res=0.25, W=34.0, H=14.0, tint=None):
    los = np.asarray(los, float); sun = np.asarray(sun, float)
    e1 = np.array([1.0, 0, 0]) - los[0] * los; e1 /= np.linalg.norm(e1); e2 = np.cross(los, e1)
    P2 = np.c_[Vb @ e1, Vb @ e2]; Wp, Hp = int(W / res), int(H / res)
    col = P2[:, 0] / res + Wp / 2; row = -P2[:, 1] / res + Hp / 2
    depth = -(Vb @ los)
    img = Image.new("RGB", (Wp, Hp), (10, 12, 16)); dr = ImageDraw.Draw(img)
    for t in np.argsort(-depth[MR.F0].mean(1)):
        if not MR.facing(Vb, t, los):
            continue
        i, j, k = MR.F0[t]
        n = np.cross(Vb[j] - Vb[i], Vb[k] - Vb[i]); n /= (np.linalg.norm(n) + 1e-12)
        lam = max(0.0, float(n @ sun))
        # блик: зеркальное отражение Солнца в сторону Sentinel (покровное стекло ячеек, плёнка шины)
        h = sun + los; h /= np.linalg.norm(h)
        spec = (max(0.0, float(n @ h)) ** 80) * (0.8 if MAT[t] in ("solar_cell", "silver", "white") else 0.2)
        kd = KD.get(MAT[t], np.array([0.3, 0.3, 0.3]))
        c = kd * (0.18 + 0.82 * lam) + spec
        if tint is not None:
            c = c * tint
        c = np.clip(c, 0, 1) ** (1 / 2.2) * 255
        dr.polygon([(col[i], row[i]), (col[j], row[j]), (col[k], row[k])], fill=tuple(int(v) for v in c))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--out", default=str(ROOT / "results/recon_figs"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = {}
    for d in a.dirs:
        f = Path(d) / "summary.json"
        if f.exists():
            for r in json.loads(f.read_text()):
                rows[r["name"]] = r
    recs = {p.stem: pickle.loads(p.read_bytes()) for p in (ROOT / "results/shape").glob("*.pkl")}
    index = []
    for name, r in sorted(rows.items()):
        rc = recs.get(name)
        if rc is None:
            continue
        MR.prepare(rc)
        los = np.asarray(rc["los"], float); sun = np.asarray(rc["sun"], float)
        pose = (math.radians(r["psi_deg"]), math.radians(r["phi_deg"]), *r["dtheta_m"])
        Vb = MR.body_vertices(pose[0], pose[1])
        pick = lambda *bs: next((b for b in bs if b in rc["bands"] and rc["bands"][b]["res"] == 10), None)
        rgbb = [pick("red", "nir", "green"), pick("green", "blue", "nir"), pick("blue", "green", "nir")]
        if any(b is None for b in rgbb):
            rgbb = [b for b in ("rededge3", "rededge2", "rededge1", "nir08", "swir16", "swir22") if b in rc["bands"]][:3]
        if len(rgbb) < 3:
            continue
        data, pred, resid = [], [], []
        for b in rgbb:
            d = rc["bands"][b]
            R = MR.regressors(rc, d, pose, los)
            c = np.array(r["colors"][b])
            occ = np.stack(R, -1)
            model = (occ * c).sum(-1) - d["g"] * occ.sum(-1)
            data.append(d["g"] + d["r"]); pred.append(d["g"] + model); resid.append((d["r"] - model) / d["s"])
        lo = np.nanpercentile(np.stack(data), 1); hi = np.nanpercentile(np.stack(data), 99.5)
        nm = lambda S: np.clip((np.stack(S, -1) - lo) / (hi - lo + 1e-9), 0, 1) ** 0.8
        # оттенок по измеренной яркости объекта в трёх полосах (среднее по материалам)
        cm = np.array([np.mean(r["colors"][b]) for b in rgbb]); tint = 0.6 + 0.8 * cm / (cm.max() + 1e-9)
        view = shaded_view(Vb, los, sun, tint=tint)
        lab = "RGB" if rgbb == ["red", "green", "blue"] else "/".join(rgbb)
        fig, ax = plt.subplots(1, 4, figsize=(19, 4.9), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.9]))
        ax[0].imshow(nm(data), interpolation="nearest"); ax[0].set_title(f"Sentinel-2, 10 м/пкс ({lab})")
        ax[1].imshow(nm(pred), interpolation="nearest"); ax[1].set_title("модель в найденной позе, те же пиксели")
        rr = np.nanmax(np.abs(np.stack(resid)), 0)
        ax[2].imshow(rr, cmap="magma", vmin=0, vmax=6, interpolation="nearest"); ax[2].set_title("остаток, σ (макс. по 3 полосам)")
        ax[3].imshow(view); ax[3].set_title("реконструкция: вид со стороны Sentinel-2, 0.25 м/пкс")
        for x in ax:
            x.set_xticks([]); x.set_yticks([])
        fig.suptitle(f"{name}   поза: рысканье {r['psi_deg']:+.0f}°, крылья {r['phi_deg']:+.0f}°   "
                     f"Δχ² {r['dchi2_all']:.0f};  поза по половине полос предсказывает другую: Δχ² {r['dchi2_holdout']:.0f}", fontsize=11)
        fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=90); plt.close(fig)
        Image.fromarray(view).save(out / f"{name}_view.png")
        index.append(dict(name=name, psi=r["psi_deg"], phi=r["phi_deg"], dchi2=r["dchi2_all"], dchi2_holdout=r["dchi2_holdout"], bands=lab))
        print("FIG", name, flush=True)
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

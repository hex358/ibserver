"""Картинки автоматической позы: снимок ↔ модель в найденной позе ↔ остаток по полосам, вид модели сверху (с Sentinel).

  python scripts/auto_figs.py [имена ...] [--out results/auto_figs]       (без имён — все results/auto/*.json)
Модель и поза — из results/auto/<имя>.json (закон, x), яркости групп материалов — НМНК в каждой полосе, как при выборе
закона. Шкала — в σ шума полосы.
"""
import argparse, json, pickle, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import autopose as AP, modelcat as MC, scenegeom as SG

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="*"); ap.add_argument("--out", default=str(ROOT / "results/auto_figs"))
ORDER = ["blue", "green", "red", "nir", "rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]


def fig(name, out):
    au = json.loads((ROOT / "results/auto" / f"{name}.json").read_text())
    rc = pickle.loads((ROOT / "results/shape" / f"{name}.pkl").read_bytes()); MR.prepare(rc)
    norad = int(rc.get("cat", 0)); spec = MC.spec_for(norad, name)
    C = np.asarray(au["C_lvlh_to_fit"]); law, x = au["law"], np.asarray(au["x"], float)
    P = AP.Problem(rc, spec, C); Vfit = P.body(law, x)
    bands = [b for b in ORDER if b in P.bands][:5]
    f, axs = plt.subplots(len(bands), 3, figsize=(7.2, 2.3 * len(bands) + 1.2))
    axs = np.atleast_2d(axs)
    for i, b in enumerate(bands):
        d = rc["bands"][b]
        occ = AP.regress(d, Vfit, spec["F"], spec["G"], P.ng, P.los, x[4:7])
        _, c = AP.band_loss(d, occ)
        O = np.stack(occ, -1); s = float(d["s"])
        pred = (O @ c - d["g"] * O.sum(-1)) / s; obs = d["r"] / s
        v = np.nanpercentile(np.abs(obs), 99.5) or 1.0
        ext = [d["X"].min(), d["X"].max(), d["Y"].min(), d["Y"].max()]
        for j, (img, t) in enumerate([(obs, "снимок"), (pred, "модель"), (obs - pred, "остаток")]):
            ax = axs[i, j]; ax.imshow(img, cmap="RdBu_r", vmin=-v, vmax=v, extent=ext, origin="upper")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(t, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{b} ({int(d['res'])} м)", fontsize=8)
    h = min(au["hypotheses"], key=lambda q: q["cv"])
    note = h.get("drives_summary") or ""
    f.suptitle(f"{name}\n{au['chosen']}; запас {au['margin_eff']:.1f} (порог 4); "
               f"{'НЕШТАТНО, ' + str(round(au.get('deviation_deg') or 0)) + '°' if au.get('anomalous') else 'штатно'}\n{note}",
               fontsize=8)
    f.tight_layout(); f.savefig(Path(out) / f"{name}.png", dpi=110); plt.close(f)
    # вид модели с Sentinel (вдоль визира), грани по группам материалов
    Vb, R, _ = AP.pose_body(spec, law, x, P.sun_l)
    los_l = C.T @ P.los; e1 = np.cross(los_l, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(los_l, e1)
    T = Vb[spec["F"]]; P2 = np.stack([T @ e1, T @ e2], -1)
    order = np.argsort((T @ los_l).mean(1))
    col = np.array([[0.15, 0.25, 0.55], [0.85, 0.85, 0.8], [0.3, 0.3, 0.3]])[np.asarray(spec["G"])[order] % 3]
    f2, ax = plt.subplots(figsize=(4, 4))
    ax.add_collection(PolyCollection(P2[order], facecolors=col, edgecolors="none")); ax.autoscale(); ax.set_aspect("equal")
    ax.set_title(f"{name.split('_')[0]}: вид с Sentinel-2, м\n{au['chosen']}", fontsize=8); ax.tick_params(labelsize=7)
    f2.tight_layout(); f2.savefig(Path(out) / f"{name}_model.png", dpi=110); plt.close(f2)
    print(name, "готово", flush=True)


if __name__ == "__main__":
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    for n in a.names or sorted(p.stem for p in (ROOT / "results/auto").glob("*.json") if not p.name.startswith("inject")):
        try:
            fig(n, a.out)
        except Exception as ex:                    # noqa: BLE001
            import traceback; traceback.print_exc(); print(n, "ОШИБКА", ex, flush=True)

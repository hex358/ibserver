"""Коллаж всех найденных объектов: снимок Sentinel-2 (полоса с самым сильным пятном) и модель в найденной позе, вид с
Sentinel-2; подписи — закон ориентации, нештатность, поправка орбиты и её погрешность.

  python scripts/collage.py [--out results/collage.png]
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
from s2sat import autopose as AP, modelcat as MC

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "results/collage.png")); ap.add_argument("--cols", type=int, default=4)
a = ap.parse_args()

names = sorted(p.stem for p in (ROOT / "results/auto").glob("*.json") if not p.name.startswith("inject"))
rows = (len(names) + a.cols - 1) // a.cols
fig = plt.figure(figsize=(4.4 * a.cols, 2.75 * rows), facecolor="white")
gs = fig.add_gridspec(rows, 2 * a.cols, wspace=0.05, hspace=0.42, top=0.955, bottom=0.03)
col = np.array([[0.13, 0.22, 0.52], [0.86, 0.86, 0.82], [0.32, 0.32, 0.32]])
for i, n in enumerate(names):
    r_, c_ = divmod(i, a.cols)
    au = json.loads((ROOT / "results/auto" / f"{n}.json").read_text())
    rc = pickle.loads((ROOT / "results/shape" / f"{n}.pkl").read_bytes()); MR.prepare(rc)
    spec = MC.spec_for(int(rc["cat"]), n); C = np.asarray(au["C_lvlh_to_fit"])
    # снимок: полоса 10 м с самым сильным пятном
    b = max([b for b in rc["bands"] if rc["bands"][b]["res"] <= 10] or list(rc["bands"]),
            key=lambda b: np.nanmax(np.abs(rc["bands"][b]["r"])) / float(rc["bands"][b]["s"]))
    d = rc["bands"][b]; z = d["r"] / float(d["s"]); v = np.nanpercentile(np.abs(z), 99.5) or 1.0
    ax = fig.add_subplot(gs[r_, 2 * c_]); ax.imshow(z, cmap="RdBu_r", vmin=-v, vmax=v); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{b}, {int(d['res'])} м/пкс", fontsize=6, labelpad=1)
    # модель в найденной позе, вид вдоль визира
    law, x = au["law"], np.asarray(au["x"], float)
    Vb, R, _ = AP.pose_body(spec, law, x, C.T @ np.asarray(rc["sun"]))
    los = C.T @ np.asarray(rc["los"]); e1 = np.cross(los, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(los, e1)
    T = Vb[spec["F"]]; o = np.argsort((T @ los).mean(1)); P = np.stack([T @ e1, T @ e2], -1)[o]
    ax2 = fig.add_subplot(gs[r_, 2 * c_ + 1])
    ax2.add_collection(PolyCollection(P, facecolors=col[np.asarray(spec["G"])[o] % 3], edgecolors="none"))
    ax2.autoscale(); ax2.set_aspect("equal"); ax2.set_xticks([]); ax2.set_yticks([])
    ext = float(np.ptp(P.reshape(-1, 2), 0).max()); ax2.set_xlabel(f"размер {ext:.0f} м", fontsize=6, labelpad=1)
    # подпись
    law_s = au["chosen"].replace("на Землю, ", "").replace("углы панелей по снимку", "панели по снимку").replace("панели следят за Солнцем", "панели на Солнце")
    tag = f"НЕШТАТНО {round(au.get('deviation_deg') or 0)}°" if au.get("anomalous") else law_s
    orb = ROOT / "results/orbits" / f"{n}.json"
    o_s = ""
    if orb.exists():
        q = json.loads(orb.read_text()); th = q["fits"].get("gps", q["fits"].get("tle"))["theta_km"]; sg = q["sigma_m"]
        o_s = f"\nпоправка орбиты к TLE: вдоль {th[0] * 1e3:+.0f} м, поперёк {th[1] * 1e3:+.0f} м (±{sg[0]:.0f}/{sg[1]:.0f} стат.)"
    obj, ts = n.split("_")
    fig.text(0.5 * (ax.get_position().x0 + ax2.get_position().x1), ax.get_position().y1 + 0.004,
             f"{obj}  {ts[6:8]}.{ts[4:6]} {ts[9:11]}:{ts[11:13]}\n{tag}; запас {au['margin_eff']:.0f}{o_s}",
             ha="center", va="bottom", fontsize=6.5, color="#a00000" if au.get("anomalous") else "black")
fig.suptitle("Ocellus: объекты на снимках Sentinel-2 — снимок (остаток в σ шума) и 3D-модель в позе, найденной по пикселям",
             fontsize=11, y=0.985)
fig.text(0.5, 0.005, "Поправка орбиты — к TLE детекции на момент снимка, орбита Sentinel-2 по бортовому GPS. Погрешность — только статистическая: "
         "сверка с CelesTrak показывает систематику поперёк орбиты до ~1 км, причина ищется калибровкой по спутникам с точной орбитой.",
         ha="center", fontsize=7, color="#444")
fig.savefig(a.out, dpi=130, bbox_inches="tight"); print(a.out, len(names), "объектов")

"""Лист одного объекта: цветной снимок Sentinel-2 (объект в каждом канале — в своём месте: каналы сняты в разные
моменты), по каналам R, G, B — снимок, предсказание модели в найденной позе и положении, разность; вид модели с
Sentinel-2; цифры (закон ориентации, поправка орбиты, карточка); внизу — фотореалистичные кадры, если есть.

  python scripts/object_sheet.py ISS_20261118T222808 [--renders results/realistic] [--out results/sheets/<имя>.png]
"""
import argparse, json, math, pickle, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
import physdet as PD
from s2sat import autopose as AP, imagery as I, modelcat as MC, scenegeom as SG
from s2sat.orbits import dt_from_jd

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--renders", default=str(ROOT / "results/realistic"))
ap.add_argument("--out", default=None)
a = ap.parse_args()
CH = [("red", "R — красный (B04)"), ("green", "G — зелёный (B03)"), ("blue", "B — синий (B02)")]


def main():
    n = a.name
    au = json.loads((ROOT / "results/auto" / f"{n}.json").read_text())
    rc = pickle.loads((ROOT / "results/shape" / f"{n}.pkl").read_bytes()); MR.prepare(rc)
    e, r = SG.event_of(n)
    spec = MC.spec_for(int(rc["cat"]), n); C = np.asarray(au["C_lvlh_to_fit"]); law, x = au["law"], np.asarray(au["x"], float)
    P = AP.Problem(rc, spec, C); Vfit = P.body(law, x)
    chans = [(b, t) for b, t in CH if b in rc["bands"]]
    jd0 = float(np.median([rc["bands"][b]["jd"] for b in rc["bands"]]))

    # предсказание модели по каналам
    pan = {}
    for b, _ in chans:
        d = rc["bands"][b]
        occ = AP.regress(d, Vfit, spec["F"], spec["G"], P.ng, P.los, x[4:7]); _, c = AP.band_loss(d, occ)
        O = np.stack(occ, -1); pr = O @ c - d["g"] * O.sum(-1)
        pan[b] = dict(raw=(d["g"] + d["r"]) / 1e4, model=(d["g"] + pr) / 1e4, diff=(d["r"] - pr) / float(d["s"]),
                      ext=[d["X"].min() - d["res"] / 2, d["X"].max() + d["res"] / 2, d["Y"].min() - d["res"] / 2, d["Y"].max() + d["res"] / 2],
                      dt=(float(d["jd"]) - jd0) * 86400, chi_raw=float(np.nansum((d["r"] / float(d["s"])) ** 2)),
                      chi_res=float(np.nansum(((d["r"] - pr) / float(d["s"])) ** 2)))

    # цветной снимок вокруг всех трёх положений
    xs = [v for b in pan for v in pan[b]["ext"][:2]]; ys = [v for b in pan for v in pan[b]["ext"][2:]]
    m = 500.0; bb = (math.floor((min(xs) - m) / 10) * 10, math.floor((min(ys) - m) / 10) * 10,
                     math.ceil((max(xs) + m) / 10) * 10, math.ceil((max(ys) + m) / 10) * 10)
    its = I.items_around(e["lon"], e["lat"], dt_from_jd(e["jd"]), e["sent"][-1], d=0.6)
    IM = PD.read_band_region(its, int(r["epsg"]), bb, ["red", "green", "blue"], 10.0)
    rgb = np.stack([IM[k] for k in ("red", "green", "blue")], -1).astype(float)
    rgb = np.where((rgb > 0).all(-1, keepdims=True), rgb / 1e4 - 0.1, np.nan)
    lo, hi = np.nanpercentile(rgb, 1), np.nanpercentile(rgb, 99.7)
    rgb8 = np.clip((np.nan_to_num(rgb, nan=lo) - lo) / (hi - lo), 0, 1) ** (1 / 1.8)

    fig = plt.figure(figsize=(16, 21.5), facecolor="white")
    gs = fig.add_gridspec(5, 4, height_ratios=[4.6, 2.75, 2.75, 2.75, 3.3], hspace=0.2, wspace=0.1, top=0.962, bottom=0.01, left=0.04, right=0.985)
    obj, ts = n.split("_")
    fig.suptitle(f"{obj}: снимок Sentinel-2 {e['sent'][-2:]} {ts[6:8]}.{ts[4:6]}.{ts[:4]} {ts[9:11]}:{ts[11:13]} UTC и модель в позе, найденной по пикселям",
                 fontsize=15, y=0.985)
    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(rgb8, extent=[bb[0], bb[2], bb[1], bb[3]]); ax.set_title("Снимок в естественных цветах (R, G, B): каналы сняты в разные моменты, "
                                                                      "объект в каждом — в своём месте", fontsize=10)
    for b, lab in chans:
        cx = 0.5 * (pan[b]["ext"][0] + pan[b]["ext"][1]); cy = 0.5 * (pan[b]["ext"][2] + pan[b]["ext"][3])
        colr = {"red": "#ff3030", "green": "#30ff30", "blue": "#4060ff"}[b]
        ax.add_patch(plt.Circle((cx, cy), 260, fill=False, color=colr, lw=1.6))
        ax.text(cx - 300, cy + 330, f"{lab[0]}  {pan[b]['dt']:+.2f} с", color=colr, fontsize=10, weight="bold", ha="right")
    ax.set_xticks([]); ax.set_yticks([])
    ax.plot([bb[0] + 300, bb[0] + 1300], [bb[1] + 300] * 2, color="white", lw=3); ax.text(bb[0] + 800, bb[1] + 420, "1 км", color="white", ha="center", fontsize=9)

    # модель в найденной позе, вид с Sentinel-2
    Vb, R, _ = AP.pose_body(spec, law, x, P.sun_l)
    los = C.T @ P.los; e1 = np.cross(los, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(los, e1)
    T = Vb[spec["F"]]; o = np.argsort((T @ los).mean(1)); P2 = np.stack([T @ e1, T @ e2], -1)[o]
    colm = np.array([[0.13, 0.22, 0.52], [0.86, 0.86, 0.82], [0.32, 0.32, 0.32]])[np.asarray(spec["G"])[o] % 3]
    ax2 = fig.add_subplot(gs[0, 2]); ax2.add_collection(PolyCollection(P2, facecolors=colm, edgecolors="none"))
    ax2.autoscale(); ax2.set_aspect("equal"); ax2.tick_params(labelsize=8); ax2.set_xlabel("м", fontsize=8)
    ax2.set_title("Модель в найденной позе\n(вид с Sentinel-2)", fontsize=10)
    sun2 = np.array([P.sun_l @ e1, P.sun_l @ e2]); sun2 = sun2 / (np.linalg.norm(sun2) + 1e-9)
    xl, yl = ax2.get_xlim(), ax2.get_ylim(); s0 = np.array([xl[0] + 0.12 * np.ptp(xl), yl[1] - 0.12 * np.ptp(yl)])
    ax2.annotate("", s0 + sun2 * 0.1 * np.ptp(xl), s0, arrowprops=dict(arrowstyle="->", color="orange", lw=2))
    ax2.text(*s0, "Солнце ", color="orange", fontsize=8, ha="right")

    # цифры
    pp = json.loads(Path(au["pose_pix"]).read_text()) if au.get("pose_pix") and Path(au["pose_pix"]).exists() else None
    if pp:                                                     # поза по пикселям (scripts/pose_pix.py)
        sd = pp["sigma"]["rot_deg"]; m2 = pp["modes"][1] if len(pp["modes"]) > 1 else None
        txt = ["Поза — по пикселям, без законов ориентации:", "корпус по всей SO(3) + угол каждого крыла, градиент",
               f"невязка {pp['L']:.0f} (без объекта {pp['null']:.0f}), κ {pp['kappa']:.2f}",
               f"σ поворота (вдоль/поперёк/радиус) {sd[0]:.1f}/{sd[1]:.1f}/{sd[2]:.1f}°",
               f"углы крыльев {', '.join(f'{v % 180:.0f}' for v in pp['drives_deg'])}° ± {', '.join(f'{v:.1f}' for v in pp['sigma']['drives_deg'])}°",
               f"вероятность лучшего минимума {pp['modes'][0]['prob']:.2f}" + (f"; второй: +{m2['dL']:.0f}, {m2['angle_deg']:.0f}° от лучшего" if m2 else ""),
               f"до «на Землю» {pp['nearest_nadir']['angle_deg']:.0f}° (рысканье {pp['nearest_nadir']['yaw_deg']:.0f}°)",
               f"нормали крыльев от Солнца {', '.join(f'{v:.0f}' for v in pp['wing_normal_from_sun_deg'])}°"]
        for c in pp.get("cross_validation", []):
            txt.append(f"проверка: поза по {'+'.join(c['train'])} предсказывает {'+'.join(c['test'])}: {c['explained_pose_from_train'] * 100:.0f}% сигнала")
        for c in pp.get("injection", []):
            txt.append(f"подстановка ({c['truth']}): ошибка {c['angle_err_deg']:.1f}°")
        if pp.get("old"):
            txt.append(f"прежняя схема (законы): невязка +{pp['old']['L'] - pp['L']:.0f}, {pp['old']['angle_to_new_deg']:.0f}° от найденной")
    else:
        txt = [f"Закон ориентации: {au['chosen']}", f"запас над второй гипотезой {au['margin_eff']:.0f} (порог 4)",
               ("НЕШТАТНАЯ ориентация, " + f"{au.get('deviation_deg'):.0f}° от ближайшего закона") if au.get("anomalous") else "ориентация штатная"]
        h = min(au["hypotheses"], key=lambda q: q["cv"])
        if h.get("drives_summary"):
            txt.append(f"приводы панелей: {h['drives_summary']}")
    orb = ROOT / "results/orbits" / f"{n}.json"
    if orb.exists():
        q = json.loads(orb.read_text()); fk = "gps3d" if "gps3d" in q["fits"] else ("gps" if "gps" in q["fits"] else "tle")
        th = q["fits"][fk]["theta_km"]; sg = q["fits"][fk].get("sigma_m", q.get("sigma_m"))
        txt += ["", f"Положение (орбита Sentinel {'GPS + 3D-углы' if fk == 'gps3d' else fk}):",
                f"поправка к TLE: вдоль {th[0] * 1e3:+.0f} м, поперёк {th[1] * 1e3:+.0f} м (±{sg[0]:.0f} / ±{sg[1]:.0f} м)"]
    card = ROOT / "results/cards" / f"{n}.json"
    if card.exists():
        k = json.loads(card.read_text())
        txt += ["", "Карточка:", f"масса {k['mass_kg']:.0f} кг ({k.get('mass_source')})",
                f"силуэт: кувыркание {k['area']['tumbling_m2']:.0f} м², по потоку {k['area']['ram_m2']:.0f} м²",
                f"срок без поддержания орбиты ≈ {(k['decay']['lifetime_days_model_mean_atm'] or 0) / 365.25:.1f} г (средняя атмосфера)"]
    txt += ["", f"дальность {rc['rng']:.0f} км; полос в подгонке {len(P.bands)}", f"модель: {spec['source'][:70]}"]
    ax3 = fig.add_subplot(gs[0, 3]); ax3.axis("off"); ax3.text(0, 1, "\n".join(txt), va="top", fontsize=9.5, wrap=True)

    # каналы: снимок / модель / разность
    for i, (b, lab) in enumerate(chans):
        p = pan[b]; lo_, hi_ = np.nanpercentile(p["raw"], 0.5), np.nanpercentile(p["raw"], 99.8)
        v = max(5.0, float(np.nanpercentile(np.abs(p["diff"]), 99.5)), float(np.nanpercentile(np.abs(rc["bands"][b]["r"] / float(rc["bands"][b]["s"])), 99.5)))
        ext = [p["ext"][0] - p["ext"][0], p["ext"][1] - p["ext"][0], p["ext"][2] - p["ext"][2], p["ext"][3] - p["ext"][2]]
        for j, (img, ttl, kw) in enumerate([(p["raw"], "снимок (отражение)", dict(cmap="gray", vmin=lo_, vmax=hi_)),
                                            (p["model"], "модель в найденной позе и положении", dict(cmap="gray", vmin=lo_, vmax=hi_)),
                                            (p["diff"], "разность снимок − модель, σ шума", dict(cmap="RdBu_r", vmin=-v, vmax=v))]):
            axc = fig.add_subplot(gs[1 + i, j]); im = axc.imshow(img, extent=ext, **kw); axc.tick_params(labelsize=7)
            axc.set_xlabel("м", fontsize=7, labelpad=0)
            axc.set_title(f"{lab}: {ttl}", fontsize=9)
            if j == 2:
                plt.colorbar(im, ax=axc, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
        axt = fig.add_subplot(gs[1 + i, 3]); axt.axis("off")
        red = 100 * (1 - p["chi_res"] / max(p["chi_raw"], 1e-9))
        axt.text(0, 0.95, f"{lab}\nснят на {p['dt']:+.2f} с от середины\nпиксель {int(rc['bands'][b]['res'])} м, окно "
                          f"{int(p['ext'][1] - p['ext'][0])} м\nσ шума {float(rc['bands'][b]['s']) / 1e4:.4f} (отражение)\n"
                          f"модель объясняет {red:.0f}% сигнала объекта\n(Σ квадратов в окне: {p['chi_raw']:.0f} → {p['chi_res']:.0f})",
                 va="top", fontsize=9.5)

    # фотореалистичные кадры
    shots = [(s, Path(a.renders) / f"{n}_{s}.png") for s in ("wide", "mid", "tele")]
    shots = [(s, f) for s, f in shots if f.exists()]
    if shots:
        sg2 = gs[4, :].subgridspec(1, len(shots), wspace=0.03)
        names = {"wide": "общий план: 2,5 м/пкс на земле", "mid": "средний план: 46 см на объекте", "tele": "крупный план: 12 см на объекте"}
        for j, (s, f) in enumerate(shots):
            axr = fig.add_subplot(sg2[0, j]); axr.imshow(mpimg.imread(f)); axr.axis("off")
            axr.set_title(names.get(s, s), fontsize=9.5)
        fig.text(0.5, axr.get_position().y1 + 0.022, "Фотореалистичные кадры: камера на месте Sentinel-2 в момент съёмки, "
                 "поза из подбора по пикселям, Земля и свет — с этого же снимка", ha="center", fontsize=11)
    out = Path(a.out) if a.out else ROOT / "results/sheets" / f"{n}.png"
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=110); print(out)


if __name__ == "__main__":
    main()

"""Улучшает ли снимок публичную орбиту? Проверка прогнозом (THEORY §25).

  python scripts/orbit_improve.py [--out results/orbits]
Для каждого пролёта с results/orbits/<имя>.json (положение по снимку, орбита Sentinel по GPS, углы полос в 3D):
  1. «Было»: самый свежий публичный TLE с эпохой ДО снимка (GP или SupGP CelesTrak) — то, что доступно в момент съёмки.
  2. «Стало»: тот же TLE со сдвигом фазы вдоль орбиты так, чтобы он проходил через положение со снимка
     (r(t) = r_TLE(t − δ), δ = Δвдоль / v).
  3. Эталон: каждый более поздний публичный TLE (эпоха через 0,05–3 сут после снимка) в СВОЮ эпоху — там подгонка
     публичной орбиты точнее всего.
  4. Ошибка прогноза «было» и «стало» на эпохи эталонов: вдоль, поперёк и по модулю.
Плюс ошибка самих публичных орбит в момент снимка в зависимости от их возраста (картинка).
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from s2sat import geometry as G, orbits as O, passgeom as PG

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "results/orbits"))
ap.add_argument("--base", default="fresh", choices=["fresh", "supgp"], help="основа: свежайший TLE до снимка или SupGP (план оператора с манёврами)")
a = ap.parse_args()


def ecef_shifted(o, jd, delta_s):
    """Положение по TLE со сдвигом по времени (в инерциальных осях), в ECEF на момент jd."""
    r, v = o.teme([jd - delta_s / 86400.0])
    rr, _ = O.teme_to_ecef(np.array([jd]), r, v)
    return rr[0]


def axes(r, v):
    uv = G.unit(v); ur = G.unit(r); uc = G.unit(np.cross(ur, uv)); return np.stack([uv, uc, ur])


orbs = PG.Orbits(until="99999999")
rows, ages = [], []
for f in sorted(Path(a.out).glob("*.json")):
    if f.name.startswith("summary"):
        continue
    d = json.loads(f.read_text())
    if "position_ecef_km" not in d:
        continue
    cat = str(d["norad"]).zfill(5) if str(d["norad"]).zfill(5) in orbs.GP or str(d["norad"]).zfill(5) in orbs.SG else str(d["norad"])
    jd = d["jd"]; rm = np.asarray(d["position_ecef_km"]); sig = d["sigma_m"]
    pool = [("GP", o) for o in orbs.GP.get(cat, [])] + [("SupGP", o) for o in orbs.SG.get(cat, [])]
    # ошибка публичных орбит в момент снимка против возраста
    for kind, o in pool:
        age = jd - o.epoch_jd
        if abs(age) > 4:
            continue
        r, v = o.ecef([jd]); A = axes(r[0], v[0]); dd = A @ (r[0] - rm)
        ages.append(dict(name=d["name"], kind=kind, age_days=age, d_along_km=float(dd[0]), d_cross_km=float(dd[1])))
    before = [(k, o) for k, o in pool if 0 < jd - o.epoch_jd <= 3]
    if not before:
        continue
    if a.base == "supgp" and any(k == "SupGP" for k, _ in before):
        before = [(k, o) for k, o in before if k == "SupGP"]
    kb, ob = max(before, key=lambda q: q[1].epoch_jd)
    r, v = ob.ecef([jd]); A = axes(r[0], v[0]); d0 = A @ (r[0] - rm)
    delta = float(d0[0]) / float(np.linalg.norm(ob.teme([jd])[1][0]))          # с: TLE впереди на Δвдоль → назад на δ
    refs = [(k, o) for k, o in pool if 0.05 <= o.epoch_jd - jd <= 3]
    for kr, orf in refs:
        t2 = orf.epoch_jd
        r_ref = orf.ecef([t2])[0][0]; rv = orf.ecef([t2]); A2 = axes(rv[0][0], rv[1][0])
        r_old = ob.ecef([t2])[0][0]; r_new = ecef_shifted(ob, t2, delta)
        e_old = A2 @ (r_old - r_ref); e_new = A2 @ (r_new - r_ref)
        rows.append(dict(name=d["name"], base=f"{kb} −{jd - ob.epoch_jd:.2f} сут", ref=f"{kr} +{t2 - jd:.2f} сут", horizon_days=t2 - jd,
                         old_along_km=float(e_old[0]), new_along_km=float(e_new[0]), old_km=float(np.linalg.norm(e_old)),
                         new_km=float(np.linalg.norm(e_new)), meas_sigma_along_m=sig[0]))

out = Path(a.out); (out / f"improve{'' if a.base == 'fresh' else '_supgp'}.json").write_text(json.dumps(dict(predictions=rows, public_vs_image=ages), ensure_ascii=False, indent=1))
print(f"прогнозов: {len(rows)} (объектов {len({r['name'] for r in rows})})")
for name in sorted({r["name"] for r in rows}):
    rr = [r for r in rows if r["name"] == name]
    print(f"  {name.split('_')[0]:15s} база {rr[0]['base']:18s} эталонов {len(rr):2d}: ошибка прогноза медиана было {np.median([r['old_km'] for r in rr]):6.2f} км → "
          f"стало {np.median([r['new_km'] for r in rr]):6.2f} км")
if rows:
    old = np.array([r["old_km"] for r in rows]); new = np.array([r["new_km"] for r in rows])
    for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 3.0)):
        m = np.array([lo <= r["horizon_days"] < hi for r in rows])
        if m.any():
            print(f"  горизонт {lo}–{hi} сут: n={m.sum():3d}, медиана было {np.median(old[m]):.2f} км → стало {np.median(new[m]):.2f} км, "
                  f"лучше в {np.mean(new[m] < old[m]) * 100:.0f}% случаев")

# картинка: ошибка публичных орбит в момент снимка против возраста + прогноз было/стало
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for kind, mk in (("GP", "o"), ("SupGP", "s")):
    q = [g for g in ages if g["kind"] == kind]
    ax[0].scatter([abs(g["age_days"]) for g in q], [abs(g["d_along_km"]) * 1e3 for g in q], marker=mk, s=18, alpha=0.7, label=f"{kind}: вдоль")
    ax[0].scatter([abs(g["age_days"]) for g in q], [abs(g["d_cross_km"]) * 1e3 for g in q], marker=mk, s=12, alpha=0.5, label=f"{kind}: поперёк")
sa = [json.loads(f.read_text())["sigma_m"][0] for f in out.glob("*_*.json") if "sigma_m" in json.loads(f.read_text())]
ax[0].axhspan(0, float(np.median(sa)), color="green", alpha=0.15, label=f"погрешность снимка вдоль (медиана {np.median(sa):.0f} м)")
ax[0].set_yscale("log"); ax[0].set_xlabel("возраст публичной орбиты, сут"); ax[0].set_ylabel("|публичная − снимок|, м")
ax[0].set_title("Публичные орбиты CelesTrak в момент снимка"); ax[0].legend(fontsize=7)
if rows:
    h = [r["horizon_days"] for r in rows]
    ax[1].scatter(h, [r["old_km"] * 1e3 for r in rows], s=16, color="#c0392b", label="было: свежайший TLE до снимка")
    ax[1].scatter(h, [r["new_km"] * 1e3 for r in rows], s=16, color="#1e8449", label="стало: тот же TLE + поправка по снимку")
    ax[1].set_yscale("log"); ax[1].set_xlabel("горизонт прогноза после снимка, сут"); ax[1].set_ylabel("ошибка прогноза против более поздней публичной орбиты, м")
    ax[1].set_title("Прогноз орбиты: было → стало"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(out / f"improve{'' if a.base == 'fresh' else '_supgp'}.png", dpi=120)

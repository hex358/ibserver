"""Окна остатков по ВСЕМ снимкам объекта в пролёте: полоса × детектор (+ полосы 60 м) — больше моментов съёмки.

  python scripts/prep_sightings.py results/events_rb4/CZ-6A_20261125T012657.json [--all60] [--until 20261127T1755]
→ results/shape/<имя>_ms.pkl (тот же формат, что у shape_fit.py prep; ключи полос: «green» — снимок основного
детектора, «green#6» — тот же объект, снятый соседним детектором в другой момент).

Зачем. Детекторы Sentinel-2 стоят в шахматном порядке: чётные и нечётные смотрят под разными углами вдоль трека.
Объект в зоне перекрытия соседних детекторов снимается каждой полосой дважды. Прежняя подготовка брала один корень на
полосу (второй — отбрасывался или считался выбросом), отсюда «2 момента съёмки» у встречных объектов. Полосы 60 м
(B01 coastal, B09 nir09) стоят на краях фокальной плоскости — ещё моменты.

Геометрия — по точной орбите Sentinel (бортовой GPS из MTD_DS) и 3D-углам полос. θ (поправка орбиты объекта, км:
вдоль, поперёк, радиус) — Гаусс–Ньютон по положениям пятна во всех снимках, веса Хьюбера, априори поперёк 1,1 км,
радиус 1,9 км. Фон каждого окна — по всем полосам (loglocal), положения объекта во всех снимках вырезаны.
"""
import argparse, json, math, os, pickle, sys, time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from s2sat import passgeom as PG, s2orbit as S2O, s2meta as M

ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("--all60", action="store_true")
ap.add_argument("--until", default="20261127T1755"); ap.add_argument("--suffix", default="_ms")
a = ap.parse_args()
orbs = PG.Orbits(until=a.until)
OUT = ROOT / "results/shape"


def key_name(k, prim):
    return k[0] if prim.get(k[0]) == k[1] else f"{k[0]}#{k[1]}"


for path in a.paths:
    x = json.loads(Path(path).read_text()); e, r = x["event"], x["result"]
    name = Path(path).stem + a.suffix; t0 = time.time()
    P = PG.Pass(e, r, orbs); P.orb = PG.pick_orbit(orbs, e)
    po = S2O.precise_for(r["item"], e["sent"], e["jd"])
    P.regrid(po, all60=a.all60)                                   # GPS Sentinel + 3D-углы (+ полосы 60 м)
    tgt = {b: tuple(v) for b, v in r["pos_best"].items()}
    th, P0, J0, rms0 = PG.gauss_newton_theta(P, tgt, [r.get("dM", 0.0), 0.0, 0.0])
    S, J = P.jacobian_sightings(th)
    # основной детектор полосы — снимок ближе всего к найденному положению (по нему шла детекция)
    prim = {}
    for (b, det), (sx, sy, _) in S.items():
        if b in P0:
            dd = math.hypot(sx - P0[b][0], sy - P0[b][1])
            if b not in prim or dd < prim[b][1]:
                prim[b] = (det, dd)
    prim = {b: v[0] for b, v in prim.items()}
    for b in {k[0] for k in S} - set(prim):                       # полосы без детекции (60 м): основной — первый
        prim[b] = min(det for (bb, det) in S if bb == b)
    pos = {k: (v[0], v[1]) for k, v in S.items()}
    mpts = [(k[0], v[0], v[1]) for k, v in S.items()]
    W = PG.window_residuals(P, pos, read_bands=[b for b in P.bands if M.BRES.get(b, 20) <= 60], mask_pts=mpts)
    # положение пятна в каждом снимке и устойчивый θ по всем снимкам
    meas = {}
    for k, d in W.items():
        z = np.nan_to_num(d["r"] / d["s"]); zs = gaussian_filter(z, 1.0) * math.sqrt(4 * math.pi)
        near = np.hypot(d["X"] - pos[k][0], d["Y"] - pos[k][1]) <= 250.0
        i = np.argmax(np.where(near, np.abs(zs), -1)); pk = float(np.abs(zs).flat[i])
        if pk >= 4.0:
            meas[k] = (float(d["X"].flat[i]), float(d["Y"].flat[i]), pk)
    for _ in range(4):
        ks = [k for k in meas if k in S and k in J]
        if len(ks) < 3:
            break
        A = np.concatenate([J[k] for k in ks]); rv = np.concatenate([np.array(meas[k][:2]) - np.array(S[k][:2]) for k in ks])
        wts = np.repeat([min(1.0, 30.0 / max(math.hypot(*(np.array(meas[k][:2]) - np.array(S[k][:2]))), 1e-6)) for k in ks], 2)
        pr = np.array([[0, 1 / 1.1, 0], [0, 0, 1 / 1.9]]) * 30.0
        dth, *_ = np.linalg.lstsq(np.vstack([A * np.sqrt(wts)[:, None], pr]), np.concatenate([rv * np.sqrt(wts), -pr @ th]), rcond=None)
        th = th + np.clip(dth, -2.0, 2.0); S, J = P.jacobian_sightings(th)
    dev = {key_name(k, prim): math.hypot(meas[k][0] - S[k][0], meas[k][1] - S[k][1]) for k in meas if k in S}
    jd_mid = float(np.median([v[2] for v in S.values()]))
    Af, sun, los, rng = P.frame(jd_mid)
    bands = {}; drop = {k: "нет окна (снимок не прочитан или фон не построен)" for k in S if k not in W}
    for k, d in W.items():
        if k not in S or k not in J:
            drop[k] = "нет якобиана"; continue
        p0 = np.array(S[k][:2])
        fit = np.hypot(d["X"] - p0[0], d["Y"] - p0[1]) <= 160.0
        if fit.sum() < 0.8 * math.pi * (160.0 / d["res"]) ** 2 or not np.isfinite(d["s"]) or np.isfinite(d["r"][fit]).mean() < 0.8:
            drop[k] = f"окно неполное (пикселей {int(fit.sum())}, конечных {np.isfinite(d['r'][fit]).mean():.0%})"; continue
        ii = np.where(fit.any(1))[0]; jj = np.where(fit.any(0))[0]
        sl = (slice(ii[0], ii[-1] + 1), slice(jj[0], jj[-1] + 1))
        bands[key_name(k, prim)] = dict(X=d["X"][sl], Y=d["Y"][sl], r=d["r"][sl], s=d["s"], g=d["g"][sl], res=d["res"], p0=p0,
                                        J=J[k], jd=S[k][2], meas=meas.get(k), band=k[0], det=k[1])
    if not bands:
        print(name, "нет окон", flush=True); continue
    rec = dict(name=name, obj=e["obj"], cat=e["cat"], theta=th.tolist(), fit_rms=float(np.median(list(dev.values()))) if dev else float("nan"),
               dev=dev, sun=sun.tolist(), los=los.tolist(), rng=rng, score=r.get("score"), bands=bands, sightings=True,
               row=[math.sin(math.radians(r["psi_deg"])), math.cos(math.radians(r["psi_deg"]))])
    (OUT / f"{name}.pkl").write_bytes(pickle.dumps(rec))
    tj = sorted((d["jd"] - min(dd["jd"] for dd in bands.values())) * 86400 for d in bands.values())
    print(f"{name}: θ {np.round(th, 3)} км; снимков {len(bands)} (было бы {len({k[0] for k in S})} полос), из них соседним "
          f"детектором {sum('#' in k for k in bands)}; разброс моментов {tj[-1]:.2f} с; пятно найдено в {len(meas)}; "
          f"медиана отклонения {rec['fit_rms']:.0f} м; {time.time() - t0:.0f} с", flush=True)
    for k, why in sorted(drop.items()):
        print(f"   выброшен снимок {k[0]}#{k[1]} (t = {(S[k][2] - min(v[2] for v in S.values())) * 86400:.2f} с): {why}", flush=True)
    print("   моменты, с:", ", ".join(f"{k} {(d['jd'] - min(dd['jd'] for dd in bands.values())) * 86400:.2f}" for k, d in sorted(bands.items(), key=lambda q: q[1]["jd"])), flush=True)

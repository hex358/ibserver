"""Точное положение спутника по снимку и сравнение с орбитами CelesTrak (THEORY §25).

  python scripts/orbit_check.py [имена ...]        (без имён — все results/shape/*.pkl)
Для каждого найденного пролёта:
  1. орбита Sentinel-2 — по бортовому GPS (s2sat/s2orbit.py), а не по TLE;
  2. сдвиг объекта θ (вдоль, поперёк, по радиусу; км) от орбиты детекции — по измеренным центрам пятен всех полос
     (Гаусс–Ньютон, веса Хьюбера, априорное условие поперёк 1,1 км и по радиусу 1,9 км — радиус по одному ракурсу
     определяется слабо); тот же расчёт с TLE Sentinel — чтобы показать вклад орбиты камеры;
  3. погрешность: большая из двух — по разбросу пятен (НК) и выбросом полосы (jackknife);
  4. положение объекта на момент снимка r = r_TLE(t) + θ в осях орбиты; разность с каждым TLE CelesTrak (GP и SupGP,
     эпоха ±4 сут), вдоль / поперёк / по радиусу и эквивалентный сдвиг по времени.
Пишет results/orbits/<имя>.json и results/orbits/summary.csv.
"""
import argparse, csv, json, math, pickle, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from s2sat import geometry as G, passgeom as PG, s2orbit as S2O, scenegeom as SG

ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="*"); ap.add_argument("--out", default=str(ROOT / "results/orbits"))
PRIOR_KM = np.array([np.inf, 1.1, 1.9])            # σ априорного условия: вдоль — нет, поперёк 1,1 км, радиус 1,9 км


def fit_theta(P, meas, th0, iters=5):
    th = np.array(th0, float)
    pr = np.array([[0, 1 / PRIOR_KM[1], 0], [0, 0, 1 / PRIOR_KM[2]]]) * 30.0
    for _ in range(iters):
        P0, J = P.jacobian(th)
        bs = [b for b in meas if b in P0 and b in J]
        if len(bs) < 3:
            return None
        A = np.concatenate([J[b] for b in bs]); rv = np.concatenate([np.array(meas[b]) - np.array(P0[b][:2]) for b in bs])
        w = np.repeat([min(1.0, 30.0 / max(math.hypot(*(np.array(meas[b]) - np.array(P0[b][:2]))), 1e-6)) for b in bs], 2)
        d, *_ = np.linalg.lstsq(np.vstack([A * np.sqrt(w)[:, None], pr]), np.concatenate([rv * np.sqrt(w), -pr @ th]), rcond=None)
        th = th + np.clip(d, -2.0, 2.0)
        if np.abs(d).max() < 1e-4:
            break
    P0, J = P.jacobian(th)
    bs = [b for b in meas if b in P0 and b in J]
    res = {b: np.array(meas[b]) - np.array(P0[b][:2]) for b in bs}
    return th, P0, J, bs, res


def covariance(J, bs, res):
    """НК-ковариация θ (км²) и jackknife по полосам; σ пятна — устойчивая по остаткам (не меньше 10 м)."""
    rn = np.array([np.linalg.norm(res[b]) for b in bs])
    sig = max(10.0, 1.4826 * float(np.median(rn)) / math.sqrt(2))
    w = {b: min(1.0, 2 * sig / max(np.linalg.norm(res[b]), 1e-6)) for b in bs}
    Pr = np.diag([0.0, 1 / PRIOR_KM[1] ** 2, 1 / PRIOR_KM[2] ** 2])
    blocks = {b: w[b] * J[b].T @ J[b] / sig ** 2 for b in bs}
    grads = {b: w[b] * J[b].T @ res[b] / sig ** 2 for b in bs}
    info = sum(blocks.values()) + Pr
    C = np.linalg.inv(info)
    jk = []
    for b in bs:
        Ib = info - blocks[b]
        try:
            jk.append(np.linalg.solve(Ib, -grads[b]))
        except np.linalg.LinAlgError:
            pass
    jk = np.array(jk); n = len(jk)
    Cj = (n - 1) / n * np.cov(jk.T, bias=True) * n / max(n, 1) if n >= 3 else np.zeros((3, 3))
    return C, Cj, sig


def axes_at(orb, jd):
    r, v = orb.ecef([jd]); r, v = r[0], v[0]
    uv = G.unit(v); ur = G.unit(r); uc = G.unit(np.cross(ur, uv))
    return r, np.linalg.norm(v), np.stack([uv, uc, ur])


def check(name, orbs):
    rc = pickle.loads((ROOT / "results/shape" / f"{name}.pkl").read_bytes())
    e, r = SG.event_of(name)
    meas = {b: tuple(d["meas"][:2]) for b, d in rc["bands"].items() if d.get("meas")}
    P = PG.Pass(e, r, orbs)
    P.orb = PG.pick_orbit(orbs, e)
    jd = float(np.median([d["jd"] for d in rc["bands"].values()]))
    sen_tle = P.sen
    po = S2O.precise_for(r["item"], e["sent"], jd)
    out = dict(name=name, object=e["obj"], norad=int(e["cat"]), jd=jd, sentinel=e["sent"], image=r["item"],
               detection_tle=dict(line1=P.orb.line1, line2=P.orb.line2, age_days=round(jd - P.orb.epoch_jd, 3)),
               bands_used=len(meas))
    fits = {}
    for tag, sen in (("tle", sen_tle), ("gps", po)):
        if sen is None:
            continue
        P.sen = sen
        f = fit_theta(P, meas, rc.get("theta", [0, 0, 0]))
        if f is None:
            continue
        th, P0, J, bs, res = f
        C, Cj, sig = covariance(J, bs, res)
        sd = np.sqrt(np.maximum(np.diag(C), np.diag(Cj))) * 1e3
        fits[tag] = dict(theta_km=th.tolist(), sigma_m=sd.tolist(), sigma_ls_m=(np.sqrt(np.diag(C)) * 1e3).tolist(),
                         sigma_jackknife_m=(np.sqrt(np.diag(Cj)) * 1e3).tolist(), spot_sigma_m=sig, bands=bs,
                         residual_m={b: [round(float(v), 1) for v in res[b]] for b in bs})
    if not fits:
        print(name, "нет решения (меньше 3 полос с пятном)", flush=True); return None
    best = fits.get("gps") or fits["tle"]
    out["sentinel_orbit"] = "GPS" if "gps" in fits else "TLE"
    out["fits"] = fits
    if "gps" in fits and "tle" in fits:
        out["sentinel_tle_effect_m"] = ((np.array(fits["tle"]["theta_km"]) - np.array(fits["gps"]["theta_km"])) * 1e3).round(1).tolist()
    r0, vmag, A = axes_at(P.orb, jd)
    th = np.array(best["theta_km"]); rpos = r0 + A.T @ th
    out["position_ecef_km"] = rpos.tolist(); out["sigma_m"] = best["sigma_m"]
    out["note"] = "положение фотоцентра (пятна) на момент jd; радиус по одному ракурсу — с априорным условием 1,9 км"
    cmp = []
    for kind, pool in (("GP", orbs.GP.get(e["cat"], [])), ("SupGP", orbs.SG.get(e["cat"], []))):
        for o in pool:
            age = jd - o.epoch_jd
            if abs(age) > 4.0:
                continue
            ro = o.ecef([jd])[0][0]
            if not np.all(np.isfinite(ro)):
                continue
            d = A @ (ro - rpos)
            cmp.append(dict(source=kind, epoch_age_days=round(age, 3), when="до снимка" if age >= 0 else "после снимка",
                            d_along_km=round(float(d[0]), 3), d_cross_km=round(float(d[1]), 3), d_radial_km=round(float(d[2]), 3),
                            d_time_s=round(float(d[0]) * 1e3 / (vmag * 1e3), 3)))
    out["celestrak"] = sorted(cmp, key=lambda q: (q["source"], q["epoch_age_days"]))
    Path(ROOT / "results/orbits").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/orbits" / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    s = best["sigma_m"]
    near = {}
    for q in cmp:
        k = (q["source"], q["when"])
        if k not in near or abs(q["epoch_age_days"]) < abs(near[k]["epoch_age_days"]):
            near[k] = q
    msg = "; ".join(f"{k[0]} {k[1]} ({abs(v['epoch_age_days']):.2f} сут): вдоль {v['d_along_km']:+.2f} поперёк {v['d_cross_km']:+.2f} км"
                    for k, v in sorted(near.items()))
    eff = out.get("sentinel_tle_effect_m")
    print(f"{name}: орбита Sentinel {out['sentinel_orbit']}, полос {len(best['bands'])}, σ вдоль {s[0]:.0f} м, поперёк {s[1]:.0f} м, "
          f"радиус {s[2]:.0f} м{'' if eff is None else f'; вклад TLE Sentinel {eff} м'}\n   CelesTrak − снимок: {msg}", flush=True)
    return out, near


if __name__ == "__main__":
    a = ap.parse_args()
    orbs = PG.Orbits(until="99999999")
    names = a.names or sorted(p.stem for p in (ROOT / "results/shape").glob("*.pkl"))
    rows = []
    for n in names:
        try:
            res = check(n, orbs)
        except Exception as ex:                    # noqa: BLE001
            import traceback; traceback.print_exc(); print(n, "ОШИБКА", ex, flush=True); continue
        if res is None:
            continue
        out, near = res
        row = dict(name=n, sentinel_orbit=out["sentinel_orbit"], sig_along_m=round(out["sigma_m"][0]), sig_cross_m=round(out["sigma_m"][1]),
                   sig_radial_m=round(out["sigma_m"][2]))
        for k in [("GP", "до снимка"), ("GP", "после снимка"), ("SupGP", "до снимка"), ("SupGP", "после снимка")]:
            v = near.get(k); tag = f"{k[0]}_{'before' if k[1] == 'до снимка' else 'after'}"
            row[f"{tag}_age_d"] = None if v is None else abs(v["epoch_age_days"])
            row[f"{tag}_along_km"] = None if v is None else v["d_along_km"]
            row[f"{tag}_cross_km"] = None if v is None else v["d_cross_km"]
        rows.append(row)
    if rows:
        with open(Path(a.out) / "summary.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
        print("\nМедиана |CelesTrak − снимок| (км), ближайшая эпоха каждого вида:")
        for tag in ["GP_before", "GP_after", "SupGP_before", "SupGP_after"]:
            al = [abs(r_[f"{tag}_along_km"]) for r_ in rows if r_[f"{tag}_along_km"] is not None]
            cr = [abs(r_[f"{tag}_cross_km"]) for r_ in rows if r_[f"{tag}_cross_km"] is not None]
            if al:
                print(f"  {tag:13s} n={len(al):2d}: вдоль {np.median(al):.2f} (макс {max(al):.2f}), поперёк {np.median(cr):.2f} (макс {max(cr):.2f})")
        sa = [r_["sig_along_m"] for r_ in rows]; sc = [r_["sig_cross_m"] for r_ in rows]
        print(f"  точность снимка (медиана σ): вдоль {np.median(sa):.0f} м, поперёк {np.median(sc):.0f} м")

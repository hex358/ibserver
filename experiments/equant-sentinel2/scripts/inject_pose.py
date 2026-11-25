"""Надёжность автоматической позы (THEORY §24): синтетический объект с известной позой в настоящих окнах снимка,
тот же алгоритм autopose.solve, доля верно угаданных режимов и пойманных нештатных поз.

  python scripts/inject_pose.py STARLINK-30580_20261120T111642 --n 6 --anom 3
Объект — сетка каталога в позе: (а) закон класса + отклонения в пределах допусков; (б) нештатная поза (рысканье
любое, тангаж и крен до ±40°). Яркость материалов — доли медианы фона по полосам; шум — гауссов, σ = 1.5·s
(в 1.5 раза больше оценки шума окна — запас на структурированный фон).
"""
import argparse, copy, json, math, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import autopose as AP, modelcat as MC, scenegeom as SG

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("--n", type=int, default=6); ap.add_argument("--anom", type=int, default=3)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--noise", type=float, default=1.5); ap.add_argument("--maxfev", type=int, default=160)
ap.add_argument("--mismatch", action="store_true", help="синтетика с несоответствием модели: неоднородная яркость граней "
                "(4 подгруппы на материал, лог-нормальный разброс 35 %%) и 10 %% выпавших граней — как у настоящего снимка")
a = ap.parse_args()


def synth(rc, spec, h, x, rng, C):
    rc2 = copy.deepcopy(rc); P = AP.Problem(rc2, spec, C)
    Vb = P.body(h, x)
    F, G, ng = spec["F"], spec["G"], P.ng
    base = np.array([0.4, 1.5, 2.5])[:ng]
    if a.mismatch:                                          # модель снимка ≠ модель подгонки
        keep = rng.random(len(F)) > 0.10
        sub = rng.integers(0, 4, len(F))
        F, G = F[keep], (G * 4 + sub)[keep]; ng = ng * 4
        base = np.repeat(base, 4) * np.exp(0.35 * rng.standard_normal(ng))
    for b, d in rc2["bands"].items():
        R = AP.regress(d, Vb, F, G, ng, P.los, x[4:7])
        gm = float(np.nanmedian(d["g"]))
        c = base * gm * (1 + 0.15 * rng.standard_normal(ng))
        occ = np.stack(R, -1)
        model = (occ * c).sum(-1) - d["g"] * occ.sum(-1)
        d["r"] = model + a.noise * float(d["s"]) * rng.standard_normal(d["r"].shape)
    return rc2


def main():
    rng = np.random.default_rng(a.seed)
    rc = pickle.loads((ROOT / "results/shape" / f"{a.base}.pkl").read_bytes()); MR.prepare(rc)
    spec = MC.spec_for(rc.get("cat", 0), a.base)
    C = SG.lvlh_to_fit(a.base, rc)
    sun = np.asarray(rc["sun"], float); sun_l = C.T @ sun
    sp = 30.0 / AP.Problem(rc, spec, C).ground_scale()     # ошибка орбиты: σ 30 м на земле (объект остаётся в окне снимка)
    out = []
    trials = [("law", i) for i in range(a.n)] + [("anom", i) for i in range(a.anom)]
    for kind, i in trials:
        t0 = time.time()
        if kind == "law":                                   # штатный объект: точно по закону, сдвиг — ошибка орбиты
            h = spec["laws"][int(rng.integers(len(spec["laws"])))]
            x = np.r_[np.zeros(4), rng.normal(0, sp, 2), 0.0]
            if h.get("arrays") == "est":                    # истинные углы приводов: случайный режим ± 5°
                Rl = AP.law_attitude(h) @ spec["M2B"]
                m = ["sun", "edge", "flat", "vertical", "fixed"][int(rng.integers(5))]
                x = np.r_[x, AP.mode_angles(spec, Rl, m, sun_l) + np.radians(rng.normal(0, 5, AP.n_drives(spec)))]
        else:                                               # нештатный: случайный поворот (вектор поворота) и угол панелей
            h = dict(name="нештатная", free3=True)
            w = rng.normal(size=3); w = w / np.linalg.norm(w) * rng.uniform(math.radians(30), math.pi)
            x = np.r_[w, rng.uniform(0, math.pi), rng.normal(0, sp, 2), 0.0]
        rc2 = synth(rc, spec, h, x, rng, C)
        res = AP.solve(rc2, spec, log=lambda *_: None, maxfev=a.maxfev, C=C)
        Rt = AP.pose_body(spec, h, x, sun_l)[1]; Re = np.asarray(res["R"])
        ang = math.degrees(math.acos(np.clip((np.trace(Rt.T @ Re) - 1) / 2, -1, 1)))
        rms = AP.pose_rms(spec, sun_l, h, x, res["law"], res["x"])               # ошибка позы, м (вершины, без сдвига)
        thr = AP.eq_rms(spec)
        label_ok = h["name"] in res.get("equivalent", [res["chosen"]]) or res["chosen"] == h["name"]
        ok = res["anomalous"] if kind == "anom" else (not res["anomalous"] and rms < 2 * thr)
        # оракул: невязка при истинных параметрах — если меньше найденной, оптимизатор недоискал
        P = AP.Problem(rc2, spec, C); P.kappa = res.get("kappa") or 1.0; L_true = P.loss(h, x, P.bands, prior=False)   # эфф. χ²
        hyp = {r["name"]: r["cv"] for r in res["hypotheses"]}
        indist = (not ok) and min(r["L"] for r in res["hypotheses"]) <= L_true     # данные не различают: найденная поза не хуже истинной
        row = dict(kind=kind, true=h["name"], indistinguishable=bool(indist), chosen=res["chosen"], equivalent=res.get("equivalent"), anomalous=res["anomalous"],
                   ok=bool(ok), label_ok=bool(label_ok), pose_rms_m=rms, eq_rms_m=thr, rot_err_deg=ang, margin=res["margin"],
                   cv=hyp, cv_free=(res.get("free") or {}).get("cv"), free_gain_eff=res.get("free_gain_eff"),
                   deviation_deg=res.get("deviation_deg"), kappa=res.get("kappa"), L_true=L_true,
                   L_found=min(r["L"] for r in res["hypotheses"]), seconds=round(time.time() - t0))
        out.append(row)
        print(f"{kind:4s} истина: {h['name'][:40]:40s} → {res['chosen'][:40]:40s} {'ВЕРНО' if ok else ('НЕРАЗЛИЧИМО' if indist else 'НЕВЕРНО')}  поза ±{rms:4.1f} м "
              f"(порог {thr:.1f})  поворот {ang:5.1f}°  своб−закон {((res.get('free') or {}).get('cv', float('nan')) - min(hyp.values())):+7.0f}  "
              f"L ист {L_true:.0f} / найд {row['L_found']:.0f}  {row['seconds']} с", flush=True)
    good = [r for r in out if r["kind"] == "law"]; bad = [r for r in out if r["kind"] == "anom"]
    fg = [r["free_gain_eff"] for r in good if r.get("free_gain_eff") is not None]
    t95 = float(np.percentile(fg, 95)) if fg else None      # порог нештатности: 95 % штатных объектов ниже
    summ = dict(base=a.base, noise=a.noise, mismatch=a.mismatch, T_anom_95=t95,
                anom_detected_at_T95=sum(1 for r in bad if (r.get("free_gain_eff") or -1e9) > (t95 or 0)) if t95 is not None else None,
                law_ok=sum(r["ok"] for r in good), law_n=len(good),
                law_indistinguishable=sum(r["indistinguishable"] for r in good), anom_ok=sum(r["ok"] for r in bad), anom_n=len(bad),
                rot_err_median_deg=float(np.median([r["rot_err_deg"] for r in good])) if good else None,
                pose_rms_median_m=float(np.median([r["pose_rms_m"] for r in good])) if good else None, trials=out)
    Path(ROOT / "results/auto").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/auto" / f"inject_{a.base}{'_mismatch' if a.mismatch else ''}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(f"ИТОГ: штатные {summ['law_ok']}/{summ['law_n']} (неразличимы {summ['law_indistinguishable']}), нештатные пойманы {summ['anom_ok']}/{summ['anom_n']}, "
          f"медианная ошибка позы {summ['pose_rms_median_m']} м; порог нештатности по 95 % штатных {summ['T_anom_95']} → поймано бы "
          f"{summ['anom_detected_at_T95']}/{summ['anom_n']}", flush=True)


if __name__ == "__main__":
    main()

"""Карточка объекта для каталога и увода мусора (PLAN_characterization §2.5) — одна для всех объектов.

  python scripts/debris_card.py [имена ...] [--auto results/auto] [--spin auto|all|none]   (без имён — все)
Из готовой автоматической позы (<auto>/<имя>.json), модели каталога и TLE объекта на момент снимка:
  * модель, тип, размеры, масса (GCAT: сухая масса), работает ли аппарат (SATCAT);
  * площадь сечения: средняя при беспорядочном вращении, в найденной позе — к Солнцу и к набегающему потоку;
    A/m; баллистический коэффициент B = C_D·A/m по модели и по TLE (B*);
  * торможение: наблюдаемое по TLE (ṅ) и по модели (экспоненциальная атмосфера); их отношение; срок до входа в
    атмосферу без манёвров — по наблюдаемому торможению и по модели;
  * тонкостенная оболочка: центр масс, тензор инерции, главные оси (в осях орбиты), ось наибольшего момента;
  * выступающие элементы: крылья/панели (размах, ось привода);
  * поправка к орбите для центра масс: вдоль и поперёк следа (м), сдвиг по времени (с); радиус из одного ракурса не
    определяется;
  * яркость групп материалов по полосам (в долях яркости фона);
  * ориентация: закон / нештатность / углы приводов с погрешностями;
  * вращение (s2sat/spin.py): по умолчанию для неработающих, ступеней, обломков и нештатных; --spin all — для всех.
Пишет results/cards/<имя>.json и сводку results/cards/summary.csv.
"""
import argparse, csv, json, math, pickle, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import autopose as AP, debris as D, modelcat as MC, scenegeom as SG, spin as SP

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="*"); ap.add_argument("--auto", default=str(ROOT / "results/auto"))
ap.add_argument("--spin", default="auto", choices=["auto", "all", "none"]); ap.add_argument("--maxfev", type=int, default=300)
ap.add_argument("--out", default=str(ROOT / "results/cards"))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def r(v, k=2):
    return None if v is None else round(float(v), k)


def card(name, a):
    au = json.loads((Path(a.auto) / f"{name}.json").read_text())
    rc = pickle.loads((ROOT / "results/shape" / f"{name}.pkl").read_bytes()); MR.prepare(rc)
    e, _ = SG.event_of(name); sen, orb = SG.choose_orbits(e, rc)
    norad = int(rc.get("cat", 0) or e.get("cat", 0)); spec = MC.spec_for(norad, name); g = spec.get("gcat") or {}
    jd = float(np.median([d["jd"] for d in rc["bands"].values()])); year = 2000 + (jd - 2451544.5) / 365.25
    active = MC.is_active(norad, int(year))
    C = np.asarray(au["C_lvlh_to_fit"]) if au.get("C_lvlh_to_fit") else SG.lvlh_to_fit(name, rc)
    sun_l = np.asarray(au["sun_lvlh"]) if au.get("sun_lvlh") else C.T @ np.asarray(rc["sun"])
    law, x = au["law"], np.asarray(au["x"], float)
    Vb, R, _ = AP.pose_body(spec, law, x, sun_l)                  # вершины в найденной позе, оси орбиты (вдоль, поперёк, радиус)
    # масса — того, что изображает модель: у сборок (станций) модель — весь комплекс, а номер в каталоге — один модуль;
    # тогда масса берётся из описания модели (catalog.json: mass_kg), иначе — сухая масса GCAT
    cat_mass = num((spec.get("catalog") or {}).get("mass_kg"))
    mass = cat_mass or num(g.get("DryMass")) or num(g.get("Mass"))
    mass_src = "модель (весь комплекс)" if cat_mass else "GCAT" if mass else None

    # площади и баллистика
    F = spec["F"]
    a_tumble = D.tumbling_area(Vb, F); a_sun = D.proj_area(Vb, sun_l, F); a_ram = D.proj_area(Vb, [1.0, 0, 0], F)
    tumbling = (not active) or spec["kind"] in ("rocketbody", "debris")
    a_drag = a_tumble if tumbling else a_ram                     # стабилизированный — сечение по потоку в позе
    B_model = None if not mass else D.CD * a_drag / mass
    tle = D.tle_decay(orb.satrec)
    h = tle["h_mean_km"]
    adot_model = None if B_model is None else D.adot_circ(h, B_model)
    adot_obs = tle["adot_km_day"] if tle["adot_km_day"] is not None and tle["adot_km_day"] < 0 else None
    k_atm = None if (adot_obs is None or adot_model is None) else adot_obs / adot_model
    # у работающего аппарата торможение в TLE включает тягу двигателей (поддержание орбиты) — срок по нему не считается
    life_obs = None if (active or adot_obs is None or B_model is None) else D.lifetime_days(h, B_model, k=k_atm)
    if not active and adot_obs is not None and B_model is None:  # массы нет — профиль плотности с уровнем по торможению
        B_eff = abs(adot_obs) / abs(D.adot_circ(h, 1.0)); life_obs = D.lifetime_days(h, B_eff)
    life_model = None if B_model is None else D.lifetime_days(h, B_model)

    # масса-инерционные свойства тонкостенной оболочки в позе (оси орбиты)
    com, I, Ip, Iax = D.shell_mass_props(Vb, F, mass)
    major = None if Iax is None else Iax[:, 2]

    # выступающие элементы
    wings = []
    for w in spec.get("arrays") or []:
        m = w.get("mask")
        if m is None or not np.any(m):
            continue
        P = Vb[m]; c = P.mean(0); ext = np.ptp(P, 0)
        wings.append(dict(span_m=r(float(np.linalg.norm(ext)), 1), center_from_com_m=[r(v, 1) for v in (c - com)],
                          drive_axis_lvlh=None if w.get("axis") is None else [r(v, 3) for v in R @ np.asarray(w["axis"], float)]))

    # поправка к орбите для центра масс: θ реконструкции (сдвиг орбиты по положениям пятен полос, км) + сдвиг начала
    # модели при подгонке позы (x[4:6], м) + центр масс модели — всё в осях реконструкции (вдоль, поперёк, радиус)
    com_fit = C @ com; th = np.asarray(rc.get("theta", [0, 0, 0]), float) * 1e3
    ofile = ROOT / "results/orbits" / f"{name}.json"            # точнее: орбита Sentinel по GPS, углы полос в 3D
    if ofile.exists():
        q = json.loads(ofile.read_text()); f_ = q["fits"].get("gps3d") or q["fits"].get("gps") or q["fits"].get("tle")
        th = np.asarray(f_["theta_km"], float) * 1e3
    along, cross = float(th[0] + x[4] + com_fit[0]), float(th[1] + x[5] + com_fit[1])
    v_orb = float(np.linalg.norm(orb.teme([jd])[1][0])) * 1e3           # инерциальная скорость, м/с

    # точки захвата (досье увода): торцы вдоль оси наименьшего момента инерции (у ступени — сопло и переходник),
    # центр масс; для каждой — расстояние до центра масс и радиус торца
    grasp = []
    if Iax is not None:
        a_min = Iax[:, 0]; proj = (Vb - com) @ a_min
        for sgn, lab in ((1, "торец +"), (-1, "торец −")):
            sel = (sgn * proj) > (np.max(sgn * proj) - 0.05 * np.ptp(proj))
            if sel.any():
                c_end = Vb[sel].mean(0); rad = float(np.linalg.norm((Vb[sel] - c_end) - np.outer((Vb[sel] - c_end) @ a_min, a_min), axis=1).max())
                grasp.append(dict(point=lab, lvlh_from_com_m=[r(v) for v in (c_end - com)], dist_com_m=r(float(np.linalg.norm(c_end - com))),
                                  radius_m=r(rad)))

    # яркость групп материалов по полосам в найденной позе; остаток для поиска фрагментов
    P = AP.Problem(rc, spec, C); P.kappa = float(au.get("kappa") or 1.0); refl = {}; resid = {}
    Vfit = P.body(law, x)
    for b in P.bands:
        d = rc["bands"][b]
        occ = AP.regress(d, Vfit, spec["F"], spec["G"], P.ng, P.los, x[4:7])
        _, cc = AP.band_loss(d, occ)
        gm = float(np.nanmedian(d["g"]))
        refl[b] = [round(float(v) / gm, 3) if gm else None for v in cc]
        O = np.stack(occ, -1); resid[b] = (d["r"] - (O @ cc - d["g"] * O.sum(-1))) / float(d["s"])

    # фрагменты: пятна в остатке после модели, повторяющиеся в ≥ 3 полосах на одном и том же смещении в осях объекта
    # (в полосе b смещение Δ объекта → на снимке J_b·Δ/1000); перебор Δ по сетке, счёт — сумма |z| по полосам
    frags = []
    bands_f = [b for b in P.bands if b in resid]
    if len(bands_f) >= 3:
        k_img = float(np.median([np.linalg.norm(np.asarray(rc["bands"][b]["J"]) / 1000.0, 2) for b in bands_f]))
        lim = 140.0 / max(k_img, 1e-6); step = max(1.0, 8.0 / max(k_img, 1e-6))
        grid = np.arange(-lim, lim + step, step)
        best = []
        for dx in grid:
            for dy in grid:
                if math.hypot(dx, dy) < 1.5 * float(np.ptp(spec["V"], 0).max()):
                    continue                                       # сам объект
                zs = []
                for b in bands_f:
                    d = rc["bands"][b]; base = np.asarray(d["meas"][:2] if d.get("meas") else d["p0"], float)
                    p = base + (np.asarray(d["J"]) / 1000.0) @ (C @ np.array([dx, dy, 0.0]) + np.r_[x[4:6], 0.0])
                    i = np.unravel_index(np.argmin((d["X"] - p[0]) ** 2 + (d["Y"] - p[1]) ** 2), d["X"].shape)
                    if (d["X"][i] - p[0]) ** 2 + (d["Y"][i] - p[1]) ** 2 < (d["res"]) ** 2:
                        zs.append(abs(float(resid[b][i])))
                if len(zs) >= 3:
                    best.append((float(np.sum(sorted(zs)[-3:])), dx, dy, len(zs), int(np.sum(np.array(zs) > 4))))
        best.sort(reverse=True)
        thr = 3 * 5.0                                               # ≥ 5σ в трёх полосах в среднем
        frags = [dict(offset_lvlh_m=[r(dx, 1), r(dy, 1)], score=r(sc, 1), bands=n, bands_over_4sigma=n4)
                 for sc, dx, dy, n, n4 in best[:3] if sc > thr and n4 >= 3]

    out = dict(name=name, norad=norad, object=rc.get("obj"), gcat_name=g.get("Name"), type=spec["kind"], active=bool(active),
               model=spec["source"], dims_m=[r(v) for v in np.ptp(spec["V"], 0)], mass_kg=mass, mass_source=mass_src,
               area=dict(tumbling_m2=r(a_tumble), sun_m2=r(a_sun), ram_m2=r(a_ram), used_for_drag="tumbling" if tumbling else "ram"),
               A_over_m=None if not mass else dict(tumbling=r(a_tumble / mass, 5), ram=r(a_ram / mass, 5)),
               ballistic=dict(Cd=D.CD, B_model_m2_kg=r(B_model, 5), B_tle_m2_kg=r(tle["B_tle"], 5), bstar=tle["bstar"],
                              ratio_tle_to_model=None if (not B_model or tle["bstar"] <= 0) else r(tle["B_tle"] / B_model, 2),
                              mass_from_drag_kg=None if (active or tle["bstar"] <= 0) else r(D.CD * a_drag / tle["B_tle"], 0),
                              note=("B* ≤ 0 — нефизично (подгонка орбиты оператора или манёвры); " if tle["bstar"] <= 0 else "")
                                   + ("работает: B* включает тягу двигателей; " if active else "")
                                   + "B* в SGP4 поглощает и ошибку плотности атмосферы — отношение ≠ 1 не значит неверную площадь"),
               decay=dict(h_mean_km=r(h, 1), perigee_km=r(tle["perigee_km"], 1), apogee_km=r(tle["apogee_km"], 1), e=r(tle["e"], 5),
                          adot_tle_km_day=r(adot_obs, 4), adot_model_km_day=r(adot_model, 4), k_atm=r(k_atm, 2),
                          lifetime_days_from_tle=r(life_obs, 0), lifetime_days_model_mean_atm=r(life_model, 0),
                          note=("работает: торможение в TLE включает поддержание орбиты двигателями, срок — если "
                                "поддержание прекратится; " if active else "") + "круговая орбита на средней высоте"
                               + ("; эксцентриситет > 0.02 — оценка грубая" if tle["e"] > 0.02 else "")),
               mass_props=dict(com_from_model_origin_lvlh_m=[r(v) for v in com],
                               inertia_principal_kg_m2=None if Ip is None else [r(v, 0) for v in Ip],
                               principal_axes_lvlh=None if Iax is None else np.round(Iax.T, 3).tolist(),
                               major_axis_lvlh=None if major is None else [r(v, 3) for v in major],
                               model="тонкостенная оболочка: масса по площади поверхности"),
               protrusions=wings, grasp_points=grasp,
               fragments=dict(found=frags, note="поиск в окнах ±140 м снимка вокруг объекта; пятна ≥ 5σ в ≥ 3 полосах "
                                                 "на одном смещении в осях объекта"),
               orbit_correction=dict(along_m=r(along, 1), cross_m=r(cross, 1), radial_prior_m=r(th[2], 1), time_s=r(along / v_orb, 4),
                                     parts_m=dict(spots_theta=[r(v, 1) for v in th[:2]], pose_shift=[r(x[4], 1), r(x[5], 1)],
                                                  com=[r(v) for v in com_fit[:2]]),
                                     note="центр масс относительно прогноза TLE на момент снимка; радиус по одному ракурсу "
                                          "не определяется (держится априорным условием)"),
               attitude=dict(chosen=au["chosen"], anomalous=au.get("anomalous"), deviation_deg=au.get("deviation_deg"),
                             nearest_law=au.get("nearest_law"), consistent=au.get("consistent"), R_mesh_to_lvlh=np.round(R, 4).tolist()),
               reflectance_rel_bg=refl, group_names=["панели/элементы", "светлое", "тёмное"][:P.ng],
               tle=dict(epoch_jd=r(orb.epoch_jd, 5), age_days=r(jd - orb.epoch_jd, 2), line1=orb.line1, line2=orb.line2))
    pp = json.loads(Path(au["pose_pix"]).read_text()) if au.get("pose_pix") and Path(au["pose_pix"]).exists() else None
    if pp:                                     # поза по пикселям (scripts/pose_pix.py): σ, минимумы, проверки
        out["attitude"].update(method="по пикселям: вся SO(3) + приводы, градиент; опоры полос подогнаны (σ 25 м)",
                               sigma_rot_deg=[r(v, 2) for v in pp["sigma"]["rot_deg"]], drive_angles_deg=[r(v % 180, 1) for v in pp["drives_deg"]],
                               drive_sigma_deg=[r(v, 2) for v in pp["sigma"]["drives_deg"]],
                               wing_normal_from_sun_deg=[r(v, 1) for v in pp["wing_normal_from_sun_deg"]],
                               nearest_nadir_deg=r(pp["nearest_nadir"]["angle_deg"], 1), nearest_nadir_yaw_deg=r(pp["nearest_nadir"]["yaw_deg"], 1),
                               best_mode_prob=r(pp["modes"][0]["prob"], 3),
                               second_mode=dict(dL=r(pp["modes"][1]["dL"], 1), angle_deg=r(pp["modes"][1]["angle_deg"], 1)) if len(pp["modes"]) > 1 else None,
                               fit_vs_null=[r(pp["L"], 0), r(pp["null"], 0)], band_offsets_m=pp.get("band_offsets_m"),
                               old_law_pose=pp.get("old"))
    best = min(au["hypotheses"], key=lambda q: q["cv"]) if au.get("hypotheses") else {}
    if best.get("drive_angles_deg"):
        out["attitude"].update(drive_angles_deg=best["drive_angles_deg"], drive_sigma_deg=best["drive_sigma_deg"],
                               drive_vs_sun_deg=best.get("sun_dev_plate_deg"), drives_summary=best.get("drives_summary"))
    # вращение внутри пролёта
    want = a.spin == "all" or (a.spin == "auto" and (tumbling or au.get("anomalous")))
    if au.get("spin"):
        out["spin"] = au["spin"]
    elif want:
        S = SP.Spin(rc, spec, C, kappa=P.kappa)
        sp = S.solve(maxfev=a.maxfev, log=lambda s: print(f"  {name}:{s}", flush=True))
        ax = np.asarray(sp["axis_lvlh"]) if sp["axis_lvlh"] else None
        out["spin"] = dict(spinning=sp["spinning"], rate_deg_s=r(sp["rate_deg_s"], 1), sigma_deg_s=r(sp["rate_sigma_deg_s"], 1),
                           axis_lvlh=None if ax is None else [r(v, 3) for v in ax], axis_sigma_deg=r(sp["axis_sigma_jackknife_deg"], 0),
                           gain_eff_chi2=r(sp["gain"], 1), band_span_s=r(sp["span_s"], 2),
                           family_rates_deg_s=sp.get("family_rates_deg_s"), ambiguous=sp.get("ambiguous"),
                           angle_to_major_axis_deg=None if (ax is None or major is None) else
                           r(math.degrees(math.acos(min(1.0, abs(float(ax @ major))))), 0))
    else:
        out["spin"] = dict(note="не считалось: работающий аппарат со штатной ориентацией (--spin all — считать)")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    lt = out["decay"]["lifetime_days_from_tle"] or out["decay"]["lifetime_days_model_mean_atm"]
    print(f"{name}: {spec['kind']}, {'работает' if active else 'не работает'}; масса {mass} кг; сечение кувырк. {a_tumble:.1f} м², "
          f"поток {a_ram:.1f} м²; B мод. {out['ballistic']['B_model_m2_kg']} / TLE {out['ballistic']['B_tle_m2_kg']} м²/кг; "
          f"ȧ TLE {out['decay']['adot_tle_km_day']} / мод. {out['decay']['adot_model_km_day']} км/сут; срок ≈ "
          f"{'—' if lt is None else f'{lt / 365.25:.1f} г'}; поправка ЦМ {along:+.0f} м вдоль ({out['orbit_correction']['time_s']:+.4f} с), "
          f"{cross:+.0f} м поперёк; {au['chosen']}"
          + (f"; вращение {out['spin'].get('rate_deg_s')} ± {out['spin'].get('sigma_deg_s')} °/с" if "rate_deg_s" in out["spin"] else ""),
          flush=True)
    return out


SUMMARY = [("name", "name"), ("type", "type"), ("active", "active"), ("mass_kg", "mass_kg"),
           ("area_tumbling_m2", ("area", "tumbling_m2")), ("area_ram_m2", ("area", "ram_m2")),
           ("B_model", ("ballistic", "B_model_m2_kg")), ("B_tle", ("ballistic", "B_tle_m2_kg")),
           ("adot_tle_km_day", ("decay", "adot_tle_km_day")), ("adot_model_km_day", ("decay", "adot_model_km_day")),
           ("k_atm", ("decay", "k_atm")), ("life_tle_days", ("decay", "lifetime_days_from_tle")),
           ("life_model_days", ("decay", "lifetime_days_model_mean_atm")),
           ("corr_along_m", ("orbit_correction", "along_m")), ("corr_cross_m", ("orbit_correction", "cross_m")),
           ("corr_time_s", ("orbit_correction", "time_s")), ("law", ("attitude", "chosen")),
           ("anomalous", ("attitude", "anomalous")), ("spin_deg_s", ("spin", "rate_deg_s")), ("spin_sigma", ("spin", "sigma_deg_s")),
           ("spinning", ("spin", "spinning"))]


def summary(out_dir):
    rows = []
    for f in sorted(Path(out_dir).glob("*.json")):
        c = json.loads(f.read_text()); row = {}
        for k, p in SUMMARY:
            v = c
            for q in (p if isinstance(p, tuple) else (p,)):
                v = v.get(q) if isinstance(v, dict) else None
            row[k] = v
        rows.append(row)
    with open(Path(out_dir) / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k, _ in SUMMARY]); w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    a = ap.parse_args()
    names = a.names or sorted(p.stem for p in Path(a.auto).glob("*.json") if not p.name.startswith("inject"))
    for n in names:
        try:
            card(n, a)
        except Exception as ex:                    # noqa: BLE001
            import traceback; traceback.print_exc()
            print(n, "ОШИБКА", type(ex).__name__, ex, flush=True)
    print(f"сводка: {len(summary(a.out))} карточек → {Path(a.out) / 'summary.csv'}", flush=True)

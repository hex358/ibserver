"""Карточка объекта для каталога и увода мусора (PLAN_characterization §2.5) — одна для всех объектов.

  python scripts/debris_card.py ISS_20261118T222808 STARLINK-30580_20261120T111642 ...   (без аргументов — все)
Из готовой автоматической позы (results/auto/<имя>.json) и модели каталога:
  * модель, размеры, масса (GCAT: сухая масса), тип;
  * площадь сечения: средняя при беспорядочном вращении (проекции на 400 направлений), в найденной позе — к Солнцу
    и к набегающему потоку (вдоль скорости); отношение площадь/масса A/m (торможение, давление света);
  * центр масс тонкостенной оболочки (по площади поверхности) относительно геометрического центра модели;
  * поправка к орбите: сдвиг вдоль следа и поперёк (м) от прогноза TLE на момент снимка, эквивалентный сдвиг по
    времени (с); радиус из одного ракурса не определяется;
  * яркость (отражательная способность) групп материалов по полосам в найденной позе (НМНК);
  * ориентация: закон / нештатность / углы приводов панелей с погрешностями; вращение — если считалось (spin).
Пишет results/cards/<имя>.json.
"""
import json, math, pickle, sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import autopose as AP, modelcat as MC, scenegeom as SG

V_ORBIT = 7.66e3                       # м/с, порядок для НОО; точнее — из орбиты объекта


def proj_area(V, d):
    """Площадь проекции (выпуклая оболочка силуэта) на плоскость ⟂ d, м²."""
    d = d / np.linalg.norm(d); e1 = np.cross(d, [0.0, 0, 1] if abs(d[2]) < 0.9 else [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1); P = np.c_[V @ e1, V @ e2]
    try:
        return float(ConvexHull(P).volume)
    except Exception:
        return 0.0


def card(name):
    au = json.loads((ROOT / "results/auto" / f"{name}.json").read_text())
    rc = pickle.loads((ROOT / "results/shape" / f"{name}.pkl").read_bytes()); MR.prepare(rc)
    norad = int(rc.get("cat", 0)); spec = MC.spec_for(norad, name); g = spec.get("gcat") or {}
    C = np.asarray(au.get("C_lvlh_to_fit", np.eye(3).tolist())); sun_l = np.asarray(au.get("sun_lvlh", (C.T @ np.asarray(rc["sun"])).tolist()))
    law, x = au["law"], np.asarray(au["x"], float)
    Vb, R, _ = AP.pose_body(spec, law, x, sun_l)
    V0 = spec["V"]
    # площадь: беспорядочное вращение — среднее по сфере; в позе — к Солнцу и вдоль скорости (оси орбиты: x — вдоль)
    k = np.arange(400) + 0.5; th = np.arccos(1 - 2 * k / 400); ph = math.pi * (1 + 5 ** 0.5) * k
    dirs = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    a_tumble = float(np.mean([proj_area(V0, d) for d in dirs]))
    a_sun = proj_area(Vb, sun_l); a_ram = proj_area(Vb, np.array([1.0, 0, 0]))
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    mass = num(g.get("DryMass")) or num(g.get("Mass"))
    # центр масс тонкостенной оболочки: по площади граней
    F = spec["F"]; T = V0[F]; A = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
    com = (T.mean(1) * A[:, None]).sum(0) / max(A.sum(), 1e-9); geo_c = 0.5 * (V0.min(0) + V0.max(0))
    com_body = (R @ (com - geo_c)).tolist()                       # в осях орбиты (вдоль, поперёк, радиус), м
    # яркость групп по полосам в найденной позе
    P = AP.Problem(rc, spec, C); refl = {}
    Vfit = P.body(law, x)
    for b in P.bands:
        d = rc["bands"][b]
        occ = AP.regress(d, Vfit, spec["F"], spec["G"], P.ng, P.los, x[4:7])
        _, c = AP.band_loss(d, occ)
        gm = float(np.nanmedian(d["g"]))
        refl[b] = [round(float(v) / gm, 3) if gm else None for v in c]     # в долях яркости фона (≈ ρ_объекта / ρ_фона)
    dx, dy = float(x[4]), float(x[5])
    out = dict(name=name, norad=norad, object=rc.get("obj"), gcat_name=g.get("Name"), type=spec["kind"], model=spec["source"],
               dims_m=[round(float(v), 2) for v in np.ptp(V0, 0)], mass_kg=mass,
               area_tumbling_m2=round(a_tumble, 2), area_sun_m2=round(a_sun, 2), area_ram_m2=round(a_ram, 2),
               A_over_m_tumbling=None if not mass else round(a_tumble / mass, 5),
               A_over_m_ram=None if not mass else round(a_ram / mass, 5),
               com_offset_thin_shell_m=[round(v, 2) for v in com_body],
               orbit_correction=dict(along_m=round(dx, 1), cross_m=round(dy, 1), time_s=round(dx / V_ORBIT, 4),
                                     note="сдвиг фотоцентра от прогноза TLE на момент снимка; радиус не определяется"),
               attitude=dict(chosen=au["chosen"], anomalous=au.get("anomalous"), deviation_deg=au.get("deviation_deg"),
                             nearest_law=au.get("nearest_law"), consistent=au.get("consistent"),
                             R_mesh_to_lvlh=np.round(R, 4).tolist()),
               reflectance_rel_bg=refl, group_names=["панели/элементы", "светлое", "тёмное"][:P.ng])
    best = min(au["hypotheses"], key=lambda r: r["cv"])
    if best.get("drive_angles_deg"):
        out["attitude"].update(drive_angles_deg=best["drive_angles_deg"], drive_sigma_deg=best["drive_sigma_deg"],
                               drive_vs_sun_deg=best.get("sun_dev_plate_deg"), drives_summary=best.get("drives_summary"))
    if au.get("spin"):
        out["spin"] = au["spin"]
    (ROOT / "results/cards").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/cards" / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"{name}: {out['model'][:60]}; {out['dims_m']} м, масса {mass} кг; сечение (кувырк.) {a_tumble:.1f} м², A/m "
          f"{out['A_over_m_tumbling']} м²/кг; поправка орбиты {dx:+.0f} м вдоль ({out['orbit_correction']['time_s']:+.4f} с), "
          f"{dy:+.0f} м поперёк; {au['chosen']}", flush=True)
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in (ROOT / "results/auto").glob("*.json") if not p.name.startswith("inject"))
    for n in names:
        try:
            card(n)
        except Exception as ex:                    # noqa: BLE001
            print(n, "ОШИБКА", type(ex).__name__, ex, flush=True)

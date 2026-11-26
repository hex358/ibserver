"""Поза объекта по пикселям (s2sat/diffpose.py) и доказательства, что она правильная.

  python scripts/pose_pix.py STARLINK-30245_20261120T110322 [--n-rot 3000] [--checks] [--write]
Опоры полос — геометрические положения по точной орбите Sentinel (results/orbits/<имя>.json, gps3d: GPS + 3D-углы),
а не яркие пики в каждой полосе (пики в разных полосах расходились до 84 м).
Вывод results/pose_pix/<имя>.json:
  * поза: поворот корпуса (тело → орбитальные оси), углы приводов каждого крыла, сдвиг; σ по Гессиану;
  * другие минимумы и их вероятность; поза прежней схемы (законы) на тех же опорах — для сравнения;
  * --checks: перекрёстная проверка (поза по одной группе полос предсказывает пиксели другой) и подстановка
    (снимок, нарисованный в известной позе с шумом снимка, — найденная поза против истинной);
  * описание: угол до ближайшей ориентации «на Землю» с любым рысканьем, отклонение крыльев от Солнца.
--write: results/auto/<имя>.json (прежний — в results/auto_laws/) и опоры в results/shape/<имя>.pkl (прежний — в
results/shape_peaks/) — по ним рисуют листы и рендеры.
"""
import argparse, copy, json, math, pickle, shutil, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import autopose as AP, modelcat as MC, scenegeom as SG
from s2sat.diffmodel import transfer_r
from s2sat.diffpose import DiffPose, rot_angle

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--n-rot", type=int, default=3000); ap.add_argument("--checks", action="store_true")
ap.add_argument("--write", action="store_true"); ap.add_argument("--shade", action="store_true"); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(a.threads)
n = a.name; t_start = time.time(); rng = np.random.default_rng(a.seed)
OUT = ROOT / ("results/pose_pix_shade" if "--shade" in sys.argv else "results/pose_pix"); OUT.mkdir(parents=True, exist_ok=True)


def log(*s):
    print(*s, flush=True)


rc = pickle.loads((ROOT / "results/shape" / f"{n}.pkl").read_bytes())
rc_raw = copy.deepcopy(rc)
orb = ROOT / "results/orbits" / f"{n}.json"; anchors = "пики яркости в каждой полосе"
if orb.exists():
    q = json.loads(orb.read_text()); f = q["fits"].get("gps3d") or q["fits"].get("gps") or {}
    bp = f.get("band_positions")
    if bp and all(b in bp for b in rc["bands"]):
        sh = {b: np.round(np.asarray(rc["bands"][b].get("meas") or rc["bands"][b]["p0"], float)[:2] - bp[b], 1).tolist() for b in rc["bands"]}
        for b, d in rc["bands"].items():
            d["p0"] = np.asarray(bp[b], float); d.pop("meas", None)
        anchors = f"геометрия по орбите Sentinel ({'GPS + 3D-углы' if 'gps3d' in q['fits'] else 'GPS'}); прежние опоры смещены на " + \
                  ", ".join(f"{b} {np.hypot(*v):.0f} м" for b, v in sh.items())
log(f"{n}: опоры полос — {anchors}")
MR.prepare(rc)
spec = MC.spec_for(int(rc["cat"]), n); C = SG.lvlh_to_fit(n, rc)
sun_l = C.T @ np.asarray(rc["sun"], float)
Df = DiffPose(rc, spec, C, shade=a.shade); Dc = DiffPose(rc, spec, C, coarse=True, shade=a.shade)
log(f"   параметры: поворот корпуса 3 + приводы {Df.nd} + сдвиг 2; пикселей {Df.npix}; невязка без объекта {Df.null():.0f}")

# --- 1. поиск по всей SO(3) × приводам ---
t0 = time.time()
modes, _ = Dc.search(n_rot=a.n_rot, fine=Df, log=log)
best = modes[0]; xb = best["x"]
sig, Cov, kap = Df.hessian_sigma(xb)
for m in modes:
    m["dL"] = m["L"] - best["L"]; m["p"] = math.exp(-0.5 * m["dL"] / kap)
Z = sum(m["p"] for m in modes)
log(f"   найдено за {time.time() - t0:.0f} с: невязка {best['L']:.0f} (без объекта {Df.null():.0f}), κ = {kap:.2f}")
for k, m in enumerate(modes[:6]):
    log(f"   минимум {k}: невязка {m['L']:.0f} (+{m['dL']:.0f}), вероятность {m['p'] / Z:.3f}, поворот от лучшего "
        f"{rot_angle(m['R'], best['R']):.0f}°, приводы {np.round(np.degrees(m['drives']) % 180).tolist()}°")
sd = np.degrees(sig[:3]); sdd = np.degrees(sig[3:3 + Df.nd]); sp = sig[-2:]
log(f"   σ поворота (оси: вдоль, поперёк, радиус) {np.round(sd, 1).tolist()}°; σ приводов {np.round(sdd, 1).tolist()}°; σ сдвига {np.round(sp, 2).tolist()} м")
near, yaw = Df.nearest_nadir(best["R"]); drv_sun = Df.drive_report(xb, sun_l)
log(f"   описание: до ближайшей ориентации «на Землю» {near:.1f}° (рысканье {yaw:.1f}°); нормали крыльев от Солнца {np.round(drv_sun, 1).tolist()}°")

# --- 2. прежняя схема (законы) на тех же опорах ---
old = None
fa = ROOT / "results/auto" / f"{n}.json"
src = ROOT / "results/auto_laws" / f"{n}.json"
au = json.loads((src if src.exists() else fa).read_text()) if (src.exists() or fa.exists()) else None
if au and not au["law"].get("pix"):
    h, x = au["law"], np.asarray(au["x"], float)
    _, Rm, _ = AP.pose_body(spec, h, x, sun_l); Rb = Rm @ np.asarray(spec["M2B"], float).T
    L_ = AP.drive_layout(spec)
    if h.get("arrays") == "est" and len(x) >= 7 + Df.nd:
        ang = x[7:7 + Df.nd]
    else:
        ang = AP.mode_angles(spec, Rm, h.get("arrays", "fixed"), sun_l, x)
    from scipy.spatial.transform import Rotation as Ro
    xo = np.r_[Ro.from_matrix(Rb).as_rotvec(), ang, x[4], x[5]]
    Lo = float(Df.L(xo)[0])
    xo_loc, Lo_loc = Df.polish(xo)
    old = dict(name=au["chosen"], L=Lo, L_local=Lo_loc, angle_to_new_deg=rot_angle(Rb, best["R"]),
               drives_deg=np.degrees(ang).tolist(), local_moves_deg=rot_angle(Df.R_of(xo_loc), Rb))
    log(f"   прежняя схема «{au['chosen']}»: невязка {Lo:.0f} (+{Lo - best['L']:.0f}); от неё же градиентом — {Lo_loc:.0f}, "
        f"поворот уходит на {old['local_moves_deg']:.0f}°; до новой позы {old['angle_to_new_deg']:.0f}°")

out = dict(name=n, anchors=anchors, x=xb.tolist(), L=best["L"], null=Df.null(), kappa=kap, npix=Df.npix,
           R_body_to_lvlh=best["R"].tolist(), drives_deg=np.degrees(best["drives"]).tolist(),
           sigma=dict(rot_deg=sd.tolist(), drives_deg=sdd.tolist(), pos_m=sp.tolist()),
           modes=[dict(L=m["L"], dL=m["dL"], prob=m["p"] / Z, angle_deg=rot_angle(m["R"], best["R"]),
                       drives_deg=np.degrees(m["drives"]).tolist()) for m in modes],
           nearest_nadir=dict(angle_deg=near, yaw_deg=yaw), wing_normal_from_sun_deg=drv_sun, old=old, C_lvlh_to_fit=C.tolist(),
           sun_lvlh=sun_l.tolist())

# --- 3. проверки ---
if a.checks:
    bs = sorted(Df.sc.bands, key=lambda b: Df.sc.t[b]); grp = [[bs[0]]]
    for b in bs[1:]:
        (grp[-1].append(b) if Df.sc.t[b] - Df.sc.t[grp[-1][-1]] <= 1.0 else grp.append([b]))
    if len(grp) < 2:
        grp = [[b for b in bs if Df.sc.B[b].res <= 10], [b for b in bs if Df.sc.B[b].res > 10]]
    cv = []
    for test in grp:
        train = [b for b in bs if b not in test]
        if not train or not test:
            continue
        mt, _ = Dc.search(bands=train, n_rot=max(1000, a.n_rot // 2), top=128, fine=Df, n_polish=8, log=lambda s: None)
        xt = mt[0]["x"]
        pr, _ = Df.predict(xt, train, test); pb, _ = Df.predict(xb, train, test)
        po = Df.predict(xo, train, test)[0] if old else None
        tot = lambda p: (sum(v["chi_null"] for v in p.values()), sum(v["chi_model"] for v in p.values()),
                         1 - sum(v["chi_model"] for v in p.values()) / sum(v["chi_null"] for v in p.values()))
        r_ = dict(train=train, test=test, angle_train_vs_all_deg=rot_angle(Df.R_of(xt), best["R"]),
                  drives_train_deg=np.degrees(xt[3:3 + Df.nd]).tolist(),
                  explained_pose_from_train=tot(pr)[2], explained_pose_all=tot(pb)[2], explained_old=tot(po)[2] if po else None,
                  per_band={b: v["explained"] for b, v in pr.items()})
        cv.append(r_)
        log(f"   перекрёстная: поза по {train} → предсказание {test}: объяснено {r_['explained_pose_from_train'] * 100:.0f}% "
            f"сигнала (поза по всем полосам {r_['explained_pose_all'] * 100:.0f}%"
            + (f", прежняя схема {r_['explained_old'] * 100:.0f}%" if po else "") + f"); поза по части полос отличается от общей на "
            f"{r_['angle_train_vs_all_deg']:.0f}°")
    out["cross_validation"] = cv
    # подстановка: известная поза → снимок модели + шум → поиск
    _, rho = Df.predict(xb, bs, bs[:1]); inj = []
    from scipy.spatial.transform import Rotation as Ro
    truths = [("найденная поза", xb)] + [(f"случайная поза {k + 1}", np.r_[Ro.random(random_state=int(rng.integers(1 << 30))).as_rotvec(),
                                                                       rng.uniform(0, math.pi, Df.nd), rng.normal(0, 2.0, 2)]) for k in range(2)]
    for lab, xt in truths:
        o1 = Df.synth(xt, rho, math.sqrt(kap), rng); o2 = {b: Dc.sc.B[b].r for b in Dc.sc.bands}; transfer_r(Df.sc, Dc.sc)
        mi, _ = Dc.search(n_rot=max(1000, a.n_rot // 2), top=128, fine=Df, n_polish=8, log=lambda s: None)
        xe = mi[0]["x"]; Lt = float(Df.L(xt)[0]); Le = mi[0]["L"]
        dd = (np.degrees(xe[3:3 + Df.nd] - xt[3:3 + Df.nd]) + 90) % 180 - 90
        r_ = dict(truth=lab, angle_err_deg=rot_angle(Df.R_of(xe), AP.rotvec(xt[:3])), drive_err_deg=dd.tolist(), L_found=Le, L_truth=Lt,
                  pos_err_m=(xe[-2:] - xt[-2:]).tolist())
        inj.append(r_)
        Df.restore(o1); Dc.restore(o2)
        log(f"   подстановка ({lab}): ошибка поворота {r_['angle_err_deg']:.1f}°, приводов {np.round(dd, 1).tolist()}°; "
            f"невязка найденной {Le:.0f} / истинной {Lt:.0f}" + (" — снимок не различает" if Le < Lt - 1 and r_['angle_err_deg'] > 10 else ""))
    out["injection"] = inj

out["seconds"] = time.time() - t_start
(OUT / f"{n}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
log(f"   → {OUT / (n + '.json')} ({out['seconds']:.0f} с)")

if a.write:
    (ROOT / "results/auto_laws").mkdir(exist_ok=True); (ROOT / "results/shape_peaks").mkdir(exist_ok=True)
    if fa.exists() and not src.exists():
        shutil.copy(fa, src)
    sp_ = ROOT / "results/shape_peaks" / f"{n}.pkl"
    if not sp_.exists():
        sp_.write_bytes(pickle.dumps(rc_raw))
    rc_w = copy.deepcopy(rc_raw)
    for b, d in rc_w["bands"].items():
        d["p0"] = np.asarray(rc["bands"][b]["p0"], float); d.pop("meas", None)
    (ROOT / "results/shape" / f"{n}.pkl").write_bytes(pickle.dumps(rc_w))
    xa = np.r_[xb[:3], 0.0, xb[-2], xb[-1], 0.0, xb[3:3 + Df.nd]]
    desc = f"по пикселям: до «на Землю» {near:.0f}° (рысканье {yaw:.0f}°), σ поворота {np.round(sd, 1).tolist()}°"
    au_new = dict(kind=spec.get("kind"), source=spec.get("source"), name=n, law=dict(name="поза по пикселям", pix=True), x=xa.tolist(),
                  chosen=desc, anomalous=False, margin_eff=float("nan"), hypotheses=[], C_lvlh_to_fit=C.tolist(),
                  sun_lvlh=sun_l.tolist(), R=(best["R"] @ np.asarray(spec["M2B"], float)).tolist(),
                  array_angles_deg=np.degrees(best["drives"]).tolist(), pose_pix=str(OUT / f"{n}.json"))
    fa.write_text(json.dumps(au_new, ensure_ascii=False, indent=1))
    log(f"   записано: {fa}, опоры — {ROOT / 'results/shape' / (n + '.pkl')}")

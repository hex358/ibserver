"""Сравнение Equant и перебора на задаче «пересечения поля зрения Sentinel-2 × каталог».

Запуск на машине с GPU, где собран Equant (build.bat k0 / build-linux.sh k0):
  set EQUANT_ROOT=D:\\...\\cull   (ветка decay-study)
  python scripts/bench_equant.py --hours 24 --radius 650

Что делает:
 1. TLE снимка (data/tle_snapshots/...) → OMM JSON (поля, которые читает equant.screen).
 2. Перебор (s2sat.crossings.find_crossings, шаг 30 с) → эталонный список пересечений и время.
 3. Equant: equant.screen(каталог, horizon, radius) → окна сближений; оставляем пары с Sentinel-2.
    Внутри каждого окна точный тест «объект в плоскости веера» (тот же, что в переборе).
 4. Печатает: время перебора, время Equant (+ уточнение), полноту Equant относительно перебора,
    сколько пар/окон Equant выдал на одно настоящее пересечение.

Радиус: пересечение поля зрения возможно только при дальности ≤ ~600 км
(объект на 200 км, угол от надира 10.5°: (786−200)/cos 10.5° ≈ 596 км), поэтому окна
сближений Equant с радиусом 650 км гарантированно содержат все пересечения.
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

from s2sat import orbits as O, crossings as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", default=None)
ap.add_argument("--hours", type=float, default=24.0)
ap.add_argument("--radius", type=float, default=650.0)
ap.add_argument("--max-perigee", type=float, default=1000.0)
ap.add_argument("--equant-root", default=os.environ.get("EQUANT_ROOT"))
a = ap.parse_args()

snaps = sorted((ROOT / "data/tle_snapshots").iterdir())
snap = (ROOT / "data/tle_snapshots" / a.snapshot) if a.snapshot else snaps[-1]
sents = O.parse_tle((snap / "sentinel2.tle").read_text())
objs = [o for o in O.parse_tle((snap / "gp_active.tle").read_text())
        if o.perigee_km < a.max_perigee and not o.name.startswith("SENTINEL-2")]
print(f"snapshot {snap.name}: {len(sents)} Sentinel-2, {len(objs)} LEO objects")


def omm(o: O.Orbit):
    l1, l2 = o.line1, o.line2
    yy = int(l1[18:20]); year = 2000 + yy if yy < 57 else 1900 + yy
    doy = float(l1[20:32])
    ep = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=doy - 1)
    def expo(s):
        s = s.strip()
        if not s: return 0.0
        sign = -1.0 if s[0] == "-" else 1.0
        s = s.lstrip("+-"); return sign * float("0." + s[:-2]) * 10 ** int(s[-2:])
    return dict(OBJECT_NAME=o.name, NORAD_CAT_ID=int(o.catnr), EPOCH=ep.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                MEAN_MOTION=float(l2[52:63]), ECCENTRICITY=float("0." + l2[26:33].strip()),
                INCLINATION=float(l2[8:16]), RA_OF_ASC_NODE=float(l2[17:25]), ARG_OF_PERICENTER=float(l2[34:42]),
                MEAN_ANOMALY=float(l2[43:51]), BSTAR=expo(l1[53:61]), MEAN_MOTION_DOT=float(l1[33:43]),
                MEAN_MOTION_DDOT=expo(l1[44:52]))


cat = [omm(o) for o in sents + objs]
cat_path = ROOT / "data" / f"omm_{snap.name}.json"
cat_path.write_text(json.dumps(cat))

# эпоха отсчёта: самая поздняя эпоха каталога (так делает Equant: «minutes after the common epoch»)
j0 = max(o.epoch_jd for o in sents + objs)
j1 = j0 + a.hours / 24

t = time.time()
ref = C.find_crossings(sents, objs, j0, j1, step_s=30)
t_brute = time.time() - t
print(f"brute force: {len(ref)} crossings in {t_brute:.1f} s")

if not a.equant_root:
    print("EQUANT_ROOT не задан: Equant не запускался (нужен GPU и собранная библиотека).")
    sys.exit(0)
sys.path.insert(0, str(Path(a.equant_root) / "python"))
import equant  # noqa: E402

sent_ids = {int(s.catnr) for s in sents}
t = time.time()
res = equant.screen(str(cat_path), horizon_minutes=a.hours * 60, radius_km=a.radius)
t_eq = time.time() - t
W = res.windows
mask = np.isin(W["norad_a"], list(sent_ids)) | np.isin(W["norad_b"], list(sent_ids))
W = W[mask]
print(f"Equant: {len(res.windows)} windows total, {len(W)} with Sentinel-2, {t_eq:.2f} s; epoch {res.epoch}")

SAT = {int(s.catnr): s for s in sents}; OBJ = {int(o.catnr): o for o in objs}
ep_jd = O.jd_from_dt(res.epoch)
t = time.time()
found = []
for w in W:
    a_, b_ = int(w["norad_a"]), int(w["norad_b"])
    s, o = (SAT[a_], OBJ.get(b_)) if a_ in SAT else (SAT.get(b_), OBJ.get(a_))
    if s is None or o is None:
        continue
    ja = ep_jd + float(w["start_minutes"]) / 1440; jb = ep_jd + float(w["end_minutes"]) / 1440
    found += C.find_crossings([s], [o], ja - 60 / 86400, jb + 60 / 86400, step_s=10)
t_ref = time.time() - t
key = lambda e: (e["sent"], e["cat"], round(e["jd"] * 86400 / 5))
got = {key(e) for e in found}
miss = [e for e in ref if key(e) not in got]
print(f"Equant + refinement: {len(found)} crossings, refine {t_ref:.1f} s; recall vs brute {1 - len(miss) / max(len(ref), 1):.4f} "
      f"({len(miss)} missed); windows per crossing {len(W) / max(len(ref), 1):.1f}")
print(f"time: brute {t_brute:.1f} s  vs  Equant {t_eq:.2f} s + refine {t_ref:.1f} s")

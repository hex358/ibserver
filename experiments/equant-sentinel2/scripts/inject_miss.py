"""Подстановка: виден ли на ЭТОМ снимке Starlink обычной яркости? (проверка «не видно» данными, THEORY §14)

  python scripts/inject_miss.py results/events_v9 --obj STARLINK-32251,STARLINK-36150 [--f 0.6]

Для пролёта из результатов v9 в реальный снимок вставляется синтетический Starlink по модели заслонения
(ρ = 0.28, доля пикселя f, длина изображения по геометрии пролёта) в точки полос при фазе 0 со сдвигом;
тем же детектором ищется в узком окне вокруг вставки (±1 км фазы, ±600 м поперёк), счёт сравнивается с
порогом настоящего поиска v9 по этому пролёту (тот же нуль). Найден подставной, а настоящий нет —
снимок позволяет видеть, а спутника не было там, где искали. Не найден и подставной — сцена ограничивает.
"""
import argparse, glob, json, math, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from s2sat import orbits as O, pipeline as P
from s2sat.threshold import threshold

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--obj", default="")
ap.add_argument("--f", type=float, default=0.6)
ap.add_argument("--rho", type=float, default=0.28)
ap.add_argument("--until", default="20261121T1217")
ap.add_argument("--shift", default="300,-200")
a = ap.parse_args()
snaps = sorted(p for p in (ROOT / "data/tle_snapshots").iterdir() if p.is_dir() and p.name <= a.until)


def load(pat):
    best = {}
    for sd in snaps:
        for f in sd.glob(pat):
            for o in O.parse_tle(f.read_text()):
                if o.catnr not in best or o.epoch_jd > best[o.catnr].epoch_jd:
                    best[o.catnr] = o
    return best


S2 = {o.name: o for o in load("sentinel2.tle").values()}
GP, SG = load("gp_*.tle"), load("supgp_*.tle")
want = set(x for x in a.obj.split(",") if x)
sx, sy = map(float, a.shift.split(","))
for f in sorted(Path(a.dir).glob("*.json")):
    x = json.loads(f.read_text()); e, r = x["event"], x["result"]
    if want and e["obj"].split()[0] not in want:
        continue
    if "score" not in r or "null_max" not in r:
        continue
    orb = SG.get(e["cat"]) if e.get("supgp") and e["cat"] in SG else GP.get(e["cat"])
    n = e["obj"].upper()
    try:
        area = 120.0 if int(n.split("-")[1].split()[0]) >= 30000 else 30.0
    except Exception:
        area = 30.0
    geo = P.image_geometry(e, area)
    L = int(max(1, min(25, round(geo["along_px"])))); W = int(max(1, round(geo["across_px"])))
    inj = dict(dm=0.0, shift=(sx, sy), len10=L, wid10=W, rho_dn=a.rho * 1e4 + 1000.0, f=a.f)
    rr = P.run_event(S2[e["sent"]], orb, e, Rm=1500.0, minb=3, dM_max=1.0, dM_center=0.0, Rc=600.0, tol_m=30.0,
                     area_m2=area, inject=inj, n_null=2, stretch=True)
    thr = threshold(r)
    if "score" not in rr:
        print(f"{f.stem}: no score {rr.get('reasons')}", flush=True); continue
    ok = rr["score"] >= thr
    print(f"{f.stem}: injected Starlink (f {a.f}, {L}x{W} px): score {rr['score']:.1f} vs v9 threshold {thr:.1f} -> "
          f"{'DETECTED' if ok else 'not detected'};  real best {r['score']:.1f}  (inj found at dM {rr['dM']:+.2f}, shift {rr['sx']:.0f},{rr['sy']:.0f})", flush=True)

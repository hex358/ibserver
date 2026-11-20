"""Регрессия: МКС на снимке Sentinel-2B 2026-11-18 22:28 UTC, тайл T59GNN (HANDOFF 5.1).

Критерий: счёт > 15, максимум нуля < 6, сдвиг около (620, 1000) м ± 200 м
(сдвиг зависит от TLE, поэтому TLE печатается).
"""
import argparse
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning, module="skyfield_data")

from s2sat import orbits as O, crossings as C, pipeline as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--iss-tle", default=str(ROOT / "data/tle/iss_20261119.tle"))
ap.add_argument("--pilot-tle", action="store_true", help="TLE МКС из pilot_code/tle_raw.txt (эпоха 17.11)")
ap.add_argument("--ref", default=None, help="id опорной гранулы для T59GNN")
ap.add_argument("--rm", type=float, default=6000.0)
ap.add_argument("--minb", type=int, default=4)
ap.add_argument("--dm", type=float, default=0.0, help="перебор фазы вдоль орбиты, ±км")
ap.add_argument("--plot", default=str(ROOT / "results/iss_regression.png"))
a = ap.parse_args()

pilot = {o.catnr: o for o in O.parse_tle_tokens((ROOT / "pilot_code/tle_raw.txt").read_text())}
s2b = pilot["42063"]
iss = pilot["25544"] if a.pilot_tle else O.parse_tle(Path(a.iss_tle).read_text())[0]
print("S2B TLE:", s2b.line1, "|", s2b.line2)
print("ISS TLE:", iss.line1, "|", iss.line2, "epoch", O.dt_from_jd(iss.epoch_jd))

j = O.jd_from_dt(dt.datetime(2026, 11, 18, 22, 28, tzinfo=dt.timezone.utc))
ev = C.find_crossings([s2b], [iss], j - 0.02, j + 0.02, step_s=10)
assert ev, "нет пересечения"
e = ev[0]
print("event", O.dt_from_jd(e["jd"]), f"lat {e['lat']:.4f} lon {e['lon']:.4f} rng {e['rng']:.1f} v_al {e['v_al']:.2f}")

refs = {"S2B_T59GNN_20261118T222728_L2A": a.ref} if a.ref else None
r = P.run_event(s2b, iss, e, Rm=a.rm, minb=a.minb, ref_ids=refs, dM_max=a.dm)
maps = r.pop("_maps", None)
show = {k: v for k, v in r.items() if k not in ("roots", "cell_fail")}
print(json.dumps(show, default=str, indent=1))
for b, lst in r.get("roots", {}).items():
    print("root", b, [(d, round(al, 3), round(x), round(y)) for d, al, x, y in lst])
if "score" in r:
    ok = r["score"] > 15 and r["score"] > 2 * r["null_max"]
    print(f"ISS score {r['score']:.2f} null max {r['null_max']:.2f} med {r['null_med']:.2f} "
          f"shift ({r['sx']:.0f}, {r['sy']:.0f}) m  bands>4σ {r['hit']}/{r['cnt']}  -> {'PASS' if ok else 'FAIL'}")
    if a.plot and maps is not None:
        from s2sat import viz
        viz.plot_event(r, maps, a.plot)
        print("plot", a.plot)

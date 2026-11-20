"""Поиск и проверка пересечений Sentinel-2 × объекты по снимку орбит из data/tle_snapshots.

Пример (SL3 из HANDOFF 5.2):
  python scripts/run_events.py --obj STARLINK-32519 --start 2026-11-19T02:00 --end 2026-11-19T04:00
Без --obj: все дневные события Starlink за окно, медленные первыми (--limit N).
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
ap.add_argument("--snapshot", default=None, help="папка в data/tle_snapshots (по умолчанию последняя)")
ap.add_argument("--orbits", default="supgp_starlink.tle", help="файл орбит объектов в снимке")
ap.add_argument("--obj", default=None)
ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
ap.add_argument("--sent", default="SENTINEL-2")
ap.add_argument("--rm", type=float, default=1500.0)
ap.add_argument("--minb", type=int, default=3)
ap.add_argument("--limit", type=int, default=5)
ap.add_argument("--max-cloud", type=float, default=30.0)
ap.add_argument("--min-sun", type=float, default=8.0)
ap.add_argument("--min-v", type=float, default=1.5)
ap.add_argument("--dm", type=float, default=20.0, help="перебор фазы вдоль орбиты, ±км")
ap.add_argument("--max-lat", type=float, default=70.0)
ap.add_argument("--agree", default=None, help="второй файл орбит для проверки согласия")
ap.add_argument("--max-disagree", type=float, default=2.0, help="км на земле")
ap.add_argument("--out", default=str(ROOT / "results/events"))
a = ap.parse_args()

snaps = sorted((ROOT / "data/tle_snapshots").iterdir())
snap = (ROOT / "data/tle_snapshots" / a.snapshot) if a.snapshot else snaps[-1]
print("snapshot", snap.name)
sents = [s for s in O.parse_tle((snap / "sentinel2.tle").read_text()) if s.name.startswith(a.sent)]
objs = O.parse_tle((snap / a.orbits).read_text())
if a.obj:
    objs = [o for o in objs if o.name.split()[0] == a.obj or o.catnr == a.obj]
print("sentinels", [s.name for s in sents], "objects", len(objs))

t0 = dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc)
t1 = dt.datetime.fromisoformat(a.end).replace(tzinfo=dt.timezone.utc)
ev = C.find_crossings(sents, objs, O.jd_from_dt(t0), O.jd_from_dt(t1), step_s=30)
day = [e for e in ev if e["desc"] and e["sun_el"] > a.min_sun and e["lit"] and abs(e["v_al"]) >= a.min_v and abs(e["lat"]) <= a.max_lat]
print(f"crossings {len(ev)}, daytime {len(day)}")
day.sort(key=lambda e: abs(e["v_al"]))
SAT = {s.name: s for s in sents}; OBJ = {o.catnr: o for o in objs}
if a.agree:
    import numpy as np
    from s2sat import geometry as G
    OBJ2 = {o.catnr: o for o in O.parse_tle((snap / a.agree).read_text())}
    keep = []
    for e in day:
        o2 = OBJ2.get(e["cat"])
        if o2 is None:
            continue
        g1 = G.apparent_track(SAT[e["sent"]], OBJ[e["cat"]], e["jd"], [0.0])["ground"][0]
        g2 = G.apparent_track(SAT[e["sent"]], o2, e["jd"], [0.0])["ground"][0]
        d = float(np.linalg.norm(g1 - g2))
        if np.isfinite(d) and d <= a.max_disagree:
            e["disagree_km"] = d; keep.append(e)
    print(f"orbit agreement <= {a.max_disagree} km: {len(keep)} of {len(day)}")
    day = keep
Path(a.out).mkdir(parents=True, exist_ok=True)
done = 0
for e in day:
    if done >= a.limit:
        break
    tag = f"{e['obj'].split()[0]}_{O.dt_from_jd(e['jd']):%Y%m%dT%H%M%S}"
    print(f"\n== {tag} {e['sent']} v_al {e['v_al']:.2f} rng {e['rng']:.0f} age {e['age']:.2f} d  dis {e.get('disagree_km', float('nan')):.2f} km  lat {e['lat']:.3f} lon {e['lon']:.3f}")
    try:
        r = P.run_event(SAT[e["sent"]], OBJ[e["cat"]], e, Rm=a.rm, minb=a.minb, max_cloud=a.max_cloud, dM_max=a.dm)
    except Exception as ex:
        print("FAIL", repr(ex)[:300]); continue
    maps = r.pop("_maps", None)
    print("item", r.get("item"), "cloud", r.get("cloud"), "bands", r.get("bands"), "reasons", r.get("reasons")[:5])
    if "score" in r:
        print(f"RESULT {tag}: score {r['score']:.2f}  null max {r['null_max']:.2f} med {r['null_med']:.2f}  "
              f"dM {r['dM']:+.1f} km shift ({r['sx']:.0f}, {r['sy']:.0f})  >4σ {r['hit']}/{r['cnt']}  per-band "
              + " ".join(f"{b}={v:.1f}" for b, v in r["per_band"].items()))
        if maps is not None:
            from s2sat import viz
            viz.plot_event(r, maps, str(Path(a.out) / f"{tag}.png"))
    r.pop("roots", None)
    (Path(a.out) / f"{tag}.json").write_text(json.dumps(dict(event=e, result=r), default=str, indent=1))
    if "score" in r:
        done += 1

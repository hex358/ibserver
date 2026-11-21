"""E2. Статистика слепых зон: для случайных дневных событий с гранулами — число полос с корнем
в зависимости от фазы dM. Вывод: CSV (событие × dM) и сводка."""
import sys, warnings, csv, random, datetime as dt
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); warnings.filterwarnings("ignore")
import numpy as np
from s2sat import orbits as O, crossings as C
from visibility import visibility

S = ROOT / "data/tle_snapshots/20261120T0320"
sents = O.parse_tle((S / "sentinel2.tle").read_text())
objs = [o for o in O.parse_tle((S / "gp_active.tle").read_text()) if o.perigee_km < 1200 and not o.name.startswith("SENTINEL-2")]
random.seed(1); objs = random.sample(objs, 4000)
j0 = O.jd_from_dt(dt.datetime(2026, 11, 19, 6, tzinfo=dt.timezone.utc))
ev = [e for e in C.find_crossings(sents, objs, j0, j0 + 1.0) if e["desc"] and e["sun_el"] > 8 and e["lit"] and abs(e["lat"]) < 70]
random.shuffle(ev)
dms = np.arange(-10, 10.01, 1.0)
SAT = {s.name: s for s in sents}; OBJ = {o.catnr: o for o in objs}
out = open(ROOT / "results/lab/visibility_stats.csv", "w", newline="")
w = csv.writer(out); w.writerow(["obj", "sent", "v_al", "rng", "lat", "lon"] + [f"dm{d:+.0f}" for d in dms])
n_done = 0
for e in ev:
    if n_done >= 60: break
    try:
        v = visibility(SAT[e["sent"]], OBJ[e["cat"]], e, dms)
    except Exception as ex:
        continue
    if v is None: continue
    w.writerow([e["obj"], e["sent"], f"{e['v_al']:.2f}", f"{e['rng']:.0f}", f"{e['lat']:.2f}", f"{e['lon']:.2f}"] + list(v)); out.flush()
    n_done += 1
    print(n_done, e["obj"], f"v_al {e['v_al']:.1f}", "bands@dM:", " ".join(str(x) for x in v), flush=True)
print("DONE")

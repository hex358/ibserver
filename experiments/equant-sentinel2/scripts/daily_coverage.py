"""Ежедневный добор: новые дневные пролёты над сушей для ещё не найденных объектов (THEORY §19).

  python scripts/daily_coverage.py --objs STARLINK-31702,STARLINK-32851,... [--days 2] [--run]

1) по самому свежему снимку орбит ищет пролёты через кадры Sentinel-2 за последние --days суток
   (днём: Солнце > 15°, нисходящий виток, объект освещён, есть гранулы — суша);
2) с --run запускает run_events.py по каждому пролёту с одними правилами для всех:
   окна по источникам орбиты (--source-windows 1.0), облака — фон, 20 нулевых прогонов.
Новый снимок орбит берётся GitHub Action fetch-tle (правка data/tle_request.txt и push).
"""
import argparse, datetime as dt, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import orbits as O, crossings as C, imagery as I

ap = argparse.ArgumentParser()
ap.add_argument("--objs", required=True); ap.add_argument("--days", type=float, default=2.0)
ap.add_argument("--run", action="store_true"); ap.add_argument("--out", default=str(ROOT / "results/events_daily"))
a = ap.parse_args()
want = set(a.objs.split(","))
snaps = sorted(p for p in (ROOT / "data/tle_snapshots").iterdir() if p.is_dir())
last = snaps[-1]
objs = {}
for f in list(last.glob("gp_*.tle")) + list(last.glob("supgp_*.tle")):
    for o in O.parse_tle(f.read_text()):
        if o.name.split()[0] in want and (o.catnr not in objs or o.epoch_jd > objs[o.catnr].epoch_jd):
            objs[o.catnr] = o
sents = [s for s in O.parse_tle((last / "sentinel2.tle").read_text()) if s.name.startswith("SENTINEL-2")]
t1 = dt.datetime.strptime(last.name, "%Y%m%dT%H%M").replace(tzinfo=dt.timezone.utc)
t0 = t1 - dt.timedelta(days=a.days)
ev = C.find_crossings(sents, list(objs.values()), O.jd_from_dt(t0), O.jd_from_dt(t1), step_s=30)
todo = []
for e in sorted(ev, key=lambda e: e["jd"]):
    if not (e["desc"] and e["lit"] and e["sun_el"] > 15):
        continue
    try:
        its = I.items_around(e["lon"], e["lat"], O.dt_from_jd(e["jd"]), e["sent"][-1], d=1.0)
    except Exception:
        its = []
    if not its:
        continue
    todo.append(e)
    print(f"PASS {e['obj']:<16} {O.dt_from_jd(e['jd']):%Y-%m-%dT%H:%M:%S} {e['sent'][-2:]} солнце {e['sun_el']:4.0f}° "
          f"lat {e['lat']:6.1f} lon {e['lon']:7.1f} облака {its[0].properties.get('eo:cloud_cover')}", flush=True)
print(f"новых пролётов над сушей: {len(todo)} (снимок орбит {last.name})", flush=True)
if a.run:
    for e in todo:
        t = O.dt_from_jd(e["jd"])
        cmd = [sys.executable, "-W", "ignore", str(ROOT / "scripts/run_events.py"), "--no-retry", "--min-sun", "15",
               "--snapshot", f"until:{last.name}", "--orbits", "gp_*.tle", "--agree", "supgp_*.tle", "--max-disagree", "100000",
               "--max-age", "5", "--order", "snr", "--obj", e["obj"].split()[0],
               "--start", (t - dt.timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M"), "--end", (t + dt.timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M"),
               "--limit", "3", "--max-cloud", "100", "--no-prefilter", "--source-windows", "1.0", "--nnull", "20",
               "--rc", "3500", "--tol", "30", "--orbit-rule", "fitspan", "--out", a.out]
        env = dict(__import__("os").environ, MALLOC_ARENA_MAX="2", S2_CAPRANK="3", S2_BG="loglocal", S2_SEARCH="bnb", S2_CLOUDMASK="0")
        subprocess.run(cmd, env=env, check=False)

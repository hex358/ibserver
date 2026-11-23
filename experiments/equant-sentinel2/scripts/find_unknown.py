"""Пролёты объектов без публичной 3D-модели (военные/засекреченные: Yaogan, Cosmos, USA, Shijian, TJS, Gaofen…)
через кадры Sentinel-2 днём. Порядок — по RCS (больше → заметнее).

  python scripts/find_unknown.py --start 2026-11-19 --end 2026-11-23
"""
import argparse, csv, datetime as dt, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import orbits as O, crossings as C

ap = argparse.ArgumentParser()
ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
ap.add_argument("--snapshot", default="20261122T2348"); ap.add_argument("--min-sun", type=float, default=25.0); ap.add_argument("--max-age", type=float, default=3.0)
ap.add_argument("--prefix", default="YAOGAN,COSMOS,USA,SHIJIAN,TJS,GAOFEN,JILIN,HUANJING,LKW,OFEQ,KONDOR,BARS,LOTOS,PERSONA,RESURS,KANOPUS,NUSAT,ICEYE,CAPELLA,HEAD,XINGYUN,SJ-")
a = ap.parse_args()
sd = ROOT / "data/tle_snapshots" / a.snapshot
rcs, otype = {}, {}
for row in csv.DictReader((sd / "satcat.csv").read_text().splitlines()):
    try:
        cid = str(int(row["NORAD_CAT_ID"])); otype[cid] = row.get("OBJECT_TYPE", "")
        v = float(row.get("RCS") or "nan")
        if v == v: rcs[cid] = v
    except Exception:
        pass
pref = tuple(p.strip() for p in a.prefix.split(","))
objs = {}
for f in sorted(sd.glob("gp_*.tle")):
    for o in O.parse_tle(f.read_text()):
        if o.name.upper().startswith(pref) and o.perigee_km <= 800 and otype.get(o.catnr, "PAY") in ("PAY", "PAYLOAD"):
            objs[o.catnr] = o
sents = [s for s in O.parse_tle((sd / "sentinel2.tle").read_text()) if s.name.startswith("SENTINEL-2")]
print("объектов", len(objs), flush=True)
t0 = dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc); t1 = dt.datetime.fromisoformat(a.end).replace(tzinfo=dt.timezone.utc)
ev = C.find_crossings(sents, list(objs.values()), O.jd_from_dt(t0), O.jd_from_dt(t1), step_s=30)
day = [e for e in ev if e["desc"] and e["sun_el"] > a.min_sun and e["lit"] and abs(e["age"]) <= a.max_age]
day.sort(key=lambda e: -rcs.get(e["cat"], 0))
print(f"пересечений {len(ev)}, днём {len(day)}")
for e in day[:40]:
    print(f"{e['obj']:<24} {e['cat']:>6} RCS {rcs.get(e['cat'], float('nan')):6.2f} м²  {e['sent']}  {O.dt_from_jd(e['jd']):%Y-%m-%dT%H:%M:%S}  "
          f"солнце {e['sun_el']:4.0f}°  высота {e['alt']:4.0f} км  дальность {e['rng']:4.0f} км  lat {e['lat']:6.1f} lon {e['lon']:7.1f}")
if "--land" in sys.argv or True:
    from s2sat import imagery as I
    land = []
    for e in day:
        try:
            its = I.items_around(e["lon"], e["lat"], O.dt_from_jd(e["jd"]), e["sent"][-1], d=1.0)
        except Exception:
            its = []
        if its:
            land.append(e)
    print(f"над сушей (есть гранулы S2): {len(land)}")
    for e in land:
        print(f"LAND {e['obj']:<24} {e['cat']:>6} RCS {rcs.get(e['cat'], float('nan')):6.2f}  {e['sent']}  {O.dt_from_jd(e['jd']):%Y-%m-%dT%H:%M:%S}  солнце {e['sun_el']:3.0f}°  дальность {e['rng']:4.0f} км  lat {e['lat']:6.1f} lon {e['lon']:7.1f}")
    print("CATS", ",".join(sorted({e["cat"] for e in land})))

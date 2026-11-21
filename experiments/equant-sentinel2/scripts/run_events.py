"""Поиск и проверка пересечений Sentinel-2 × объекты по снимку орбит из data/tle_snapshots.

Пример (SL3 из HANDOFF 5.2):
  python scripts/run_events.py --obj STARLINK-32519 --start 2026-11-19T02:00 --end 2026-11-19T04:00
Без --obj: все дневные события Starlink за окно, медленные первыми (--limit N).
"""
import argparse
import collections
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning, module="skyfield_data")

import numpy as np
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
ap.add_argument("--no-prefilter", action="store_true", help="не фильтровать события по контурам снимков")
ap.add_argument("--resume", action="store_true", help="пропускать события, для которых уже есть JSON в --out")
ap.add_argument("--min-sun", type=float, default=8.0)
ap.add_argument("--min-v", type=float, default=1.5)
ap.add_argument("--method", default="glrt")
ap.add_argument("--dm", default="auto", help="перебор фазы вдоль орбиты, ±км, или auto (по типу объекта и возрасту TLE, THEORY.md §2)")
ap.add_argument("--max-lat", type=float, default=70.0)
ap.add_argument("--agree", default=None, help="второй набор орбит (маски через запятую) для проверки согласия; объекты без него проходят по --max-age")
ap.add_argument("--max-age", type=float, default=1.5, help="макс. возраст TLE, сут")
ap.add_argument("--order", default="snr", choices=["snr", "slow"], help="порядок проверки событий")
ap.add_argument("--max-disagree", type=float, default=2.0, help="км на земле")
ap.add_argument("--out", default=str(ROOT / "results/events"))
a = ap.parse_args()

snaps = sorted(p_ for p_ in (ROOT / "data/tle_snapshots").iterdir() if p_.is_dir())
if a.snapshot == "all":
    snap_list = snaps            # объединить все снимки: для каждого объекта самая свежая эпоха
else:
    snap_list = [(ROOT / "data/tle_snapshots" / a.snapshot) if a.snapshot else snaps[-1]]
snap = snap_list[-1]
print("snapshots", [x.name for x in snap_list])
def _load(patterns, exclude_s2=True):
    best = {}
    for sd in snap_list:
        for pat in patterns.split(","):
            for fpath in sorted(_glob.glob(str(sd / pat.strip()))):
                for o in O.parse_tle(Path(fpath).read_text()):
                    if exclude_s2 and o.name.startswith("SENTINEL-2"):
                        continue
                    if o.catnr not in best or o.epoch_jd > best[o.catnr].epoch_jd:
                        best[o.catnr] = o
    return best

import glob as _glob
sents = [s for s in _load("sentinel2.tle", exclude_s2=False).values() if s.name.startswith(a.sent)]
objs = [o for o in _load(a.orbits).values() if o.perigee_km <= 2000]
if a.obj:
    objs = [o for o in objs if o.name.split()[0] == a.obj or o.catnr == a.obj]
print("sentinels", [s.name for s in sents], "objects", len(objs))

t0 = dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc)
t1 = dt.datetime.fromisoformat(a.end).replace(tzinfo=dt.timezone.utc)
ev = C.find_crossings(sents, objs, O.jd_from_dt(t0), O.jd_from_dt(t1), step_s=30)
day = [e for e in ev if e["desc"] and e["sun_el"] > a.min_sun and e["lit"] and abs(e["v_al"]) >= a.min_v
       and abs(e["lat"]) <= a.max_lat and abs(e["age"]) <= a.max_age]
print(f"crossings {len(ev)}, daytime {len(day)}")
def area_m2(e):
    """Площадь объекта (м²) для прогноза: оценка по имени (Starlink V2 mini ~120, v1.5 ~30, ступень ~36, станции)."""
    n = e["obj"].upper()
    if "ISS" in n: return 2500.0
    if any(k in n for k in ("CSS", "TIANHE", "TIANGONG")): return 600.0
    if "R/B" in n: return 36.0
    if n.startswith("STARLINK"):
        try: return 120.0 if int(n.split("-")[1].split()[0]) >= 30000 else 30.0
        except Exception: return 30.0
    if "DEB" in n: return 0.5
    return 4.0


def size_m(e):
    """Грубая оценка размера по имени (для порядка проверки): станции, ступени, Starlink, прочее."""
    n = e["obj"].upper()
    if any(k in n for k in ("ISS", "CSS", "TIANHE", "TIANGONG")): return 100.0
    if "R/B" in n or "ROCKET" in n: return 12.0
    if n.startswith("STARLINK"):
        try: return 30.0 if int(n.split("-")[1].split()[0]) >= 30000 else 10.0
        except Exception: return 10.0
    if "DEB" in n: return 1.0
    return 5.0

def dm_for(e):
    """Окно по фазе (±км) по данным E1: Starlink манёврирует, прочие объекты известны до ~0.2–1.6 км."""
    if a.dm != "auto":
        return float(a.dm)
    if e.get("supgp"):
        return float(np.clip(3 + 1.5 * e.get("disagree_km", 0.0), 3, 25))
    if e["obj"].upper().startswith("STARLINK"):
        return float(np.clip(10 + 30 * abs(e["age"]), 10, 25))
    return float(np.clip(1.5 + 2 * abs(e["age"]), 1.5, 6))


def retry_dm(e, r):
    """Окно повтора: сильный прогноз (≥ 2× порога) оправдывает широкий поиск (до ±60 км), иначе ×1.6."""
    fc_ = r.get("forecast") or {}
    if fc_.get("T_raw", 0) >= 2 * (r.get("null_max", 99) + 3):
        return 60.0
    return min(1.6 * dm_for(e), 40.0)


def status(r):
    """Диагноз: detected / unobservable / too_faint (прогноз ниже порога) / MISSED (прогноз выше порога — исправимо)."""
    if r.get("early_exit"):
        return "blind_nominal" if (r.get("forecast") or {}).get("u") == {} else "too_faint"
    if (r.get("forecast") or {}).get("u") == {}:
        fc_blind = True
    else:
        fc_blind = False
    if "score" not in r or r["score"] < -1e8:
        return "unobservable"
    thr = r["null_max"] + 3
    if r["score"] >= thr:
        return "detected"
    fc = r.get("forecast") or {}
    tp = fc.get("T_pred", float("nan"))
    if fc_blind:
        return "blind_nominal"   # при номинальной орбите ни одна полоса не видит объект (слепая зона детекторов)
    if np.isfinite(tp) and tp < thr:
        return "too_faint"
    if fc.get("T_low", tp) < thr:
        return "borderline"      # виден только при удачной ориентации (f близко к 1)
    return "MISSED"


def snr_key(e):
    # сигнал ∝ площадь изображения × растяжение push-broom (медленные сильнее)
    stretch = 6.6 / max(abs(e["v_al"]) * e["lam"] / e["rng"] - 0.0, 1.0)
    return -(size_m(e) ** 2) * (e["lam"] / e["rng"]) ** 2 * min(stretch, 10.0)

def footprints(t0, t1):
    """Все снимки коллекции за окно (только контур, время, облачность), с кэшем на диске."""
    from s2sat import imagery as I
    cache = ROOT / "data/cache" / f"footprints_{t0:%Y%m%dT%H%M}_{t1:%Y%m%dT%H%M}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out, t = [], t0
    while t < t1:
        t2 = min(t + dt.timedelta(hours=2), t1)
        s_ = I.client().search(collections=[I.COLL], datetime=f"{t:%Y-%m-%dT%H:%M:%SZ}/{t2:%Y-%m-%dT%H:%M:%SZ}", limit=500,
                               fields={"include": ["id", "geometry", "properties.datetime", "properties.eo:cloud_cover"], "exclude": ["assets", "links"]})
        for it in s_.items_as_dicts():
            out.append(dict(id=it["id"], geom=it["geometry"], t=it["properties"]["datetime"], cloud=it["properties"].get("eo:cloud_cover", 100)))
        t = t2
    cache.write_text(json.dumps(out))
    return out


if not a.no_prefilter and day:
    from shapely.geometry import shape, Point
    from shapely.strtree import STRtree
    fps = footprints(t0 - dt.timedelta(minutes=10), t1 + dt.timedelta(minutes=10))
    geoms = [shape(f["geom"]) for f in fps]; tree = STRtree(geoms)
    tfp = [dt.datetime.fromisoformat(f["t"].replace("Z", "+00:00")) for f in fps]
    keep, why = [], collections.Counter()
    for e in day:
        te = O.dt_from_jd(e["jd"]); pt = Point(e["lon"], e["lat"]); best = None
        for k in tree.query(pt):
            f = fps[k]
            if f["id"][:3] != "S2" + e["sent"][-1] or abs((tfp[k] - te).total_seconds()) > 300 or not geoms[k].contains(pt):
                continue
            best = f if best is None or f["cloud"] < best["cloud"] else best
        if best is None:
            why["no_image"] += 1
        elif best["cloud"] > a.max_cloud:
            why["cloud"] += 1
        else:
            e["prefilter_cloud"] = best["cloud"]; keep.append(e)
    print(f"prefilter: {len(fps)} footprints; kept {len(keep)} of {len(day)} events ({dict(why)})", flush=True)
    day = keep

day.sort(key=(snr_key if a.order == "snr" else (lambda e: abs(e["v_al"]))))
_u, _seen_ev = [], set()
for e in day:   # объекты с одной орбитой (модули МКС и т.п.) дают одно событие
    k = (e["sent"], round(e["jd"] * 86400 / 3))
    if k not in _seen_ev:
        _seen_ev.add(k); _u.append(e)
day = _u
SAT = {s.name: s for s in sents}; OBJ = {o.catnr: o for o in objs}
if a.agree:
    import numpy as np
    from s2sat import geometry as G
    OBJ2 = _load(a.agree)
    keep = []
    for e in day:
        o2 = OBJ2.get(e["cat"])
        if o2 is None:
            e["disagree_km"] = float("nan"); keep.append(e)  # второй орбиты нет: только фильтр возраста
            continue
        g1 = G.apparent_track(SAT[e["sent"]], OBJ[e["cat"]], e["jd"], [0.0])["ground"][0]
        g2 = G.apparent_track(SAT[e["sent"]], o2, e["jd"], [0.0])["ground"][0]
        d = float(np.linalg.norm(g1 - g2))
        if np.isfinite(d) and d <= a.max_disagree:
            e["disagree_km"] = d; e["supgp"] = True; keep.append(e)
    print(f"orbit agreement <= {a.max_disagree} km: {len(keep)} of {len(day)}")
    day = keep
Path(a.out).mkdir(parents=True, exist_ok=True)
done = 0
for e in day:
    if done >= a.limit:
        break
    e["size_m"] = size_m(e)
    _tag = f"{e['obj'].split()[0]}_{O.dt_from_jd(e['jd']):%Y%m%dT%H%M%S}"
    if a.resume and (Path(a.out) / f"{_tag}.json").exists():
        _prev = json.loads((Path(a.out) / f"{_tag}.json").read_text()).get("result", {})
        if "score" in _prev:
            done += 1
        continue
    tag = f"{e['obj'].split()[0]}_{O.dt_from_jd(e['jd']):%Y%m%dT%H%M%S}"
    print(f"\n== {tag} {e['sent']} size~{size_m(e):.0f}m v_al {e['v_al']:.2f} rng {e['rng']:.0f} age {e['age']:.2f} d  dis {e.get('disagree_km', float('nan')):.2f} km  lat {e['lat']:.3f} lon {e['lon']:.3f}")
    try:
        orb = OBJ[e["cat"]]
        if a.agree and e.get("supgp") and e["cat"] in OBJ2:
            orb = OBJ2[e["cat"]]          # SupGP (эфемериды оператора) точнее GP для Starlink
        r = P.run_event(SAT[e["sent"]], orb, e, Rm=a.rm, minb=a.minb, max_cloud=a.max_cloud, dM_max=dm_for(e), method=a.method, area_m2=area_m2(e))
    except Exception as ex:
        print("FAIL", repr(ex)[:300]); continue
    maps = r.pop("_maps", None)
    print("item", r.get("item"), "cloud", r.get("cloud"), "bands", r.get("bands"), "reasons", r.get("reasons")[:5])
    r["status"] = status(r); r["dm_window"] = dm_for(e)
    fc_ = r.get("forecast") or {}
    strong = fc_.get("T_raw", 0) >= 2 * (r.get("null_max", 99) + 3)
    edge = abs(r.get("dM", 0)) > 0.9 * dm_for(e) or max(abs(r.get("sx", 0)), abs(r.get("sy", 0))) >= a.rm - 1
    if r["status"] == "detected" and strong and edge:
        r["status"] = "MISSED"   # «детекция» на краю окна при сильном прогнозе — признак, что объект за окном
    if r["status"] in ("MISSED", "borderline") and fc_.get("T_low", 0) >= (r.get("null_max", 99) + 3) * 0.8:   # прогноз выше порога, но не найден: сразу пробуем шире окно орбиты/сдвига
        try:
            shift_edge = max(abs(r.get("sx", 0)), abs(r.get("sy", 0))) >= a.rm - 1
            r2 = P.run_event(SAT[e["sent"]], orb, e, Rm=(2 * a.rm if shift_edge else a.rm), minb=a.minb, max_cloud=a.max_cloud,
                             dM_max=retry_dm(e, r), method=a.method, area_m2=area_m2(e))
            r2.pop("_maps", None)
            r2["status"] = status(r2)
            print(f"RETRY {tag}: wider window ±{retry_dm(e, r):.0f} km -> {r2['status']} "
                  f"score {r2.get('score', float('nan')):.1f} thr {r2.get('null_max', float('nan')) + 3:.1f}", flush=True)
            if r2["status"] == "detected":
                r = r2; r["dm_window"] = retry_dm(e, r); r["retry"] = True
        except Exception as ex:
            print("RETRY FAIL", repr(ex)[:150])
    fc = r.get("forecast") or {}
    print(f"STATUS {tag}: {r['status']}  score {r.get('score', float('nan')):.1f} thr {r.get('null_max', float('nan')) + 3:.1f}"
          f"  T_pred {fc.get('T_pred', float('nan')):.1f} (gain {fc.get('gain', float('nan')):.2f}, u {fc.get('u', {})})"
          f"  window ±{r['dm_window']:.1f} km  reasons {r.get('reasons', [])[:2]}", flush=True)
    if "score" in r:
        print(f"RESULT {tag}: spectrum {r.get('spectrum','')} f {r.get('f_hat',0):.2f} score {r['score']:.2f}  null max {r['null_max']:.2f} med {r['null_med']:.2f}  "
              f"dM {r['dM']:+.1f} km shift ({r['sx']:.0f}, {r['sy']:.0f})  >4σ {r['hit']}/{r['cnt']}  per-band "
              + " ".join(f"{b}={v:.1f}" for b, v in r["per_band"].items()))
        if maps is not None:
            from s2sat import viz
            viz.plot_event(r, maps, str(Path(a.out) / f"{tag}.png"))
    r.pop("roots", None)
    (Path(a.out) / f"{tag}.json").write_text(json.dumps(dict(event=e, result=r), default=str, indent=1))
    if "score" in r:
        done += 1

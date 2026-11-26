"""Поиск и проверка пересечений Sentinel-2 × объекты по снимку орбит из data/tle_snapshots.

Пример (SL3 из HANDOFF 5.2):
  python scripts/run_events.py --obj STARLINK-32519 --start 2026-11-19T02:00 --end 2026-11-19T04:00
Без --obj: все дневные события Starlink за окно, медленные первыми (--limit N).
"""
import argparse
import math
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
ap.add_argument("--save-maps", default=None, help="каталог для сохранения карт событий (npz) для офлайн-анализа")
ap.add_argument("--sample", default=None, help="выборка событий: top:N,random:M (после префильтра)")
ap.add_argument("--no-retry", action="store_true", help="без автоповтора (режим сбора данных)")
ap.add_argument("--max-image-km", type=float, default=None, help="предел сдвига на снимке (км) для окна по фазе")
ap.add_argument("--shard", default=None, help="k/n: обработать только события с номером ≡ k (mod n)")
ap.add_argument("--rc", type=float, default=None, help="радиус поиска поперёк трассы на снимке, м (поперечная ошибка орбиты × ~2.5)")
ap.add_argument("--tol", type=float, default=30.0, help="допуск положения полосы поперёк трассы, м")
ap.add_argument("--bootstrap-from", default=None, help="каталоги с результатами: для объекта с найденным пролётом окно по фазе вокруг найденной поправки")
ap.add_argument("--boot-drift", type=float, default=5.0, help="допустимый дрейф поправки вдоль орбиты, км/сут")
ap.add_argument("--boot-min", type=float, default=2.0, help="минимальная полуширина окна вокруг найденной поправки, км")
ap.add_argument("--orbit-rule", default="newest", choices=["newest", "fitspan"],
                help="fitspan: SupGP с последней эпохой ДО снимка (прогноз оператора вперёд), иначе GP с первой эпохой ПОСЛЕ (подгонка по наблюдениям назад); THEORY §14")
ap.add_argument("--adaptive-window", action="store_true", help="окно = max(dm-floor, 1.5·расхождение источников орбиты), ≤ 45 км")
ap.add_argument("--dm-floor", type=float, default=8.0)
ap.add_argument("--dual-orbit", action="store_true", help="если не найден, а второй источник орбиты за краем окна — поиск вокруг него")
ap.add_argument("--prior-window", action="store_true", help="окно = ±3σ_вдоль, σ из расхождения источников орбиты (§17)")
ap.add_argument("--reverse", action="store_true", help="обработать список с конца (второй параллельный процесс)")
ap.add_argument("--resume", action="store_true", help="пропускать события, для которых уже есть JSON в --out")
ap.add_argument("--nnull", type=int, default=10, help="число нулевых прогонов (хвост распределения)")
ap.add_argument("--source-windows", type=float, default=0.0,
                help="THEORY §19: σ (км) окон вокруг каждого источника орбиты (GP/SupGP с |эпоха−снимок| ≤ 4 сут); 0 — выкл.")
ap.add_argument("--min-sun", type=float, default=40.0)   # правило съёмки: THEORY §12
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
elif a.snapshot and a.snapshot.startswith("until:"):
    snap_list = [x for x in snaps if x.name <= a.snapshot[6:]]   # воспроизводимый набор: снимки не позже указанного
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
    _want = set(x.strip() for x in a.obj.split(","))
    objs = [o for o in objs if o.name.split()[0] in _want or o.catnr in _want]
print("sentinels", [s.name for s in sents], "objects", len(objs))

t0 = dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc)
t1 = dt.datetime.fromisoformat(a.end).replace(tzinfo=dt.timezone.utc)
ev = C.find_crossings(sents, objs, O.jd_from_dt(t0), O.jd_from_dt(t1), step_s=30)
if a.orbit_rule == "fitspan" and len(snap_list) > 1:
    # пересечения по орбитам каждого снимка каталога: свежая орбита при пересчёте назад уводит объект на десятки км,
    # и пролёт выпадает из снимков; берём пролёт, если объект мог быть в кадре хотя бы по одной орбите
    _want_c = {o.catnr for o in objs}
    _seen = {(x["sent"], x["cat"], round(x["jd"] * 1440)) for x in ev}
    for sd in snap_list[:-1]:
        _objs_s = {}
        for pat in a.orbits.split(","):
            for fpath in sorted(_glob.glob(str(sd / pat.strip()))):
                for o in O.parse_tle(Path(fpath).read_text()):
                    if o.catnr in _want_c and (o.catnr not in _objs_s or o.epoch_jd > _objs_s[o.catnr].epoch_jd):
                        _objs_s[o.catnr] = o
        if not _objs_s:
            continue
        for x in C.find_crossings(sents, list(_objs_s.values()), O.jd_from_dt(t0), O.jd_from_dt(t1), step_s=30):
            k_ = (x["sent"], x["cat"], round(x["jd"] * 1440))
            if not any((k_[0], k_[1], k_[2] + d_) in _seen for d_ in (-1, 0, 1)):
                _seen.add(k_); ev.append(x)
    print(f"crossings incl. older orbits: {len(ev)}", flush=True)
day = [e for e in ev if e["desc"] and e["sun_el"] > a.min_sun and e["lit"] and abs(e["v_al"]) >= a.min_v
       and abs(e["lat"]) <= a.max_lat and abs(e["age"]) <= a.max_age]
print(f"crossings {len(ev)}, daytime {len(day)}")
def _load_rcs():
    """RCS (м²) из последнего satcat.csv в снимках, если есть."""
    import csv
    for sd in reversed(snap_list):
        f = sd / "satcat.csv"
        if f.exists():
            out = {}
            for row in csv.DictReader(f.read_text().splitlines()):
                try:
                    cid = str(int(row["NORAD_CAT_ID"])); OTYPE[cid] = row.get("OBJECT_TYPE", "")
                    v = float(row.get("RCS") or "nan")
                    if v == v and v > 0: out[cid] = v
                except Exception:
                    pass
            print(f"SATCAT: RCS for {len(out)} objects, types for {len(OTYPE)} ({f.parent.name})", flush=True)
            return out
    return {}


OTYPE = {}
RCS = _load_rcs()


def area_m2(e):
    """Площадь объекта (м²) для прогноза: оценка по имени (Starlink V2 mini ~120, v1.5 ~30, ступень ~36, станции)."""
    rcs = RCS.get(str(int(e["cat"]))) if e.get("cat") else None
    if rcs:
        return float(rcs)          # радарное сечение из SATCAT (м²) — оценка площади по данным
    n = e["obj"].upper()
    if n.startswith("ISS (") or e.get("cat") in ("25544",): return 2500.0
    if n.startswith(("CSS (", "TIANHE", "TIANGONG")): return 600.0
    if "OBJECT" in n: return 0.1   # неопознанный объект (часто кубсат)
    t = OTYPE.get(str(int(e["cat"]))) if e.get("cat") else None
    if t == "R/B": return 36.0
    if t == "DEB": return 0.5
    if "R/B" in n: return 36.0
    if n.startswith("STARLINK"):
        if "DTC" in n:
            return 120.0     # Direct-to-Cell: шина V2 mini (docs/3d_models.md); номера 11xxx
        try: return 120.0 if int(n.split("-")[1].split()[0]) >= 30000 else 30.0
        except Exception: return 30.0
    if "DEB" in n: return 0.5
    return 1.0   # аппарат без RCS: размер неизвестен (часто кубсат/микроспутник) — консервативно 1 м²


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
    if e.get("dm_adaptive"):
        return float(e["dm_adaptive"])          # по расхождению источников орбиты (THEORY §14)
    if a.dm != "auto":
        return float(a.dm)
    if e.get("supgp"):
        return 8.0   # по детекциям: реальная ошибка SupGP 1.2–2.9 км (31550 — 7 км) при расхождении GP–SupGP до 32 км
    if e["obj"].upper().startswith("STARLINK"):
        return float(np.clip(10 + 30 * abs(e["age"]), 10, 25))
    return float(np.clip(1.5 + 2 * abs(e["age"]), 1.5, 6))


def retry_dm(e, r):
    """Окно повтора: сильный прогноз (≥ 2× порога) оправдывает широкий поиск (до ±60 км), иначе ×1.6."""
    fc_ = r.get("forecast") or {}
    if fc_.get("T_raw", 0) >= 2 * (r.get("null_max", 99) + 3):
        return 60.0
    return min(max(2.5 * dm_for(e), 20.0), 40.0)   # ошибки SupGP по найденным: 1.2–7 км, у 31272 — 15 км


from s2sat.threshold import threshold as _threshold


def status(r):
    """Диагноз: detected / unobservable / too_faint (прогноз ниже порога) / MISSED (прогноз выше порога — исправимо)."""
    if (r.get("masked_frac") or 0) >= 0.5 and not ("score" in r and "null_max" in r and r["score"] >= _threshold(r)):
        return "cloud_masked"     # точки объекта под маской облаков/теней: не видно, а не пропуск
    if r.get("early_exit"):
        return "blind_nominal" if (r.get("forecast") or {}).get("u") == {} else "too_faint"
    if (r.get("forecast") or {}).get("u") == {}:
        fc_blind = True
    else:
        fc_blind = False
    if "score" not in r or r["score"] < -1e8:
        return "unobservable"
    thr = _threshold(r)
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
if a.reverse:
    day.reverse()
if a.sample:
    import random as _rnd
    spec = dict(x.split(":") for x in a.sample.split(","))
    ntop, nrnd = int(spec.get("top", 0)), int(spec.get("random", 0))
    top = day[:ntop]; rest = day[ntop:]; _rnd.seed(7)
    day = top + _rnd.sample(rest, min(nrnd, len(rest)))
    print(f"sample: top {len(top)} + random {len(day) - len(top)}", flush=True)
_u, _seen_ev = [], set()
for e in day:   # объекты с одной орбитой (модули МКС и т.п.) дают одно событие
    k = (e["sent"], round(e["jd"] * 86400 / 3))
    if k not in _seen_ev:
        _seen_ev.add(k); _u.append(e)
day = _u
if a.shard:
    _k, _n = map(int, a.shard.split("/"))
    day = [e for i, e in enumerate(day) if i % _n == _k]
    print(f"shard {_k}/{_n}: {len(day)} events", flush=True)
SAT = {s.name: s for s in sents}; OBJ = {o.catnr: o for o in objs}


def _load_all(patterns):
    """Все эпохи каждого объекта из всех снимков каталога (для выбора по интервалу подгонки)."""
    allo = {}
    for sd in snap_list:
        for pat in patterns.split(","):
            for fpath in sorted(_glob.glob(str(sd / pat.strip()))):
                for o in O.parse_tle(Path(fpath).read_text()):
                    allo.setdefault(o.catnr, {})[round(o.epoch_jd, 6)] = o
    return {k: list(v.values()) for k, v in allo.items()}


if a.orbit_rule == "fitspan":
    ALL_SG = _load_all(a.agree or "supgp_*.tle"); ALL_GP = _load_all(a.orbits)


def orbit_for(e):
    """Орбита для снимка по интервалу подгонки (по 17 найденным: SupGP до снимка — медиана 1.3 км, макс 6.3;
    GP после — 1.5 / 20.4; SupGP после — 3.0 / 30.3; GP до — 3.2 / 107)."""
    sg = [o for o in ALL_SG.get(e["cat"], []) if o.epoch_jd <= e["jd"] and e["jd"] - o.epoch_jd <= 2.0]
    if sg:
        o = max(sg, key=lambda o: o.epoch_jd); return o, "supgp_before", e["jd"] - o.epoch_jd
    gp = [o for o in ALL_GP.get(e["cat"], []) if o.epoch_jd > e["jd"] and o.epoch_jd - e["jd"] <= 2.0]
    if gp:
        o = min(gp, key=lambda o: o.epoch_jd); return o, "gp_after", e["jd"] - o.epoch_jd
    return None, "none", float("nan")


def orbit_disagreement(e, o, src):
    """|вдоль| между выбранной орбитой и ближайшей по эпохе орбитой другого источника в момент снимка, км.
    По 17 найденным: реальная ошибка ≤ max(8, 1.2·расхождение) во всех случаях (THEORY §14)."""
    alts = ALL_GP.get(e["cat"], []) if src == "supgp_before" else ALL_SG.get(e["cat"], [])
    if not alts:
        return float("nan")
    alt = min(alts, key=lambda x: abs(x.epoch_jd - e["jd"]))
    import numpy as _np
    jd = _np.array([e["jd"]]); r1, v1 = o.ecef(jd); r2, _ = alt.ecef(jd)
    u = v1[0] / _np.linalg.norm(v1[0])
    return abs(float((r2[0] - r1[0]) @ u))
def orbit_alt(e):
    """Другой источник орбиты для снимка: к SupGP-до — GP-после (или ближайшая GP), к GP-после — ближайшая SupGP."""
    if e.get("orbit_src") == "supgp_before":
        gp = [o for o in ALL_GP.get(e["cat"], []) if o.epoch_jd > e["jd"] and o.epoch_jd - e["jd"] <= 2.0]
        if gp:
            return min(gp, key=lambda o: o.epoch_jd)
        alts = ALL_GP.get(e["cat"], [])
    else:
        alts = ALL_SG.get(e["cat"], [])
    return min(alts, key=lambda x: abs(x.epoch_jd - e["jd"])) if alts else None


BOOT = {}   # catnr -> [(jd, dM)] найденных пролётов (по --bootstrap-from)
if a.bootstrap_from:
    from s2sat.threshold import threshold as _thr_b
    for _d in a.bootstrap_from.split(","):
        for _f in Path(_d).glob("*.json"):
            try:
                _x = json.loads(_f.read_text()); _e, _r = _x["event"], _x["result"]
            except Exception:
                continue
            if "score" in _r and "null_max" in _r and _r.get("dM") is not None and _r["score"] >= _thr_b(_r):
                BOOT.setdefault(_e["cat"], []).append((_e["jd"], _r["dM"]))
    print("bootstrap: detected crossings for", len(BOOT), "objects", flush=True)
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
    _n = e["obj"].upper(); _c = str(int(e["cat"])) if e.get("cat") else ""
    e["area_source"] = ("rcs" if _c in RCS else "starlink" if _n.startswith("STARLINK") else
                        "name/type" if (_n.startswith(("ISS (", "CSS (", "TIANHE", "TIANGONG")) or OTYPE.get(_c) in ("R/B", "DEB")) else "unknown")
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
        if a.orbit_rule == "fitspan":
            _o, _src, _age = orbit_for(e)
            if _o is not None:
                orb = _o; e["orbit_src"] = _src; e["orbit_age"] = _age
                _D = orbit_disagreement(e, _o, _src); e["orbit_disagree_km"] = _D
                if a.prior_window and _D == _D:
                    # THEORY §17: σ_вдоль = √(0.85² + (0.78·D)²) км — по 17 эталонным пролётам; окно ±3σ
                    _sa = math.sqrt(0.85 ** 2 + (0.78 * _D) ** 2); e["sigma_along"] = _sa
                    if not e.get("boot"):
                        e["dm_adaptive"] = min(60.0, max(3.0, 3.0 * _sa))
                elif a.adaptive_window and _D == _D:
                    _w = min(45.0, max(a.dm_floor, 1.5 * _D)); _c = (e.get("boot") or {}).get("center", 0.0)
                    if not e.get("boot"):
                        e["dm_adaptive"] = _w
                print(f"  orbit: {_src}, age {_age:+.2f} d, other source differs by {_D:.1f} km", flush=True)
        _c, _w = 0.0, dm_for(e)
        _bl = [(jd, dm) for jd, dm in BOOT.get(e["cat"], []) if abs(jd - e["jd"]) > 1e-4]   # не сам этот пролёт
        if _bl:
            jd_b, dm_b = min(_bl, key=lambda t: abs(t[0] - e["jd"]))
            _c, _w = dm_b, max(a.boot_min, a.boot_drift * abs(e["jd"] - jd_b))
            e["boot"] = dict(center=_c, half=_w, from_jd=jd_b)
            print(f"  bootstrap: window {_c:+.1f} ± {_w:.1f} km (from crossing {e['jd'] - jd_b:+.2f} d)", flush=True)
        _cs = None
        if a.source_windows > 0:
            # смесь по источникам: истина у одного из источников с σ ≈ 0.85 км (все 17 эталонов — ≤ 1.5 км от
            # ближайшего источника при разбросе источников 30–411 км); окна ±3σ вокруг смещения каждого источника
            _rO, _vO = orb.ecef([e["jd"]]); _uv = _vO[0] / np.linalg.norm(_vO[0])
            _offs = []
            for _pat in ("gp_*.tle", "supgp_*.tle"):
                for _sd in snap_list:
                    for _fp in sorted(_glob.glob(str(_sd / _pat))):
                        for _o in O.parse_tle(Path(_fp).read_text()):
                            if _o.catnr == e["cat"] and abs(_o.epoch_jd - e["jd"]) <= 4.0:
                                _r2, _ = _o.ecef([e["jd"]]); _offs.append(float((_r2[0] - _rO[0]) @ _uv))
            _offs = sorted(set(round(v, 1) for v in _offs + [0.0]))
            _cs = []
            for v in _offs:
                if not _cs or v - _cs[-1] > a.source_windows:
                    _cs.append(v)
            # ≥ 3 источников — истина у одного из них с σ ≈ 0.85 км (17/17 эталонов ≤ 1.5 км); 1–2 источника
            # (архивные снимки каталога) — ошибка одиночной GP до 20 км (fit-span: GP после — медиана 1.5, макс 20.4 км;
            # находки по архиву: 31702 +15.5 км, 34093 −11 км) → окно ±20 км
            _w = 3.0 * a.source_windows if len(_cs) >= 3 else 20.0; _c = 0.0
            e["source_centers"] = _cs
            print(f"  source windows: {len(_cs)} × ±{_w:.1f} км вокруг {_cs}", flush=True)
        r = P.run_event(SAT[e["sent"]], orb, e, Rm=a.rm, minb=a.minb, max_cloud=a.max_cloud, dM_max=_w, dM_center=_c, method=a.method, area_m2=area_m2(e), max_image_km=None if _cs else a.max_image_km, Rc=a.rc, tol_m=a.tol, n_null=a.nnull, dM_centers=_cs)
    except Exception as ex:
        print("FAIL", repr(ex)[:300]); continue
    maps = r.pop("_maps", None)
    try:
        from s2sat import imagery as _I
        _I.prune_cache()
    except Exception:
        pass
    if a.save_maps and maps is not None:
        Path(a.save_maps).mkdir(parents=True, exist_ok=True)
        arr = {}
        for (ci, cj), T in maps["tiles"].items():
            k = f"{ci}_{cj}"
            for b, U in T["U"].items():
                arr[f"U|{k}|{b}"] = np.clip(np.nan_to_num(U.astype(np.float32)) * 15, -127, 127).astype(np.int8)
            if T.get("RGB") is not None:
                arr[f"RGB|{k}"] = np.clip(T["RGB"].astype(np.float32) / 40, 0, 255).astype(np.uint8)
        meta = dict(event=e, result={k: v for k, v in r.items() if k not in ("roots",)})
        np.savez_compressed(Path(a.save_maps) / f"{tag}.npz", meta=json.dumps(meta, default=float), **arr)
    print("item", r.get("item"), "cloud", r.get("cloud"), "bands", r.get("bands"), "reasons", r.get("reasons")[:5])
    r["status"] = status(r); r["dm_window"] = max((abs(x - (e.get("boot") or {}).get("center", 0.0)) for x in (r.get("profile_dm") or [])), default=dm_for(e))
    # второй источник орбиты за краем окна (THEORY §16, 3358): отдельный поиск вокруг него
    if (a.orbit_rule == "fitspan" and a.dual_orbit and "score" in r and r["status"] not in ("detected", "detected_edge")
            and (e.get("orbit_disagree_km") or 0) + 8.0 > r["dm_window"]):
        _alt = orbit_alt(e)
        if _alt is not None:
            try:
                _e2 = dict(e); _e2["dm_adaptive"] = 10.0
                r2 = P.run_event(SAT[e["sent"]], _alt, _e2, Rm=a.rm, minb=a.minb, max_cloud=a.max_cloud, dM_max=10.0, dM_center=0.0,
                                 method=a.method, area_m2=area_m2(e), max_image_km=a.max_image_km, Rc=a.rc, tol_m=a.tol)
                maps2 = r2.pop("_maps", None); r2["status"] = status(r2); r2["dm_window"] = 10.0
                print(f"DUAL {tag}: alternative orbit ({'gp' if e.get('orbit_src') == 'supgp_before' else 'supgp'}) -> {r2['status']} "
                      f"score {r2.get('score', float('nan')):.1f}", flush=True)
                if r2["status"] in ("detected", "detected_edge"):
                    r = r2; maps = maps2; r["dual_orbit"] = True; e["orbit_src"] = (e.get("orbit_src") or "") + "+alt"
                else:
                    r["dual_tried"] = dict(score=r2.get("score"), status=r2["status"])
            except Exception as ex:
                print("DUAL FAIL", repr(ex)[:150])
    fc_ = r.get("forecast") or {}
    strong = fc_.get("T_raw", 0) >= 2 * (r.get("null_max", 99) + 3)
    edge = abs(r.get("dM", 0) - (e.get("boot") or {}).get("center", 0.0)) > 0.9 * r["dm_window"] or max(abs(r.get("sx", 0)), abs(r.get("sy", 0))) >= a.rm - 1
    if r["status"] == "detected" and strong and edge:
        r["status"] = "detected_edge"   # найдено на краю окна: при разрешённом повторе — подтвердить шире
    if not a.no_retry and (r["status"] == "detected_edge" or r["status"] in ("MISSED", "borderline") and fc_.get("T_low", 0) >= (r.get("null_max", 99) + 3) * 0.8):   # прогноз выше порога, но не найден: сразу пробуем шире окно орбиты/сдвига
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
    print(f"STATUS {tag}: {r['status']}  score {r.get('score', float('nan')):.1f} thr {(_threshold(r) if 'null_max' in r else float('nan')):.1f}"
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

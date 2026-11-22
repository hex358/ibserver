"""Сводка по обработанным событиям (JSON из run_events.py): статусы по текущим правилам, без пересчёта снимков.

  python scripts/summary.py results/events_v4 [results/v4 ...]
"""
import json, math, sys, collections
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from s2sat.threshold import threshold
from pathlib import Path


def clipT(us, f):
    return math.sqrt(sum(min(abs(f * v), 8.0) ** 2 for v in us.values()))


def classify(r):
    if r.get("early_exit"):
        return "blind_nominal" if (r.get("forecast") or {}).get("u") == {} else "too_faint"
    if "score" not in r or r["score"] < -1e8:
        return "unobservable"
    thr = threshold(r)
    fc = r.get("forecast") or {}
    us = fc.get("u")
    if "T_low" in fc:                     # v6+: сохранены T при f = 0.7 и f = 0.25
        T, Tl = fc.get("T_pred", float("nan")), fc.get("T_low", float("nan"))
    else:
        T, Tl = (clipT(us, 1.0), clipT(us, 0.25)) if us else (float("nan"), float("nan"))
    if r["score"] >= thr:
        return "detected"
    if us == {}:
        return "blind_nominal"
    if us and T < thr:
        return "too_faint"
    if us and Tl < thr:
        return "borderline"
    return "MISSED"


rows = []
for d in sys.argv[1:]:
    for f in sorted(Path(d).glob("*.json")):
        x = json.loads(f.read_text()); e, r = x.get("event", {}), x.get("result", {})
        reason = (r.get("reasons") or [""])[0]
        st = classify(r)
        if st == "unobservable":
            st = "no_image" if "granules" in reason else ("cloud" if "cloud" in reason else "unobservable")
        fc_ = r.get("forecast") or {}; us = fc_.get("u") or {}
        T1 = fc_.get("T_pred") if "T_low" in fc_ else (clipT(us, 1.0) if us else None)
        T025 = fc_.get("T_low") if "T_low" in fc_ else (clipT(us, 0.25) if us else None)
        rows.append(dict(tag=f.stem, obj=e.get("obj"), sent=e.get("sent"), status=st, score=r.get("score"),
                         thr=threshold(r) if "null_max" in r else None,
                         T1=T1, T025=T025,
                         v_al=e.get("v_al"), lat=e.get("lat"), lon=e.get("lon"), dM=r.get("dM"), window=r.get("dm_window")))
seen = {}
for r in rows:   # одно событие могло быть в нескольких каталогах результатов: берём лучший статус
    order = ["detected", "MISSED", "borderline", "too_faint", "blind_nominal", "cloud", "no_image", "unobservable"]
    k = r["tag"]
    if k not in seen:          # первый каталог в аргументах — самый свежий результат
        seen[k] = r
rows = list(seen.values())
c = collections.Counter(r["status"] for r in rows)
print("events:", len(rows), dict(c))
obs = [r for r in rows if r["status"] not in ("no_image", "cloud", "unobservable")]
vis = [r for r in obs if r["status"] not in ("blind_nominal",)]
exp = [r for r in vis if r["status"] in ("detected", "MISSED")]
print(f"observable (image, no cloud): {len(obs)}; not blind: {len(vis)}; "
      f"expected visible even at f=0.25 or detected: {len(exp)}; detected: {c['detected']}")
print(f"{'event':34s} {'status':13s} {'score':>6s} {'thr':>5s} {'T(.7)':>7s} {'T(.25)':>7s} {'v_al':>6s} {'lat':>7s} {'lon':>8s}")
for r in sorted(obs, key=lambda r: ["detected", "MISSED", "borderline", "too_faint", "blind_nominal"].index(r["status"])):
    f = lambda v, w=6, p=1: (f"{v:{w}.{p}f}" if isinstance(v, (int, float)) and v is not None and math.isfinite(v) else " " * (w - 1) + "-")
    print(f"{r['tag'][:34]:34s} {r['status']:13s} {f(r['score'])} {f(r['thr'],5)} {f(r['T1'],7)} {f(r['T025'],7)} {f(r['v_al'],6)} {f(r['lat'],7,2)} {f(r['lon'],8,2)}")

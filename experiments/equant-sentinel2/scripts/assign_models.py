"""Модель и законы ориентации для КАЖДОГО объекта на орбите — один алгоритм для всех (THEORY §24).

  python scripts/assign_models.py [--max-perigee 786]
Для всех объектов CelesTrak SATCAT на орбите вокруг Земли с перигеем ниже Sentinel-2 (786 км) — только они могут
оказаться между Sentinel-2 и Землёй и попасть на снимок: modelcat.resolve (NORAD → шина GCAT → имя →
ступень), класс модели, число крыльев и приводов из манифеста, законы ориентации класса. Пишет
results/catalog/assignments.csv.gz (строка на объект) и results/catalog/coverage_below_s2.json (сводка в процентах).
"""
import argparse, csv, gzip, io, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import modelcat as MC

ap = argparse.ArgumentParser(); ap.add_argument("--max-perigee", type=float, default=786.0); a = ap.parse_args()


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main():
    rows = list(csv.DictReader(gzip.open(ROOT / "data/catalogs/celestrak_satcat.csv.gz", "rt")))
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(["norad", "name", "type", "perigee_km", "apogee_km", "rcs_m2", "gcat_bus", "gcat_shape", "size_m", "model", "model_class",
                "frame", "wings", "two_axis_wings", "laws", "detectable"])
    cnt = Counter(); cnt_t = defaultdict(Counter); det = Counter(); det_t = defaultdict(Counter)
    man = {e["key"]: e for e in MC._manifest()}
    for r in rows:
        if r["DECAY_DATE"] or r["ORBIT_CENTER"] != "EA":
            continue
        pe = num(r["PERIGEE"])
        if pe is None or pe >= a.max_perigee:
            continue
        n = int(r["NORAD_CAT_ID"]); g = MC.gcat(n) or {}
        e = MC.resolve(n, r["OBJECT_NAME"])
        L, D, S = num(g.get("Length")), num(g.get("Diameter")), num(g.get("Span"))
        size = max([v for v in (L, D, S) if v] or [0.0])
        rcs = num(r["RCS"])
        detect = size >= 2.0 or (rcs is not None and rcs >= 1.0)
        typ = (g.get("Type") or r["OBJECT_TYPE"] or "?")[:1]
        if e is None:
            cls = ("параметрическая (размеры GCAT)" if ((L or 0) > 0 or (D or 0) > 0)
                   else "обломок: пластина по ЭПР (SATCAT)" if (rcs or 0) > 0 else "нет данных")
            model, frame, wings, two = "", "", 0, 0
        else:
            m = man[e["key"]]
            frame = e["frame"]; wings = int(m.get("arrays") or 0); two = 0; model = e["key"]
            exact = n in (e["match"].get("norad") or [])
            cls = ("ступень: сетка KeepTrack по форме и размерам" if frame == "rb"
                   else "детальная, оси выверены" + (" (этот аппарат)" if exact else " (класс/шина)") if frame == "satvis"
                   else "детальная, оси не выверены")
        kind = "rocketbody" if typ == "R" else "debris" if typ == "D" else "payload"
        year = int((g.get("LDate") or "2026")[:4]) if (g.get("LDate") or "")[:4].isdigit() else 2026
        laws = MC.laws_for(kind, wings > 0, (e is None) or e["frame"] == "satvis", year)
        w.writerow([n, r["OBJECT_NAME"], r["OBJECT_TYPE"], pe, r["APOGEE"], r["RCS"], g.get("Bus", ""), g.get("Shape", ""), size,
                    model, cls, frame, wings, two, len(laws), int(detect)])
        t = r["OBJECT_TYPE"]
        cnt[cls] += 1; cnt_t[t][cls] += 1
        if detect:
            det[cls] += 1; det_t[t][cls] += 1
    (ROOT / "results/catalog").mkdir(parents=True, exist_ok=True)
    with gzip.open(ROOT / "results/catalog/assignments.csv.gz", "wt") as f:
        f.write(out.getvalue())

    def pct(c):
        tot = sum(c.values()) or 1
        return {k: dict(n=v, pct=round(100 * v / tot, 1)) for k, v in c.most_common()}
    summ = dict(max_perigee_km=a.max_perigee, all=dict(total=sum(cnt.values()), classes=pct(cnt), by_type={k: pct(v) for k, v in cnt_t.items()}),
                detectable=dict(rule="размер GCAT ≥ 2 м или ЭПР ≥ 1 м²", total=sum(det.values()), classes=pct(det),
                                by_type={k: pct(v) for k, v in det_t.items()}))
    (ROOT / "results/catalog/coverage_below_s2.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    for k in ("all", "detectable"):
        print(k, summ[k]["total"])
        for c, v in summ[k]["classes"].items():
            print(f"   {c:48s} {v['n']:6d}  {v['pct']:5.1f}%")


if __name__ == "__main__":
    main()

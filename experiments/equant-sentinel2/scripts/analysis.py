"""Общий разбор набора данных (results/data/*.npz + results/events_ds/*.json) без пересчёта снимков.

  python scripts/analysis.py results/data

Для каждого события:
* статус, прогноз (T при f = 1 и f = 0.25), порог;
* найденная фаза и сдвиг (у детекций) — реальная ошибка орбиты, против расхождения GP–SupGP;
* видимость: число полос с корнем на каждой фазе; в пределах ±PH км от номинала — максимум и доля фаз с ≥ 6 полос;
* маска: доля пустых пикселей карты U в радиусе 150 м от точки прогноза по видимым полосам (облака, тени, края).
Итог: сводная таблица и доли причин.
"""
import json, math, sys, collections
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_spec = importlib.util.spec_from_file_location("study", str(Path(__file__).resolve().parent / "study.py"))
st = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(st)

PH = 7.0   # км: максимальная ошибка SupGP по найденным объектам (0.6–7 км)


def clipT(us, f):
    return math.sqrt(sum(min(abs(f * v), 8.0) ** 2 for v in us.values())) if us else float("nan")


def analyse(p):
    meta, U, RGB = st.load(p)
    e, r = meta["event"], meta["result"]
    g = np.array(r.get("profile_dm") or [0.0]); ra = r.get("roots_all") or []
    nb = np.array([len(x) for x in ra]) if ra else np.array([0])
    near = np.abs(g) <= PH
    fc = r.get("forecast") or {}; us = dict(fc.get("u") or {})
    # пересчёт прогноза под текущие правила площади (прогноз линеен по площади)
    n = e["obj"].upper(); a_old = 2500.0 if "ISS" in n else None
    if a_old and not n.startswith("ISS ("):
        us = {b: v * 0.1 / a_old for b, v in us.items()}   # «ISS OBJECT …» — кубсат, не станция
    thr = r.get("null_max", float("nan")) + 3
    st_ = "detected" if r.get("score", -1e9) >= thr else ("blind" if not us else ("too_faint" if clipT(us, 1) < thr else ("borderline" if clipT(us, .25) < thr else "MISSED")))
    # маска в точках номинала
    k0 = int(np.argmin(np.abs(g))); roots0 = ra[k0] if ra else {}
    o = np.arange(-15, 16) * 10.0; OX, OY = np.meshgrid(o, o); m = np.hypot(OX, OY) <= 150
    masked = []
    for b, lst in roots0.items():
        if b in U:
            v = st.Cells(U[b]).get(lst[0][2] + OX[m], lst[0][3] + OY[m])
            masked.append(float(np.mean(~np.isfinite(v) | (v == 0))))
    return dict(tag=p.stem, obj=e["obj"], v_al=e["v_al"], dis=e.get("disagree_km"), status=st_, score=r.get("score"), thr=thr,
                T1=clipT(us, 1), T025=clipT(us, .25), dM=r.get("dM"), sx=r.get("sx"), sy=r.get("sy"),
                gain=r.get("gain_m_per_km"), vis_nom=int(nb[k0]) if ra else 0,
                vis_max_near=int(nb[near].max()) if ra and near.any() else 0,
                vis6_frac_near=float(np.mean(nb[near] >= 6)) if ra and near.any() else 0.0,
                masked_nom=float(np.mean(masked)) if masked else float("nan"), cloud=r.get("cloud"))


if __name__ == "__main__":
    rows = [analyse(p) for p in sorted(Path(sys.argv[1]).glob("*.npz"))]
    out = Path(sys.argv[1]) / "study"; out.mkdir(exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(rows, default=float))
    f = lambda v, w=6, d=1: (f"{v:{w}.{d}f}" if isinstance(v, (int, float)) and v is not None and math.isfinite(v) else " " * (w - 1) + "-")
    print(f"{'event':32s} {'status':10s} {'score':>6s} {'thr':>5s} {'T1':>5s} {'T.25':>5s} {'v_al':>5s} {'disGP':>6s} {'dM':>6s} "
          f"{'vis0':>4s} {'visMx':>5s} {'vis6%':>5s} {'mask%':>5s} {'cloud':>5s}")
    for x in sorted(rows, key=lambda x: x["status"]):
        print(f"{x['tag'][:32]:32s} {x['status']:10s} {f(x['score'])} {f(x['thr'],5)} {f(x['T1'],5)} {f(x['T025'],5)} {f(x['v_al'],5)} "
              f"{f(x['dis'],6)} {f(x['dM'],6)} {x['vis_nom']:4d} {x['vis_max_near']:5d} {100*x['vis6_frac_near']:5.0f} {f(100*x['masked_nom'],5,0)} {f(x['cloud'],5,0)}")
    c = collections.Counter(x["status"] for x in rows); print("\nstatus:", dict(c))
    det = [x for x in rows if x["status"] == "detected"]
    if det:
        print("detections: |dM| (SupGP error, km):", [round(abs(x["dM"]), 1) for x in det], " GP–SupGP:", [f(x["dis"], 1) for x in det])
    miss = [x for x in rows if x["status"] in ("MISSED", "borderline")]
    if miss:
        print("misses: vis6% near nominal:", [round(100 * x["vis6_frac_near"]) for x in miss],
              " masked% at nominal:", [f(100 * x["masked_nom"], 1, 0) for x in miss])

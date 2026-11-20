"""Видимость объекта детекторами Sentinel-2 как функция фазы вдоль орбиты dM (км).

Для каждого dM считаем, в скольких полосах есть корень «α трека = α детектора».
0 полос — слепая зона шахматного расположения детекторов: объект в этом снимке не виден вообще.
"""
import argparse, sys, warnings, datetime as dt
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); warnings.filterwarnings("ignore")
import numpy as np
from s2sat import orbits as O, crossings as C, imagery as I, geometry as G, s2meta as M

def visibility(sen, ob, e, dms, d=0.8):
    its = I.items_around(e["lon"], e["lat"], O.dt_from_jd(e["jd"]), e["sent"][-1], d=d)
    inside = [i for i in its if i.bbox[0] <= e["lon"] <= i.bbox[2] and i.bbox[1] <= e["lat"] <= i.bbox[3]] or its
    if not inside:
        return None
    main = min(inside, key=lambda i: i.properties.get("eo:cloud_cover", 100)); epsg = I.epsg_of(main)
    rS, _ = sen.ecef([e["jd"]]); psi = G.track_azimuth(sen, e["jd"]); H = float(np.linalg.norm(rS[0])) - G.R_LOC
    DG = {}
    for it in its:
        if I.epsg_of(it) != epsg: continue
        for b, l in M.parse_grids(I.get_text(it.assets["granule_metadata"].href), psi, H, it.id).items():
            DG.setdefault(b, []).extend(l)
    out = []
    for dm in dms:
        tr = G.apparent_track(sen, ob, e["jd"], np.linspace(-2.6, 2.6, 2000), dpos_km=float(dm))
        ok = np.isfinite(tr["ground"][:, 0])
        X, Y = G.ecef_to_utm(tr["ground"][ok], epsg)
        out.append(len(M.band_roots(tr["alpha"][ok], X, Y, DG, list(M.BID))))
    return np.array(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="OBJ@YYYY-mm-ddTHH:MM (начало 30-мин окна)")
    ap.add_argument("--orbits", default="gp_starlink.tle")
    a = ap.parse_args()
    snap = sorted((ROOT / "data/tle_snapshots").iterdir())[-1]
    sents = O.parse_tle((snap / "sentinel2.tle").read_text())
    allo = {o.name.split()[0]: o for o in O.parse_tle((snap / a.orbits).read_text())}
    dms = np.arange(-10, 10.01, 1.0)
    print("dM km:", " ".join(f"{d:+3.0f}" for d in dms))
    for spec in a.specs:
        name, t = spec.split("@"); ob = allo[name]
        t0 = dt.datetime.fromisoformat(t).replace(tzinfo=dt.timezone.utc)
        ev = [x for x in C.find_crossings(sents, [ob], O.jd_from_dt(t0), O.jd_from_dt(t0) + 1 / 48) if x["desc"] and x["sun_el"] > 8]
        if not ev: print(name, "no event"); continue
        e = ev[0]; sen = {s.name: s for s in sents}[e["sent"]]
        v = visibility(sen, ob, e, dms)
        print(f"{name:15s}", " ".join(f"{n:3d}" for n in v), f"  v_al {e['v_al']:.1f}")

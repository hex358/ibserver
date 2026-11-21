"""Совместный поиск одного объекта по нескольким пересечениям (Sentinel-2A/B/C, разные моменты).

Ошибка орбиты вдоль трассы меняется во времени плавно: δ(t) ≈ a + b·(t − t0) (снос до ~20 км/сут для
Starlink, THEORY.md §2). Каждое пересечение даёт профиль «лучший счёт при фазе dM» (и такие же профили
нуль-прогонов). Совместный счёт J(a, b) = Σ_i z_i(a + b·Δt_i) / √n, где z_i — профиль события,
стандартизованный по его нулю. Нуль J считается так же по нуль-профилям. Фазы, несогласованные между
пересечениями, отбрасываются: меньше проверок, сигнал складывается.

  python scripts/joint.py --obj STARLINK-32702 --start 2026-11-19T12:00 --end 2026-11-21T04:00
"""
import argparse, json, math, sys, warnings, datetime as dt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); warnings.filterwarnings("ignore")
import numpy as np
from s2sat import orbits as O, crossings as C

S = ROOT / "data/tle_snapshots"


def load_merged(patterns):
    best = {}
    for sd in sorted(p for p in S.iterdir() if p.is_dir()):
        for pat in patterns.split(","):
            for f in sorted(sd.glob(pat.strip())):
                for o in O.parse_tle(f.read_text()):
                    if o.catnr not in best or o.epoch_jd > best[o.catnr].epoch_jd:
                        best[o.catnr] = o
    return best


def run_one(args):
    sen_lines, ob_lines, e, W, rm, max_cloud, n_null = args
    sys.path.insert(0, str(ROOT))
    from sgp4.api import Satrec
    from s2sat import orbits as O2, pipeline as P
    sen = O2.Orbit(sen_lines[0], Satrec.twoline2rv(sen_lines[1], sen_lines[2]), sen_lines[1], sen_lines[2])
    ob = O2.Orbit(ob_lines[0], Satrec.twoline2rv(ob_lines[1], ob_lines[2]), ob_lines[1], ob_lines[2])
    try:
        r = P.run_event(sen, ob, e, Rm=rm, minb=3, max_cloud=max_cloud, dM_max=W, n_null=n_null)
    except Exception as ex:
        return dict(error=repr(ex)[:200], event=e)
    r.pop("_maps", None); r.pop("roots", None); r.pop("roots0", None)
    return dict(event=e, result=r)


def standardize(prof, nulls):
    v = np.array([x for row in nulls for x in row if x is not None and np.isfinite(x) and x > -1e8])
    if v.size < 10:
        return None, None
    mu, sd = np.median(v), 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-6
    z = (np.array(prof, float) - mu) / sd
    zn = (np.array(nulls, float) - mu) / sd
    z[~np.isfinite(z) | (np.array(prof, float) < -1e8)] = np.nan
    zn[~np.isfinite(zn) | (np.array(nulls, float) < -1e8)] = np.nan
    return z, zn


def joint_max(curves, times, a_grid, b_grid):
    """curves: [(dm_grid, z)], times: Δt (сут). Возвращает (J_max, a, b, n_used)."""
    best = (-1e9, 0, 0, 0)
    for b in b_grid:
        acc = np.zeros_like(a_grid); cnt = np.zeros_like(a_grid)
        for (g, z), t in zip(curves, times):
            dm = a_grid + b * t
            ok = (dm >= g[0]) & (dm <= g[-1])
            zi = np.interp(dm, g, np.nan_to_num(z, nan=-3.0))
            acc += np.where(ok, zi, 0); cnt += ok
        J = np.where(cnt >= 2, acc / np.sqrt(np.maximum(cnt, 1)), -1e9)
        k = int(np.argmax(J))
        if J[k] > best[0]:
            best = (float(J[k]), float(a_grid[k]), float(b), int(cnt[k]))
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--orbits", default="gp_*.tle,debris_*.tle")
    ap.add_argument("--supgp", default="supgp_*.tle")
    ap.add_argument("--window", type=float, default=None, help="±км фазы (по умолчанию: Starlink 25, прочие 6)")
    ap.add_argument("--drift", type=float, default=None, help="±км/сут сноса (по умолчанию: Starlink 40, прочие 5)")
    ap.add_argument("--rm", type=float, default=1500.0)
    ap.add_argument("--max-cloud", type=float, default=40.0)
    ap.add_argument("--n-null", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    objs = load_merged(a.orbits); sup = load_merged(a.supgp)
    sents = [s for s in load_merged("sentinel2.tle").values() if s.name.startswith("SENTINEL-2")]
    ob = [o for o in objs.values() if o.name.split()[0] == a.obj]
    if not ob:
        sys.exit(f"{a.obj} not found")
    ob = sup.get(ob[0].catnr, ob[0])
    star = a.obj.upper().startswith("STARLINK")
    W = a.window or (25.0 if star else 6.0); B = a.drift or (40.0 if star else 5.0)
    t0 = dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc); t1 = dt.datetime.fromisoformat(a.end).replace(tzinfo=dt.timezone.utc)
    ev = [e for e in C.find_crossings(sents, [ob], O.jd_from_dt(t0), O.jd_from_dt(t1))
          if e["desc"] and e["sun_el"] > 8 and e["lit"] and abs(e["lat"]) < 70]
    print(f"{a.obj} ({ob.name}, epoch {O.dt_from_jd(ob.epoch_jd):%m-%d %H:%M}): {len(ev)} daytime crossings, window ±{W} km, drift ±{B} km/d", flush=True)
    SL = {s.name: s for s in sents}
    jobs = [((SL[e["sent"]].name, SL[e["sent"]].line1, SL[e["sent"]].line2), (ob.name, ob.line1, ob.line2), e, W, a.rm, a.max_cloud, a.n_null) for e in ev]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_one, jobs))
    out_dir = ROOT / "results/joint"; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{a.obj}.json").write_text(json.dumps(res, default=float))
    curves, times, nulls = [], [], []
    jref = None
    for x in res:
        e = x["event"]; r = x.get("result", {})
        tag = f"{e['sent'][-2:]} {O.dt_from_jd(e['jd']):%m-%d %H:%M}"
        if "profile" not in r:
            print(f"  {tag}: unobservable {x.get('error') or r.get('reasons', [])[:2]}", flush=True); continue
        z, zn = standardize(r["profile"], r["null_profiles"])
        if z is None:
            print(f"  {tag}: no null"); continue
        g = np.array(r["profile_dm"])
        jref = e["jd"] if jref is None else jref
        curves.append((g, z)); times.append(e["jd"] - jref); nulls.append(zn)
        print(f"  {tag}: score {r['score']:.1f} null max {r['null_max']:.1f} best dM {r['dM']:+.1f}  z_max {np.nanmax(z):.1f}  cloud {r.get('cloud', 0):.0f}%", flush=True)
    if len(curves) < 2:
        print("JOINT: fewer than 2 observable crossings"); sys.exit(0)
    a_grid = np.arange(-W, W + 1e-9, 0.25); b_grid = np.arange(-B, B + 1e-9, max(B / 40, 0.25))
    J, aa, bb, n = joint_max(curves, times, a_grid, b_grid)
    nn = min(len(x) for x in nulls); Jn = []
    for k in range(nn):
        Jn.append(joint_max([(g, zn[k]) for (g, _), zn in zip(curves, nulls)], times, a_grid, b_grid)[0])
    print(f"JOINT {a.obj}: J {J:.2f} (a {aa:+.2f} km, b {bb:+.1f} km/d, {n} crossings)  null max {max(Jn):.2f} med {np.median(Jn):.2f}"
          f"  -> {'DETECTED' if J >= max(Jn) + 1.5 else 'not detected'}", flush=True)

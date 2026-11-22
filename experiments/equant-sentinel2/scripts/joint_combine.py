"""Совместная оценка по нескольким пересечениям одного объекта (результаты run_events.py).

  python scripts/joint_combine.py results/events_joint

Каждое пересечение даёт профиль «лучший счёт при фазе dM» (`profile`, `profile_dm`) и такие же профили
нуль-прогонов (`null_profiles`). Профиль стандартизуется по своему нулю: z(dM) = (score − медиана нуля)/MAD нуля.
Ошибка орбиты вдоль трассы меняется во времени плавно: dM_i = a + b·(t_i − t0). Совместный счёт
J(a, b) = Σ_i z_i(a + b·Δt_i) / √n (n — пересечения, где фаза попала в профиль). Нуль J — тем же способом по
нуль-профилям (k-й нуль каждого пересечения). Детекция: J ≥ max(нуль J) + запас.
"""
import json, math, sys, collections
from pathlib import Path
import numpy as np


def standardize(prof, nulls):
    v = np.array([x for row in nulls for x in row if x is not None and np.isfinite(x) and x > -1e8], float)
    if v.size < 20:
        return None, None
    mu = np.median(v); sd = 1.4826 * np.median(np.abs(v - mu)) + 1e-6
    P = np.array(prof, float); N = np.array(nulls, float)
    z = (P - mu) / sd; zn = (N - mu) / sd
    z[~np.isfinite(P) | (P < -1e8)] = np.nan; zn[~np.isfinite(N) | (N < -1e8)] = np.nan
    return z, zn


def joint_max(curves, times, a_grid, b_grid, min_n=2):
    best = (-1e9, 0.0, 0.0, 0)
    for b in b_grid:
        acc = np.zeros_like(a_grid); cnt = np.zeros_like(a_grid)
        for (g, z), t in zip(curves, times):
            dm = a_grid + b * t
            ok = (dm >= g[0]) & (dm <= g[-1])
            zi = np.interp(dm, g, np.nan_to_num(z, nan=-2.0))
            acc += np.where(ok, zi, 0.0); cnt += ok
        J = np.where(cnt >= min_n, acc / np.sqrt(np.maximum(cnt, 1)), -1e9)
        k = int(np.argmax(J))
        if J[k] > best[0]:
            best = (float(J[k]), float(a_grid[k]), float(b), int(cnt[k]))
    return best


if __name__ == "__main__":
    groups = collections.defaultdict(list)
    for d in sys.argv[1:]:
        for f in sorted(Path(d).glob("*.json")):
            x = json.loads(f.read_text()); e, r = x["event"], x["result"]
            if "profile" not in r or "null_profiles" not in r:
                continue
            groups[(e["cat"], e["obj"].split()[0])].append((e, r))
    rows = []
    for (cat, name), lst in sorted(groups.items(), key=lambda kv: kv[0][1]):
        lst.sort(key=lambda er: er[0]["jd"])
        t0 = lst[0][0]["jd"]
        curves, times, nulls, ind = [], [], [], []
        for e, r in lst:
            z, zn = standardize(r["profile"], r["null_profiles"])
            if z is None:
                continue
            curves.append((np.array(r["profile_dm"], float), z)); times.append(e["jd"] - t0); nulls.append(zn)
            ind.append((e["sent"][-2:], f"{e['jd'] - t0:+.2f}d", round(float(np.nanmax(z)), 1), round(r["score"] - (r["null_max"] + 3), 1)))
        if len(curves) < 2:
            rows.append(dict(obj=name, n=len(curves), ind=ind, J=None)); continue
        star = name.upper().startswith("STARLINK")
        W = max(float(c[0][-1]) for c in curves)
        a_grid = np.arange(-W, W + 1e-9, 0.25)
        B = 40.0 if star else 5.0
        b_grid = np.arange(-B, B + 1e-9, B / 40)
        J, aa, bb, n = joint_max(curves, times, a_grid, b_grid)
        nn = min(len(x) for x in nulls)
        Jn = [joint_max([(g, zn[k]) for (g, _), zn in zip(curves, nulls)], times, a_grid, b_grid)[0] for k in range(nn)]
        det = J >= max(Jn) + 1.0
        rows.append(dict(obj=name, n=len(curves), ind=ind, J=J, a=aa, b=bb, n_used=n, null_max=max(Jn), null_med=float(np.median(Jn)), det=det))
        print(f"{name:16s} crossings {len(curves)}  J {J:6.2f}  null max {max(Jn):5.2f} med {np.median(Jn):5.2f}  a {aa:+6.2f} km  b {bb:+5.1f} km/d  "
              f"used {n}  -> {'DETECTED' if det else 'no'}   per-crossing (sat, Δt, z_max, margin): {ind}", flush=True)
    out = Path(sys.argv[1]) / "joint_summary.json"
    out.write_text(json.dumps(rows, default=float))
    dets = [r for r in rows if r.get("det")]
    print(f"\nobjects {len(rows)}, with ≥2 crossings {sum(1 for r in rows if r['J'] is not None)}, joint detections {len(dets)}")

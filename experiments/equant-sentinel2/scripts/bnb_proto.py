"""Прототип перебора гипотез по схеме Equant (ветви и границы с закрытой оценкой), CPU/numpy.

  python scripts/bnb_proto.py results/events_v9/STARLINK-30245_20261120T110322.json [--null k]

Гипотеза h = (n — индекс фазы dM, e — растяжение, a — сдвиг вдоль видимой трассы, c — поперёк), та же сетка, что у
полного перебора в конвейере. Блок гипотез → для каждой полосы прямоугольник, куда попадают её точки → максимум |U|
по пирамиде максимумов M_b → верхняя граница счёта UB = √Σ min(M_b, M_(3))² (GLRT — проекция, T ≤ ‖u‖; отсечка по
3-й полосе). Блоки с UB ≤ лучшего найденного отбрасываются (как K1/K2 в Equant), выжившие делятся, точный счёт —
в центрах блоков и в листьях (как K3). Сравнение: максимум против полного перебора и число вычисленных гипотез.
"""
import argparse, glob, json, math, pickle, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2sat import imagery as I, detect, orbits as O, geometry as G
from s2sat.pipeline import CellRaster, CELL

ap = argparse.ArgumentParser()
ap.add_argument("json")
ap.add_argument("--null", type=int, default=-1, help="номер нуль-прохода (смещения полос из JSON), -1 — настоящий")
ap.add_argument("--chunk", type=int, default=8)
ap.add_argument("--until", default="20261121T1217")
a = ap.parse_args()
x = json.loads(Path(a.json).read_text()); e, r = x["event"], x["result"]
RA = r["roots_all"]; N = len(RA)
smax = 0.05
STRETCH = np.round(np.arange(-smax, smax + 1e-5, 0.004), 4)
R_al = r.get("search_along_m") or 1500.0; Rc = 3500.0; step = 30.0
Aa = np.arange(-R_al, R_al + 1e-6, step); Cc = np.arange(-Rc, Rc + 1e-6, step)
NE, NA, NC = len(STRETCH), len(Aa), len(Cc)
kmdeg = r["kmdeg"]; minb = 3
disp = {}
if a.null >= 0:
    disp = {b: tuple(v) for b, v in r["null_disp"][a.null].items()}

# направление видимой трассы — как в конвейере (по номинальной орбите)
snaps = sorted(p for p in (ROOT / "data/tle_snapshots").iterdir() if p.is_dir() and p.name <= a.until)
def load(pat):
    best = {}
    for sd in snaps:
        for f in sd.glob(pat):
            for o in O.parse_tle(f.read_text()):
                if o.catnr not in best or o.epoch_jd > best[o.catnr].epoch_jd:
                    best[o.catnr] = o
    return best
S2 = {o.name: o for o in load("sentinel2.tle").values()}
orb = (load("supgp_*.tle").get(e["cat"]) if e.get("supgp") else None) or load("gp_*.tle")[e["cat"]]
tr = G.apparent_track(S2[e["sent"]], orb, e["jd"], np.linspace(-2.6, 2.6, 2000), dpos_km=float((e.get("boot") or {}).get("center", 0.0)) or None)
ok = np.isfinite(tr["ground"][:, 0]); X_, Y_ = G.ecef_to_utm(tr["ground"][ok], r["epsg"]); k = len(X_) // 2
ang = math.atan2(Y_[min(k + 5, len(Y_) - 1)] - Y_[k], X_[min(k + 5, len(X_) - 1)] - X_[k])
t = np.array([math.cos(ang), math.sin(ang)]); nrm = np.array([-t[1], t[0]])

# ячейки из кэша
key = f"{e['obj'].split()[0]}_{e['sent'][-2:]}_{round(e['jd'] * 86400)}_glrt_z*loglocal*"
ds = sorted(glob.glob(str(I.CACHE / "tiles" / key)))
tiles = {}
for p in Path(ds[-1]).glob("*.pkl"):
    ci, cj = map(int, p.stem.split("_")); T = pickle.loads(p.read_bytes())
    tiles[(ci, cj)] = {"U": T["U"], "S": T["S"], "G": T["G"]}
bands = sorted({b for R in RA for b in R} & set(next(iter(tiles.values()))["U"]))
mosU, mosS, mosG = CellRaster(tiles, bands, "U"), CellRaster(tiles, bands, "S"), CellRaster(tiles, bands, "G")

# корни: для каждой фазы n и полосы — список (α, x, y) по детекторам
roots = [{b: [(q[1], q[2], q[3]) for q in R[b]] for b in R if b in bands} for R in RA]
aref = np.array([min([v[0][0] for v in R.values()]) if R else 0.0 for R in roots])

# плотная мозаика |U| (uint8, ×10) и пирамида максимумов для оценок
ci = [c[0] for c in tiles]; cj = [c[1] for c in tiles]; i0, i1, j0, j1 = min(ci), max(ci), min(cj), max(cj)
n_px = next(iter(tiles.values()))["U"][bands[0]].shape[0]; res = CELL / n_px
H, W = (j1 - j0 + 1) * n_px, (i1 - i0 + 1) * n_px
X0, Y1 = i0 * CELL, (j1 + 1) * CELL
PYR = {}
for b in bands:
    D = np.zeros((H, W), np.uint8)
    for (ii, jj), T in tiles.items():
        A = np.nan_to_num(np.abs(T["U"][b].astype(np.float32)))
        D[(j1 - jj) * n_px:(j1 - jj + 1) * n_px, (ii - i0) * n_px:(ii - i0 + 1) * n_px] = np.clip(np.ceil(A * 10), 0, 255).astype(np.uint8)
    lv = [D]
    while max(lv[-1].shape) > 1:
        P = lv[-1]; h2, w2 = (P.shape[0] + 1) // 2, (P.shape[1] + 1) // 2
        Q = np.zeros((h2 * 2, w2 * 2), np.uint8); Q[:P.shape[0], :P.shape[1]] = P
        lv.append(Q.reshape(h2, 2, w2, 2).max(axis=(1, 3)))
    PYR[b] = lv


def rect_max(b, x0, x1, y0, y1):
    """Верхняя граница max|U| полосы b в прямоугольниках [x0,x1]×[y0,y1] (векторно)."""
    c0 = np.floor((x0 - X0) / res).astype(int); c1 = np.floor((x1 - X0) / res).astype(int)
    r0 = np.floor((Y1 - y1) / res).astype(int); r1 = np.floor((Y1 - y0) / res).astype(int)
    c0 = np.clip(c0, 0, W - 1); c1 = np.clip(c1, 0, W - 1); r0 = np.clip(r0, 0, H - 1); r1 = np.clip(r1, 0, H - 1)
    span = np.maximum(c1 - c0, r1 - r0) + 1
    L = np.ceil(np.log2(np.maximum(span, 1))).astype(int)
    out = np.zeros(len(x0))
    lv = PYR[b]
    for l in np.unique(L):
        m = L == l; l_ = min(l, len(lv) - 1); P = lv[l_]
        rr0, rr1, cc0, cc1 = r0[m] >> l_, r1[m] >> l_, c0[m] >> l_, c1[m] >> l_
        v = np.maximum(np.maximum(P[rr0, cc0], P[rr0, np.minimum(cc1, P.shape[1] - 1)]),
                       np.maximum(P[np.minimum(rr1, P.shape[0] - 1), cc0], P[np.minimum(rr1, P.shape[0] - 1), np.minimum(cc1, P.shape[1] - 1)]))
        out[m] = v / 10.0
    return out


def positions(n, e_idx, b, root):
    """Смещение растяжения для полосы b при фазе n (угол первого детектора полосы — как в конвейере)."""
    al = roots[n][b][0][0]
    sig = STRETCH[e_idx] * kmdeg * 1000.0
    return sig * (al - aref[n])


def score_points(n, e_idx, AA, CC):
    """Точный счёт для гипотез (n, e) × набор (a, c) — повтор detect.occlusion_search."""
    R = roots[n]
    if sum(1 for b in R) < minb:
        return np.full(len(AA), -1e9)
    SX = AA * t[0] + CC * nrm[0]; SY = AA * t[1] + CC * nrm[1]
    per = []; cnt = np.zeros(len(AA))
    for b, lst in R.items():
        dx, dy = disp.get(b, (0.0, 0.0)); bu = bs = bg = None
        for root in lst:
            off = positions(n, e_idx, b, root)
            X = root[1] + dx + off * t[0] + SX; Y = root[2] + dy + off * t[1] + SY
            u = mosU.get(b, X, Y).astype(np.float64); sv = mosS.get(b, X, Y).astype(np.float64); gv = mosG.get(b, X, Y).astype(np.float64)
            if bu is None:
                bu, bs, bg = u, sv, gv
            else:
                tk = np.abs(np.nan_to_num(u)) > np.abs(np.nan_to_num(bu)); bu, bs, bg = np.where(tk, u, bu), np.where(tk, sv, bs), np.where(tk, gv, bg)
        okb = np.isfinite(bu) & np.isfinite(bs) & (bs > 0) & np.isfinite(bg)
        inv = np.where(okb, 1.0 / np.where(okb, bs, 1), 0); uu = np.where(okb, bu, 0); x2 = -inv * (np.where(okb, bg, 1000.0) - 1000.0)
        per.append((b, inv, x2, uu)); cnt += okb
    if len(per) >= 3:
        Au = np.sort(np.stack([np.abs(p[3]) for p in per]), axis=0)[-3]
        per = [(b, inv, x2, np.clip(uu, -Au, Au)) for b, inv, x2, uu in per]
    best = np.full(len(AA), -1e9)
    for sp in detect.SPECTRA.values():
        A11 = A12 = A22 = B1 = B2 = 0.0
        for b, inv, x2, uu in per:
            x1 = sp.get(b, 1.0) * 1e3 * inv
            A11 = A11 + x1 * x1; A12 = A12 + x1 * x2; A22 = A22 + x2 * x2; B1 = B1 + x1 * uu; B2 = B2 + x2 * uu
        det = A11 * A22 - A12 * A12
        with np.errstate(invalid="ignore", divide="ignore"):
            ah = (A22 * B1 - A12 * B2) / (det + 1e-12 * (A11 * A22 + 1e-30)); fh = (A11 * B2 - A12 * B1) / (det + 1e-12 * (A11 * A22 + 1e-30))
            T2 = np.where(fh >= 0, ah * B1 + fh * B2, B1 * B1 / A11)
            Tm = np.where(cnt >= minb, np.sqrt(np.maximum(np.nan_to_num(T2, nan=0.0), 0)), -1e9)
        best = np.maximum(best, Tm)
    return best


def bound(boxes):
    """UB для блоков [n0,n1)×[e0,e1)×[a0,a1)×[c0,c1)."""
    nb = len(boxes); M = {}
    navail = np.zeros(nb)
    for b in bands:
        mx = np.zeros(nb); have = np.zeros(nb, bool)
        dx, dy = disp.get(b, (0.0, 0.0))
        for i, (n0, n1, e0, e1, a0, a1, c0, c1) in enumerate(boxes):
            xs = []; ys = []
            for n in range(n0, n1):
                for root in roots[n].get(b, []):
                    for ee in (e0, e1 - 1):
                        off = positions(n, ee, b, root)
                        for aa in (Aa[a0], Aa[a1 - 1]):
                            for cc in (Cc[c0], Cc[c1 - 1]):
                                xs.append(root[1] + dx + (off + aa) * t[0] + cc * nrm[0]); ys.append(root[2] + dy + (off + aa) * t[1] + cc * nrm[1])
            if xs:
                have[i] = True; mx[i] = rect_max(b, np.array([min(xs)]), np.array([max(xs)]), np.array([min(ys)]), np.array([max(ys)]))[0]
        M[b] = np.where(have, mx, 0.0); navail += have
    Ms = np.stack([M[b] for b in bands])                 # (bands, boxes)
    third = np.sort(Ms, axis=0)[-3] if len(bands) >= 3 else np.zeros(nb)
    ub = np.sqrt((np.minimum(Ms, third) ** 2).sum(0))
    return np.where(navail >= minb, ub, 0.0)


t0 = time.time()
boxes = [(n, min(n + a.chunk, N), 0, NE, 0, NA, 0, NC) for n in range(0, N, a.chunk)]
best = -1e9; best_h = None; n_eval = 0; n_bound = 0; it = 0
while boxes:
    it += 1
    ub = bound(boxes); n_bound += len(boxes)
    boxes = [bx for bx, u in zip(boxes, ub) if u > best]
    nxt = []
    for bx in boxes:
        n0, n1, e0, e1, a0, a1, c0, c1 = bx
        if (n1 - n0) * (e1 - e0) * (a1 - a0) * (c1 - c0) <= 64:      # лист: точный счёт всех гипотез блока
            for n in range(n0, n1):
                for ee in range(e0, e1):
                    AA, CC = np.meshgrid(Aa[a0:a1], Cc[c0:c1]); s = score_points(n, ee, AA.ravel(), CC.ravel()); n_eval += s.size
                    j = int(np.argmax(s))
                    if s[j] > best:
                        best = float(s[j]); best_h = (n, ee, float(AA.ravel()[j]), float(CC.ravel()[j]))
            continue
        nm, em, am, cm = (n0 + n1) // 2, (e0 + e1) // 2, (a0 + a1) // 2, (c0 + c1) // 2
        s = score_points(nm, em, np.array([Aa[am]]), np.array([Cc[cm]])); n_eval += 1
        if s[0] > best:
            best = float(s[0]); best_h = (nm, em, float(Aa[am]), float(Cc[cm]))
        # делим по измерению с наибольшим размахом на снимке
        ext = {"n": (n1 - n0) * (r.get("gain_m_per_km") or 1000) * (float(np.median(np.diff(r["profile_dm"]))) if N > 1 else 0),
               "e": (e1 - e0) * 0.004 * kmdeg * 1000 * 2.5, "a": (a1 - a0) * step, "c": (c1 - c0) * step}
        d = max(ext, key=ext.get)
        if d == "n" and n1 - n0 > 1:
            nxt += [(n0, nm, e0, e1, a0, a1, c0, c1), (nm, n1, e0, e1, a0, a1, c0, c1)]
        elif d == "e" and e1 - e0 > 1:
            nxt += [(n0, n1, e0, em, a0, a1, c0, c1), (n0, n1, em, e1, a0, a1, c0, c1)]
        elif d == "a" and a1 - a0 > 1:
            nxt += [(n0, n1, e0, e1, a0, am, c0, c1), (n0, n1, e0, e1, am, a1, c0, c1)]
        elif c1 - c0 > 1:
            nxt += [(n0, n1, e0, e1, a0, a1, c0, cm), (n0, n1, e0, e1, a0, a1, cm, c1)]
        else:
            for dd in ("n", "e", "a"):
                pass
            nxt += [(n0, nm, e0, e1, a0, a1, c0, c1), (nm, n1, e0, e1, a0, a1, c0, c1)] if n1 - n0 > 1 else \
                   [(n0, n1, e0, em, a0, a1, c0, c1), (n0, n1, em, e1, a0, a1, c0, c1)] if e1 - e0 > 1 else \
                   [(n0, n1, e0, e1, a0, am, c0, c1), (n0, n1, e0, e1, am, a1, c0, c1)]
    boxes = nxt
n_brute = sum(1 for n in range(N) if len(roots[n]) >= minb) * NE * NA * NC
ref = r["score"] if a.null < 0 else r["nulls"][a.null]
print(f"{Path(a.json).stem} {'real' if a.null < 0 else 'null ' + str(a.null)}: B&B max {best:.3f} at n={best_h[0]} e={STRETCH[best_h[1]]:+.3f} a={best_h[2]:.0f} c={best_h[3]:.0f}"
      f" | brute-force max {ref:.3f} | hypotheses: brute {n_brute:.2e}, B&B exact {n_eval:.2e}, bounds {n_bound:.2e}"
      f" -> pruning x{n_brute / max(n_eval, 1):.0f} | {time.time() - t0:.0f} s, {it} levels", flush=True)

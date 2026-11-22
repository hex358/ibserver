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
ap.add_argument("--floor", type=float, default=None, help="уровень отсечения: блоки с UB ≤ floor отбрасываются (решение «найден» не меняют)")
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


# массивы корней: RX/RY[n, b, k], VAL[n, b, k], DA[n, b] = α_первого − α_ref (для растяжения)
KMAX = max(len(v) for R in roots for v in R.values())
NB = len(bands); BI = {b: i for i, b in enumerate(bands)}
RX = np.zeros((N, NB, KMAX)); RY = np.zeros((N, NB, KMAX)); VAL = np.zeros((N, NB, KMAX), bool); DA = np.zeros((N, NB))
DXY = np.array([disp.get(b, (0.0, 0.0)) for b in bands])
for n, R in enumerate(roots):
    for b, lst in R.items():
        i = BI[b]; DA[n, i] = lst[0][0] - aref[n]
        for k_, q in enumerate(lst):
            RX[n, i, k_] = q[1] + DXY[i, 0]; RY[n, i, k_] = q[2] + DXY[i, 1]; VAL[n, i, k_] = True
KST = kmdeg * 1000.0


def bound(boxes):
    """UB для блоков [n0,n1)×[e0,e1)×[a0,a1)×[c0,c1), векторно по блокам × полосам × детекторам × фазам."""
    B = np.array(boxes); nb = len(B)
    n0, n1, e0, e1, a0, a1, c0, c1 = B.T
    alo, ahi = Aa[a0], Aa[a1 - 1]; clo, chi = Cc[c0], Cc[c1 - 1]
    slo, shi = STRETCH[e0], STRETCH[e1 - 1]
    span = int((n1 - n0).max())
    xmin = np.full((nb, NB, KMAX), np.inf); xmax = np.full((nb, NB, KMAX), -np.inf)
    ymin = np.full((nb, NB, KMAX), np.inf); ymax = np.full((nb, NB, KMAX), -np.inf)
    anyv = np.zeros((nb, NB, KMAX), bool); nav = np.zeros((nb, span), int)
    for j in range(span):
        n = n0 + j; okn = n < n1; n = np.minimum(n, N - 1)
        da = DA[n]                                                   # (nb, NB)
        o1 = slo[:, None] * KST * da; o2 = shi[:, None] * KST * da
        olo, ohi = np.minimum(o1, o2), np.maximum(o1, o2)            # (nb, NB)
        al_lo = (olo + alo[:, None]); al_hi = (ohi + ahi[:, None])
        def rng(coef, lo, hi):
            v1, v2 = lo * coef, hi * coef
            return np.minimum(v1, v2), np.maximum(v1, v2)
        tx_lo, tx_hi = rng(t[0], al_lo, al_hi); ty_lo, ty_hi = rng(t[1], al_lo, al_hi)
        nx_lo, nx_hi = rng(nrm[0], clo[:, None], chi[:, None]); ny_lo, ny_hi = rng(nrm[1], clo[:, None], chi[:, None])
        v = VAL[n] & okn[:, None, None]                              # (nb, NB, K)
        X = RX[n]; Y = RY[n]
        xl = X + (tx_lo + nx_lo)[..., None]; xh = X + (tx_hi + nx_hi)[..., None]
        yl = Y + (ty_lo + ny_lo)[..., None]; yh = Y + (ty_hi + ny_hi)[..., None]
        xmin = np.where(v, np.minimum(xmin, xl), xmin); xmax = np.where(v, np.maximum(xmax, xh), xmax)
        ymin = np.where(v, np.minimum(ymin, yl), ymin); ymax = np.where(v, np.maximum(ymax, yh), ymax)
        anyv |= v; nav[:, j] = np.where(okn, v.any(-1).sum(-1), 0)
    M = np.zeros((nb, NB))
    for bi, b in enumerate(bands):
        for k_ in range(KMAX):
            m = anyv[:, bi, k_]
            if m.any():
                val = rect_max(b, xmin[m, bi, k_], xmax[m, bi, k_], ymin[m, bi, k_], ymax[m, bi, k_])
                M[m, bi] = np.maximum(M[m, bi], val)
    Ms = M.T
    third = np.sort(Ms, axis=0)[-3] if NB >= 3 else np.zeros(nb)
    ub = np.sqrt((np.minimum(Ms, third) ** 2).sum(0))
    return np.where(nav.max(1) >= minb, ub, 0.0)


t0 = time.time()
import heapq


def split(bx):
    n0, n1, e0, e1, a0, a1, c0, c1 = bx
    nm, em, am, cm = (n0 + n1) // 2, (e0 + e1) // 2, (a0 + a1) // 2, (c0 + c1) // 2
    gstep = (r.get("gain_m_per_km") or 1000) * (float(np.median(np.diff(r["profile_dm"]))) if N > 1 else 0)
    ext = {"n": (n1 - n0 - 1) * gstep, "e": (e1 - e0 - 1) * 0.004 * KST * float(np.abs(DA).max()),
           "a": (a1 - a0 - 1) * step, "c": (c1 - c0 - 1) * step}
    d = max(ext, key=ext.get)
    if d == "n" and n1 - n0 > 1: return [(n0, nm, e0, e1, a0, a1, c0, c1), (nm, n1, e0, e1, a0, a1, c0, c1)]
    if d == "e" and e1 - e0 > 1: return [(n0, n1, e0, em, a0, a1, c0, c1), (n0, n1, em, e1, a0, a1, c0, c1)]
    if d == "a" and a1 - a0 > 1: return [(n0, n1, e0, e1, a0, am, c0, c1), (n0, n1, e0, e1, am, a1, c0, c1)]
    return [(n0, n1, e0, e1, a0, a1, c0, cm), (n0, n1, e0, e1, a0, a1, cm, c1)]


def nhyp(bx):
    n0, n1, e0, e1, a0, a1, c0, c1 = bx
    return (n1 - n0) * (e1 - e0) * (a1 - a0) * (c1 - c0)


init = [(n, min(n + a.chunk, N), 0, NE, 0, NA, 0, NC) for n in range(0, N, a.chunk)]
ub0 = bound(init)
heap = [(-u, bx) for u, bx in zip(ub0, init) if u > 0]; heapq.heapify(heap)
best = -1e9 if a.floor is None else a.floor; best_h = None; n_eval = 0; n_bound = len(init); it = 0
while heap and -heap[0][0] > best:
    it += 1
    batch = []
    while heap and len(batch) < 512 and -heap[0][0] > best:
        batch.append(heapq.heappop(heap)[1])
    kids = []
    for bx in batch:
        if nhyp(bx) <= 64:                                           # лист: точный счёт всех гипотез
            n0, n1, e0, e1, a0, a1, c0, c1 = bx
            for n in range(n0, n1):
                for ee in range(e0, e1):
                    AA, CC = np.meshgrid(Aa[a0:a1], Cc[c0:c1]); sc = score_points(n, ee, AA.ravel(), CC.ravel()); n_eval += sc.size
                    jj = int(np.argmax(sc))
                    if sc[jj] > best:
                        best = float(sc[jj]); best_h = (n, ee, float(AA.ravel()[jj]), float(CC.ravel()[jj]))
        else:
            kids += split(bx)
    if kids:
        uk = bound(kids); n_bound += len(kids)
        for u, bx in zip(uk, kids):
            if u > best:
                heapq.heappush(heap, (-u, bx))
    if it % 20 == 0:
        print(f"  iter {it}: queue {len(heap)}, top UB {(-heap[0][0]) if heap else 0:.2f}, best {best:.2f}, exact {n_eval}, bounds {n_bound}", flush=True)
n_brute = sum(1 for n in range(N) if len(roots[n]) >= minb) * NE * NA * NC
ref = r["score"] if a.null < 0 else r["nulls"][a.null]
if best_h is None:
    best_h = (-1, 0, float("nan"), float("nan"))
print(f"{Path(a.json).stem} {'real' if a.null < 0 else 'null ' + str(a.null)}{'' if a.floor is None else f' floor {a.floor:.2f}'}: B&B max {best:.3f}{' (= below floor)' if best_h[0] < 0 else ''} at n={best_h[0]} e={STRETCH[best_h[1]]:+.3f} a={best_h[2]:.0f} c={best_h[3]:.0f}"
      f" | brute-force max {ref:.3f} | hypotheses: brute {n_brute:.2e}, B&B exact {n_eval:.2e}, bounds {n_bound:.2e}"
      f" -> pruning x{n_brute / max(n_eval, 1):.0f} | {time.time() - t0:.0f} s, {it} levels", flush=True)

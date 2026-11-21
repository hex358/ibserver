"""Офлайн-изучение сохранённых событий (results/data/*.npz): листы снимков и поиск «вслепую».

  python scripts/study.py results/data            # все события
  python scripts/study.py results/data --only 31550

Для каждого события:
* blind: узор взаимного расположения полос (вектор корень_b − корень_ref при номинальной фазе) ищется по ВСЕЙ
  сохранённой полосе. В каждой точке p: T(p) = √(Σ_b U_b(p + Δ_b)²/n). Узор от фазы почти не зависит, поэтому
  объект находится, даже если орбита ошибается сильнее окна. Нуль — тот же узор со случайными перестановками Δ_b.
* sheet: PNG — RGB и U вокруг предсказанной точки (номинальная фаза) и вокруг лучшего кандидата.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CELL, RES = 5000.0, 10.0
N = int(CELL / RES)


def load(p):
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    U, RGB = {}, {}
    for k in z.files:
        if k.startswith("U|"):
            _, c, b = k.split("|"); U.setdefault(b, {})[tuple(map(int, c.split("_")))] = z[k].astype(np.float32) / 15
        elif k.startswith("RGB|"):
            _, c = k.split("|"); RGB[tuple(map(int, c.split("_")))] = z[k].astype(np.float32) * 40
    return meta, U, RGB


def mosaic(cells):
    """{(i,j): (N,N)} → большой массив + (x0, y1) угол."""
    ks = list(cells); I = [k[0] for k in ks]; J = [k[1] for k in ks]
    i0, i1, j0, j1 = min(I), max(I), min(J), max(J)
    sh = next(iter(cells.values())).shape
    big = np.full(((j1 - j0 + 1) * N, (i1 - i0 + 1) * N) + sh[2:], np.nan, np.float16)
    for (i, j), A in cells.items():
        r0 = (j1 - j) * N; c0 = (i - i0) * N
        big[r0:r0 + N, c0:c0 + N] = A[:N, :N]
    return big, i0 * CELL, (j1 + 1) * CELL


def rc(x, y, x0, y1):
    return int((y1 - y) // RES), int((x - x0) // RES)


class Cells:
    """Доступ к значениям полосы по координатам UTM через ячейки (без общей мозаики)."""

    def __init__(self, cells):
        self.c = cells

    def get(self, X, Y):
        I_ = np.floor(X / CELL).astype(int); J_ = np.floor(Y / CELL).astype(int)
        col = np.clip(((X - I_ * CELL) // RES).astype(int), 0, N - 1); row = np.clip((((J_ + 1) * CELL - Y) // RES).astype(int), 0, N - 1)
        out = np.full(X.shape, np.nan, np.float32)
        for key in set(zip(I_.ravel().tolist(), J_.ravel().tolist())):
            A = self.c.get(key)
            if A is None:
                continue
            m = (I_ == key[0]) & (J_ == key[1])
            out[m] = A[row[m], col[m]]
        return out


def blind(meta, U, n_null=10, seed=0, step=2):
    """Узор полос по всем сохранённым ячейкам (шаг step пикселей = 20 м), без мозаики."""
    r = meta["result"]; roots_all = r.get("roots_all") or []
    pdm = r.get("profile_dm") or [0.0]
    k0 = int(np.argmin(np.abs(np.array(pdm)))) if roots_all else None
    roots0 = roots_all[k0] if roots_all else {}
    bands = [b for b in roots0 if b in U]
    if len(bands) < 3:
        return None
    ref = bands[0]
    off = {b: (roots0[b][0][2] - roots0[ref][0][2], roots0[b][0][3] - roots0[ref][0][3]) for b in bands}
    C = {b: Cells(U[b]) for b in bands}
    g = (np.arange(0, N, step) + 0.5) * RES
    GX, GY = np.meshgrid(g, g)
    need = max(3, 0.6 * len(bands))

    def scan(offs):
        best = (-1.0, 0, 0)
        for (ci, cj) in U[ref]:
            X = ci * CELL + GX; Y = (cj + 1) * CELL - GY
            acc = np.zeros(X.shape, np.float32); cnt = np.zeros(X.shape, np.float32)
            for b in bands:
                v = C[b].get(X + offs[b][0], Y + offs[b][1]); ok = np.isfinite(v)
                acc[ok] += v[ok] ** 2; cnt[ok] += 1
            T = np.where(cnt >= need, np.sqrt(acc / np.maximum(cnt, 1)), 0)
            k = int(np.argmax(T))
            if T.flat[k] > best[0]:
                best = (float(T.flat[k]), float(X.flat[k]), float(Y.flat[k]))
        return best

    T0, bx, by = scan(off)
    rng = np.random.default_rng(seed); nulls = []
    for _ in range(n_null):
        o2 = {b: (off[b][0] + rng.uniform(-3000, 3000), off[b][1] + rng.uniform(-3000, 3000)) for b in bands}; o2[ref] = off[ref]
        nulls.append(scan(o2)[0])
    px, py = roots0[ref][0][2], roots0[ref][0][3]
    return dict(bands=bands, ref=ref, null_max=max(nulls), null_med=float(np.median(nulls)),
                cands=[dict(T=T0, dx=bx - px, dy=by - py, dist_km=math.hypot(bx - px, by - py) / 1000)],
                area_km2=len(U[ref]) * 25.0)


def sheet(meta, U, RGB, path, cand=None, half=40):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r = meta["result"]; roots_all = r.get("roots_all") or []
    pdm = r.get("profile_dm") or [0.0]; k0 = int(np.argmin(np.abs(np.array(pdm))))
    roots0 = roots_all[k0] if roots_all else {}
    bands = [b for b in ("blue", "green", "red", "nir", "rededge3", "swir16") if b in roots0 and b in U]
    if not bands:
        return
    rgbC = [Cells({k: v[..., i] for k, v in RGB.items()}) for i in range(3)] if RGB else None
    rows = [("номинал", None)] + ([("кандидат", cand)] if cand else [])
    fig, ax = plt.subplots(2 * len(rows), len(bands), figsize=(2.4 * len(bands), 4.8 * len(rows)), squeeze=False)
    o = (np.arange(-half, half) + 0.5) * RES; OX, OY = np.meshgrid(o, -o)
    for ri, (lab, c) in enumerate(rows):
        for k, b in enumerate(bands):
            x, y = roots0[b][0][2], roots0[b][0][3]
            if c is not None:
                x += c["dx"]; y += c["dy"]
            Ub = Cells(U[b]).get(x + OX, y + OY)
            if rgbC:
                im = np.stack([cc.get(x + OX, y + OY) for cc in rgbC], -1)
                ok = np.isfinite(im) & (im > 0)
                lo, hi = (np.percentile(im[ok], [1, 99]) if ok.any() else (0, 1))
                ax[2 * ri, k].imshow(np.clip((np.nan_to_num(im) - lo) / (hi - lo + 1e-6), 0, 1))
            ax[2 * ri + 1, k].imshow(np.nan_to_num(Ub), cmap="RdBu_r", vmin=-8, vmax=8)
            ax[2 * ri, k].set_title(f"{lab}: {b}", fontsize=8)
            for a_ in (ax[2 * ri, k], ax[2 * ri + 1, k]):
                a_.set_xticks([]); a_.set_yticks([]); a_.plot([half], [half], "c+", ms=10)
    e = meta["event"]
    fig.suptitle(f"{e['obj']} {e['sent']} v_al {e['v_al']:.1f}  score {r.get('score', float('nan')):.1f}"
                 + (f"  | blind T {cand['T']:.1f} at {cand['dist_km']:.1f} km" if cand else ""), fontsize=9)
    plt.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--only", default=None); ap.add_argument("--sheets", action="store_true")
    a = ap.parse_args()
    out = Path(a.dir) / "study"; out.mkdir(exist_ok=True)
    rows = []
    for p in sorted(Path(a.dir).glob("*.npz")):
        if a.only and a.only not in p.name:
            continue
        meta, U, RGB = load(p)
        r = meta["result"]; e = meta["event"]
        bl = blind(meta, U)
        if bl is None:
            print(f"{p.stem:34s} too few bands"); continue
        c0 = bl["cands"][0]
        fc = r.get("forecast") or {}
        print(f"{p.stem:34s} {r.get('status', '?'):13s} score {r.get('score', float('nan')):5.1f}/{r.get('null_max', float('nan')) + 3:5.1f}"
              f" | blind T {c0['T']:5.1f} null max {bl['null_max']:5.1f} at {c0['dist_km']:5.1f} km  (area {bl['area_km2']:.0f} km²)"
              f" | T_pred {fc.get('T_pred', float('nan')):5.1f}", flush=True)
        rows.append(dict(tag=p.stem, status=r.get("status"), score=r.get("score"), thr=r.get("null_max", 0) + 3, blind=bl))
        if a.sheets:
            sheet(meta, U, RGB, out / f"{p.stem}.png", cand=c0 if c0["T"] > bl["null_max"] else None)
    (out / "blind.json").write_text(json.dumps(rows, default=float))

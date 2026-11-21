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
    big = np.full(((j1 - j0 + 1) * N, (i1 - i0 + 1) * N) + sh[2:], np.nan, np.float32)
    for (i, j), A in cells.items():
        r0 = (j1 - j) * N; c0 = (i - i0) * N
        big[r0:r0 + N, c0:c0 + N] = A[:N, :N]
    return big, i0 * CELL, (j1 + 1) * CELL


def rc(x, y, x0, y1):
    return int((y1 - y) // RES), int((x - x0) // RES)


def blind(meta, U, n_null=20, seed=0):
    r = meta["result"]; roots_all = r.get("roots_all") or []
    pdm = r.get("profile_dm") or [0.0]
    k0 = int(np.argmin(np.abs(np.array(pdm)))) if roots_all else None
    roots0 = roots_all[k0] if roots_all else {}
    bands = [b for b in roots0 if b in U]
    if len(bands) < 3:
        return None
    ref = bands[0]
    off = {b: (roots0[b][0][2] - roots0[ref][0][2], roots0[b][0][3] - roots0[ref][0][3]) for b in bands}
    M = {}
    for b in bands:
        M[b], x0, y1 = mosaic(U[b])
    H, W = M[ref].shape
    shifts = {b: (int(round(-off[b][1] / RES)), int(round(off[b][0] / RES))) for b in bands}   # (drow, dcol)

    def stat(sh):
        acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
        for b in bands:
            dr, dc = sh[b]; A = M[b]
            S = np.full((H, W), np.nan, np.float32)
            rs, re_ = max(0, -dr), min(H, H - dr); cs, ce = max(0, -dc), min(W, W - dc)
            if rs < re_ and cs < ce:
                S[rs:re_, cs:ce] = A[rs + dr:re_ + dr, cs + dc:ce + dc]
            ok = np.isfinite(S); acc[ok] += S[ok] ** 2; cnt[ok] += 1
        T = np.where(cnt >= max(3, 0.6 * len(bands)), np.sqrt(acc / np.maximum(cnt, 1)), 0)
        return T, cnt

    T, cnt = stat(shifts)
    rng = np.random.default_rng(seed); nulls = []
    for _ in range(n_null):
        sh = {b: (shifts[b][0] + int(rng.integers(-300, 300)), shifts[b][1] + int(rng.integers(-300, 300))) for b in bands}
        sh[ref] = shifts[ref]
        nulls.append(float(np.nanmax(stat(sh)[0])))
    # топ кандидатов (разнесённые > 300 м)
    flat = np.argsort(T.ravel())[::-1]; cands = []
    for idx in flat[:5000]:
        rr, cc = divmod(int(idx), W)
        x = x0 + (cc + 0.5) * RES; y = y1 - (rr + 0.5) * RES
        if all(math.hypot(x - c[1], y - c[2]) > 300 for c in cands):
            cands.append((float(T[rr, cc]), x, y))
        if len(cands) >= 5:
            break
    # положение кандидата относительно прогноза ref при номинальной фазе (м)
    px, py = roots0[ref][0][2], roots0[ref][0][3]
    return dict(bands=bands, ref=ref, null_max=max(nulls), null_med=float(np.median(nulls)),
                cands=[dict(T=t, dx=x - px, dy=y - py, dist_km=math.hypot(x - px, y - py) / 1000) for t, x, y in cands],
                area_km2=float(np.isfinite(M[ref]).sum() * RES * RES / 1e6))


def sheet(meta, U, RGB, path, cand=None, half=40):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r = meta["result"]; roots_all = r.get("roots_all") or []
    pdm = r.get("profile_dm") or [0.0]; k0 = int(np.argmin(np.abs(np.array(pdm))))
    roots0 = roots_all[k0] if roots_all else {}
    bands = [b for b in ("blue", "green", "red", "nir", "rededge3", "swir16") if b in roots0 and b in U]
    if not bands:
        return
    RGBm, x0, y1 = mosaic(RGB) if RGB else (None, 0, 0)
    rows = [("номинал", None)] + ([("кандидат", cand)] if cand else [])
    fig, ax = plt.subplots(2 * len(rows), len(bands), figsize=(2.4 * len(bands), 4.8 * len(rows)), squeeze=False)
    for ri, (lab, c) in enumerate(rows):
        for k, b in enumerate(bands):
            M, mx0, my1 = mosaic(U[b])
            x, y = roots0[b][0][2], roots0[b][0][3]
            if c is not None:
                x += c["dx"]; y += c["dy"]
            r_, c_ = rc(x, y, mx0, my1)
            sl = (slice(max(0, r_ - half), r_ + half), slice(max(0, c_ - half), c_ + half))
            if RGBm is not None:
                rr2, cc2 = rc(x, y, x0, y1); sl2 = (slice(max(0, rr2 - half), rr2 + half), slice(max(0, cc2 - half), cc2 + half))
                im = RGBm[sl2]
                ok = np.isfinite(im) & (im > 0)
                lo, hi = (np.percentile(im[ok], [1, 99]) if ok.any() else (0, 1))
                ax[2 * ri, k].imshow(np.clip((np.nan_to_num(im) - lo) / (hi - lo + 1e-6), 0, 1))
            ax[2 * ri + 1, k].imshow(np.nan_to_num(M[sl]), cmap="RdBu_r", vmin=-8, vmax=8)
            ax[2 * ri, k].set_title(f"{lab}: {b}", fontsize=8)
            for a_ in (ax[2 * ri, k], ax[2 * ri + 1, k]):
                a_.set_xticks([]); a_.set_yticks([]); a_.plot([half], [half], "c+", ms=10)
    e = meta["event"]
    fig.suptitle(f"{e['obj']} {e['sent']} v_al {e['v_al']:.1f}  status {r.get('status')}  score {r.get('score', float('nan')):.1f}"
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

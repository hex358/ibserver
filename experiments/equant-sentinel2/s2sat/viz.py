"""Картинка результата: RGB, z и E вокруг найденного положения в каждой полосе."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .detect import RES

CELL = 5000.0


def _crop(tiles, key, b, x, y, half):
    """Кроп (2·half)² пикселей 20 м вокруг (x, y), склеенный из ячеек."""
    out = np.full((2 * half, 2 * half) + ((3,) if key == "RGB" else ()), np.nan, np.float32)
    xs = x + (np.arange(2 * half) - half + 0.5) * RES
    ys = y - (np.arange(2 * half) - half + 0.5) * RES
    XX, YY = np.meshgrid(xs, ys)
    I = np.floor(XX / CELL).astype(int); J = np.floor(YY / CELL).astype(int)
    col = np.floor((XX - I * CELL) / RES).astype(int); row = np.floor(((J + 1) * CELL - YY) / RES).astype(int)
    for (i, j), T in tiles.items():
        m = (I == i) & (J == j)
        if not m.any():
            continue
        A = T[key] if key == "RGB" else T[key].get(b)
        if A is None:
            continue
        out[m] = A[row[m], col[m]]
    return out


def plot_event(r, maps, path, half=30):
    tiles = maps["tiles"]
    bands = [b for b in ("blue", "nir", "green", "red") if b in r["roots"]] or list(r["roots"])[:4]
    fig, ax = plt.subplots(3, len(bands), figsize=(3 * len(bands), 9), squeeze=False)
    for k, b in enumerate(bands):
        x, y = r["roots"][b][0][2] + r["sx"], r["roots"][b][0][3] + r["sy"]
        rgb = _crop(tiles, "RGB", b, x, y, half)
        ok = np.isfinite(rgb) & (rgb > 0)
        lo, hi = np.percentile(rgb[ok], [1, 99]) if ok.any() else (0, 1)
        ax[0, k].imshow(np.clip((np.nan_to_num(rgb) - lo) / (hi - lo + 1e-6), 0, 1)); ax[0, k].set_title(f"{b}: RGB", fontsize=9)
        z = _crop(tiles, "Z", b, x, y, half)
        ax[1, k].imshow(np.nan_to_num(z), cmap="RdBu_r", vmin=-10, vmax=10); ax[1, k].set_title(f"{b}: z изменений", fontsize=9)
        E = _crop(tiles, "E", b, x, y, half)
        ax[2, k].imshow(np.nan_to_num(E), cmap="magma", vmin=0, vmax=10)
        ax[2, k].set_title(f"{b}: E = {r['per_band'].get(b, float('nan')):.1f}", fontsize=9)
        for a in ax[:, k]:
            a.set_xticks([]); a.set_yticks([]); a.plot([half], [half], "c+", ms=14)
    fig.suptitle(f"{r['obj']}: счёт {r['score']:.1f}, нуль max {r['null_max']:.1f}, dM {r.get('dM', 0):+.1f} км, "
                 f"сдвиг ({r['sx']:.0f}, {r['sy']:.0f}) м; кропы 1.2 км, 20 м/пкс", fontsize=10)
    plt.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)

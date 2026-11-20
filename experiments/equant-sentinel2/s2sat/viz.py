"""Картинка результата: RGB, z и E вокруг найденного положения в каждой полосе."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .detect import RES


def plot_event(r, maps, path, half=30):
    bands = [b for b in ("blue", "nir", "green", "red") if b in r["roots"]]
    minx, miny, maxx, maxy = r["bounds"]
    fig, ax = plt.subplots(3, len(bands), figsize=(3 * len(bands), 9), squeeze=False)
    for k, b in enumerate(bands):
        x, y = r["roots"][b][0][2] + r["sx"], r["roots"][b][0][3] + r["sy"]
        col = int((x - minx) // RES); row = int((maxy - y) // RES)
        sl = (slice(max(0, row - half), row + half), slice(max(0, col - half), col + half))
        rgb = maps["RGB"][sl]; lo, hi = np.percentile(rgb[rgb > 0], [1, 99]) if (rgb > 0).any() else (0, 1)
        ax[0, k].imshow(np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)); ax[0, k].set_title(f"{b}: RGB", fontsize=9)
        ax[1, k].imshow(np.nan_to_num(maps["Z"][b][sl]), cmap="RdBu_r", vmin=-10, vmax=10); ax[1, k].set_title(f"{b}: z изменений", fontsize=9)
        ax[2, k].imshow(np.nan_to_num(maps["E"][b][sl]), cmap="magma", vmin=0, vmax=10)
        ax[2, k].set_title(f"{b}: E = {r['per_band'][b]:.1f}", fontsize=9)
        for a in ax[:, k]:
            a.set_xticks([]); a.set_yticks([]); a.plot([half], [half], "c+", ms=14)
    fig.suptitle(f"{r['obj']}: счёт {r['score']:.1f}, нуль max {r['null_max']:.1f}, сдвиг ({r['sx']:.0f}, {r['sy']:.0f}) м; кропы 1.2 км, 20 м/пкс")
    plt.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)

"""Лист «по фазам»: карты U и RGB в точке прогноза для фаз dM = −3…+3 км (шаг 1) в полосах с сильным прогнозом."""
import sys, json, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location("study", str(Path(__file__).resolve().parent / "study.py")); st = importlib.util.module_from_spec(_s); _s.loader.exec_module(st)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def strip(npz, out, phases=(-3, -2, -1, 0, 1, 2, 3), half=30, nb=4):
    meta, U, RGB = st.load(Path(npz)); r = meta["result"]; e = meta["event"]
    g = np.array(r["profile_dm"]); ra = r["roots_all"]; us = (r.get("forecast") or {}).get("u") or {}
    bands = [b for b, _ in sorted(us.items(), key=lambda kv: -abs(kv[1])) if b in U][:nb]
    o = (np.arange(-half, half) + 0.5) * 10; OX, OY = np.meshgrid(o, -o)
    rgbC = [st.Cells({k: v[..., i] for k, v in RGB.items()}) for i in range(3)] if RGB else None
    fig, ax = plt.subplots(2 * len(bands), len(phases), figsize=(1.6 * len(phases), 3.3 * len(bands)), squeeze=False)
    for j, ph in enumerate(phases):
        k = int(np.argmin(np.abs(g - ph))); rr = ra[k]
        for i, b in enumerate(bands):
            for a_ in (ax[2 * i, j], ax[2 * i + 1, j]):
                a_.set_xticks([]); a_.set_yticks([])
            if b not in rr:
                ax[2 * i, j].set_title(f"{b} dM{ph:+d}: слепо", fontsize=6); continue
            x, y = rr[b][0][2], rr[b][0][3]
            Ub = st.Cells(U[b]).get(x + OX, y + OY)
            ax[2 * i + 1, j].imshow(np.nan_to_num(Ub), cmap="RdBu_r", vmin=-8, vmax=8)
            if rgbC:
                im = np.stack([c.get(x + OX, y + OY) for c in rgbC], -1); ok = np.isfinite(im) & (im > 0)
                lo, hi = (np.percentile(im[ok], [1, 99]) if ok.any() else (0, 1))
                ax[2 * i, j].imshow(np.clip((np.nan_to_num(im) - lo) / (hi - lo + 1e-6), 0, 1))
            ax[2 * i, j].set_title(f"{b} dM{ph:+d} u{us.get(b, 0):+.0f}", fontsize=6)
            for a_ in (ax[2 * i, j], ax[2 * i + 1, j]): a_.plot([half], [half], "c+", ms=7)
    fig.suptitle(f"{e['obj']} {e['sent']} v_al {e['v_al']:.1f}  status {r.get('status')}  score {r.get('score', float('nan')):.1f}", fontsize=8)
    plt.tight_layout(); fig.savefig(out, dpi=75); plt.close(fig)

if __name__ == "__main__":
    for t in sys.argv[1:]:
        p = next(Path("results/data").glob(f"*{t}*.npz")); strip(p, f"results/data/study/phase_{t}.png"); print("ok", t)

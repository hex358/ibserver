"""Кадры вращения: как вращение видно на снимке и как его определяет система — лист по моментам съёмки.

  python scripts/spin_frames.py STARLINK-30245_20261120T110322_ms                       # настоящий снимок
  python scripts/spin_frames.py STARLINK-30245_20261120T110322_ms --inject 60 --inject-axis major
Строки — снимки объекта по времени (полоса × детектор, каждый — свой момент). Столбцы:
  1) снимок: остаток после вычитания фона, в σ шума (то, что видит система);
  2) модель БЕЗ вращения (одна поза на все моменты, яркости материалов общие для снимков одной полосы);
  3) снимок − модель без вращения: если объект поворачивается, здесь остаётся структура;
  4) модель С вращением R(t) = exp([ω]×(t − t₀))·R₀ (при подстановке — истинное вращение);
  5) снимок − модель с вращением;
  6) поза объекта в этот момент — вид со стороны Sentinel (точки — элементы поверхности модели по материалам).
Справа внизу: невязка по моментам для обеих моделей — вращение видно как рост невязки неподвижной модели к краям
по времени.
"""
import argparse, json, math, pickle, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR
from s2sat import modelcat as MC, scenegeom as SG
from s2sat.diffpose import DiffSpinPose
from s2sat.diffmodel import rotvec_to_mat
from scipy.optimize import nnls

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--inject", type=float, default=None)
ap.add_argument("--inject-axis", choices=["random", "major"], default="random"); ap.add_argument("--rows", type=int, default=8)
a = ap.parse_args(); torch.set_num_threads(2); n = a.name
rc = pickle.loads((ROOT / "results/shape" / f"{n}.pkl").read_bytes()); MR.prepare(rc)
spec = MC.spec_for(int(rc["cat"]), n); C = SG.lvlh_to_fit(n, rc)
W = DiffSpinPose(rc, spec, C); W.grey = False
G = W.groups(); gt = [float(np.median([W.sc.t[b] for b in g])) for g in G]; W.t0 = gt[int(np.argmax([len(g) for g in G]))]
pp = json.loads((ROOT / "results/pose_pix_off" / f"{n}.json").read_text()); kap = float(pp["kappa"])
nd = W.nd; xs0 = np.asarray(pp["x"], float)[:3 + nd + 2]
bands = sorted(W.sc.bands, key=lambda b: W.sc.t[b])
grp = {}
for b in bands:
    grp.setdefault(b.split("#")[0], []).append(b)


def model_images(x):
    """Предсказание каждого снимка моделью x (яркости — НК, общие для снимков одной полосы)."""
    OC = {b: W.columns(x, b) for b in bands}; out = {}
    for bb in grp.values():
        A_ = np.vstack([OC[b][1].T / W.sc.B[b].s for b in bb])
        y_ = np.concatenate([(W.sc.B[b].r.numpy() + W.sc.B[b].g.numpy() * OC[b][0].sum(0)) / W.sc.B[b].s for b in bb])
        rho = nnls(A_, y_)[0]
        for b in bb:
            occ, cols = OC[b]; out[b] = cols.T @ rho - W.sc.B[b].g.numpy() * occ.sum(0)
    return out


title_mode = "настоящий снимок"
if a.inject is not None:                       # подстановка: объект в позе по пикселям, вращение, яркости по снимку
    rng = np.random.default_rng(7); ax_t = rng.normal(size=3); ax_t /= np.linalg.norm(ax_t)
    if a.inject_axis == "major":
        from s2sat import debris as DB, autopose as AP_
        _, _, ev_, axm_ = DB.shell_mass_props(np.asarray(spec["V"], float), np.asarray(spec["F"]), 1.0)
        ax_t = AP_.rotvec(xs0[:3]) @ W.Rbase.numpy() @ np.asarray(spec["M2B"], float) @ axm_[:, int(np.argmax(ev_))]
        ax_t /= np.linalg.norm(ax_t)
    xt = np.r_[xs0, ax_t * math.radians(a.inject)]
    m_true = model_images(xt)
    for b in bands:
        bd = W.sc.B[b]; bd.r = torch.tensor(m_true[b] + math.sqrt(kap) * bd.s * rng.standard_normal(len(m_true[b])), dtype=torch.float32)
    title_mode = f"подстановка: вращение {a.inject:.0f} °/с" + (" вокруг оси наибольшего момента инерции" if a.inject_axis == "major" else "")
    x_spin = xt
else:
    x_spin = None
# модель без вращения: та же поза, дожатая по всем снимкам (ω = 0)
from s2sat.diffpose import DiffPose
S = DiffPose(rc, spec, C); S.grey = False
from s2sat.diffmodel import transfer_r
transfer_r(W.sc, S.sc)
xs_fit, _ = S.polish(xs0, iters=100)
m_static = model_images(np.r_[xs_fit, 0.0, 0.0, 0.0])
if x_spin is not None:
    x_spin, _ = W.polish(x_spin, iters=60); m_spin = model_images(x_spin)
else:
    m_spin = None


def to2d(b, v):
    d = rc["bands"][b]; img = np.full(d["r"].shape, np.nan); img.flat[W.sc.B[b].idx] = v; return img, d


sel = bands if len(bands) <= a.rows else [bands[int(round(i))] for i in np.linspace(0, len(bands) - 1, a.rows)]
ncol = 6 if m_spin is not None else 4
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig = plt.figure(figsize=(2.3 * ncol + 3.2, 2.25 * len(sel) + 1.2))
gs = fig.add_gridspec(len(sel), ncol + 1, width_ratios=[1] * ncol + [1.4])
t0 = min(W.sc.t[b] for b in bands)
chi_s, chi_w = {}, {}
for b in bands:
    s_ = W.sc.B[b].s; r_ = W.sc.B[b].r.numpy()
    chi_s[b] = float(np.mean(((r_ - m_static[b]) / s_) ** 2))
    if m_spin is not None:
        chi_w[b] = float(np.mean(((r_ - m_spin[b]) / s_) ** 2))
for i, b in enumerate(sel):
    s_ = W.sc.B[b].s; r_ = W.sc.B[b].r.numpy()
    panels = [("снимок", r_ / s_), ("без вращения", m_static[b] / s_), ("снимок − без вращения", (r_ - m_static[b]) / s_)]
    if m_spin is not None:
        panels += [("с вращением", m_spin[b] / s_), ("снимок − с вращением", (r_ - m_spin[b]) / s_)]
    vmax = max(3.0, float(np.nanpercentile(np.abs(r_ / s_), 99.5)))
    for j, (lab, v) in enumerate(panels):
        ax = fig.add_subplot(gs[i, j]); img, d = to2d(b, v)
        ext = [d["X"].min() - d["X"].mean(), d["X"].max() - d["X"].mean(), d["Y"].min() - d["Y"].mean(), d["Y"].max() - d["Y"].mean()]
        ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=ext, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(lab, fontsize=9)
        if j == 0:
            ax.set_ylabel(f"{b}\nt = {W.sc.t[b] - t0:.2f} с", fontsize=8)
    # поза в этот момент: элементы поверхности модели, вид со стороны Sentinel (проекция полосы)
    ax = fig.add_subplot(gs[i, len(panels)]); x_now = x_spin if x_spin is not None else np.r_[xs_fit, 0.0, 0.0, 0.0]
    with torch.no_grad():
        xt_ = torch.tensor(x_now[None], dtype=torch.float32); xs_, w_ = xt_[:, :-3], xt_[:, -3:]
        cw, Sw = W.spots(xs_, rotvec_to_mat(w_ * (W.sc.t[b] - W.t0)))
        bd = W.sc.B[b]; P = torch.einsum("ij,bmj->bmi", bd.Mt, cw)[0].numpy(); Avis = torch.abs(torch.einsum("bmj,j->bm", Sw, bd.qt))[0].numpy()
        depth = (cw[0] @ W.sc.los_t).numpy()
    o = np.argsort(-depth); g_ = W.sc.grp.numpy() if hasattr(W.sc.grp, "numpy") else np.asarray(W.sc.grp)
    ax.scatter(P[o, 0], P[o, 1], s=0.6 + 4 * Avis[o] / max(Avis.max(), 1e-9), c=g_[o], cmap="tab10", vmin=0, vmax=9, linewidths=0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if i == 0:
        ax.set_title("поза в этот момент\n(вид с Sentinel)", fontsize=9)
# невязка по моментам
axc = fig.add_subplot(gs[:, ncol]) if False else None
fig.suptitle(f"{n}: {title_mode}. Строки — снимки по времени (полоса × детектор); цвет — σ шума", fontsize=11)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
tt = np.array([W.sc.t[b] - t0 for b in bands])
inset = fig.add_axes([0.70, 0.005, 0.28, 0.06]) if False else None
out = ROOT / "results/spin_frames"; out.mkdir(exist_ok=True)
tag = "" if a.inject is None else f"_inj{a.inject:.0f}{'_major' if a.inject_axis == 'major' else ''}"
fig.savefig(out / f"{n}{tag}.png", dpi=110); plt.close(fig)
# отдельный график: невязка по моментам
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(tt, [chi_s[b] for b in bands], "o-", label="без вращения")
if chi_w:
    ax.plot(tt, [chi_w[b] for b in bands], "s-", label="с вращением")
ax.axhline(kap, color="gray", lw=0.8, ls="--", label=f"уровень шума (κ = {kap:.2f})")
ax.set_xlabel("момент снимка, с"); ax.set_ylabel("невязка на пиксель, σ²"); ax.legend(fontsize=8); ax.set_title(f"{n}: {title_mode}", fontsize=9)
fig.tight_layout(); fig.savefig(out / f"{n}{tag}_chi.png", dpi=110); plt.close(fig)
print("готово:", out / f"{n}{tag}.png", "; невязка без вращения по моментам:", {b: round(v, 2) for b, v in chi_s.items()},
      ("; с вращением: " + str({b: round(v, 2) for b, v in chi_w.items()})) if chi_w else "")

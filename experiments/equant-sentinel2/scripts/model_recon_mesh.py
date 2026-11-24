"""Реконструкция по произвольной сетке (МКС, Тяньгун, ступени): то же, что model_recon.py (THEORY §21), но поза —
три угла (рысканье, тангаж, крен) + поворот панелей вокруг оси фермы (SARJ у МКС) + уточнение θ.

  python scripts/model_recon_mesh.py --mesh data/models/iss.obj --name ISS_20261118T222808 \
      --axes Z,X,Y --panel-axis X --out results/recon_iss

--axes: какие оси сетки идут вдоль полёта, поперёк и по радиусу при нулевых углах (у МКС: модули вдоль полёта,
ферма поперёк — ориентация XVV). Яркость групп материалов (панели / светлый металл / тёмный металл) — НМНК по
каждой полосе; проверка — поза по половине полос предсказывает другую половину.
"""
import argparse, json, math, pickle, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import model_recon as MR

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--axes", default="Z,X,Y"); ap.add_argument("--panel-axis", default="X")
ap.add_argument("--panel-mat", default="SolarPanels"); ap.add_argument("--reg-meas", action="store_true")
ap.add_argument("--maxfev", type=int, default=500); ap.add_argument("--out", default=str(ROOT / "results/recon_mesh"))
a = ap.parse_args()
REG_MEAS = a.reg_meas

V, F, MAT = [], [], []
mat = None
for ln in open(ROOT / a.mesh):
    if ln.startswith("v "):
        V.append([float(x) for x in ln.split()[1:4]])
    elif ln.startswith("usemtl"):
        mat = ln.split()[1]
    elif ln.startswith("f "):
        idx = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
        for k in range(1, len(idx) - 1):
            F.append([idx[0], idx[k], idx[k + 1]]); MAT.append(mat)
V = np.array(V); F = np.array(F); MAT = np.array(MAT)
KD = {}
for ln in open(ROOT / a.mesh.replace(".obj", ".mtl")):
    if ln.startswith("newmtl"):
        cur = ln.split()[1]
    elif ln.strip().startswith("Kd"):
        KD[cur] = np.array([float(v) for v in ln.split()[1:4]])
bright = {m for m, k in KD.items() if k.mean() > 0.45}
GROUP = np.array([0 if m == a.panel_mat else (1 if m in bright else 2) for m in MAT])
MR.F0, MR.M0 = F, GROUP
AX = {"X": 0, "Y": 1, "Z": 2}
ax_al, ax_ct, ax_rd = (AX[s.strip()] for s in a.axes.split(","))
pax = AX[a.panel_axis]
panel_v = np.zeros(len(V), bool); panel_v[np.unique(F[MAT == a.panel_mat])] = True


def rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def body(pose):
    yaw, pitch, roll, alpha = pose[:4]
    W = V.copy()
    # панели — поворот вокруг оси фермы (через центр масс панелей каждой стороны)
    if abs(alpha) > 1e-9:
        o = [i for i in range(3) if i != pax]
        c, s = math.cos(alpha), math.sin(alpha)
        for side in (-1, 1):
            m = panel_v & (np.sign(V[:, pax]) == side)
            if m.any():
                cen = V[m][:, o].mean(0)
                p = V[m][:, o] - cen
                W[np.ix_(m, o)] = cen + np.c_[c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]]
    B = np.c_[W[:, ax_al], W[:, ax_ct], W[:, ax_rd]]
    return B @ rot(yaw, pitch, roll).T


def regressors(rc, d, pose, los):
    Vb = body(pose)
    # привязка полосы: измеренный центр пятна (у близких быстрых объектов расчётные положения 10- и 20-м полос
    # расходятся на ~100 м — систематика модели времени съёмки полос); общий сдвиг θ уточняется подгонкой
    base = np.asarray(d["meas"][:2]) if (REG_MEAS and d.get("meas")) else d["p0"]
    p = base + d["J"] @ np.asarray(pose[4:7]) / 1000.0
    X, Y = d["X"], d["Y"]
    x0 = X.min() - d["res"]; y1 = Y.max() + d["res"]
    Wd = int((X.max() - X.min() + 2 * d["res"]) / MR.FINE) + 1; Hd = int((Y.max() - Y.min() + 2 * d["res"]) / MR.FINE) + 1
    masks = MR.raster(Vb, los, d["J"], p, x0, y1, Wd, Hd)
    from scipy.signal import fftconvolve
    from scipy.ndimage import map_coordinates
    out = []
    for m in masks:
        cm = fftconvolve(m, d["_k"], mode="same") if m.any() else np.zeros_like(m)
        out.append(map_coordinates(cm, [(y1 - Y) / MR.FINE, (X - x0) / MR.FINE], order=1, mode="constant"))
    return out


def total(rc, pose, bands, los):
    L = L0 = 0.0
    for b in bands:
        d = rc["bands"][b]; l, l0, _ = MR.band_fit(d, regressors(rc, d, pose, los)); L += l; L0 += l0
    return L, L0


def fit(rc, bands, inits, los):
    f = lambda x: total(rc, x, bands, los)[0]
    best = None
    for x0 in inits:
        r = minimize(f, np.array(x0, float), method="Powell", options=dict(maxfev=a.maxfev, xtol=1e-3, ftol=1e-4))
        print(f"   старт {np.round(np.degrees(x0[:4])).tolist()}: {r.fun:.0f}", flush=True)
        if best is None or r.fun < best.fun:
            best = r
    return best


def view(Vb, los, sun, tint, res=0.25, Wm=None, Hm=None):
    los = np.asarray(los, float); sun = np.asarray(sun, float)
    e1 = np.array([1.0, 0, 0]) - los[0] * los; e1 /= np.linalg.norm(e1); e2 = np.cross(los, e1)
    P2 = np.c_[Vb @ e1, Vb @ e2]
    ext = np.abs(P2).max(0) * 2 + 4
    Wp, Hp = int(ext[0] / res), int(ext[1] / res)
    col = P2[:, 0] / res + Wp / 2; row = -P2[:, 1] / res + Hp / 2
    depth = -(Vb @ los)
    img = Image.new("RGB", (Wp, Hp), (10, 12, 16)); dr = ImageDraw.Draw(img)
    for t in np.argsort(-depth[F].mean(1)):
        if not MR.facing(Vb, t, los):
            continue
        i, j, k = F[t]
        n = np.cross(Vb[j] - Vb[i], Vb[k] - Vb[i]); n /= (np.linalg.norm(n) + 1e-12)
        lam = max(0.0, float(n @ sun)); h = sun + los; h /= np.linalg.norm(h)
        spec = (max(0.0, float(n @ h)) ** 60) * (0.7 if MAT[t] == a.panel_mat else 0.3)
        c = KD.get(MAT[t], np.array([0.5, 0.5, 0.5])) * (0.15 + 0.85 * lam) + spec
        c = np.clip(c * tint, 0, 1) ** (1 / 2.2) * 255
        dr.polygon([(col[i], row[i]), (col[j], row[j]), (col[k], row[k])], fill=tuple(int(v) for v in c))
    return np.asarray(img)


def main():
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rc = pickle.loads((ROOT / "results/shape" / f"{a.name}.pkl").read_bytes())
    MR.prepare(rc)
    los = np.asarray(rc["los"], float); sun = np.asarray(rc["sun"], float)
    bands = sorted(b for b in rc["bands"] if (not REG_MEAS) or rc["bands"][b].get("meas"))
    inits = [[math.radians(y), 0, 0, math.radians(al), 0, 0, 0] for y in (0, 90) for al in (0, 45, 90, 135)]
    hA, hB = bands[0::2], bands[1::2]
    print("поза по половине полос", hA, flush=True)
    fa = fit(rc, hA, inits, los)
    LB, LB0 = total(rc, fa.x, hB, los)
    print("полная поза", flush=True)
    fb = fit(rc, bands, [list(fa.x)], los)
    L, L0 = total(rc, fb.x, bands, los)
    cs = {b: MR.band_fit(rc["bands"][b], regressors(rc, rc["bands"][b], fb.x, los))[2] for b in bands}
    rec = dict(name=a.name, mesh=a.mesh, pose_deg=[float(v) for v in np.degrees(fb.x[:4])], dtheta_m=[float(v) for v in fb.x[4:]],
               dchi2_all=float(L0 - L), dchi2_holdout=float(LB0 - LB), holdout_bands=hB, colors={b: list(map(float, c)) for b, c in cs.items()})
    (out / f"{a.name}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"RECON {a.name}: углы {np.round(rec['pose_deg']).tolist()}°, Δχ² {rec['dchi2_all']:.0f}, отложенные {hB}: Δχ² {rec['dchi2_holdout']:.0f}", flush=True)
    # картинка: исходный 10 м, модель в тех же пикселях, остаток, вид 0.25 м
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pick = lambda *bs: next((b for b in bs if b in rc["bands"] and rc["bands"][b]["res"] == 10), None)
    rgbb = [pick("red", "nir"), pick("green", "nir"), pick("blue", "green")]
    data, pred, resid = [], [], []
    for b in rgbb:
        d = rc["bands"][b]; R = regressors(rc, d, fb.x, los); occ = np.stack(R, -1); c = cs[b]
        model = (occ * c).sum(-1) - d["g"] * occ.sum(-1)
        data.append(d["g"] + d["r"]); pred.append(d["g"] + model); resid.append((d["r"] - model) / d["s"])
    lo = np.nanpercentile(np.stack(data), 1); hi = np.nanpercentile(np.stack(data), 99.7)
    nm = lambda S: np.clip((np.stack(S, -1) - lo) / (hi - lo + 1e-9), 0, 1) ** 0.8
    cm = np.array([np.mean(cs[b]) for b in rgbb]); tint = 0.6 + 0.8 * cm / (cm.max() + 1e-9)
    Vb = body(fb.x); vw = view(Vb, los, sun, tint)
    Image.fromarray(vw).save(out / f"{a.name}_view.png")
    fig, ax = plt.subplots(1, 4, figsize=(19, 5), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.6]))
    ax[0].imshow(nm(data), interpolation="nearest"); ax[0].set_title("Sentinel-2, 10 м/пкс (RGB)")
    ax[1].imshow(nm(pred), interpolation="nearest"); ax[1].set_title("модель в найденной позе, те же пиксели")
    ax[2].imshow(np.nanmax(np.abs(np.stack(resid)), 0), cmap="magma", vmin=0, vmax=8, interpolation="nearest"); ax[2].set_title("остаток, σ")
    ax[3].imshow(vw); ax[3].set_title("реконструкция: вид со стороны Sentinel-2, 0.25 м/пкс")
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.suptitle(f"{a.name}  ({Path(a.mesh).stem}): Δχ² {rec['dchi2_all']:.0f}; поза по половине полос предсказывает другую: Δχ² {rec['dchi2_holdout']:.0f}", fontsize=11)
    fig.tight_layout(); fig.savefig(out / f"{a.name}.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    main()

"""Реконструкция снимка спутника по известной 3D-модели (THEORY §21).

  python scripts/model_recon.py [--names STARLINK-30580,...] [--out results/recon]

Метод (анализ через синтез с моделью-приором, «model-based super-resolution»):
  1) сетка модели (KeepTrack starlink-v2mini.obj: размах 29 м, крылья 12.6×4.15 м, шина 2.7×4.1 м) ставится в оси
     орбиты: ось крыльев u — горизонтальна под рысканьем ψ, крылья повёрнуты вокруг u на φ, центр — уточнение θ;
  2) для каждой полосы — точная камера Sentinel (якобиан J_b, §18): проекция граней на снимок, z-буфер (алгоритм
     художника по глубине вдоль визира), маски видимых материалов на сетке 1 м, свёртка с ядром точки
     (ФРТ MSI + пиксель + смаз push-broom вдоль видимой трассы), отсчёты в центрах пикселей полосы;
  3) яркость материалов (ячейки, тыл панели, шина) в каждой полосе — неотрицательные МНК; поза (ψ, φ, θ) —
     Пауэлл по устойчивой невязке; проверка — поза по половине полос предсказывает другую половину;
  4) «улучшенный снимок»: та же сцена с разрешением 1 м — земля из снимка (интерполяция), спутник — рендер модели
     в найденной позе с измеренными цветами материалов; и вид спутника со стороны Sentinel в метрах объекта (0.25 м).
Детали мельче 10 м приходят из модели (априорного знания), а не из пикселей: данные определяют позу, цвет и
подтверждают крупную форму (проверка по отложенным полосам).
"""
import argparse, glob, json, math, pickle, re, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import minimize, nnls
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
MESH = ROOT / "data/models/starlink-v2mini.obj"
PSF = {10: 4.2, 20: 8.4, 60: 25.0}
V_SCAN = 6600.0
GROUPS = {"solar_cell": 0, "panel_back": 1}          # остальное — шина (2)
FINE = 1.0


def load_obj(path):
    V, F, M = [], [], []
    mat = None
    for ln in open(path):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("usemtl"):
            mat = ln.split()[1]
        elif ln.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
            for k in range(1, len(idx) - 1):
                F.append([idx[0], idx[k], idx[k + 1]]); M.append(GROUPS.get(mat, 2))
    return np.array(V), np.array(F), np.array(M)


V0, F0, M0 = load_obj(MESH)
BUS_HALF = 2.2          # |X| > BUS_HALF — крылья (поворот вокруг оси X на φ)


def body_vertices(psi, phi):
    """Вершины в осях орбиты (вдоль, поперёк, радиус), м. Сетка: X — ось крыльев, +Y — зенит, Z — третья ось."""
    X, Y, Z = V0[:, 0].copy(), V0[:, 1].copy(), V0[:, 2].copy()
    wing = np.abs(X) > BUS_HALF
    c, s = math.cos(phi), math.sin(phi)
    Y2 = np.where(wing, c * Y - s * Z, Y); Z2 = np.where(wing, s * Y + c * Z, Z)
    u = np.array([math.cos(psi), math.sin(psi), 0.0]); er = np.array([0.0, 0.0, 1.0]); w = np.cross(u, er)
    return X[:, None] * u + Y2[:, None] * er + Z2[:, None] * w


def kernel(d, res):
    t = np.asarray(d["row"]); vg = np.asarray(d["vapp"])
    sv = vg * res / max(abs(V_SCAN - vg @ t), 0.05 * V_SCAN)
    C = (PSF[int(res)] ** 2 + res ** 2 / 12.0) * np.eye(2) + np.outer(sv, sv) / 12.0
    h = int(math.ceil(3.5 * math.sqrt(np.linalg.eigvalsh(C).max()) / FINE))
    g = np.arange(-h, h + 1) * FINE; Xg, Yg = np.meshgrid(g, -g)
    iC = np.linalg.inv(C)
    k = np.exp(-0.5 * (iC[0, 0] * Xg ** 2 + 2 * iC[0, 1] * Xg * Yg + iC[1, 1] * Yg ** 2))
    return k / k.sum()


def facing(Vb, t, los):
    """Грань обращена к Sentinel (порядок обхода OBJ — против часовой снаружи)."""
    i, j, k = F0[t]
    n = np.cross(Vb[j] - Vb[i], Vb[k] - Vb[i])
    return float(n @ los) > 0


def raster(Vb, los, J, p, x0, y1, W, H):
    """Маски видимых групп материалов на сетке FINE (UTM), алгоритм художника по глубине вдоль визира."""
    P2 = p[None, :] + Vb @ (J / 1000.0).T
    col = (P2[:, 0] - x0) / FINE; row = (y1 - P2[:, 1]) / FINE
    depth = -(Vb @ los)                                  # больше — дальше от Sentinel
    tri_d = depth[F0].mean(1); order = [t for t in np.argsort(-tri_d) if facing(Vb, t, los)]
    img = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(img)
    for t in order:
        i, j, k = F0[t]
        dr.polygon([(col[i], row[i]), (col[j], row[j]), (col[k], row[k])], fill=int(M0[t]) + 1)
    A = np.asarray(img)
    return [(A == g + 1).astype(np.float32) for g in range(3)]


def regressors(rc, d, pose, los):
    psi, phi, dx, dy, dz = pose
    Vb = body_vertices(psi, phi)
    p = d["p0"] + d["J"] @ np.array([dx, dy, dz]) / 1000.0
    X, Y = d["X"], d["Y"]
    x0 = X.min() - d["res"]; y1 = Y.max() + d["res"]
    W = int((X.max() - X.min() + 2 * d["res"]) / FINE) + 1; H = int((Y.max() - Y.min() + 2 * d["res"]) / FINE) + 1
    masks = raster(Vb, los, d["J"], p, x0, y1, W, H)
    k = d["_k"]
    out = []
    for m in masks:
        if m.any():
            cm = fftconvolve(m, k, mode="same")
        else:
            cm = np.zeros_like(m)
        out.append(map_coordinates(cm, [(y1 - Y) / FINE, (X - x0) / FINE], order=1, mode="constant"))
    return out


def band_fit(d, R, hub=3.0):
    """y = r + g·Σocc = Σ c_m occ_m, c_m ≥ 0 (НМНК), невязка — псевдо-Хьюбер в σ."""
    ok = np.isfinite(d["r"]) & np.isfinite(d["g"])
    occ = np.stack([q[ok] for q in R], 1)
    y = d["r"][ok] + d["g"][ok] * occ.sum(1)
    if occ.max() <= 1e-6:
        c = np.zeros(3)
    else:
        c, _ = nnls(occ / d["s"], y / d["s"])
    model = occ @ c - d["g"][ok] * occ.sum(1)
    z = (d["r"][ok] - model) / d["s"]
    z0 = d["r"][ok] / d["s"]
    hl = lambda v: float((2 * hub ** 2 * (np.sqrt(1 + (v / hub) ** 2) - 1)).sum())
    return hl(z), hl(z0), c


def total(rc, pose, bands, los):
    L = L0 = 0.0
    for b in bands:
        d = rc["bands"][b]
        R = regressors(rc, d, pose, los)
        l, l0, _ = band_fit(d, R)
        L += l; L0 += l0
    return L, L0


def fit_pose(rc, bands, init, los):
    f = lambda x: total(rc, (x[0], x[1], x[2], x[3], x[4]), bands, los)[0]
    best = None
    for x0 in init:
        r = minimize(f, np.array(x0, float), method="Powell",
                     options=dict(maxfev=400, xtol=1e-3, ftol=1e-4))
        if best is None or r.fun < best.fun:
            best = r
    return best


def prepare(rc):
    from shape_fit import row_dir
    row = np.array(row_dir(rc))
    # кажущаяся скорость объекта на снимке — по снимкам ОДНОГО детектора (у соседнего детектора другой угол обзора,
    # положение отличается параллаксом, а не движением); детектор с одним снимком — скорость самого большого
    grp = {}
    for d in rc["bands"].values():
        grp.setdefault(d.get("det"), []).append(d)

    def fit_v(ds):
        P = np.array([d["p0"] for d in ds]); T = np.array([d["jd"] for d in ds]) * 86400
        A = np.c_[T - T.mean(), np.ones_like(T)]
        return np.array([np.linalg.lstsq(A, P[:, 0], rcond=None)[0][0], np.linalg.lstsq(A, P[:, 1], rcond=None)[0][0]])
    vmain = fit_v(max(grp.values(), key=len))
    for det, ds in grp.items():
        vapp = fit_v(ds) if len(ds) >= 2 else vmain
        for d in ds:
            d["row"] = row; d["vapp"] = vapp; d["_k"] = kernel(d, d["res"])
            d["s"] = float(d["s"])


def init_from_log(name):
    log = ROOT / "results/shape/fit.log"
    if log.exists():
        for ln in log.read_text().splitlines():
            m = re.match(rf"({re.escape(name)}): .*ψ\s+([\d.]+)°, φ\s+([+-][\d.]+)° .*dθ \[([^\]]+)\]", ln)
            if m:
                dth = [float(v) for v in m.group(4).split(",")]
                return [[math.radians(float(m.group(2))), math.radians(float(m.group(3)))] + dth]
    return []


def enhanced(rc, pose, cs, los, out_png, title):
    """Слева: исходный снимок 10 м (RGB = red, green, blue); в центре — улучшенный 1 м (земля — интерполяция снимка,
    спутник — рендер модели в найденной позе с измеренными цветами); справа — вид спутника со стороны Sentinel, 0.25 м."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pick = lambda *bs: next((b for b in bs if b in rc["bands"] and rc["bands"][b]["res"] == 10), None)
    rgbb = [pick("red", "nir", "green"), pick("green", "blue", "nir"), pick("blue", "green", "nir")]
    if any(b is None for b in rgbb):
        return False
    dref = rc["bands"][rgbb[1]]
    X, Y = dref["X"], dref["Y"]; x0 = X.min() - 5; y1 = Y.max() + 5
    W = int((X.max() - X.min() + 10) / FINE); H = int((Y.max() - Y.min() + 10) / FINE)
    low, high = [], []
    psi, phi, dx, dy, dz = pose
    Vb = body_vertices(psi, phi)
    for b in rgbb:
        d = rc["bands"][b]
        G = d["g"] + d["r"]                                   # исходная яркость (DN − 1000)
        low.append(G)
        gx = (np.arange(W) + 0.5) * FINE + x0; gy = y1 - (np.arange(H) + 0.5) * FINE
        GX, GY = np.meshgrid(gx, gy)
        ground = map_coordinates(np.nan_to_num(d["g"]), [(d["Y"].max() - GY) / d["res"], (GX - d["X"].min()) / d["res"]], order=3, mode="nearest")
        p = d["p0"] + d["J"] @ np.array([dx, dy, dz]) / 1000.0
        masks = raster(Vb, los, d["J"], p, x0, y1, W, H)
        k1 = np.exp(-0.5 * (np.arange(-3, 4) ** 2)[:, None] / 1.0 - 0.5 * (np.arange(-3, 4) ** 2)[None, :] / 1.0); k1 /= k1.sum()
        img = ground.copy()
        for m, c in zip(masks, cs[b]):
            ms = fftconvolve(m, k1, mode="same")
            img = img * (1 - ms) + c * ms
        high.append(img)
    def norm(stack):
        S = np.stack(stack, -1); lo, hi = np.nanpercentile(S, 1), np.nanpercentile(S, 99.5)
        return np.clip((S - lo) / (hi - lo + 1e-9), 0, 1) ** 0.8
    # вид со стороны Sentinel: ортографическая проекция на плоскость ⟂ визиру, 0.25 м
    los_ = np.asarray(los, float); e1 = np.array([1.0, 0, 0]) - los_[0] * los_; e1 /= np.linalg.norm(e1); e2 = np.cross(los_, e1)
    P2 = np.c_[Vb @ e1, Vb @ e2]; s = 0.25; Wv = int(34 / s); Hv = int(14 / s)
    col = P2[:, 0] / s + Wv / 2; row = -P2[:, 1] / s + Hv / 2
    depth = -(Vb @ los_); order = [t for t in np.argsort(-depth[F0].mean(1)) if facing(Vb, t, los_)]
    view = np.zeros((Hv, Wv, 3)); cols = []
    for g in range(3):
        cols.append(np.array([cs[b][g] for b in rgbb]))
    allc = np.array(cols); cmax = max(float(np.nanmax(allc)), 1.0)
    img = Image.new("RGB", (Wv, Hv), (12, 14, 18)); dr = ImageDraw.Draw(img)
    for t in order:
        i, j, k = F0[t]; c = np.clip(allc[M0[t]] / cmax, 0, 1) ** 0.7 * 255
        dr.polygon([(col[i], row[i]), (col[j], row[j]), (col[k], row[k])], fill=tuple(int(v) for v in c))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    lab = "RGB" if rgbb == ["red", "green", "blue"] else "/".join(rgbb)
    ax[0].imshow(norm(low), interpolation="nearest"); ax[0].set_title(f"Sentinel-2, 10 м/пкс ({lab})")
    ax[1].imshow(norm(high), interpolation="nearest"); ax[1].set_title("реконструкция, 1 м/пкс")
    ax[2].imshow(np.asarray(img)); ax[2].set_title("вид со стороны Sentinel-2, 0.25 м/пкс")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, dpi=100); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=""); ap.add_argument("--out", default=str(ROOT / "results/recon"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = [pickle.loads(p.read_bytes()) for p in sorted((ROOT / "results/shape").glob("*.pkl"))]
    recs = [rc for rc in recs if rc.get("fit_rms", 1e9) <= 80.0]
    if a.names:
        recs = [rc for rc in recs if rc["name"].startswith(tuple(a.names.split(",")))]
    summary = []
    for rc in recs:
        prepare(rc)
        los = np.asarray(rc["los"], float)
        bands = sorted(rc["bands"])
        init = init_from_log(rc["name"]) or [[math.radians(p_), math.radians(f_), 0, 0, 0] for p_ in (-20, 0, 20) for f_ in (-60, 0, 60)]
        # проверка: поза по половине полос → предсказание другой половины
        hA, hB = bands[0::2], bands[1::2]
        fa = fit_pose(rc, hA, init, los)
        LB, LB0 = total(rc, tuple(fa.x), hB, los)
        # компактный объект той же площади в том же месте — для сравнения
        # (крылья нулевой длины: шина как точка)
        # полная поза
        fb = fit_pose(rc, bands, [list(fa.x)] + init[:1], los)
        L, L0 = total(rc, tuple(fb.x), bands, los)
        cs = {}
        for b in bands:
            d = rc["bands"][b]; R = regressors(rc, d, tuple(fb.x), los); cs[b] = band_fit(d, R)[2]
        title = (f"{rc['name']}: ψ {math.degrees(fb.x[0]):+.0f}°, φ {math.degrees(fb.x[1]):+.0f}°;  "
                 f"Δχ² все полосы {L0 - L:.0f}; поза по половине полос → другая половина: Δχ² {LB0 - LB:.0f}")
        ok = enhanced(rc, tuple(fb.x), cs, los, out / f"{rc['name']}.png", title)
        rec = dict(name=rc["name"], psi_deg=math.degrees(fb.x[0]), phi_deg=math.degrees(fb.x[1]), dtheta_m=list(map(float, fb.x[2:])),
                   dchi2_all=L0 - L, dchi2_holdout=LB0 - LB, holdout_bands=hB, colors={b: list(map(float, c)) for b, c in cs.items()}, figure=ok)
        summary.append(rec)
        print(f"RECON {rc['name']}: ψ {rec['psi_deg']:+.1f}° φ {rec['phi_deg']:+.1f}°  Δχ² {rec['dchi2_all']:.0f}, отложенные полосы {hB}: Δχ² {rec['dchi2_holdout']:.0f}", flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

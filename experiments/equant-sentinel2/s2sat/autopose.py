"""Автоматическая поза объекта по снимку без ручных решений (THEORY §24).

Поворот штатного объекта ВЫЧИСЛЯЕТСЯ, а не подгоняется:
  * оси орбиты (вдоль, поперёк, радиус) — из орбиты на момент снимка (rc["sun"], rc["los"] даны в этих осях);
  * поворот тела относительно них — по опубликованному закону ориентации класса (modelcat.spec_for): рысканье режима и
    опубликованные углы (у МКС — TEA +XVV/−XVV по NASA), без подбираемых отклонений;
  * солнечные батареи — наведение на Солнце по эфемериде всеми приводами, найденными в модели (modelcat._wings: первая
    ось — от корпуса к корню крыла, вторая — вдоль крыла, если не совпадает с первой), либо по режиму («ребром»,
    «плашмя», «перпендикулярно корпусу», «неподвижны»).
Снимок только выбирает закон: для каждой гипотезы подбираются положение (ошибка орбиты) и яркость материалов по полосам
(НМНК); уточнение по половине полос A → невязка на половине B и наоборот. Побеждает наименьшая невязка на отложенных.
Поворот оценивается только там, где закона нет: ступени и обломки (ось по сфере), объект в нештатной ориентации
(свободный поворот — вектор поворота относительно штатных осей). Нештатность: свободный поворот лучше лучшего закона
более чем на T_ANOM в единицах эффективного χ² (невязка, делённая на κ — несоответствие модели снимку) и поза различима.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize, nnls
from scipy.signal import fftconvolve

FINE = 1.0
SIG_POS = 60.0         # априорная ошибка положения по орбите, м
SIG_LIB = math.radians(10)   # либрации при гравитационной стабилизации
T_ANOM = 25.0          # порог значимости свободного поворота над законом, эффективный χ²
ANOM_DEG = 30.0        # нештатный режим — поворот относительно лучшего закона больше этого; меньше — отклонение в режиме


def rot_axis(a, ang):
    a = np.asarray(a, float); a = a / np.linalg.norm(a); c, s = math.cos(ang), math.sin(ang)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def rotvec(w):
    th = float(np.linalg.norm(w))
    return np.eye(3) if th < 1e-12 else rot_axis(np.asarray(w, float) / th, th)


def att(yaw, pitch, roll):
    return rot_axis([0, 0, 1], yaw) @ rot_axis([0, 1, 0], pitch) @ rot_axis([1, 0, 0], roll)


def law_attitude(h):
    """Поворот штатных осей тела в орбитальные по закону: рысканье режима + опубликованные углы (в наших осях)."""
    a = h.get("att_deg", (0.0, 0.0, 0.0))
    return att(math.radians(h.get("yaw", 0.0) + a[0]), math.radians(a[1]), math.radians(a[2]))


def _align(u, v):
    """Поворот, переводящий единичный u в v."""
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v); c = float(u @ v); ax = np.cross(u, v); s = np.linalg.norm(ax)
    if s < 1e-9:
        return np.eye(3) if c > 0 else rot_axis(np.cross(u, [0.3, 0.5, 0.8]), math.pi)
    return rot_axis(ax / s, math.atan2(s, c))


def array_angle(n0, a, target):
    """Угол поворота вокруг оси a, переводящий нормаль n0 в проекцию target на плоскость, перпендикулярную a."""
    t = target - (target @ a) * a
    return math.atan2(float(t @ np.cross(a, n0)), float(t @ n0))


def _turn(V, m, axis, point, ang):
    Rm = rot_axis(axis, ang); V[m] = (V[m] - point) @ Rm.T + point


def pose_body(spec, h, x, sun_b):
    """Вершины в орбитальных осях (м), матрица R (сетка → орбитальные оси) и углы приводов панелей.
    x = [3 параметра поворота (только для оценки: кувыркание / свободный поворот / либрации), угол панелей (только для
    свободного поворота), сдвиг положения 3]."""
    sun_b = np.asarray(sun_b, float)
    if h.get("gg"):                                        # гравитационная стабилизация: продольная ось к Земле
        R = att(x[0], x[1], x[2]) @ _align(spec["M2B"] @ np.array([1.0, 0, 0]), np.array([0.0, 0, 1])) @ spec["M2B"]
        return spec["V"] @ R.T, R, []
    if h.get("tumble"):                                    # ось ступени по направлению (азимут, высота)
        az, el = x[0], x[1]; d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        R = _align(spec["M2B"] @ np.array([1.0, 0, 0]), d) @ spec["M2B"]
        return spec["V"] @ R.T, R, []
    if h.get("free3"):                                     # свободный поворот: вектор поворота относительно штатных осей
        R = rotvec(x[:3]) @ spec["M2B"]
        V = spec["V"].copy(); angs = []
        for ar in spec["arrays"]:
            _turn(V, ar["mask"], ar["axis"], ar["point"], x[3]); angs.append(float(x[3]))
        return V @ R.T, R, angs
    R = law_attitude(h) @ spec["M2B"]
    V = spec["V"].copy(); angs = []
    mode = h.get("arrays", "fixed")
    t_m = R.T @ sun_b                                      # Солнце в осях сетки
    for ar in spec["arrays"]:
        a, n0 = np.asarray(ar["axis"], float), np.asarray(ar["normal0"], float)
        if mode == "fixed":
            angs.append(0.0); continue
        if mode == "flat":
            tgt = R.T @ np.array([0.0, 0, 1])
        elif mode == "vertical":
            tg = sun_b.copy(); tg[2] = 0.0
            tgt = R.T @ (tg if np.linalg.norm(tg) > 1e-6 else np.array([1.0, 0, 0]))
        else:
            tgt = t_m
        alpha = array_angle(n0, a, tgt) + (math.pi / 2 if mode == "edge" else 0.0)
        if mode == "sun" and ar.get("sub"):                # два привода: первый ставит длину крыла ⟂ Солнцу,
            a2 = np.asarray(ar["sub"][0]["axis"], float)    # второй поворачивает крыло вокруг длины нормалью на Солнце
            d = np.cross(a, t_m)
            if np.linalg.norm(d) > 1e-6:
                alpha = array_angle(a2, a, d / np.linalg.norm(d))
            t1 = rot_axis(a, alpha).T @ t_m                # Солнце в осях крыла до поворота первым приводом
            for sw in ar["sub"]:
                b = array_angle(np.asarray(sw["normal0"], float), np.asarray(sw["axis"], float), t1)
                b = (b + math.pi / 2) % math.pi - math.pi / 2   # ближайшая из двух сторон
                _turn(V, sw["mask"], sw["axis"], sw["point"], b)
        _turn(V, ar["mask"], a, ar["point"], alpha); angs.append(alpha)
    return V @ R.T, R, angs


def raster(Vb, F, G, ng, los, J, p, x0, y1, W, H):
    """Маски групп материалов на сетке FINE (UTM): алгоритм художника по глубине вдоль визира, обе стороны граней."""
    P2 = p[None, :] + Vb @ (J / 1000.0).T
    col = (P2[:, 0] - x0) / FINE; row = (y1 - P2[:, 1]) / FINE
    depth = -(Vb @ los)
    img = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(img)
    cf, rf = col[F], row[F]
    area = np.abs((cf[:, 1] - cf[:, 0]) * (rf[:, 2] - rf[:, 0]) - (cf[:, 2] - cf[:, 0]) * (rf[:, 1] - rf[:, 0]))
    inside = (cf.max(1) >= 0) & (cf.min(1) < W) & (rf.max(1) >= 0) & (rf.min(1) < H) & (area > 1e-4)
    order = np.where(inside)[0]; order = order[np.argsort(-depth[F[order]].mean(1))]
    for t in order:
        dr.polygon([(cf[t, 0], rf[t, 0]), (cf[t, 1], rf[t, 1]), (cf[t, 2], rf[t, 2])], fill=int(G[t]) + 1)
    A = np.asarray(img)
    return [(A == g + 1).astype(np.float32) for g in range(ng)]


def regress(d, Vb, F, G, ng, los, dpos, cache=None, key=None):
    """Карты групп в пикселях полосы. Маски рисуются вокруг опорного положения (без сдвига) и кэшируются по позе;
    сдвиг положения dpos — только сдвиг точек выборки (интерполяция), без перерисовки."""
    base = np.asarray(d["meas"][:2], float) if d.get("meas") else np.asarray(d["p0"], float)
    sh = d["J"] @ np.asarray(dpos, float) / 1000.0
    X, Y = d["X"], d["Y"]
    ent = cache.get(key) if cache is not None else None
    if ent is None:
        x0 = X.min() - d["res"] - 120.0; y1 = Y.max() + d["res"] + 120.0
        Wd = int((X.max() - X.min() + 2 * d["res"] + 240.0) / FINE) + 1; Hd = int((Y.max() - Y.min() + 2 * d["res"] + 240.0) / FINE) + 1
        cms = []
        for m in raster(Vb, F, G, ng, los, d["J"], base, x0, y1, Wd, Hd):
            cms.append(fftconvolve(m, d["_k"], mode="same") if m.any() else None)
        ent = (x0, y1, cms)
        if cache is not None:
            if len(cache) > 400:
                cache.clear()
            cache[key] = ent
    x0, y1, cms = ent
    out = []
    for cm in cms:
        if cm is None:
            out.append(np.zeros_like(X, dtype=float)); continue
        out.append(map_coordinates(cm, [(y1 - (Y - sh[1])) / FINE, ((X - sh[0]) - x0) / FINE], order=1, mode="constant"))
    return out


def band_loss(d, R, hub=3.0):
    ok = np.isfinite(d["r"]) & np.isfinite(d["g"])
    occ = np.stack([q[ok] for q in R], 1); s = float(d["s"])
    y = d["r"][ok] + d["g"][ok] * occ.sum(1)
    c = nnls(occ / s, y / s)[0] if occ.max() > 1e-6 else np.zeros(occ.shape[1])
    z = (d["r"][ok] - (occ @ c - d["g"][ok] * occ.sum(1))) / s
    return float((2 * hub ** 2 * (np.sqrt(1 + (z / hub) ** 2) - 1)).sum()), c


def null_loss(d, hub=3.0):
    ok = np.isfinite(d["r"]); z = d["r"][ok] / float(d["s"])
    return float((2 * hub ** 2 * (np.sqrt(1 + (z / hub) ** 2) - 1)).sum())


class Problem:
    def __init__(self, rc, spec, C=None):
        """C — поворот из инерциальных орбитальных осей (в них законы ориентации) в оси реконструкции (в них записаны
        якобианы полос, визир и Солнце rc): scenegeom.lvlh_to_fit."""
        self.rc, self.spec = rc, spec
        self.C = np.eye(3) if C is None else np.asarray(C, float)
        self.sun = np.asarray(rc["sun"], float); self.los = np.asarray(rc["los"], float)
        self.sun_l = self.C.T @ self.sun                   # Солнце в инерциальных орбитальных осях
        self.ng = int(spec["G"].max()) + 1
        self.bands = sorted(rc["bands"], key=lambda b: rc["bands"][b]["jd"])
        self.cache = {}

    @staticmethod
    def free_idx(h):
        """Какие компоненты x подбираются: у закона — только положение; у оценки поворота — поворот и положение."""
        if h.get("free3"):
            return [0, 1, 2, 3, 4, 5, 6]
        if h.get("gg"):
            return [0, 1, 2, 4, 5, 6]
        if h.get("tumble"):
            return [0, 1, 4, 5, 6]
        return [4, 5, 6]

    def body(self, h, x):
        """Вершины в осях реконструкции: поза по закону в инерциальных орбитальных осях, затем поворот C."""
        return pose_body(self.spec, h, x, self.sun_l)[0] @ self.C.T

    def loss(self, h, x, bands, prior=True):
        Vb = self.body(h, x)
        L = 0.0
        rot_free = h.get("free3") or h.get("gg") or h.get("tumble")
        ak = (h["name"],) + (tuple(np.round(np.asarray(x[:4], float), 7)) if rot_free else ())
        for b in bands:
            d = self.rc["bands"][b]
            L += band_loss(d, regress(d, Vb, self.spec["F"], self.spec["G"], self.ng, self.los, x[-3:], self.cache, ak + (b,)))[0]
        if prior:
            L += float(np.sum((np.asarray(x[-3:]) / SIG_POS) ** 2))                         # ошибка орбиты
            if h.get("gg"):
                L += (x[1] / SIG_LIB) ** 2 + (x[2] / SIG_LIB) ** 2                          # либрации
        return L

    def fit(self, h, bands, x0, maxfev=160):
        x0 = np.asarray(x0, float); idx = self.free_idx(h)

        def f(u):
            x = x0.copy(); x[idx] = u
            return self.loss(h, x, bands)
        r = minimize(f, x0[idx], method="Powell", options=dict(maxfev=maxfev, xtol=1e-3, ftol=1e-4))
        x = x0.copy(); x[idx] = r.x
        return x, r.fun

    def starts(self, h, bands):
        if h.get("tumble"):
            # ось по сфере (полусфера: цилиндр почти симметричен), 60 направлений — лучший старт по всем полосам
            k = np.arange(60) + 0.5; el = np.arcsin(k / 60); az = math.pi * (1 + 5 ** 0.5) * k
            cand = [np.array([a_ % (2 * math.pi), e_, 0, 0, 0, 0, 0]) for a_, e_ in zip(az, el)]
            sc = [self.loss(h, c, bands) for c in cand]
            return [cand[i] for i in np.argsort(sc)[:3]]
        if h.get("free3"):
            return [np.array([0, 0, math.radians(y), math.radians(a_), 0, 0, 0]) for y in (0, 90, 180, 270) for a_ in (0, 90)]
        if h.get("gg"):
            return [np.array([math.radians(y), 0, 0, 0, 0, 0, 0]) for y in (0, 45, 90, 135)]
        return [np.zeros(7)]

    def pos_grid(self, h, bands, x0, half=60.0, step=15.0):
        """Положение — перебор по сетке (вдоль, поперёк) ±60 м с шагом 15 м: маски не перерисовываются (только сдвиг
        выборки), поэтому дёшево; без него оптимизатор застревает, если объект смещён на десятки метров."""
        best, bl = np.asarray(x0, float), self.loss(h, x0, bands)
        g = np.arange(-half, half + 1e-6, step)
        for dx in g:
            for dy in g:
                x = np.array(x0, float); x[-3] += dx; x[-2] += dy
                L = self.loss(h, x, bands)
                if L < bl:
                    best, bl = x, L
        return best

    def evaluate(self, h, maxfev=160):
        """Кросс-проверка: поза по A → невязка на B и наоборот; затем поза по всем полосам."""
        A, B = self.bands[0::2], self.bands[1::2]
        st = [self.pos_grid(h, A, s) for s in self.starts(h, A)]
        if len(st) > 1:
            best = min(st, key=lambda s: self.loss(h, s, A)); xa, _ = self.fit(h, A, best, maxfev)
        else:
            xa, _ = self.fit(h, A, st[0], maxfev)
        xb, _ = self.fit(h, B, xa, maxfev)
        cv = self.loss(h, xa, B, prior=False) + self.loss(h, xb, A, prior=False)
        xf, Lf = self.fit(h, self.bands, xb, maxfev)
        null = sum(null_loss(self.rc["bands"][b]) for b in self.bands)
        return dict(name=h["name"], x=[float(v) for v in xf], cv=float(cv), L=float(self.loss(h, xf, self.bands, prior=False)), null=float(null))


def pose_rms(spec, sun, ha, xa, hb, xb):
    """Различие двух поз: СКО смещения вершин (м) без учёта сдвига положения. На пикселе 10 м позы, различающиеся
    меньше чем на EQ_RMS, снимком не различимы."""
    Va = pose_body(spec, ha, np.asarray(xa, float), sun)[0]; Vb = pose_body(spec, hb, np.asarray(xb, float), sun)[0]
    D = (Va - Va.mean(0)) - (Vb - Vb.mean(0))
    return float(np.sqrt((D ** 2).sum(1).mean()))


def eq_rms(spec):
    ext = spec["V"].max(0) - spec["V"].min(0)
    return max(1.0, 0.06 * float(np.max(ext)))


def decide(out, spec, sun_l, law_best, free, kappa, thr):
    """Штатно или нет. Свободный поворот значимо лучше закона (T_ANOM в эффективном χ²) и поза различима:
    поворот > ANOM_DEG — нештатный режим (кувыркание, авария); меньше — штатный режим с измеренным отклонением."""
    Rb = pose_body(spec, law_best["law"], np.asarray(law_best["x"]), sun_l)[1]
    Rf = pose_body(spec, free["law"], np.asarray(free["x"]), sun_l)[1]
    dev = math.degrees(math.acos(np.clip((np.trace(Rf @ Rb.T) - 1) / 2, -1, 1)))
    gain = (law_best["cv"] - free["cv"]) / kappa
    sig = gain > T_ANOM and free["rms_vs_best_m"] >= thr
    out["free_gain_eff"] = gain
    out["deviation_deg"] = dev if sig else None              # измеренное отклонение от закона (только если значимо)
    out["anomalous"] = bool(sig and dev > ANOM_DEG)
    if out["anomalous"]:
        out["chosen"] = free["name"]
    return out


def solve(rc, spec, log=print, maxfev=160, C=None):
    P = Problem(rc, spec, C)
    res = []
    for h in spec["laws"]:
        r = P.evaluate(h, maxfev); r["law"] = h; res.append(r)
        log(f"   {h['name']:52s} невязка на отложенных {r['cv']:10.0f}   по всем {r['L']:10.0f}")
    npix = sum(int(np.isfinite(P.rc["bands"][b]["r"]).sum()) for b in P.bands)
    law_best = min(res, key=lambda r: r["cv"])
    thr = eq_rms(spec)
    for r in res:                                        # насколько поза каждой гипотезы отличается от лучшей
        r["rms_vs_best_m"] = pose_rms(spec, P.sun_l, law_best["law"], law_best["x"], r["law"], r["x"])
    equiv = [r["name"] for r in res if r["rms_vs_best_m"] < thr]
    dist = sorted(r["cv"] for r in res if r["rms_vs_best_m"] >= thr)
    free = None
    if spec["kind"] in ("station", "starlink", "payload"):
        hf = dict(name="свободный поворот (оценка)", free3=True)
        free = P.evaluate(hf, maxfev); free["law"] = hf
        free["rms_vs_best_m"] = pose_rms(spec, P.sun_l, law_best["law"], law_best["x"], hf, free["x"])
        log(f"   {hf['name']:52s} невязка на отложенных {free['cv']:10.0f}   по всем {free['L']:10.0f}   отличие позы {free['rms_vs_best_m']:.1f} м")
    # κ: средняя невязка на пиксель у лучшей позы — несоответствие модели снимку (яркость, BRDF, смаз) сверх шума.
    # Разности невязок делятся на κ — «эффективный χ²»; без этого запас и порог нештатности завышены во много раз.
    kappa = max(1.0, min([r["L"] for r in res] + ([free["L"]] if free else [])) / max(1, npix))
    margin = float(dist[0] - law_best["cv"]) if dist else None
    # законы, которые снимок не отличает от лучшего статистически: невязка на отложенных хуже менее чем на 4 эфф. χ²
    consistent = [r["name"] for r in sorted(res, key=lambda r: r["cv"]) if (r["cv"] - law_best["cv"]) / kappa < 4.0]
    out = dict(kind=spec["kind"], source=spec["source"], hypotheses=[{k: v for k, v in r.items() if k != "law"} for r in res],
               chosen=law_best["name"], equivalent=equiv, consistent=consistent, eq_rms_m=thr, kappa=kappa, npix=npix,
               margin=margin, margin_eff=None if margin is None else margin / kappa, anomalous=False)
    chosen = law_best
    if free is not None:
        out["free"] = {k: v for k, v in free.items() if k != "law"}
        decide(out, spec, P.sun_l, law_best, free, kappa, thr)
        if out["anomalous"]:
            chosen = free
    Vb, R, angs = pose_body(spec, chosen["law"], np.asarray(chosen["x"]), P.sun_l)
    out["sun_lvlh"] = P.sun_l.tolist(); out["C_lvlh_to_fit"] = P.C.tolist()
    out["R"] = R.tolist(); out["array_angles_deg"] = [math.degrees(a_) for a_ in angs]; out["x"] = chosen["x"]; out["law"] = {k: v for k, v in chosen["law"].items()}
    return out

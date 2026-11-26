"""Физика для карточки объекта и досье увода мусора (PLAN_characterization §2.4–2.5). Одна для всех объектов.

* площадь сечения: средняя по сфере (беспорядочное вращение) и по направлению (к Солнцу, к набегающему потоку);
* масса-инерционные свойства тонкостенной оболочки (масса по площади поверхности): центр масс, тензор инерции,
  главные оси; ось наибольшего момента — ось, к которой приходит вращение тела с потерями энергии (плоское вращение);
* баллистический коэффициент B = C_D·A/m: из модели (A — из позы или среднее по сфере, m — каталог) и из TLE
  (B = 12.741621·B*, B* в 1/ER — стандартная связь SGP4);
* торможение: наблюдаемое по TLE (ṅ из строки 1 → ȧ = −⅔·a·ṅ/n) и по модели с экспоненциальной атмосферой
  (таблица Vallado 8-4, средняя солнечная активность); отношение — поправка «атмосфера × C_D × площадь» к модели;
* срок до входа в атмосферу (до 120 км) без манёвров: интеграл ∫dh/|ȧ(h)|, профиль плотности из таблицы, уровень —
  по наблюдаемому торможению объекта (если есть) или по модели.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull

MU = 3.986004418e14                     # м³/с²
RE = 6378.137                           # км
CD = 2.2                                # коэффициент сопротивления, свободномолекулярный поток, типовое значение
BSTAR_TO_B = 12.741621                  # B [м²/кг] = 12.741621 · B* [1/ER]

# Vallado, «Fundamentals of Astrodynamics», табл. 8-4: h₀ (км), ρ₀ (кг/м³), H (км)
_ATM = np.array([
    (0, 1.225, 7.249), (25, 3.899e-2, 6.349), (30, 1.774e-2, 6.682), (40, 3.972e-3, 7.554), (50, 1.057e-3, 8.382),
    (60, 3.206e-4, 7.714), (70, 8.770e-5, 6.549), (80, 1.905e-5, 5.799), (90, 3.396e-6, 5.382), (100, 5.297e-7, 5.877),
    (110, 9.661e-8, 7.263), (120, 2.438e-8, 9.473), (130, 8.484e-9, 12.636), (140, 3.845e-9, 16.149),
    (150, 2.070e-9, 22.523), (180, 5.464e-10, 29.740), (200, 2.789e-10, 37.105), (250, 7.248e-11, 45.546),
    (300, 2.418e-11, 53.628), (350, 9.518e-12, 53.298), (400, 3.725e-12, 58.515), (450, 1.585e-12, 60.828),
    (500, 6.967e-13, 63.822), (600, 1.454e-13, 71.835), (700, 3.614e-14, 88.667), (800, 1.170e-14, 124.64),
    (900, 5.245e-15, 181.05), (1000, 3.019e-15, 268.00)])


def density(h_km):
    """Плотность атмосферы, кг/м³ (экспоненциальная модель по слоям)."""
    h = np.atleast_1d(np.asarray(h_km, float))
    i = np.clip(np.searchsorted(_ATM[:, 0], h, side="right") - 1, 0, len(_ATM) - 1)
    h0, r0, H = _ATM[i].T
    out = r0 * np.exp(-(h - h0) / H)
    return out if np.ndim(h_km) else float(out[0])


def adot_circ(h_km, B):
    """ȧ круговой орбиты, км/сут: da/dt = −ρ·B·√(μa)."""
    a = (RE + h_km) * 1e3
    return -density(h_km) * B * math.sqrt(MU * a) * 86400.0 / 1e3


def lifetime_days(h_km, B, k=1.0, h_end=120.0, dh=0.5):
    """Время спуска круговой орбиты с h_km до h_end, сут; k — множитель к плотности (уровень атмосферы)."""
    if B is None or B <= 0 or h_km <= h_end:
        return None
    hs = np.arange(h_end + dh / 2, h_km, dh)
    v = np.array([abs(adot_circ(h, B)) * k for h in hs])
    return float(np.sum(dh / np.maximum(v, 1e-30)))


def proj_area(V, d, F=None, npx=512):
    """Площадь силуэта на плоскости ⟂ d, м². С гранями F — растеризация проекций всех граней (настоящий силуэт:
    просветы между панелями, фермой и корпусом не считаются); без F — выпуклая оболочка (только для выпуклых тел)."""
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    e1 = np.cross(d, [0.0, 0, 1] if abs(d[2]) < 0.9 else [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1); P = np.c_[V @ e1, V @ e2]
    if F is None:
        try:
            return float(ConvexHull(P).volume)
        except Exception:                                       # noqa: BLE001
            return 0.0
    from PIL import Image, ImageDraw
    lo = P.min(0); ext = float(np.max(P.max(0) - lo)) or 1e-6; px = ext / (npx - 2)
    Q = (P - lo) / px + 1.0
    img = Image.new("1", (npx, npx), 0); dr = ImageDraw.Draw(img)
    for tri in Q[np.asarray(F)]:
        dr.polygon([tuple(v) for v in tri], fill=1)
    return float(np.count_nonzero(np.asarray(img))) * px * px


def sphere_dirs(n=400):
    k = np.arange(n) + 0.5; th = np.arccos(1 - 2 * k / n); ph = math.pi * (1 + 5 ** 0.5) * k
    return np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]


def tumbling_area(V, F=None, n=100):
    """Средняя площадь силуэта при беспорядочном вращении, м² (по n направлениям сферы; у выпуклого тела = ¼ площади
    поверхности)."""
    return float(np.mean([proj_area(V, d, F) for d in sphere_dirs(n)]))


def shell_mass_props(V, F, mass):
    """Тонкостенная оболочка: масса распределена по площади граней. Центр масс, тензор инерции (относительно центра
    масс, в осях V), главные моменты по возрастанию и оси (столбцы)."""
    T = V[F]; A = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
    w = A / max(A.sum(), 1e-12)
    c = T.mean(1); com = (c * w[:, None]).sum(0)
    if not mass:
        return com, None, None, None
    m = w * mass; r = c - com
    # грань как треугольная пластина: второй момент = m/12·(Σv_i v_iᵀ + 9 c cᵀ) относительно центра масс; проще и
    # достаточно для крупных моделей — точечные массы в центрах граней плюс собственный момент грани
    d = T - c[:, None, :]
    S = np.einsum("f,fij,fik->jk", m / 12.0, d, d) + np.einsum("f,fj,fk->jk", m, r, r)   # ∫ r rᵀ dm
    I = np.trace(S) * np.eye(3) - S
    ev, ax = np.linalg.eigh(I)
    return com, I, ev, ax


def tle_decay(satrec):
    """Наблюдаемое торможение по TLE: ȧ (км/сут) из ṅ; B* → B (м²/кг). ṅ в satrec хранится как ṅ/2 (рад/мин²)."""
    n = float(satrec.no_kozai)                                   # рад/мин
    a = float(satrec.a) * float(satrec.radiusearthkm)            # км
    nd = 2.0 * float(satrec.ndot)                                # рад/мин²
    adot = -2.0 / 3.0 * a * nd / n * 1440.0 if n > 0 else None   # км/сут
    return dict(a_km=a, e=float(satrec.ecco), h_mean_km=a - RE, perigee_km=float(satrec.altp) * float(satrec.radiusearthkm),
                apogee_km=float(satrec.alta) * float(satrec.radiusearthkm), adot_km_day=adot,
                bstar=float(satrec.bstar), B_tle=BSTAR_TO_B * float(satrec.bstar))

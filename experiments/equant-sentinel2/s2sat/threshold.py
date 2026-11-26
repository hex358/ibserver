"""Порог детекции по нуль-прогонам (THEORY.md §12).

Нуль: счёт при случайных смещениях полос. По 1908 нуль-прогонам (194 события) превышение одного прогона над медианой
остальных: 99 % — 2.6, 99.5 % — 3.2; порог «медиана + 4» даёт ложную тревогу ≤ 0.37 % на событие и не зависит от
одного выброса (прежний «максимум из 10 + 3» прыгал на 2–3 единицы от одного прогона). В сценах с тяжёлым хвостом
нуля (максимум − медиана > 4: края облаков и т. п.) оставлен прежний строгий порог.
"""
import numpy as np

MARGIN_MED = 4.0
HEAVY = 4.0


def threshold(r):
    n = [v for v in (r.get("nulls") or []) if v is not None and v > -1e8]
    if len(n) >= 5:
        med, mx = float(np.median(n)), float(max(n))
        return med + MARGIN_MED if mx - med <= HEAVY else mx + 3.0
    return r["null_max"] + 3.0


# --- решение с учётом априорного положения по орбите (THEORY §19) -------------------------------------------------
# Порог «медиана нуля + 4» — для поиска по ВСЕМУ окну фаз (поправка на множественность по всему окну). Но истина почти
# всегда у одного из источников орбиты: σ ≈ 0,85 км (17/17 эталонов ≤ 1,5 км от ближайшего источника). Поэтому второй
# ярус: внутри ±2σ вокруг каждого источника — свой нуль (максимумы нуль-прогонов по тем же фазам), p-значение по
# распределению Гумбеля максимумов и поправка Бонферрони на число источников. Общая ложная тревога — та же 0,37 %.
P_FA = 0.0037
SIG_SRC = 0.85
# Гумбель по 10–20 нулям занижает хвост (на 3980 нуль-прогонах «по одному против остальных» p ≤ 0,0037 дало 0,90 %
# ложных) → отсечка p откалибрована на тех же нулях: 0,37 % нулей ниже неё (results/prior_tier_calibration.json)
P_CUT = 8.93e-4


def gumbel_p(s, maxima):
    m = np.asarray([v for v in maxima if np.isfinite(v)], float)
    if len(m) < 8:
        return float("nan")
    beta = max(1e-3, m.std(ddof=1) * np.sqrt(6) / np.pi); mu = m.mean() - 0.5772 * beta
    return float(1.0 - np.exp(-np.exp(-(s - mu) / beta)))


def prior_test(r, centers=None, sig=SIG_SRC, k=2.0):
    """→ dict(p=наименьшее p с поправкой, center, score, null_max_local) или None, если профилей нет."""
    centers = centers if centers is not None else (r.get("source_centers") or [0.0])
    if not r.get("profile") or not r.get("null_profiles") or not r.get("profile_dm"):
        return None
    dm = np.asarray(r["profile_dm"], float); P = np.asarray(r["profile"], float); N = np.asarray(r["null_profiles"], float)
    best = None
    for c in centers:
        m = np.abs(dm - c) <= k * sig
        if not m.any() or not np.isfinite(P[m]).any():
            continue
        s = float(np.nanmax(P[m])); loc = [float(np.nanmax(row[m])) for row in N if np.isfinite(row[m]).any()]
        p = gumbel_p(s, loc) * len(centers)
        if np.isfinite(p) and (best is None or p < best["p"]):
            best = dict(p=float(min(1.0, p)), center=float(c), score=s, null_max_local=float(max(loc)),
                        dm=float(dm[m][np.nanargmax(P[m])]))
    return best

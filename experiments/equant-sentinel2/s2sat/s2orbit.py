"""Точная орбита Sentinel-2 по бортовому GPS (THEORY §25).

В метаданных полосы съёмки L1C (DATASTRIP/*/MTD_DS.xml) записаны положения и скорости спутника по GPS: шаг 1 с,
ошибка ~1 м, система ITRF (вращается с Землёй), шкала времени GPS (UTC + 18 с). TLE самого Sentinel ошибается на
сотни метров (S2B 18.11.2026: 134 м вдоль, 125 поперёк, 106 по радиусу), и эта ошибка примерно наполовину переходит
в положение найденного объекта. Метаданные берутся из открытого архива Google Cloud (gcp-public-data-sentinel-2), без
ключей; кэш — data/cache/s2gps/.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from s2sat import orbits as O

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/s2gps"
BUCKET = "gcp-public-data-sentinel-2"
GPS_UTC = 18.0                                     # с: GPS − UTC (с 2017)


def _list(prefix):
    out, tok = [], None
    while True:
        q = {"prefix": prefix, "fields": "items(name),nextPageToken", "maxResults": "1000"}
        if tok:
            q["pageToken"] = tok
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=60) as f:
            d = json.load(f)
        out += [it["name"] for it in d.get("items", [])]
        tok = d.get("nextPageToken")
        if not tok:
            return out


def mtd_ds_name(item_id, jd):
    """Путь MTD_DS.xml в архиве для снимка Earth Search вида S2B_T59GNN_20261118T222728_L2A: тот же спутник, тайл и
    дата; если полос съёмки несколько — с началом съёмки ближе всего к jd."""
    sat, tile, stamp = item_id.split("_")[:3]
    zone, lat, sq = tile[1:3], tile[3], tile[4:6]            # зона в архиве — двумя цифрами («02»)
    names = [n for n in _list(f"tiles/{zone}/{lat}/{sq}/{sat}_MSIL1C_{stamp[:8]}") if n.endswith("/MTD_DS.xml")]
    if not names:
        return None

    def start(n):
        m = re.search(r"_S(\d{8}T\d{6})/MTD_DS\.xml$", n)
        return O.jd_from_dt(dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc)) if m else jd
    return min(names, key=lambda n: abs(start(n) - jd))


def gps_points(item_id, jd):
    """(jd UTC, r ECEF км, v ECEF км/с) из MTD_DS.xml; кэш npz."""
    CACHE.mkdir(parents=True, exist_ok=True)
    name = mtd_ds_name(item_id, jd)
    if name is None:
        return None
    key = re.sub(r"[^A-Za-z0-9_]", "_", name.split("/")[-2])
    f = CACHE / f"{key}.npz"
    if not f.exists():
        url = f"https://storage.googleapis.com/{BUCKET}/" + urllib.parse.quote(name)
        with urllib.request.urlopen(url, timeout=120) as h:
            x = h.read().decode("utf-8", "replace")
        pts = re.findall(r'<GPS_Point>\s*<POSITION_VALUES unit="mm">([^<]+)</POSITION_VALUES>.*?'
                         r'<VELOCITY_VALUES unit="mm/s">([^<]+)</VELOCITY_VALUES>.*?<GPS_TIME>([^<]+)</GPS_TIME>', x, re.S)
        r = np.array([[float(v) for v in p.split()] for p, _, _ in pts]) / 1e6
        v = np.array([[float(q) for q in w.split()] for _, w, _ in pts]) / 1e6
        t = np.array([O.jd_from_dt(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)) for _, _, s in pts]) - GPS_UTC / 86400
        o = np.argsort(t)
        np.savez(f, t=t[o], r=r[o], v=v[o], name=name)
    z = np.load(f)
    return z["t"], z["r"], z["v"]


class PreciseOrbit:
    """Орбита по точкам GPS: кубический эрмитов сплайн по положению и скорости (шаг 1 с — ошибка интерполяции ≪ 1 м).
    Вне интервала точек — None (такую орбиту не используют)."""

    def __init__(self, name, t, r, v):
        self.name, self.t, self.r, self.v = name, np.asarray(t), np.asarray(r), np.asarray(v)
        self.epoch_jd = float(self.t[len(self.t) // 2]); self.source = "GPS (MTD_DS)"

    def covers(self, jd, margin_s=150.0):
        return self.t[0] - margin_s / 86400 <= jd <= self.t[-1] + margin_s / 86400

    def ecef(self, jd):
        jd = np.atleast_1d(np.asarray(jd, float))
        i = np.clip(np.searchsorted(self.t, jd) - 1, 0, len(self.t) - 2)
        t0, t1 = self.t[i], self.t[i + 1]; h = (t1 - t0) * 86400.0
        s = ((jd - t0) * 86400.0 / h)[:, None]
        p0, p1, m0, m1 = self.r[i], self.r[i + 1], self.v[i] * h[:, None], self.v[i + 1] * h[:, None]
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
        r = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        d00 = (6 * s ** 2 - 6 * s); d10 = (3 * s ** 2 - 4 * s + 1); d01 = (-6 * s ** 2 + 6 * s); d11 = (3 * s ** 2 - 2 * s)
        v = (d00 * p0 + d10 * m0 + d01 * p1 + d11 * m1) / h[:, None]
        return r, v


def precise_for(item_id, sent_name, jd):
    """Точная орбита Sentinel для снимка или None (нет метаданных в архиве)."""
    try:
        g = gps_points(item_id, jd)
    except Exception:                                   # noqa: BLE001 — нет сети/файла: остаётся TLE
        return None
    if g is None or len(g[0]) < 4:
        return None
    po = PreciseOrbit(sent_name, *g)
    return po if po.covers(jd) else None

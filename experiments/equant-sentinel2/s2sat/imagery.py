"""Снимки Sentinel-2: STAC Earth Search + чтение окон из COG через rasterio, кэш на диске."""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLL = "sentinel-2-c1-l2a"
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
if Path("/root/.ccr/ca-bundle.crt").exists():
    os.environ.setdefault("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")

_client = None


def client():
    global _client
    if _client is None:
        _client = Client.open(STAC_URL)
    return _client


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def search(**kw):
    for k in range(4):
        try:
            return list(client().search(collections=[COLL], **kw).items())
        except Exception:
            if k == 3:
                raise
            time.sleep(2 * (k + 1))


def epsg_of(it) -> int:
    c = it.properties.get("proj:code") or it.properties.get("proj:epsg") or it.assets["red"].extra_fields.get("proj:code")
    return int(str(c).split(":")[-1])


def relative_orbit(it) -> int | None:
    m = re.search(r"_R(\d{3})_", it.properties.get("s2:product_uri", ""))
    return int(m.group(1)) if m else it.properties.get("sat:relative_orbit")


def items_around(lon, lat, t: dt.datetime, sat_letter: str, d=0.6, minutes=4):
    """Все гранулы того же пролёта вокруг точки (трек может уйти за край тайла)."""
    poly = {"type": "Polygon", "coordinates": [[[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d],
                                                 [lon - d, lat + d], [lon - d, lat - d]]]}
    its = search(intersects=poly, datetime=f"{_iso(t - dt.timedelta(minutes=minutes))}/{_iso(t + dt.timedelta(minutes=minutes))}", limit=50)
    return [i for i in its if i.id.startswith("S2" + sat_letter)]


def ref_same_orbit(it, days=75, max_cloud=40, good_cloud=8):
    """Опорный снимок: тот же тайл и та же относительная орбита, свежий и малооблачный."""
    t = it.datetime
    feats = search(datetime=f"{_iso(t - dt.timedelta(days=days))}/{_iso(t - dt.timedelta(hours=12))}",
                   query={"grid:code": {"eq": it.properties["grid:code"]}, "eo:cloud_cover": {"lt": max_cloud}}, limit=100)
    ro = relative_orbit(it)
    feats = [f for f in feats if f.id[:3] in ("S2A", "S2B", "S2C") and (ro is None or relative_orbit(f) == ro)]
    if not feats:
        return None
    good = [f for f in feats if f.properties.get("eo:cloud_cover", 100) < good_cloud]
    if good:
        return max(good, key=lambda f: f.datetime)
    return min(feats, key=lambda f: f.properties.get("eo:cloud_cover", 100))


def ref_candidates(it, n=3, days=90, max_cloud=20):
    """До n опорных снимков той же орбиты и тайла, малооблачные, самые свежие (для медианы)."""
    t = it.datetime
    feats = search(datetime=f"{_iso(t - dt.timedelta(days=days))}/{_iso(t - dt.timedelta(hours=12))}",
                   query={"grid:code": {"eq": it.properties["grid:code"]}, "eo:cloud_cover": {"lt": max_cloud}}, limit=100)
    ro = relative_orbit(it)
    same = [f for f in feats if f.id[:3] in ("S2A", "S2B", "S2C") and (ro is None or relative_orbit(f) == ro)]
    pool = same if len(same) >= n else feats
    pool = sorted(pool, key=lambda f: (f.properties.get("eo:cloud_cover", 100) > 5, -f.datetime.timestamp()))
    return pool[:n]


def get_text(url: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / ("txt_" + hashlib.sha1(url.encode()).hexdigest()[:16])
    if p.exists():
        return p.read_text()
    for k in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                s = r.read().decode()
            p.write_text(s)
            return s
        except OSError:
            if k == 3:
                raise
            time.sleep(2 * (k + 1))


def read_window(href: str, epsg: int, bounds, res=20.0, resampling="average"):
    """Окно (minx, miny, maxx, maxy) в EPSG на сетке res м. 0 = нет данных. Кэш .npy.

    10-метровые полосы усредняются до 20 м (аналог block2 в пилоте), другая зона UTM
    перепроецируется (WarpedVRT).
    """
    minx, miny, maxx, maxy = bounds
    W = int(round((maxx - minx) / res)); H = int(round((maxy - miny) / res))
    key = hashlib.sha1(f"{href}|{epsg}|{bounds}|{res}".encode() + (b"" if resampling == "average" else resampling.encode())).hexdigest()[:20]
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"win_{key}.npy"
    if p.exists():
        return np.load(p)
    tr = from_origin(minx, maxy, res, res)
    last = None
    for k in range(4):
        try:
            with rasterio.open(href) as ds:
                with WarpedVRT(ds, crs=f"EPSG:{epsg}", transform=tr, width=W, height=H,
                               resampling=getattr(Resampling, resampling), nodata=0, src_nodata=0) as vrt:
                    a = vrt.read(1).astype(np.float32)
            np.save(p, a)
            return a
        except Exception as ex:  # сеть
            last = ex
            time.sleep(2 * (k + 1))
    raise last


def prune_cache(max_gb=40.0, target_gb=30.0):
    """Ограничить кэш: при превышении max_gb удалить самые старые файлы до target_gb — сначала окна снимков
    (их можно перечитать), обработанные ячейки — в последнюю очередь (их пересчёт дорог)."""
    win = sorted(CACHE.glob("win_*.npy"), key=lambda p: p.stat().st_mtime)
    til = sorted(CACHE.glob("tiles/*/*.pkl"), key=lambda p: p.stat().st_mtime)
    files = win + til
    total = sum(p.stat().st_size for p in files)
    if total <= max_gb * 1e9:
        return 0
    removed = 0
    for p in files:
        if total <= target_gb * 1e9:
            break
        sz = p.stat().st_size
        try:
            p.unlink(); total -= sz; removed += 1
        except OSError:
            pass
    return removed

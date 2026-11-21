import datetime as dt
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore", category=RuntimeWarning)

from s2sat import orbits as O, geometry as G, crossings as C, detect as D, s2meta as M  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pilot_tle():
    return {o.catnr: o for o in O.parse_tle_tokens((ROOT / "pilot_code/tle_raw.txt").read_text())}


def test_tle_tokens(pilot_tle):
    s2b = pilot_tle["42063"]
    assert s2b.line1 == "1 42063U 17013A   26321.98374977 -.00000167  00000+0 -47230-4 0  9990"
    assert len(pilot_tle) > 100


def test_ut1_offset_small():
    # регрессия бага: перевод JD→skyfield съезжал на 27 с (високосные секунды)
    jd = O.jd_from_dt(dt.datetime(2026, 11, 18, 22, 28, tzinfo=dt.timezone.utc))
    dut1, dtt = O._offsets(jd)
    assert abs(dut1 * 86400) < 1.0
    assert abs(dtt * 86400 - 69.184) < 0.01


def test_iss_crossing_and_track(pilot_tle):
    """Пересечение и видимый трек МКС совпадают с пилотом (HANDOFF 5.1)."""
    s2b, iss = pilot_tle["42063"], pilot_tle["25544"]
    j = O.jd_from_dt(dt.datetime(2026, 11, 18, 22, 28, tzinfo=dt.timezone.utc))
    ev = C.find_crossings([s2b], [iss], j - 0.02, j + 0.02, step_s=10)
    assert len(ev) == 1
    e = ev[0]
    assert abs((O.dt_from_jd(e["jd"]) - dt.datetime(2026, 11, 18, 22, 28, 8, tzinfo=dt.timezone.utc)).total_seconds()) < 1
    assert abs(e["rng"] - 374) < 2 and e["desc"] and e["lit"]
    tr = G.apparent_track(s2b, iss, e["jd"], [0.612, 1.096])
    X, Y = G.ecef_to_utm(tr["ground"], 32759)
    # пилот (mega4.apparent, GMST без UT1): B02 (550753, 5245408), B04 (546846, 5241155)
    assert np.hypot(X[0] - 550753, Y[0] - 5245408) < 100
    assert np.hypot(X[1] - 546846, Y[1] - 5241155) < 100
    assert 5.5 < np.hypot(X[0] - X[1], Y[0] - Y[1]) / 1000 < 6.1  # B02–B04 около 5.8 км


def test_band_roots_synthetic():
    ulx, uly = 500000.0, 8200000.0
    gx = np.arange(23) * 5000
    G0 = 0.3 + 0.00001 * gx[None, :] * np.ones((23, 1)); G0[:, 12:] = np.nan
    DG = {"blue": [dict(det=7, g=G0, ulx=ulx, uly=uly)], "red": [dict(det=7, g=G0 + 0.2, ulx=ulx, uly=uly)]}
    al = np.linspace(-2.2, 2.2, 1500)
    X = ulx + 20000 + al * 30000; Y = uly - 50000 + al * 5000
    R = M.band_roots(al, X, Y, DG, ["blue", "red"])
    for b, off in (("blue", 0.3), ("red", 0.5)):
        (_, a, x, _y), = R[b]
        assert abs(a - (off + 0.00001 * (x - ulx))) < 1e-3


def test_shift_search_synthetic():
    """Перенос tests/test19.py: вставленный объект восстанавливается, нуль заметно ниже."""
    rng = np.random.default_rng(1)
    bands = list(M.BRES)
    bounds = (500000.0, 8140000.0, 530000.0, 8170000.0)
    n = int((bounds[2] - bounds[0]) / 20)
    E = {}
    for b in bands:
        z = np.abs(rng.standard_normal((n, n))); F = D.maximum_filter(np.abs(rng.standard_normal((n, n))), 7)
        E[b] = D.maximum_filter(np.maximum(0, z - F), 3).astype(np.float32)
    roots = {b: [(1, 0.0, 512000.0 + 1500 * k, 8150000.0 + 300 * k)] for k, b in enumerate(bands)}
    true = (1340.0, -760.0)
    for b in bands:
        x, y = roots[b][0][2] + true[0], roots[b][0][3] + true[1]
        c = int((x - bounds[0]) // 20); r = int((bounds[3] - y) // 20)
        E[b][r - 1:r + 2, c - 1:c + 2] = 6.0
    mos = D.Raster(E, bounds)
    res = D.shift_search(roots, mos, bands, 4000.0)
    nulls = D.null_test(roots, mos, bands, 4000.0, n=5)
    assert abs(res["sx"] - true[0]) <= 40 and abs(res["sy"] - true[1]) <= 40
    assert res["score"] > 2 * max(nulls)


@pytest.mark.network
def test_iss_regression(pilot_tle):
    """HANDOFF 9.2: счёт > 15, нуль < 6, сдвиг (620, 1000) ± 200 м с TLE пилота (эпоха 17.11 20:24)."""
    from s2sat import pipeline as P
    s2b, iss = pilot_tle["42063"], pilot_tle["25544"]
    j = O.jd_from_dt(dt.datetime(2026, 11, 18, 22, 28, tzinfo=dt.timezone.utc))
    e = C.find_crossings([s2b], [iss], j - 0.02, j + 0.02, step_s=10)[0]
    r = P.run_event(s2b, iss, e, Rm=6000.0, minb=4)
    assert r["score"] >= r["null_max"] + 3
    assert math.hypot(r["sx"] - 620, r["sy"] - 1000) < 200

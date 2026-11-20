"""Орбиты: загрузка TLE с кэшем, SGP4 (пакет sgp4), перевод в земную систему (ITRS/ECEF).

Все вычисления геометрии в проекте ведутся в ECEF (км, км/с). Скорость в ECEF уже
относительно вращающейся Земли, то есть это и есть v_E = v - ω×r из HANDOFF (4.1).
"""
from __future__ import annotations

import datetime as dt
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sgp4.api import Satrec
from skyfield.api import load as _sf_load
from skyfield.sgp4lib import TEME_to_ITRF

CACHE = Path(__file__).resolve().parent.parent / "data" / "tle"
CELESTRAK = "https://celestrak.org/NORAD/elements/"
SUPGP_STARLINK = CELESTRAK + "supplemental/sup-gp.php?FILE=starlink&FORMAT=tle"

_TS = _sf_load.timescale(builtin=True)


def fetch_text(url: str, name: str, max_age_h: float = 6.0, offline: bool = False) -> str:
    """Скачать текст с кэшем в data/tle/<name>. Кэш свежее max_age_h часов берётся без сети."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.exists():
        age_h = (dt.datetime.now().timestamp() - path.stat().st_mtime) / 3600
        if offline or age_h < max_age_h:
            return path.read_text()
    if offline:
        raise FileNotFoundError(path)
    text, last = "", None
    for k in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode()
            break
        except OSError as ex:
            last = ex
            time.sleep(2 * (k + 1))
    else:
        if path.exists():
            return path.read_text()
        raise last
    if text.strip():
        path.write_text(text)
    return text


@dataclass
class Orbit:
    name: str
    satrec: Satrec
    line1: str
    line2: str

    @property
    def catnr(self) -> str:
        return self.line1[2:7].strip()

    @property
    def epoch_jd(self) -> float:
        return self.satrec.jdsatepoch + self.satrec.jdsatepochF

    @property
    def perigee_km(self) -> float:
        return self.satrec.altp * self.satrec.radiusearthkm

    def teme(self, jd):
        """TEME, км и км/с. jd: скаляр или массив (UTC JD)."""
        jd = np.atleast_1d(np.asarray(jd, dtype=float))
        whole = np.floor(jd - 0.5) + 0.5
        e, r, v = self.satrec.sgp4_array(whole, jd - whole)
        if np.any(e != 0):
            r = r.copy(); r[e != 0] = np.nan
        return r, v

    def ecef(self, jd):
        """ITRS (без полярного движения), км и км/с; форма (n, 3)."""
        jd = np.atleast_1d(np.asarray(jd, dtype=float))
        r, v = self.teme(jd)
        return teme_to_ecef(jd, r, v)


def teme_to_ecef(jd_utc, r, v):
    jd_utc = np.atleast_1d(np.asarray(jd_utc, dtype=float))
    # UT1 = UTC + DUT1; DUT1 из встроенной таблицы skyfield
    ut1 = jd_utc + _offsets(jd_utc[0])[0]
    whole = np.floor(ut1)
    rr, vv = TEME_to_ITRF(whole, r.T, v.T * 86400.0, fraction_ut1=ut1 - whole)
    return rr.T, vv.T / 86400.0


def _offsets(jd_utc: float):
    """(UT1−UTC, TT−UTC) в сутках на момент jd_utc (постоянны в пределах одного вызова)."""
    t = _TS.from_datetime(dt_from_jd(jd_utc))
    return t.ut1 - jd_utc, t.tt - jd_utc


def parse_tle(text: str) -> list[Orbit]:
    lines = [l.rstrip() for l in text.replace("\r", "").split("\n") if l.strip()]
    out, i = [], 0
    while i < len(lines) - 2:
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            try:
                out.append(Orbit(lines[i].strip(), Satrec.twoline2rv(lines[i + 1], lines[i + 2]), lines[i + 1], lines[i + 2]))
            except Exception:
                pass
            i += 3
        else:
            i += 1
    return out


def _cks(line: str) -> str:
    s = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return line[:68] + str(s % 10)


def parse_tle_tokens(text: str) -> list[Orbit]:
    """TLE со схлопнутыми пробелами (pilot_code/tle_raw.txt): восстановить колонки."""
    tok = text.split()
    out, i = [], 0
    while i < len(tok):
        if tok[i] == "1" and i + 9 < len(tok) and tok[i + 1][:5].isdigit() and tok[i + 1][-1].isalpha():
            # имя — токены после предыдущего TLE
            j = i - 1
            while j >= 0 and not (tok[j] == "2" or tok[j].replace(".", "").isdigit() and len(tok[j]) >= 12):
                j -= 1
            name = " ".join(tok[j + 1:i]) if j >= 0 else " ".join(tok[:i])
            a = tok[i:i + 9]
            l1 = (f"1 {a[1][:5]}{a[1][5]} {a[2]:<8} {a[3]:>14} {a[4]:>10} {a[5]:>8} {a[6]:>8} {a[7]} {a[8][:-1]:>4}0")
            k = i + 9
            if k + 7 < len(tok) and tok[k] == "2":
                b = tok[k:k + 8]
                l2 = (f"2 {b[1]:<5} {b[2]:>8} {b[3]:>8} {b[4]:>7} {b[5]:>8} {b[6]:>8} {b[7][:11]:>11}{b[7][11:-1]:>5}0")
                l1, l2 = _cks(l1), _cks(l2)
                try:
                    out.append(Orbit(name, Satrec.twoline2rv(l1, l2), l1, l2))
                except Exception:
                    pass
                i = k + 8
                continue
        i += 1
    return out


def celestrak(query: str, name: str, **kw) -> list[Orbit]:
    """query, например 'CATNR=25544' или 'GROUP=active' или 'NAME=SENTINEL-2'."""
    return parse_tle(fetch_text(CELESTRAK + "gp.php?" + query + "&FORMAT=tle", name, **kw))


def jd_from_dt(d: dt.datetime) -> float:
    return d.timestamp() / 86400.0 + 2440587.5


def dt_from_jd(jd: float) -> dt.datetime:
    return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=float(jd) - 2440587.5)


def sun_ecef(jd):
    """Единичный вектор на Солнце в ITRS."""
    from skyfield.framelib import itrs
    eph = _ephem()
    jd = np.atleast_1d(np.asarray(jd, float))
    t = _TS.tt_jd(jd + _offsets(jd[0])[1])
    p = (eph["sun"] - eph["earth"]).at(t).frame_xyz(itrs).km
    p = np.asarray(p).T
    return p / np.linalg.norm(p, axis=-1, keepdims=True)


_EPH = None


def _ephem():
    global _EPH
    if _EPH is None:
        from skyfield.api import Loader
        from skyfield_data import get_skyfield_data_path
        _EPH = Loader(get_skyfield_data_path())("de421.bsp")
    return _EPH

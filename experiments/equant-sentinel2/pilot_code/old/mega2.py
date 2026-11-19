import math, datetime as dt
import numpy as np
MU = 398600.8; RE = 6378.135; XKE = 60.0 / math.sqrt(RE**3 / MU)
J2 = 0.001082616; J3 = -0.00000253881; J4 = -0.00000165597; J3OJ2 = J3 / J2
X2O3 = 2.0 / 3.0; TWOPI = 2 * math.pi; VKMS = RE * XKE / 60.0
def _expo(s):
    s = s.strip()
    if not s: return 0.0
    sign = -1.0 if s[0] == '-' else 1.0
    s = s.lstrip('+-')
    return sign * float('0.' + s[:-2]) * 10 ** int(s[-2:])
def jday(year, doy):
    return 367 * year - int(7 * year / 4) + 30 + 1 + 1721013.5 + doy - 1.0
def gstime(jd):
    t = (jd - 2451545.0) / 36525.0
    g = -6.2e-6 * t**3 + 0.093104 * t**2 + (876600.0 * 3600 + 8640184.812866) * t + 67310.54841
    return np.radians(g / 240.0) % TWOPI
class Sat:
    def __init__(self, name, l1, l2):
        self.name = name.strip(); self.cat = l1[2:7].strip()
        yy = int(l1[18:20]); year = 2000 + yy if yy < 57 else 1900 + yy
        self.jdepoch = jday(year, float(l1[20:32]))
        self.bstar = _expo(l1[53:61])
        self.inclo = math.radians(float(l2[8:16])); self.nodeo = math.radians(float(l2[17:25]))
        self.ecco = float('0.' + l2[26:33].strip()); self.argpo = math.radians(float(l2[34:42]))
        self.mo = math.radians(float(l2[43:51])); self.no_kozai = float(l2[52:63]) * TWOPI / 1440.0
        self._init()
    def _init(self):
        e = self.ecco; inc = self.inclo
        omeosq = 1 - e * e; rteosq = math.sqrt(omeosq)
        cosio = math.cos(inc); cosio2 = cosio * cosio
        ak = (XKE / self.no_kozai) ** X2O3
        d1 = 0.75 * J2 * (3 * cosio2 - 1) / (rteosq * omeosq)
        dl = d1 / (ak * ak)
        adel = ak * (1 - dl * dl - dl * (1.0 / 3.0 + 134 * dl * dl / 81.0))
        dl = d1 / (adel * adel)
        self.no = self.no_kozai / (1 + dl)
        ao = (XKE / self.no) ** X2O3
        self.ao = ao
        sinio = math.sin(inc); po = ao * omeosq
        con42 = 1 - 5 * cosio2; self.con41 = -con42 - cosio2 - cosio2
        posq = po * po; rp = ao * (1 - e)
        self.perigee_km = (rp - 1) * RE; self.apogee_km = (ao * (1 + e) - 1) * RE
        if TWOPI / self.no >= 225: raise ValueError('deep space')
        ss = 78.0 / RE + 1; qzms2t = ((120.0 - 78.0) / RE) ** 4
        self.isimp = 1 if rp < (220.0 / RE + 1) else 0
        sfour = ss; qzms24 = qzms2t; perige = (rp - 1) * RE
        if perige < 156:
            sfour = perige - 78
            if perige < 98: sfour = 20
            qzms24 = ((120 - sfour) / RE) ** 4; sfour = sfour / RE + 1
        pinvsq = 1 / posq; tsi = 1 / (ao - sfour)
        self.eta = eta = ao * e * tsi
        etasq = eta * eta; eeta = e * eta; psisq = abs(1 - etasq)
        coef = qzms24 * tsi ** 4; coef1 = coef / psisq ** 3.5
        cc2 = coef1 * self.no * (ao * (1 + 1.5 * etasq + eeta * (4 + etasq)) +
                                 0.375 * J2 * tsi / psisq * self.con41 * (8 + 3 * etasq * (8 + etasq)))
        self.cc1 = cc1 = self.bstar * cc2
        cc3 = -2 * coef * tsi * J3OJ2 * self.no * sinio / e if e > 1.0e-4 else 0.0
        self.x1mth2 = 1 - cosio2
        self.cc4 = 2 * self.no * coef1 * ao * omeosq * (
            eta * (2 + 0.5 * etasq) + e * (0.5 + 2 * etasq) -
            J2 * tsi / (ao * psisq) * (-3 * self.con41 * (1 - 2 * eeta + etasq * (1.5 - 0.5 * eeta)) +
                                       0.75 * self.x1mth2 * (2 * etasq - eeta * (1 + etasq)) * math.cos(2 * self.argpo)))
        self.cc5 = 2 * coef1 * ao * omeosq * (1 + 2.75 * (etasq + eeta) + eeta * etasq)
        cosio4 = cosio2 * cosio2
        temp1 = 1.5 * J2 * pinvsq * self.no; temp2 = 0.5 * temp1 * J2 * pinvsq
        temp3 = -0.46875 * J4 * pinvsq * pinvsq * self.no
        self.mdot = self.no + 0.5 * temp1 * rteosq * self.con41 + 0.0625 * temp2 * rteosq * (13 - 78 * cosio2 + 137 * cosio4)
        self.argpdot = -0.5 * temp1 * con42 + 0.0625 * temp2 * (7 - 114 * cosio2 + 395 * cosio4) + temp3 * (3 - 36 * cosio2 + 49 * cosio4)
        xhdot1 = -temp1 * cosio
        self.nodedot = xhdot1 + (0.5 * temp2 * (4 - 19 * cosio2) + 2 * temp3 * (3 - 7 * cosio2)) * cosio
        self.omgcof = self.bstar * cc3 * math.cos(self.argpo)
        self.xmcof = -X2O3 * coef * self.bstar / eeta if e > 1.0e-4 else 0.0
        self.nodecf = 3.5 * omeosq * xhdot1 * cc1; self.t2cof = 1.5 * cc1
        den = (1 + cosio) if abs(cosio + 1.0) > 1.5e-12 else 1.5e-12
        self.xlcof = -0.25 * J3OJ2 * sinio * (3 + 5 * cosio) / den
        self.aycof = -0.5 * J3OJ2 * sinio
        self.delmo = (1 + eta * math.cos(self.mo)) ** 3; self.sinmao = math.sin(self.mo)
        self.x7thm1 = 7 * cosio2 - 1
        if self.isimp != 1:
            cc1sq = cc1 * cc1
            self.d2 = d2 = 4 * ao * tsi * cc1sq
            temp = d2 * tsi * cc1 / 3.0
            self.d3 = d3 = (17 * ao + sfour) * temp
            self.d4 = d4 = 0.5 * temp * ao * tsi * (221 * ao + 31 * sfour) * cc1
            self.t3cof = d2 + 2 * cc1sq
            self.t4cof = 0.25 * (3 * d3 + cc1 * (12 * d2 + 10 * cc1sq))
            self.t5cof = 0.2 * (3 * d4 + 12 * cc1 * d3 + 6 * d2 * d2 + 15 * cc1sq * (2 * d2 + cc1sq))
    def prop(self, tsince):
        t = np.atleast_1d(np.asarray(tsince, dtype=float))
        xmdf = self.mo + self.mdot * t; argpdf = self.argpo + self.argpdot * t; nodedf = self.nodeo + self.nodedot * t
        argpm = argpdf.copy(); mm = xmdf.copy(); t2 = t * t
        nodem = nodedf + self.nodecf * t2
        tempa = 1 - self.cc1 * t; tempe = self.bstar * self.cc4 * t; templ = self.t2cof * t2
        if self.isimp != 1:
            delomg = self.omgcof * t
            delm = self.xmcof * ((1 + self.eta * np.cos(xmdf)) ** 3 - self.delmo)
            temp = delomg + delm; mm = xmdf + temp; argpm = argpdf - temp
            t3 = t2 * t; t4 = t3 * t
            tempa = tempa - self.d2 * t2 - self.d3 * t3 - self.d4 * t4
            tempe = tempe + self.bstar * self.cc5 * (np.sin(mm) - self.sinmao)
            templ = templ + self.t3cof * t3 + t4 * (self.t4cof + t * self.t5cof)
        am = (XKE / self.no) ** X2O3 * tempa * tempa
        nm = XKE / am ** 1.5
        em = np.maximum(self.ecco - tempe, 1e-6)
        mm = mm + self.no * templ
        xlm = mm + argpm + nodem
        nodem = np.fmod(nodem, TWOPI); argpm = np.fmod(argpm, TWOPI); xlm = np.fmod(xlm, TWOPI)
        mm = np.fmod(xlm - argpm - nodem, TWOPI)
        sinip = math.sin(self.inclo); cosip = math.cos(self.inclo)
        axnl = em * np.cos(argpm)
        temp = 1.0 / (am * (1 - em * em))
        aynl = em * np.sin(argpm) + temp * self.aycof
        xl = mm + argpm + nodem + temp * self.xlcof * axnl
        u = np.fmod(xl - nodem, TWOPI); eo1 = u.copy()
        for _ in range(10):
            s1 = np.sin(eo1); c1 = np.cos(eo1)
            tem5 = (u - aynl * c1 + axnl * s1 - eo1) / (1 - c1 * axnl - s1 * aynl)
            tem5 = np.clip(tem5, -0.95, 0.95); eo1 = eo1 + tem5
            if np.all(np.abs(tem5) < 1e-12): break
        sineo1 = np.sin(eo1); coseo1 = np.cos(eo1)
        ecose = axnl * coseo1 + aynl * sineo1; esine = axnl * sineo1 - aynl * coseo1
        el2 = axnl * axnl + aynl * aynl; pl = am * (1 - el2); rl = am * (1 - ecose)
        rdotl = np.sqrt(am) * esine / rl; rvdotl = np.sqrt(pl) / rl; betal = np.sqrt(1 - el2)
        temp = esine / (1 + betal)
        sinu = am / rl * (sineo1 - aynl - axnl * temp); cosu = am / rl * (coseo1 - axnl + aynl * temp)
        su = np.arctan2(sinu, cosu); sin2u = (cosu + cosu) * sinu; cos2u = 1 - 2 * sinu * sinu
        temp = 1.0 / pl; temp1 = 0.5 * J2 * temp; temp2 = temp1 * temp
        mrt = rl * (1 - 1.5 * temp2 * betal * self.con41) + 0.5 * temp1 * self.x1mth2 * cos2u
        su = su - 0.25 * temp2 * self.x7thm1 * sin2u
        xnode = nodem + 1.5 * temp2 * cosip * sin2u
        xinc = self.inclo + 1.5 * temp2 * cosip * sinip * cos2u
        mvt = rdotl - nm * temp1 * self.x1mth2 * sin2u / XKE
        rvdot = rvdotl + nm * temp1 * (self.x1mth2 * cos2u + 1.5 * self.con41) / XKE
        sinsu = np.sin(su); cossu = np.cos(su); snod = np.sin(xnode); cnod = np.cos(xnode)
        sini = np.sin(xinc); cosi = np.cos(xinc); xmx = -snod * cosi; xmy = cnod * cosi
        U = np.stack([xmx * sinsu + cnod * cossu, xmy * sinsu + snod * cossu, sini * sinsu], 1)
        V = np.stack([xmx * cossu - cnod * sinsu, xmy * cossu - snod * sinsu, sini * cossu], 1)
        return (mrt * RE)[:, None] * U, VKMS * (mvt[:, None] * U + rvdot[:, None] * V)
    def prop_jd(self, jd):
        return self.prop((np.asarray(jd, dtype=float) - self.jdepoch) * 1440.0)
def parse_tle_text(text):
    lines = [l.rstrip() for l in text.replace('\r', '').split('\n') if l.strip()]
    out = []
    i = 0
    while i < len(lines) - 2:
        if lines[i + 1].startswith('1 ') and lines[i + 2].startswith('2 '):
            try: out.append(Sat(lines[i], lines[i + 1], lines[i + 2]))
            except Exception: pass
            i += 3
        else:
            i += 1
    return out
A84 = 6378.137; B84 = 6356.752314245; E2 = 1 - (B84 / A84) ** 2
def jd_from_dt(d): return d.timestamp() / 86400.0 + 2440587.5
def dt_from_jd(jd): return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=jd - 2440587.5)
def sun_unit(jd):
    n = jd - 2451545.0
    L = math.radians((280.460 + 0.9856474 * n) % 360); g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 0.0000004 * n)
    return np.array([math.cos(lam), math.cos(eps) * math.sin(lam), math.sin(eps) * math.sin(lam)])
def ray_ellipsoid(p, d):
    a2, b2 = A84 ** 2, B84 ** 2
    A = (d[0] ** 2 + d[1] ** 2) / a2 + d[2] ** 2 / b2
    B = 2 * ((p[0] * d[0] + p[1] * d[1]) / a2 + p[2] * d[2] / b2)
    C = (p[0] ** 2 + p[1] ** 2) / a2 + p[2] ** 2 / b2 - 1
    disc = B * B - 4 * A * C
    return None if disc < 0 else (-B - math.sqrt(disc)) / (2 * A)
def teme_to_geodetic(r, jd):
    g = float(gstime(jd))
    x = math.cos(g) * r[0] + math.sin(g) * r[1]; y = -math.sin(g) * r[0] + math.cos(g) * r[1]; z = r[2]
    p = math.hypot(x, y); lat = math.atan2(z, p * (1 - E2))
    for _ in range(6):
        N = A84 / math.sqrt(1 - E2 * math.sin(lat) ** 2); h = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1 - E2 * N / (N + h)))
    return math.degrees(lat), (math.degrees(math.atan2(y, x)) + 180) % 360 - 180
def fan_normal(rs, vs):
    rh = rs / np.linalg.norm(rs, axis=-1, keepdims=True)
    vh = vs - np.sum(vs * rh, axis=-1, keepdims=True) * rh
    return vh / np.linalg.norm(vh, axis=-1, keepdims=True)
def band_plane(rS, vS, alpha):
    rh = rS / np.linalg.norm(rS); n = fan_normal(rS, vS); ct = np.cross(rh, n)
    look = -rh * math.cos(alpha) + n * math.sin(alpha)
    return np.cross(look, ct)
def latlon_to_utm(lat, lon, zone):
    a = 6378137.0; f = 1 / 298.257223563; k0 = 0.9996
    lon0 = math.radians((zone - 1) * 6 - 180 + 3); e2 = f * (2 - f); ep2 = e2 / (1 - e2)
    phi = math.radians(lat); lam = math.radians(lon)
    N = a / math.sqrt(1 - e2 * math.sin(phi) ** 2); T = math.tan(phi) ** 2; C = ep2 * math.cos(phi) ** 2; A = math.cos(phi) * (lam - lon0)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * phi) - (35 * e2 ** 3 / 3072) * math.sin(6 * phi))
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000
    y = k0 * (M + N * math.tan(phi) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24 + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, (y + 10000000 if lat < 0 else y)
T_LINE = 10.0 / 6.63  # ms per 10 m line
def find_crossings(sents, objs, jd0, jd1, step_s=30.0, half_fov_deg=10.5):
    grid = np.arange(jd0, jd1, step_s / 86400.0)
    S = []
    for s in sents:
        rs, vs = s.prop_jd(grid); S.append((s, rs, fan_normal(rs, vs)))
    ev = []
    hf = math.radians(half_fov_deg)
    for ob in objs:
        try: ro, _ = ob.prop_jd(grid)
        except Exception: continue
        if not np.all(np.isfinite(ro)): continue
        for s, rs, nn in S:
            d = ro - rs
            f = np.sum(d * nn, axis=1)
            dist = np.linalg.norm(d, axis=1)
            idx = np.where((np.sign(f[:-1]) != np.sign(f[1:])) & (dist[:-1] < 1500))[0]
            for i in idx:
                w = f[i] / (f[i] - f[i + 1])
                dl = d[i] + w * (d[i + 1] - d[i]); rl = rs[i] + w * (rs[i + 1] - rs[i])
                if math.acos(np.clip(np.dot(dl / np.linalg.norm(dl), -rl / np.linalg.norm(rl)), -1, 1)) > hf + math.radians(3):
                    continue
                ta, tb = grid[i], grid[i + 1]
                def F(t):
                    a, b = s.prop_jd([t]); c, _v = ob.prop_jd([t])
                    return float(np.dot(c[0] - a[0], fan_normal(a[0], b[0])))
                fa, fb = f[i], f[i + 1]
                for _ in range(8):
                    tm = ta - fa * (tb - ta) / (fb - fa); fm = F(tm)
                    if abs(fm) < 1e-5: break
                    if np.sign(fm) == np.sign(fa): ta, fa = tm, fm
                    else: tb, fb = tm, fm
                rS, vS = s.prop_jd([tm]); rO, vO = ob.prop_jd([tm]); rS, vS, rO, vO = rS[0], vS[0], rO[0], vO[0]
                dv = rO - rS; rng = np.linalg.norm(dv); nadir = -rS / np.linalg.norm(rS)
                if math.acos(np.clip(np.dot(dv / rng, nadir), -1, 1)) > hf: continue
                lam = ray_ellipsoid(rS, dv / rng)
                if lam is None or lam < rng: continue
                gp = rS + lam * dv / rng
                lat, lon = teme_to_geodetic(gp, tm)
                su = sun_unit(tm)
                sun_el = math.degrees(math.asin(np.dot(su, gp / np.linalg.norm(gp))))
                proj = np.dot(rO, su); lit = proj > 0 or np.linalg.norm(rO - proj * su) > RE
                n = fan_normal(rS, vS); vrel = vO - vS
                v_al = float(np.dot(vrel, n))
                pix_obj = 10.0 * rng / lam  # m per pixel at the object
                ev.append(dict(sent=s.name, obj=ob.name, cat=ob.cat, jd=tm, lat=lat, lon=lon, rng=rng, lam=lam,
                               alt=np.linalg.norm(rO) - RE, sun_el=sun_el, lit=bool(lit), desc=bool(vS[2] < 0),
                               v_al=v_al, pix_obj=pix_obj, age=tm - ob.jdepoch))
    ev.sort(key=lambda e: e['jd'])
    return ev
def apparent(sen, ob, jd0, alpha_deg, dM=0.0):
    a = math.radians(alpha_deg)
    def F(t):
        rS, vS = sen.prop_jd([t]); rO, vO = ob.prop_jd([t]); rS, vS, rO, vO = rS[0], vS[0], rO[0], vO[0]
        rO = rO + dM * vO / np.linalg.norm(vO)
        return float(np.dot(rO - rS, band_plane(rS, vS, a))), rS, rO
    ta, tb = jd0 - 4 / 86400, jd0 + 4 / 86400
    fa = F(ta)[0]; fb = F(tb)[0]
    k = 4
    while np.sign(fa) == np.sign(fb) and k < 120:
        k *= 2; ta, tb = jd0 - k / 86400, jd0 + k / 86400; fa = F(ta)[0]; fb = F(tb)[0]
    for _ in range(40):
        tm = 0.5 * (ta + tb); fm, rS, rO = F(tm)
        if np.sign(fm) == np.sign(fa): ta, fa = tm, fm
        else: tb = tm
    d = (rO - rS) / np.linalg.norm(rO - rS)
    lam = ray_ellipsoid(rS, d)
    lat, lon = teme_to_geodetic(rS + lam * d, tm)
    return tm, lat, lon
import asyncio
import time, json, io, micropip
from pyodide.http import pyfetch
await micropip.install('scipy')
async def gettext(u):
    for t in range(3):
        try:
            r = await pyfetch(u); return await r.string()
        except Exception:
            await asyncio.sleep(2)
    raise RuntimeError('fetch failed ' + u)
CT = 'https://celestrak.org/NORAD/elements/'
t_sup = await gettext(CT + 'supplemental/sup-gp.php?FILE=starlink&FORMAT=tle')
t_act = await gettext(CT + 'gp.php?GROUP=active&FORMAT=tle')
t_s2 = await gettext(CT + 'gp.php?NAME=SENTINEL-2&FORMAT=tle')
sents = parse_tle_text(t_s2); star = parse_tle_text(t_sup); seen = set(s.cat for s in star); others = []
for s in parse_tle_text(t_act):
    if s.cat in seen or s.name.startswith('STARLINK') or s.name.startswith('SENTINEL-2'): continue
    seen.add(s.cat); others.append(s)
objs = [s for s in star + others if s.perigee_km < 800]
SENT = {s.name: s for s in sents}; OBJ = {o.cat: o for o in objs}
print('MEGA sentinels', [s.name for s in sents], 'supgp', len(star), 'others', len(others), 'objs', len(objs))
def epsg_of(it):
    p = it['properties']
    if 'proj:epsg' in p and p['proj:epsg']: return int(p['proj:epsg'])
    c = p.get('proj:code') or it['assets']['red'].get('proj:code')
    return int(str(c).split(':')[-1])
def locus(e, alphas, dM=0.0):
    zone = e['epsg'] % 100; pts = []
    for al in alphas:
        try: tm, la, lo = apparent(SENT[e['sent']], OBJ[e['cat']], e['jd'], float(al), dM / 1000.0)
        except Exception: continue
        x, y = latlon_to_utm(la, lo, zone); pts.append((al, tm, x, y))
    return np.array(pts)
def size_m(o):
    n = o['obj']; c = o['cat']
    if c == '25544': return 100.
    if c == '48274': return 50.
    if n.startswith('STARLINK'):
        try: k = int(n.split('-')[1].split()[0])
        except Exception: k = 0
        return 30. if k >= 30000 else 10.
    return 5.
import json, asyncio, time, io, gc, datetime as dtm, traceback
import matplotlib.pyplot as plt
from pyodide.http import pyfetch
from scipy.ndimage import median_filter, maximum_filter, minimum_filter
BID = {'blue': '1', 'green': '2', 'red': '3', 'rededge1': '4', 'rededge2': '5', 'rededge3': '6', 'nir': '7', 'nir08': '8', 'swir16': '11', 'swir22': '12'}
BRES = {'blue': 10, 'green': 10, 'red': 10, 'nir': 10, 'rededge1': 20, 'rededge2': 20, 'rededge3': 20, 'nir08': 20, 'swir16': 20, 'swir22': 20}
GRID, PAD, CLIP = 5000.0, 500.0, 5.0
async def stac_post(q):
    for t in range(3):
        try:
            r = await pyfetch('https://earth-search.aws.element84.com/v1/search', method='POST', headers={'Content-Type': 'application/json'}, body=json.dumps(q))
            return (await r.json()).get('features', [])
        except Exception:
            await asyncio.sleep(2)
    return []
def iso(t): return t.strftime('%Y-%m-%dT%H:%M:%SZ')
async def ref_same_orbit(e):
    pr = e['item']['properties']; t = dtm.datetime.fromisoformat(pr['datetime'].replace('Z', '+00:00'))
    qq = {'grid:code': {'eq': pr['grid:code']}, 'eo:cloud_cover': {'lt': 40}}
    if 'sat:relative_orbit' in pr: qq['sat:relative_orbit'] = {'eq': pr['sat:relative_orbit']}
    feats = await stac_post({'collections': ['sentinel-2-c1-l2a'], 'limit': 50, 'datetime': f"{iso(t - dtm.timedelta(days=75))}/{iso(t - dtm.timedelta(hours=12))}", 'query': qq})
    if not feats: return None
    good = [f for f in feats if f['properties'].get('eo:cloud_cover', 100) < 8]
    pool = good if good else feats
    pool.sort(key=(lambda f: f['properties']['datetime']) if good else (lambda f: -f['properties'].get('eo:cloud_cover', 100)))
    return pool[-1]
async def getnpy2(href, bb, E, res):
    minx, miny, maxx, maxy = bb; W = int(round((maxx - minx) / res)); H = int(round((maxy - miny) / res))
    u = f"https://titiler.xyz/cog/bbox/{minx},{miny},{maxx},{maxy}/{W}x{H}.npy?url={href}&coord_crs=epsg:{E}&dst_crs=epsg:{E}"
    for t in range(4):
        try:
            r = await pyfetch(u)
            if r.status == 200: return np.load(io.BytesIO(await r.bytes()))
        except Exception:
            pass
        await asyncio.sleep(1.5 * (t + 1))
    return None
def zmap(ev, evm, rf, rfm, k):
    v = evm & (ev > 0); mode = 'self'; Rr = ev
    if rf is not None:
        vr = v & rfm & (rf > 0)
        if vr.mean() > 0.6:
            g = np.median(ev[vr] / rf[vr]); Rr = ev - g * rf; v = vr; mode = 'ref'
    Rr = Rr - median_filter(Rr, k)
    if v.sum() < 300: return None, 'empty'
    md = np.median(Rr[v]); sg = 1.4826 * np.median(np.abs(Rr[v] - md)) + 1e-6
    Z = ((Rr - md) / sg).astype(np.float32); Z[~v] = np.nan
    return Z, mode
def dens(P, n=240):
    a = np.linspace(P[0, 0], P[-1, 0], n)
    return np.c_[a, np.interp(a, P[:, 0], P[:, 1]), np.interp(a, P[:, 0], P[:, 2]), np.interp(a, P[:, 0], P[:, 3])]
def normals(P):
    tx, ty = np.gradient(P[:, 2]), np.gradient(P[:, 3]); nn = np.hypot(tx, ty) + 1e-9
    return -ty / nn, tx / nn
async def fetch_tile(e, ref, cell, bands):
    i, j = cell; bb = (i * GRID - PAD, j * GRID - PAD, (i + 1) * GRID + PAD, (j + 1) * GRID + PAD); E = e['epsg']
    jobs = []
    for b in bands:
        jobs.append(getnpy2(e['item']['assets'][b]['href'], bb, E, BRES[b]))
        jobs.append(getnpy2(ref['assets'][b]['href'], bb, E, BRES[b]) if ref is not None and b in ref['assets'] else asyncio.sleep(0))
    out = []
    for s in range(0, len(jobs), 6): out += await asyncio.gather(*jobs[s:s + 6])
    T = dict(cell=cell, bb=bb, dark={}, bright={}, Z={}, RGB=None, modes=[])
    rgb = {}
    for n, b in enumerate(bands):
        a, r = out[2 * n], out[2 * n + 1]
        if a is None: return None
        ev = a[0].astype(np.float32); evm = a[-1] > 0
        rf = r[0].astype(np.float32) if isinstance(r, np.ndarray) else None; rfm = (r[-1] > 0) if isinstance(r, np.ndarray) else None
        Z, m = zmap(ev, evm, rf, rfm, 9 if BRES[b] == 10 else 7); T['modes'].append(m)
        if Z is None: Z = np.full(ev.shape, np.nan, np.float32)
        Zf = np.nan_to_num(Z, nan=0.0)
        T['dark'][b] = np.where(np.isnan(Z), np.nan, -minimum_filter(Zf, size=3)).astype(np.float16)
        T['bright'][b] = np.where(np.isnan(Z), np.nan, maximum_filter(Zf, size=3)).astype(np.float16)
        if b in ('blue', 'green', 'red', 'nir'): T['Z'][b] = Z.astype(np.float16)
        if b in ('red', 'green', 'blue'): rgb[b] = a[0]
    T['RGB'] = np.stack([rgb['red'], rgb['green'], rgb['blue']], -1)
    return T
class Mosaic:
    def __init__(s, tiles, bands):
        ci = np.array([T['cell'][0] for T in tiles]); cj = np.array([T['cell'][1] for T in tiles])
        s.i0, s.j0 = ci.min(), cj.min(); s.M = -np.ones((ci.max() - s.i0 + 1, cj.max() - s.j0 + 1), int)
        for k, T in enumerate(tiles): s.M[T['cell'][0] - s.i0, T['cell'][1] - s.j0] = k
        s.st = {key: {b: np.stack([T[key][b] for T in tiles]) for b in bands} for key in ('dark', 'bright')}
        s.mu = {key: {b: float(np.nanmean(s.st[key][b][:, ::3, ::3].astype(np.float32))) for b in bands} for key in s.st}
        s.sd = {key: {b: float(np.nanstd(s.st[key][b][:, ::3, ::3].astype(np.float32))) + 1e-6 for b in bands} for key in s.st}
    def get(s, key, b, X, Y):
        res = BRES[b]; I = np.floor(X / GRID).astype(int) - s.i0; J = np.floor(Y / GRID).astype(int) - s.j0
        ok = (I >= 0) & (I < s.M.shape[0]) & (J >= 0) & (J < s.M.shape[1]); t = np.full(X.shape, -1)
        t[ok] = s.M[I[ok], J[ok]]; ok &= t >= 0
        A = s.st[key][b]; H, W = A.shape[1:]
        col = np.floor((X - ((I + s.i0) * GRID - PAD)) / res).astype(int); row = np.floor((((J + s.j0) + 1) * GRID + PAD - Y) / res).astype(int)
        ok &= (col >= 0) & (col < W) & (row >= 0) & (row < H)
        V = np.full(X.shape, np.nan, np.float32); V[ok] = A[t[ok], row[ok], col[ok]]
        return V
async def match_tile(e):
    t = dt_from_jd(e['jd']); lt = e['sent'][-1]
    feats = await stac_post({'collections': ['sentinel-2-c1-l2a'], 'limit': 10, 'intersects': {'type': 'Point', 'coordinates': [e['lon'], e['lat']]}, 'datetime': f"{iso(t - dtm.timedelta(minutes=6))}/{iso(t + dtm.timedelta(minutes=6))}"})
    feats = [f for f in feats if f['id'].startswith('S2' + lt)]
    if not feats: return False
    feats.sort(key=lambda f: f['properties'].get('eo:cloud_cover', 100)); e['item'] = feats[0]; e['epsg'] = epsg_of(feats[0]); e['cloud'] = feats[0]['properties'].get('eo:cloud_cover') or 0.0
    return True
import xml.etree.ElementTree as ET
OMEGA_E = 7.2921158553e-5
def band_plane(rS, vS, alpha):
    rS = np.asarray(rS, float); vS = np.asarray(vS, float)
    vE = vS - OMEGA_E * np.array([-rS[1], rS[0], 0.0])
    rh = rS / np.linalg.norm(rS); n = fan_normal(rS, vE); ct = np.cross(rh, n)
    look = -rh * math.cos(alpha) + n * math.sin(alpha)
    return np.cross(look, ct)
async def band_alphas2(e, near_km=15):
    root = ET.fromstring(await gettext(e['item']['assets']['granule_metadata']['href']))
    for g in root.iter('Geoposition'):
        if g.get('resolution') == '10': ulx = float(g.find('ULX').text); uly = float(g.find('ULY').text)
    x0, y0 = latlon_to_utm(e['lat'], e['lon'], e['epsg'] % 100); sen = SENT[e['sent']]
    rS, _ = sen.prop_jd([e['jd']]); la1, lo1 = teme_to_geodetic(rS[0], e['jd'])
    rS2, _ = sen.prop_jd([e['jd'] + 1 / 86400]); la2, lo2 = teme_to_geodetic(rS2[0], e['jd'] + 1 / 86400)
    dlo = (lo2 - lo1 + 540) % 360 - 180; psi = math.atan2(dlo * math.cos(math.radians(la1)), la2 - la1)
    out = {}
    for grid in root.iter('Viewing_Incidence_Angles_Grids'):
        names = [k for k, v in BID.items() if v == grid.get('bandId')]
        if not names: continue
        rd = lambda tag: np.array([[float(v) for v in row.text.split()] for row in grid.find(tag).find('Values_List').findall('VALUES')])
        Zg, Ag = rd('Zenith'), rd('Azimuth')
        al = np.degrees(np.arctan(-np.tan(np.radians(Zg)) * np.cos(np.radians(Ag) - psi)))
        rr, cc = np.mgrid[0:Zg.shape[0], 0:Zg.shape[1]]
        near = np.isfinite(al) & (np.hypot(ulx + cc * 5000 - x0, uly - rr * 5000 - y0) < near_km * 1000)
        if near.sum() == 0: continue
        out.setdefault(names[0], []).append(dict(det=int(grid.get('detectorId')), a=float(np.nanmedian(al[near])), ok=np.isfinite(al), ulx=ulx, uly=uly))
    return out, math.degrees(psi)
def inside(dd, X, Y):
    r = np.rint((dd['uly'] - Y) / 5000).astype(int); c = np.rint((X - dd['ulx']) / 5000).astype(int)
    nr, nc = dd['ok'].shape; ins = (r >= 0) & (r < nr) & (c >= 0) & (c < nc)
    ins[ins] = dd['ok'][r[ins], c[ins]]
    return ins
def xy_at(P, nx, ny, a, C):
    AF = P[:, 0]
    return (np.interp(a, AF, P[:, 2])[:, None] + C[None, :] * np.interp(a, AF, nx)[:, None], np.interp(a, AF, P[:, 3])[:, None] + C[None, :] * np.interp(a, AF, ny)[:, None])
def plan_tiles2(loci, DA, bands, Wd, cmax):
    pts = []
    for P in loci.values():
        nx, ny = normals(P)
        for b in bands:
            for dd in DA[b]:
                a = np.arange(dd['a'] - Wd, dd['a'] + Wd + 1e-9, 0.01); X, Y = xy_at(P, nx, ny, a, np.array([-cmax, -cmax / 2, 0, cmax / 2, cmax]))
                m = inside(dd, X, Y); pts.append(np.c_[X[m], Y[m]])
    pts = np.vstack(pts)
    return sorted({(int(np.floor(x / GRID)), int(np.floor(y / GRID))) for x, y in pts})
def search4(loci, mos, DA, bands, Wd, cmax=1500, cstep=20, dstep=0.003):
    best = (-1e9, None); C = np.arange(-cmax, cmax + 1, cstep); D = np.arange(-Wd, Wd + 1e-9, dstep)
    for dM, P in loci.items():
        nx, ny = normals(P)
        for key in ('dark', 'bright'):
            sm = np.zeros((len(D), len(C)), np.float32); cnt = np.zeros_like(sm)
            for b in bands:
                Vb = np.full(sm.shape, np.nan, np.float32)
                for dd in DA[b]:
                    X, Y = xy_at(P, nx, ny, dd['a'] + D, C); V = (mos.get(key, b, X, Y) - mos.mu[key][b]) / mos.sd[key][b]
                    V[~inside(dd, X, Y)] = np.nan; Vb = np.fmax(Vb, V)
                okv = ~np.isnan(Vb); sm[okv] += np.minimum(Vb[okv], CLIP); cnt[okv] += 1
            S = np.where(cnt >= max(4, 0.6 * len(bands)), sm / np.sqrt(np.maximum(cnt, 1)), -1e9)
            i, j = np.unravel_index(np.argmax(S), S.shape)
            if S[i, j] > best[0]: best = (float(S[i, j]), dict(dM=dM, d=float(D[i]), c=float(C[j]), sign=key))
    return best
def best_positions(loci, DA, bands, info):
    P = loci[info['dM']]; nx, ny = normals(P); out = {}
    for b in bands:
        for dd in DA[b]:
            X, Y = xy_at(P, nx, ny, np.array([dd['a'] + info['d']]), np.array([info['c']]))
            if inside(dd, X, Y)[0, 0]: out[b] = (float(X[0, 0]), float(Y[0, 0]), dd['det']); break
    return out
RES4 = {}
async def run4(e, tag, dM_half=3000, dM_step=750, n_null=8, max_tiles=48):
    t0 = time.time()
    DA, psi = await band_alphas2(e)
    bands = [b for b in BID if b in DA and b in e['item']['assets']]
    if not all(b in bands for b in ('blue', 'green', 'red', 'nir')): print('RESULT16', tag, 'skip: no angles'); return
    allA = [dd['a'] for b in bands for dd in DA[b]]; cA = float(np.mean(allA))
    Pp = locus(e, np.array([cA - 0.5, cA + 0.5])); kmdeg = math.hypot(Pp[1, 2] - Pp[0, 2], Pp[1, 3] - Pp[0, 3]) / 1000
    Wd = float(np.clip(12 / kmdeg, 0.2, 0.8)); span = max(allA) - min(allA) + 2 * Wd + 0.1
    al = np.linspace(min(allA) - Wd - 0.05, max(allA) + Wd + 0.05, int(np.clip(span / 0.06, 25, 70)))
    loci = {float(dM): dens(locus(e, al, dM), n=500) for dM in np.arange(-dM_half, dM_half + 1, dM_step)}
    ref = await ref_same_orbit(e); cells = plan_tiles2(loci, DA, bands, Wd, 1500)
    dets = {b: [(dd['det'], round(dd['a'], 3)) for dd in DA[b]] for b in ('blue', 'red')}
    print(f"RESULT16 {tag} {e['obj']} {e['sent']} v={e['v_al']:.1f} rng={e['rng']:.0f} km/deg={kmdeg:.1f} Wd={Wd:.2f} dets={dets} ref={ref['id'][:30] if ref else None} tiles={len(cells)} loci {time.time() - t0:.0f}s")
    if len(cells) > max_tiles:
        P0 = loci[0.0] if 0.0 in loci else list(loci.values())[len(loci) // 2]; cx, cy = np.mean(P0[:, 2]), np.mean(P0[:, 3])
        cells = sorted(cells, key=lambda c: math.hypot((c[0] + 0.5) * GRID - cx, (c[1] + 0.5) * GRID - cy))[:max_tiles]
    tiles = []
    for cell in cells:
        T = await fetch_tile(e, ref, cell, bands)
        if T: tiles.append(T)
    if not tiles: print('RESULT16', tag, 'no tiles'); return
    modes = [m for T in tiles for m in T['modes']]; mos = Mosaic(tiles, bands)
    score, info = search4(loci, mos, DA, bands, Wd)
    rng = np.random.default_rng(11 + len(RES4)); nl = []
    for _ in range(n_null):
        p = rng.permutation(len(bands)); DAp = {b: DA[bands[p[i]]] for i, b in enumerate(bands)}
        nl.append(search4(loci, mos, DAp, bands, Wd)[0])
    s4 = search4(loci, mos, DA, ['blue', 'green', 'red', 'nir'], Wd)[0]
    pos = best_positions(loci, DA, bands, info) if info else {}
    vals = {b: float(mos.get(info['sign'], b, np.array([x]), np.array([y]))[0]) for b, (x, y, d) in pos.items()} if info else {}
    RES4[tag] = dict(e=e, DA=DA, bands=bands, score=score, info=info, nulls=nl, s4=s4, pos=pos, vals=vals, ref=ref, kmdeg=kmdeg, Wd=Wd, loci0=loci.get(0.0))
    print(f"RESULT16 {tag} done {time.time() - t0:.0f}s ref/self/empty={modes.count('ref')}/{modes.count('self')}/{modes.count('empty')} | TRUE={score:.2f} NULLmax={max(nl):.2f} med={np.median(nl):.2f} | 10m={s4:.2f} | {info} | " + ' '.join(f"{b}={v:.1f}" for b, v in vals.items()))
    del tiles, mos; gc.collect()
async def look4(tag, res=20, half_km=4.0):
    R = RES4[tag]; e = R['e']; pos = R['pos']
    if not pos: return
    xs = [v[0] for v in pos.values()]; ys = [v[1] for v in pos.values()]
    cx, cy = float(np.mean(xs)), float(np.mean(ys)); hw = round((max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + half_km * 1000) / res) * res
    bb = (cx - hw, cy - hw, cx + hw, cy + hw); Zs = {}; EVs = {}
    for b in ('blue', 'green', 'red'):
        a = await getnpy2(e['item']['assets'][b]['href'], bb, e['epsg'], res); ev = a[0].astype(np.float32); EVs[b] = ev
        r = await getnpy2(R['ref']['assets'][b]['href'], bb, e['epsg'], res) if R['ref'] else None
        if r is not None:
            rf = r[0].astype(np.float32); v = (ev > 0) & (rf > 0); g = np.median(ev[v] / rf[v]); Rr = ev - g * rf
        else: Rr = ev - median_filter(ev, 15)
        Rr = Rr - median_filter(Rr, 9); Zs[b] = Rr / (1.4826 * np.median(np.abs(Rr - np.median(Rr))) + 1e-6)
    tr = lambda x, y: ((x - bb[0]) / res, (bb[3] - y) / res)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    rgb = np.stack([EVs['red'], EVs['green'], EVs['blue']], -1); lo, hi = np.percentile(rgb, [1, 99])
    ax[0].imshow(np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)); ax[0].set_title(f"{tag} {e['obj']} {e['item']['id'][:24]} RGB")
    sgn = -1 if R['info']['sign'] == 'dark' else 1
    ax[1].imshow(np.stack([np.clip(sgn * Zs[b], 0, 10) for b in ('red', 'green', 'blue')], -1) / 10); ax[1].set_title(f"{R['info']['sign']} anomalies, R/G/B = band; TRUE {R['score']:.1f} vs NULL max {max(R['nulls']):.1f}")
    col = {'blue': 'cyan', 'green': 'lime', 'red': 'red', 'nir': 'white'}
    for a in ax:
        if R['loci0'] is not None:
            px, py = tr(R['loci0'][:, 2], R['loci0'][:, 3]); a.plot(px, py, 'y--', lw=0.6)
        for b in ('blue', 'green', 'red', 'nir'):
            if b in pos:
                u, w = tr(pos[b][0], pos[b][1]); a.add_patch(plt.Circle((u, w), 300 / res, fill=False, ec=col[b], lw=1.5))
        a.set_xlim(0, rgb.shape[1]); a.set_ylim(rgb.shape[0], 0); a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.show()
J1 = jd_from_dt(dt.datetime(2026, 11, 19, 12, 0, tzinfo=dt.timezone.utc))
BIG = list({o.cat: o for o in objs if o.cat in ('25544', '48274')}.values())
bev = [x for x in find_crossings(sents, BIG, J1 - 4.0, J1) if x['desc'] and x['sun_el'] > 8]
BEV = []
for x in bev:
    if await match_tile(x) and x['cloud'] < 70: BEV.append(x)
print('MEGA big-object events with tile:', [(x['obj'], str(dt_from_jd(x['jd']))[:16], x['item']['id'][:24], round(x['cloud'])) for x in BEV])
t0 = time.time()
EV = find_crossings(sents, [o for o in objs if o.name.startswith('STARLINK')], J1 - 1.0, J1, step_s=30)
day = [e for e in EV if e['desc'] and e['sun_el'] > 8 and size_m(e) >= 30]
day.sort(key=lambda e: abs(e['v_al']))
print(f'MEGA starlink crossings {len(EV)}, daylit large {len(day)}, {time.time() - t0:.0f}s')
CAND = []
for x in day[:60]:
    if await match_tile(x) and x['cloud'] < 25: CAND.append(x)
    if len(CAND) >= 6: break
print('MEGA starlink events:', [(x['obj'], round(x['v_al'], 1), x['item']['id'][:24], round(x['cloud'])) for x in CAND])
Q4 = [(x, 'BIG%d' % i, 15000, 1500) for i, x in enumerate(BEV)] + [(x, 'SL%d' % i, 3000, 750) for i, x in enumerate(CAND)]
for e, tag, dmh, dms in Q4:
    try: await run4(e, tag, dmh, dms)
    except Exception: print('RESULT16 fail', tag, traceback.format_exc()[-300:].replace('\n', ' | '))
ranked = sorted(RES4, key=lambda t: -(RES4[t]['score'] - max(RES4[t]['nulls'])))
print('RESULT16 RANK', [(t, round(RES4[t]['score'], 1), round(max(RES4[t]['nulls']), 1)) for t in ranked])
for t in ranked[:2]:
    try: await look4(t)
    except Exception: print('RESULT16 lookfail', t, traceback.format_exc()[-200:])
print('RESULT16 ALLDONE')

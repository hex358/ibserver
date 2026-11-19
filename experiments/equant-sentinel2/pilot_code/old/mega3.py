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
RLOC = 6371.0; R20 = 20.0; CLIPE = 8.0
def utm_to_latlon(x, y, zone, south):
    a = 6378137.0; f = 1 / 298.257223563; k0 = 0.9996; e2 = f * (2 - f); ep2 = e2 / (1 - e2)
    x = x - 500000.0; y = y - (10000000.0 if south else 0.0); M = y / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256)); e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    p1 = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu) + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu) + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
    N1 = a / math.sqrt(1 - e2 * math.sin(p1) ** 2); T1 = math.tan(p1) ** 2; C1 = ep2 * math.cos(p1) ** 2
    R1 = a * (1 - e2) / (1 - e2 * math.sin(p1) ** 2) ** 1.5; D = x / (N1 * k0)
    lat = p1 - (N1 * math.tan(p1) / R1) * (D * D / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * ep2) * D ** 4 / 24 + (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * ep2 - 3 * C1 * C1) * D ** 6 / 720)
    lon = (D - (1 + 2 * T1 + C1) * D ** 3 / 6 + (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * ep2 + 24 * T1 * T1) * D ** 5 / 120) / math.cos(p1)
    return math.degrees(lat), (zone - 1) * 6 - 180 + 3 + math.degrees(lon)
async def items_around(e, d=0.45):
    t = dt_from_jd(e['jd']); lt = e['sent'][-1]
    poly = {'type': 'Polygon', 'coordinates': [[[e['lon'] - d, e['lat'] - d], [e['lon'] + d, e['lat'] - d], [e['lon'] + d, e['lat'] + d], [e['lon'] - d, e['lat'] + d], [e['lon'] - d, e['lat'] - d]]]}
    its = [f for f in await stac_post({'collections': ['sentinel-2-c1-l2a'], 'limit': 30, 'intersects': poly, 'datetime': f"{iso(t - dtm.timedelta(minutes=4))}/{iso(t + dtm.timedelta(minutes=4))}"}) if f['id'].startswith('S2' + lt)]
    its.sort(key=lambda f: f['id'] != e['item']['id'])
    return its
async def band_grids(e, its):
    sen = SENT[e['sent']]; rS, _ = sen.prop_jd([e['jd']]); H = float(np.linalg.norm(rS[0])) - RLOC
    la1, lo1 = teme_to_geodetic(rS[0], e['jd']); rS2, _ = sen.prop_jd([e['jd'] + 1 / 86400]); la2, lo2 = teme_to_geodetic(rS2[0], e['jd'] + 1 / 86400)
    psi = math.atan2(((lo2 - lo1 + 540) % 360 - 180) * math.cos(math.radians(la1)), la2 - la1)
    out = {}
    for it in its:
        if epsg_of(it) != e['epsg']: continue
        try: root = ET.fromstring(await gettext(it['assets']['granule_metadata']['href']))
        except Exception: continue
        ulx = uly = None
        for g in root.iter('Geoposition'):
            if g.get('resolution') == '10': ulx = float(g.find('ULX').text); uly = float(g.find('ULY').text)
        for grid in root.iter('Viewing_Incidence_Angles_Grids'):
            names = [k for k, v in BID.items() if v == grid.get('bandId')]
            if not names: continue
            rd = lambda tag: np.array([[float(v) for v in row.text.split()] for row in grid.find(tag).find('Values_List').findall('VALUES')])
            Zg, Ag = rd('Zenith'), rd('Azimuth')
            eta = np.arcsin(RLOC * np.sin(np.radians(Zg)) / (RLOC + H))
            al = np.degrees(np.arctan(-np.tan(eta) * np.cos(np.radians(Ag) - psi)))
            out.setdefault(names[0], []).append(dict(det=int(grid.get('detectorId')), g=al, ulx=ulx, uly=uly, item=it['id']))
    return out, math.degrees(psi), H
def gval(dd, X, Y):
    g = dd['g']; nr, nc = g.shape; fr = (dd['uly'] - Y) / 5000.0; fc = (X - dd['ulx']) / 5000.0
    r0 = np.floor(fr).astype(int); c0 = np.floor(fc).astype(int); wr = fr - r0; wc = fc - c0
    ok = (r0 >= 0) & (c0 >= 0) & (r0 < nr - 1) & (c0 < nc - 1)
    acc = np.zeros(X.shape); wac = np.zeros(X.shape)
    for dr, dc, w in ((0, 0, (1 - wr) * (1 - wc)), (0, 1, (1 - wr) * wc), (1, 0, wr * (1 - wc)), (1, 1, wr * wc)):
        v = g[np.clip(r0 + dr, 0, nr - 1), np.clip(c0 + dc, 0, nc - 1)]; m = ok & np.isfinite(v)
        acc[m] += w[m] * v[m]; wac[m] += w[m]
    out = np.full(X.shape, np.nan); m = wac > 0.5; out[m] = acc[m] / wac[m]
    return out
def band_roots(P, DG, bands):
    A = P[:, 0]; X = P[:, 2]; Y = P[:, 3]; R = {}
    for b in bands:
        lst = []
        for dd in DG.get(b, []):
            f = A - gval(dd, X, Y)
            ok = np.isfinite(f[:-1]) & np.isfinite(f[1:]) & (np.sign(f[:-1]) != np.sign(f[1:]))
            for i in np.where(ok)[0]:
                w = f[i] / (f[i] - f[i + 1]); x = X[i] + w * (X[i + 1] - X[i]); y = Y[i] + w * (Y[i + 1] - Y[i])
                if all(math.hypot(x - q[2], y - q[3]) > 200 for q in lst): lst.append((dd['det'], float(A[i] + w * (A[i + 1] - A[i])), float(x), float(y)))
        if lst: R[b] = lst
    return R
def block2(a): return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean(axis=(1, 3))
async def fetch_cell2(cell, cands, bands, E, Arep, thr):
    i, j = cell; bb = (i * GRID - PAD, j * GRID - PAD, (i + 1) * GRID + PAD, (j + 1) * GRID + PAD)
    best = None
    for it, rf in cands[:3]:
        a = await getnpy2(it['assets']['blue']['href'], bb, E, 10)
        if a is None: continue
        cov = float((a[-1] > 0).mean())
        if best is None or cov > best[2]: best = (it, rf, cov, a)
        if cov > 0.97: break
    if best is None or best[2] < 0.2: return None
    it, rf, cov, a_blue = best; Z = {}; RGB = {}
    for b in bands:
        res = BRES[b]
        a = a_blue if b == 'blue' else await getnpy2(it['assets'][b]['href'], bb, E, res)
        r = await getnpy2(rf['assets'][b]['href'], bb, E, res) if (rf is not None and b in rf['assets']) else None
        if a is None: return None
        ev = a[0].astype(np.float32); evm = a[-1] > 0
        rv = r[0].astype(np.float32) if r is not None else None; rm = (r[-1] > 0) if r is not None else None
        if res == 10:
            ev = block2(ev); evm = block2(evm.astype(np.float32)) > 0.99
            if rv is not None: rv = block2(rv); rm = block2(rm.astype(np.float32)) > 0.99
        z, m = zmap(ev, evm, rv, rm, 7)
        Z[b] = z if z is not None else np.full(ev.shape, np.nan, np.float32)
        if b in ('red', 'green', 'blue'): RGB[b] = ev
    AZ = {b: np.abs(np.nan_to_num(Z[b])) for b in bands}; Ex = {}
    for b in bands:
        far = [c for c in bands if abs(Arep[c] - Arep[b]) > thr]
        F = maximum_filter(np.max(np.stack([AZ[c] for c in far]), 0), size=7) if far else 0.0
        ex = maximum_filter(np.maximum(0.0, AZ[b] - F), size=3).astype(np.float32); ex[np.isnan(Z[b])] = np.nan; Ex[b] = ex.astype(np.float16)
    return dict(cell=cell, bb=bb, E=Ex, Z={b: Z[b].astype(np.float16) for b in ('blue', 'green', 'red', 'nir')}, RGB=np.stack([RGB['red'], RGB['green'], RGB['blue']], -1), item=it['id'], cov=cov)
class Mos2:
    def __init__(s, tiles, bands):
        ci = [T['cell'][0] for T in tiles]; cj = [T['cell'][1] for T in tiles]
        s.i0, s.j0 = min(ci), min(cj); s.M = -np.ones((max(ci) - s.i0 + 1, max(cj) - s.j0 + 1), int)
        for k, T in enumerate(tiles): s.M[T['cell'][0] - s.i0, T['cell'][1] - s.j0] = k
        s.st = {b: np.stack([T['E'][b] for T in tiles]) for b in bands}
    def get(s, b, X, Y):
        I = np.floor(X / GRID).astype(int) - s.i0; J = np.floor(Y / GRID).astype(int) - s.j0
        ok = (I >= 0) & (I < s.M.shape[0]) & (J >= 0) & (J < s.M.shape[1]); t = np.full(X.shape, -1); t[ok] = s.M[I[ok], J[ok]]; ok &= t >= 0
        A = s.st[b]; H, W = A.shape[1:]
        col = np.floor((X - ((I + s.i0) * GRID - PAD)) / R20).astype(int); row = np.floor((((J + s.j0) + 1) * GRID + PAD - Y) / R20).astype(int)
        ok &= (col >= 0) & (col < W) & (row >= 0) & (row < H)
        V = np.full(X.shape, np.nan, np.float32); V[ok] = A[t[ok], row[ok], col[ok]]
        return V
def shift_search(roots, mos, bands, Rm, step=20.0, disp=None):
    S = np.arange(-Rm, Rm + 1e-6, step); SX, SY = np.meshgrid(S, S); tot = np.zeros(SX.shape, np.float32); cnt = np.zeros_like(tot); hit = np.zeros_like(tot)
    for b in bands:
        if b not in roots: continue
        dx, dy = disp[b] if disp else (0.0, 0.0); Vb = np.full(SX.shape, np.nan, np.float32)
        for (det, a, x, y) in roots[b]: Vb = np.fmax(Vb, mos.get(b, x + dx + SX, y + dy + SY))
        ok = ~np.isnan(Vb); tot[ok] += np.minimum(Vb[ok], CLIPE); cnt[ok] += 1; hit[ok] += Vb[ok] > 4
    sc = np.where(cnt >= max(4, 0.6 * len(roots)), tot / np.sqrt(np.maximum(cnt, 1)), -1e9)
    i, j = np.unravel_index(np.argmax(sc), sc.shape)
    return float(sc[i, j]), float(SX[i, j]), float(SY[i, j]), int(hit[i, j]), int(cnt[i, j])
RES5 = {}
async def run5(e, tag, Rm=4000.0, n_null=10):
    t0 = time.time(); its = await items_around(e)
    DG, psi, H = await band_grids(e, its)
    bands = [b for b in BID if b in DG and b in e['item']['assets']]
    P = dens(locus(e, np.linspace(-2.2, 2.2, 89), 0.0), n=1500)
    roots = band_roots(P, DG, bands); bands = [b for b in bands if b in roots]
    Arep = {b: float(np.median([q[1] for q in roots[b]])) for b in bands}
    kmdeg = float(np.median(np.hypot(np.diff(P[:, 2]), np.diff(P[:, 3])) / np.diff(P[:, 0]))) / 1000
    thr = max(0.03, 0.6 / kmdeg)
    print(f"RESULT19 {tag} {e['obj']} v={e['v_al']:.1f} rng={e['rng']:.0f} H={H:.0f} km/deg={kmdeg:.1f} items={len(its)} roots=" + ' '.join(f"{b}:{[(q[0], round(q[1], 3)) for q in roots[b]]}" for b in ('blue', 'nir', 'green', 'red') if b in roots) + f" ({time.time() - t0:.0f}s)")
    if len(bands) < 4: print('RESULT19', tag, 'too few bands with roots'); return
    zone = e['epsg'] % 100; south = e['epsg'] // 100 == 327
    cells = set()
    for b in bands:
        for (det, a, x, y) in roots[b]:
            for xx in np.arange(x - Rm - 400, x + Rm + 401, 2500):
                for yy in np.arange(y - Rm - 400, y + Rm + 401, 2500): cells.add((int(np.floor(xx / GRID)), int(np.floor(yy / GRID))))
    cells = sorted(cells)
    refs = {}
    for it in its: refs[it['id']] = await ref_same_orbit({'item': it})
    tiles = []
    for c in cells:
        la, lo = utm_to_latlon((c[0] + 0.5) * GRID, (c[1] + 0.5) * GRID, zone, south)
        cands = [(it, refs[it['id']]) for it in its if it['bbox'][0] <= lo <= it['bbox'][2] and it['bbox'][1] <= la <= it['bbox'][3]]
        cands.sort(key=lambda p: math.hypot(lo - (p[0]['bbox'][0] + p[0]['bbox'][2]) / 2, la - (p[0]['bbox'][1] + p[0]['bbox'][3]) / 2))
        try: T = await fetch_cell2(c, cands, bands, e['epsg'], Arep, thr)
        except Exception: T = None
        if T: tiles.append(T)
    print(f"RESULT19 {tag} cells {len(tiles)}/{len(cells)} fetched ({time.time() - t0:.0f}s)")
    if not tiles: return
    mos = Mos2(tiles, bands)
    sc, sx, sy, hit, cnt = shift_search(roots, mos, bands, Rm)
    rng = np.random.default_rng(5); nl = []
    for _ in range(n_null):
        disp = {}
        for b in bands:
            r = rng.uniform(1000, 4000); th = rng.uniform(0, 2 * math.pi); disp[b] = (r * math.cos(th), r * math.sin(th))
        nl.append(shift_search(roots, mos, bands, Rm, disp=disp)[0])
    per = {}
    for b in bands:
        per[b] = max(float(np.nan_to_num(mos.get(b, np.array([x + sx]), np.array([y + sy]))[0], nan=-1)) for (d, a, x, y) in roots[b])
    crops = {}
    for b in ('blue', 'green', 'red', 'nir'):
        if b not in roots: continue
        x, y = roots[b][0][2] + sx, roots[b][0][3] + sy
        for T in tiles:
            minx, miny, maxx, maxy = T['bb']; col = int((x - minx) // R20); row = int((maxy - y) // R20)
            if 0 <= col < 300 and 0 <= row < 300:
                sl = (slice(max(0, row - 30), row + 30), slice(max(0, col - 30), col + 30))
                crops[b] = dict(Z=T['Z'][b][sl].astype(np.float32), RGB=T['RGB'][sl].astype(np.float32), E=T['E'][b][sl].astype(np.float32)); break
    RES5[tag] = dict(e=e, roots=roots, bands=bands, score=sc, shift=(sx, sy), hit=hit, cnt=cnt, nulls=nl, per=per, crops=crops, kmdeg=kmdeg)
    print(f"RESULT19 {tag} TRUE={sc:.2f} (bands>4sigma {hit}/{cnt}) NULLmax={max(nl):.2f} med={np.median(nl):.2f} shift=({sx:.0f},{sy:.0f}) m | " + ' '.join(f"{b}={v:.1f}" for b, v in per.items()) + f" ({time.time() - t0:.0f}s)")
    del tiles, mos; gc.collect()
def plot5(tag):
    R = RES5[tag]; cr = R['crops']; fig, ax = plt.subplots(3, 4, figsize=(12, 9))
    for i, b in enumerate(('blue', 'nir', 'green', 'red')):
        if b not in cr: continue
        rgb = cr[b]['RGB']; lo, hi = np.percentile(rgb, [1, 99])
        ax[0, i].imshow(np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)); ax[0, i].set_title(f"{b} predicted spot, RGB", fontsize=9)
        ax[1, i].imshow(np.nan_to_num(cr[b]['Z']), cmap='RdBu_r', vmin=-10, vmax=10); ax[1, i].set_title(f"{b} change z", fontsize=9)
        ax[2, i].imshow(np.nan_to_num(cr[b]['E']), cmap='magma', vmin=0, vmax=10); ax[2, i].set_title(f"{b} band-exclusive, {R['per'][b]:.1f}", fontsize=9)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([]); a.plot([30], [30], 'c+', ms=14)
    fig.suptitle(f"{tag} {R['e']['obj']}: TRUE {R['score']:.1f} vs NULL max {max(R['nulls']):.1f}  (crops 1.2 km, 20 m px)"); plt.tight_layout(); plt.show()
for e, tag, Rm in [(CAND[0], 'SL0', 4000.0), (BEV[0], 'ISS', 6000.0), (CAND[1], 'SL1', 4000.0)]:
    try: await run5(e, tag, Rm)
    except Exception: print('RESULT19 fail', tag, traceback.format_exc()[-400:].replace('\n', ' | '))
for t in RES5: plot5(t)
print('RESULT19 ALLDONE')

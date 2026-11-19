import math, datetime as dt
import numpy as np

# ---------------- SGP4 (near-earth, WGS72), vectorised over time ----------------
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

# ---------------- geometry ----------------
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

# ---------------- crossing search ----------------
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

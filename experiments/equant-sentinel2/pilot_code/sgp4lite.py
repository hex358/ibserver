"""Near-earth SGP4 (WGS72), vectorised over time with numpy.
Follows Vallado et al. (2006) sgp4unit, near-earth branch only (period < 225 min).
"""
import math, re
import numpy as np

MU = 398600.8
RE = 6378.135
XKE = 60.0 / math.sqrt(RE**3 / MU)
J2 = 0.001082616
J3 = -0.00000253881
J4 = -0.00000165597
J3OJ2 = J3 / J2
X2O3 = 2.0 / 3.0
TWOPI = 2 * math.pi
VKMS = RE * XKE / 60.0


def _expo(s):
    # TLE implied-decimal exponent field, e.g. '-60382-4' -> -0.60382e-4
    s = s.strip()
    sign = -1.0 if s[0] == '-' else 1.0
    s = s.lstrip('+-')
    mant, exp = s[:-2], s[-2:]
    return sign * float('0.' + mant) * 10 ** int(exp)


def jday(year, doy):
    # Julian date of Jan 0.0 of year + doy
    jd_jan1 = 367 * year - int(7 * year / 4) + 30 + 1 + 1721013.5  # Jan 1.0 of year (1901-2099)
    return jd_jan1 + doy - 1.0


def gstime(jdut1):
    tut1 = (jdut1 - 2451545.0) / 36525.0
    temp = (-6.2e-6 * tut1**3 + 0.093104 * tut1**2 +
            (876600.0 * 3600 + 8640184.812866) * tut1 + 67310.54841)
    temp = math.radians(temp / 240.0) % TWOPI
    return temp


class Sat:
    def __init__(self, name, cat, epoch_str, bstar, inc, raan, ecc, argp, ma, mm):
        self.name, self.cat = name.strip(), cat
        yy = int(epoch_str[:2]); doy = float(epoch_str[2:])
        year = 2000 + yy if yy < 57 else 1900 + yy
        self.jdepoch = jday(year, doy)
        self.bstar = bstar
        self.inclo = math.radians(inc); self.nodeo = math.radians(raan)
        self.ecco = ecc; self.argpo = math.radians(argp); self.mo = math.radians(ma)
        self.no_kozai = mm * TWOPI / 1440.0
        self._init()

    def _init(self):
        e = self.ecco; inc = self.inclo
        eccsq = e * e; omeosq = 1 - eccsq; rteosq = math.sqrt(omeosq)
        cosio = math.cos(inc); cosio2 = cosio * cosio
        ak = (XKE / self.no_kozai) ** X2O3
        d1 = 0.75 * J2 * (3 * cosio2 - 1) / (rteosq * omeosq)
        dl = d1 / (ak * ak)
        adel = ak * (1 - dl * dl - dl * (1.0 / 3.0 + 134 * dl * dl / 81.0))
        dl = d1 / (adel * adel)
        self.no = self.no_kozai / (1 + dl)
        ao = (XKE / self.no) ** X2O3
        sinio = math.sin(inc)
        po = ao * omeosq
        con42 = 1 - 5 * cosio2
        self.con41 = -con42 - cosio2 - cosio2
        posq = po * po
        rp = ao * (1 - e)
        self.period_min = TWOPI / self.no
        if self.period_min >= 225:
            raise ValueError('deep-space object not supported')
        ss = 78.0 / RE + 1
        qzms2t = ((120.0 - 78.0) / RE) ** 4
        self.isimp = 1 if rp < (220.0 / RE + 1) else 0
        sfour = ss; qzms24 = qzms2t
        perige = (rp - 1) * RE
        if perige < 156:
            sfour = perige - 78
            if perige < 98:
                sfour = 20
            qzms24 = ((120 - sfour) / RE) ** 4
            sfour = sfour / RE + 1
        pinvsq = 1 / posq
        tsi = 1 / (ao - sfour)
        self.eta = eta = ao * e * tsi
        etasq = eta * eta; eeta = e * eta
        psisq = abs(1 - etasq)
        coef = qzms24 * tsi ** 4
        coef1 = coef / psisq ** 3.5
        cc2 = coef1 * self.no * (ao * (1 + 1.5 * etasq + eeta * (4 + etasq)) +
                                 0.375 * J2 * tsi / psisq * self.con41 * (8 + 3 * etasq * (8 + etasq)))
        self.cc1 = cc1 = self.bstar * cc2
        cc3 = 0.0
        if e > 1.0e-4:
            cc3 = -2 * coef * tsi * J3OJ2 * self.no * sinio / e
        self.x1mth2 = 1 - cosio2
        self.cc4 = 2 * self.no * coef1 * ao * omeosq * (
            eta * (2 + 0.5 * etasq) + e * (0.5 + 2 * etasq) -
            J2 * tsi / (ao * psisq) * (-3 * self.con41 * (1 - 2 * eeta + etasq * (1.5 - 0.5 * eeta)) +
                                       0.75 * self.x1mth2 * (2 * etasq - eeta * (1 + etasq)) * math.cos(2 * self.argpo)))
        self.cc5 = 2 * coef1 * ao * omeosq * (1 + 2.75 * (etasq + eeta) + eeta * etasq)
        cosio4 = cosio2 * cosio2
        temp1 = 1.5 * J2 * pinvsq * self.no
        temp2 = 0.5 * temp1 * J2 * pinvsq
        temp3 = -0.46875 * J4 * pinvsq * pinvsq * self.no
        self.mdot = self.no + 0.5 * temp1 * rteosq * self.con41 + 0.0625 * temp2 * rteosq * (13 - 78 * cosio2 + 137 * cosio4)
        self.argpdot = -0.5 * temp1 * con42 + 0.0625 * temp2 * (7 - 114 * cosio2 + 395 * cosio4) + temp3 * (3 - 36 * cosio2 + 49 * cosio4)
        xhdot1 = -temp1 * cosio
        self.nodedot = xhdot1 + (0.5 * temp2 * (4 - 19 * cosio2) + 2 * temp3 * (3 - 7 * cosio2)) * cosio
        self.omgcof = self.bstar * cc3 * math.cos(self.argpo)
        self.xmcof = 0.0
        if e > 1.0e-4:
            self.xmcof = -X2O3 * coef * self.bstar / eeta
        self.nodecf = 3.5 * omeosq * xhdot1 * cc1
        self.t2cof = 1.5 * cc1
        if abs(cosio + 1.0) > 1.5e-12:
            self.xlcof = -0.25 * J3OJ2 * sinio * (3 + 5 * cosio) / (1 + cosio)
        else:
            self.xlcof = -0.25 * J3OJ2 * sinio * (3 + 5 * cosio) / 1.5e-12
        self.aycof = -0.5 * J3OJ2 * sinio
        self.delmo = (1 + eta * math.cos(self.mo)) ** 3
        self.sinmao = math.sin(self.mo)
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
        """tsince: minutes from epoch (array). Returns r, v in TEME (km, km/s), shape (N,3)."""
        t = np.atleast_1d(np.asarray(tsince, dtype=float))
        xmdf = self.mo + self.mdot * t
        argpdf = self.argpo + self.argpdot * t
        nodedf = self.nodeo + self.nodedot * t
        argpm = argpdf.copy(); mm = xmdf.copy()
        t2 = t * t
        nodem = nodedf + self.nodecf * t2
        tempa = 1 - self.cc1 * t
        tempe = self.bstar * self.cc4 * t
        templ = self.t2cof * t2
        if self.isimp != 1:
            delomg = self.omgcof * t
            delm = self.xmcof * ((1 + self.eta * np.cos(xmdf)) ** 3 - self.delmo)
            temp = delomg + delm
            mm = xmdf + temp
            argpm = argpdf - temp
            t3 = t2 * t; t4 = t3 * t
            tempa = tempa - self.d2 * t2 - self.d3 * t3 - self.d4 * t4
            tempe = tempe + self.bstar * self.cc5 * (np.sin(mm) - self.sinmao)
            templ = templ + self.t3cof * t3 + t4 * (self.t4cof + t * self.t5cof)
        am = (XKE / self.no) ** X2O3 * tempa * tempa
        nm = XKE / am ** 1.5
        em = self.ecco - tempe
        em = np.where(em < 1e-6, 1e-6, em)
        mm = mm + self.no * templ
        xlm = mm + argpm + nodem
        nodem = np.fmod(nodem, TWOPI); argpm = np.fmod(argpm, TWOPI); xlm = np.fmod(xlm, TWOPI)
        mm = np.fmod(xlm - argpm - nodem, TWOPI)
        inclm = self.inclo
        sinip = math.sin(inclm); cosip = math.cos(inclm)
        ep, argpp, nodep, mp = em, argpm, nodem, mm
        axnl = ep * np.cos(argpp)
        temp = 1.0 / (am * (1 - ep * ep))
        aynl = ep * np.sin(argpp) + temp * self.aycof
        xl = mp + argpp + nodep + temp * self.xlcof * axnl
        u = np.fmod(xl - nodep, TWOPI)
        eo1 = u.copy()
        for _ in range(10):
            s1 = np.sin(eo1); c1 = np.cos(eo1)
            tem5 = 1 - c1 * axnl - s1 * aynl
            tem5 = (u - aynl * c1 + axnl * s1 - eo1) / tem5
            tem5 = np.clip(tem5, -0.95, 0.95)
            eo1 = eo1 + tem5
            if np.all(np.abs(tem5) < 1e-12):
                break
        sineo1 = np.sin(eo1); coseo1 = np.cos(eo1)
        ecose = axnl * coseo1 + aynl * sineo1
        esine = axnl * sineo1 - aynl * coseo1
        el2 = axnl * axnl + aynl * aynl
        pl = am * (1 - el2)
        rl = am * (1 - ecose)
        rdotl = np.sqrt(am) * esine / rl
        rvdotl = np.sqrt(pl) / rl
        betal = np.sqrt(1 - el2)
        temp = esine / (1 + betal)
        sinu = am / rl * (sineo1 - aynl - axnl * temp)
        cosu = am / rl * (coseo1 - axnl + aynl * temp)
        su = np.arctan2(sinu, cosu)
        sin2u = (cosu + cosu) * sinu
        cos2u = 1 - 2 * sinu * sinu
        temp = 1.0 / pl
        temp1 = 0.5 * J2 * temp
        temp2 = temp1 * temp
        mrt = rl * (1 - 1.5 * temp2 * betal * self.con41) + 0.5 * temp1 * self.x1mth2 * cos2u
        su = su - 0.25 * temp2 * self.x7thm1 * sin2u
        xnode = nodep + 1.5 * temp2 * cosip * sin2u
        xinc = inclm + 1.5 * temp2 * cosip * sinip * cos2u
        mvt = rdotl - nm * temp1 * self.x1mth2 * sin2u / XKE
        rvdot = rvdotl + nm * temp1 * (self.x1mth2 * cos2u + 1.5 * self.con41) / XKE
        sinsu = np.sin(su); cossu = np.cos(su)
        snod = np.sin(xnode); cnod = np.cos(xnode)
        sini = np.sin(xinc); cosi = np.cos(xinc)
        xmx = -snod * cosi; xmy = cnod * cosi
        ux = xmx * sinsu + cnod * cossu
        uy = xmy * sinsu + snod * cossu
        uz = sini * sinsu
        vx = xmx * cossu - cnod * sinsu
        vy = xmy * cossu - snod * sinsu
        vz = sini * cossu
        r = (mrt * RE)[:, None] * np.stack([ux, uy, uz], axis=1)
        v = VKMS * (mvt[:, None] * np.stack([ux, uy, uz], axis=1) + rvdot[:, None] * np.stack([vx, vy, vz], axis=1))
        return r, v

    def prop_jd(self, jd):
        return self.prop((np.asarray(jd) - self.jdepoch) * 1440.0)


TLE_RE = re.compile(
    r'(?P<name>[^\n]*?)\s*1 (?P<cat>\d{5})U (?:(?P<intl>\S+) )?(?P<epoch>\d{5}\.\d+) (?P<ndot>\S+) (?P<nddot>\S+) '
    r'(?P<bstar>\S+) \d (?P<el>\d+) 2 (?P=cat) (?P<inc>\S+) (?P<raan>\S+) (?P<ecc>\d{7}) (?P<argp>\S+) '
    r'(?P<ma>\S+) (?P<mm>\d{1,2}\.\d{8})\d*(?: \d+)?')


def parse_collapsed(text):
    sats = []
    for m in TLE_RE.finditer(text.replace('\n', ' ')):
        try:
            s = Sat(m['name'], m['cat'], m['epoch'], _expo(m['bstar']), float(m['inc']), float(m['raan']),
                    float('0.' + m['ecc']), float(m['argp']), float(m['ma']), float(m['mm']))
            sats.append(s)
        except ValueError:
            pass
    return sats

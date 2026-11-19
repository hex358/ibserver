"""Find moments when catalogue objects cross the Sentinel-2 MSI viewing fan (nadir cross-track plane, +-10.5 deg),
between the satellite and the Earth, on the daylit descending pass. Pilot uses a brute-force time grid
(fine for ~150 objects); Equant replaces this for the full catalogue x years.
"""
import math, sys, datetime as dt
import numpy as np
import sgp4lite as S

A_WGS84 = 6378.137
B_WGS84 = 6356.752314245
HALF_FOV = math.radians(10.5)


def jd_from_dt(d):
    return d.timestamp() / 86400.0 + 2440587.5


def dt_from_jd(jd):
    return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=jd - 2440587.5)


def sun_unit(jd):
    n = jd - 2451545.0
    L = math.radians((280.460 + 0.9856474 * n) % 360)
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 0.0000004 * n)
    return np.array([math.cos(lam), math.cos(eps) * math.sin(lam), math.sin(eps) * math.sin(lam)])


def ray_ellipsoid(p, d):
    # smallest positive lambda with p + lambda d on WGS84 ellipsoid
    a2, b2 = A_WGS84 ** 2, B_WGS84 ** 2
    A = (d[0] ** 2 + d[1] ** 2) / a2 + d[2] ** 2 / b2
    B = 2 * ((p[0] * d[0] + p[1] * d[1]) / a2 + p[2] * d[2] / b2)
    C = (p[0] ** 2 + p[1] ** 2) / a2 + p[2] ** 2 / b2 - 1
    disc = B * B - 4 * A * C
    if disc < 0:
        return None
    return (-B - math.sqrt(disc)) / (2 * A)


def teme_to_geodetic(r, jd):
    g = S.gstime(jd)
    x = math.cos(g) * r[0] + math.sin(g) * r[1]
    y = -math.sin(g) * r[0] + math.cos(g) * r[1]
    z = r[2]
    lon = math.degrees(math.atan2(y, x))
    e2 = 1 - (B_WGS84 / A_WGS84) ** 2
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(6):
        N = A_WGS84 / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1 - e2 * N / (N + h)))
    return math.degrees(lat), (lon + 180) % 360 - 180


def fan_f(rs, vs, ro):
    rhat = rs / np.linalg.norm(rs, axis=-1, keepdims=True)
    vh = vs - np.sum(vs * rhat, axis=-1, keepdims=True) * rhat
    n = vh / np.linalg.norm(vh, axis=-1, keepdims=True)
    return np.sum((ro - rs) * n, axis=-1)


def main(t_start, t_end, step_s=2.0):
    sats = S.parse_collapsed(open('tle_raw.txt').read())
    sents = [s for s in sats if s.name.startswith('SENTINEL-2')]
    objs = [s for s in sats if not s.name.startswith('SENTINEL-2')]
    jd0, jd1 = jd_from_dt(t_start), jd_from_dt(t_end)
    grid = np.arange(jd0, jd1, step_s / 86400.0)
    events = []
    for sen in sents:
        rs, vs = sen.prop_jd(grid)
        for ob in objs:
            ro, vo = ob.prop_jd(grid)
            f = fan_f(rs, vs, ro)
            dist = np.linalg.norm(ro - rs, axis=1)
            idx = np.where((np.sign(f[:-1]) != np.sign(f[1:])) & (dist[:-1] < 1200))[0]
            for i in idx:
                # secant refinement on exact propagation
                ta, tb = grid[i], grid[i + 1]
                fa, fb = f[i], f[i + 1]
                for _ in range(6):
                    tm = ta - fa * (tb - ta) / (fb - fa)
                    a, b_ = sen.prop_jd(np.array([tm])); c, _v = ob.prop_jd(np.array([tm]))
                    fm = fan_f(a, b_, c)[0]
                    if abs(fm) < 1e-4:
                        break
                    if np.sign(fm) == np.sign(fa):
                        ta, fa = tm, fm
                    else:
                        tb, fb = tm, fm
                rS, vS = sen.prop_jd(np.array([tm])); rO, vO = ob.prop_jd(np.array([tm]))
                rS, vS, rO, vO = rS[0], vS[0], rO[0], vO[0]
                d = rO - rS
                rng = np.linalg.norm(d)
                nadir = -rS / np.linalg.norm(rS)
                ang = math.acos(np.clip(np.dot(d / rng, nadir), -1, 1))
                if ang > HALF_FOV:
                    continue
                lam = ray_ellipsoid(rS, d / rng)
                if lam is None or lam < rng:
                    continue  # object not between satellite and ground
                gp = rS + lam * d / rng
                lat, lon = teme_to_geodetic(gp, tm)
                su = sun_unit(tm)
                sun_el = math.degrees(math.asin(np.dot(su, gp / np.linalg.norm(gp))))
                # object illuminated? (cylindrical shadow)
                proj = np.dot(rO, su)
                lit = proj > 0 or np.linalg.norm(rO - proj * su) > S.RE
                desc = vS[2] < 0
                # cross-track offset of ground point from nadir (signed, km): use cross-track unit
                rh = rS / np.linalg.norm(rS)
                vh = vS - np.dot(vS, rh) * rh
                ct = np.cross(rh, vh / np.linalg.norm(vh))
                xtrack = math.degrees(math.atan2(np.dot(d, ct), np.dot(d, nadir)))
                vrel = vO - vS
                alt = np.linalg.norm(rO) - S.RE
                events.append(dict(sentinel=sen.name, obj=ob.name, cat=ob.cat, utc=dt_from_jd(tm),
                                   lat=lat, lon=lon, off_nadir_deg=xtrack, range_km=rng, alt_km=alt,
                                   sun_el=sun_el, lit=lit, desc=desc, vrel=np.linalg.norm(vrel)))
    events.sort(key=lambda e: e['utc'])
    return events


if __name__ == '__main__':
    t1 = dt.datetime(2026, 11, 18, 9, 30, tzinfo=dt.timezone.utc)
    t0 = t1 - dt.timedelta(days=float(sys.argv[1]) if len(sys.argv) > 1 else 4)
    ev = main(t0, t1)
    print(f'{len(ev)} fan crossings within FOV (any lighting)')
    day = [e for e in ev if e['desc'] and e['sun_el'] > 5 and e['lit']]
    print(f'{len(day)} on daylit descending passes with object sunlit\n')
    for e in day:
        print(f"{e['utc']:%Y-%m-%d %H:%M:%S.%f}"[:-3], f"{e['sentinel']:<11} {e['obj']:<22} {e['cat']} "
              f"lat={e['lat']:7.3f} lon={e['lon']:8.3f} offnadir={e['off_nadir_deg']:+5.1f} "
              f"range={e['range_km']:5.0f} alt={e['alt_km']:4.0f} sunEl={e['sun_el']:4.1f} vrel={e['vrel']:4.1f}")

"""Apparent ground location of an object in a push-broom image for band along-track tilt alpha."""
import math, sys, numpy as np, datetime as dt
import sgp4lite as S, find_crossings as F
sats = {s.cat: s for s in S.parse_collapsed(open('tle_raw.txt').read())}
sen = {s.name: s for s in sats.values() if s.name.startswith('SENTINEL')}

def plane_f(jd, s, o, alpha, dM=0.0):
    rS, vS = s.prop_jd(np.array([jd])); rO, vO = o.prop_jd(np.array([jd]))
    rS, vS, rO, vO = rS[0], vS[0], rO[0], vO[0]
    rO = rO + dM * vO / np.linalg.norm(vO)  # along-track error of the object (km)
    rh = rS/np.linalg.norm(rS); vh = vS - vS.dot(rh)*rh; n = vh/np.linalg.norm(vh)
    look = -rh*math.cos(alpha) + n*math.sin(alpha)      # band boresight tilted forward by alpha
    nrm = np.cross(look, np.cross(rh, n))               # normal of the band's view plane
    return (rO - rS).dot(nrm), rS, rO

def apparent(utc, sname, cat, alpha_deg, dM=0.0):
    s, o = sen[sname], sats[cat]
    t0 = F.jd_from_dt(dt.datetime.strptime(utc, '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc))
    a = math.radians(alpha_deg)
    ta, tb = t0 - 3/86400, t0 + 3/86400
    fa, *_ = plane_f(ta, s, o, a, dM); fb, *_ = plane_f(tb, s, o, a, dM)
    if np.sign(fa) == np.sign(fb):
        ta, tb = t0 - 15/86400, t0 + 15/86400
        fa, *_ = plane_f(ta, s, o, a, dM); fb, *_ = plane_f(tb, s, o, a, dM)
    for _ in range(40):
        tm = 0.5*(ta+tb); fm, rS, rO = plane_f(tm, s, o, a, dM)
        if np.sign(fm) == np.sign(fa): ta, fa = tm, fm
        else: tb = tm
    d = (rO - rS)/np.linalg.norm(rO - rS)
    lam = F.ray_ellipsoid(rS, d)
    lat, lon = F.teme_to_geodetic(rS + lam*d, tm)
    return F.dt_from_jd(tm), lat, lon

if __name__ == '__main__':
    utc, sname, cat = sys.argv[1], sys.argv[2], sys.argv[3]
    for dM in (0.0, -2.0, 2.0):
        print(f'object along-track error {dM:+.0f} km:')
        for al in (-0.8, -0.4, 0.0, 0.4, 0.8):
            t, la, lo = apparent(utc, sname, cat, al, dM)
            print(f'   band tilt {al:+.1f} deg -> t={t:%H:%M:%S.%f} lat={la:.5f} lon={lo:.5f}')

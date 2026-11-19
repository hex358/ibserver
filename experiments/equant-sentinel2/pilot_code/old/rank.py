import math, numpy as np, datetime as dt
import sgp4lite as S, find_crossings as F
sats = {s.cat: s for s in S.parse_collapsed(open('tle_raw.txt').read())}
sen = {s.name: s for s in sats.values() if s.name.startswith('SENTINEL')}
cands = [  # (utc, sentinel, cat, label)
 ('2026-11-14 17:26:15.458','SENTINEL-2B','39679','Texas/Laredo'),
 ('2026-11-16 18:00:27.438','SENTINEL-2B','43682','North Dakota'),
 ('2026-11-16 16:22:30.384','SENTINEL-2B','20663','Ohio/WV'),
 ('2026-11-16 19:49:54.921','SENTINEL-2A','12139','British Columbia'),
 ('2026-11-17 10:58:25.777','SENTINEL-2A','17567','France'),
 ('2026-11-17 09:09:07.320','SENTINEL-2B','39679','Turkey/Dardanelles'),
 ('2026-11-17 16:41:44.151','SENTINEL-2C','14699','Michigan'),
 ('2026-11-16 09:50:39.801','SENTINEL-2A','39679','Sicily SE'),
 ('2026-11-18 05:26:31.939','SENTINEL-2A','11267','Xinjiang'),
 ('2026-11-18 07:00:09.438','SENTINEL-2B','11267','Iran'),
 ('2026-11-18 07:05:07.584','SENTINEL-2A','11267','RU/KZ border'),
 ('2026-11-18 08:38:45.056','SENTINEL-2B','11267','Turkey/Corum'),
]
IFOV = 10/786.0  # mrad -> km per km
for utc, sn, cat, lab in cands:
    t = dt.datetime.strptime(utc, '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc)
    jd = F.jd_from_dt(t)
    rS, vS = sen[sn].prop_jd(np.array([jd])); rO, vO = sats[cat].prop_jd(np.array([jd]))
    rS, vS, rO, vO = rS[0], vS[0], rO[0], vO[0]
    rh = rS/np.linalg.norm(rS); vh = vS - vS.dot(rh)*rh; n = vh/np.linalg.norm(vh); ct = np.cross(rh, n)
    vr = vO - vS
    rng = np.linalg.norm(rO-rS)
    foot = rng*IFOV*1000  # m per pixel at object
    vn, vc = vr.dot(n), vr.dot(ct)
    # ground track heading of Sentinel nadir (deg from north)
    lat, lon = F.teme_to_geodetic(rS, jd)
    lat2, lon2 = F.teme_to_geodetic(rS + vS*1.0, jd + 1/86400)
    hd = math.degrees(math.atan2((lon2-lon)*math.cos(math.radians(lat)), lat2-lat)) % 360
    dwell_ms = foot/abs(vn*1000)*1000
    print(f"{lab:<18} {sats[cat].name:<14} range={rng:5.0f}km pix@obj={foot:4.1f}m  v_rel along-track={vn:+6.2f} cross={vc:+6.2f} km/s  dwell/pixel={dwell_ms:5.2f}ms  S2 heading={hd:5.1f}")

import math, numpy as np, datetime as dt, sys
import sgp4lite as S, find_crossings as F
t1 = dt.datetime(2026,11,18,9,30,tzinfo=dt.timezone.utc)
ev = F.main(t1-dt.timedelta(days=float(sys.argv[1])), t1)
sats = {s.cat: s for s in S.parse_collapsed(open('tle_raw.txt').read())}
sen = {s.name: s for s in sats.values() if s.name.startswith('SENTINEL')}
rows=[]
for e in ev:
    if not (e['desc'] and e['sun_el']>5 and e['lit']): continue
    jd = F.jd_from_dt(e['utc'])
    rS,vS = sen[e['sentinel']].prop_jd(np.array([jd])); rO,vO = sats[e['cat']].prop_jd(np.array([jd]))
    rS,vS,rO,vO = rS[0],vS[0],rO[0],vO[0]
    rh=rS/np.linalg.norm(rS); vh=vS-vS.dot(rh)*rh; n=vh/np.linalg.norm(vh)
    vn=(vO-vS).dot(n); foot=e['range_km']*10/786.0  # metres per pixel at object
    rows.append((abs(vn), e, vn, foot, foot/abs(vn*1000)*1000))
rows.sort(key=lambda r:r[0])
for a,e,vn,foot,dw in rows[:15]:
    print(f"{e['utc']:%Y-%m-%d %H:%M:%S.%f}"[:-3], f"{e['sentinel']:<11} {e['obj']:<16} lat={e['lat']:7.3f} lon={e['lon']:8.3f} offnadir={e['off_nadir_deg']:+5.1f} range={e['range_km']:4.0f} pix@obj={foot:3.1f}m v_along={vn:+6.2f} dwell={dw:5.2f}ms sunEl={e['sun_el']:4.1f}")

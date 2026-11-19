import json
def size_m(o):
    n = o['obj']; c = o['cat']
    if c == '25544': return 100.
    if c == '48274': return 50.
    if n.startswith('STARLINK'):
        try: k = int(n.split('-')[1].split()[0])
        except Exception: k = 0
        return 30. if k >= 30000 else 10.
    if n.startswith('SPACEMOBILE') or n.startswith('BLUEBIRD'): return 12.
    if 'R/B' in n: return 8.
    return 5.
def score(e):
    L = size_m(e); p = e['pix_obj']
    ft = min(1.0, (L + p) / (abs(e['v_al']) * 1000 * T_LINE / 1000))
    fs = min(1.0, L / (2 * p))
    return ft * fs
t0 = time.time()
J1 = jd_from_dt(dt.datetime(2026, 11, 19, 9, 0, tzinfo=dt.timezone.utc)); J0 = J1 - 27 / 24
EV = find_crossings(sents, objs, J0, J1, step_s=30)
day = [e for e in EV if e['desc'] and e['sun_el'] > 8 and e['lit']]
for e in day: e['size'] = size_m(e); e['score'] = score(e)
day.sort(key=lambda e: -e['score'])
print(f'{len(EV)} crossings in FOV, {len(day)} daylit; search {time.time()-t0:.0f}s')
print('score>0.5:', sum(e['score'] > 0.5 for e in day), ' score>0.2:', sum(e['score'] > 0.2 for e in day))
for e in day[:12]:
    print(f"{dt_from_jd(e['jd']):%m-%d %H:%M:%S} {e['sent'][-2:]} {e['obj'][:16]:<16} lat={e['lat']:7.2f} lon={e['lon']:8.2f} rng={e['rng']:4.0f} v_al={e['v_al']:+6.2f} pix={e['pix_obj']:4.1f} L={e['size']:3.0f} score={e['score']:.2f} age={e['age']:+.2f}d")

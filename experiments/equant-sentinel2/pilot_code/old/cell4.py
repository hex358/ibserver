STAC = 'https://earth-search.aws.element84.com/v1/search'
async def stac(q):
    r = await pyfetch(STAC, method='POST', headers={'Content-Type': 'application/json'}, body=json.dumps(q))
    return json.loads(await r.string())
def epsg_of(it):
    p = it['properties']
    if 'proj:epsg' in p and p['proj:epsg']: return int(p['proj:epsg'])
    c = p.get('proj:code') or it['assets']['red'].get('proj:code')
    return int(str(c).split(':')[-1])
async def match(e):
    t = dt_from_jd(e['jd'])
    q = {'collections': ['sentinel-2-c1-l2a'], 'limit': 10,
         'intersects': {'type': 'Point', 'coordinates': [e['lon'], e['lat']]},
         'datetime': (t - dt.timedelta(minutes=4)).strftime('%Y-%m-%dT%H:%M:%SZ') + '/' + (t + dt.timedelta(minutes=4)).strftime('%Y-%m-%dT%H:%M:%SZ')}
    res = await stac(q)
    want = 'S2' + e['sent'][-1]
    for it in res.get('features', []):
        if it['id'].startswith(want):
            return it
    return None
CAND = []
for e in [x for x in day if x['score'] > 0.25][:80]:
    try: it = await match(e)
    except Exception as ex: it = None; print('stac err', ex)
    if it is None: continue
    e['item'] = it; e['cloud'] = it['properties'].get('eo:cloud_cover'); e['epsg'] = epsg_of(it)
    CAND.append(e)
print(len(CAND), 'events with a matching Sentinel-2 L2A tile')
for k, e in enumerate(CAND[:25]):
    print(f"#{k:2d} {dt_from_jd(e['jd']):%m-%d %H:%M:%S} {e['obj'][:16]:<16} {e['item']['id'][:24]} cloud={e['cloud']:5.1f}% lat={e['lat']:7.2f} lon={e['lon']:8.2f} v_al={e['v_al']:+6.2f} pix={e['pix_obj']:4.1f} L={e['size']:3.0f} score={e['score']:.2f}")

import matplotlib.pyplot as plt
async def reference(e):
    it = e['item']; t = dt_from_jd(e['jd'])
    q = {'collections': ['sentinel-2-c1-l2a'], 'limit': 30,
         'query': {'grid:code': {'eq': it['properties']['grid:code']}},
         'datetime': (t - dt.timedelta(days=40)).strftime('%Y-%m-%dT%H:%M:%SZ') + '/' + (t - dt.timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ')}
    res = await stac(q)
    fs = [f for f in res.get('features', []) if f['properties'].get('eo:cloud_cover', 100) < 15]
    fs.sort(key=lambda f: f['properties']['datetime'], reverse=True)
    return fs[0] if fs else None
def seg_dist(xx, yy, lx, ly, la):
    d = np.full(xx.shape, 1e12); a = np.zeros(xx.shape)
    for k in range(len(lx) - 1):
        x0, y0, x1, y1 = lx[k], ly[k], lx[k + 1], ly[k + 1]
        dx, dy = x1 - x0, y1 - y0; L2 = dx * dx + dy * dy
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / L2, 0, 1)
        d2 = (xx - x0 - t * dx) ** 2 + (yy - y0 - t * dy) ** 2
        upd = d2 < d; d[upd] = d2[upd]; a[upd] = (la[k] + t * (la[k + 1] - la[k]))[upd]
    return np.sqrt(d), a
async def analyze(e, half_w=2000):
    P = locus(e, AL)
    minx = int((P[:, 2].min() - half_w) // 10 * 10); maxx = int((P[:, 2].max() + half_w) // 10 * 10)
    miny = int((P[:, 3].min() - half_w) // 10 * 10); maxy = int((P[:, 3].max() + half_w) // 10 * 10)
    bb = (minx, miny, maxx, maxy); E = e['epsg']
    ref = await reference(e)
    if ref is None: return None
    jobs = [getnpy(e['item']['assets'][b]['href'], bb, E) for b in BANDS] + [getnpy(ref['assets'][b]['href'], bb, E) for b in BANDS]
    arrs = await asyncio.gather(*jobs)
    H, W = arrs[0].shape[1:]
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    lx = (P[:, 2] - minx) / 10; ly = (maxy - P[:, 3]) / 10
    dist, alf = seg_dist(xx, yy, lx, ly, P[:, 0])
    corr = dist * 10 < half_w
    Z = {}; EV = {}
    for i, b in enumerate(BANDS):
        ev = arrs[i][0].astype(float); rf = arrs[4 + i][0].astype(float)
        ok = (arrs[i][-1] > 0) & (arrs[4 + i][-1] > 0) & (ev > 0) & (rf > 0)
        g = np.median(ev[ok & corr] / rf[ok & corr])
        R = ev - g * rf
        R = R - median_filter(R, size=7)
        s = 1.4826 * np.median(np.abs(R[ok & corr]))
        Z[b] = np.where(ok & corr, R / s, 0); EV[b] = ev
    stack = np.stack([Z[b] for b in BANDS])
    cands = []
    for i, b in enumerate(BANDS):
        A = np.abs(Z[b]); others = np.delete(np.abs(stack), i, axis=0).max(0)
        pk = (A == maximum_filter(A, size=7)) & (A > 6) & (others < 0.4 * A)
        for y, x in zip(*np.where(pk)):
            cands.append(dict(band=b, x=int(x), y=int(y), z=float(Z[b][y, x]), oth=float(others[y, x]),
                              alpha=float(alf[y, x]), off=float(dist[y, x] * 10)))
    return dict(P=P, bb=bb, ref=ref['id'], Z=Z, EV=EV, cands=cands, lx=lx, ly=ly, shape=(H, W))
def triplets(c, tol=0.12):
    by = {b: [x for x in c if x['band'] == b] for b in BANDS}
    out = []
    for cb in by['blue']:
        for cg in by['green']:
            dg = cg['alpha'] - cb['alpha']
            if not (0.12 < abs(dg) < 0.40): continue
            for cr in by['red']:
                dr = cr['alpha'] - cb['alpha']
                if np.sign(dr) != np.sign(dg) or not (1.4 < dr / dg < 2.6): continue
                spread = max(abs(cb['off'] - cg['off']), abs(cb['off'] - cr['off']))
                out.append(dict(b=cb, g=cg, r=cr, dg=dg, dr=dr, spread=spread, zmin=min(abs(cb['z']), abs(cg['z']), abs(cr['z']))))
    out.sort(key=lambda t: -t['zmin'])
    return out
RES = {}
t0 = time.time()
R0 = await analyze(CAND[0]); RES[0] = R0
H, W = R0['shape']
print(f"RESULT6 window {W}x{H} px, ref {R0['ref']}, {time.time()-t0:.0f}s; single-band candidates per band:",
      {b: sum(c['band'] == b for c in R0['cands']) for b in BANDS})
for c in sorted(R0['cands'], key=lambda c: -abs(c['z']))[:15]:
    print(f"  {c['band']:<5} z={c['z']:+7.1f} others={c['oth']:5.1f} alpha={c['alpha']:+.3f} off_locus={c['off']:5.0f} m px=({c['x']},{c['y']})")
T0 = triplets(R0['cands'])
print('RESULT6T triplets B02-B03-B04 consistent with band timing:', len(T0))
for t in T0[:5]:
    print(f"  dg={t['dg']:+.3f} dr={t['dr']:+.3f} ratio={t['dr']/t['dg']:.2f} zmin={t['zmin']:.1f} spread={t['spread']:.0f} m  blue@({t['b']['x']},{t['b']['y']}) green@({t['g']['x']},{t['g']['y']}) red@({t['r']['x']},{t['r']['y']})")

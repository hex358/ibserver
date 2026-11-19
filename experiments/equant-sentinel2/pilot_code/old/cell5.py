import io, micropip
await micropip.install('scipy')
from scipy.ndimage import median_filter, maximum_filter
import matplotlib.pyplot as plt
SENT = {s.name: s for s in sents}; OBJ = {o.cat: o for o in objs}
BANDS = ['blue', 'nir', 'green', 'red']   # MSI acquisition order B02, B08, B03, B04
async def getnpy(href, bb, E):
    minx, miny, maxx, maxy = bb; W = int((maxx - minx) / 10); H = int((maxy - miny) / 10)
    u = f"https://titiler.xyz/cog/bbox/{minx},{miny},{maxx},{maxy}/{W}x{H}.npy?url={href}&coord_crs=epsg:{E}&dst_crs=epsg:{E}&resampling=nearest"
    r = await pyfetch(u)
    if r.status != 200: raise RuntimeError(f'titiler {r.status}')
    a = np.load(io.BytesIO(await r.bytes()))
    return a[0].astype(float), (a[-1] > 0 if a.shape[0] > 1 else np.ones(a.shape[1:], bool))
def locus(e, E, alphas):
    zone = E % 100; pts = []
    for al in alphas:
        tm, la, lo = apparent(SENT[e['sent']], OBJ[e['cat']], e['jd'], al)
        x, y = latlon_to_utm(la, lo, zone); pts.append((al, tm, x, y))
    return np.array(pts)
async def reference(e):
    it = e['item']; t = dt_from_jd(e['jd'])
    q = {'collections': ['sentinel-2-c1-l2a'], 'limit': 20,
         'query': {'grid:code': {'eq': it['properties']['grid:code']}, 'eo:cloud_cover': {'lt': 40}},
         'datetime': (t - dt.timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%SZ') + '/' + (t + dt.timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')}
    res = await stac(q)
    fs = [f for f in res.get('features', []) if f['id'] != it['id']]
    fs.sort(key=lambda f: f['properties']['eo:cloud_cover'])
    return fs[0] if fs else None
async def analyze(e, alphas=np.linspace(-1.0, 1.0, 41), margin=1500, show=True):
    E = e['epsg']; P = locus(e, E, alphas)
    minx = int((P[:, 2].min() - margin) // 10 * 10); maxx = int((P[:, 2].max() + margin) // 10 * 10)
    miny = int((P[:, 3].min() - margin) // 10 * 10); maxy = int((P[:, 3].max() + margin) // 10 * 10)
    bb = (minx, miny, maxx, maxy)
    ref = await reference(e)
    if ref is None: print('no reference scene'); return None
    res = {'bb': bb, 'ref': ref['id'], 'locus': P, 'bands': {}}
    H = int((maxy - miny) / 10); W = int((maxx - minx) / 10)
    yy, xx = np.mgrid[0:H, 0:W]
    lx = (P[:, 2] - minx) / 10; ly = (maxy - P[:, 3]) / 10
    # distance of every pixel to the locus polyline (dense sampling)
    dmin = np.full((H, W), 1e9); amin = np.zeros((H, W))
    for k in range(len(lx)):
        d2 = (xx - lx[k]) ** 2 + (yy - ly[k]) ** 2
        upd = d2 < dmin; dmin[upd] = d2[upd]; amin[upd] = P[k, 1]
    corridor = np.sqrt(dmin) < margin / 10
    for b in BANDS:
        ev, m1 = await getnpy(e['item']['assets'][b]['href'], bb, E)
        rf, m2 = await getnpy(ref['assets'][b]['href'], bb, E)
        ok = m1 & m2 & (ev > 0) & (rf > 0)
        g = np.median(ev[ok] / rf[ok])
        R = ev - g * rf
        R = R - median_filter(R, size=15)
        sig = 1.4826 * np.median(np.abs(R[ok & corridor] - np.median(R[ok & corridor])))
        Z = np.where(ok & corridor, R / sig, 0)
        res['bands'][b] = dict(ev=ev, Z=Z, sig=sig)
    # candidates: strongest compact spots per band inside the corridor
    cands = []
    for b in BANDS:
        Z = res['bands'][b]['Z']; A = np.abs(Z)
        pk = (A == maximum_filter(A, size=9)) & (A > 6)
        ys, xs = np.where(pk)
        for y, x in zip(ys, xs):
            other = [abs(res['bands'][o]['Z'][y, x]) for o in BANDS if o != b]
            cands.append(dict(band=b, x=int(x), y=int(y), z=float(Z[y, x]), other_max=float(max(other)),
                              t_locus=float(amin[y, x]), off_m=float(np.sqrt(dmin[y, x]) * 10)))
    cands.sort(key=lambda c: -abs(c['z']))
    res['cands'] = cands
    moving = [c for c in cands if abs(c['z']) > 8 and c['other_max'] < 0.35 * abs(c['z'])]
    res['moving'] = moving
    if show:
        fig, ax = plt.subplots(2, 4, figsize=(16, 8))
        for i, b in enumerate(BANDS):
            ev = res['bands'][b]['ev']; Z = res['bands'][b]['Z']
            lo, hi = np.percentile(ev[ev > 0], [2, 98])
            ax[0, i].imshow(ev, cmap='gray', vmin=lo, vmax=hi); ax[0, i].plot(lx, ly, 'c-', lw=0.6); ax[0, i].set_title(b + ' (event)')
            ax[1, i].imshow(np.clip(Z, -10, 10), cmap='RdBu_r', vmin=-10, vmax=10); ax[1, i].plot(lx, ly, 'k-', lw=0.4)
            for c in moving:
                if c['band'] == b: ax[1, i].plot(c['x'], c['y'], 'o', mfc='none', mec='lime', ms=14, mew=2)
            ax[1, i].set_title(f"{b} residual / sigma")
        for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{e['obj']}  {dt_from_jd(e['jd']):%Y-%m-%d %H:%M:%S} UTC  {e['item']['id']}  ref {ref['id'][:24]}", fontsize=10)
        plt.tight_layout(); plt.show()
    print(f"window {W}x{H} px, ref {ref['id']}, sigma per band:", {b: round(res['bands'][b]['sig'], 1) for b in BANDS})
    print('moving-object candidates (strong in one band only):')
    for c in moving[:8]:
        print(f"  {c['band']:<5} z={c['z']:+6.1f} others<= {c['other_max']:4.1f} px=({c['x']},{c['y']}) offset_from_locus={c['off_m']:5.0f} m  t={dt_from_jd(c['t_locus']):%H:%M:%S.%f}")
    return res

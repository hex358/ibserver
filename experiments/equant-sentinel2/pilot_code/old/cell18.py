async def wide_look(e, tag, half_km=14.0, res=20):
    DA, psi = await band_alphas2(e)
    P = dens(locus(e, np.linspace(-2.0, 2.2, 43), 0.0), n=400); nx, ny = normals(P)
    marks = []
    for b in ('blue', 'green', 'red'):
        for dd in DA.get(b, []):
            X, Y = xy_at(P, nx, ny, np.array([dd['a']]), np.array([0.0])); marks.append((b, dd['det'], float(X[0, 0]), float(Y[0, 0]), bool(inside(dd, X, Y)[0, 0])))
    cx = float(np.mean([m[2] for m in marks])); cy = float(np.mean([m[3] for m in marks]))
    hw = round(half_km * 1000 / res) * res; bb = (cx - hw, cy - hw, cx + hw, cy + hw)
    t = dt_from_jd(e['jd']); lt = e['sent'][-1]; d = 0.3
    poly = {'type': 'Polygon', 'coordinates': [[[e['lon'] - d, e['lat'] - d], [e['lon'] + d, e['lat'] - d], [e['lon'] + d, e['lat'] + d], [e['lon'] - d, e['lat'] + d], [e['lon'] - d, e['lat'] - d]]]}
    items = [f for f in await stac_post({'collections': ['sentinel-2-c1-l2a'], 'limit': 20, 'intersects': poly, 'datetime': f"{iso(t - dtm.timedelta(minutes=4))}/{iso(t + dtm.timedelta(minutes=4))}"}) if f['id'].startswith('S2' + lt)]
    refs = [await ref_same_orbit({'item': it}) for it in items]
    Zs = {}; EVs = {}
    for b in ('blue', 'green', 'red'):
        ev = None; rf = None
        for it, rr in zip(items, refs):
            a = await getnpy2(it['assets'][b]['href'], bb, e['epsg'], res)
            if a is None: continue
            m = a[-1] > 0
            if ev is None: ev = np.zeros(a[0].shape, np.float32); rf = np.zeros(a[0].shape, np.float32)
            put = m & (ev == 0); ev[put] = a[0][put]
            if rr is not None:
                q = await getnpy2(rr['assets'][b]['href'], bb, e['epsg'], res)
                if q is not None:
                    pq = put & (q[-1] > 0); rf[pq] = q[0][pq]
        EVs[b] = ev; v = (ev > 0) & (rf > 0)
        if v.mean() > 0.5:
            g = np.median(ev[v] / rf[v]); Rr = np.where(v, ev - g * rf, 0)
        else: Rr = ev - median_filter(ev, 15)
        Rr = Rr - median_filter(Rr, 7); Zs[b] = Rr / (1.4826 * np.median(np.abs(Rr - np.median(Rr))) + 1e-6)
    tr = lambda x, y: ((x - bb[0]) / res, (bb[3] - y) / res)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.6))
    rgb = np.stack([EVs['red'], EVs['green'], EVs['blue']], -1); lo, hi = np.percentile(rgb, [1, 99])
    ax[0].imshow(np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)); ax[0].set_title(f"{tag} {e['obj']} {e['item']['id'][:26]} ({res} m)")
    D = np.stack([np.abs(Zs['red']), np.abs(Zs['green']), np.abs(Zs['blue'])], -1)
    ax[1].imshow(np.clip(D / 12, 0, 1)); ax[1].set_title('|change| per band: R=B04 G=B03 B=B02')
    col = {'blue': 'cyan', 'green': 'lime', 'red': 'red'}
    for a in ax:
        px, py = tr(P[:, 2], P[:, 3]); a.plot(px, py, 'y--', lw=0.7)
        for b, det, x, y, ok in marks:
            u, w = tr(x, y); a.add_patch(plt.Circle((u, w), 1500 / res, fill=False, ec=col[b], lw=1.6 if ok else 0.6, ls='-' if ok else ':'))
            a.text(u + 30, w, f"{b[0].upper()}{det}", color=col[b], fontsize=8)
        a.set_xlim(0, rgb.shape[1]); a.set_ylim(rgb.shape[0], 0); a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.show()
    fig, ax = plt.subplots(2, 3, figsize=(12, 8)); k = 0
    for det in sorted(set(m[1] for m in marks)):
        for jb, b in enumerate(('blue', 'green', 'red')):
            mm = [m for m in marks if m[0] == b and m[1] == det]
            a = ax[min(k, 1), jb]
            if mm:
                u, w = tr(mm[0][2], mm[0][3]); u, w = int(u), int(w); h = int(2000 / res)
                Z = Zs[b][max(0, w - h):w + h, max(0, u - h):u + h]
                a.imshow(np.clip(Z, -10, 10), cmap='RdBu_r', vmin=-10, vmax=10); a.set_title(f"{b} det{det} {'IN' if mm[0][4] else 'out'} (+-2 km)", fontsize=9)
            a.set_xticks([]); a.set_yticks([])
        k += 1
        if k > 1: break
    plt.tight_layout(); plt.show()
    S = np.abs(Zs['red']) + np.abs(Zs['green']) + np.abs(Zs['blue'])
    peaks = []
    for b in ('blue', 'green', 'red'):
        Z = np.abs(Zs[b]); m = (maximum_filter(Z, 9) == Z) & (Z > 15)
        for i, j in np.argwhere(m): peaks.append((b, round(float(Zs[b][i, j]), 1), int(j), int(i)))
    peaks.sort(key=lambda t: -abs(t[1]))
    print('RESULT17', tag, 'items', [it['id'][:24] for it in items], 'marks', [(b, d, ok, round(tr(x, y)[0]), round(tr(x, y)[1])) for b, d, x, y, ok in marks], 'peaks', peaks[:15])
for e, tag in [(x, 'SL%d' % i) for i, x in enumerate(CAND)]:
    try: await wide_look(e, tag)
    except Exception: print('RESULT17 fail', tag, traceback.format_exc()[-300:].replace('\n', ' | '))
print('RESULT17 ALLDONE')

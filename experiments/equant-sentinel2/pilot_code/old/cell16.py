import xml.etree.ElementTree as ET
OMEGA_E = 7.2921158553e-5
def band_plane(rS, vS, alpha):
    rS = np.asarray(rS, float); vS = np.asarray(vS, float)
    vE = vS - OMEGA_E * np.array([-rS[1], rS[0], 0.0])
    rh = rS / np.linalg.norm(rS); n = fan_normal(rS, vE); ct = np.cross(rh, n)
    look = -rh * math.cos(alpha) + n * math.sin(alpha)
    return np.cross(look, ct)
async def band_alphas2(e, near_km=15):
    root = ET.fromstring(await gettext(e['item']['assets']['granule_metadata']['href']))
    for g in root.iter('Geoposition'):
        if g.get('resolution') == '10': ulx = float(g.find('ULX').text); uly = float(g.find('ULY').text)
    x0, y0 = latlon_to_utm(e['lat'], e['lon'], e['epsg'] % 100); sen = SENT[e['sent']]
    rS, _ = sen.prop_jd([e['jd']]); la1, lo1 = teme_to_geodetic(rS[0], e['jd'])
    rS2, _ = sen.prop_jd([e['jd'] + 1 / 86400]); la2, lo2 = teme_to_geodetic(rS2[0], e['jd'] + 1 / 86400)
    dlo = (lo2 - lo1 + 540) % 360 - 180; psi = math.atan2(dlo * math.cos(math.radians(la1)), la2 - la1)
    out = {}
    for grid in root.iter('Viewing_Incidence_Angles_Grids'):
        names = [k for k, v in BID.items() if v == grid.get('bandId')]
        if not names: continue
        rd = lambda tag: np.array([[float(v) for v in row.text.split()] for row in grid.find(tag).find('Values_List').findall('VALUES')])
        Zg, Ag = rd('Zenith'), rd('Azimuth')
        al = np.degrees(np.arctan(-np.tan(np.radians(Zg)) * np.cos(np.radians(Ag) - psi)))
        rr, cc = np.mgrid[0:Zg.shape[0], 0:Zg.shape[1]]
        near = np.isfinite(al) & (np.hypot(ulx + cc * 5000 - x0, uly - rr * 5000 - y0) < near_km * 1000)
        if near.sum() == 0: continue
        out.setdefault(names[0], []).append(dict(det=int(grid.get('detectorId')), a=float(np.nanmedian(al[near])), ok=np.isfinite(al), ulx=ulx, uly=uly))
    return out, math.degrees(psi)
def inside(dd, X, Y):
    r = np.rint((dd['uly'] - Y) / 5000).astype(int); c = np.rint((X - dd['ulx']) / 5000).astype(int)
    nr, nc = dd['ok'].shape; ins = (r >= 0) & (r < nr) & (c >= 0) & (c < nc)
    ins[ins] = dd['ok'][r[ins], c[ins]]
    return ins
def xy_at(P, nx, ny, a, C):
    AF = P[:, 0]
    return (np.interp(a, AF, P[:, 2])[:, None] + C[None, :] * np.interp(a, AF, nx)[:, None], np.interp(a, AF, P[:, 3])[:, None] + C[None, :] * np.interp(a, AF, ny)[:, None])
def plan_tiles2(loci, DA, bands, Wd, cmax):
    pts = []
    for P in loci.values():
        nx, ny = normals(P)
        for b in bands:
            for dd in DA[b]:
                a = np.arange(dd['a'] - Wd, dd['a'] + Wd + 1e-9, 0.01); X, Y = xy_at(P, nx, ny, a, np.array([-cmax, -cmax / 2, 0, cmax / 2, cmax]))
                m = inside(dd, X, Y); pts.append(np.c_[X[m], Y[m]])
    pts = np.vstack(pts)
    return sorted({(int(np.floor(x / GRID)), int(np.floor(y / GRID))) for x, y in pts})
def search4(loci, mos, DA, bands, Wd, cmax=1500, cstep=20, dstep=0.003):
    best = (-1e9, None); C = np.arange(-cmax, cmax + 1, cstep); D = np.arange(-Wd, Wd + 1e-9, dstep)
    for dM, P in loci.items():
        nx, ny = normals(P)
        for key in ('dark', 'bright'):
            sm = np.zeros((len(D), len(C)), np.float32); cnt = np.zeros_like(sm)
            for b in bands:
                Vb = np.full(sm.shape, np.nan, np.float32)
                for dd in DA[b]:
                    X, Y = xy_at(P, nx, ny, dd['a'] + D, C); V = (mos.get(key, b, X, Y) - mos.mu[key][b]) / mos.sd[key][b]
                    V[~inside(dd, X, Y)] = np.nan; Vb = np.fmax(Vb, V)
                okv = ~np.isnan(Vb); sm[okv] += np.minimum(Vb[okv], CLIP); cnt[okv] += 1
            S = np.where(cnt >= max(4, 0.6 * len(bands)), sm / np.sqrt(np.maximum(cnt, 1)), -1e9)
            i, j = np.unravel_index(np.argmax(S), S.shape)
            if S[i, j] > best[0]: best = (float(S[i, j]), dict(dM=dM, d=float(D[i]), c=float(C[j]), sign=key))
    return best
def best_positions(loci, DA, bands, info):
    P = loci[info['dM']]; nx, ny = normals(P); out = {}
    for b in bands:
        for dd in DA[b]:
            X, Y = xy_at(P, nx, ny, np.array([dd['a'] + info['d']]), np.array([info['c']]))
            if inside(dd, X, Y)[0, 0]: out[b] = (float(X[0, 0]), float(Y[0, 0]), dd['det']); break
    return out
RES4 = {}
async def run4(e, tag, dM_half=3000, dM_step=750, n_null=8, max_tiles=48):
    t0 = time.time()
    DA, psi = await band_alphas2(e)
    bands = [b for b in BID if b in DA and b in e['item']['assets']]
    if not all(b in bands for b in ('blue', 'green', 'red', 'nir')): print('RESULT16', tag, 'skip: no angles'); return
    allA = [dd['a'] for b in bands for dd in DA[b]]; cA = float(np.mean(allA))
    Pp = locus(e, np.array([cA - 0.5, cA + 0.5])); kmdeg = math.hypot(Pp[1, 2] - Pp[0, 2], Pp[1, 3] - Pp[0, 3]) / 1000
    Wd = float(np.clip(12 / kmdeg, 0.2, 0.8)); span = max(allA) - min(allA) + 2 * Wd + 0.1
    al = np.linspace(min(allA) - Wd - 0.05, max(allA) + Wd + 0.05, int(np.clip(span / 0.06, 25, 70)))
    loci = {float(dM): dens(locus(e, al, dM), n=500) for dM in np.arange(-dM_half, dM_half + 1, dM_step)}
    ref = await ref_same_orbit(e); cells = plan_tiles2(loci, DA, bands, Wd, 1500)
    dets = {b: [(dd['det'], round(dd['a'], 3)) for dd in DA[b]] for b in ('blue', 'red')}
    print(f"RESULT16 {tag} {e['obj']} {e['sent']} v={e['v_al']:.1f} rng={e['rng']:.0f} km/deg={kmdeg:.1f} Wd={Wd:.2f} dets={dets} ref={ref['id'][:30] if ref else None} tiles={len(cells)} loci {time.time() - t0:.0f}s")
    if len(cells) > max_tiles:
        P0 = loci[0.0] if 0.0 in loci else list(loci.values())[len(loci) // 2]; cx, cy = np.mean(P0[:, 2]), np.mean(P0[:, 3])
        cells = sorted(cells, key=lambda c: math.hypot((c[0] + 0.5) * GRID - cx, (c[1] + 0.5) * GRID - cy))[:max_tiles]
    tiles = []
    for cell in cells:
        T = await fetch_tile(e, ref, cell, bands)
        if T: tiles.append(T)
    if not tiles: print('RESULT16', tag, 'no tiles'); return
    modes = [m for T in tiles for m in T['modes']]; mos = Mosaic(tiles, bands)
    score, info = search4(loci, mos, DA, bands, Wd)
    rng = np.random.default_rng(11 + len(RES4)); nl = []
    for _ in range(n_null):
        p = rng.permutation(len(bands)); DAp = {b: DA[bands[p[i]]] for i, b in enumerate(bands)}
        nl.append(search4(loci, mos, DAp, bands, Wd)[0])
    s4 = search4(loci, mos, DA, ['blue', 'green', 'red', 'nir'], Wd)[0]
    pos = best_positions(loci, DA, bands, info) if info else {}
    vals = {b: float(mos.get(info['sign'], b, np.array([x]), np.array([y]))[0]) for b, (x, y, d) in pos.items()} if info else {}
    RES4[tag] = dict(e=e, DA=DA, bands=bands, score=score, info=info, nulls=nl, s4=s4, pos=pos, vals=vals, ref=ref, kmdeg=kmdeg, Wd=Wd, loci0=loci.get(0.0))
    print(f"RESULT16 {tag} done {time.time() - t0:.0f}s ref/self/empty={modes.count('ref')}/{modes.count('self')}/{modes.count('empty')} | TRUE={score:.2f} NULLmax={max(nl):.2f} med={np.median(nl):.2f} | 10m={s4:.2f} | {info} | " + ' '.join(f"{b}={v:.1f}" for b, v in vals.items()))
    del tiles, mos; gc.collect()
async def look4(tag, res=20, half_km=4.0):
    R = RES4[tag]; e = R['e']; pos = R['pos']
    if not pos: return
    xs = [v[0] for v in pos.values()]; ys = [v[1] for v in pos.values()]
    cx, cy = float(np.mean(xs)), float(np.mean(ys)); hw = round((max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + half_km * 1000) / res) * res
    bb = (cx - hw, cy - hw, cx + hw, cy + hw); Zs = {}; EVs = {}
    for b in ('blue', 'green', 'red'):
        a = await getnpy2(e['item']['assets'][b]['href'], bb, e['epsg'], res); ev = a[0].astype(np.float32); EVs[b] = ev
        r = await getnpy2(R['ref']['assets'][b]['href'], bb, e['epsg'], res) if R['ref'] else None
        if r is not None:
            rf = r[0].astype(np.float32); v = (ev > 0) & (rf > 0); g = np.median(ev[v] / rf[v]); Rr = ev - g * rf
        else: Rr = ev - median_filter(ev, 15)
        Rr = Rr - median_filter(Rr, 9); Zs[b] = Rr / (1.4826 * np.median(np.abs(Rr - np.median(Rr))) + 1e-6)
    tr = lambda x, y: ((x - bb[0]) / res, (bb[3] - y) / res)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    rgb = np.stack([EVs['red'], EVs['green'], EVs['blue']], -1); lo, hi = np.percentile(rgb, [1, 99])
    ax[0].imshow(np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)); ax[0].set_title(f"{tag} {e['obj']} {e['item']['id'][:24]} RGB")
    sgn = -1 if R['info']['sign'] == 'dark' else 1
    ax[1].imshow(np.stack([np.clip(sgn * Zs[b], 0, 10) for b in ('red', 'green', 'blue')], -1) / 10); ax[1].set_title(f"{R['info']['sign']} anomalies, R/G/B = band; TRUE {R['score']:.1f} vs NULL max {max(R['nulls']):.1f}")
    col = {'blue': 'cyan', 'green': 'lime', 'red': 'red', 'nir': 'white'}
    for a in ax:
        if R['loci0'] is not None:
            px, py = tr(R['loci0'][:, 2], R['loci0'][:, 3]); a.plot(px, py, 'y--', lw=0.6)
        for b in ('blue', 'green', 'red', 'nir'):
            if b in pos:
                u, w = tr(pos[b][0], pos[b][1]); a.add_patch(plt.Circle((u, w), 300 / res, fill=False, ec=col[b], lw=1.5))
        a.set_xlim(0, rgb.shape[1]); a.set_ylim(rgb.shape[0], 0); a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.show()
Q4 = [(x, 'ISS%d' % i, 15000, 1500) for i, x in enumerate(BEV)] + [(CAND[k], '#%d' % k, 3000, 750) for k in [0, 1, 2, 3, 8, 4, 5, 6, 7]]
for e, tag, dmh, dms in Q4:
    try: await run4(e, tag, dmh, dms)
    except Exception: print('RESULT16 fail', tag, traceback.format_exc()[-300:].replace('\n', ' | '))
ranked = sorted(RES4, key=lambda t: -(RES4[t]['score'] - max(RES4[t]['nulls'])))
print('RESULT16 RANK', [(t, round(RES4[t]['score'], 1), round(max(RES4[t]['nulls']), 1)) for t in ranked])
for t in ranked[:2]:
    try: await look4(t)
    except Exception: print('RESULT16 lookfail', t, traceback.format_exc()[-200:])
print('RESULT16 ALLDONE')

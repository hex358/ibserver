RLOC = 6371.0; R20 = 20.0; CLIPE = 8.0
def utm_to_latlon(x, y, zone, south):
    a = 6378137.0; f = 1 / 298.257223563; k0 = 0.9996; e2 = f * (2 - f); ep2 = e2 / (1 - e2)
    x = x - 500000.0; y = y - (10000000.0 if south else 0.0); M = y / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256)); e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    p1 = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu) + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu) + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
    N1 = a / math.sqrt(1 - e2 * math.sin(p1) ** 2); T1 = math.tan(p1) ** 2; C1 = ep2 * math.cos(p1) ** 2
    R1 = a * (1 - e2) / (1 - e2 * math.sin(p1) ** 2) ** 1.5; D = x / (N1 * k0)
    lat = p1 - (N1 * math.tan(p1) / R1) * (D * D / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * ep2) * D ** 4 / 24 + (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * ep2 - 3 * C1 * C1) * D ** 6 / 720)
    lon = (D - (1 + 2 * T1 + C1) * D ** 3 / 6 + (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * ep2 + 24 * T1 * T1) * D ** 5 / 120) / math.cos(p1)
    return math.degrees(lat), (zone - 1) * 6 - 180 + 3 + math.degrees(lon)
async def items_around(e, d=0.45):
    t = dt_from_jd(e['jd']); lt = e['sent'][-1]
    poly = {'type': 'Polygon', 'coordinates': [[[e['lon'] - d, e['lat'] - d], [e['lon'] + d, e['lat'] - d], [e['lon'] + d, e['lat'] + d], [e['lon'] - d, e['lat'] + d], [e['lon'] - d, e['lat'] - d]]]}
    its = [f for f in await stac_post({'collections': ['sentinel-2-c1-l2a'], 'limit': 30, 'intersects': poly, 'datetime': f"{iso(t - dtm.timedelta(minutes=4))}/{iso(t + dtm.timedelta(minutes=4))}"}) if f['id'].startswith('S2' + lt)]
    its.sort(key=lambda f: f['id'] != e['item']['id'])
    return its
async def band_grids(e, its):
    sen = SENT[e['sent']]; rS, _ = sen.prop_jd([e['jd']]); H = float(np.linalg.norm(rS[0])) - RLOC
    la1, lo1 = teme_to_geodetic(rS[0], e['jd']); rS2, _ = sen.prop_jd([e['jd'] + 1 / 86400]); la2, lo2 = teme_to_geodetic(rS2[0], e['jd'] + 1 / 86400)
    psi = math.atan2(((lo2 - lo1 + 540) % 360 - 180) * math.cos(math.radians(la1)), la2 - la1)
    out = {}
    for it in its:
        if epsg_of(it) != e['epsg']: continue
        try: root = ET.fromstring(await gettext(it['assets']['granule_metadata']['href']))
        except Exception: continue
        ulx = uly = None
        for g in root.iter('Geoposition'):
            if g.get('resolution') == '10': ulx = float(g.find('ULX').text); uly = float(g.find('ULY').text)
        for grid in root.iter('Viewing_Incidence_Angles_Grids'):
            names = [k for k, v in BID.items() if v == grid.get('bandId')]
            if not names: continue
            rd = lambda tag: np.array([[float(v) for v in row.text.split()] for row in grid.find(tag).find('Values_List').findall('VALUES')])
            Zg, Ag = rd('Zenith'), rd('Azimuth')
            eta = np.arcsin(RLOC * np.sin(np.radians(Zg)) / (RLOC + H))
            al = np.degrees(np.arctan(-np.tan(eta) * np.cos(np.radians(Ag) - psi)))
            out.setdefault(names[0], []).append(dict(det=int(grid.get('detectorId')), g=al, ulx=ulx, uly=uly, item=it['id']))
    return out, math.degrees(psi), H
def gval(dd, X, Y):
    g = dd['g']; nr, nc = g.shape; fr = (dd['uly'] - Y) / 5000.0; fc = (X - dd['ulx']) / 5000.0
    r0 = np.floor(fr).astype(int); c0 = np.floor(fc).astype(int); wr = fr - r0; wc = fc - c0
    ok = (r0 >= 0) & (c0 >= 0) & (r0 < nr - 1) & (c0 < nc - 1)
    acc = np.zeros(X.shape); wac = np.zeros(X.shape)
    for dr, dc, w in ((0, 0, (1 - wr) * (1 - wc)), (0, 1, (1 - wr) * wc), (1, 0, wr * (1 - wc)), (1, 1, wr * wc)):
        v = g[np.clip(r0 + dr, 0, nr - 1), np.clip(c0 + dc, 0, nc - 1)]; m = ok & np.isfinite(v)
        acc[m] += w[m] * v[m]; wac[m] += w[m]
    out = np.full(X.shape, np.nan); m = wac > 0.5; out[m] = acc[m] / wac[m]
    return out
def band_roots(P, DG, bands):
    A = P[:, 0]; X = P[:, 2]; Y = P[:, 3]; R = {}
    for b in bands:
        lst = []
        for dd in DG.get(b, []):
            f = A - gval(dd, X, Y)
            ok = np.isfinite(f[:-1]) & np.isfinite(f[1:]) & (np.sign(f[:-1]) != np.sign(f[1:]))
            for i in np.where(ok)[0]:
                w = f[i] / (f[i] - f[i + 1]); x = X[i] + w * (X[i + 1] - X[i]); y = Y[i] + w * (Y[i + 1] - Y[i])
                if all(math.hypot(x - q[2], y - q[3]) > 200 for q in lst): lst.append((dd['det'], float(A[i] + w * (A[i + 1] - A[i])), float(x), float(y)))
        if lst: R[b] = lst
    return R
def block2(a): return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean(axis=(1, 3))
async def fetch_cell2(cell, cands, bands, E, Arep, thr):
    i, j = cell; bb = (i * GRID - PAD, j * GRID - PAD, (i + 1) * GRID + PAD, (j + 1) * GRID + PAD)
    best = None
    for it, rf in cands[:3]:
        a = await getnpy2(it['assets']['blue']['href'], bb, E, 10)
        if a is None: continue
        cov = float((a[-1] > 0).mean())
        if best is None or cov > best[2]: best = (it, rf, cov, a)
        if cov > 0.97: break
    if best is None or best[2] < 0.2: return None
    it, rf, cov, a_blue = best; Z = {}; RGB = {}
    for b in bands:
        res = BRES[b]
        a = a_blue if b == 'blue' else await getnpy2(it['assets'][b]['href'], bb, E, res)
        r = await getnpy2(rf['assets'][b]['href'], bb, E, res) if (rf is not None and b in rf['assets']) else None
        if a is None: return None
        ev = a[0].astype(np.float32); evm = a[-1] > 0
        rv = r[0].astype(np.float32) if r is not None else None; rm = (r[-1] > 0) if r is not None else None
        if res == 10:
            ev = block2(ev); evm = block2(evm.astype(np.float32)) > 0.99
            if rv is not None: rv = block2(rv); rm = block2(rm.astype(np.float32)) > 0.99
        z, m = zmap(ev, evm, rv, rm, 7)
        Z[b] = z if z is not None else np.full(ev.shape, np.nan, np.float32)
        if b in ('red', 'green', 'blue'): RGB[b] = ev
    AZ = {b: np.abs(np.nan_to_num(Z[b])) for b in bands}; Ex = {}
    for b in bands:
        far = [c for c in bands if abs(Arep[c] - Arep[b]) > thr]
        F = maximum_filter(np.max(np.stack([AZ[c] for c in far]), 0), size=7) if far else 0.0
        ex = maximum_filter(np.maximum(0.0, AZ[b] - F), size=3).astype(np.float32); ex[np.isnan(Z[b])] = np.nan; Ex[b] = ex.astype(np.float16)
    return dict(cell=cell, bb=bb, E=Ex, Z={b: Z[b].astype(np.float16) for b in ('blue', 'green', 'red', 'nir')}, RGB=np.stack([RGB['red'], RGB['green'], RGB['blue']], -1), item=it['id'], cov=cov)
class Mos2:
    def __init__(s, tiles, bands):
        ci = [T['cell'][0] for T in tiles]; cj = [T['cell'][1] for T in tiles]
        s.i0, s.j0 = min(ci), min(cj); s.M = -np.ones((max(ci) - s.i0 + 1, max(cj) - s.j0 + 1), int)
        for k, T in enumerate(tiles): s.M[T['cell'][0] - s.i0, T['cell'][1] - s.j0] = k
        s.st = {b: np.stack([T['E'][b] for T in tiles]) for b in bands}
    def get(s, b, X, Y):
        I = np.floor(X / GRID).astype(int) - s.i0; J = np.floor(Y / GRID).astype(int) - s.j0
        ok = (I >= 0) & (I < s.M.shape[0]) & (J >= 0) & (J < s.M.shape[1]); t = np.full(X.shape, -1); t[ok] = s.M[I[ok], J[ok]]; ok &= t >= 0
        A = s.st[b]; H, W = A.shape[1:]
        col = np.floor((X - ((I + s.i0) * GRID - PAD)) / R20).astype(int); row = np.floor((((J + s.j0) + 1) * GRID + PAD - Y) / R20).astype(int)
        ok &= (col >= 0) & (col < W) & (row >= 0) & (row < H)
        V = np.full(X.shape, np.nan, np.float32); V[ok] = A[t[ok], row[ok], col[ok]]
        return V
def shift_search(roots, mos, bands, Rm, step=20.0, disp=None):
    S = np.arange(-Rm, Rm + 1e-6, step); SX, SY = np.meshgrid(S, S); tot = np.zeros(SX.shape, np.float32); cnt = np.zeros_like(tot); hit = np.zeros_like(tot)
    for b in bands:
        if b not in roots: continue
        dx, dy = disp[b] if disp else (0.0, 0.0); Vb = np.full(SX.shape, np.nan, np.float32)
        for (det, a, x, y) in roots[b]: Vb = np.fmax(Vb, mos.get(b, x + dx + SX, y + dy + SY))
        ok = ~np.isnan(Vb); tot[ok] += np.minimum(Vb[ok], CLIPE); cnt[ok] += 1; hit[ok] += Vb[ok] > 4
    sc = np.where(cnt >= max(4, 0.6 * len(roots)), tot / np.sqrt(np.maximum(cnt, 1)), -1e9)
    i, j = np.unravel_index(np.argmax(sc), sc.shape)
    return float(sc[i, j]), float(SX[i, j]), float(SY[i, j]), int(hit[i, j]), int(cnt[i, j])
RES5 = {}
async def run5(e, tag, Rm=4000.0, n_null=10):
    t0 = time.time(); its = await items_around(e)
    DG, psi, H = await band_grids(e, its)
    bands = [b for b in BID if b in DG and b in e['item']['assets']]
    P = dens(locus(e, np.linspace(-2.2, 2.2, 89), 0.0), n=1500)
    roots = band_roots(P, DG, bands); bands = [b for b in bands if b in roots]
    Arep = {b: float(np.median([q[1] for q in roots[b]])) for b in bands}
    kmdeg = float(np.median(np.hypot(np.diff(P[:, 2]), np.diff(P[:, 3])) / np.diff(P[:, 0]))) / 1000
    thr = max(0.03, 0.6 / kmdeg)
    print(f"RESULT19 {tag} {e['obj']} v={e['v_al']:.1f} rng={e['rng']:.0f} H={H:.0f} km/deg={kmdeg:.1f} items={len(its)} roots=" + ' '.join(f"{b}:{[(q[0], round(q[1], 3)) for q in roots[b]]}" for b in ('blue', 'nir', 'green', 'red') if b in roots) + f" ({time.time() - t0:.0f}s)")
    if len(bands) < 4: print('RESULT19', tag, 'too few bands with roots'); return
    zone = e['epsg'] % 100; south = e['epsg'] // 100 == 327
    cells = set()
    for b in bands:
        for (det, a, x, y) in roots[b]:
            for xx in np.arange(x - Rm - 400, x + Rm + 401, 2500):
                for yy in np.arange(y - Rm - 400, y + Rm + 401, 2500): cells.add((int(np.floor(xx / GRID)), int(np.floor(yy / GRID))))
    cells = sorted(cells)
    refs = {}
    for it in its: refs[it['id']] = await ref_same_orbit({'item': it})
    tiles = []
    for c in cells:
        la, lo = utm_to_latlon((c[0] + 0.5) * GRID, (c[1] + 0.5) * GRID, zone, south)
        cands = [(it, refs[it['id']]) for it in its if it['bbox'][0] <= lo <= it['bbox'][2] and it['bbox'][1] <= la <= it['bbox'][3]]
        cands.sort(key=lambda p: math.hypot(lo - (p[0]['bbox'][0] + p[0]['bbox'][2]) / 2, la - (p[0]['bbox'][1] + p[0]['bbox'][3]) / 2))
        try: T = await fetch_cell2(c, cands, bands, e['epsg'], Arep, thr)
        except Exception: T = None
        if T: tiles.append(T)
    print(f"RESULT19 {tag} cells {len(tiles)}/{len(cells)} fetched ({time.time() - t0:.0f}s)")
    if not tiles: return
    mos = Mos2(tiles, bands)
    sc, sx, sy, hit, cnt = shift_search(roots, mos, bands, Rm)
    rng = np.random.default_rng(5); nl = []
    for _ in range(n_null):
        disp = {}
        for b in bands:
            r = rng.uniform(1000, 4000); th = rng.uniform(0, 2 * math.pi); disp[b] = (r * math.cos(th), r * math.sin(th))
        nl.append(shift_search(roots, mos, bands, Rm, disp=disp)[0])
    per = {}
    for b in bands:
        per[b] = max(float(np.nan_to_num(mos.get(b, np.array([x + sx]), np.array([y + sy]))[0], nan=-1)) for (d, a, x, y) in roots[b])
    crops = {}
    for b in ('blue', 'green', 'red', 'nir'):
        if b not in roots: continue
        x, y = roots[b][0][2] + sx, roots[b][0][3] + sy
        for T in tiles:
            minx, miny, maxx, maxy = T['bb']; col = int((x - minx) // R20); row = int((maxy - y) // R20)
            if 0 <= col < 300 and 0 <= row < 300:
                sl = (slice(max(0, row - 30), row + 30), slice(max(0, col - 30), col + 30))
                crops[b] = dict(Z=T['Z'][b][sl].astype(np.float32), RGB=T['RGB'][sl].astype(np.float32), E=T['E'][b][sl].astype(np.float32)); break
    RES5[tag] = dict(e=e, roots=roots, bands=bands, score=sc, shift=(sx, sy), hit=hit, cnt=cnt, nulls=nl, per=per, crops=crops, kmdeg=kmdeg)
    print(f"RESULT19 {tag} TRUE={sc:.2f} (bands>4sigma {hit}/{cnt}) NULLmax={max(nl):.2f} med={np.median(nl):.2f} shift=({sx:.0f},{sy:.0f}) m | " + ' '.join(f"{b}={v:.1f}" for b, v in per.items()) + f" ({time.time() - t0:.0f}s)")
    del tiles, mos; gc.collect()
def plot5(tag):
    R = RES5[tag]; cr = R['crops']; fig, ax = plt.subplots(3, 4, figsize=(12, 9))
    for i, b in enumerate(('blue', 'nir', 'green', 'red')):
        if b not in cr: continue
        rgb = cr[b]['RGB']; lo, hi = np.percentile(rgb, [1, 99])
        ax[0, i].imshow(np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)); ax[0, i].set_title(f"{b} predicted spot, RGB", fontsize=9)
        ax[1, i].imshow(np.nan_to_num(cr[b]['Z']), cmap='RdBu_r', vmin=-10, vmax=10); ax[1, i].set_title(f"{b} change z", fontsize=9)
        ax[2, i].imshow(np.nan_to_num(cr[b]['E']), cmap='magma', vmin=0, vmax=10); ax[2, i].set_title(f"{b} band-exclusive, {R['per'][b]:.1f}", fontsize=9)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([]); a.plot([30], [30], 'c+', ms=14)
    fig.suptitle(f"{tag} {R['e']['obj']}: TRUE {R['score']:.1f} vs NULL max {max(R['nulls']):.1f}  (crops 1.2 km, 20 m px)"); plt.tight_layout(); plt.show()
for e, tag, Rm in [(CAND[0], 'SL0', 4000.0), (BEV[0], 'ISS', 6000.0), (CAND[1], 'SL1', 4000.0)]:
    try: await run5(e, tag, Rm)
    except Exception: print('RESULT19 fail', tag, traceback.format_exc()[-400:].replace('\n', ' | '))
for t in RES5: plot5(t)
print('RESULT19 ALLDONE')

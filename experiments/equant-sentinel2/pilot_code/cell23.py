def shift_search(roots, mos, bands, Rm, step=20.0, disp=None):
    S = np.arange(-Rm, Rm + 1e-6, step); SX, SY = np.meshgrid(S, S); tot = np.zeros(SX.shape, np.float32); cnt = np.zeros_like(tot); hit = np.zeros_like(tot)
    for b in bands:
        if b not in roots: continue
        dx, dy = disp[b] if disp else (0.0, 0.0); Vb = np.full(SX.shape, np.nan, np.float32)
        for (det, a, x, y) in roots[b]: Vb = np.fmax(Vb, mos.get(b, x + dx + SX, y + dy + SY))
        ok = ~np.isnan(Vb); tot[ok] += np.minimum(Vb[ok], CLIPE); cnt[ok] += 1; hit[ok] += Vb[ok] > 4
    sc = np.where(cnt >= max(3, 0.6 * len([b for b in bands if b in roots])), tot / np.sqrt(np.maximum(cnt, 1)), -1e9)
    i, j = np.unravel_index(np.argmax(sc), sc.shape)
    return float(sc[i, j]), float(SX[i, j]), float(SY[i, j]), int(hit[i, j]), int(cnt[i, j])
DIAG = {}
async def getnpy3(href, bb, E, res):
    minx, miny, maxx, maxy = bb; W = int(round((maxx - minx) / res)); H = int(round((maxy - miny) / res))
    u = f"https://titiler.xyz/cog/bbox/{minx},{miny},{maxx},{maxy}/{W}x{H}.npy?url={href}&coord_crs=epsg:{E}&dst_crs=epsg:{E}"
    last = None
    for t in range(3):
        try:
            r = await pyfetch(u)
            if r.status == 200: return np.load(io.BytesIO(await r.bytes())), 200
            last = r.status
            if r.status in (404, 422): return None, last
        except Exception as ex:
            last = 'exc:' + type(ex).__name__
        await asyncio.sleep(2.0 * (t + 1))
    return None, last
async def fetch_cell3(cell, cands, bands, E, Arep, thr):
    i, j = cell; bb = (i * GRID - PAD, j * GRID - PAD, (i + 1) * GRID + PAD, (j + 1) * GRID + PAD)
    best = None; why = []
    for it, rf in cands[:3]:
        a, st = await getnpy3(it['assets']['blue']['href'], bb, E, 10)
        if a is None: why.append(('blue', st)); continue
        cov = float((a[-1] > 0).mean())
        if best is None or cov > best[2]: best = (it, rf, cov, a)
        if cov > 0.97: break
    if best is None: return None, ('nofetch', why, len(cands))
    if best[2] < 0.05: return None, ('nodata', round(best[2], 3))
    it, rf, cov, a_blue = best; Z = {}; RGB = {}
    for b in bands:
        res = BRES[b]; n = int(round((bb[2] - bb[0]) / 20))
        a = a_blue if b == 'blue' else (await getnpy3(it['assets'][b]['href'], bb, E, res))[0]
        r = (await getnpy3(rf['assets'][b]['href'], bb, E, res))[0] if (rf is not None and b in rf['assets']) else None
        if a is None: Z[b] = np.full((n, n), np.nan, np.float32); continue
        ev = a[0].astype(np.float32); evm = a[-1] > 0
        rv = r[0].astype(np.float32) if r is not None else None; rm = (r[-1] > 0) if r is not None else None
        if res == 10:
            ev = block2(ev); evm = block2(evm.astype(np.float32)) > 0.99
            if rv is not None: rv = block2(rv); rm = block2(rm.astype(np.float32)) > 0.99
        z, m = zmap(ev, evm, rv, rm, 7)
        Z[b] = z if z is not None else np.full(ev.shape, np.nan, np.float32)
        if b in ('red', 'green', 'blue'): RGB[b] = ev
    for b in ('red', 'green', 'blue'):
        if b not in RGB: RGB[b] = np.zeros(Z['blue'].shape, np.float32)
    AZ = {b: np.abs(np.nan_to_num(Z[b])) for b in bands}; Ex = {}
    for b in bands:
        far = [c for c in bands if abs(Arep[c] - Arep[b]) > thr]
        F = maximum_filter(np.max(np.stack([AZ[c] for c in far]), 0), size=7) if far else 0.0
        ex = maximum_filter(np.maximum(0.0, AZ[b] - F), size=3).astype(np.float32); ex[np.isnan(Z[b])] = np.nan; Ex[b] = ex.astype(np.float16)
    return dict(cell=cell, bb=bb, E=Ex, Z={b: Z[b].astype(np.float16) for b in ('blue', 'green', 'red', 'nir') if b in Z}, RGB=np.stack([RGB['red'], RGB['green'], RGB['blue']], -1), item=it['id'], cov=cov), 'ok'
async def run6(e, tag, Rm=4000.0, n_null=10, minb=3):
    t0 = time.time(); its = await items_around(e, d=0.6)
    DG, psi, H = await band_grids(e, its)
    bands = [b for b in BID if b in DG and b in e['item']['assets']]
    P = dens(locus(e, np.linspace(-2.6, 2.6, 105), 0.0), n=2000)
    roots = band_roots(P, DG, bands); bands = [b for b in bands if b in roots]
    if len(bands) < minb: print('RESULT23', tag, 'too few bands with roots', list(roots)); return
    Arep = {b: float(np.median([q[1] for q in roots[b]])) for b in bands}
    kmdeg = float(np.median(np.hypot(np.diff(P[:, 2]), np.diff(P[:, 3])) / np.diff(P[:, 0]))) / 1000
    thr = max(0.03, 0.6 / kmdeg)
    zone = e['epsg'] % 100; south = e['epsg'] // 100 == 327
    cells = set()
    for b in bands:
        for (det, a, x, y) in roots[b]:
            for xx in np.arange(x - Rm - 400, x + Rm + 401, 2500):
                for yy in np.arange(y - Rm - 400, y + Rm + 401, 2500): cells.add((int(np.floor(xx / GRID)), int(np.floor(yy / GRID))))
    cells = sorted(cells); refs = {}
    for it in its: refs[it['id']] = await ref_same_orbit({'item': it})
    tiles = []; reasons = []
    for c in cells:
        la, lo = utm_to_latlon((c[0] + 0.5) * GRID, (c[1] + 0.5) * GRID, zone, south)
        cands = [(it, refs[it['id']]) for it in its if it['bbox'][0] - 0.05 <= lo <= it['bbox'][2] + 0.05 and it['bbox'][1] - 0.05 <= la <= it['bbox'][3] + 0.05]
        cands.sort(key=lambda p: math.hypot(lo - (p[0]['bbox'][0] + p[0]['bbox'][2]) / 2, la - (p[0]['bbox'][1] + p[0]['bbox'][3]) / 2))
        try: T, why = await fetch_cell3(c, cands, bands, e['epsg'], Arep, thr)
        except Exception as ex: T, why = None, 'exc ' + repr(ex)[:80]
        if T: tiles.append(T)
        else: reasons.append(why)
    print(f"RESULT23 {tag} {e['obj']} v={e['v_al']:.1f} km/deg={kmdeg:.1f} bands={bands} cells {len(tiles)}/{len(cells)} ({time.time() - t0:.0f}s) fail-reasons {reasons[:4]}")
    DIAG[tag] = reasons
    if len(tiles) < 2: return
    mos = Mos2(tiles, bands)
    sc, sx, sy, hit, cnt = shift_search(roots, mos, bands, Rm)
    rng = np.random.default_rng(5); nl = []
    for _ in range(n_null):
        disp = {}
        for b in bands:
            r = rng.uniform(1000, 4000); th = rng.uniform(0, 2 * math.pi); disp[b] = (r * math.cos(th), r * math.sin(th))
        nl.append(shift_search(roots, mos, bands, Rm, disp=disp)[0])
    per = {b: max(float(np.nan_to_num(mos.get(b, np.array([x + sx]), np.array([y + sy]))[0], nan=-1)) for (d, a, x, y) in roots[b]) for b in bands}
    crops = {}
    for b in ('blue', 'green', 'red', 'nir'):
        if b not in roots: continue
        x, y = roots[b][0][2] + sx, roots[b][0][3] + sy
        for T in tiles:
            minx, miny, maxx, maxy = T['bb']; col = int((x - minx) // R20); row = int((maxy - y) // R20)
            if 0 <= col < 300 and 0 <= row < 300 and b in T['Z']:
                sl = (slice(max(0, row - 30), row + 30), slice(max(0, col - 30), col + 30))
                crops[b] = dict(Z=T['Z'][b][sl].astype(np.float32), RGB=T['RGB'][sl].astype(np.float32), E=T['E'][b][sl].astype(np.float32)); break
    RES5[tag] = dict(e=e, roots=roots, bands=bands, score=sc, shift=(sx, sy), hit=hit, cnt=cnt, nulls=nl, per=per, crops=crops, kmdeg=kmdeg)
    print(f"RESULT23 {tag} TRUE={sc:.2f} (bands>4sigma {hit}/{cnt}) NULLmax={max(nl):.2f} med={np.median(nl):.2f} shift=({sx:.0f},{sy:.0f}) m | " + ' '.join(f"{b}={v:.1f}" for b, v in per.items()) + f" ({time.time() - t0:.0f}s)")
    del tiles, mos; gc.collect()
    plot5(tag)
for i, x in enumerate(CAND):
    try: await run6(x, 'SL%d' % i)
    except Exception: print('RESULT23 fail', i, traceback.format_exc()[-400:].replace('\n', ' | '))
print('RESULT23 ALLDONE')

import json, asyncio, time, io, gc, datetime as dtm, traceback
import matplotlib.pyplot as plt
from pyodide.http import pyfetch
from scipy.ndimage import median_filter, maximum_filter, minimum_filter
BID = {'blue': '1', 'green': '2', 'red': '3', 'rededge1': '4', 'rededge2': '5', 'rededge3': '6', 'nir': '7', 'nir08': '8', 'swir16': '11', 'swir22': '12'}
BRES = {'blue': 10, 'green': 10, 'red': 10, 'nir': 10, 'rededge1': 20, 'rededge2': 20, 'rededge3': 20, 'nir08': 20, 'swir16': 20, 'swir22': 20}
GRID, PAD, CLIP = 5000.0, 500.0, 5.0
async def stac_post(q):
    for t in range(3):
        try:
            r = await pyfetch('https://earth-search.aws.element84.com/v1/search', method='POST', headers={'Content-Type': 'application/json'}, body=json.dumps(q))
            return (await r.json()).get('features', [])
        except Exception:
            await asyncio.sleep(2)
    return []
def iso(t): return t.strftime('%Y-%m-%dT%H:%M:%SZ')
async def ref_same_orbit(e):
    pr = e['item']['properties']; t = dtm.datetime.fromisoformat(pr['datetime'].replace('Z', '+00:00'))
    qq = {'grid:code': {'eq': pr['grid:code']}, 'eo:cloud_cover': {'lt': 40}}
    if 'sat:relative_orbit' in pr: qq['sat:relative_orbit'] = {'eq': pr['sat:relative_orbit']}
    feats = await stac_post({'collections': ['sentinel-2-c1-l2a'], 'limit': 50, 'datetime': f"{iso(t - dtm.timedelta(days=75))}/{iso(t - dtm.timedelta(hours=12))}", 'query': qq})
    if not feats: return None
    good = [f for f in feats if f['properties'].get('eo:cloud_cover', 100) < 8]
    pool = good if good else feats
    pool.sort(key=(lambda f: f['properties']['datetime']) if good else (lambda f: -f['properties'].get('eo:cloud_cover', 100)))
    return pool[-1]
async def getnpy2(href, bb, E, res):
    minx, miny, maxx, maxy = bb; W = int(round((maxx - minx) / res)); H = int(round((maxy - miny) / res))
    u = f"https://titiler.xyz/cog/bbox/{minx},{miny},{maxx},{maxy}/{W}x{H}.npy?url={href}&coord_crs=epsg:{E}&dst_crs=epsg:{E}"
    for t in range(4):
        try:
            r = await pyfetch(u)
            if r.status == 200: return np.load(io.BytesIO(await r.bytes()))
        except Exception:
            pass
        await asyncio.sleep(1.5 * (t + 1))
    return None
def zmap(ev, evm, rf, rfm, k):
    v = evm & (ev > 0); mode = 'self'; Rr = ev
    if rf is not None:
        vr = v & rfm & (rf > 0)
        if vr.mean() > 0.6:
            g = np.median(ev[vr] / rf[vr]); Rr = ev - g * rf; v = vr; mode = 'ref'
    Rr = Rr - median_filter(Rr, k)
    if v.sum() < 300: return None, 'empty'
    md = np.median(Rr[v]); sg = 1.4826 * np.median(np.abs(Rr[v] - md)) + 1e-6
    Z = ((Rr - md) / sg).astype(np.float32); Z[~v] = np.nan
    return Z, mode
def dens(P, n=240):
    a = np.linspace(P[0, 0], P[-1, 0], n)
    return np.c_[a, np.interp(a, P[:, 0], P[:, 1]), np.interp(a, P[:, 0], P[:, 2]), np.interp(a, P[:, 0], P[:, 3])]
def normals(P):
    tx, ty = np.gradient(P[:, 2]), np.gradient(P[:, 3]); nn = np.hypot(tx, ty) + 1e-9
    return -ty / nn, tx / nn
def plan_tiles(loci, cmax):
    pts = []
    for P in loci.values():
        nx, ny = normals(P)
        for c in (-cmax, -cmax / 2, 0, cmax / 2, cmax): pts.append(np.c_[P[:, 2] + c * nx, P[:, 3] + c * ny])
    pts = np.vstack(pts)
    return sorted({(int(np.floor(x / GRID)), int(np.floor(y / GRID))) for x, y in pts})
async def fetch_tile(e, ref, cell, bands):
    i, j = cell; bb = (i * GRID - PAD, j * GRID - PAD, (i + 1) * GRID + PAD, (j + 1) * GRID + PAD); E = e['epsg']
    jobs = []
    for b in bands:
        jobs.append(getnpy2(e['item']['assets'][b]['href'], bb, E, BRES[b]))
        jobs.append(getnpy2(ref['assets'][b]['href'], bb, E, BRES[b]) if ref is not None and b in ref['assets'] else asyncio.sleep(0))
    out = []
    for s in range(0, len(jobs), 6): out += await asyncio.gather(*jobs[s:s + 6])
    T = dict(cell=cell, bb=bb, dark={}, bright={}, Z={}, RGB=None, modes=[])
    rgb = {}
    for n, b in enumerate(bands):
        a, r = out[2 * n], out[2 * n + 1]
        if a is None: return None
        ev = a[0].astype(np.float32); evm = a[-1] > 0
        rf = r[0].astype(np.float32) if isinstance(r, np.ndarray) else None; rfm = (r[-1] > 0) if isinstance(r, np.ndarray) else None
        Z, m = zmap(ev, evm, rf, rfm, 9 if BRES[b] == 10 else 7); T['modes'].append(m)
        if Z is None: Z = np.full(ev.shape, np.nan, np.float32)
        Zf = np.nan_to_num(Z, nan=0.0)
        T['dark'][b] = np.where(np.isnan(Z), np.nan, -minimum_filter(Zf, size=3)).astype(np.float16)
        T['bright'][b] = np.where(np.isnan(Z), np.nan, maximum_filter(Zf, size=3)).astype(np.float16)
        if b in ('blue', 'green', 'red', 'nir'): T['Z'][b] = Z.astype(np.float16)
        if b in ('red', 'green', 'blue'): rgb[b] = a[0]
    T['RGB'] = np.stack([rgb['red'], rgb['green'], rgb['blue']], -1)
    return T
class Mosaic:
    def __init__(s, tiles, bands):
        ci = np.array([T['cell'][0] for T in tiles]); cj = np.array([T['cell'][1] for T in tiles])
        s.i0, s.j0 = ci.min(), cj.min(); s.M = -np.ones((ci.max() - s.i0 + 1, cj.max() - s.j0 + 1), int)
        for k, T in enumerate(tiles): s.M[T['cell'][0] - s.i0, T['cell'][1] - s.j0] = k
        s.st = {key: {b: np.stack([T[key][b] for T in tiles]) for b in bands} for key in ('dark', 'bright')}
        s.mu = {key: {b: float(np.nanmean(s.st[key][b][:, ::3, ::3].astype(np.float32))) for b in bands} for key in s.st}
        s.sd = {key: {b: float(np.nanstd(s.st[key][b][:, ::3, ::3].astype(np.float32))) + 1e-6 for b in bands} for key in s.st}
    def get(s, key, b, X, Y):
        res = BRES[b]; I = np.floor(X / GRID).astype(int) - s.i0; J = np.floor(Y / GRID).astype(int) - s.j0
        ok = (I >= 0) & (I < s.M.shape[0]) & (J >= 0) & (J < s.M.shape[1]); t = np.full(X.shape, -1)
        t[ok] = s.M[I[ok], J[ok]]; ok &= t >= 0
        A = s.st[key][b]; H, W = A.shape[1:]
        col = np.floor((X - ((I + s.i0) * GRID - PAD)) / res).astype(int); row = np.floor((((J + s.j0) + 1) * GRID + PAD - Y) / res).astype(int)
        ok &= (col >= 0) & (col < W) & (row >= 0) & (row < H)
        V = np.full(X.shape, np.nan, np.float32); V[ok] = A[t[ok], row[ok], col[ok]]
        return V
def search(loci, mos, A, bands, cmax=1500, cstep=20, dstep=0.003):
    best = (-1e9, None); C = np.arange(-cmax, cmax + 1, cstep)
    for dM, P in loci.items():
        AF, PX, PY = P[:, 0], P[:, 2], P[:, 3]; nx, ny = normals(P)
        d0 = AF[0] - min(A[b] for b in bands); d1 = AF[-1] - max(A[b] for b in bands)
        if d1 <= d0: continue
        D = np.arange(d0, d1, dstep); acc = {}
        for key in ('dark', 'bright'):
            sm = np.zeros((len(D), len(C)), np.float32); cnt = np.zeros_like(sm)
            for b in bands:
                a = A[b] + D
                X = np.interp(a, AF, PX)[:, None] + C[None, :] * np.interp(a, AF, nx)[:, None]
                Y = np.interp(a, AF, PY)[:, None] + C[None, :] * np.interp(a, AF, ny)[:, None]
                V = (mos.get(key, b, X, Y) - mos.mu[key][b]) / mos.sd[key][b]; okv = ~np.isnan(V)
                sm[okv] += np.minimum(V[okv], CLIP); cnt[okv] += 1
            S = np.where(cnt >= max(4, 0.6 * len(bands)), sm / np.sqrt(np.maximum(cnt, 1)), -1e9); acc[key] = S
        for key, S in acc.items():
            i, j = np.unravel_index(np.argmax(S), S.shape)
            if S[i, j] > best[0]: best = (float(S[i, j]), dict(dM=dM, d=float(D[i]), c=float(C[j]), sign=key))
    return best
def nulls(loci, mos, A, bands, n, seed):
    rng = np.random.default_rng(seed); lo = min(A[b] for b in bands); sp = max(A[b] for b in bands) - lo; out = []
    for _ in range(n):
        r = rng.uniform(0, sp, len(bands)); Ar = {b: lo + r[i] for i, b in enumerate(bands)}
        out.append(search(loci, mos, Ar, bands)[0])
    return out
def band_values(loci, mos, A, bands, info):
    P = loci[info['dM']]; AF, PX, PY = P[:, 0], P[:, 2], P[:, 3]; nx, ny = normals(P); out = {}
    for b in bands:
        a = A[b] + info['d']; x = np.interp(a, AF, PX) + info['c'] * np.interp(a, AF, nx); y = np.interp(a, AF, PY) + info['c'] * np.interp(a, AF, ny)
        out[b] = (float(mos.get(info['sign'], b, np.array([x]), np.array([y]))[0]), x, y)
    return out
def crops3(tiles, bv, half=25):
    out = {}
    for b in ('blue', 'green', 'red', 'nir'):
        if b not in bv: continue
        _, x, y = bv[b]
        for T in tiles:
            minx, miny, maxx, maxy = T['bb']; col = int((x - minx) // 10); row = int((maxy - y) // 10)
            if 0 <= col < 600 and 0 <= row < 600:
                sl = (slice(max(0, row - half), row + half), slice(max(0, col - half), col + half))
                out[b] = dict(Z=T['Z'][b][sl].astype(np.float32), RGB=T['RGB'][sl].astype(float)); break
    return out
RES3 = {}
async def run3(e, tag, dM_half=3000, dM_step=1000, n_null=8):
    t0 = time.time()
    BAk, psi = await band_alphas(e)
    bands = [b for b in BID if b in BAk and b in e['item']['assets']]
    if not all(b in bands for b in ('blue', 'green', 'red', 'nir')): print('RESULT13', tag, 'skip: no angles'); return
    A = {b: BAk[b][0][3] for b in bands}
    lo, hi = min(A.values()), max(A.values()); cA = (lo + hi) / 2; sp = hi - lo
    Pp = locus(e, np.array([cA - 0.5, cA + 0.5])); kmdeg = math.hypot(Pp[1, 2] - Pp[0, 2], Pp[1, 3] - Pp[0, 3]) / 1000
    w = float(np.clip(8 / kmdeg, 0.15, 0.6)); al = np.linspace(lo - w, hi + w, 25)
    dMs = list(np.arange(-dM_half, dM_half + 1, dM_step)); dMs += [x for x in (-500, 500) if x not in dMs]
    loci = {float(dM): dens(locus(e, al, dM)) for dM in sorted(dMs)}
    ref = await ref_same_orbit(e); cells = plan_tiles(loci, 1500)
    print(f"RESULT13 {tag} {e['obj']} {e['sent']} v={e['v_al']:.1f} rng={e['rng']:.0f} km/deg={kmdeg:.1f} bands={len(bands)} B02-B04 gap {(A['red'] - A['blue']) * kmdeg:.1f} km ref={ref['id'][:30] if ref else None} tiles={len(cells)}")
    if len(cells) > 45: print('RESULT13', tag, 'too many tiles, trimming'); cells = cells[:45]
    tiles = []
    for cell in cells:
        T = await fetch_tile(e, ref, cell, bands)
        if T: tiles.append(T)
    if not tiles: print('RESULT13', tag, 'no tiles'); return
    modes = [m for T in tiles for m in T['modes']]; mos = Mosaic(tiles, bands)
    score, info = search(loci, mos, A, bands); nl = nulls(loci, mos, A, bands, n_null, 7 + len(RES3))
    s4, i4 = search(loci, mos, A, ['blue', 'green', 'red', 'nir'])
    bv = band_values(loci, mos, A, bands, info) if info else {}
    RES3[tag] = dict(e=e, A=A, bands=bands, score=score, info=info, nulls=nl, s4=s4, bv=bv, kmdeg=kmdeg, crops=crops3(tiles, bv) if info else {})
    rank = sum(x >= score for x in nl)
    print(f"RESULT13 {tag} done {time.time() - t0:.0f}s ref/self/empty={modes.count('ref')}/{modes.count('self')}/{modes.count('empty')} | TRUE={score:.2f} NULLmax={max(nl):.2f} med={np.median(nl):.2f} (nulls>=true: {rank}/{len(nl)}) | 10m-only={s4:.2f} | {info} | per-band " + ' '.join(f"{b}={v[0]:.1f}" for b, v in bv.items()))
    del tiles, mos; gc.collect()
def plot3(tag):
    R = RES3[tag]; cr = R['crops']; fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for i, b in enumerate(('blue', 'green', 'red', 'nir')):
        if b not in cr: continue
        rgb = cr[b]['RGB']; lo, hi = np.percentile(rgb, [1, 99])
        ax[0, i].imshow(np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)); ax[0, i].set_title(f"{b}: predicted spot", fontsize=9)
        ax[1, i].imshow(np.nan_to_num(cr[b]['Z']), cmap='RdBu_r', vmin=-8, vmax=8); ax[1, i].set_title(f"{b} z ({R['info']['sign']} {R['bv'][b][0]:.1f})", fontsize=9)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{tag} {R['e']['obj']}: TRUE {R['score']:.1f} vs NULL max {max(R['nulls']):.1f}"); plt.tight_layout(); plt.show()
async def match_tile(e):
    t = dt_from_jd(e['jd']); lt = e['sent'][-1]
    feats = await stac_post({'collections': ['sentinel-2-c1-l2a'], 'limit': 10, 'intersects': {'type': 'Point', 'coordinates': [e['lon'], e['lat']]}, 'datetime': f"{iso(t - dtm.timedelta(minutes=6))}/{iso(t + dtm.timedelta(minutes=6))}"})
    feats = [f for f in feats if f['id'].startswith('S2' + lt)]
    if not feats: return False
    feats.sort(key=lambda f: f['properties'].get('eo:cloud_cover', 100)); e['item'] = feats[0]; e['epsg'] = epsg_of(feats[0]); e['cloud'] = feats[0]['properties'].get('eo:cloud_cover', 100)
    return True
BIG = list({o.cat: o for o in objs if ('ZARYA' in o.name or 'TIANHE' in o.name)}.values())
bev = find_crossings(sents, BIG, J1 - 3.0, J1 + 0.15)
bev = [x for x in bev if x['desc'] and x['sun_el'] > 8]
print('RESULT13 big-object daylit crossings:', len(bev), [(x['obj'], x['sent'], str(dt_from_jd(x['jd']))[:16]) for x in bev])
BEV = []
for x in bev:
    if await match_tile(x) and x['cloud'] < 60: BEV.append(x)
print('RESULT13 with S2 tile:', [(x['obj'], x['item']['id'][:24], round(x['cloud'])) for x in BEV])
QUEUE = [(x, 'BIG%d' % i, 8000) for i, x in enumerate(BEV)] + [(CAND[k], '#%d' % k, 3000) for k in [8, 5, 6, 0, 1, 2, 3, 4, 7] + [k for k in range(9, len(CAND)) if (CAND[k].get('cloud') or 100) < 50]]
for e, tag, dmh in QUEUE:
    try: await run3(e, tag, dM_half=dmh)
    except Exception as ex: print('RESULT13 fail', tag, traceback.format_exc()[-300:].replace('\n', ' | '))
good = sorted([t for t in RES3 if RES3[t]['score'] > max(RES3[t]['nulls'])], key=lambda t: -(RES3[t]['score'] - max(RES3[t]['nulls'])))
print('RESULT13 SUMMARY above-null:', good)
for t in good[:3]: plot3(t)
print('RESULT13 ALLDONE')

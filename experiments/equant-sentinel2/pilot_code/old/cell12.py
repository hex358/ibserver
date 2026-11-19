import json, asyncio, time, datetime as dtm, traceback
import matplotlib.pyplot as plt
from pyodide.http import pyfetch
from scipy.ndimage import median_filter, maximum_filter
async def stac_post(q):
    for t in range(3):
        try:
            r = await pyfetch('https://earth-search.aws.element84.com/v1/search', method='POST', headers={'Content-Type': 'application/json'}, body=json.dumps(q))
            return (await r.json()).get('features', [])
        except Exception:
            await asyncio.sleep(2)
    return []
async def ref_same_orbit(e):
    pr = e['item']['properties']; t = dtm.datetime.fromisoformat(pr['datetime'].replace('Z', '+00:00'))
    win = f"{(t - dtm.timedelta(days=75)).strftime('%Y-%m-%dT%H:%M:%SZ')}/{(t - dtm.timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    qq = {'grid:code': {'eq': pr['grid:code']}, 'eo:cloud_cover': {'lt': 40}}
    if 'sat:relative_orbit' in pr: qq['sat:relative_orbit'] = {'eq': pr['sat:relative_orbit']}
    feats = await stac_post({'collections': ['sentinel-2-c1-l2a'], 'limit': 50, 'datetime': win, 'query': qq})
    if not feats: return None
    good = [f for f in feats if f['properties'].get('eo:cloud_cover', 100) < 8]
    pool = good if good else feats
    pool.sort(key=(lambda f: f['properties']['datetime']) if good else (lambda f: -f['properties'].get('eo:cloud_cover', 100)))
    return pool[-1]
async def gnp(href, bb, E):
    for t in range(4):
        try:
            a = await getnpy(href, bb, E)
            if a is not None: return a
        except Exception:
            pass
        await asyncio.sleep(1.5 * (t + 1))
    return None
def zmap(ev, evm, rf, rfm):
    v = evm & (ev > 0); mode = 'self'; Rr = ev
    if rf is not None:
        vr = v & rfm & (rf > 0)
        if vr.mean() > 0.6:
            g = np.median(ev[vr] / rf[vr]); Rr = ev - g * rf; v = vr; mode = 'ref'
    Rr = Rr - median_filter(Rr, 9)
    if v.sum() < 500: return np.zeros(ev.shape, np.float32), 'empty'
    md = np.median(Rr[v]); sg = 1.4826 * np.median(np.abs(Rr[v] - md)) + 1e-6
    Z = ((Rr - md) / sg).astype(np.float32); Z[~v] = 0
    return Z, mode
def dens(P, n=200):
    a = np.linspace(P[0, 0], P[-1, 0], n)
    return np.c_[a, np.interp(a, P[:, 0], P[:, 1]), np.interp(a, P[:, 0], P[:, 2]), np.interp(a, P[:, 0], P[:, 3])]
def normals(P):
    tx, ty = np.gradient(P[:, 2]), np.gradient(P[:, 3]); nn = np.hypot(tx, ty) + 1e-9
    return -ty / nn, tx / nn
def plan_tiles(loci, cmax, grid=5000.0):
    pts = []
    for P in loci.values():
        nx, ny = normals(P)
        for c in (-cmax, 0, cmax): pts.append(np.c_[P[:, 2] + c * nx, P[:, 3] + c * ny])
    pts = np.vstack(pts)
    return sorted({(int(np.floor(x / grid)), int(np.floor(y / grid))) for x, y in pts})
async def fetch_tile(e, ref, cell, grid=5000.0, pad=500.0):
    i, j = cell; bb = (i * grid - pad, j * grid - pad, (i + 1) * grid + pad, (j + 1) * grid + pad)
    E = e['epsg']; EV = {}; Z = {}; modes = []
    for b in BANDS:
        a = await gnp(e['item']['assets'][b]['href'], bb, E)
        if a is None: return None
        ev = a[0].astype(np.float32); evm = a[-1] > 0; rf = rfm = None
        if ref is not None:
            r = await gnp(ref['assets'][b]['href'], bb, E)
            if r is not None: rf = r[0].astype(np.float32); rfm = r[-1] > 0
        Z[b], m = zmap(ev, evm, rf, rfm); modes.append(m); EV[b] = a[0]
    return dict(bb=bb, EV=EV, Z=Z, MA={b: maximum_filter(np.abs(Z[b]), size=3) for b in BANDS}, modes=modes)
def lookup(tiles, b, X, Y):
    V = np.zeros(X.shape, np.float32)
    for T in tiles:
        minx, miny, maxx, maxy = T['bb']
        col = np.floor((X - minx) / 10).astype(int); row = np.floor((maxy - Y) / 10).astype(int)
        H, W = T['MA'][b].shape; m = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        if m.any(): V[m] = np.maximum(V[m], T['MA'][b][row[m], col[m]])
    return V
def search(loci, tiles, A, cmax=1500, cstep=20, dstep=0.003, bands=BANDS):
    best = (-1.0, None); C = np.arange(-cmax, cmax + 1, cstep)
    for dM, P in loci.items():
        AF, PX, PY = P[:, 0], P[:, 2], P[:, 3]; nx, ny = normals(P)
        d0 = AF[0] - min(A[b] for b in bands); d1 = AF[-1] - max(A[b] for b in bands)
        if d1 <= d0: continue
        D = np.arange(d0, d1, dstep); S = None; vals = {}
        for b in bands:
            a = A[b] + D
            X = np.interp(a, AF, PX)[:, None] + C[None, :] * np.interp(a, AF, nx)[:, None]
            Y = np.interp(a, AF, PY)[:, None] + C[None, :] * np.interp(a, AF, ny)[:, None]
            vals[b] = lookup(tiles, b, X, Y); S = vals[b] if S is None else np.minimum(S, vals[b])
        i, j = np.unravel_index(np.argmax(S), S.shape)
        if S[i, j] > best[0]: best = (float(S[i, j]), dict(dM=dM, d=float(D[i]), c=float(C[j]), per={b: float(vals[b][i, j]) for b in bands}))
    return best
def nulls(loci, tiles, A, n, seed, bands=BANDS):
    rng = np.random.default_rng(seed); sp = max(A.values()) - min(A.values()); out = []
    for _ in range(n):
        r = rng.uniform(0, sp, len(BANDS)); Ar = {b: min(A.values()) + r[i] for i, b in enumerate(BANDS)}
        out.append(search(loci, tiles, Ar, bands=bands)[0])
    return out
def crops2(loci, tiles, A, info, half=25):
    P = loci[info['dM']]; AF, PX, PY = P[:, 0], P[:, 2], P[:, 3]; nx, ny = normals(P); out = {}
    for b in BANDS:
        a = A[b] + info['d']
        x = np.interp(a, AF, PX) + info['c'] * np.interp(a, AF, nx); y = np.interp(a, AF, PY) + info['c'] * np.interp(a, AF, ny)
        for T in tiles:
            minx, miny, maxx, maxy = T['bb']; col = int((x - minx) // 10); row = int((maxy - y) // 10); H, W = T['Z'][b].shape
            if 0 <= col < W and 0 <= row < H:
                sl = (slice(max(0, row - half), row + half), slice(max(0, col - half), col + half))
                out[b] = dict(Z=T['Z'][b][sl].copy(), RGB=np.stack([T['EV'][q][sl] for q in ('red', 'green', 'blue')], -1).astype(float), xy=(x, y)); break
    return out
RES2 = {}
async def run2(k, n_null=8):
    e = CAND[k]; t0 = time.time()
    BAk, psi = await band_alphas(e)
    if any(b not in BAk for b in BANDS): print('RESULT12 skip', k); return
    A = {b: BAk[b][0][3] for b in BANDS}
    sp = max(A.values()) - min(A.values()); cA = (max(A.values()) + min(A.values())) / 2
    Pp = locus(e, np.array([cA - 0.5, cA + 0.5])); kmdeg = math.hypot(Pp[1, 2] - Pp[0, 2], Pp[1, 3] - Pp[0, 3]) / 1000
    w = float(np.clip(8 / kmdeg, 0.15, 0.6)); al = np.linspace(cA - sp / 2 - w, cA + sp / 2 + w, 25)
    loci = {dM: dens(locus(e, al, dM)) for dM in (-3000, -2000, -1000, -500, 0, 500, 1000, 2000, 3000)}
    ref = await ref_same_orbit(e); cells = plan_tiles(loci, 1500)
    print(f"RESULT12 #{k} {e['obj']} v={e['v_al']:.1f} rng={e['rng']:.0f}km km/deg={kmdeg:.1f} blue-red gap {sp * kmdeg:.1f} km ref={ref['id'] if ref else None} tiles={len(cells)}")
    tiles = []
    for cell in cells:
        T = await fetch_tile(e, ref, cell)
        if T: tiles.append(T)
    modes = [m for T in tiles for m in T['modes']]
    score, info = search(loci, tiles, A); nl = nulls(loci, tiles, A, n_null, 300 + k)
    RES2[k] = dict(e=e, A=A, score=score, info=info, nulls=nl, kmdeg=kmdeg, ref=ref['id'] if ref else None, crops=crops2(loci, tiles, A, info) if info else {})
    per = {b: round(v, 1) for b, v in info['per'].items()} if info else None
    print(f"RESULT12 #{k} done {time.time() - t0:.0f}s ref/self/empty={modes.count('ref')}/{modes.count('self')}/{modes.count('empty')} | TRUE={score:.1f} NULLmax={max(nl):.1f} med={np.median(nl):.1f} | dM={info['dM'] if info else None} d={info['d'] if info else 0:+.3f} c={info['c'] if info else 0:+.0f} {per}")
def plot2(k):
    R = RES2[k]; cr = R['crops']; fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for i, b in enumerate(BANDS):
        if b not in cr: continue
        rgb = cr[b]['RGB']; lo, hi = np.percentile(rgb, [1, 99])
        ax[0, i].imshow(np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)); ax[0, i].set_title(f"{b} predicted spot (RGB)", fontsize=9)
        ax[1, i].imshow(cr[b]['Z'], cmap='RdBu_r', vmin=-10, vmax=10); ax[1, i].set_title(f"{b} z, peak {R['info']['per'][b]:.1f}", fontsize=9)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"#{k} {R['e']['obj']}  TRUE {R['score']:.1f} vs NULL max {max(R['nulls']):.1f}"); plt.tight_layout(); plt.show()
for k in [8, 5, 6, 0, 1, 2, 3, 4, 7]:
    try: await run2(k)
    except Exception as ex: print('RESULT12 fail', k, traceback.format_exc()[-300:].replace('\n', ' | '))
if RES2:
    bk = max(RES2, key=lambda k: RES2[k]['score'] - max(RES2[k]['nulls'])); plot2(bk)
print('RESULT12 ALLDONE')

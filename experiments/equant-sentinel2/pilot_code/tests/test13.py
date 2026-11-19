import numpy as np, math, re
from scipy.ndimage import maximum_filter, minimum_filter
src = open('../old/cell13.py').read()
ns = dict(np=np, math=math, maximum_filter=maximum_filter, minimum_filter=minimum_filter)
exec(re.search(r"^BID = .*?$", src, re.M).group(0), ns); exec(re.search(r"^BRES = .*?$", src, re.M).group(0), ns)
exec(re.search(r"^GRID.*$", src, re.M).group(0), ns)
for name in ['dens', 'normals', 'plan_tiles', 'search', 'nulls', 'band_values']:
    exec(re.search(r'^def ' + name + r'\(.*?(?=^\S)', src, re.S | re.M).group(0), ns)
exec(re.search(r'^class Mosaic.*?(?=^def )', src, re.S | re.M).group(0), ns)
BID, BRES = ns['BID'], ns['BRES']; bands = list(BID)
rng = np.random.default_rng(0)
A = {b: 0.30 + 0.07 * i for i, b in enumerate(bands)}
def mk(dM):
    al = np.linspace(0.0, 1.6, 25); s = al * 12000 + dM * 0.3
    return ns['dens'](np.c_[al, al * 0, 400000 + s * math.sin(0.5) + dM * 0.1, 3100000 + s * math.cos(0.5)])
loci = {float(d): mk(d) for d in (-3000, -1000, 0, 1000, 3000)}
cells = ns['plan_tiles'](loci, 1500)
tiles = []
for c in cells:
    T = dict(cell=c, bb=(c[0] * 5000 - 500, c[1] * 5000 - 500, (c[0] + 1) * 5000 + 500, (c[1] + 1) * 5000 + 500), dark={}, bright={}, Z={})
    for b in bands:
        n = 600 if BRES[b] == 10 else 300
        Z = rng.standard_normal((n, n)).astype(np.float32); T['Z'][b] = Z
    tiles.append(T)
P = loci[1000.0]; nx, ny = ns['normals'](P)
for b in bands:  # weak dark object: z=-3.5 per band
    a = A[b] + 0.1; x = np.interp(a, P[:, 0], P[:, 2]) + 400 * np.interp(a, P[:, 0], nx); y = np.interp(a, P[:, 0], P[:, 3]) + 400 * np.interp(a, P[:, 0], ny)
    r = BRES[b]
    for T in tiles:
        minx, miny, maxx, maxy = T['bb']; col = int((x - minx) // r); row = int((maxy - y) // r); n = T['Z'][b].shape[0]
        if 0 <= col < n and 0 <= row < n: T['Z'][b][row, col] -= 3.5; break
for T in tiles:
    for b in bands:
        T['dark'][b] = (-minimum_filter(T['Z'][b], 3)).astype(np.float16); T['bright'][b] = maximum_filter(T['Z'][b], 3).astype(np.float16)
mos = ns['Mosaic'](tiles, bands)
s, info = ns['search'](loci, mos, A, bands)
nl = ns['nulls'](loci, mos, A, bands, 6, 1)
s4, _ = ns['search'](loci, mos, A, ['blue', 'green', 'red', 'nir'])
print('tiles', len(tiles), 'TRUE', round(s, 2), info, 'NULL', [round(x, 2) for x in nl], '10m-only', round(s4, 2))

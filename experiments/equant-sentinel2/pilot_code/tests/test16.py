import numpy as np, math, re
from scipy.ndimage import maximum_filter, minimum_filter
s13 = open('../old/cell13.py').read(); s16 = open('../old/cell16.py').read()
ns = dict(np=np, math=math)
for pat in [r"^BID = .*?$", r"^BRES = .*?$", r"^GRID.*$"]: exec(re.search(pat, s13, re.M).group(0), ns)
for name in ['dens', 'normals']: exec(re.search(r'^def ' + name + r'\(.*?(?=^\S)', s13, re.S | re.M).group(0), ns)
exec(re.search(r'^class Mosaic.*?(?=^def )', s13, re.S | re.M).group(0), ns)
for name in ['inside', 'xy_at', 'plan_tiles2', 'search4', 'best_positions']: exec(re.search(r'^def ' + name + r'\(.*?(?=^\S)', s16, re.S | re.M).group(0), ns)
BID, BRES = ns['BID'], ns['BRES']; bands = list(BID); rng = np.random.default_rng(0)
ulx, uly = 399960.0, 3200040.0
ok1 = np.zeros((23, 23), bool); ok1[:, :12] = True; ok2 = np.zeros((23, 23), bool); ok2[:, 11:] = True
DA = {b: [dict(det=7, a=0.3 + 0.07 * i, ok=ok1, ulx=ulx, uly=uly), dict(det=8, a=-1.4 + 0.07 * i, ok=ok2, ulx=ulx, uly=uly)] for i, b in enumerate(bands)}
def mk(dM):
    al = np.linspace(-2.5, 2.0, 60); s = al * 12000 + dM * 0.3
    return ns['dens'](np.c_[al, al * 0, 452000 + s * math.cos(1.2) + dM * 0.1, 3140000 + s * math.sin(1.2)], n=500)
loci = {float(d): mk(d) for d in (-1500, 0, 1500)}
Wd = 0.6; cells = ns['plan_tiles2'](loci, DA, bands, Wd, 1500); print('cells', len(cells))
tiles = []
for c in cells:
    T = dict(cell=c, bb=(c[0] * 5000 - 500, c[1] * 5000 - 500, (c[0] + 1) * 5000 + 500, (c[1] + 1) * 5000 + 500), dark={}, bright={}, Z={})
    for b in bands: T['Z'][b] = rng.standard_normal((600, 600) if BRES[b] == 10 else (300, 300)).astype(np.float32)
    tiles.append(T)
P = loci[0.0]; nx, ny = ns['normals'](P); hits = 0
for b in bands:
    for dd in DA[b]:
        X, Y = ns['xy_at'](P, nx, ny, np.array([dd['a'] + 0.2]), np.array([300.0]))
        if not ns['inside'](dd, X, Y)[0, 0]: continue
        x, y = X[0, 0], Y[0, 0]; r = BRES[b]
        for T in tiles:
            minx, miny, maxx, maxy = T['bb']; col = int((x - minx) // r); row = int((maxy - y) // r); n = T['Z'][b].shape[0]
            if 0 <= col < n and 0 <= row < n: T['Z'][b][row, col] -= 5.0; hits += 1; break
print('injected', hits)
for T in tiles:
    for b in bands: T['dark'][b] = (-minimum_filter(T['Z'][b], 3)).astype(np.float16); T['bright'][b] = maximum_filter(T['Z'][b], 3).astype(np.float16)
mos = ns['Mosaic'](tiles, bands)
sc, info = ns['search4'](loci, mos, DA, bands, Wd); nl = []
for k in range(5):
    p = rng.permutation(len(bands)); DAp = {b: DA[bands[p[i]]] for i, b in enumerate(bands)}; nl.append(round(ns['search4'](loci, mos, DAp, bands, Wd)[0], 2))
print('TRUE', round(sc, 2), info, 'NULL', nl, ns['best_positions'](loci, DA, bands, info)['blue'])

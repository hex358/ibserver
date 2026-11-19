import numpy as np, math, re, sys
from scipy.ndimage import maximum_filter, median_filter
src = open('../cell19.py').read()
mega = open('../old/mega2.py').read()
ns = dict(np=np, math=math, maximum_filter=maximum_filter, median_filter=median_filter)
exec(re.search(r"^BRES = .*?$", mega, re.M).group(0), ns); exec(re.search(r"^GRID.*$", mega, re.M).group(0), ns)
exec(re.search(r"^RLOC.*$", src, re.M).group(0), ns)
for name in ['utm_to_latlon', 'gval', 'band_roots', 'block2', 'shift_search']:
    exec(re.search(r'^def ' + name + r'\(.*?(?=^\S)', src, re.S | re.M).group(0), ns)
exec(re.search(r'^class Mos2.*?(?=^def )', src, re.S | re.M).group(0), ns)
exec(re.search(r'^def latlon_to_utm.*?(?=^def |^T_LINE)', mega, re.S | re.M).group(0), ns)
# utm round trip
x, y = ns['latlon_to_utm'](-17.06, 53.74, 39); print('utm rt', ns['utm_to_latlon'](x, y, 39, True))
# roots on synthetic grid: alpha field increasing eastward; locus heading east-north
ulx, uly = 500000.0, 8200000.0
gx = np.arange(23) * 5000; G = 0.3 + 0.00001 * gx[None, :] * np.ones((23, 1)); G[:, 12:] = np.nan
DG = {'blue': [dict(det=7, g=G, ulx=ulx, uly=uly)], 'red': [dict(det=7, g=G + 0.5, ulx=ulx, uly=uly)]}
al = np.linspace(-2.2, 2.2, 1500); P = np.c_[al, al * 0, ulx + 20000 + al * 30000, uly - 50000 + al * 5000]
R = ns['band_roots'](P, DG, ['blue', 'red']); print('roots', R)
# shift search synthetic
bands = list(ns['BRES']); rng = np.random.default_rng(1)
roots = {b: [(1, 0.0, 512000.0 + 1500 * k, 8150000.0 + 300 * k)] for k, b in enumerate(bands)}
cells = set()
for b in bands:
    for (_, _, x, y) in roots[b]:
        for xx in np.arange(x - 4400, x + 4401, 2500):
            for yy in np.arange(y - 4400, y + 4401, 2500): cells.add((int(np.floor(xx / 5000)), int(np.floor(yy / 5000))))
tiles = []
for c in sorted(cells):
    E = {}
    for b in bands:
        z = np.abs(rng.standard_normal((300, 300))); F = maximum_filter(np.abs(rng.standard_normal((300, 300))), 7)
        E[b] = maximum_filter(np.maximum(0, z - F), 3).astype(np.float16)
    tiles.append(dict(cell=c, bb=(c[0] * 5000 - 500, c[1] * 5000 - 500, (c[0] + 1) * 5000 + 500, (c[1] + 1) * 5000 + 500), E=E))
true = (1340.0, -760.0)
for b in bands:
    x, y = roots[b][0][2] + true[0], roots[b][0][3] + true[1]
    for T in tiles:
        minx, miny, maxx, maxy = T['bb']; col = int((x - minx) // 20); row = int((maxy - y) // 20)
        if 0 <= col < 300 and 0 <= row < 300:
            T['E'][b][max(0, row - 1):row + 2, max(0, col - 1):col + 2] = 6.0; break
mos = ns['Mos2'](tiles, bands)
import time; t0 = time.time()
print('TRUE', ns['shift_search'](roots, mos, bands, 4000.0), round(time.time() - t0, 1), 's')
nl = []
for _ in range(5):
    disp = {b: (rng.uniform(-3000, 3000), rng.uniform(-3000, 3000)) for b in bands}; nl.append(round(ns['shift_search'](roots, mos, bands, 4000.0, disp=disp)[0], 2))
print('NULL', nl, 'cells', len(cells))

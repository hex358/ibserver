import numpy as np, sys
from PIL import Image
from scipy.ndimage import median_filter, binary_dilation, maximum_filter
mos = np.array(Image.open(sys.argv[1])).astype(float)
p0 = np.array([float(v) for v in sys.argv[2].split(',')]); p1 = np.array([float(v) for v in sys.argv[3].split(',')])
km_per_deg = float(sys.argv[4])           # along-locus km per degree of band tilt
u = (p1 - p0) / np.linalg.norm(p1 - p0); nrm = np.array([-u[1], u[0]])
hp = mos - np.stack([median_filter(mos[..., c], size=9) for c in range(3)], -1)
chroma = hp - np.median(hp, axis=2, keepdims=True)
cloud = binary_dilation(mos.max(2) > 200, iterations=25)
H, W = mos.shape[:2]
# B02 leads: offsets along the locus (towards p1) for B03 and B04, in pixels (10 m)
dG = 0.255 * km_per_deg * 100; dR = 0.484 * km_per_deg * 100
best = []
for sign in (+1, -1):
    sb = sign * chroma[..., 2]; sg = sign * chroma[..., 1]; sr = sign * chroma[..., 0]
    sb = np.where(cloud, 0, sb); sg = np.where(cloud, 0, sg); sr = np.where(cloud, 0, sr)
    # allow +-2 px tolerance by max filtering
    sgm = maximum_filter(sg, size=5); srm = maximum_filter(sr, size=7)
    ys, xs = np.mgrid[0:H, 0:W]
    gx = np.round(xs + u[0] * dG).astype(int); gy = np.round(ys + u[1] * dG).astype(int)
    rx = np.round(xs + u[0] * dR).astype(int); ry = np.round(ys + u[1] * dR).astype(int)
    ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H) & (rx >= 0) & (rx < W) & (ry >= 0) & (ry < H)
    trip = np.zeros_like(sb)
    trip[ok] = np.minimum(np.minimum(sb[ok], sgm[gy[ok], gx[ok]]), srm[ry[ok], rx[ok]])
    rel = np.stack([xs - p0[0], ys - p0[1]], -1)
    across = rel @ nrm * 10 / 1000; along = rel @ u * 10 / 1000
    band = np.abs(across) < 3.5
    t = np.where(band, trip, 0)
    for _ in range(8):
        i = np.argmax(t); y, x = divmod(i, W)
        if t[y, x] <= 0: break
        best.append((t[y, x], sign, x, y, along[y, x], across[y, x], chroma[y, x].round(0)))
        t[max(0, y-15):y+16, max(0, x-15):x+16] = 0
best.sort(key=lambda b: -b[0])
for b in best[:10]:
    print(f"triplet strength={b[0]:5.1f} sign={'+' if b[1]>0 else '-'} B02 ghost at px=({b[2]},{b[3]}) along={b[4]:.2f}km across={b[5]:+.2f}km chroma(B02 px)={b[6]}")

import numpy as np, math
from PIL import Image
from scipy.ndimage import median_filter, label
def crop(f): return np.array(Image.open(f).convert('RGB')).astype(float)[26:966, 20:1020]
mos = np.zeros((1880, 2000, 3))
mos[940:, :1000] = crop('img/c1_480_4502.png')
mos[940:, 1000:] = crop('img/c2_490_4502.png')
mos[:940, :1000] = crop('img/c3_480_4511.png')
mos[:940, 1000:] = crop('img/c4_490_4511.png')
Image.fromarray(mos.astype(np.uint8)).save('img/mosaic_turkey.png')
X0, Y1 = 480000, 4520800
def to_pix(x, y): return ((x - X0) / 10, (Y1 - y) / 10)
# locus (band tilt -0.8..+0.8 deg), nominal
p0 = np.array(to_pix(483264.8, 4503130.7)); p1 = np.array(to_pix(495324.4, 4518207.7))
u = (p1 - p0) / np.linalg.norm(p1 - p0); nrm = np.array([-u[1], u[0]])
hp = mos - np.stack([median_filter(mos[..., c], size=7) for c in range(3)], -1)
med = np.median(hp, axis=2, keepdims=True)
chroma = hp - med               # channel-specific excess
score = np.abs(chroma).max(2)
rr, cc = np.mgrid[0:1880, 0:2000]
rel = np.stack([cc - p0[0], rr - p0[1]], -1)
along = rel @ u; across = rel @ nrm
L = np.linalg.norm(p1 - p0)
band = (np.abs(across) < 300) & (along > -300) & (along < L + 300)
# ignore saturated/cloud pixels
bright = mos.max(2) > 235
s = np.where(band & ~bright, score, 0)
thr = np.percentile(s[band], 99.99)
lab, n = label(s > max(thr, 25))
out = []
for k in range(1, n + 1):
    ys, xs = np.where(lab == k)
    i = np.argmax(s[ys, xs]); y, x = ys[i], xs[i]
    ch = 'RGB'[int(np.argmax(np.abs(chroma[y, x])))]
    sign = '+' if chroma[y, x][np.argmax(np.abs(chroma[y, x]))] > 0 else '-'
    out.append((s[y, x], x, y, ch + sign, along[y, x] * 10 / 1000, across[y, x] * 10 / 1000, len(ys), mos[y, x].astype(int)))
out.sort(reverse=True)
print('threshold', thr, 'components', n)
for o in out[:40]:
    print(f"score={o[0]:5.1f} px=({o[1]},{o[2]}) {o[3]} along={o[4]:6.2f}km across={o[5]:+5.2f}km size={o[6]} rgb={o[7]}")

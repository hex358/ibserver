import numpy as np, sys
from PIL import Image
from scipy.ndimage import map_coordinates, median_filter
mos = np.array(Image.open(sys.argv[1])).astype(float)
p0 = np.array([float(v) for v in sys.argv[2].split(',')]); p1 = np.array([float(v) for v in sys.argv[3].split(',')])
half = int(sys.argv[4]); out = sys.argv[5]
u = (p1 - p0) / np.linalg.norm(p1 - p0); n = np.array([-u[1], u[0]])
L = int(np.linalg.norm(p1 - p0)) + 200
s = np.arange(-100, L - 100); t = np.arange(-half, half + 1)
S, T = np.meshgrid(s, t)
X = p0[0] + S * u[0] + T * n[0]; Y = p0[1] + S * u[1] + T * n[1]
strip = np.stack([map_coordinates(mos[..., c], [Y, X], order=0, cval=0) for c in range(3)], -1)
hp = strip - np.stack([median_filter(strip[..., c], size=9) for c in range(3)], -1)
ch = hp - np.median(hp, axis=2, keepdims=True)
vis = np.clip(128 + 4 * ch, 0, 255)
Image.fromarray(strip.astype(np.uint8)).save(out + '_rgb.png')
Image.fromarray(vis.astype(np.uint8)).save(out + '_chroma.png')
print(strip.shape)

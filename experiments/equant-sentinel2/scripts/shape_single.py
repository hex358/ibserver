"""Уточнение формы по одному пролёту с большим увеличением вдоль полёта (камера push-broom растягивает объект
в |J·e_вдоль| раз; у STARLINK-30580 ×5.8 → эффективное разрешение вдоль оси ~1.7 м).

Модель: шина Lb×Wb и два крыла Lw×Ww (как в shape.py), но яркость крыльев и шины — раздельные линейные параметры
в каждой полосе (крылья — тёмные элементы, шина может быть светлее). ФРТ фиксирована по паспорту MSI
(MTF на Найквисте ≈ 0.27 → σ ≈ 4.2 м без ящика пикселя; 20-м полосы — вдвое). Перебор начальных ψ, φ + L-BFGS,
неопределённость — гессиан.

  python scripts/shape_single.py STARLINK-30580
"""
import glob, math, pickle, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import jax, jax.numpy as jnp
from scipy.optimize import minimize
from s2sat import shape as SH

name = sys.argv[1]
rc = pickle.loads(Path(glob.glob(str(ROOT / f"results/shape/{name}*.pkl"))[0]).read_bytes())
from shape_fit import row_dir
D = SH.to_jax(rc["bands"], row_dir(rc))
SIG = (4.2, 8.4)


def model_band(z, d):
    lw, ww, lb, wb = jnp.exp(z[0:4]); dth = z[4:7] / 1000.0; psi, phi = z[7], z[8]
    plates = SH.starlink_plates(jnp.array([lw, ww, lb, wb]), psi, phi)
    p = d["p0"] + d["J"] @ dth
    C = SH.kernel_cov(d, SIG[0] if d["res"] <= 10 else SIG[1])
    ow = SH.render_occ(plates[:2], d["J"] / 1000.0, p, d["X"], d["Y"], C)
    ob = SH.render_occ(plates[2:], d["J"] / 1000.0, p, d["X"], d["Y"], C)
    # r = ow·(cw − g) + ob·(cb − g) → r + (ow+ob)·g = cw·ow + cb·ob: два линейных параметра, cw, cb ≥ 0
    w = d["m"] / d["s"] ** 2; y = d["r"] + (ow + ob) * d["g"]
    A = jnp.array([[jnp.sum(w * ow * ow), jnp.sum(w * ow * ob)], [jnp.sum(w * ow * ob), jnp.sum(w * ob * ob)]]) + 1e-9 * jnp.eye(2)
    bb = jnp.array([jnp.sum(w * ow * y), jnp.sum(w * ob * y)])
    c = jnp.maximum(jnp.linalg.solve(A, bb), 0.0)
    return ow * (c[0] - d["g"]) + ob * (c[1] - d["g"]), c


def loss(z):
    tot = 0.0
    for d in D:
        m, _ = model_band(z, d)
        zz = jnp.where(d["m"] > 0, (d["r"] - m) / d["s"], 0.0)
        tot = tot + jnp.sum(d["m"] * 18.0 * (jnp.sqrt(1 + (zz / 3.0) ** 2) - 1))
    return tot


vg = jax.jit(jax.value_and_grad(loss))
f = lambda z: tuple(np.asarray(v, float) for v in vg(jnp.asarray(z)))
best = None; t0 = time.time()
for psi0 in np.radians([-30, -15, 0, 15, 30]):
    for phi0 in np.radians([-60, -20, 20, 60]):
        for init in ((12.8, 4.1), (6.0, 2.0)):
            z0 = np.array([math.log(init[0]), math.log(init[1]), math.log(4.1), math.log(2.7), 0, 0, 0, psi0, phi0])
            r = minimize(f, z0, jac=True, method="L-BFGS-B", options=dict(maxiter=400))
            if best is None or r.fun < best.fun:
                best = r
print(f"{name}: лучший χ² {best.fun:.1f} ({time.time() - t0:.0f} с)", flush=True)
hvp = jax.jit(lambda x, v: jax.jvp(jax.grad(loss), (x,), (v,))[1])
xj = jnp.asarray(best.x); n = len(best.x); H = np.zeros((n, n))
for i in range(n):
    H[:, i] = np.asarray(hvp(xj, jnp.zeros(n).at[i].set(1.0)))
H = 0.5 * (H + H.T)
try:
    C = np.linalg.inv(H / 2.0); sd = np.sqrt(np.clip(np.diag(C), 0, None))
except np.linalg.LinAlgError:
    sd = np.full(n, np.nan)
lw, ww, lb, wb = np.exp(best.x[:4])
print(f"  крыло {lw:.1f} ± {lw * sd[0]:.1f} м × {ww:.1f} ± {ww * sd[1]:.1f} м (известно 12.8 × 4.1); шина {lb:.1f} ± {lb * sd[2]:.1f} × {wb:.1f} ± {wb * sd[3]:.1f} м "
      f"(известно ~4.1 × 2.7); размах {2 * lw + lb:.1f} м (известно 29.0–29.7)", flush=True)
print(f"  ψ {math.degrees(best.x[7]):+.1f}° от направления полёта, φ {math.degrees(best.x[8]):+.1f}° (по Солнцу {math.degrees(SH.sun_phi(np.array(rc['sun']), best.x[7])):+.1f}°), dθ {np.round(best.x[4:7]).tolist()} м", flush=True)
for d in D:
    _, c = model_band(jnp.asarray(best.x), d)
    print(f"  {d['b']:9s} яркость крыльев {float(c[0]):6.0f}, шины {float(c[1]):6.0f}, земли {float(jnp.median(d['g'])):6.0f} (DN−1000)")
np.save(ROOT / f"results/shape/single_{name}.npy", best.x)

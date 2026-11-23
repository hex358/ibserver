"""Уточнение 3D-модели Starlink V2 mini по найденным пролётам (тест перед засекреченными аппаратами).

  python scripts/shape_fit.py prep results/events_v11/STARLINK-31550_*.json ...   # окна + камера пролёта
  python scripts/shape_fit.py fit  [--init 6,2]                                   # эксперименты

prep: θ (ошибка орбиты, км) по положениям pos_best (Гаусс–Ньютон), якобиан J_b каждой полосы (камера формы),
сырые остатки в окне ±600 м вокруг объекта в каждой полосе (родное разрешение), Солнце и Sentinel в осях объекта.
fit: (1) каждый пролёт — номинальная форма V2 mini (крылья 12.8×4.1 м, шина 4.1×2.7 м) против компактного
объекта той же площади (разрешена ли форма); (2) общая форма по всем пролётам из неверного начального
приближения (сходится ли к известной); (3) неопределённость по гессиану; (4) ориентация крыльев против
слежения за Солнцем.
"""
import argparse, glob, json, math, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "shape"
NOMINAL = (12.8, 4.1, 4.1, 2.7)


def prep(paths):
    from s2sat import passgeom as PG
    orbs = PG.Orbits()
    OUT.mkdir(parents=True, exist_ok=True)
    for path in paths:
        x = json.loads(Path(path).read_text()); e, r = x["event"], x["result"]
        name = Path(path).stem; t0 = time.time()
        if (OUT / f"{name}.pkl").exists():
            print(name, "есть"); continue
        P = PG.Pass(e, r, orbs)
        P.orb = PG.pick_orbit(orbs, e)
        tgt = {b: tuple(v) for b, v in r["pos_best"].items()}
        th, P0, J, rms0 = PG.gauss_newton_theta(P, tgt, [r.get("dM", 0.0), 0.0, 0.0])
        W = PG.window_residuals(P, tgt)
        # положение пятна в каждой полосе по сырым остаткам (экстремум сглаженного |r|/σ в 250 м от pos_best)
        from scipy.ndimage import gaussian_filter
        meas = {}
        for b, d in W.items():
            z = np.nan_to_num(d["r"] / d["s"]); zs = gaussian_filter(z, 1.0) * math.sqrt(4 * math.pi)
            near = np.hypot(d["X"] - tgt[b][0], d["Y"] - tgt[b][1]) <= 250.0
            k = np.argmax(np.where(near, np.abs(zs), -1)); pk = float(np.abs(zs).flat[k])
            if pk >= 4.0:
                meas[b] = (float(d["X"].flat[k]), float(d["Y"].flat[k]), pk)
        # устойчивый θ: Гаусс–Ньютон с весами Хьюбера (c = 30 м) по измеренным положениям
        for _ in range(4):
            bs = [b for b in meas if b in P0 and b in J]
            if len(bs) < 3:
                break
            A = np.concatenate([J[b] for b in bs]); rv = np.concatenate([np.array(meas[b][:2]) - np.array(P0[b][:2]) for b in bs])
            wts = np.repeat([min(1.0, 30.0 / max(math.hypot(*(np.array(meas[b][:2]) - np.array(P0[b][:2]))), 1e-6)) for b in bs], 2)
            pr = np.array([[0, 1 / 1.1, 0], [0, 0, 1 / 1.9]]) * 30.0
            dth, *_ = np.linalg.lstsq(np.vstack([A * np.sqrt(wts)[:, None], pr]), np.concatenate([rv * np.sqrt(wts), -pr @ th]), rcond=None)
            th = th + np.clip(dth, -2.0, 2.0); P0, J = P.jacobian(th)
        dev = {b: math.hypot(meas[b][0] - P0[b][0], meas[b][1] - P0[b][1]) for b in meas if b in P0}
        rms = float(np.median(list(dev.values()))) if dev else float("nan")
        pos = {b: P0[b][:2] for b in P0}
        jd_mid = float(np.median([P0[b][2] for b in P0]))
        A, sun, los, rng = P.frame(jd_mid)
        bands = {}
        for b, d in W.items():
            if b not in pos:
                continue
            fit = np.hypot(d["X"] - pos[b][0], d["Y"] - pos[b][1]) <= 160.0
            if fit.sum() < 0.8 * math.pi * (160.0 / d["res"]) ** 2 or not np.isfinite(d["s"]) or np.isfinite(d["r"][fit]).mean() < 0.8:
                continue
            ii = np.where(fit.any(1))[0]; jj = np.where(fit.any(0))[0]
            sl = (slice(ii[0], ii[-1] + 1), slice(jj[0], jj[-1] + 1))
            bands[b] = dict(X=d["X"][sl], Y=d["Y"][sl], r=d["r"][sl], s=d["s"], g=d["g"][sl], res=d["res"],
                            p0=np.array(pos[b]), J=J[b], jd=P0[b][2], meas=meas.get(b))
        if not bands:
            print(name, "нет окон", {b: (np.isfinite(d["r"]).mean().round(2), d["s"]) for b, d in W.items()}, flush=True); continue
        rec = dict(name=name, obj=e["obj"], cat=e["cat"], theta=th.tolist(), fit_rms=rms, dev=dev, sun=sun.tolist(), los=los.tolist(),
                   rng=rng, score=r.get("score"), bands=bands)
        (OUT / f"{name}.pkl").write_bytes(pickle.dumps(rec))
        snr = {b: m_[2] for b, m_ in meas.items()}
        print(f"{name}: θ {np.round(th, 3)} км, медиана отклонения пятна {rms:.0f} м (до {rms0:.0f}), полос {len(bands)}, пятно найдено в {len(meas)}, пик {max(snr.values(), default=0):.1f}σ, "
              f"Солнце·зенит {sun[2]:+.2f}, визир·зенит {los[2]:+.2f}, {time.time() - t0:.0f} с", flush=True)


def load():
    from s2sat import shape as SH
    recs = [pickle.loads(p.read_bytes()) for p in sorted(OUT.glob("*.pkl"))]
    recs = [rc for rc in recs if rc.get("fit_rms", 1e9) <= 80.0]       # пятно совпадает с геометрией пролёта
    for rc in recs:
        rc["D"] = SH.to_jax(rc["bands"])
    return recs


def fit(args):
    import jax, jax.numpy as jnp
    from scipy.optimize import minimize
    from s2sat import shape as SH
    recs = load()
    print(f"пролётов: {len(recs)}", flush=True)
    sig0 = jnp.array([5.0, 10.0])

    _vg = {}

    def vgf(rc, kind):
        key = (rc["name"], kind)
        if key not in _vg:
            D = rc["D"]
            _vg[key] = jax.jit(jax.value_and_grad(lambda pp, shape, sig: SH.pass_loss(shape, sig, pp, D, kind)))
        return _vg[key]

    def fit_pass(rc, shape, kind, sig=sig0):
        """Лучшие (dθ, ψ, φ) для пролёта при заданной форме: перебор начальных ψ, φ + L-BFGS."""
        sun = np.array(rc["sun"]); best = None
        g_ = vgf(rc, kind); shp = jnp.asarray(shape, float)
        f = lambda pp: tuple(np.asarray(v, float) for v in g_(jnp.asarray(pp), shp, sig))
        for psi0 in np.radians(np.arange(0, 180, 22.5)):
            for phi0 in (np.radians([-60.0, -20.0, 20.0, 60.0]).tolist() + [SH.sun_phi(sun, psi0)] if kind == "starlink" else [0.0]):
                x0 = np.array([0.0, 0.0, 0.0, psi0, phi0])
                res = minimize(f, x0, jac=True, method="L-BFGS-B",
                               bounds=[(-300, 300), (-300, 300), (-2000, 2000), (None, None), (-math.pi / 2, math.pi / 2)], options=dict(maxiter=200))
                if best is None or res.fun < best.fun:
                    best = res
        return best

    # 0) «нет объекта»: потеря при occ = 0
    def null_loss(rc):
        tot = 0.0
        for d in rc["D"]:
            z = d["r"] / d["s"]; tot += float(jnp.sum(d["m"] * 2 * 9.0 * (jnp.sqrt(1 + (z / 3.0) ** 2) - 1)))
        return tot

    # 1) по пролётам: номинальная форма против компактной той же площади
    area = 2 * NOMINAL[0] * NOMINAL[1] + NOMINAL[2] * NOMINAL[3]
    rows = []
    for rc in recs:
        L0 = null_loss(rc)
        fs = fit_pass(rc, NOMINAL, "starlink"); fc = fit_pass(rc, (area, 0, 0, 0), "compact")
        psi, phi = fs.x[3], fs.x[4]; phis = SH.sun_phi(np.array(rc["sun"]), psi)
        rows.append(dict(name=rc["name"], det=L0 - fs.fun, d_shape=fc.fun - fs.fun, psi=math.degrees(psi) % 180,
                         phi=math.degrees(phi), phi_sun=math.degrees(phis), x=fs.x))
        print(f"{rc['name']}: χ²(нет объекта) − χ²(V2 mini) = {L0 - fs.fun:7.1f}; χ²(компактный) − χ²(V2 mini) = {fc.fun - fs.fun:6.1f}; "
              f"ψ {math.degrees(psi) % 180:5.1f}°, φ {math.degrees(phi):+5.1f}° (по Солнцу {math.degrees(phis):+5.1f}°), dθ {np.round(fs.x[:3]).tolist()} м", flush=True)
    good = [rc for rc, rw in zip(recs, rows) if rw["det"] > 25]
    print(f"для общей формы: {len(good)} пролётов с χ²-выигрышем > 25", flush=True)

    # 2) общая форма: L-BFGS по (log Lw, log Ww, log σ10, log σ20) + все параметры пролётов, из неверного начала
    init = [float(v) for v in args.init.split(",")]
    n = len(good)

    def total(z):
        lw, ww, ls10, ls20 = z[:4]
        shape = jnp.array([jnp.exp(lw), jnp.exp(ww), NOMINAL[2], NOMINAL[3]]); sig = jnp.exp(jnp.array([ls10, ls20]))
        tot = 0.0
        for k, rc in enumerate(good):
            tot = tot + SH.pass_loss(shape, sig, z[4 + 5 * k: 9 + 5 * k], rc["D"])
        return tot
    tvg = jax.jit(jax.value_and_grad(total))
    for start in [init, list(NOMINAL[:2])]:
        z0 = [math.log(start[0]), math.log(start[1]), math.log(5.0), math.log(10.0)]
        for rc in good:
            fp = fit_pass(rc, (start[0], start[1], NOMINAL[2], NOMINAL[3]), "starlink")
            z0 += list(fp.x)
        z0 = np.array(z0)
        f = lambda z: tuple(np.asarray(v, float) for v in tvg(jnp.asarray(z)))
        res = minimize(f, z0, jac=True, method="L-BFGS-B", options=dict(maxiter=3000, maxfun=6000))
        H = np.asarray(jax.hessian(total)(jnp.asarray(res.x)))
        try:
            C = np.linalg.inv(H / 2.0)                  # χ²/2 ≈ −log L (псевдо-Хьюбер ≈ χ² у центра)
            sd = np.sqrt(np.clip(np.diag(C)[:4], 0, None))
        except np.linalg.LinAlgError:
            sd = np.full(4, np.nan)
        Lw, Ww = math.exp(res.x[0]), math.exp(res.x[1]); s10, s20 = math.exp(res.x[2]), math.exp(res.x[3])
        print(f"старт Lw={start[0]} Ww={start[1]}: крыло {Lw:.1f} ± {Lw * sd[0]:.1f} м × {Ww:.1f} ± {Ww * sd[1]:.1f} м "
              f"(известно 12.8 × 4.1), размах {2 * Lw + NOMINAL[2]:.1f} м (известно ~30), ФРТ σ10 {s10:.1f} м σ20 {s20:.1f} м, "
              f"χ² {res.fun:.1f}, {res.message}", flush=True)
    json.dump(rows, open(OUT / "per_pass.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), indent=1)


def gauss(args):
    """Непараметрическая форма: K гауссиан в осях орбиты, общие для всех выбранных пролётов (3DGS + аффинная камера)."""
    import jax, jax.numpy as jnp
    from scipy.optimize import minimize
    from s2sat import shape as SH
    recs = [rc for rc in load() if rc["obj"].startswith(tuple(args.objs.split(",")))]
    K = args.K; rng = np.random.default_rng(0)
    print(f"пролётов: {len(recs)} ({sorted({rc['obj'] for rc in recs})}), гауссиан {K}", flush=True)
    sig = jnp.array([5.0, 10.0])

    def total(z):
        th = z[:10 * K]; tot = 0.0
        for k, rc in enumerate(recs):
            tot = tot + SH.pass_loss_gauss(th, K, sig, z[10 * K + 3 * k: 10 * K + 3 * k + 3], rc["D"])
        T = th.reshape(K, 10)
        # априорно: компоненты не шире 30 м и не уже 0.5 м; центры в пределах ~25 м; радиус сверху не виден → к нулю
        prior = jnp.sum(((T[:, 3:6] - jnp.log(3.0)) / 1.5) ** 2) + jnp.sum((T[:, :2] / 25.0) ** 2) + jnp.sum((T[:, 2] / 3.0) ** 2)
        dth = z[10 * K:].reshape(-1, 3)
        prior = prior + jnp.sum((dth / jnp.array([300.0, 300.0, 1500.0])) ** 2)
        return tot + prior
    vg = jax.jit(jax.value_and_grad(total))
    best = None
    for rs in range(args.restarts):
        T0 = np.zeros((K, 10))
        T0[:, 0] = rng.uniform(-15, 15, K); T0[:, 1] = rng.uniform(-6, 6, K)
        T0[:, 3:6] = np.log([3.0, 2.0, 0.5]); T0[:, 6] = rng.uniform(-0.3, 0.3, K); T0[:, 9] = np.log(40.0)
        z0 = np.concatenate([T0.ravel(), np.zeros(3 * len(recs))])
        f = lambda z: tuple(np.asarray(v, float) for v in vg(jnp.asarray(z)))
        res = minimize(f, z0, jac=True, method="L-BFGS-B", options=dict(maxiter=3000, maxfun=6000))
        print(f"  старт {rs}: χ² {res.fun:.1f} ({res.nit} итераций)", flush=True)
        if best is None or res.fun < best.fun:
            best = res
    th = jnp.asarray(best.x[:10 * K]); mu, Sig, a = (np.asarray(v) for v in SH.gauss_params(th, K))
    # вид сверху (вдоль × поперёк): покрытие на сетке 0.5 м
    g = np.arange(-30, 30.01, 0.5); GA, GC = np.meshgrid(g, g, indexing="ij")
    dens = np.zeros_like(GA)
    for k in range(K):
        S2 = Sig[k][:2, :2] + 1e-6 * np.eye(2); iS = np.linalg.inv(S2); d0 = GA - mu[k, 0]; d1 = GC - mu[k, 1]
        q = iS[0, 0] * d0 ** 2 + 2 * iS[0, 1] * d0 * d1 + iS[1, 1] * d1 ** 2
        dens += a[k] / (2 * np.pi * np.sqrt(np.linalg.det(S2))) * np.exp(-0.5 * q)
    w = dens / dens.sum(); ma = (w * GA).sum(); mc = (w * GC).sum()
    C = np.array([[(w * (GA - ma) ** 2).sum(), (w * (GA - ma) * (GC - mc)).sum()], [0, (w * (GC - mc) ** 2).sum()]]); C[1, 0] = C[0, 1]
    ev, V = np.linalg.eigh(C)
    L, Wd = math.sqrt(12 * ev[1]), math.sqrt(12 * ev[0]); ang = math.degrees(math.atan2(V[1, 1], V[0, 1])) % 180
    print(f"вид сверху: длина {L:.1f} м × ширина {Wd:.1f} м (по моментам, как у равномерного бруса), ось {ang:.0f}° от направления полёта; "
          f"площадь Σa {a.sum():.0f} м² на снимке", flush=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(dens.T, origin="lower", extent=[g[0], g[-1], g[0], g[-1]], cmap="magma")
    ax.set_xlabel("вдоль полёта, м"); ax.set_ylabel("поперёк, м"); ax.set_title(f"{args.objs}: {len(recs)} пролётов, {K} гауссиан\nдлина {L:.1f} м, ширина {Wd:.1f} м")
    if args.objs.startswith("STARLINK"):
        ax.add_patch(plt.Rectangle((-14.85, -2.05), 29.7, 4.1, fill=False, ec="cyan", lw=1, ls="--"))
    out = OUT / f"gauss_{args.objs.split(',')[0]}_K{K}.png"; fig.savefig(out, dpi=110, bbox_inches="tight"); print("рисунок", out, flush=True)
    json.dump(dict(mu=mu.tolist(), Sig=Sig.tolist(), a=a.tolist(), L=L, W=Wd, axis_deg=ang, chi2=float(best.fun), passes=[rc["name"] for rc in recs]),
              open(OUT / f"gauss_{args.objs.split(',')[0]}_K{K}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("paths", nargs="*"); ap.add_argument("--init", default="6,2")
    ap.add_argument("--objs", default="STARLINK"); ap.add_argument("--K", type=int, default=6); ap.add_argument("--restarts", type=int, default=3)
    a = ap.parse_args()
    {"prep": lambda: prep(a.paths), "fit": lambda: fit(a), "gauss": lambda: gauss(a)}[a.cmd]()

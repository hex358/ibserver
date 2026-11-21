"""Лаборатория E3–E6 на окнах вокруг корней (быстро: окна маленькие, кэшируются).

E3  анатомия детекции: смещение пятна от прогноза по полосам, форма, амплитуда, контраст, фон.
E4  шум остатка после регрессии по другим полосам: σ, хвосты, корреляция, по типам фона.
E5  нулевое распределение статистики при известном положении (случайные точки окон).
E6  вставки объектов расчётной формы: измеренный SNR против предсказанного.

Отражение: ρ = DN·1e−4 − 0.1 (L2A, коллекция sentinel-2-c1-l2a).
"""
import json, math, sys, warnings, datetime as dt
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); warnings.filterwarnings("ignore")
import numpy as np
from scipy.ndimage import label, convolve, binary_dilation
from s2sat import orbits as O, crossings as C, geometry as G, imagery as I, s2meta as M, bgsub

OUT = ROOT / "results/lab"; OUT.mkdir(parents=True, exist_ok=True)
S = ROOT / "data/tle_snapshots"
HALF = 600.0            # окно ±600 м вокруг корня
BANDS = list(M.BID)


def load_merged(patterns, snaps):
    best = {}
    for sd in snaps:
        for pat in patterns:
            for f in sorted((S / sd).glob(pat)):
                for o in O.parse_tle(f.read_text()):
                    if o.catnr not in best or o.epoch_jd > best[o.catnr].epoch_jd:
                        best[o.catnr] = o
    return best


def event_for(name, t0, objs, sents):
    ob = [o for o in objs.values() if o.name.split()[0] == name][0]
    j0 = O.jd_from_dt(t0)
    ev = [e for e in C.find_crossings(sents, [ob], j0, j0 + 1 / 48) if e["desc"] and e["sun_el"] > 8]
    return ob, ev[0]


def setup(sen, ob, e):
    its = I.items_around(e["lon"], e["lat"], O.dt_from_jd(e["jd"]), e["sent"][-1], d=1.0)
    inside = [i for i in its if i.bbox[0] <= e["lon"] <= i.bbox[2] and i.bbox[1] <= e["lat"] <= i.bbox[3]] or its
    main = min(inside, key=lambda i: i.properties.get("eo:cloud_cover", 100)); epsg = I.epsg_of(main)
    rS, _ = sen.ecef([e["jd"]]); psi = G.track_azimuth(sen, e["jd"]); H = float(np.linalg.norm(rS[0])) - G.R_LOC
    DG = {}
    for it in its:
        if I.epsg_of(it) != epsg: continue
        for b, l in M.parse_grids(I.get_text(it.assets["granule_metadata"].href), psi, H, it.id).items():
            DG.setdefault(b, []).extend(l)
    return its, main, epsg, DG


def roots_at(sen, ob, e, DG, epsg, dm):
    tr = G.apparent_track(sen, ob, e["jd"], np.linspace(-2.6, 2.6, 2000), dpos_km=dm)
    ok = np.isfinite(tr["ground"][:, 0]); X, Y = G.ecef_to_utm(tr["ground"][ok], epsg); A = tr["alpha"][ok]
    R = M.band_roots(A, X, Y, DG, BANDS)
    kmdeg = float(np.median(np.hypot(np.diff(X), np.diff(Y)) / np.diff(A))) / 1000
    # направление трека (для осей «вдоль/поперёк» на снимке): касательная к видимому треку
    tx, ty = np.gradient(X), np.gradient(Y); k = len(X) // 2; tn = math.hypot(tx[k], ty[k])
    return R, kmdeg, (tx[k] / tn, ty[k] / tn)


def item_for(its, main, epsg, x, y):
    lon, lat = G.utm_to_lonlat(x, y, epsg)
    c = [i for i in its if i.bbox[0] <= lon <= i.bbox[2] and i.bbox[1] <= lat <= i.bbox[3]]
    c.sort(key=lambda i: (i.id != main.id, I.epsg_of(i) != epsg, i.properties.get("eo:cloud_cover", 100)))
    return c[0] if c else main


def window(it, epsg, x, y):
    bb = (round(x / 10) * 10 - HALF, round(y / 10) * 10 - HALF, round(x / 10) * 10 + HALF, round(y / 10) * 10 + HALF)
    IM = {b: I.read_window(it.assets[b].href, epsg, bb, res=10) for b in BANDS if b in it.assets}
    scl = I.read_window(it.assets["scl"].href, epsg, bb, res=10, resampling="nearest") if "scl" in it.assets else None
    return bb, IM, scl


def residual(IM, b, near, valid):
    regs = [IM[c] for c in IM if c != b and c not in near]
    return bgsub.predict_band(IM[b], regs, valid, block=60, win=120)


def analyse_event(tag, sen, ob, e, dm, shift, inject_trials=0, rng=None):
    its, main, epsg, DG = setup(sen, ob, e)
    R, kmdeg, tdir = roots_at(sen, ob, e, DG, epsg, dm)
    Arep = {b: float(np.median([q[1] for q in R[b]])) for b in R}
    per_band, noise_rows, null_u, inj_rows = [], [], {}, []
    k_img = e["lam"] / e["rng"]
    for b in BANDS:
        if b not in R: continue
        det, al, x0, y0 = R[b][0]
        x, y = x0 + shift[0], y0 + shift[1]
        it = item_for(its, main, epsg, x, y)
        bb, IM, scl = window(it, epsg, x, y)
        valid = np.all(np.stack([IM[c] > 0 for c in IM]), 0)
        if scl is not None:
            valid &= ~binary_dilation(np.isin(scl.astype(int), (1, 3, 8, 9, 10)), iterations=5)
        near = {c for c in IM if c in Arep and abs(Arep[c] - Arep[b]) * kmdeg < 0.3}
        Rb = residual(IM, b, near, valid)
        n = Rb.shape[0]; cc = n // 2
        yy, xx = np.mgrid[0:n, 0:n]
        rr = np.hypot(xx - cc, yy - cc) * 10
        bgm = valid & (rr > 250)
        v = Rb[bgm]; v = v[np.isfinite(v)]
        med = np.median(v); sig = 1.4826 * np.median(np.abs(v - med)) + 1e-6
        z = (Rb - med) / sig
        # шум (E4)
        zf = z[bgm & np.isfinite(z)]
        ac1 = np.nanmean(z[:, 1:] * z[:, :-1]) / np.nanmean(z * z)
        mf = convolve(np.nan_to_num(z), bgsub.KERNEL); mfv = mf[bgm]
        noise_rows.append(dict(tag=tag, band=b, bg_dn=float(np.median(IM[b][valid])), sigma_dn=float(sig),
                               frac_gt5=float(np.mean(np.abs(zf) > 5)), frac_gt8=float(np.mean(np.abs(zf) > 8)),
                               kurt=float(np.mean(zf ** 4) / max(np.mean(zf ** 2), 1e-9) ** 2), ac_lag1=float(ac1),
                               mf_sigma=float(1.4826 * np.median(np.abs(mfv - np.median(mfv))))))
        null_u[b] = (mf[bgm] / (1.4826 * np.median(np.abs(mfv - np.median(mfv))) + 1e-6))
        # анатомия пятна (E3): ближайшее к центру связное пятно |z|>4 в радиусе 150 м
        inner = valid & (rr <= 150) & np.isfinite(z)
        if inner.any():
            zi = np.where(inner, z, 0)
            k = np.unravel_index(np.argmax(np.abs(zi)), zi.shape); sgn = np.sign(zi[k])
            lab_, _ = label((sgn * np.nan_to_num(z) > 3) & valid)
            blob = lab_ == lab_[k] if lab_[k] > 0 else np.zeros_like(valid)
            if blob.any():
                w = np.abs(np.nan_to_num(Rb))[blob]
                bx = xx[blob]; by = yy[blob]
                cx = float((w * bx).sum() / w.sum()); cy = float((w * by).sum() / w.sum())
                dxm = (cx - cc) * 10; dym = -(cy - cc) * 10
                along = dxm * tdir[0] + dym * tdir[1]; cross = -dxm * tdir[1] + dym * tdir[0]
                # размеры по осям трека
                px = (bx - cx) * 10; py = -(by - cy) * 10
                pa = px * tdir[0] + py * tdir[1]; pc = -px * tdir[1] + py * tdir[0]
                bgval = float(np.median(IM[b][valid & (rr < 400) & (rr > 150)]))
                per_band.append(dict(tag=tag, band=b, det=det, alpha=al, peak_z=float(zi[k]), peak_dn=float(Rb[k]),
                                     flux_dn=float(np.nansum(Rb[blob])), npix=int(blob.sum()),
                                     off_along_m=along, off_cross_m=cross,
                                     len_along_m=float(pa.max() - pa.min() + 10), len_cross_m=float(pc.max() - pc.min() + 10),
                                     bg_refl=bgval * 1e-4 - 0.1, contrast_refl=float(Rb[k]) * 1e-4))
        # вставки (E6): объект расчётной формы в случайном месте фона
        if inject_trials and rng is not None:
            for t in range(inject_trials):
                for shape in ((1, 3), (1, 7), (2, 7), (4, 4)):
                    for rho in (0.03, 0.06, 0.12):
                        r0 = rng.integers(10, n - 10); c0 = rng.integers(10, n - 10)
                        if rr[r0, c0] < 300 or not valid[r0, c0]: continue
                        ev2 = {c: IM[c].copy() for c in IM}
                        sl = (slice(r0, r0 + shape[0]), slice(c0, c0 + shape[1]))
                        rho_dn = (rho + 0.1) * 1e4
                        ev2[b][sl] = np.where(ev2[b][sl] > 0, rho_dn, 0)
                        R2 = residual(ev2, b, near, valid)
                        z2 = convolve(np.nan_to_num((R2 - med) / sig), bgsub.KERNEL)
                        u = z2[r0:r0 + shape[0], c0:c0 + shape[1]].flat[np.argmax(np.abs(z2[r0:r0 + shape[0], c0:c0 + shape[1]]))] / noise_rows[-1]["mf_sigma"]
                        bg_dn = float(np.median(IM[b][sl][IM[b][sl] > 0])) if (IM[b][sl] > 0).any() else np.nan
                        contrast = rho - (bg_dn * 1e-4 - 0.1)
                        # предсказание для гауссова ядра σ=0.8 px: отклик = Σ ядро·контраст по пикселям объекта
                        kk = bgsub.KERNEL; kc = kk.shape[0] // 2
                        pred = 0.0
                        for dr in range(shape[0]):
                            for dc in range(shape[1]):
                                ir = dr - shape[0] // 2 + kc; ic = dc - shape[1] // 2 + kc
                                if 0 <= ir < kk.shape[0] and 0 <= ic < kk.shape[1]: pred += kk[ir, ic]
                        pred_u = pred * contrast * 1e4 / sig
                        inj_rows.append(dict(tag=tag, band=b, shape=f"{shape[0]}x{shape[1]}", rho=rho, bg_refl=bg_dn * 1e-4 - 0.1,
                                             contrast=contrast, pred_u=float(pred_u), meas_u=float(u)))
    return dict(tag=tag, kmdeg=kmdeg, k_img=k_img, v_al=e["v_al"], rng=e["rng"], lam=e["lam"], sent=e["sent"], obj=e["obj"],
                per_band=per_band, noise=noise_rows, inj=inj_rows, null_u={b: v.tolist()[:20000] for b, v in null_u.items()})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--inject", type=int, default=0); ap.add_argument("--only", default=None)
    ap.add_argument("--spec", action="append", default=[], help="NAME,src(gp20|merged|supgp),YYYY-mm-ddTHH:MM,dm,sx,sy,snaps(через +)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    sents20 = O.parse_tle((S / "20261120T0320/sentinel2.tle").read_text())
    pilot = {o.catnr: o for o in O.parse_tle_tokens((ROOT / "pilot_code/tle_raw.txt").read_text())}
    gp20 = load_merged(["gp_starlink.tle"], ["20261120T0320"])
    merged = load_merged(["gp_*.tle", "debris_*.tle"], ["20261120T0320", "20261121T0627"])
    sents_m = [s for s in load_merged(["sentinel2.tle"], ["20261120T0320", "20261121T0627"]).values() if s.name.startswith("SENTINEL-2")]
    U = dt.timezone.utc
    specs = [
        ("ISS", "pilot", dt.datetime(2026, 11, 18, 22, 20, tzinfo=U), 0.0, (620, 860)),
        ("STARLINK-31550", "gp20", dt.datetime(2026, 11, 19, 18, 0, tzinfo=U), -1.5, (-480, -900)),
        ("STARLINK-32702", "merged", dt.datetime(2026, 11, 20, 8, 30, tzinfo=U), -5.0, (1320, 1500)),
        ("STARLINK-4026", "merged", dt.datetime(2026, 11, 21, 2, 45, tzinfo=U), 5.5, (920, 740)),
        ("STARLINK-6044", "merged", dt.datetime(2026, 11, 20, 14, 15, tzinfo=U), -3.0, (700, -660)),
        ("STARLINK-37359", "gp20", dt.datetime(2026, 11, 19, 18, 0, tzinfo=U), 6.0, (0, 0)),     # фон: пустыня (вставки)
        ("STARLINK-33770", "gp20", dt.datetime(2026, 11, 19, 17, 10, tzinfo=U), 0.0, (0, 0)),    # фон: поля
    ]
    supgp = load_merged(["supgp_*.tle"], sorted(p.name for p in S.iterdir() if p.is_dir()))
    if a.spec:
        specs = []
        SNAPS = {}
        for sp in a.spec:
            n_, src_, t_, dm_, sx_, sy_, sn_ = sp.split(",")
            SNAPS[n_] = sn_.split("+")
            specs.append((n_, src_, dt.datetime.fromisoformat(t_).replace(tzinfo=U), float(dm_), (float(sx_), float(sy_))))
    for name, src, t0, dm, shift in specs:
        if a.only and a.only not in name: continue
        try:
            if src == "pilot":
                sen, ob = pilot["42063"], pilot["25544"]
                e = [x for x in C.find_crossings([sen], [ob], O.jd_from_dt(t0), O.jd_from_dt(t0) + 1 / 48)][0]
            else:
                if src == "supgp" and a.spec:
                    objs = load_merged(["supgp_*.tle"], SNAPS[name])
                    ss = [s_ for s_ in load_merged(["sentinel2.tle"], SNAPS[name]).values() if s_.name.startswith("SENTINEL-2")]
                else:
                    objs = gp20 if src == "gp20" else (supgp if src == "supgp" else merged)
                    ss = sents20 if src == "gp20" else sents_m
                ob, e = event_for(name, t0, objs, ss); sen = {s.name: s for s in ss}[e["sent"]]
            res = analyse_event(name, sen, ob, e, dm, shift, inject_trials=a.inject, rng=rng)
            (OUT / f"lab_{name}.json").write_text(json.dumps(res, default=float))
            print(f"== {name} {e['sent']} v_al {e['v_al']:.1f} k {res['k_img']:.2f} kmdeg {res['kmdeg']:.1f}", flush=True)
            for p in res["per_band"]:
                print(f"  {p['band']:9s} det{p['det']:2d} z {p['peak_z']:7.1f} peakDN {p['peak_dn']:7.0f} flux {p['flux_dn']:8.0f} npix {p['npix']:3d} "
                      f"off along {p['off_along_m']:6.0f} cross {p['off_cross_m']:6.0f}  size {p['len_along_m']:4.0f}x{p['len_cross_m']:4.0f} m  "
                      f"bg ρ {p['bg_refl']:.3f} contrast {p['contrast_refl']:+.3f}", flush=True)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print("FAIL", name, repr(ex)[:200], flush=True)
    print("ALLDONE")

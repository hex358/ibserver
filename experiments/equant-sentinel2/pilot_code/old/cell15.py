async def look(e, tag, half_km=6.0, res=20):
    BAk, _ = await band_alphas(e); A = {b: BAk[b][0][3] for b in ('blue', 'green', 'red', 'nir')}
    pos = {b: tuple(locus(e, np.array([A[b]]))[0][2:4]) for b in A}
    xs = [v[0] for v in pos.values()]; ys = [v[1] for v in pos.values()]
    cx, cy = float(np.mean(xs)), float(np.mean(ys)); hw = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + half_km * 1000
    hw = round(hw / res) * res; bb = (cx - hw, cy - hw, cx + hw, cy + hw)
    ref = await ref_same_orbit(e); EVs, Zs = {}, {}
    for b in ('blue', 'green', 'red'):
        a = await getnpy2(e['item']['assets'][b]['href'], bb, e['epsg'], res)
        r = await getnpy2(ref['assets'][b]['href'], bb, e['epsg'], res) if ref else None
        ev = a[0].astype(np.float32); EVs[b] = ev
        if r is not None:
            rf = r[0].astype(np.float32); v = (ev > 0) & (rf > 0); g = np.median(ev[v] / rf[v]); R = ev - g * rf
        else:
            R = ev - median_filter(ev, 15)
        R = R - median_filter(R, 9); s = 1.4826 * np.median(np.abs(R - np.median(R))) + 1e-6; Zs[b] = R / s
    tr = lambda x, y: ((x - bb[0]) / res, (bb[3] - y) / res)
    Pl = locus(e, np.linspace(min(A.values()) - 1.0, max(A.values()) + 1.0, 41))
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))
    rgb = np.stack([EVs['red'], EVs['green'], EVs['blue']], -1); lo, hi = np.percentile(rgb, [1, 99])
    ax[0].imshow(np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)); ax[0].set_title(f"{tag} {e['obj']} event RGB ({res} m)")
    dk = np.stack([np.clip(-Zs['red'], 0, 12), np.clip(-Zs['green'], 0, 12), np.clip(-Zs['blue'], 0, 12)], -1) / 12
    br = np.stack([np.clip(Zs['red'], 0, 12), np.clip(Zs['green'], 0, 12), np.clip(Zs['blue'], 0, 12)], -1) / 12
    ax[1].imshow(dk); ax[1].set_title('DARK anomalies (R,G,B = band)'); ax[2].imshow(br); ax[2].set_title('BRIGHT anomalies (R,G,B = band)')
    col = {'blue': 'cyan', 'green': 'lime', 'red': 'red', 'nir': 'white'}
    for a in ax:
        px, py = tr(Pl[:, 2], Pl[:, 3]); a.plot(px, py, 'y--', lw=0.6)
        for b, (x, y) in pos.items():
            u, w = tr(x, y); a.add_patch(plt.Circle((u, w), 1500 / res, fill=False, ec=col[b], lw=1.2))
        a.set_xlim(0, rgb.shape[1]); a.set_ylim(rgb.shape[0], 0); a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.show()
    top = []
    for b in ('blue', 'green', 'red'):
        Z = Zs[b]; m = maximum_filter(np.abs(Z), 5) == np.abs(Z)
        idx = np.argwhere(m & (np.abs(Z) > 10))
        top += [(b, float(Z[i, j]), int(j), int(i)) for i, j in idx]
    top.sort(key=lambda t: -abs(t[1]))
    print('RESULT15', tag, 'ref', ref['id'][:30] if ref else None, 'box km', round(2 * hw / 1000, 1), 'strongest:', [(b, round(z, 1), x, y) for b, z, x, y in top[:12]])
for e, tag in [(BEV[0], 'ISS') if BEV else (CAND[0], '#0'), (CAND[8], '#8'), (CAND[0], '#0')]:
    try: await look(e, tag)
    except Exception: print('RESULT15 fail', tag, traceback.format_exc()[-300:].replace('\n', ' | '))
print('RESULT15 ALLDONE')

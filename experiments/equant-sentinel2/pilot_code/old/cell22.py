R = RES5['ISS0']; e = R['e']; sx, sy = R['shift']
lab = {'blue': 'B02 blue', 'nir': 'B08 NIR', 'green': 'B03 green', 'red': 'B04 red', 'rededge1': 'B05', 'swir16': 'B11'}
pos = {b: (R['roots'][b][0][2] + sx, R['roots'][b][0][3] + sy) for b in ('blue', 'nir', 'green', 'red')}
xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
m = 1500.0; bb = (round(min(xs) - m, -1), round(min(ys) - m, -1), round(max(xs) + m, -1), round(max(ys) + m, -1))
its = await items_around(e); its = [it for it in its if epsg_of(it) == e['epsg']]
best = None
for it in its:
    a = await getnpy2(it['assets']['blue']['href'], bb, e['epsg'], 10)
    if a is not None and (best is None or (a[-1] > 0).mean() > best[1]): best = (it, (a[-1] > 0).mean())
it = best[0]
C = {b: (await getnpy2(it['assets'][b]['href'], bb, e['epsg'], 10))[0].astype(np.float32) for b in ('red', 'green', 'blue')}
rgb = np.stack([C['red'], C['green'], C['blue']], -1)
def stretch(x, lo=1, hi=99.7):
    a, b = np.percentile(x[x > 0], [lo, hi]); return np.clip((x - a) / (b - a + 1e-6), 0, 1) ** 0.8
tr = lambda x, y: ((x - bb[0]) / 10, (bb[3] - y) / 10)
P = dens(locus(e, np.linspace(0.45, 1.25, 17), 0.0), n=200)
col = {'blue': '#3cf', 'nir': '#fff', 'green': '#5f5', 'red': '#f44'}
fig = plt.figure(figsize=(16, 8.6), dpi=100); gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 1.35, 1, 1])
ax0 = fig.add_subplot(gs[:, :2]); ax0.imshow(stretch(rgb))
px, py = tr(P[:, 2] + sx, P[:, 3] + sy); ax0.plot(px, py, '--', color='yellow', lw=0.8, alpha=0.7, label='predicted apparent track')
for b, (x, y) in pos.items():
    u, w = tr(x, y); ax0.add_patch(plt.Circle((u, w), 22, fill=False, ec=col[b], lw=1.8)); ax0.text(u + 26, w - 8, lab[b], color=col[b], fontsize=10, weight='bold')
ax0.set_title(f"ISS caught by Sentinel-2B, {str(dt_from_jd(e['jd']))[:19]} UTC, tile {it['id'][4:10]}, 10 m true colour", fontsize=11)
ax0.legend(loc='lower left', fontsize=8); ax0.set_xticks([]); ax0.set_yticks([])
for k, b in enumerate(('blue', 'nir', 'green', 'red')):
    ax = fig.add_subplot(gs[k // 2, 2 + k % 2]); u, w = tr(*pos[b]); u, w = int(u), int(w); h = 40
    crop = rgb[max(0, w - h):w + h, max(0, u - h):u + h]
    ax.imshow(stretch(crop, 1, 99.9), interpolation='nearest'); ax.set_title(f"{lab[b]} predicted spot (800 m)", fontsize=10, color='k')
    ax.add_patch(plt.Circle((h, h), 9, fill=False, ec=col[b] if b != 'nir' else 'k', lw=1.2, ls='--')); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"One object, four places: each band images the ISS at a different moment. Detection score {R['score']:.1f} vs {max(R['nulls']):.1f} for random geometry", fontsize=12)
plt.tight_layout(); plt.show()
print('RESULT21 done', it['id'], bb)
if len(star) == 0:
    t_sl = await gettext(CT + 'gp.php?GROUP=starlink&FORMAT=tle')
    star = parse_tle_text(t_sl)
    objs = [s for s in star + others if s.perigee_km < 800]; OBJ = {o.cat: o for o in objs}
print('RESULT20 starlink orbits', len(star), 'epoch span', str(dt_from_jd(min(s.jdepoch for s in star)))[:16], str(dt_from_jd(max(s.jdepoch for s in star)))[:16])
EV = find_crossings(sents, [o for o in objs if o.name.startswith('STARLINK')], J1 - 1.0, J1, step_s=30)
day = [e for e in EV if e['desc'] and e['sun_el'] > 8 and size_m(e) >= 30]
day.sort(key=lambda e: abs(e['v_al']))
CAND = []
for x in day[:80]:
    if await match_tile(x) and x['cloud'] < 25: CAND.append(x)
    if len(CAND) >= 4: break
print('RESULT20 starlink events:', [(x['obj'], round(x['v_al'], 1), x['item']['id'][:24], round(x['cloud']), round(x['age'], 2)) for x in CAND])
for i, x in enumerate(CAND):
    tag = 'SL%d' % i
    try: await run5(x, tag, 4000.0)
    except Exception: print('RESULT19 fail', tag, traceback.format_exc()[-400:].replace('\n', ' | '))
    if tag in RES5: plot5(tag)
print('RESULT20 ALLDONE')

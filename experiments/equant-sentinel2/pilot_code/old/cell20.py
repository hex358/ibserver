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

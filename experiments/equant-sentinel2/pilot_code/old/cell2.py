import time
from pyodide.http import pyfetch
async def gettext(u):
    r = await pyfetch(u); return await r.string()
CT = 'https://celestrak.org/NORAD/elements/'
t_sup = out['sup'] if 'sup' in out else await gettext(CT + 'supplemental/sup-gp.php?FILE=starlink&FORMAT=tle')
t_act = await gettext(CT + 'gp.php?GROUP=active&FORMAT=tle')
t_vis = await gettext(CT + 'gp.php?GROUP=visual&FORMAT=tle')
t_s2 = await gettext(CT + 'gp.php?NAME=SENTINEL-2&FORMAT=tle')
sents = parse_tle_text(t_s2)
star = parse_tle_text(t_sup)
seen = set(s.cat for s in star)
others = []
for s in parse_tle_text(t_act) + parse_tle_text(t_vis):
    if s.cat in seen or s.name.startswith('STARLINK') or s.name.startswith('SENTINEL-2'): continue
    seen.add(s.cat); others.append(s)
objs = [s for s in star + others if s.perigee_km < 800]
print('sentinels', [s.name for s in sents], '| starlink supgp', len(star), '| others', len(others), '| below 800 km', len(objs))
print('SupGP epochs (UTC):', dt_from_jd(min(s.jdepoch for s in star)), '->', dt_from_jd(max(s.jdepoch for s in star)))

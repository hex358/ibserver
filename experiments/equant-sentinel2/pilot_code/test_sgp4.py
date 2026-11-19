import numpy as np, sgp4lite as S
# Vallado verification case 00005
s = S.Sat('00005','00005','00179.78495062', S._expo('28098-4'), 34.2682, 348.7242, 0.1859667, 331.7664, 19.3264, 10.82419157)
r,v = s.prop(np.array([0.0, 360.0, 720.0]))
np.set_printoptions(precision=6, suppress=True)
print(r); print(v)
print('expected t=0  :', [7022.46529266, -1400.08296755, 0.03995155], [1.893841015, 6.405893759, 4.534807250])
print('expected t=360:', [-7154.03120202, -3783.17682504, -3536.19412294], [4.741887409, -4.151817765, -2.093935425])
print(S.jday(2000,1.0), 'expect 2451544.5')
txt = open('tle_raw.txt').read()
sats = S.parse_collapsed(txt)
print(len(sats), 'sats parsed;', [x.name for x in sats[:5]], '...', sats[-1].name)
iss = [x for x in sats if x.cat=='25544'][0]
r,v = iss.prop(np.array([0.0])); print('ISS |r|-RE', np.linalg.norm(r)-6378.135, 'km |v|', np.linalg.norm(v))

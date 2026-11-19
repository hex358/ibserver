import math
def latlon_to_utm(lat, lon, zone=None):
    a=6378137.0; f=1/298.257223563; k0=0.9996
    if zone is None: zone=int((lon+180)//6)+1
    lon0=math.radians((zone-1)*6-180+3)
    e2=f*(2-f); ep2=e2/(1-e2)
    phi=math.radians(lat); lam=math.radians(lon)
    N=a/math.sqrt(1-e2*math.sin(phi)**2); T=math.tan(phi)**2; C=ep2*math.cos(phi)**2; A=math.cos(phi)*(lam-lon0)
    M=a*((1-e2/4-3*e2**2/64-5*e2**3/256)*phi-(3*e2/8+3*e2**2/32+45*e2**3/1024)*math.sin(2*phi)+(15*e2**2/256+45*e2**3/1024)*math.sin(4*phi)-(35*e2**3/3072)*math.sin(6*phi))
    x=k0*N*(A+(1-T+C)*A**3/6+(5-18*T+T*T+72*C-58*ep2)*A**5/120)+500000
    y=k0*(M+N*math.tan(phi)*(A*A/2+(5-T+9*C+4*C*C)*A**4/24+(61-58*T+T*T+600*C-330*ep2)*A**6/720))
    if lat<0: y+=10000000
    return zone,x,y
if __name__=='__main__':
    import sys
    for p in [(40.67889,26.80197),(40.74689,26.87319),(40.81487,26.94456),(40.81855,26.96759),(40.67516,26.77899)]:
        print(p, latlon_to_utm(*p, zone=35))

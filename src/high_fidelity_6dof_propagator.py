import numpy as np
import scipy as sp
import spiceypy as spice
import matplotlib.pyplot as plt

spice.furnsh('naif0012.tls')
spice.furnsh('de440s.bsp')
spice.furnsh('gm_de440.tpc')
spice.furnsh('pck00010.tpc')

# initial epoch
t0=spice.str2et('12 AUG 2010 12:00 UTC')

# initial state [km, km/s]
x0=np.zeros(6)
x0[:3]=np.array([4622.232026629, 5399.3369588058, -0.0212138165769957])
x0[3:]=np.array([0.812221125483763,-0.721512914578826,7.42665302729053])
"""x0[0]=spice.bodvrd('Earth','RADII',3)[1][0]+515. 
# x0[4]=np.sqrt(spice.bodvrd('Earth','GM',1)[1][0]/x0[0])
x0[4]=np.sqrt(spice.bodvrd('Earth','GM',1)[1][0]/x0[0])*np.cos(np.deg2rad(97.5))
x0[5]=np.sqrt(spice.bodvrd('Earth','GM',1)[1][0]/x0[0])*np.sin(np.deg2rad(97.5))"""

# initial attitude (DCM)
A0=np.eye(3)

# initial angular velocity [rad/s] (body frame)
w0=np.array([0.,0.,0.])

# geometrical properties
m=94.                                   # mass [kg]
J=np.diag([6.6583, 7.8333, 8.8517])     # main inertial axis [kg*m2] (body frame)
md=np.array([0.37, 0.32, 0.28])         # magnetic dipole [A*m2] (body frame)
S=.7*.8*1.2                             # spacecraft surface [m2] (max surface and 1.2 safety factor)
cD=2.                                   # aerodynamic drag coefficient

# propagation settings
NO=3   # number of orbits
# npo=10  # number of points per orbit
nM=3    # magnetic model order nM<=5

#--------------------------------------------------------------------------------------

mu=spice.bodvrd('Earth','GM',1)[1][0]                       # gravitational constant

LU=1/(2/np.linalg.norm(x0[:3])-np.dot(x0[3:],x0[3:])/mu)    # Length Unit, semi-major axis
TU=np.sqrt(LU**3/mu)                                        # Time Unit, set to mu=1
MU=m                                                        # Mass Unit, spacecraft mass

REs=spice.bodvrd('Earth','RADII',3)[1]/LU
RE=REs[0]
f=(RE-REs[2])/RE

# constants and adimensionalization units
cun={
    'LU' : LU,          # length unit
    'VU' : LU/TU,       # velocity unit
    'TU' : TU,          # time unit
    'MU' : MU,          # mass unit
    't0' : t0,          # initial time
    'RE' : RE,
    'f'  : f
}

# inertial matrix inverse
IJ=np.linalg.inv(J)

# optimized cross product
def crossp(a, b, out):
    out[0]=a[1]*b[2]-a[2]*b[1]
    out[1]=a[2]*b[0]-a[0]*b[2]
    out[2]=a[0]*b[1]-a[1]*b[0]

def car2kep(x):
    # cartesian coordinates to keplerian parameters

    r=x[:3]
    v=x[3:]

    a=1/(2/np.linalg.norm(r)-np.dot(v,v)/mu)

    # h=np.cross(r,v)
    h=np.empty(3)
    crossp(r,v,h)

    i=np.acos(h[2]/np.linalg.norm(h))

    # n=np.cross(np.array([0,0,1]),h)
    n=np.empty(3)
    crossp(np.array([0,0,1]),h,n)

    if np.linalg.norm(n)==0:
        n=np.array([1,0,0])

    Om=np.acos(n[0]/np.linalg.norm(n)) if n[1]>=0 else 2*np.pi-np.acos(n[0]/np.linalg.norm(n))

    temp=np.empty(3)
    crossp(v,h,temp)

    ee=temp/mu-r/np.linalg.norm(r)
    e=np.linalg.norm(ee)

    if e!=0:

        if np.linalg.norm(ee)==0:
            ee=np.array([1,0,0])

        om=np.acos(np.dot(n,ee)/(np.linalg.norm(n)*e)) if ee[2]>=0 else 2*np.pi-np.acos(np.dot(n,ee)/(np.linalg.norm(n)*e))

        nu=np.acos(np.dot(ee,r)/(e*np.linalg.norm(r))) if np.dot(r,v)>=0 else 2*np.pi-np.acos(np.dot(ee,r)/(e*np.linalg.norm(r)))

        E=2*np.atan2(np.tan(nu/2)/np.sqrt((1+e)),np.sqrt((1-e)))
        th=2*np.atan2(np.tan(E/2)/np.sqrt((1+e)),np.sqrt((1-e)))

    else:

        om=0.
        th=np.acos(r[0]/np.linalg.norm(r)) if r[1]>=0 else 2*np.pi-np.acos(r[0]/np.linalg.norm(r))

    kep_param=np.zeros(6)
    kep_param[0]=a
    kep_param[1]=e
    kep_param[2]=i
    kep_param[3]=Om
    kep_param[4]=om
    kep_param[5]=th

    return kep_param

# initial orbital elements
kp=car2kep(x0)
print(f'a:  {kp[0]:.4f} km')
print(f'e:  {kp[1]:.4f}')
print(f'i:  {np.rad2deg(kp[2]):.4f} deg')
print(f'OM: {np.rad2deg(kp[3]):.4f} deg')
print(f'om: {np.rad2deg(kp[4]):.4f} deg')
print(f'th: {np.rad2deg(kp[5]):.4f} deg')

# adimensional initial state
x0[:3]/=cun['LU']
x0[3:]/=cun['VU']

# integration time interval
T=2*np.pi*np.sqrt(cun['LU']**3/mu)
tf=t0+NO*T
tspan=[0, (tf-t0)/cun['TU']]

# complete state
z0=np.concatenate([np.reshape(A0,9),w0])
y0=np.concatenate([x0,z0])


# magnetic field misc

# coefficients
GC=np.array([
        np.array([
            [-29615., -1728., 0., 0., 0., 0.],
            [0., 5186., 0., 0., 0., 0.]
        ]).T,
        np.array([
            [-2267., 3072., 1672., 0., 0., 0.],
            [0., -2478., -458., 0., 0., 0.]
        ]).T,
        np.array([
            [1341., -2290., 1253., 715., 0., 0.],
            [0., -277., 296., -492., 0., 0.]
        ]).T,
        np.array([
            [935., 787., 251., -405., 110., 0.],
            [0., 272., -232., 119., -304., 0.]
        ]).T,
        np.array([
            [-217., 351., 222., -131., -169., -12.],
            [0., 44., 172., -134., -40., 107.]
        ]).T
    ])

def Snm(n,m):

    if n+m==0:
        S=1
    elif n>=1 and m==0:
        S=Snm(n-1,m)*(2*n-1)/n
    else:
        S=Snm(n,m-1)*np.sqrt((float(m==1)+1)*(n-m+1)/(n+m))

    return S

# normalized coefficients
GCS=GC.copy()

for i in range(nM):
    n=i+1
    for j in range(n+1):
        m=j
        GCS[i,j]=GC[i,j]*Snm(n,m)

def Knm(n,m):

    if n==1:
        K=0.
    else:
        K=((n-1)**2-m**2)/((2*n-1)*(2*n-3))

    return K

# # gauss-legendre polynomials
# def glp(n,m,theta):

#     if n+m==0:
#         P=1.
#     elif n==m:
#         P=np.sin(theta)*glp(n-1,m-1,theta)
#     else:
#         if n==1 or n==m+1:
#             P=np.cos(theta)*glp(n-1,m,theta)
#         else:
#             P=np.cos(theta)*glp(n-1,m,theta)-Knm(n,m)*glp(n-2,m,theta)

#     return P

# def nglp(n,m,costh,sinth):

#     if n==m:
#         P=sinth**n
#     else:
#         P=costh*(n-1,m,costh,sinth)-Knm(n,m)*(n-2,m,costh,sinth)

#     return P

# # differentiated gauss-legendre polynomials
# def dglp(n,m,theta):

#     if n+m==0:
#         dP=0.
#     elif n==m:
#         dP=np.sin(theta)*dglp(n-1,n-1,theta)+np.cos(theta)*glp(n-1,n-1,theta)
#     else:
#         if n==1 or n==m+1:
#             dP=np.cos(theta)*dglp(n-1,m,theta)-np.sin(theta)*glp(n-1,m,theta)
#         else:
#             dP=np.cos(theta)*dglp(n-1,m,theta)-np.sin(theta)*glp(n-1,m,theta)-Knm(n,m)*dglp(n-2,m,theta)

#     return dP



# air drag misc

# dimensional air density [kg/m3]
def dens(r):

    h=np.linalg.norm(r)*cun['LU']-spice.bodvrd('Earth','RADII',3)[1][0]

    if h>0 and h<=25:
        h0=0
        rho0=1.225
        H=7.249
    elif h>25 and h<=30:
        h0=25
        rho0=3.899e-2
        H=6.349
    elif h>30 and h<=40:
        h0=30
        rho0=1.774e-2
        H=6.682
    elif h>40 and h<=50:
        h0=40
        rho0=3.972e-3
        H=7.554
    elif h>50 and h<=60:
        h0=50
        rho0=1.057e-3
        H=8.382
    elif h>60 and h<=70:
        h0=60
        rho0=3.206e-4
        H=7.714
    elif h>70 and h<=80:
        h0=70
        rho0=8.770e-5
        H=6.549
    elif h>80 and h<=90:
        h0=80
        rho0=1.905e-5
        H=5.799
    elif h>90 and h<=100:
        h0=90
        rho0=3.396e-6
        H=5.382
    elif h>100 and h<=110:
        h0=100
        rho0=5.297e-7
        H=5.877
    elif h>110 and h<=120:
        h0=110
        rho0=9.661e-8
        H=7.263
    elif h>120 and h<=130:
        h0=120
        rho0=2.438e-8
        H=9.473
    elif h>130 and h<=140:
        h0=130
        rho0=8.484e-9
        H=12.636
    elif h>140 and h<=150:
        h0=140
        rho0=3.845e-9
        H=16.149
    elif h>150 and h<=180:
        h0=150
        rho0=2.070e-9
        H=22.523
    elif h>180 and h<=200:
        h0=180
        rho0=5.464e-10
        H=29.740
    elif h>200 and h<=250:
        h0=200
        rho0=2.789e-10
        H=37.105
    elif h>250 and h<=300:
        h0=250
        rho0=7.248e-11
        H=45.546
    elif h>300 and h<=350:
        h0=300
        rho0=2.418e-11
        H=53.628
    elif h>350 and h<=400:
        h0=350
        rho0=9.158e-12
        H=53.298
    elif h>400 and h<=450:
        h0=400
        rho0=3.725e-12
        H=58.515
    elif h>450 and h<=500:
        h0=450
        rho0=1.585e-12
        H=60.828
    elif h>500 and h<=600:
        h0=500
        rho0=6.967e-13
        H=63.822
    elif h>600 and h<=700:
        h0=600
        rho0=1.454e-13
        H=71.835
    elif h>700 and h<=800:
        h0=700
        rho0=3.614e-14
        H=88.667
    elif h>800 and h<=900:
        h0=800
        rho0=1.170e-14
        H=124.64
    elif h>900 and h<=1000:
        h0=900
        rho0=5.245e-15
        H=181.05
    else:
        h0=1000
        rho0=3.019e-15
        H=268.00

    rho=rho0*np.exp(-(h-h0)/H)
    return rho



# propagation functions

# orbital dynamics
def twobp(t,x):
    # 2BP adimensional dynamics

    r=x[:3]     # geocentric position
    v=x[3:]     # geocentric velocity

    rn=np.linalg.norm(r)

    dx=np.zeros(6)
    dx[:3]=v
    dx[3:]=-r/rn**3

    return dx

def aj2(t,x):
    # adimensional j2 acceleration

    r=x[:3]                         # geocentric position
    td=t*cun['TU']+cun['t0']      # dimensional epoch [s]

    rn=np.linalg.norm(r)

    RR=spice.pxform('J2000','IAU_EARTH',td)     # ECI to ECEF rotation matrix

    re=RR@r         # ECEF adimensional position
    xx,yy,zz=re     # ECEF adimensional position components

    # RE=spice.bodvrd('Earth','RADII',3)[1][0]/cun['LU'] # adimentional earth equatorial radius
    RE=cun['RE']
    j2=1.08262668e-3

    kf=3/2*j2*RE**2/rn**5
    kc=5*(zz/rn)**2

    aj2e=kf*np.array([      # ECEF adimensional acceleration
        xx*(kc-1),
        yy*(kc-1),
        zz*(kc-3),
    ])

    aj2=RR.T@aj2e           # ECI adimensional acceleration

    return aj2

def a3b(t,x):
    # adimensional 3rd body acceleration

    r=x[:3]     # geocentric position

    bd='Moon'
    mu3=spice.bodvrd(bd,'GM',1)[1][0]*cun['TU']**2/cun['LU']**3   # adimensional 3rd body grav constant
    td=t*cun['TU']+cun['t0']                                      # dimensional epoch [s]

    r3E=spice.spkpos(bd,td,'J2000','NONE','Earth')[0]/cun['LU']    # adimensional 3rd body - Earth position

    rs3=r-r3E                                                       # adimensional SC - 3rd body position

    a3b=mu3*(rs3/np.linalg.norm(rs3)**3-r3E/np.linalg.norm(r3E)**3) # 3rd body acceleration

    return a3b

def ad(t,x):
    # adimensional drag acceleration

    r=x[:3]     # geocentric position
    v=x[3:]     # geocentric velocity

    TE=23*3600+56*60+4          # Earth revolution period [s]
    wE=2*np.pi/TE*cun['TU']    # adimensional Earth angular velocity

    # vw=np.cross(wE*np.array([0,0,1]),r) # adimensional wind velocity
    vw=np.empty(3)
    crossp(np.array([0,0,wE]),r,vw)

    vr=v-vw                             # adimensional relative wind velocity
    vrn=np.linalg.norm(vr)              

    K=S*cD*dens(r)*1e3*cun['LU']/cun['MU']    # adimensional aerodynamic coefficient

    ad=-0.5*K*vrn*vr

    return ad

# attitude dynamics
def euler(t,z):
    # dimensional Euler equations

    A=np.reshape(z[:9],(3,3))   # DCM
    w=z[9:]                     # dimensional angular velocity [rad/s]

    # orthonormalization check
    U,_,Vt=np.linalg.svd(A)
    A=U@Vt

    wcr=np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

    dA=-wcr@A                               # DCM update
    # dw=np.linalg.inv(J)@(np.cross(J@w,w))   # angular velocity update
    # T=np.cross(J@w,w)   # Torque [N/m]
    T=np.empty(3)
    crossp(J@w,w,T)

    dAT=np.concatenate([np.reshape(dA,9),T])

    return dAT

def gdt(t,x,z):
    # dimensional gravity disturbance torque

    r=x[:3]                     # geocentric position
    A=np.reshape(z[:9],(3,3))   # DCM, ECI to BODY rotation matrix

    rn=np.linalg.norm(r)*cun['LU']     # dimensional geocentric distance
    rb=A.T@r                    # body frame geocentric position
    rbu=rb/np.linalg.norm(rb)   # body frame position direction

    temp=np.empty(3)
    crossp(rbu,J@rbu,temp)
    Tgg=3*mu/(rn**3)*temp    # torque [N]
    # Tgg=3*mu/(rn**3)*np.cross(rbu,J@rbu)    # torque [N]
    # dw=np.linalg.inv(J)@Tgg                 # angular acceleration [rad/s2]
    T=Tgg                 # torque [N/m]

    return T

def mdt(t,x,z):
    # dimensional magnetic disturbance torque

    r=x[:3]     # geocentric position
    rn=np.linalg.norm(r)

    A=np.reshape(z[:9],(3,3))   # DCM

    td=t*cun['TU']+cun['t0']                          # dimensional time
    f=cun['f']                             # Earth polar flattening
    # RE=spice.bodvrd('Earth','RADII',3)[1][0]/cun['LU'] # adimensional earth equatorial radius
    RE=cun['RE']

    re=spice.pxform('J2000','IAU_EARTH',td)@r   # ECEF adimensional position
    lon,lat,_=spice.recgeo(re,RE,f)

    the=np.pi/2-lat
    phi=lon

    costh=np.cos(the)
    sinth=np.sin(the)

    cosphv=np.array([np.cos(m*phi) for m in range(nM+1)])
    sinphv=np.array([np.sin(m*phi) for m in range(nM+1)])

    # Gaussian normalized associated Legendre polynomials
    P=np.diag(sinth**np.arange(nM+1))
    P[np.arange(1,nM+1),np.arange(nM)]=costh*P[np.arange(nM),np.arange(nM)]

    dP=np.zeros((nM+1,nM+1))
    dP[np.arange(nM)+1,np.arange(nM)+1]=(np.arange(nM)+1)*costh*sinth**np.arange(nM)
    dP[np.arange(1,nM+1),np.arange(nM)]=costh*dP[np.arange(nM),np.arange(nM)]-sinth*P[np.arange(nM),np.arange(nM)]

    for m in range(nM-1):
        for n in np.arange(2+m,nM+1):    
            P[n,m]=costh*P[n-1,m]-Knm(n,m)*P[n-2,m]
            dP[n,m]=costh*dP[n-1,m]-sinth*P[n-1,m]-Knm(n,m)*dP[n-2,m]

    br=bt=bp=0

    # for i in range(nM):
    #     n=i+1
    #     for j in range(n+1):
    #         m=j

    #         br+=(RE/rn)**(n+2)*(n+1)*(GCS[i,j,0]*np.cos(m*phi)+GCS[i,j,1]*np.sin(m*phi))*glp(n,m,the)
    #         bt+=-(RE/rn)**(n+2)*(GCS[i,j,0]*np.cos(m*phi)+GCS[i,j,1]*np.sin(m*phi))*dglp(n,m,the)
    #         bp+=-1/np.sin(the)*(RE/rn)**(n+2)*m*(-GCS[i,j,0]*np.sin(m*phi)+GCS[i,j,1]*np.cos(m*phi))*glp(n,m,the)

    # brn=btn=bpn=0

    for i in range(nM):
        n=i+1

        kc=(RE/rn)**(n+2)

        for j in range(n+1):
            m=j

            br+=kc*(n+1)*(GCS[i,j,0]*cosphv[m]+GCS[i,j,1]*sinphv[m])*P[n,m]
            bt+=-kc*(GCS[i,j,0]*cosphv[m]+GCS[i,j,1]*sinphv[m])*dP[n,m]
            bp+=-kc*m*(-GCS[i,j,0]*sinphv[m]+GCS[i,j,1]*cosphv[m])*P[n,m]

    bp/=sinth

    ru=re/rn # --------------------------------------> switched r_eci for r_ecef
    xe,ye,ze=re

    pv=np.array([-ye,xe,0.])
    pu=pv/np.linalg.norm(pv)

    tv=np.array([xe*ze, ye*ze, -xe*xe-ye*ye])
    tu=tv/np.linalg.norm(tv)

    B=(br*ru+bt*tu+bp*pu)*1e-9

    # Tm=np.cross(md,A.T@B)       # torque [N]
    Tm=np.empty(3)
    crossp(md,A.T@B,Tm)

    # dw=np.linalg.inv(J)@Tm      # angular acceleration [rad/s2]
    T=Tm      # torque [N/m]

    return T

""""- both aj2 and mdt use ECI to ECEF rotation matrix,
    which can be updated separately on larger time stes"""

# complete dynamics
def fdyn(t,y):
    # full dynamics: adimensional orbital dynamics + dimensional attitude dynamics

    x=y[:6]     # state parameters
    z=y[6:]     # attitude parameters

    # adimensional orbital dynamics

    dx=np.zeros(len(x))
    dx+=twobp(t,x)      # 2BP state dynamics
    dx[3:]+=aj2(t,x)    # J2 acceleration
    dx[3:]+=a3b(t,x)    # 3rd body acceleration
    dx[3:]+=ad(t,x)     # drag acceleration

    # dimensional attitude dynamics

    dz=np.zeros(len(z))
    dz+=euler(t,z)      # DCM kin + Euler equations
    dz[9:]+=gdt(t,x,z)  # gravity disturbance torque
    dz[9:]+=mdt(t,x,z)  # magnetic disturbance torque

    # angular velocity rate of change
    dz[9:]=IJ@dz[9:]

    # dimensional derivative correction (d/dt)_adim = TU*(d/dt)_dim
    dz*=cun['TU']

    dy=np.concatenate([dx,dz])

    return dy


sol=sp.integrate.solve_ivp(
    fun=fdyn,
    t_span=tspan,
    y0=y0,
    atol=1e-12,
    rtol=1e-12,
    method='DOP853'
    )

t_plot=sol.t
y_plot=sol.y
y_plot[:3]*=cun['LU']
y_plot[3:6]*=cun['VU']

tt=t_plot*cun['TU']
rr=y_plot[:3]

dd=np.array([np.linalg.norm(rr[:,i]) for i in range(len(t_plot))])

ww=np.array([np.linalg.norm(y_plot[-3:,i]) for i in range(len(t_plot))])

plt.figure()
plt.plot(tt,dd)
plt.legend(['r'])

plt.figure()
plt.plot(tt,ww)
plt.legend(['w'])
plt.show()
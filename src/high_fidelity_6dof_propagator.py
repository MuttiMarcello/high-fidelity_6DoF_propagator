import numpy as np
import scipy as sp
import spiceypy as spice
import matplotlib.pyplot as plt

spice.furnsh('naif0012.tls')
spice.furnsh('de440s.bsp')
spice.furnsh('gm_de440.tpc')
spice.furnsh('pck00010.tpc')

#initial state

t0=spice.str2et('12 AUG 2010 12:00 UTC')

x0=np.zeros(6)
x0[:3]=[4622.232026629, 5399.3369588058, -0.0212138165769957]
x0[3:]=[0.812221125483763,-0.721512914578826,7.42665302729053]
# x0[0]=spice.bodvrd('Earth','RADII',3)[1][0]+515
# x0[4]=np.sqrt(spice.bodvrd('Earth','GM',1)[1][0]/x0[0])
# # x0[4]=np.sqrt(spice.bodvrd('Earth','GM',1)[1][0]/x0[0])*np.cos(np.deg2rad(97.5))
# # x0[5]=np.sqrt(spice.bodvrd('Earth','GM',1)[1][0]/x0[0])*np.sin(np.deg2rad(97.5))

A0=np.eye(3)
w0=np.array([0.01,0.,0.]) # in body frame

# geometrical properties

J=np.diag([6.6583, 7.8333, 8.8517])*1e-6    # kg*m2
md=np.array([0.37, 0.32, 0.28])             # A*m2

# propagation settings

NO=3   # number of orbits
npo=25  # number of points per orbit

nM=3    # magnetic model order nM<=5

#----------

mu=spice.bodvrd('Earth','GM',1)[1][0]

LU=1/(2/np.linalg.norm(x0[:3])-np.dot(x0[3:],x0[3:])/mu)
TU=np.sqrt(LU**3/mu)
MU=94

adun={
    'LU' : LU,          # length unit
    'VU' : LU/TU,       # velocity unit
    'TU' : TU,          # time unit
    'AVU': 1/TU,        # angular velocity unit
    'MU' : MU,          # mass unit
    'IU' : MU*LU**2,    # inertial unit
    't0' : t0           # initial time
}

x0[:3]/=adun['LU']
x0[3:]/=adun['VU']

T=2*np.pi*np.sqrt(LU**3/mu)
tf=t0+NO*T

w0/=adun['AVU']
J/=adun['IU']

z0=np.concatenate([np.reshape(A0,9),w0])

y0=np.concatenate([x0,z0])

# magnetic field shiss

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

def glp(n,m,theta):

    if n+m==0:
        P=1.
    elif n==m:
        P=np.sin(theta)*glp(n-1,m-1,theta)
    else:
        if n==1 or n==m+1:
            P=np.cos(theta)*glp(n-1,m,theta)
        else:
            P=np.cos(theta)*glp(n-1,m,theta)-Knm(n,m)*glp(n-2,m,theta)

    return P

def dglp(n,m,theta):

    if n+m==0:
        dP=0.
    elif n==m:
        dP=np.sin(theta)*dglp(n-1,n-1,theta)+np.cos(theta)*glp(n-1,n-1,theta)
    else:
        if n==1 or n==m+1:
            dP=np.cos(theta)*dglp(n-1,m,theta)-np.sin(theta)*glp(n-1,m,theta)
        else:
            dP=np.cos(theta)*dglp(n-1,m,theta)-np.sin(theta)*glp(n-1,m,theta)-Knm(n,m)*dglp(n-2,m,theta)

    return dP

# functions

"""def stt_prop(t,x):

    r=x[:3]
    v=x[3:]

    # 2bp dynamics

    a2bp=np.zeros(6)
    a2bp[:3]=v
    a2bp[3:]=-r/np.linalg.norm(r)**3

    # third-body perturbation

    mu3=spice.bodvrd('Moon','GM',1)[1][0]*adun['TU']**2/adun['LU']**3
    td=t*adun['TU']+adun['t0']

    rE3=spice.spkpos('Earth',td,'J2000','NONE','Moon')[0]/adun['LU']
    rs3=r+rE3

    a3b=mu3*(rs3/np.linalg.norm(rs3)**3-rE3/np.linalg.norm(rE3)**3)

    # J2 perturbation

    RE=spice.bodvrd('Earth','RADII',3)[1][0]/adun['LU']
    j2=1.08262668e-3

    aj2=-3/2*j2*RE**2/np.linalg.norm(r)**5*np.array([
        r[0]*(1-5*r[2]**2/np.dot(r,r)),
        r[1]*(1-5*r[2]**2/np.dot(r,r)),
        r[2]*(3-5*r[2]**2/np.dot(r,r))
    ])

    dx=np.zeros(6)
    dx+=a2bp
    # dx[3:]+=aj2
    # dx[3:]+=a3b
    dx[3:]+=aj2+a3b

    return dx

def att_prop(t,z):

    A=np.reshape(z[:9],(3,3))

    # A=3/2*A-1/2*A@np.transpose(A)@A
    U,_,Vt=np.linalg.svd(A)
    A=U@Vt

    w=z[9:]

    wc=np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

    dA=-wc@A
    dAv=np.reshape(dA,9)

    dw=np.linalg.inv(J)@(np.cross(J@w,w))    

    dz=np.concatenate([dAv,dw])
    return dz"""

def full_prop(t,y):

    x=y[:6]
    z=y[6:]

    # orbital dynamics

    r=x[:3]
    v=x[3:]

    # 2bp dynamics

    a2bp=np.zeros(6)
    a2bp[:3]=v
    a2bp[3:]=-r/np.linalg.norm(r)**3

    # third-body perturbation

    mu3=spice.bodvrd('Moon','GM',1)[1][0]*adun['TU']**2/adun['LU']**3
    td=t*adun['TU']+adun['t0']

    rE3=spice.spkpos('Earth',td,'J2000','NONE','Moon')[0]/adun['LU']
    rs3=r+rE3

    a3b=mu3*(rs3/np.linalg.norm(rs3)**3-rE3/np.linalg.norm(rE3)**3)

    # J2 perturbation
    RII=spice.bodvrd('Earth','RADII',3)[1]/adun['LU']
    RE=RII[0]
    j2=1.08262668e-3

    aj2=-3/2*j2*RE**2/np.linalg.norm(r)**5*np.array([
        r[0]*(1-5*r[2]**2/np.dot(r,r)),
        r[1]*(1-5*r[2]**2/np.dot(r,r)),
        r[2]*(3-5*r[2]**2/np.dot(r,r))
    ])

    dx=np.zeros(6)
    dx+=a2bp
    # dx[3:]+=aj2
    # dx[3:]+=a3b
    dx[3:]+=aj2+a3b

    # attitude dynamics

    A=np.reshape(z[:9],(3,3))

    # A=3/2*A-1/2*A@np.transpose(A)@A
    U,_,Vt=np.linalg.svd(A)
    A=U@Vt

    w=z[9:]

    wc=np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

    dA=-wc@A
    dAv=np.reshape(dA,9)

    dw=np.linalg.inv(J)@(np.cross(J@w,w))

    # gravity torque

    rb=np.transpose(A)@r
    rbu=rb/np.linalg.norm(rb)

    Tgg=3/(np.linalg.norm(r)**3)*np.cross(rbu,J@rbu)

    """# ru=r/np.linalg.norm(r)
    # ku=np.cross(r,v)/np.linalg.norm(np.cross(r,v))
    # yu=np.cross(ku,ru)
    #
    # A_lvlh_i=np.array([ru,yu,ku])
    # A_b_lvlh=A@np.transpose(A_lvlh_i)
    #
    # c=np.array([
    #     1,
    #     (A_b_lvlh[1,0]-A_b_lvlh[0,1])/2,
    #     (A_b_lvlh[0,2]-A_b_lvlh[2,0])/2
    # ])
    #
    # Tgg=np.array([
    #     3/np.linalg.norm(r)**3*(J[2,2]-J[1,1])*c[2]*c[1],
    #     3/np.linalg.norm(r)**3*(J[0,0]-J[2,2])*c[0]*c[2],
    #     3/np.linalg.norm(r)**3*(J[1,1]-J[0,0])*c[1]*c[0]
    # ])"""
    
    dw+=np.linalg.inv(J)@Tgg

    # magnetic torque

    RP=RII[2]
    Ef=(RE-RP)/RE

    r_ecef=spice.pxform('J2000','IAU_EARTH',td)@r
    lon,lat,_=spice.recgeo(r_ecef,RE,Ef)
    
    re=np.linalg.norm(r_ecef)
    the=np.pi/2-lat
    phi=lon

    br=bt=bp=0

    for i in range(nM):
        n=i+1
        for j in range(n+1):
            m=j

            br+=(RE/re)**(n+2)*(n+1)*(GCS[i,j,0]*np.cos(m*phi)+GCS[i,j,1]*np.sin(m*phi))*glp(n,m,the)
            bt+=-(RE/re)**(n+2)*(GCS[i,j,0]*np.cos(m*phi)+GCS[i,j,1]*np.sin(m*phi))*dglp(n,m,the)
            bp+=-1/np.sin(the)*(RE/re)**(n+2)*m*(-GCS[i,j,0]*np.sin(m*phi)+GCS[i,j,1]*np.cos(m*phi))*glp(n,m,the)

    ru=r/np.linalg.norm(r)
    pu=np.cross(np.array([0,0,1]),ru)/np.linalg.norm(np.cross(np.array([0,0,1]),ru))
    tu=np.cross(pu,ru)

    B=(br*ru+bt*tu+bp*pu)
    B*=1e-9

    Tm=np.cross(md,A.T@B)*1e-6      # 1e-6 kg*m2/s2 -> kg km2/s2
    Tm*=adun['TU']**2/adun['IU']

    dw+=np.linalg.inv(J)@Tm

    # composition

    dz=np.concatenate([dAv,dw])

    dy=np.concatenate([dx,dz])
    return dy

"""# sol=sp.integrate.solve_ivp(
#     fun=stt_prop,
#     t_span=[0, (tf-t0)/adun['TU']],
#     y0=x0,
#     atol=1e-12,
#     rtol=1e-12,
#     method='DOP853',
#     dense_output='True'
# )

# t_plot=np.linspace(0,(tf-t0)/adun['TU'],NO*npo,1)
# y_plot=sol.sol(t_plot)

# plt.figure()
# ax=plt.axes(projection='3d')
# ax.plot3D(y_plot[0],y_plot[1],y_plot[2])
# plt.show()

# print(f'{y_plot[0,0]*adun['LU']:.5f} {y_plot[1,0]*adun['LU']:.5f} {y_plot[2,0]*adun['LU']:.5f}')
# print(f'{y_plot[0,-1]*adun['LU']:.5f} {y_plot[1,-1]*adun['LU']:.5f} {y_plot[2,-1]*adun['LU']:.5f}')
# print((y_plot[:3,-1]-y_plot[:3,0])*adun['LU'])

# sol=sp.integrate.solve_ivp(
#     fun=att_prop,
#     t_span=[0, (tf-t0)/adun['TU']],
#     y0=z0,
#     atol=1e-12,
#     rtol=1e-12,
#     method='DOP853',
#     dense_output='True'
# )
#
# t_plot=np.linspace(0,(tf-t0)/adun['TU'],NO*npo,1)
# z_plot=sol.sol(t_plot)"""

sol=sp.integrate.solve_ivp(
    fun=full_prop,
    t_span=[0, (tf-t0)/adun['TU']],
    y0=y0,
    atol=1e-12,
    rtol=1e-12,
    method='DOP853',
    dense_output='True'
)

t_plot=np.linspace(0,(tf-t0)/adun['TU'],NO*npo,1)
y_plot=sol.sol(t_plot)

tt=t_plot*adun['TU']
rr=y_plot[:3]*adun['LU']

plt.figure()
plt.plot(tt,rr[0])
plt.plot(tt,rr[1])
plt.plot(tt,rr[2])
plt.legend(['rx','ry','rz'])
plt.show()

ww=y_plot[-3:]*adun['AVU']

plt.figure()
plt.plot(tt,ww[0])
plt.plot(tt,ww[1])
plt.plot(tt,ww[2])
plt.legend(['wx','wy','wz'])
plt.show()

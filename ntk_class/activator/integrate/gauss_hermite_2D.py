import numpy as np
import math
from scipy.special import roots_hermite

global Gh2_verbose
Gh2_verbose=True
def verbose(flag):
    global Gh2_verbose
    Gh2_verbose=flag

# u^T x u = w^T [[-1,0],[0,-1]] w by u=diag_coord(x) @ w
# x must be negative definite
def diag_coord(x):
    p0=np.linalg.eigh(x)
    l0=p0[0][0]
    l1=p0[0][1]
    p=p0[1]
    q = p @ np.array([[1/np.sqrt(-l0),0],[0,1/np.sqrt(-l1)]])
    return [q,p0[0]]
"""
x=np.array([[-18,23],[23,-30]])
q=diag_coord(x)
print(q.T @ x @ q)
"""

def integrate_2D_normal_main(f,x,q,degx,degy):
    [px,wx]=roots_hermite(degx)
    [py,wy]=roots_hermite(degy)
    nx=len(px)
    ny=len(py)
    val=0
    for i in range(nx):
        for j in range(ny):
            val = val + f(q @ np.array([px[i],py[j]]))*wx[i]*wy[j]
    return val*np.linalg.det(q)

def integrate_2D_normal_adaptive(f,x,qq,degx,accuracy_digits):
    global Gh2_verbose
    max_retry=10
    magnify=2  # magnifier of degx, degy
    ad=accuracy_digits
    q=qq[0]
    l0=qq[1][0]
    l1=qq[1][1]
    ratio=l0/l1
    degy=math.floor(degx*(1+math.log(ratio)))
    for i in range(max_retry):
        val_orig=integrate_2D_normal_main(f,x,q,degx,degy)
        val_new=integrate_2D_normal_main(f,x,q,degx*2,degy*2)
        degx=degx*magnify; degy=degy*magnify
        if Gh2_verbose: print('[i,degx,degy]=',[i,degx,degy])
        rel_error=np.abs((val_new-val_orig)/val_orig)
        if Gh2_verbose: print('relative error by doubling the mesh=',rel_error)
        if (rel_error < 1/10**ad): return val_new
    print('Warning: the interation of integrate_2D_normal_adaptive reached  to {%max_retry}. There may be an error in value.')
    return val_new

# Evaluate numerically integral_{R^2} f(u) exp(u^T x u) du by Gauss-Hermite quadrature
def integrate_2D_normal(f,x,ad=3,degx=100,degy=100):
    global Gh2_verbose
    qq=diag_coord(x)
    q=qq[0]
    if Gh2_verbose: print('integrate_2D_normal [ad,degx,degy]=',[ad,degx,degy])
    if (ad > 0): 
        return integrate_2D_normal_adaptive(f,x,qq,degx,accuracy_digits=ad);
    return integrate_2D_normal_main(f,x,q,degx,degy)

def resin_prod(u,k):
    if (u[0] < 0): return 0
    if (u[1] < 0): return 0
    return np.sin(u[0])*np.sin(u[1])*u[0]**(2*k[0]+k[1])*u[1]**(k[1]+2*k[2])*2**k[1]

## return the value of (1,dx11,dx12,dx22,dx11^2,dx12^2,dx22^2,dx11*dx12*dx22)resin
def resin_grad(x,ad=3):
    val=[0]*8
    f=lambda u : resin_prod(u,[0,0,0])
    val[0]=integrate_2D_normal(f,x,ad)
    f=lambda u : resin_prod(u,[1,0,0])  #dx11
    val[1]=integrate_2D_normal(f,x,ad)
    f=lambda u : resin_prod(u,[0,1,0])  #dx12
    val[2]=integrate_2D_normal(f,x,ad)
    f=lambda u : resin_prod(u,[0,0,1])  #dx22
    val[3]=integrate_2D_normal(f,x,ad)
    f=lambda u : resin_prod(u,[2,0,0])  #dx11^2
    val[4]=integrate_2D_normal(f,x,ad)
    f=lambda u : resin_prod(u,[0,2,0])  #dx12^2
    val[5]=integrate_2D_normal(f,x,ad)
    f=lambda u : resin_prod(u,[0,0,2])  #dx22^2
    val[6]=integrate_2D_normal(f,x,ad)
    f=lambda u : resin_prod(u,[1,1,1])  #dx11 dx12 dx22
    val[7]=integrate_2D_normal(f,x,ad)
    return val

def resin_kernel(u):
    if (u[0] < 0): return 0
    if (u[1] < 0): return 0
    return np.sin(u[0])*np.sin(u[1])


"""
##Test inputs.
def f0(u):
    if (u[0] < 0): return 0
    if (u[1] < 0): return 0
    return np.sin(u[0])*np.sin(u[1])

x=np.array([[-18,23],[23,-30]])
print(np.linalg.eigh(x))
print('True val=0.1977 (by MC)?. By integrate_2D_normal=',integrate_2D_normal(f0,x,ad=5,degx=100,degy=10000))
# uee1[-18,23,-30] --> 0.0965756 may be wrong.
# uueMC[-18,23,-30] --> 0.197726 (adaptive quasi montecarlo)
# tt=Integrate[Sin[u]*Sin[v]*Exp[-18*u^2+2*23*u*v-30*v^2],{u,0,Infinity},{v,0,Infinity}]; N[tt]  ---> it does not finish

x=np.array([[-1,0.2],[0.2,-1]])
print(np.linalg.eigh(x))
print('True val=0.240364. By integrate_2D_normal=',integrate_2D_normal(f0,x,degx=100,degy=100))

x=np.array([[-1,0],[0,-1]])
print('True val=0.180146. By integrate_2D_normal=',integrate_2D_normal(f0,x,ad=3,degx=10,degy=10))
print('True val=0.180146. By integrate_2D_normal non-adaptive=',integrate_2D_normal(f0,x,ad=0,degx=100,degy=100))

def f1(u):
    if (u[0] < 0): return 0
    if (u[1] < 0): return 0
    return u[0]**2*u[1]

x=np.array([[-18,23],[23,-30]])
print('True val=0.922698? . By integrate_2D_normal=',integrate_2D_normal(f1,x,degx=100,degy=1000))
# uee1[x11_,x12_,x22_]:=NIntegrate[ u^2*v*Exp[x11*u^2+2*x12*u*v+x22*v^2],{u,0,Infinity},{v,0,Infinity}]
# uee1[-18,23,-30] --> 0.8984  wrong.
# N[Integrate[u^2*v*Exp[-18*u^2+2*23*u*v-30*v^2],{u,0,Infinity},{v,0,Infinity}]  ] --> 0.922698
# NIntegrate[ u^2*v*Exp[-18*u^2+2*23*u*v-30*v^2],{u,0,Infinity},{v,0,Infinity},Method->"AdaptiveQuasiMonteCarlo"]  --> 0.912641

def f2(u):
    if (u[0] < 0): return 0
    if (u[1] < 0): return 0
    return u[0]**2*u[1]**2

x=np.array([[-200,0],[0,-0.002]])
print('True val=0.77614. By integrate_2D_normal=',integrate_2D_normal(f2,x,degx=10,degy=1000))
# uee2[x11_,x12_,x22_]:=NIntegrate[ u^2*v^2*Exp[x11*u^2+2*x12*u*v+x22*v^2],{u,0,Infinity},{v,0,Infinity}]; uee2[-20,0,-0.002] --> 24.5423 Wrong
# NIntegrate[u^2*Exp[-200*u^2],{u,0,Infinity}]*NIntegrate[v^2*Exp[-0.002*v^2],{v,0,Infinity}] --> 0.77614  agree with below
"""


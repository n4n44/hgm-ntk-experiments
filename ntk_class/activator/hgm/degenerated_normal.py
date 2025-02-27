import numpy as np
import scipy

#Ref: T.W.Anderson, An introduction to multivariate statistical analysis, p.30.
#covariance cov が逆行列を持たない時に c1, c2 を求める. 2x2 専用
def get_c1_c2(cov,debug=False):
    sigma=cov
    if np.linalg.det(sigma)!=0.0:
        print('Warning: get_c1_c2 is called non-singular matrix:', sigma)
    s=np.linalg.svd(sigma)
    b=np.linalg.inv(s[0])
    b=b/np.sqrt(s[1][0])
    if debug:
        print('diagonal=[*,0]? ',s[1])
        bp=np.linalg.inv(s[2])
        bp=bp/np.sqrt(s[1][0])
        print('b=',b,'\nb.T-bp==0 matrix?',b.T-bp,'\nb@cov@b.T==[[1,0],[0,0]]? ',b@sigma@bp,end='\n\n')
    binv=np.linalg.inv(b)
    return [binv[0,0],binv[1,0]]

# print(get_c1_c2(np.array([[1,1],[1,1]]),debug=True))
# print(get_c1_c2(np.array([[1,-1],[-1,1]]),debug=True))
# print(get_c1_c2(np.array([[1,2],[2,4]]),debug=True))

# https://docs.scipy.org/doc/scipy/tutorial/integrate.html
# cov が退化した場合の x*(1+erf(x)) での期待値.
# 最初の要素が期待値. 2番目は誤差.
def e_relu_degenerated(cov):
    [c1,c2]=get_c1_c2(cov)
    e0=scipy.integrate.quad(lambda z: (c1*z)*(c2*z)*np.exp(-z**2/2),0,np.inf)
    e=e0/np.sqrt(2*np.pi)
    return e
def e_relu_diff_degenerated(cov):
    [c1,c2]=get_c1_c2(cov)
    e0=scipy.integrate.quad(lambda z: np.exp(-z**2/2),0,np.inf)
    e=e0/np.sqrt(2*np.pi)
    return e

def e_erf_degenerated(cov):
    [c1,c2]=get_c1_c2(cov)
    e0=scipy.integrate.quad(lambda z: ((c1*z)*(1+scipy.special.erf(c1*z)))*((c2*z)*(1+scipy.special.erf(c2*z)))*np.exp(-z**2/2),-np.inf,np.inf)
    e=e0/np.sqrt(2*np.pi)
    return e

#e = e_erf_degenerated(np.array([[1,1],[1,1]]))

# print(e_erf_degenerated(np.array([[1,1],[1,1]])))
# print(2*e)
# print(e_erf_degenerated(np.array([[1,2],[2,4]])))

# cov が退化した場合の x*(1+erf(x)) の微分での期待値.
# 最初の要素が期待値. 2番目は誤差.
# 微分は 1+erf(x)+x*erf'(x)=1+erf(x)+x*exp(-x^2)*2/np.sqrt(np.pi) 
def e_erf_diff_degenerated(cov):
    [c1,c2]=get_c1_c2(cov)
    e0=scipy.integrate.quad(lambda z: (1+scipy.special.erf(c1*z)+c1*z*np.exp(-(c1*z)**2)*2/np.sqrt(np.pi))*(1+scipy.special.erf(c2*z)+c2*z*np.exp(-(c2*z)**2)*2/np.sqrt(np.pi))*np.exp(-z**2/2),-np.inf,np.inf)
    e=e0/np.sqrt(2*np.pi)
    return e

def e_erf_degenerated(cov):
    [c1,c2]=get_c1_c2(cov)
    e0=scipy.integrate.quad(lambda z: np.sin(c1*z)*np.sin(c2*z)*np.exp(-z**2/2),0,np.inf)
    e=e0/np.sqrt(2*np.pi)
    return e

def e_resin_diff_degenerated(cov):
    [c1,c2]=get_c1_c2(cov)
    e0=scipy.integrate.quad(lambda z: np.cos(c1*z)*np.cos(c2*z)*np.exp(-z**2/2),0,np.inf)
    e=e0/np.sqrt(2*np.pi)
    return e

# print(e_erf_diff_degenerated(np.array([[1,1],[1,1]])))
# print(e_erf_diff_degenerated(np.array([[1,2],[2,4]])))



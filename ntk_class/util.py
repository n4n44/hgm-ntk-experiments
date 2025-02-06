import numpy as np
import scipy
import time

def gauss_herm(f,covar,q = 50):
    a = np.sqrt(covar[0])
    b = np.sqrt(covar[2])
    c = np.min([covar[1]/(a*b),1])

    k_sigma = 0
    x , w = scipy.special.roots_hermite(q)
    for i in range(q):
        for j in range(q):
            x_i = x[i]
            w_i = w[i]
            x_j = x[j]
            w_j = w[j]

            k_sigma += w_i*w_j*(f(np.sqrt(2)*a*x_i)*f(np.sqrt(2)*b*c*x_i + np.sqrt(2)*b*np.sqrt(1-c**2)*x_j))

    return k_sigma/np.pi


def make_covar(pts):
    
    return covars

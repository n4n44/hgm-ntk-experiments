import numpy as np
import scipy
from .hgm import tmp_resin
from .hgm import hgm_batch_resin
from .ntk_activator import Activator
from .hgm import degenerated_normal as degene_norm

class Resin(Activator):
    def __init__(self):
        super().__init__()
        self.hgm_large_covar = 1
        self.batch_init = np.array([[-1.0,1/100,-1.0, -1]])
    def activator(self,x):
        if x < 0:
            return 0
        else:
            return np.sin(x)

    def activator_dot(self,x):
        if x < 0:
            return 0
        else:
            return np.cos(x)
        

    def _closed_exp(self,covar):
        c1,c2,r = self._get_c1_c2_r(covar)
        return 0

    
    def _closed_exp_dot(self,covar):
        c1,c2,r = self._get_c1_c2_r(covar)
        return 0
    
    def _hgm_exp(self,covar):
        if np.linalg.det(covar) >= 0.000001:
            input_mat = -np.linalg.inv(covar)/2
            sigma =e_relu.e_relu([input_mat[0][0],input_mat[0][1],input_mat[1][1]])
        return sigma
            
    def _hgm_exp_dot(self,covar):
        return 0
    def _mc_exp(self,covar):
        return 0
    def _mc_exp_dot(self,covar):
        return 0
    def _closed_sigmas(self,covar):
        print('closed form of Resin is unknown')
        return 0

    def _hgm_sigmas(self,covar,rtol = 1e-10,atol = 1e-10):
        c1,c2,r = self._get_c1_c2_r(covar)
        if np.linalg.det(covar) >= 1e-3:
            input_mat = -np.linalg.inv(covar)/2
            sigma =tmp_resin.e_resin([input_mat[0][0],input_mat[0][1],input_mat[1][1]],rtol = rtol,atol= atol)
            sigma_dot = tmp_resin.e_resin_diff([input_mat[0][0],input_mat[0][1],input_mat[1][1]],rtol = rtol,atol= atol)

            var1,var2,_,_ =  self._gauss_herm_sigmas(covar)

        else:
            var1,var2,sigma,sigma_dot = self._gauss_herm_sigmas(covar)

        return var1, var2, sigma, sigma_dot

        
    def batch_hgm(self,dat,qsize=-1,rtol = 1e-10,atol = 1e-10):
        sol = hgm_batch_resin.batch_hgm(dat,qsize,rtol,atol)
        return sol.y

    def batch_hgm_dot(self,dat,qsize=-1,rtol = 1e-10,atol = 1e-10):
        sol = hgm_batch_resin.batch_hgm_diff(dat,qsize,rtol,atol)
        return sol.y

    def _hgm_sigma(self,dat,rtol = 1e-10,atol = 1e-10):
        tmp_sol = tmp_resin.e_resin(dat[0:3],rtol,atol)
        return tmp_sol

    def _hgm_sigma_dot(self,dat,rtol = 1e-10,atol = 1e-10):
        tmp_diff_sol = tmp_resin.e_resin_diff(dat[0:3],rtol,atol)
        return tmp_diff_sol

    def _degene_sigmas(self,covar):
        tmp = degene_norm.e_resin_degenerated(covar)[0]
        tmp_diff = degene_norm.e_resin_diff_degenerated(covar)[0]

        return tmp,tmp_diff
    def _small_sigmas(self,covar):
        tmp, tmp_diff = self.gauss_herm_sigmas(covar, 50)
        return tmp,tmp_diff
    # def _gauss_herm_sigmas(self,covar):
    #     def resin_kernel
        
    #     x = np.linalg.inv(-covar/2)
    #     return integrate_

import numpy as np
import scipy
#from .hgm import e_relu
from .hgm import tmp_relu
from .hgm import tmp_ReLU_diff
from .ntk_activator import Activator
#from .hgm import hgm_batch_relu
from .hgm import tmp_batch_relu
from .hgm import hgm_batch_relu_diff
from .hgm import degenerated_normal as degene_norm
class ReLU(Activator):
    def __init__(self):
        super().__init__()
        self.hgm_small_covar = 1e-3
        
    def activator(self,x):
        if x > 0:
            return x
        else:
            return 0

    def activator_dot(self,x):
        if x > 0:
            return 1
        else:
            return 0

    def _get_c_sigma(self):
        return 2
        

    def _closed_exp(self,covar):
        c1,c2,r = self._get_c1_c2_r(covar)
        return ((r*(np.pi-np.arccos(np.max([np.min([r,1]),-1])))+np.sqrt(1-np.min([r*r,1])))*(c1*c2))/(2*np.pi)

    
    def _closed_exp_dot(self,covar):
        c1,c2,r = self._get_c1_c2_r(covar)
        return (np.pi-np.arccos(np.max([np.min([r,1]),-1])))/(2*np.pi)
    
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
        c1,c2,r = self._get_c1_c2_r(covar)
        sigma = ((r*(np.pi-np.arccos(np.max([np.min([r,1]),-1])))+np.sqrt(1-np.min([r*r,1])))*(c1*c2))/(2*np.pi)
        
        sigma_dot = (np.pi-np.arccos(np.max([np.min([r,1]),-1])))/(2*np.pi)

        return sigma,sigma_dot

    def _hgm_sigmas(self,covar,rtol = 1e-10,atol = 1e-10):
        c1,c2,r = self._get_c1_c2_r(covar)
        if np.linalg.det(covar) >= 0.000001:
            input_mat = -np.linalg.inv(covar)/2
            sigma =tmp_relu.e_relu([input_mat[0][0],input_mat[0][1],input_mat[1][1]],rtol,atol)
            sigma_dot = tmp_ReLU_diff.e_ReLU_diff([input_mat[0][0],input_mat[0][1],input_mat[1][1]],rtol,atol)

        else:
            sigma = self._closed_exp(covar)
            sigma_dot = self._closed_exp_dot(covar)

        var1 = (c1**2)/2
        var2 = (c2**2)/2

        return var1, var2, sigma, sigma_dot

    # def _mc_sigmas(self,covar):
    #     c1,c2,_ = self._get_c1_c2_r(covar)
    #     sigma = 0
    #     sigma_dot = 0
    #     for j in range(self.mc_size):
    #         vec = np.random.multivariate_normal([0,0],covar)
    #         u = np.random.normal(0.,np.sqrt(covar[0][0]))
    #         v = np.random.normal(0.,np.sqrt(covar[1][1]))
    #         sigma += self.activator(vec[0])*self.activator(vec[1])
    #         sigma_dot += self.activator_dot(vec[0])*self.activator_dot(vec[1])
    #     sigma = sigma/self.mc_size
    #     sigma_dot = sigma_dot/self.mc_size
    #     var1 = (c1**2)/2
    #     var2 = (c2**2)/2
    #     return var1,var2,sigma,sigma_dot
        
    def batch_hgm(self,dat,qsize=-1,rtol = 1e-10,atol = 1e-10):
        initial_value =  [1/4,np.pi/8]
        sol = tmp_batch_relu.batch_hgm(dat,initial_value,qsize=qsize,rtol = rtol,atol = atol)
        return sol.y

    def batch_hgm_dot(self,dat,qsize=-1,rtol = 1e-10,atol = 1e-10):

        sol = hgm_batch_relu_diff.batch_hgm(dat,qsize=qsize,rtol=rtol,atol=atol)
        return sol.y

    def _hgm_sigma(self,dat,rtol = 1e-10,atol = 1e-10):
        iv = [1/4,np.pi/8]
        #最後の点の解を求める
        if atol == 0:
            tmp_sol = scipy.integrate.solve_ivp(tmp_relu.f_ReLU,[0.0,1.0], iv,method='RK45',args=(dat,),rtol=rtol)
        else :
            tmp_sol = scipy.integrate.solve_ivp(tmp_relu.f_ReLU,[0.0,1.0], iv,method='RK45',args=(dat,),rtol=rtol,atol=atol)
        return tmp_sol.y

    def _hgm_sigma_dot(self,dat,rtol = 1e-10,atol = 1e-10):
        iv=[np.pi/4,1/2]
        if atol == 0:
            tmp_diff_sol = scipy.integrate.solve_ivp(tmp_ReLU_diff.f_ReLU_diff,[0.0,1.0], iv,method='RK45',args=(dat,),rtol=rtol)
        else :
            tmp_diff_sol = scipy.integrate.solve_ivp(tmp_ReLU_diff.f_ReLU_diff,[0.0,1.0], iv,method='RK45',args=(dat,),rtol=rtol,atol=atol)
        return tmp_diff_sol.y

    # def _ada_gauss_herm(self,x,qq = 50,degx,accuracy_digits = )

    def _degene_sigmas(self,covar):
        tmp = degene_norm.e_relu_degenerated(covar)[0]
        tmp_diff = degene_norm.e_relu_diff_degenerated(covar)[0]

        return tmp,tmp_diff

    def _small_sigmas(self, covar):
        det = covar[0]*covar[2] - covar[1]**2
        if det < 1e-7:
            covar_mat = np.array([[covar[0], covar[1]],[covar[1],covar[2]]])
            tmp, tmp_diff = self._degene_sigmas(covar_mat)
        else:
        
            tmp, tmp_diff = self.gauss_herm_sigmas(covar,50)
        return tmp,tmp_diff

import numpy as np
import scipy.special
from .hgm import tmp_erf
from .hgm import e_erf_diff_by_sum as e_erf_diff
from .hgm import degenerated_normal as degene_norm
from .ntk_activator import Activator

class Airy(Activator):
    # 数値積分で推定
    def ev_sigmas(self,covar,mode):
        if mode == 'closed':
            var1, var2, sigma, sigma_dot = self._closed_sigmas(covar)
        elif mode == 'hgm':
            var1, var2, sigma, sigma_dot = self._hgm_sigmas(covar)
        elif mode == 'monte_carlo':
            var1, var2, sigma, sigma_dot = self._mc_sigmas(covar)
        elif mode == 'gauss_herm':
            var1, var2, sigma, sigma_dot = self._gauss_herm_sigmas(covar)
        else :
            print(f'invalid mode: mode=\'{mode}\'')
            return 0

        var1 *= self.c_sigma
        var2 *= self.c_sigma
        sigma *= self.c_sigma
        sigma_dot *= self.c_sigma
        return var1,var2,sigma,sigma_dot

    def activator(self,x):
        return scipy.special.airy(-x)[0]

    def activator_dot(self,x):
        return scipy.special.airy(-x)[1]

    def _closed_exp(self,covar):
        c1,c2,r = self._get_c1_c2_r(covar)
        sigma = ((c1*c2*r)/4)+(((c1**2)*(c2**2))/(2*np.pi))*(((r**2 + 1 + c1**2 + c2**2 + ((c1*c2)**2)*(1-r**2))/((1+c1**2)*(1+c2**2)*np.sqrt(1+c1**2+c2**2 + ((c1*c2)**2)*(1-r**2))))+(r/(c1*c2))*np.arctan((c1*c2*r)/np.sqrt(1+c1**2+c2**2 + ((c1*c2)**2)*(1-r**2))))
        return sigma

    
    def _closed_exp_dot(self,covar):
        c1,c2,r = self._get_c1_c2_r(covar)
        sigma_dot = (1/4) + ((((2-(c1*c2)**2)*c1*c2*r*(1+c1**2)*(1+c2**2))+(((c1*c2)**2) - 1)*((c1*c2*r)**3))/((2*np.pi)*(1+c1**2)*(1+c2**2)*((np.sqrt(1+c1**2+c2**2 + ((c1*c2)**2)*(1-r**2)))**3))) + (1/(2*np.pi))*np.arctan((c1*c2*r)/np.sqrt(1+c1**2+c2**2 + ((c1*c2)**2)*(1-r**2))) + (c1*c2*r/(2*np.pi))*(1/np.sqrt(1+c1**2+c2**2 + ((c1*c2)**2)*(1-r**2)))
        return sigma_dot
    
    def _hgm_exp(self,covar):
        if np.linalg.det(covar) >= 0.0000001:
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
        sigma =  ((c1*c2*r)/4)+(((c1**2)*(c2**2))/(2*np.pi))*(((r**2 + 1 + c1**2 + c2**2 + ((c1*c2)**2)*(1-r**2))/((1+c1**2)*(1+c2**2)*np.sqrt(1+c1**2+c2**2 + ((c1*c2)**2)*(1-r**2))))+(r/(c1*c2))*np.arctan((c1*c2*r)/np.sqrt(1+c1**2+c2**2 + ((c1*c2)**2)*(1-r**2))))
        
        sigma_dot = (1/4) + ((((2-(c1*c2)**2)*c1*c2*r*(1+c1**2)*(1+c2**2))+(((c1*c2)**2) - 1)*((c1*c2*r)**3))/((2*np.pi)*(1+c1**2)*(1+c2**2)*((np.sqrt(1+c1**2+c2**2 + ((c1*c2)**2)*(1-r**2)))**3))) + (1/(2*np.pi))*np.arctan((c1*c2*r)/np.sqrt(1+c1**2+c2**2 + ((c1*c2)**2)*(1-r**2))) + (c1*c2*r/(2*np.pi))*(1/np.sqrt(1+c1**2+c2**2 + ((c1*c2)**2)*(1-r**2)))
        c1 = np.sqrt(covar[0][0])
        c2 = np.sqrt(covar[0][0])
        r = 1
        var1 =((c1*c2*r)/4)+(((c1**2)*(c2**2))/(2*np.pi))*(((r**2 + 1 + c1**2 + c2**2 + ((c1*c2)**2)*(1-r**2))/((1+c1**2)*(1+c2**2)*np.sqrt(1+c1**2+c2**2 + ((c1*c2)**2)*(1-r**2))))+(r/(c1*c2))*np.arctan((c1*c2*r)/np.sqrt(1+c1**2+c2**2 + ((c1*c2)**2)*(1-r**2))))
        
        c1 = np.sqrt(covar[1][1])
        c2 = np.sqrt(covar[1][1])
        var2 =((c1*c2*r)/4)+(((c1**2)*(c2**2))/(2*np.pi))*(((r**2 + 1 + c1**2 + c2**2 + ((c1*c2)**2)*(1-r**2))/((1+c1**2)*(1+c2**2)*np.sqrt(1+c1**2+c2**2 + ((c1*c2)**2)*(1-r**2))))+(r/(c1*c2))*np.arctan((c1*c2*r)/np.sqrt(1+c1**2+c2**2 + ((c1*c2)**2)*(1-r**2))))
        return var1,var2,sigma,sigma_dot

    def _gauss_herm_sigmas(self,covar):
        a,b,c = self._get_c1_c2_r(covar)
        covar_det = np.linalg.det(covar)
        if covar_det >= 0.00000001:
            k_sigma = 0
            k_sigma_dot = 0
            var1 = 0
            var2 = 0
            q = 150
            x , w = scipy.special.roots_hermite(q)
            for i in range(q):
                for j in range(q):
                    x_i = x[i]
                    w_i = w[i]
                    x_j = x[j]
                    w_j = w[j]
                    k_sigma += w_i*w_j*(self.activator(np.sqrt(2)*a*x_i)*self.activator(np.sqrt(2)*b*c*x_i + np.sqrt(2)*b*np.sqrt(1-c**2)*x_j))

                    k_sigma_dot +=w_i*w_j*(self.activator_dot(np.sqrt(2)*a*x_i)*self.activator_dot(np.sqrt(2)*b*c*x_i + np.sqrt(2)*b*np.sqrt(1-c**2)*x_j))
                    var1 += w_i*w_j*(self.activator(np.sqrt(2)*a*x_i)*self.activator(np.sqrt(2)*a*1*x_i))
                    var2 += w_i*w_j*(self.activator(np.sqrt(2)*b*x_i)*self.activator(np.sqrt(2)*b*1*x_i))
            sigma = k_sigma/np.pi
            sigma_dot = k_sigma_dot/np.pi
            var1 /= np.pi
            var2 /=np.pi
        else:
            k_sigma = 0
            k_sigma_dot = 0
            var1 = 0
            var2 = 0
            q = 150
            x , w = scipy.special.roots_hermite(q)
            for i in range(q):
                for j in range(q):
                    x_i = x[i]
                    w_i = w[i]
                    x_j = x[j]
                    w_j = w[j]
                    k_sigma += w_i*w_j*(self.activator(np.sqrt(2)*a*x_i)*self.activator(np.sqrt(2)*b*c*x_i))

                    k_sigma_dot +=w_i*w_j*(self.activator_dot(np.sqrt(2)*a*x_i)*self.activator_dot(np.sqrt(2)*b*c*x_i))
                    var1 += w_i*w_j*(self.activator(np.sqrt(2)*a*x_i)*self.activator(np.sqrt(2)*a*1*x_i))
                    var2 += w_i*w_j*(self.activator(np.sqrt(2)*b*x_i)*self.activator(np.sqrt(2)*b*1*x_i))
            sigma = k_sigma/np.pi
            sigma_dot = k_sigma_dot/np.pi
            var1 /= np.pi
            var2 /=np.pi
        return var1, var2, sigma, sigma_dot
    
    def _hgm_sigmas(self,covar):
        c1,c2,r = self._get_c1_c2_r(covar)
        covar_det = np.linalg.det(covar)
        if covar_det >= 0.00000001:
            input_mat = -np.linalg.inv(covar)/2
            if covar_det >= 1e-3:
                sigma =tmp_erf.e_gelu([input_mat[0][0],input_mat[0][1],input_mat[1][1]])
                sigma_dot = e_erf_diff.e_erf_diff_value_by_sum([input_mat[0][0],input_mat[0][1],input_mat[1][1]])
            else:
                # print('use monte carlo approximation')
                # sigma,sigma_dot = self._mc_sigma_sigma_dot(covar)
                print('use gauss hermite quadrature')
                k_sigma = 0
                k_sigma_dot = 0
                q = 150
                a = np.sqrt(covar[0][0])
                b = np.sqrt(covar[1][1])
                c = covar[0][1]/(a*b)
                x , w = scipy.special.roots_hermite(q)
                
                for i in range(q):
                    for j in range(q):
                        x_i = x[i]
                        w_i = w[i]
                        x_j = x[j]
                        w_j = w[j]
            
                        k_sigma += w_i*w_j*(self.activator(np.sqrt(2)*a*x_i)*self.activator(np.sqrt(2)*b*c*x_i + np.sqrt(2)*b*np.sqrt(1-c**2)*x_j))
                        k_sigma_dot +=w_i*w_j*(self.activator_dot(np.sqrt(2)*a*x_i)*self.activator_dot(np.sqrt(2)*b*c*x_i + np.sqrt(2)*b*np.sqrt(1-c**2)*x_j))
                sigma = k_sigma/np.pi
                sigma_dot = k_sigma_dot/np.pi


        else:
            sigma = degene_norm.e_erf_degenerated(covar)[0]
            sigma_dot = degene_norm.e_erf_diff_degenerated(covar)[0]
            
        
        var1 = degene_norm.e_erf_degenerated(np.array(([c1**2,c1**2],[c1**2,c1**2])))[0]
        var2 = degene_norm.e_erf_degenerated(np.array(([c2**2,c2**2],[c2**2,c2**2])))[0]

        return var1, var2, sigma, sigma_dot

    def _mc_sigma_sigma_dot(self,covar):
        sigma = 0
        sigma_dot = 0
        for j in range(self.mc_size):
            vec = np.random.multivariate_normal([0,0],covar)
            u = np.random.normal(0.,np.sqrt(covar[0][0]))
            v = np.random.normal(0.,np.sqrt(covar[1][1]))
            sigma += self.activator(vec[0])*self.activator(vec[1])
            sigma_dot += self.activator_dot(vec[0])*self.activator_dot(vec[1])
        sigma = sigma/self.mc_size
        sigma_dot = sigma_dot/self.mc_size

        return sigma,sigma_dot

    def _mc_sigmas(self,covar):
        c1,c2,_ = self._get_c1_c2_r(covar)
        sigma = 0
        sigma_dot = 0
        for j in range(self.mc_size):
            vec = np.random.multivariate_normal([0,0],covar)
            u = np.random.normal(0.,np.sqrt(covar[0][0]))
            v = np.random.normal(0.,np.sqrt(covar[1][1]))
            sigma += self.activator(vec[0])*self.activator(vec[1])
            sigma_dot += self.activator_dot(vec[0])*self.activator_dot(vec[1])
        sigma = sigma/self.mc_size
        sigma_dot = sigma_dot/self.mc_size
        var1 = (c1**2)/2
        var2 = (c2**2)/2
        return var1,var2,sigma,sigma_dot


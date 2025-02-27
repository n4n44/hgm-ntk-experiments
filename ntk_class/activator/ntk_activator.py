from abc import ABC,abstractmethod
import numpy as np
import scipy.special

class Activator(ABC):
    def __init__(self,mc_size=5000):
        c_sigma = self._get_c_sigma()
        self.mc_size = mc_size
        self.c_sigma = c_sigma
        self.hgm_small_covar = 1e-3
        self.hgm_large_covar = np.inf
        self.batch_init = np.array([[-1.0, 0.0, -1.0, -1]])
    def __call__(self,x):
        
        return self.activator(x)
    
    @abstractmethod
    def activator(self,x):
        pass
    
    @abstractmethod
    def activator_dot(self,x):
        pass
    
    def nc2(self,x):
        return (np.pi/np.sqrt(x[0]*x[2]-x[1]**2))

    def _get_c_sigma(self):
        # gauss-hermite estimation of c_sigma 
        k_sigma = 0
        k_sigma_dot = 0
        var1 = 0
        var2 = 0
        q = 150
        x , w = scipy.special.roots_hermite(q)
        a = 1
        b = 1
        c = 1
        for i in range(q):
            for j in range(q):
                x_i = x[i]
                w_i = w[i]
                x_j = x[j]
                w_j = w[j]
                k_sigma += w_i*w_j*(self.activator(np.sqrt(2)*a*x_i)*self.activator(np.sqrt(2)*b*c*x_i))

        exp = k_sigma/np.pi

        return 1/exp 

    def _get_c1_c2_r(self,covar):
        c1 = np.sqrt(covar[0])
        c2 = np.sqrt(covar[2])
        if c1 != 0 and c2!= 0:
            r = covar[1]/(c1*c2)
        else:
            r = covar[1]

        return c1,c2,r
    
    def ev_sigmas(self,covar,mode,rtol = 1e-7,atol = 0):
        if mode == 'closed':
            var1, var2, sigma, sigma_dot = self._closed_sigmas(covar)
        elif mode == 'hgm':
            var1, var2, sigma, sigma_dot = self._hgm_sigmas(covar,rtol = rtol,atol=atol)
        elif mode == 'monte_carlo':
            var1, var2, sigma, sigma_dot = self._mc_sigmas(covar)
        elif mode == 'gauss_herm':
            var1, var2, sigma, sigma_dot = self._gauss_herm_sigmas(covar,accuracy_dig = rtol)
        else :
            print(f'invalid mode: mode=\'{mode}\'')
            return 0

        var1 *= self.c_sigma
        var2 *= self.c_sigma
        sigma *= self.c_sigma
        sigma_dot *= self.c_sigma
        return var1,var2,sigma,sigma_dot
    
    def dual_expectation(self,covar,mode):
        if mode == 'closed':
            return self._closed_exp(covar)
        elif mode == 'hgm':
            return self._hgm_exp(covar)
        elif mode == 'monte_carlo':
            return self._mc_exp(covar)
        print(f'mode = {mode} is invalid mode')
        
    def dual_expectation_dot(self,covar,mode):
        if mode == 'closed':
            return self._closed_exp(covar)
        elif mode == 'hgm':
            return self._hgm_exp(covar)
        elif mode == 'monte_carlo':
            return self._mc_exp(covar)
        print(f'mode = {mode} is invalid mode')


    @abstractmethod
    def _closed_exp(self,covar):
        pass

    @abstractmethod
    def _closed_exp_dot(self,covar):
        pass

    @abstractmethod
    def _hgm_exp(self,covar,rtol=1e-10,atol = 1e-10):
        pass
    @abstractmethod
    def _hgm_exp_dot(self,covar,rtol=1e-10,atol = 1e-10):
        pass
    @abstractmethod
    def _mc_exp(self,covar):
        pass
    @abstractmethod
    def _mc_exp_dot(self,covar):
        pass
    @abstractmethod
    def _closed_sigmas(self,covar):
        pass
    @abstractmethod
    def batch_hgm(dat,qsize=-1,rtol = 1e-10, atol = 1e-10):
        pass
    @abstractmethod    
    def batch_hgm_dot(dat,qsize=-1,rtol = 1e-10, atol = 1e-10):
        pass
    def _gauss_herm_sigmas(self,covar,accuracy_dig = 1e-10):
        q1 = 10
        q2 = q1
        if accuracy_dig == -1:
            sigma1 = self.gauss_herm_sigmas(covar,q1)
            return sigma1
        else:
            for i in range(5):
                sigma1 = self.gauss_herm_sigmas(covar,q1)
                q2 = 2*q1
                sigma2 = self.gauss_herm_sigmas(covar,q2)
                sig_arr1 = np.array(sigma1)
                sig_arr2 = np.array(sigma2)
                diff = sig_arr1-sig_arr2
                for j in range(2):
                    rer = 0
                    tmp = np.abs(diff[j]/sig_arr1[j])
                    if tmp > rer:
                        rer = tmp
                    if  rer < accuracy_dig:
                        return sigma2
                        # return sigma2[0],sigma2[1],sigma2[2],sigma2[3]
                q1 = q2
        return sigma2
            # return sigma2[0],sigma2[1],sigma2[2],sigma2[3]
    def gauss_herm_sigmas(self,covar,q1):
        a,b,c = self._get_c1_c2_r(covar)
        covar_det = covar[0]*covar[2] - covar[1]**2
        if covar_det >= 0.00000001:
            k_sigma = 0
            k_sigma_dot = 0
            q = q1
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
            k_sigma = 0
            k_sigma_dot = 0
            q = q1
            x , w = scipy.special.roots_hermite(q)
            for i in range(q):
                for j in range(q):
                    x_i = x[i]
                    w_i = w[i]
                    x_j = x[j]
                    w_j = w[j]
                    k_sigma += w_i*w_j*(self.activator(np.sqrt(2)*a*x_i)*self.activator(np.sqrt(2)*b*c*x_i))

                    k_sigma_dot +=w_i*w_j*(self.activator_dot(np.sqrt(2)*a*x_i)*self.activator_dot(np.sqrt(2)*b*c*x_i))

            sigma = k_sigma/np.pi
            sigma_dot = k_sigma_dot/np.pi

        return sigma, sigma_dot

    def _mc_sigmas(self, covar):
        sigma = 0
        sigma_dot = 0
        covar_mat = np.array([[covar[0],covar[1]],[covar[1],covar[2]]])
        for i in range(self.mc_size):
            vec = np.random.multivariate_normal([0,0],covar_mat)
            sigma += self.activator(vec[0])*self.activator(vec[1])
            sigma_dot += self.activator_dot(vec[0])*self.activator_dot(vec[1])
        sigma = sigma/self.mc_size
        sigma_dot = sigma_dot/self.mc_size
        return sigma,sigma_dot
    def _adaptive_gauss_herm(self,f,covar,accuracy_dig = 1e-7):
        q1 = 10
        for i in range(5):
            sigma1 = self.gauss_herm(f,covar,q1)
            q2 = 2*q1
            sigma2 = self.gauss_herm(f,covar,q2)
            diff = sigma1-sigma2
            rer = np.abs(diff/sigma1)
            if rer < accuracy_dig:
                return sigma2
            q1 = q2
        return sigma2
    def gauss_herm(self,f,covar,q = 10):
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

import numpy as np
import time
import scipy.linalg
from .activator.hgm import batch_data_loader as batch_dl
from .util import gauss_herm

class ntk:
    def __init__(self,layers,activator,beta=0,cuda_mode = False,debug_mode = False):
        self.activator = activator
        self.layers = layers
        self.beta = beta
        self.pred_mode = False
        self.gauss_herm_q = 50
        self.threshould = 1e-5
        self.cuda_mode = cuda_mode
        self._debug_mode = debug_mode
        self._hgm_interpolate = 'copy'


    def ev_kernel(self,covars,mode = 'hgm_fast',rtol = 1e-10,atol = 1e-10):
        if mode == 'hgm':
            sigma,sigma_dot = self._hgm_ev_kernel(covars,rtol,atol)
        if mode == 'hgm_fast':
            sigma,sigma_dot = self._fast_hgm_ev_kernel(covars,rtol,atol)
        if mode == 'closed':
            sigma,sigma_dot = self._other_ev_kernel(covars,mode,rtol,atol)
        if mode == 'gauss_herm':
            sigma,sigma_dot = self._other_ev_kernel(covars,mode, rtol,atol)
        if mode == 'monte_carlo':
            sigma,sigma_dot = self._other_ev_kernel(covars,mode, rtol,atol)
        return sigma,sigma_dot
        
    def train(self,sample_x,sample_y,diag_reg=0.001,mode = 'hgm_fast',get_timing = False,rtol = 1e-10, atol = 1e-10):
        # covarsは(3,n)のndarrayにする.
        
        self.sample_x = sample_x
        self.sample_y = sample_y
        self.pred_mode = False
        self.kernel_mat = np.zeros((self.sample_x.shape[0]*self.sample_x.shape[0]))
        if get_timing == True:
            start = time.time()

        covars = self.make_covars(self.sample_x)
        
        sigma_array, sigma_dot_array = self.ev_kernel(covars,rtol = rtol, mode = mode,atol = atol)
        
        if self._debug_mode:
            print('sigma_array:',sigma_array)
            print('sigma_dot_array:',sigma_dot_array)
        for i in range(self.sample_x.shape[0]**2):
            ntk = 0
            for s in range(self.layers):

                sigma_dot = 1
                for t in range(s+1,self.layers):
                    #print(sigma_dot_array[t-1][i])
                    
                    sigma_dot = sigma_dot*sigma_dot_array[t-1][i]
                
                ntk = ntk +(sigma_dot*sigma_array[s][i])
            self.kernel_mat[i] = ntk

        self.kernel_mat = self.kernel_mat.reshape(self.sample_x.shape[0],-1)
        if get_timing == True:
           end = time.time()
           training_time = end - start
           start = time.time()
        self.pred_matrix = scipy.linalg.solve(self.kernel_mat+diag_reg*np.identity(self.sample_x.shape[0]),self.sample_y,assume_a = 'sym')
        if get_timing == True:
            end = time.time()
            solve_time = end-start
        self.pred_mode = True
        print('Training finished.')
        if get_timing == True:
            print(f'Training time:{training_time}, solve_time: {solve_time}')

    def pred(self,test_x,mode = 'hgm_fast',get_timing = True,rtol = 1e-10, atol = 1e-10):
        if self.pred_mode != True:
            print('model has not been trained yet.')
            return 0

        else:
            if get_timing:
                start = time.time()
            covars =self.make_covars_pred(test_x,self.sample_x)
            sigma_array, sigma_dot_array = self.ev_kernel(covars,rtol = rtol, mode = mode,atol = atol)

            kernel_list = np.zeros(test_x.shape[0]*self.sample_x.shape[0])
            
            for i in range(test_x.shape[0]*self.sample_x.shape[0]):
                ntk = 0
           
                for s in range(self.layers):

                    sigma_dot = 1
                    for t in range(s+1,self.layers):
                    #print(sigma_dot_array[t-1][i])
                        sigma_dot = sigma_dot*sigma_dot_array[t-1][i]
                
                    ntk = ntk +(sigma_dot*sigma_array[s][i])
                    kernel_list[i] = ntk
            kernel_list = kernel_list.reshape(-1,self.sample_x.shape[0])
            # print(kernel_list.shape)
            if self._debug_mode == True:
                
                print(self.pred_matrix)
                print(kernel_list)
            output = kernel_list@self.pred_matrix
            output = output.reshape(-1)
            if get_timing:
                end = time.time()
                print('pred time:', end-start)
            return output

            
    def train_pred(self,sample_x,test_x,sample_y,diag_reg=0.001,mode = 'hgm_fast',get_timing = False,rtol = 1e-10, atol = 1e-10):
        self.sample_x = sample_x
        self.sample_y = sample_y
        self.pred_mode = False
        if get_timing == True:
            start = time.time()
        eval_pts = np.vstack((sample_x,test_x))
        covars = self.make_covars(eval_pts)
        
        sigma_array, sigma_dot_array = self.ev_kernel(covars,rtol = rtol, mode = mode,atol = atol)
        kernel_table = np.zeros((eval_pts.shape[0]**2))
        for i in range(sigma_array.shape[1]):
            ntk = 0
            for s in range(self.layers):
                sigma_dot = 1
                for t in range(s+1,self.layers):

                    sigma_dot = sigma_dot*sigma_dot_array[t-1][i]

                ntk = ntk+(sigma_dot*sigma_array[s][i])
            kernel_table[i] = ntk
        train_size = self.sample_x.shape[0]
        kernel_table = kernel_table.reshape(eval_pts.shape[0],-1)
        self.kernel_mat = kernel_table[0:train_size,0:train_size]
        kernel_list = kernel_table[0:train_size,train_size:]
        self.pred_matrix = scipy.linalg.solve(self.kernel_mat+diag_reg*np.identity(train_size),sample_y,assume_a='sym')
        output = kernel_list.T@self.pred_matrix
        if get_timing:
            end = time.time()
            print('pred time:', end-start)

        return output

    def _other_ev_kernel(self, covars, mode,rtol = 1e-10, atol=1e-10):
        sigma_array = np.zeros((self.layers,covars.shape[0]))
        sigma_dot_array = np.zeros((self.layers,covars.shape[0]))
        for l in range(self.layers-1):
            if l == 0:
                dat = covars
                for i in range(covars.shape[0]):
                    sigma_array[0][i] = covars[i][1]

            else:
                dat = self._make_covars_from_sigma(sigma_array[l])

            for i in range(covars.shape[0]):
                covar = covars[i]

                if mode == 'closed':
                    sigma,sigma_dot = self.activator._closed_sigmas(covar)
                if mode == 'gauss_herm':
                    sigma,sigma_dot = self.activator.gauss_herm_sigmas(covar,50)
                if mode == 'monte_carlo':
                    sigma,sigma_dot = self.activator._mc_sigmas(covar)
                                    
                if l < self.layers-1:
                    sigma_array[l+1][i] = self.activator.c_sigma*sigma+self.beta**2
                    sigma_dot_array[l][i] = self.activator.c_sigma*sigma_dot

        return sigma_array, sigma_dot_array
    def _hgm_ev_kernel(self,covars,rtol = 1e-10,atol = 1e-10):
        return 0
    
    def _fast_hgm_ev_kernel(self,covars,rtol, atol):
        if self._debug_mode:
            print(covars)
        sigma_array = np.zeros((self.layers,covars.shape[0]))
        sigma_dot_array = np.zeros((self.layers,covars.shape[0]))
        for l in range(self.layers-1):
            degene_covars = []
            
            if l == 0:
                dat = covars
                for i in range(covars.shape[0]):
                    sigma_array[0][i] = covars[i][1] # first sigmas
            else :
                dat = self._make_covars_from_sigma(sigma_array[l])
                
            [dat2,removed,removed_small_covar,removed_large_covar]=batch_dl.sort_dat(dat,small_covar=self.activator.hgm_small_covar,large_covar = self.activator.hgm_large_covar,threshould=1e-5,initial_point = self.activator.batch_init,even = None)
            if self._debug_mode:
                print('dat2.shape = ',dat2.shape)
                print(removed.shape)
                print(removed_small_covar.shape)
                print(removed_large_covar.shape)

            removed = removed.reshape(-1,5)

            dat2 = np.vstack((dat2,dat2[0])) # Add dummy data to the end to evaluate all points in solve_ivp
            sol = self.activator.batch_hgm(dat2,qsize=-1,rtol = rtol,atol = atol)
            hgm_sol = sol[0].reshape(-1,1)
            diff_sol = self.activator.batch_hgm_dot(dat2,qsize=-1,rtol = rtol,atol = atol)
            hgm_diff_sol = diff_sol[0].reshape(-1,1)

            degene_sol, degene_diff_sol = self._ev_degene_sigmas(removed_small_covar,rtol,atol)
            #removed_sol, removed_diff_sol = self._ev_removed_sigmas(removed)
            large_sol,large_diff_sol = self._ev_large_sigmas(removed_large_covar,rtol,atol)
            
            for i in range(dat2.shape[0]-1):
                idx =int(dat2[i][3])
                x = np.array([[dat2[i][0],dat2[i][1]],[dat2[i][1],dat2[i][2]]])
                covar = np.linalg.inv(-2*x)
                nc2x = self.activator.nc2(dat2[i][0:3])
                
                if idx >= 0:
                    if l < self.layers-1:
                        #print(hgm_sol[i],hgm_diff_sol[i])
                        sigma_array[l+1][idx] = (self.activator.c_sigma*hgm_sol[i])/nc2x+self.beta**2
                        sigma_dot_array[l][idx] = (self.activator.c_sigma*hgm_diff_sol[i])/nc2x

            for i in range(removed_small_covar.shape[0]):
                idx = int(removed_small_covar[i][3])
                if l < self.layers-1:
                    sigma_array[l+1][idx] = self.activator.c_sigma*degene_sol[i]+self.beta**2
                    sigma_dot_array[l][idx] = self.activator.c_sigma*degene_diff_sol[i]

            for i in range(removed_large_covar.shape[0]):
                idx = int(removed_large_covar[i][3])
                if l < self.layers-1:
                    sigma_array[l+1][idx] = self.activator.c_sigma*small_covar_sol[i]+self.beta**2
                    sigma_dot_array[l][idx] = self.activator.c_sigma*small_covar_diff_sol[i]

            for i in range(removed.shape[0]):
                idx1 = int(removed[i][3])
                idx2 = int(removed[i][4])
                if l < self.layers-1:
                    # gauss-hermの値と比較する
                    sigma_array[l+1][idx1] = sigma_array[l+1][idx2]
                    sigma_dot_array[l][idx1] = sigma_dot_array[l][idx2]
        return sigma_array,sigma_dot_array

    def _ev_degene_sigmas(self,degene_covars,rtol,atol):
        degene_sol = []
        degene_diff_sol = []
        for i in range(degene_covars.shape[0]):
            # covar = np.array([[degene_covars[i][0],degene_covars[i][1]],[degene_covars[i][1],degene_covars[i][2]]])
            #tmp_sol,tmp_diff_sol = self.activator._degene_sigmas(covar)
            covar = degene_covars[i]
            tmp_sol,tmp_diff_sol = self.activator._small_sigmas(covar)
            degene_sol = np.append(degene_sol,tmp_sol)
            degene_diff_sol = np.append(degene_diff_sol,tmp_diff_sol)
        return degene_sol,degene_diff_sol

    def _ev_large_sigmas(self,degene_covars,rtol,atol):
        degene_sol = []
        degene_diff_sol = []
        for i in range(degene_covars.shape[0]):
            tmp_sol,tmp_diff_sol = self.activator._gauss_herm_sigmas(degene_covars[i])
            degene_sol = np.append(degene_sol,tmp_sol)
            degene_diff_sol = np.append(degene_diff_sol,tmp_diff_sol)
        return degene_sol,degene_diff_sol

    # def _ev_removed_sigmas(removed,hgm_sol,hgm_diff_sol):
    #     removed_sol =[]
    #     degene_diff_sol = []
    #     for i in range(removed)
    def make_covars(self,pts):
        covars = []
        sigmas= np.dot(pts,pts.T)+self.beta**2
        for i in range(sigmas.shape[0]):
            for j in range(sigmas.shape[0]):
                tmp_covar = np.array([sigmas[i][i],sigmas[i][j],sigmas[j][j]])
                covars = np.append(covars,tmp_covar)
        covars = np.array(covars)
        
        covars = batch_dl.add_index(covars.reshape(-1,3))
        print(covars.shape)
        return covars

    def make_covars_pred(self,test_pts,sample_pts):
        covars = []
        sigmas = np.dot(test_pts,sample_pts.T)+self.beta**2

        for i in range(sigmas.shape[0]):
            for j in range(sigmas.shape[1]):
                c1 = np.linalg.norm(sample_pts[j])**2+self.beta**2
                c2 = np.linalg.norm(test_pts[i])**2+self.beta**2
                
                tmp_covar = np.array([c1,sigmas[i][j],c2])
                covars = np.append(covars,tmp_covar)
        covars = np.array(covars)
        covars = batch_dl.add_index(covars.reshape(-1,3))
        return covars
    
    def make_covars_from_sigma(self,sigma_array):
        covars = []
        size = int(np.sqrt(sigma_array.shape[0]))
        for i in range(sigma_array.shape[0]):
            idx1 = int(i/size)
            idx2 = int(i%size)
            tmp_covar = np.array(sigma_array[idx1*size+idx1],sigma_array[i],sigma_array[idx2*size+idx2])

            covars = np.append(covars,tmp_covar)
        covars = np.array(covars)
        covars = batch_dl.add_index(covars.reshape(-1,3))
        return covars

    
    def set_mc_size(self,size):
        self.activator.mc_size = size

        

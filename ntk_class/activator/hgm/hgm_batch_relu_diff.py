import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

import time
# import hgm_batch_erf
# import batch_data_loader
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# import e_erf_diff_by_sum
# import tmp_erf_diff_fg
# import tmp_erf_diff_gf
# import tmp_erf_diff_gg

def f_relu_diff(t,f):
    global Q
    global N_calls
    N_calls = N_calls+1
    k = int(np.floor(t))
    dx=Q[k+1]-Q[k]
    [x11,x12,x22]=Q[k]+dx*(t-k)

    #------ from tmp_pf_relu_diff.py
    d1=-((x12)**(2))+(x22)*(x11)
    p11=np.array([[(-1/2)/(x11),(-((1/2)*(x12)))/(x11)],[(-((1/2)*(x12)))/((d1)*(x11)),((-((1/2)*((x12)**(2))))-((x22)*(x11)))/((d1)*(x11))]])
    p12=np.array([[0,1],[(1)/(d1),((3)*(x12))/(d1)]])
    p22=np.array([[(-1/2)/(x22),(-((1/2)*(x12)))/(x22)],[(-((1/2)*(x12)))/((d1)*(x22)),((-((1/2)*((x12)**(2))))-((x22)*(x11)))/((d1)*(x22))]])
    #------
    p=p11*dx[0]+p12*dx[1]+p22*dx[2]
    return p@f

#cook_Q_for_hgm(dat,small_covar=1e-3,threshould=1e-5,initial_point=np.array([[-1,0,-1]]),even=True):
#  defined in 2024-09-12-batch-erf.py
# 戻り値 np.array([[x11,x12,x22,idx0],[x11,x12,x22,idx1],...])
#   x が sort されてる. idx は元のデータで何番目か?

# dat は cook_Q_for_hgm の戻り値を使う
#  qsize  何個まとめて計算するか?  -1 は全部.
def batch_hgm(dat,qsize=-1,rtol = 1e-10, atol = 1e-10):
    global Q
    global N_calls 
    global Last_value
    Q=dat
    if (qsize < 0): qsize=Q.shape[0]
    Q=Q[0:qsize]
    if (qsize <= 20):
        print(Q)
    print('Data size = ',Q.shape)
    NN = Q.shape[0]
    QQ=(Q.T)[0:-1] # remove index
    Q=QQ.T
    Q_orig=Q.copy()
    N_calls=0

    t_span=[0.0,NN-2]
    # tmp_erf.py
    iv=[np.pi/4,1/2]
    start=time.time()
    #print(list(range(NN-1)))

    # sol = solve_ivp(f_erf,t_span,iv,method='RK45',t_eval=list(range(NN-1)),rtol=1e-10,atol=1e-10)
    if atol == 0:
        sol = solve_ivp(f_relu_diff,t_span,iv,method='RK45',t_eval=list(range(NN-1)),rtol=rtol)
    else:
        sol = solve_ivp(f_relu_diff,t_span,iv,method='RK45',t_eval=list(range(NN-1)),rtol=rtol,atol=atol)
    end=time.time(); print('Time=',end-start,'s')
    print(sol)
    print('sol.y[0,-1]=',sol.y[0,-1])
    Last_value=sol.y[0,-1]
    print('N_calls=',N_calls)
    print('Q[NN-2]=',Q[NN-2])
    return sol


# num_pts = 15
# sample_x = np.linspace(-1,1,num_pts).reshape(num_pts, -1)
# beta = 0.5
# dat = []
# for i in range(num_pts):
#     for j in range(num_pts):
        
#         c1 = sample_x[i].item()**2 + beta**2
#         c2 = sample_x[j].item()**2 + beta**2
#         sigma = sample_x[i].item()*sample_x[j].item() + beta**2
#         covar = np.array([[c1,sigma],[sigma,c2]])
#         try:
#             mat_inv =(-1/2)* np.linalg.inv(covar)
#             print(mat_inv)

#         except np.linalg.LinAlgError as e:
#             print(e)
#             print('did not append inverse matrix')

#         else:
#             covar_inv =np.array([mat_inv[0][0],mat_inv[0][1],mat_inv[1][1]])
#             dat = np.append(dat,covar_inv)


# dat = dat.reshape(-1,3)
# print(dat)
# [dat2,removed,removed_small_covar]=batch_data_loader.cook_Q_for_hgm(dat,small_covar=1e-3,threshould=1e-5,initial_point = np.array([[-1.0,0.0,-1.0, -1]]),even = None)

# sol=batch_hgm_erf_diff(dat2,qsize=-1) 
# def nc2(x):
#     return (np.pi/np.sqrt(x[0]*x[2]-x[1]**2))


# import tmp_erf_diff_fg
# import tmp_erf_diff_gf
# import tmp_erf_diff_gg

# print(sol)
# sol2 = []
# for i in range(0,dat2.shape[0]-1):
#     x =np.array([dat2[i][0],dat2[i][1],dat2[i][2]])
#     t_span=[0.0,1.0]
#     iv_fg=[0,1/2];
#     iv_gf=[0,1/2];  # x を入れ替えるので iv は fg に同じ.
#     iv_gg=[np.pi,1];

    ##Ref: 2024-08-19-erf-diff-by-sum.rr
    #  sol_ff=(4/np.pi)*(np.pi*x[1])/(2*np.power((x[0]-1)*(x[2]-1) - x[1]**2,3/2))
#    sol_ff=4*x[1]/(2*np.power((x[0]-1)*(x[2]-1) - x[1]**2,3/2))
    #sol_fg = tmp_erf_diff_fg.solve_ivp(tmp_erf_diff_fg.f_erf_diff_fg,t_span,iv_fg,method='RK45',args=(x,),rtol=1e-10,atol=1e-10)
    #sol_gf = scipy.integrate.solve_ivp(tmp_erf_diff_gf.f_erf_diff_gf,t_span,iv_gf,method='RK45',args=(x,),rtol=1e-10,atol=1e-10)
    #sol_gg = tmp_erf_diff_gg.solve_ivp(tmp_erf_diff_gg.f_erf_diff_gg,t_span,iv_gg,method='RK45',args=(x,),rtol=1e-10,atol=1e-10)
    #print(sol_gg.y[0,-1])
#     tmp = e_erf_diff_by_sum.e_erf_diff_value_by_sum(x)
#     sol2 = np.append(sol2,tmp)

# print(sol2)

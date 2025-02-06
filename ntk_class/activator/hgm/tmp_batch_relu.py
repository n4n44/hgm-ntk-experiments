#Do pip3 install --upgrade --force-reinstall scipy on Ubuntu 20.04
# P11, P12, P22 are Pfaffian matrices for airy. x11, x12, x22 are x_{11}, x_{12}, x_{22}.
#Use this code to build your own function.
global N_calls; global Q; global Last_value
def f_activator(t,f):
# activator = relu
  global Q; global N_calls
  N_calls = N_calls+1; k = int(np.floor(t))
  dx=Q[k+1]-Q[k]
  [x11,x12,x22]=Q[k]+dx*(t-k)
  d1=-((x12)**(2))+(x22)*(x11)
  p11=np.array([[(-1)/(x11),(-((1/2)*(x12)))/(x11)],[(-((2)*(x12)))/((d1)*(x11)),((-((x12)**(2)))-(((3/2)*(x22))*(x11)))/((d1)*(x11))]])
  p12=np.array([[0,1],[(4)/(d1),((5)*(x12))/(d1)]])
  p22=np.array([[(-1)/(x22),(-((1/2)*(x12)))/(x22)],[(-((2)*(x12)))/((d1)*(x22)),((-((x12)**(2)))-(((3/2)*(x22))*(x11)))/((d1)*(x22))]])
  p=p11*dx[0]+p12*dx[1]+p22*dx[2]
  return p@f

# hgm_batch  templates
#
# 
# Sample code
#

##Example. include tmp_pf_resin.py 

import numpy as np
#import batch_data_loader
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
#import gauss_hermite_2D


#
# changelog
#   2024-09-13-hgm-batch-erf.py,  hgm_batch_erf.py, 
#
# Note
#   Example 1. include files generated by Derive-other/2024-10-10-ReSin.rr
#

# dat は cook_Q_for_hgm の戻り値を使う
#  qsize  何個まとめて計算するか?  -1 は全部.
def batch_hgm(dat,initial_value,qsize=-1,rtol = 1e-10,atol = 1e-10):
  global Q
  global N_calls 
  global Last_value
  iv = initial_value
  Q=dat
  if (qsize < 0): qsize=Q.shape[0]
  print('qsize:',qsize)
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
  start=time.time()
  if atol == 0:
    sol = solve_ivp(f_activator,t_span,iv,method='RK45',t_eval=list(range(NN-1)),rtol=rtol)
  else :
    sol = solve_ivp(f_activator,t_span,iv,method='RK45',t_eval=list(range(NN-1)),rtol=rtol,atol = atol)
  end=time.time(); print('Time=',end-start,'s')
  print(sol)
  print('sol.y[0,-1]=',sol.y[0,-1])
  
  print('Last Q is ',Q[qsize-2])
  lastq=Q[qsize-2]
  Last_value=sol.y[0,-1]
  print('Last_value         = ',Last_value)
##Edit  resin_kernel is for resin  
#  gauss_hermite_2D.verbose(False)
  #print('By gauss_hermite_2D= ',gauss_hermite_2D.integrate_2D_normal(gauss_hermite_2D.resin_kernel,np.array([[lastq[0],lastq[1]],[lastq[1],lastq[2]]])))

  print('N_calls=',N_calls)
  print('Q[NN-2]=',Q[NN-2])
  return sol

'''
# test code for resin
##Edit
fName='../../Data/test_input3.csv'
dat = np.loadtxt(fName)
##Edit params. small_covar=1e-3, large_covar=10 はだめ.  append_dummy=True とすると計算しない最後の点を追加.
[dat2,removed,removed_small_covar]=batch_data_loader.cook_Q_for_hgm(dat,small_covar=1e-2,large_covar=1,threshould=1e-5,initial_point=[[-1,1/100,-1]]);
# 戻り値 np.array([[x11,x12,x22,idx0],[x11,x12,x22,idx1],...])
#   x が sort されてる. idx は元のデータで何番目か?

# dat2 を図示.
[pos1,pos2]=[0,1]; to=dat2.shape[0]
for i in range(to):
  print(dat2[i][0]*dat2[i][2]-dat2[i][1]**2)
plt.plot(dat2.T[pos1][0:to],dat2.T[pos2][0:to],color='blue')
plt.show()


dat=dat2
# tmp_erf.py
#iv=[1/2,5/8,1/2*np.pi,5/8,3/4*np.pi,3/4*np.pi,25/32,9/8*np.pi]
##Edit initial value from Derive-other/2024-10-07-ReSin.rr
iv=[0.18255369716266250124729815805,0.1541654230495132614215893156,0.24333194878201707952525389896,0.1541654230495132614215893156,0.2562117075877708176849345367,0.52242298789508171655711316607,0.2562117075877708176849345367,0.38635347953862556976588171668]  # resin initial value at (-1,1/100,-1)\n";

# 期待値をまとめて(all at once)計算.
#sol=batch_hgm(dat2,initial_value=iv,qsize=30)  # 30個のみ計算
sol=batch_hgm(dat,initial_value=iv)
print('Expectation values are in sol.y[0].')

# end of test codes.
'''

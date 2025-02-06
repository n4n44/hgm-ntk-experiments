# hgm_batch_erf
#
# 
# Sample code
#
"""
import numpy as np
import hgm_batch_erf
import batch_data_loader
import matplotlib.pyplot as plt
fName='../hgm-ntk/Data/test_input3.csv'
dat = np.loadtxt(fName)
[dat2,removed,removed_small_covar]=batch_data_loader.cook_Q_for_hgm(dat,small_covar=1e-3,threshould=1e-5);

# dat2 を図示.
[pos1,pos2]=[0,1]; to=dat2.shape[0]
plt.plot(dat2.T[pos1][0:to],dat2.T[pos2][0:to],color='blue')
plt.show()

# 期待値をまとめて(all at once)計算.
#sol=hgm_batch_erf.batch_hgm_erf(dat2,qsize=30)  # 30個のみ計算
sol=hgm_batch_erf.batch_hgm_erf(dat2,qsize=-1)   # 全部計算.
print('Expectation values are in sol.y[0].')
"""
#
# changelog
#   2024-09-13-hgm-batch-erf
#
# Note
#
#Do pip3 install --upgrade --force-reinstall scipy on Ubuntu 20.04
# P11, P12, P22 are Pfaffian matrices for ReLU. x11, x12, x22 are x_{11}, x_{12}, x_{22}.
#
import numpy as np
from scipy.integrate import solve_ivp
import time

global N_calls
global Q
global Last_value

# 

def f_relu(t,f):
    global Q
    global N_calls
    N_calls = N_calls+1
    k = int(np.floor(t))
    dx=Q[k+1]-Q[k]
    [x11,x12,x22]=Q[k]+dx*(t-k)

    #------ from tmp_pf_relu.py
    d1=-((x12)**(2))+(x22)*(x11)
    d2=(x12)**(2)+(x22)*(x11)
    p11=np.array([[0,1,0,0,0],[(-1/2)/((x11)**(2)),(((1/2)*((x12)**(4)))-(((5/2)*((x22)**(2)))*((x11)**(2))))/(((d2)*(d1))*(x11)),(-((x12)**(3))+(((1/2)*(x22))*(x11))*(x12))/((d2)*((x11)**(2))),(((2)*((x22)**(2)))*((x12)**(2)))/(((d2)*(d1))*(x11)),(-((x12)**(4))+(((3)*(x22))*(x11))*((x12)**(2)))/((d2)*((x11)**(2)))],[0,((2)*((x12)**(3)))/((d2)*(d1)),(-((3/2)*(x22)))/(d2),(-(((2)*((x22)**(2)))*(x12)))/((d2)*(d1)),(-(((4)*(x22))*(x12)))/(d2)],[0,0,0,0,1],[(-((x22)*((x12)**(2))))/((d1)**(3)),((-((3/2)*((x12)**(6)))+(((3)*(x22))*(x11))*((x12)**(4)))-((((5/2)*((x22)**(2)))*((x11)**(2)))*((x12)**(2))))/((d2)*((d1)**(3))),((-(((5)*(x22))*((x12)**(5)))+(((19/4)*((x22)**(2)))*(x11))*((x12)**(3)))-((((3/4)*((x22)**(3)))*((x11)**(2)))*(x12)))/((d2)*((d1)**(3))),(((((7/2)*((x22)**(2)))*((x12)**(4)))-((((4)*((x22)**(3)))*(x11))*((x12)**(2))))-(((1/2)*((x22)**(4)))*((x11)**(2))))/((d2)*((d1)**(3))),((-(((19/2)*(x22))*((x12)**(2))))-(((5/2)*((x22)**(2)))*(x11)))/((d2)*(d1))]])
    p12=np.array([[0,0,1,0,0],[0,((2)*((x12)**(3)))/((d2)*(d1)),(-((3/2)*(x22)))/(d2),(-(((2)*((x22)**(2)))*(x12)))/((d2)*(d1)),(-(((4)*(x22))*(x12)))/(d2)],[0,0,0,0,4],[0,(-(((2)*((x11)**(2)))*(x12)))/((d2)*(d1)),(-((3/2)*(x11)))/(d2),((2)*((x12)**(3)))/((d2)*(d1)),(-(((4)*(x11))*(x12)))/(d2)],[((d2)*(x12))/((d1)**(3)),(-((((3)*(x22))*((x11)**(2)))*((x12)**(3)))+(((5)*((x22)**(2)))*((x11)**(3)))*(x12))/((d2)*((d1)**(3))),(((7/2)*((x12)**(6))+(((19/4)*(x22))*(x11))*((x12)**(4)))-((((17/2)*((x22)**(2)))*((x11)**(2)))*((x12)**(2)))+((9/4)*((x22)**(3)))*((x11)**(3)))/((d2)*((d1)**(3))),(-((((3)*((x22)**(2)))*(x11))*((x12)**(3)))+(((5)*((x22)**(3)))*((x11)**(2)))*(x12))/((d2)*((d1)**(3))),((8)*((x12)**(3))+(((16)*(x22))*(x11))*(x12))/((d2)*(d1))]])
    p22=np.array([[0,0,0,1,0],[0,0,0,0,1],[0,(-(((2)*((x11)**(2)))*(x12)))/((d2)*(d1)),(-((3/2)*(x11)))/(d2),((2)*((x12)**(3)))/((d2)*(d1)),(-(((4)*(x11))*(x12)))/(d2)],[(-1/2)/((x22)**(2)),(((2)*((x11)**(2)))*((x12)**(2)))/(((d2)*(d1))*(x22)),(-((x12)**(3))+(((1/2)*(x22))*(x11))*(x12))/((d2)*((x22)**(2))),(((1/2)*((x12)**(4)))-(((5/2)*((x22)**(2)))*((x11)**(2))))/(((d2)*(d1))*(x22)),(-((x12)**(4))+(((3)*(x22))*(x11))*((x12)**(2)))/((d2)*((x22)**(2)))],[(-((x11)*((x12)**(2))))/((d1)**(3)),(((((7/2)*((x11)**(2)))*((x12)**(4)))-((((4)*(x22))*((x11)**(3)))*((x12)**(2))))-(((1/2)*((x22)**(2)))*((x11)**(4))))/((d2)*((d1)**(3))),((-(((5)*(x11))*((x12)**(5)))+(((19/4)*(x22))*((x11)**(2)))*((x12)**(3)))-((((3/4)*((x22)**(2)))*((x11)**(3)))*(x12)))/((d2)*((d1)**(3))),((-((3/2)*((x12)**(6)))+(((3)*(x22))*(x11))*((x12)**(4)))-((((5/2)*((x22)**(2)))*((x11)**(2)))*((x12)**(2))))/((d2)*((d1)**(3))),((-(((19/2)*(x11))*((x12)**(2))))-(((5/2)*(x22))*((x11)**(2))))/((d2)*(d1))]])
    #------
    p=p11*dx[0]+p12*dx[1]+p22*dx[2]
    return p@f

#cook_Q_for_hgm(dat,small_covar=1e-3,threshould=1e-5,initial_point=np.array([[-1,0,-1]]),even=True):
#  defined in 2024-09-12-batch-erf.py
# 戻り値 np.array([[x11,x12,x22,idx0],[x11,x12,x22,idx1],...])
#   x が sort されてる. idx は元のデータで何番目か?

# dat は cook_Q_for_hgm の戻り値を使う
#  qsize  何個まとめて計算するか?  -1 は全部.
def batch_hgm(dat,qsize=-1,rtol = 1e-10,atol = 1e-10):
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
  iv=[1/4,1/4,np.pi/8,1/4,1/4]
  start=time.time()
  #print(list(range(NN-1)))

#  sol = solve_ivp(f_erf,t_span,iv,method='RK45',t_eval=list(range(NN-1)),rtol=1e-10,atol=1e-10)
  sol = solve_ivp(f_relu,t_span,iv,method='RK45',t_eval=list(range(NN-1)),rtol=rtol,atol=atol)
  end=time.time(); print('Time=',end-start,'s')
  print(sol)
  print('sol.y[0,-1]=',sol.y[0,-1])
  Last_value=sol.y[0,-1]
  print('N_calls=',N_calls)
  print('Q[NN-2]=',Q[NN-2])
  return sol

def batch_hgm_erf2(dat,iv,qsize=-1,rtol = 1e-10,atol = 1e-10):
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

  t_span=[0.0,1.0]
  # tmp_erf.py
  iv=iv
  start=time.time()
  #print(list(range(NN-1)))

#  sol = solve_ivp(f_erf,t_span,iv,method='RK45',t_eval=list(range(NN-1)),rtol=1e-10,atol=1e-10)
  sol = solve_ivp(f_erf,t_span,iv,method='RK45',t_eval=list(range(NN-1)),rtol=rtol,atol=atol)
  end=time.time(); print('Time=',end-start,'s')
  print(sol)
  print('sol.y[0,-1]=',sol.y[0,-1])
  Last_value=sol.y[0,-1]
  print('N_calls=',N_calls)
  print('Q[NN-2]=',Q[NN-2])
  return sol
# test codes


"""
npyName='tmp-Q-data-may-remove'
print('Loading %s.npy' % (npyName))
dat=np.load(npyName+'.npy')
#sol=batch_hgm_erf(dat,qsize=30)
sol=batch_hgm_erf(dat)

print('\nComputing the each steps straightly.')
start=time.time()
N_calls=0
npyName='tmp-Q-data-may-remove'
print('Loading %s.npy' % (npyName))
Q_orig=np.load(npyName+'.npy')
QQ_orig=(Q_orig.T)[0:-1] # remove index
Q_orig=QQ_orig.T


iv=[1/2,5/8,1/2*np.pi,5/8,3/4*np.pi,3/4*np.pi,25/32,9/8*np.pi]
NN = Q.shape[0]
for i in range(1,NN-1):
  Q=[Q_orig[0],Q_orig[i],Q_orig[i]]
  t_span=[0.0,1.0]
  sol2 = solve_ivp(f_erf,t_span,iv,method='RK45',rtol=1e-10,atol=1e-10)
end=time.time(); print('Time=',end-start,'s')
print('sol2.y[0,-1]=',sol2.y[0,-1])
last_value2=sol2.y[0,-1]
print('N_calls=',N_calls)
print('diff of last value=',last_value2-Last_value)
print('ratio of last value=',(last_value2-Last_value)/Last_value)
"""
# end of test codes.

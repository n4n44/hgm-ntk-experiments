#Do pip3 install --upgrade --force-reinstall scipy on Ubuntu 20.04
import numpy as np
from scipy.integrate import solve_ivp
def nc2(x):
  return (np.pi/np.sqrt(x[0]*x[2]-x[1]**2))
# z: initial value, x: evaluation point
def f_ReLU_diff(t,z,x):
  p11=x[0]; p12=x[1]; p22=x[2];
  d1=((((p22+1) )*(p11)+-((p12)**(2))+p22+1) )*((t)**(2))+((-(p11)+(-(p22))-(2)) )*(t)+1
  p=np.array([[((((((-(p22))-(1)) )*(p11)+(-(p22))-(1)) )*(t)+(1/2)*(p11)+(1/2)*(p22)+1)/(((((p22+1) )*(p11)+p22+1) )*((t)**(2))+((-(p11)+(-(p22))-(2)) )*(t)+1),(((-(((1/2)*(p12))*(p11))+(((-((1/2)*(p22)))-(1)) )*(p12)) )*(t)+p12)/(((((p22+1) )*(p11)+p22+1) )*((t)**(2))+((-(p11)+(-(p22))-(2)) )*(t)+1)],[(((-(((1/2)*(p12))*(p11))+(((-((1/2)*(p22)))-(1)) )*(p12)) )*(t)+p12)/((((((p22+1) )*(p11)+p22+1) )*(d1))*((t)**(2))+(((-(p11)+(-(p22))-(2)) )*(d1))*(t)+d1),(((((((-((2)*((p22)**(2))))-((4)*(p22)))-(2)) )*((p11)**(2))+(((((2)*(p22)+2) )*((p12)**(2))+((-((4)*((p22)**(2))))-((8)*(p22)))-(4)) )*(p11)+(((2)*(p22)+2) )*((p12)**(2))+((-((2)*((p22)**(2))))-((4)*(p22)))-(2)) )*((t)**(3))+(((((3)*(p22)+3) )*((p11)**(2))+((-((5/2)*((p12)**(2)))+(3)*((p22)**(2))+(12)*(p22)+9) )*(p11)+(((-((5/2)*(p22)))-(5)) )*((p12)**(2))+(3)*((p22)**(2))+(9)*(p22)+6) )*((t)**(2))+((-((p11)**(2))+(((-((4)*(p22)))-(6)) )*(p11)+(3)*((p12)**(2))+((-((p22)**(2)))-((6)*(p22)))-(6)) )*(t)+p11+p22+2)/((((((p22+1) )*(p11)+p22+1) )*(d1))*((t)**(2))+(((-(p11)+(-(p22))-(2)) )*(d1))*(t)+d1)]])
  return (p@z)

def e_ReLU_diff(x,rtol = 1e-10,atol = 1e-10):
  t_span=[0.0,1.0]
  iv=[np.pi/4,1/2]
  if atol == 0:
    sol = solve_ivp(f_ReLU_diff,t_span,iv,method='RK45',args=(x,),rtol=rtol)
  else:
    sol = solve_ivp(f_ReLU_diff,t_span,iv,method='RK45',args=(x,),rtol=rtol,atol=atol)
  return sol.y[0,-1]/nc2(x)
  
# t_span=[0.0,1.0]
# iv=[np.pi/4,1/2]
# x=[-2,-0.5,-1.5] #sample pt
# #x=[-2,-0.1,-2] #sample pt
# #x=[-10,-7,-5] #sample pt
# #x=[-10,-6,-5] #sample pt
# sol = solve_ivp(f_ReLU_diff,t_span,iv,method='RK45',args=(x,),rtol=1e-10,atol=1e-10)
# print(sol)
# print(sol.y[0,-1])
# print('x=',x,end=', ')
# print('E[ReLU diff]=',sol.y[0,-1]/nc2(x))

#Do pip3 install --upgrade --force-reinstall scipy on Ubuntu 20.04
import numpy as np
from scipy.integrate import solve_ivp
def nc2(x):
  return (np.pi/np.sqrt(x[0]*x[2]-x[1]**2))
# z: initial value, x: evaluation point
def f_relu(t,z,x):
  p11=x[0]; p12=x[1]; p22=x[2];
  d1=((((p22+1) )*(p11)+-((p12)**(2))+p22+1) )*((t)**(2))+((-(p11)+(-(p22))-(2)) )*(t)+1
  p=np.array([[((((((-((2)*(p22)))-(2)) )*(p11)+(-((2)*(p22)))-(2)) )*(t)+p11+p22+2)/(((((p22+1) )*(p11)+p22+1) )*((t)**(2))+((-(p11)+(-(p22))-(2)) )*(t)+1),(((-(((1/2)*(p12))*(p11))+(((-((1/2)*(p22)))-(1)) )*(p12)) )*(t)+p12)/(((((p22+1) )*(p11)+p22+1) )*((t)**(2))+((-(p11)+(-(p22))-(2)) )*(t)+1)],[(((-(((2)*(p12))*(p11))+(((-((2)*(p22)))-(4)) )*(p12)) )*(t)+(4)*(p12))/((((((p22+1) )*(p11)+p22+1) )*(d1))*((t)**(2))+(((-(p11)+(-(p22))-(2)) )*(d1))*(t)+d1),(((((((-((3)*((p22)**(2))))-((6)*(p22)))-(3)) )*((p11)**(2))+(((((3)*(p22)+3) )*((p12)**(2))+((-((6)*((p22)**(2))))-((12)*(p22)))-(6)) )*(p11)+(((3)*(p22)+3) )*((p12)**(2))+((-((3)*((p22)**(2))))-((6)*(p22)))-(3)) )*((t)**(3))+(((((9/2)*(p22)+9/2) )*((p11)**(2))+((-((4)*((p12)**(2)))+(9/2)*((p22)**(2))+(18)*(p22)+27/2) )*(p11)+(((-((4)*(p22)))-(8)) )*((p12)**(2))+(9/2)*((p22)**(2))+(27/2)*(p22)+9) )*((t)**(2))+((-((3/2)*((p11)**(2)))+(((-((6)*(p22)))-(9)) )*(p11)+(5)*((p12)**(2))+((-((3/2)*((p22)**(2))))-((9)*(p22)))-(9)) )*(t)+(3/2)*(p11)+(3/2)*(p22)+3)/((((((p22+1) )*(p11)+p22+1) )*(d1))*((t)**(2))+(((-(p11)+(-(p22))-(2)) )*(d1))*(t)+d1)]])
  return (p@z)

def e_relu(x,rtol = 1e-10, atol = 1e-10):
  if x[0]*x[2]-x[1]**2 <= 0:
    print('Error: input x is not positive definite.')
    return -1
  t_span=[0.0,1.0]
  iv=[1/4, np.pi/8];  #ReLU initial value at -1,0,-1
  if atol == 0:
    sol = solve_ivp(f_relu,t_span,iv,method='RK45',args=(x,),rtol=rtol)
  else:
    sol = solve_ivp(f_relu,t_span,iv,method='RK45',args=(x,),rtol=rtol,atol = atol)
  return sol.y[0,-1]/nc2(x)

'''
t_span=[0.0,1.0]
iv=[1/4, np.pi/8];  #ReLU initial value at -1,0,-1 
pts=[[-2,-0.5,-1.5],[-2,-0.1,-2],[-3,-0.6,-1],[-2.4,-0.7,-1.5],[-2.5,-0.5,-1.1],[-2,-1/2,-2],[-10,-9.1,-9],[-18,23,-30]]
print('Pts=',pts)
q=[]
for i in range(len(pts)):
   x = pts[i]
   sol = solve_ivp(f_relu,t_span,iv,method='RK45',args=(x,),rtol=1e-10)
   print('x=',x,end=', ')
   print(sol.y[0,-1])
   q.append(sol.y[0,-1])
print('Q=',q)
print('Last (sol.y).T[-1]=',(sol.y).T[-1])
'''

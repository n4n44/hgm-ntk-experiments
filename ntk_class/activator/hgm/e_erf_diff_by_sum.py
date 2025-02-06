from . import tmp_erf_diff_fg
from . import tmp_erf_diff_gf
from . import tmp_erf_diff_gg
import numpy as np
# x は [x11, x12, x22]
def nc2(x):
  return (np.pi/np.sqrt(x[0]*x[2]-x[1]**2))
def e_erf_diff_value_by_sum(x, rtol = 1e-10, atol = 1e-10):
  t_span=[0.0,1.0]
  iv_fg=[0,1/2];
  iv_gf=[0,1/2];  # x を入れ替えるので iv は fg に同じ.
  iv_gg=[np.pi,1];
##Ref: 2024-08-19-erf-diff-by-sum.rr
#  sol_ff=(4/np.pi)*(np.pi*x[1])/(2*np.power((x[0]-1)*(x[2]-1) - x[1]**2,3/2))
  sol_ff=4*x[1]/(2*np.power((x[0]-1)*(x[2]-1) - x[1]**2,3/2))
  if atol == 0:
    sol_fg = tmp_erf_diff_fg.solve_ivp(tmp_erf_diff_fg.f_erf_diff_fg,t_span,iv_fg,method='RK45',args=(x,),rtol=rtol)
    sol_gf = tmp_erf_diff_gf.solve_ivp(tmp_erf_diff_gf.f_erf_diff_gf,t_span,iv_gf,method='RK45',args=(x,),rtol=rtol)
    sol_gg = tmp_erf_diff_gg.solve_ivp(tmp_erf_diff_gg.f_erf_diff_gg,t_span,iv_gg,method='RK45',args=(x,),rtol=rtol)

  else:
    sol_fg = tmp_erf_diff_fg.solve_ivp(tmp_erf_diff_fg.f_erf_diff_fg,t_span,iv_fg,method='RK45',args=(x,),rtol=rtol,atol=atol)
    sol_gf = tmp_erf_diff_gf.solve_ivp(tmp_erf_diff_gf.f_erf_diff_gf,t_span,iv_gf,method='RK45',args=(x,),rtol=rtol,atol=atol)
    sol_gg = tmp_erf_diff_gg.solve_ivp(tmp_erf_diff_gg.f_erf_diff_gg,t_span,iv_gg,method='RK45',args=(x,),rtol=rtol,atol=atol)

  ee=(sol_ff+sol_fg.y[0,-1])+(sol_gf.y[0,-1])+(sol_gg.y[0,-1])
  #print('x=',x,', uEE=',ee,end='\n')
  return ee/nc2(x)
'''
# test inputs. 2024-08-14-erf-diff-init.rr の値と比べる --> OK.
e_erf_diff_value_by_sum([-1,0,-1])
e_erf_diff_value_by_sum([-1.1,-0.1,-1.2])
e_erf_diff_value_by_sum([-1.2,-0.2,-1.3])
e_erf_diff_value_by_sum([-0.2,-0.05,-0.15])
'''

# たくさんの点での値を求める方は value でなく values.
def e_erf_diff_ff_values(x,t_points):
  r=np.array([x[0]-(-1),x[1],x[2]-(-1)])
  r0=np.array([-1,0,-1])
  v=[]
  for t in t_points:
##Ref: 2024-08-19-erf-diff-by-sum.rr
#  sol_ff=(4/np.pi)*(np.pi*x[1])/(2*np.power((x[0]-1)*(x[2]-1) - x[1]**2,3/2))
    xxx=r*t+r0
    ff=4*xxx[1]/(2*np.power((xxx[0]-1)*(xxx[2]-1) - xxx[1]**2,3/2))
    v.append(ff)
  return v

def e_erf_diff_values_by_sum(x,t_points):
  t_span=[0.0,1.0]
  iv_fg=[0,1/2];
  iv_gf=[0,1/2];  # x を入れ替えるので iv は fg に同じ.
  iv_gg=[np.pi,1];

  sol_ff=e_erf_diff_ff_values(x,t_points)
  sol_fg = tmp_erf_diff_fg.solve_ivp(tmp_erf_diff_fg.f_erf_diff_fg,t_span,iv_fg,t_eval=t_points,method='RK45',args=(x,),rtol=1e-10,atol=1e-10)
  sol_gf = tmp_erf_diff_gf.solve_ivp(tmp_erf_diff_gf.f_erf_diff_gf,t_span,iv_gf,t_eval=t_points,method='RK45',args=(x,),rtol=1e-10,atol=1e-10)
  sol_gg = tmp_erf_diff_gg.solve_ivp(tmp_erf_diff_gg.f_erf_diff_gg,t_span,iv_gg,t_eval=t_points,method='RK45',args=(x,),rtol=1e-10,atol=1e-10)
  ee=(sol_ff+sol_fg.y[0])+(sol_gf.y[0])+(sol_gg.y[0])
  print('x=',x,', uEE=',ee,end='\n')
  return ee

# e_erf_diff_values_by_sum([-1.2,-0.2,-1.3],np.linspace(0,1,5))



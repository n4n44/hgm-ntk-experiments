import numpy as np
import ntk_class
import ntk_class.activator
import matplotlib.pyplot as plt

def test_func(x):
    return np.sin(np.pi*x)

np.random.seed(0)
num_pts = 15
sample_x = np.linspace(-1,1,num_pts).reshape(num_pts, -1)
sample_y = test_func(sample_x).reshape(num_pts,1)

num_test = 20
test_x = np.linspace(-1, 1,num_test).reshape(num_test,1)
test_y = test_func(test_x).reshape(num_test,1)
layers = 2

print('----gauss-herm----')
activator = ntk_class.activator.Resin()
model = ntk_class.ntk(layers,activator,beta=0.5,debug_mode = False)

model.train(sample_x,sample_y,diag_reg = 0.001, mode = 'gauss_herm', get_timing = True)
kernel_gh = model.kernel_mat.copy()
pred_gh = model.pred(test_x,mode = 'gauss_herm')

print('----hgm-----')
activator = ntk_class.activator.Resin()
model = ntk_class.ntk(layers,activator,beta=0.5,debug_mode = False)

model.train(sample_x,sample_y,diag_reg = 0.001, mode = 'hgm_fast', get_timing = True)
kernel_hgm = model.kernel_mat.copy()
pred_hgm = model.pred(test_x,mode = 'hgm_fast')

print('----monte_calro----')
activator = ntk_class.activator.Resin()
model = ntk_class.ntk(layers,activator,beta=0.5,debug_mode = False)

model.train(sample_x,sample_y,diag_reg = 0.001, mode = 'monte_carlo', get_timing = True)
kernel_mc = model.kernel_mat.copy()
pred_mc = model.pred(test_x,mode = 'monte_carlo')

kernel_error_gh_hgm = np.linalg.norm(kernel_gh-kernel_hgm)/(sample_x.shape[0]**2)
kernel_error_gh_mc = np.linalg.norm(kernel_gh-kernel_mc)/(sample_x.shape[0]**2)
pred_error_gh_hgm  = np.linalg.norm(pred_gh-pred_hgm)/(test_x.shape[0])
pred_error_gh_mc  = np.linalg.norm(pred_gh-pred_mc)/(test_x.shape[0])
print(kernel_error_gh_hgm)
print(kernel_error_gh_mc)
print(pred_error_gh_hgm)
print(pred_error_gh_mc)
plt.plot(test_x,pred_hgm, color = 'green',label='hgm_fast')
plt.plot(test_x,pred_gh, color = 'magenta',label = 'gauss_herm')
plt.plot(test_x,pred_mc,color = 'purple',label = 'monte_carlo')
plt.legend()
plt.show()

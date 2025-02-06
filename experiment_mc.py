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
size_list = [10,50,100,500,1000,5000]
activator = ntk_class.activator.ReLU()
model = ntk_class.ntk(layers,activator,beta=0.5,debug_mode = False)
model.train(sample_x,sample_y,diag_reg = 0.001, mode = 'closed', get_timing = True)

kernel_mat_closed = model.kernel_mat
pred_closed = model.pred(test_x,mode = 'closed')

for size in size_list:
    print(f'n = {size}')
    activator = ntk_class.activator.ReLU()
    model = ntk_class.ntk(layers,activator,beta=0.5,debug_mode = False)
    model.set_mc_size(size)
    model.train(sample_x,sample_y,diag_reg = 0.001, mode = 'monte_carlo', get_timing = True)

    pred = model.pred(test_x,mode = 'monte_carlo')
    
    kernel_error = np.linalg.norm(kernel_mat_closed-model.kernel_mat)/sample_x.shape[0]**2
    pred_error = np.linalg.norm(np.asarray(pred_closed)-np.asarray(pred))/test_x.shape[0]
    print('kernel_error:',kernel_error)
    print('pred_error:',pred_error)
    print()
# pred = model.train_pred(sample_x,test_x, sample_y,diag_reg = 0.001, mode = 'hgm_fast', get_timing = True)
plt.plot(test_x,test_y,color = 'red')
plt.plot(test_x,pred,color = 'blue')
plt.show()

'''
(225, 4)
Training finished.
Training time:0.009811162948608398, solve_time: 0.00028061866760253906
pred time: 0.014391660690307617
n = 10
(225, 4)
Training finished.
Training time:0.1633439064025879, solve_time: 0.00016832351684570312
pred time: 0.2155287265777588
kernel_error: 0.027303849887845885
pred_error: 0.666342004222004

n = 50
(225, 4)
Training finished.
Training time:0.7870621681213379, solve_time: 7.462501525878906e-05
pred time: 1.0509436130523682
kernel_error: 0.011296481001493287
pred_error: 0.26718826238265875

n = 100
(225, 4)
Training finished.
Training time:1.5673964023590088, solve_time: 0.00018525123596191406
pred time: 2.0937647819519043
kernel_error: 0.007608710585293065
pred_error: 0.5893258824429946

n = 500
(225, 4)
Training finished.
Training time:7.8305583000183105, solve_time: 0.00021028518676757812
pred time: 10.45906662940979
kernel_error: 0.0035035539062165835
pred_error: 3.738040954665725

n = 1000
(225, 4)
Training finished.
Training time:15.645933151245117, solve_time: 0.0002465248107910156
pred time: 21.018333196640015
kernel_error: 0.0027396878131718226
pred_error: 0.3588496946794468

n = 5000
(225, 4)
Training finished.
Training time:78.38959908485413, solve_time: 0.0002608299255371094
pred time: 104.86528277397156
kernel_error: 0.0013190718002394244
pred_error: 0.15750577821059533

Training finished.
Training time:157.00646829605103, solve_time: 0.0002562999725341797
pred time: 211.19979667663574
kernel_error: 0.0008570235993981631
pred_error: 0.14584475737243333

n = 15000
(225, 4)
Training finished.
Training time:240.0711109638214, solve_time: 0.00021886825561523438
pred time: 312.07188534736633
kernel_error: 0.0006749231108517454
pred_error: 0.12389470040132242
'''

#Do pip3 install --upgrade --force-reinstall scipy on Ubuntu 20.04
# P11, P12, P22 are Pfaffian matrices for diff(ReLU). x11, x12, x22 are x_{11}, x_{12}, x_{22}.
#Use this code to build your own function.
def f_ReLU_diff(t,f):
  d1=-((x12)**(2))+(x22)*(x11)
  p11=np.array([[(-1/2)/(x11),(-((1/2)*(x12)))/(x11)],[(-((1/2)*(x12)))/((d1)*(x11)),((-((1/2)*((x12)**(2))))-((x22)*(x11)))/((d1)*(x11))]])
  p12=np.array([[0,1],[(1)/(d1),((3)*(x12))/(d1)]])
  p22=np.array([[(-1/2)/(x22),(-((1/2)*(x12)))/(x22)],[(-((1/2)*(x12)))/((d1)*(x22)),((-((1/2)*((x12)**(2))))-((x22)*(x11)))/((d1)*(x22))]])

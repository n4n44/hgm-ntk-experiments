import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time

def det_covar(x):
    return x[0]*x[2]-x[1]**2
# dat[i] は [x00,x12,x22]
# dat から det(covar) < 1e-3 を除く. det(covar)=1/(4*det(x))
# small_covar=1e-3
def remove_covar(dat,small_covar,large_covar):
    n=dat.shape[0]
    newdat=[]
    removed_small=[]
    removed_large = []
    for i in range(n):
        mydet=det_covar(list(dat[i]))
        if mydet < small_covar :
            #print('det',mydet)
            removed_small.append(list(dat[i]))
        elif mydet > large_covar:
            removed_large.append(list(dat[i]))
        else:
            newdat.append([-dat[i][2]/(2*mydet),dat[i][1]/(2*mydet),-dat[i][0]/(2*mydet),dat[i][3]])
    return [np.array(newdat),removed_small,removed_large]

# dat から偶数行 or 奇数行のみを取り出す.
def get_even_or_odd_lines(dat,even=True):
    n=dat.shape[0]
    newdat=[]
    if even:
        bias=0
    else:
        bias=1
    for i in range(bias,n,2):
        newdat.append(list(dat[i]))
    return np.array(newdat)

def distance(v1,v2):
    return np.sum(np.abs(v1-v2))

# dat から distance を用いて近い点を取り除く.
def remove_close_points(dat,threshould=1e-5):
    n=dat.shape[0]
    newdat=[dat[0]]
    removed=[]
    for i in range(1,n):
        if (distance(newdat[-1][0:3],dat[i][0:3]) > threshould):
            newdat.append(list(dat[i]))
        else:
            tt=list(dat[i])
            tt.append(newdat[-1][3]) # index of near points
            removed.append(tt)
    return [np.array(newdat),np.array(removed)]

def add_index(dat):

    n=dat.shape[0]
    newdat=[]
    for i in range(n):
        y=list(dat[i])
        y.append(i)
        #      print(y)
        newdat.append(y)
        #    print(newdat)
    return np.array(newdat)

def make_pair(values):
    pair_list = []
    for i in range(values.shape[0]):
        for j in range(values.shape[0]):
            pair_list = np.append(pair_list,np.array([values[i],values[j]]))
    pair_array = np.array(pair_list).reshape(-1,values.shape[1]*2)
    return pair_array


def sort_dat(dat,small_covar=1e-3,large_covar=np.inf,threshould=1e-5,initial_point=np.array([[-1,0,-1, -1]]),even=True):
    start=time.time()
    removed=[]
    if (type(even)==type(True)):
        dat1 = get_even_or_odd_lines(dat,even)
        #print('dat1=',dat1)
    else:
        dat1=dat
    [dat2,removed_small_covar,removed_large_covar]=remove_covar(dat1,small_covar,large_covar)
    #print('dat=',dat)
    #print('dat2=',dat2)
    print(np.array(dat2).shape)
    df2=pd.DataFrame(dat2)
    df3=df2.sort_values([0,1,2,3], ascending=False)
    #print(df3)
    dat4=df3.to_numpy(dtype='double')
    #print('dat4=',dat4)

    [dat5,tt]=remove_close_points(dat4,threshould)
    removed.append(tt)
    if (isinstance(initial_point,np.ndarray)):
        if (distance(dat5[0][0:3],initial_point[0][0:3])>threshould):
           print(distance(dat5[0][0:3],initial_point[0][0:3]))
           dat5=np.vstack([initial_point,dat5])
    end=time.time();
    print('time of cook_Q_for_hgm: ',end-start)
    return [dat5,np.array(removed),np.array(removed_small_covar),np.array(removed_large_covar)]

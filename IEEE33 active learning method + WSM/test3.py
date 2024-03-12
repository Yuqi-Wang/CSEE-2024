from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF,DotProduct, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt
import main_py_fun_GK_OLTC as mfGK
from mpl_toolkits.mplot3d import Axes3D
import joblib
import ParetoCheck3 as PC
import config
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import multiprocessing
import main_py_fun_ws_OLTC as mfws
from keras import models
import random

initial_num = 1003
test_prop = 0.2
relax = config.get_relax()
MSE = config.get_MSE()
k = config.get_k()
MAE = config.get_MAE()
diff = config.get_diff()
test_prop = config.get_test_prop()
test_new = config.get_test_new()
num_iterations = config.get_num_iterations()
testing_num = 485

# Load the initial training data
w_set = np.load('w_train_ws_woPC_OLTC_'+str(initial_num)+'.npy')
w_set_new = np.zeros((initial_num,3))

test_w = np.load('w_test_ws_OLTC_'+str(testing_num)+'.npy')
test_w_new = np.zeros((testing_num,3))

for i in range(initial_num):
    w_set_new[i,0:2] = w_set[i,:]
    w_set_new[i,2] = 1 - w_set[i,:].sum()

for i in range(testing_num):
    test_w_new[i,0:2] = test_w[i,:]
    test_w_new[i,2] = 1 - test_w[i,:].sum()

print(w_set_new[0:3,:])
print(test_w_new[0:3,:])

np.save('w3_train_ws_woPC_OLTC_'+str(initial_num)+'.npy',w_set_new)
np.save('w3_test_ws_OLTC_'+str(testing_num)+'.npy',test_w_new)
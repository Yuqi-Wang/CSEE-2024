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

initial_num = 235
test_prop = 0.2
relax = config.get_relax()
MSE = config.get_MSE()
k = config.get_k()
MAE = config.get_MAE()
diff = config.get_diff()
test_prop = config.get_test_prop()
test_new = config.get_test_new()
num_iterations = config.get_num_iterations()

# for objective function 1
mse_list1 = np.load('mse_list1_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
mae_list1 = np.load('mae_list1_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
diff_list1 = np.load('diff_list1_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')

plt.figure(figsize=(12,3))
plt.subplot(131)
plt.plot(range(len(mse_list1)),mse_list1,'r--')
plt.subplot(132)
plt.plot(range(len(mae_list1)),mae_list1,'bs')
plt.subplot(133)
plt.plot(range(len(diff_list1)),diff_list1,'g^')
plt.savefig('Objective function 1 iterating '+str(num_iterations))
plt.show()


# for objective function 2
mse_list2 = np.load('mse_list2_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
mae_list2 = np.load('mae_list2_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
diff_list2 = np.load('diff_list2_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')

plt.figure(figsize=(12,3))
plt.subplot(131)
plt.plot(range(len(mse_list2)),mse_list2,'r--')
plt.subplot(132)
plt.plot(range(len(mae_list2)),mae_list2,'bs')
plt.subplot(133)
plt.plot(range(len(diff_list2)),diff_list2,'g^')
plt.savefig('Objective function 2 iterating '+str(num_iterations))
plt.show()


# for objective function 3
mse_list3 = np.load('mse_list3_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
mae_list3 = np.load('mae_list3_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
diff_list3 = np.load('diff_list3_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')

plt.figure(figsize=(12,3))
plt.subplot(131)
plt.plot(range(len(mse_list3)),mse_list3,'r--')
plt.subplot(132)
plt.plot(range(len(mae_list3)),mae_list3,'bs')
plt.subplot(133)
plt.plot(range(len(diff_list3)),diff_list3,'g^')
plt.savefig('Objective function 3 iterating '+str(num_iterations))
plt.show()



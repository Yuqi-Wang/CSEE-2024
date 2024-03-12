import numpy as np
import config
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ParetoCheck4 as PC4
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF,DotProduct, WhiteKernel
from keras import models
import random
import scipy.io as io

initial_num = config.get_initial_num()
num_iterations = config.get_num_iterations()
MSE = config.get_MSE()
MAE = config.get_MAE()
diff = config.get_diff_max()

f_train = np.load('f_train_iteration_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')

mse_list = np.load('mse_list_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
mae_list = np.load('mae_list_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')

diff_list1 = np.load('diff_list1_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
diff_list2 = np.load('diff_list2_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
diff_list3 = np.load('diff_list3_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')

# 调整成论文中的目标函数编号
# 第一列是碳排放，对应论文中的目标2；第二列是电压偏移，对应论文中的目标3；第三列是经济性，对应论文中的目标1
final_y = np.array([f_train[:,0], f_train[:,2], f_train[:,1]])
io.savemat('final_y'+'.mat',{'final_y':final_y})
io.savemat('mse_list'+'.mat',{'mse_list':mse_list})
io.savemat('mae_list'+'.mat',{'mae_list':mae_list})

io.savemat('diff_list1.mat',{'diff_list1':diff_list1})
io.savemat('diff_list2.mat',{'diff_list2':diff_list2})
io.savemat('diff_list3.mat',{'diff_list3':diff_list3})
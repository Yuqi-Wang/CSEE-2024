import numpy as np
import auxiliary2_fun as a2f
import scipy.io as io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
# 该文件用来计算不同w下三个目标的函数值，w包括benchmark中的所有数据点

FU_1 = config.get_FU_1()
FU_2 = config.get_FU_2()
FU_3 = config.get_FU_3()

FN_1 = config.get_FN_1()
FN_2 = config.get_FN_2()
FN_3 = config.get_FN_3()

initial_num = config.get_initial_num()
num_iterations = config.get_num_iterations()
MSE = config.get_MSE()
MAE = config.get_MAE()
diff = config.get_diff()
model = models.load_model('M_GK_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.h5')

num_sample = 30000
con_weight = io.loadmat('weights_wPC_'+str(num_sample)+'_89.mat')['weights'].T
NoLCA_y = io.loadmat('objectives_name_y_wPC_'+str(num_sample)+'_89.mat')['objectives']
con_y = model.predict(con_weight)

# Pareto check
con_y, con_weight = PC4.ParetoCheck(con_y, con_weight)
tune_y = np.array([con_y[:,0], con_y[:,2], con_y[:,1]])
name_tune_y = np.zeros([num_sample,3])

for i in range(num_sample):
    name_tune_y[i,0] = tune_y[0,i]*(FN_1-FU_1) + FU_1
    name_tune_y[i,1] = tune_y[1,i]*(FN_3-FU_3) + FU_3
    name_tune_y[i,2] = tune_y[2,i]*(FN_2-FU_2) + FU_2


delta_y = (name_tune_y-NoLCA_y)/np.abs(NoLCA_y)
io.savemat('objectives_delta_name_abs_y_wPC_'+str(num_sample)+'.mat',{'objectives':delta_y})
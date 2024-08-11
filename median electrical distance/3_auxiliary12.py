import numpy as np
import auxiliary2_fun as a2f
import scipy.io as io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ParetoCheck4 as PC
import config
# 该文件用来计算不同w下三个目标的函数值，w包括benchmark中的所有数据点

FU_1 = config.get_FU_1()
FU_2 = config.get_FU_2()
FU_3 = config.get_FU_3()

FN_1 = config.get_FN_1()
FN_2 = config.get_FN_2()
FN_3 = config.get_FN_3()


num_sample = 30000
tune_y = io.loadmat('objectives_wPC_'+str(num_sample)+'.mat')['objectives']
name_tune_y = np.zeros([num_sample,3])

for i in range(num_sample):
    name_tune_y[i,0] = tune_y[0,i]*(FN_1-FU_1) + FU_1
    name_tune_y[i,1] = tune_y[1,i]*(FN_3-FU_3) + FU_3
    name_tune_y[i,2] = tune_y[2,i]*(FN_2-FU_2) + FU_2

io.savemat('objectives_name_y_wPC_'+str(num_sample)+'.mat',{'objectives':name_tune_y})
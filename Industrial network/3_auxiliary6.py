import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import config
import os
import time
import scipy.io as io

obj = 4
P_pv_max = 10/100

relax = config.get_relax()
MSE = config.get_MSE()
k = config.get_k()
MAE = config.get_MAE()
diff = config.get_diff()
test_prop = config.get_test_prop()
test_new = config.get_test_new()
num_iterations = config.get_num_iterations()
diff_max = config.get_diff_max()
initial_num = config.get_initial_num()
testing_num = config.get_testing_num()
BatchSize = config.get_BatchSize()
Epochs = config.get_Epochs()
LearningRate = config.get_LearningRate()

row_list = [
    122,532,857,934,1345,1172
]

num_sample = 30000
exam_x = io.loadmat('weights_wPC_'+str(num_sample)+'.mat')['weights'].T


w_case = np.zeros([6,3])

for i,row in enumerate(row_list):
    print('w is :',exam_x[row,:])
    
    w_case[i,1],w_case[i,2],w_case[i,0] = exam_x[row,0],exam_x[row,1],exam_x[row,2]

io.savemat('w_case'+'.mat',{'w_case':w_case})
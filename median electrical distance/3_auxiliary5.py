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

w_case = io.loadmat('w_case.mat')['w_case']


pv_case = np.zeros([6,6])
ene_case = np.zeros([6,3])
pei_case = np.zeros([6,3])

for i in range(len(row_list)):
    print('w is :',w_case[i,:])
    result_data = np.load('.\DataFile\\result_obj'+str(obj)+'_w1_'+str(round(w_case[i,1],8))+'_w2_'+str(round(w_case[i,0],8))+'_pv'+str(P_pv_max)+'.npz')


    pv_case[i,:] = result_data['pv_cap'].squeeze()
    ene_case[i,:] = result_data['E_ess'].squeeze()
    pei_case[i,:] = result_data['p_ess'].squeeze()
    print('PV unit capacity: ', pv_case)
    print('ESS energy capacity: ', ene_case)
    print('ESS power capacity: ', pei_case)


io.savemat('pv_case.mat',{'pv_case':pv_case})
io.savemat('ene_case.mat',{'ene_case':ene_case})
io.savemat('pei_case.mat',{'pei_case':pei_case})

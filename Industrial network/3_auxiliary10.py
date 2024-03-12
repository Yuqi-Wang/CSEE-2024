import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import config
import os
import time
import scipy.io as io

obj = 4
P_pv_max = 10/100
year = 365 # the days in one year
c_e_b = io.loadmat('c_e_t.mat')['c_e_t']*1000*100 # the electricity price for buy at each hour and the unit is $/kWh，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MWh
c_e_b = c_e_b.reshape((1,1,24))
c_e_s = np.zeros((1,1,24)) # the electricity price for sell
for i in range(24):
    c_e_s[0,0,i] = c_e_b[0,0,i] * 0.7
ess_num = 3
pv_num = 6
c_ess_op = 0.00027*1000*100 # the operation cost of ess unit is $/kWh，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MWh
c_pv_cut = 0.033*1000*100 # the punishment on pv deserting, and the unit is $/kWh，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MWh
N = 20 # 随机场景的数量
p_pv = io.loadmat('p_pv.mat')['p_pv'][0:pv_num,0:N,:] # the value is in p.u. value and represents the fluctuation, the power factor of PV is 0.9，滞后

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

w_train = np.load('w_train_iteration_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
total_num = len(w_train)

w_final_train_x = np.zeros([total_num,3]) # 添加“_x”是因为后面减掉了一个point
pv_case = np.zeros([total_num,1])
ene_case = np.zeros([total_num,1])
pei_case = np.zeros([total_num,1])
ene2pei = np.zeros([total_num,1])
op_cost = np.zeros([total_num,1])

count = 0
for i in range(total_num):
    if (round(w_train[i,0],8) != 0.85486488):
        print('w is :',w_train[i,:])
        w_final_train_x[count,:] = w_train[i,:]
        
        result_data = np.load('.\DataFile2\\result_obj'+str(obj)+'_w1_'+str(round(w_train[i,0],8))+'_w2_'+str(round(w_train[i,1],8))+'_pv'+str(P_pv_max)+'.npz')
        pv_case[count,:] = result_data['pv_cap'].sum() * 100
        ene_case[count,:] = result_data['E_ess'].sum() * 100
        pei_case[count,:] = result_data['p_ess'].sum() * 100
        ene2pei[count,:] = ene_case[count,:] / pei_case[count,:]

        p_sub_b = result_data['p_sub_b']
        p_ch = result_data['p_ch']
        p_dch = result_data['p_dch']
        pv_cap = result_data['pv_cap']
        power_pv = result_data['power_pv']

        op_cost[count,:] = year * 1/N*sum(sum(p_sub_b[0,j,t]*c_e_b[0,0,t] for t in range(24)) + sum(p_ch[i,j,t]+p_dch[i,j,t] for i in range(ess_num) for t in range(24))*c_ess_op + sum(pv_cap[i,0]*p_pv[i,j,t]-power_pv[i,j,t] for i in range(pv_num) for t in range(24))*c_pv_cut for j in range(N))

        count = count + 1

print('PV unit capacity: ', pv_case)
print('ESS energy capacity: ', ene_case)
print('ESS power capacity: ', pei_case)
print(ene2pei)

io.savemat('w_final_train_x2.mat',{'w_final_train_x2':w_final_train_x})
io.savemat('pv_final_x2.mat',{'pv_final_x2':pv_case})
io.savemat('ene_final_x2.mat',{'ene_final_x2':ene_case})
io.savemat('pei_final_x2.mat',{'pei_final_x2':pei_case})
io.savemat('ene2pei_x2.mat',{'ene2pei_x2':ene2pei})
io.savemat('op_cost_x2.mat',{'op_cost_x2':op_cost})
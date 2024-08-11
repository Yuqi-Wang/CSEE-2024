import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import config
import os
import time
import scipy.io as io

w1 = 1/3
w2 = 1/3
RX = 2
result = np.load('result_obj4_w1_'+str(w1)+'_w2_'+str(w2)+'_pv0.1_RX_'+str(RX)+'.npz')

c_ess = 150*1000*100 # the installation cost of ess. the unit is 150 $/100MWh，或者说150$/p.u.，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MWh
c_ess_inv = 150*1000*100 # the installation cost of ess. the unit is 150 $/100MW，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MW

baseMV = 100

ene_cap = result['E_ess'] * baseMV
pei_cap = result['p_ess'] * baseMV
pv_cap = result['pv_cap'] * baseMV

print('ene_cap',ene_cap)
print('pei_cap',pei_cap)
print('pv_cap',pv_cap)

cost_ene = ene_cap.sum() * c_ess
cost_pei = pei_cap.sum() * c_ess_inv
cost_tol = cost_ene + cost_pei
print('cost ene',cost_ene)
print('cost pei',cost_pei)
print('cost tol',cost_tol)

io.savemat('cap and cost_RX_'+str(RX)+'.mat',{'ene_cap':ene_cap,'pei_cap':pei_cap,'cost_ene_pei_tol':[[cost_ene],[cost_pei],[cost_tol]]})
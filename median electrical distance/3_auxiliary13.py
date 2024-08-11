import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import config
import os
import time
import scipy.io as io

w1 = 0.1
w2 = 0.1
result = np.load('result_obj4_w1_'+str(w1)+'_w2_'+str(w2)+'_pv0.1.npz')

baseMV = 100

ene_cap = result['E_ess'] * baseMV
pei_cap = result['p_ess'] * baseMV
pv_cap = result['pv_cap'] * baseMV

print('ene_cap',ene_cap)
print('pei_cap',pei_cap)
print('pv_cap',pv_cap)
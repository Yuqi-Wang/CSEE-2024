import numpy as np
import os
import matplotlib.pyplot as plt
import random
import auxiliary2_fun as a2f
import scipy.io as io
import config

FU_1 = config.get_FU_1()
FU_2 = config.get_FU_2()
FU_3 = config.get_FU_3()

FN_1 = config.get_FN_1()
FN_2 = config.get_FN_2()
FN_3 = config.get_FN_3()

file_path = 'E:\ResearchCode\ThreeObjects\V85\DataFile'
file_list = os.listdir(file_path)

w = [] # all the weight values
for item in file_list:
    w1 = float(item.split('_')[3])
    w2 = float(item.split('_')[5])
    w.append([w1,w2,1-w1-w2])
w = np.array(w)
print(len(w))

w_testing = w
sample_num = 89
w_test = []
f_test = []
for i in range(sample_num):
    testing_index = random.randint(0,len(w_testing)-1)
    w_test.append(w_testing[testing_index,:])
    f1,f2,f3 = a2f.obj_values(w_testing[testing_index,0],w_testing
    [testing_index,1])
    f1 = (f1-FU_1)/(FN_1-FU_1)
    f2 = (f2-FU_2)/(FN_2-FU_2)
    f3 = (f3-FU_3)/(FN_3-FU_3)
    f_test.append([f1,f2,f3])
    w_testing = np.delete(w_testing,testing_index,axis=0)

w_test = np.array(w_test)
f_test = np.array(f_test)

io.savemat('w_sample_weights_89.mat',{'w_sample_weights_89':w_test})
io.savemat('f_sample_weights_89.mat',{'f_sample_weights_89':f_test})
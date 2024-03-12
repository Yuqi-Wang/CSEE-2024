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

F1_1 = (FU_1-FU_1)/(FN_1-FU_1)
F1_2 = (1799982022.306740000000-FU_1)/(FN_1-FU_1)
F1_3 = (1896737759.664720000000-FU_1)/(FN_1-FU_1)

F2_1 = (29685953.880699800000-FU_2)/(FN_2-FU_2)
F2_2 = (FU_2-FU_2)/(FN_2-FU_2)
F2_3 = (37608729.071093800000-FU_2)/(FN_2-FU_2)

F3_1 = (154.061449799987-FU_3)/(FN_3-FU_3)
F3_2 = (153.860469827555-FU_3)/(FN_3-FU_3)
F3_3 = (FU_3-FU_3)/(FN_3-FU_3)

# the training set
##################
##################
##################
train_w = np.load('train_ws_w.npy')
train_sum1_list = []
train_sum2_list = []
train_sum3_list = []
for i in range(len(train_w)):
    temp_w1 = round(train_w[i][0],8)
    temp_w2 = round(train_w[i][1],8)
    sum_1,sum_2,sum_3 = a2f.obj_values(temp_w1,temp_w2)
    sum_1 = (sum_1-FU_1)/(FN_1-FU_1)
    sum_2 = (sum_2-FU_2)/(FN_2-FU_2)
    sum_3 = (sum_3-FU_3)/(FN_3-FU_3)
    train_sum1_list.append(sum_1)
    train_sum2_list.append(sum_2)
    train_sum3_list.append(sum_3)
f_train = np.zeros([len(train_sum1_list),3])
for i in range(len(train_sum1_list)):
    f_train[i,0] = train_sum1_list[i]
    f_train[i,1] = train_sum2_list[i]
    f_train[i,2] = train_sum3_list[i]

w_bound = [
    [1,0,0],
    [0,1,0],
    [0,0,1]
]
w_train = np.vstack((train_w, w_bound))
f_bound = [
    [F1_1,F2_1,F3_1],
    [F1_2,F2_2,F3_2],
    [F1_3,F2_3,F3_3]
]
f_train = np.vstack((f_train, f_bound))
print('Total points before ParetoCheck: '+str(len(f_train)))
np.save('3w_train_ws_woPC_OLTC_'+str(len(f_train))+'.npy', w_train)
np.save('3f_train_ws_woPC_OLTC_'+str(len(f_train))+'.npy', f_train)

# F1
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(w_train[:,0], w_train[:,1], f_train[:,0])
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('F1', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('w2', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('w1', fontdict={'size': 15, 'color': 'red'})
ax.set_title('Points before Pareto check')
fig.savefig('Pareto_ws_woPC_F1_OLTC_'+str(len(f_train)))
plt.show()
# F2
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(w_train[:,0], w_train[:,1], f_train[:,1])
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('F2', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('w2', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('w1', fontdict={'size': 15, 'color': 'red'})
ax.set_title('Points before Pareto check')
fig.savefig('Pareto_ws_woPC_F2_OLTC_'+str(len(f_train)))
plt.show()
# F3
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(w_train[:,0], w_train[:,1], f_train[:,2])
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('F3', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('w2', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('w1', fontdict={'size': 15, 'color': 'red'})
ax.set_title('Points before Pareto check')
fig.savefig('Pareto_ws_woPC_F3_OLTC_'+str(len(f_train)))
plt.show()
# F2(F1,F3)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(f_train[:,0], f_train[:,2], f_train[:,1])
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('F2', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('F3', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('F1', fontdict={'size': 15, 'color': 'red'})
ax.set_title('Points before Pareto check')
fig.savefig('Pareto_ws_woPC_F123_OLTC_'+str(len(f_train)))
plt.show()

# training w
plt.scatter(w_train[:,0],w_train[:,1])
plt.show()

# Pareto selection
f_train, w_train = PC.ParetoCheck(f_train,w_train)
print('Total points after ParetoCheck: '+str(len(f_train)))
np.save('3f_train_ws_wPC_OLTC_'+str(len(f_train))+'.npy', f_train)
np.save('3w_train_ws_wPC_OLTC_'+str(len(f_train))+'.npy', w_train)

# F2(F1,F3)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(f_train[:,0], f_train[:,2], f_train[:,1])
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('F2', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('F3', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('F1', fontdict={'size': 15, 'color': 'red'})
ax.set_title('Points before Pareto check')
fig.savefig('Pareto_ws_wPC_F123_OLTC_'+str(len(f_train)))
plt.show()

# figure for the weights
fig = plt.figure()
plt.scatter(w_train[:,0],w_train[:,1])
plt.title('Pareto weights')
fig.savefig('Pareto_weight_ws_wPC_OLTC_'+str(len(f_train)))
plt.show()


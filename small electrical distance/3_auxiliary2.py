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
diff = config.get_diff()
model = models.load_model('M_GK_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.h5')

FU_1 = config.get_FU_1()
FU_2 = config.get_FU_2()
FU_3 = config.get_FU_3()

FN_1 = config.get_FN_1()
FN_2 = config.get_FN_2()
FN_3 = config.get_FN_3()

exam_x = []
num_sample = 30000
for i in range(num_sample):
    x1 = random.uniform(0,1)
    x2 = random.uniform(0,1-x1)
    x3 = 1-x1-x2
    exam_x.append([x1,x2,x3])
exam_x = np.array(exam_x)
exam_y = model.predict(exam_x)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(exam_y[:,0], exam_y[:,2], exam_y[:,1])
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('F2', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('F3', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('F1', fontdict={'size': 15, 'color': 'red'})
plt.title('Modeled points before Pareto check')
plt.show()
fig.savefig('Modeled_Pareto_woPC_ws_wc_OLTC_'+str(len(exam_y))+str(num_sample))
# 调整成论文中的目标函数编号
# 第一列是碳排放，对应论文中的目标2；第二列是电压偏移，对应论文中的目标3；第三列是经济性，对应论文中的目标1
tune_x = np.array([exam_x[:,0], exam_x[:,2], exam_x[:,1]])
tune_y = np.array([exam_y[:,0], exam_y[:,2], exam_y[:,1]])
tune_y_name = np.array([exam_y[:,0]*(FN_1-FU_1) + FU_1, exam_y[:,2]*(FN_3-FU_3) + FU_3, exam_y[:,1]*(FN_2-FU_2) + FU_2])
io.savemat('weights_woPC_'+str(num_sample)+'_near'+'.mat',{'weights':tune_x})
io.savemat('objectives_woPC_'+str(num_sample)+'_near'+'.mat',{'objectives':tune_y})
io.savemat('objectives_woPC_'+str(num_sample)+'_name_near'+'.mat',{'objectives':tune_y_name})


# # Pareto check
# exam_y, exam_x = PC4.ParetoCheck(exam_y, exam_x)
# print('exam_x length: ', len(exam_x))
# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.scatter3D(exam_y[:,0], exam_y[:,2], exam_y[:,1])
# # 添加坐标轴(顺序是Z, Y, X)
# ax.set_zlabel('F2', fontdict={'size': 15, 'color': 'red'})
# ax.set_ylabel('F3', fontdict={'size': 15, 'color': 'red'})
# ax.set_xlabel('F1', fontdict={'size': 15, 'color': 'red'})
# plt.title('Modeled points after Pareto check')
# plt.show()
# fig.savefig('Modeled_Pareto_wPC_ws_wc_OLTC_'+str(len(exam_y))+str(num_sample))
# # 调整成论文中的目标函数编号
# # 第一列是碳排放，对应论文中的目标2；第二列是电压偏移，对应论文中的目标3；第三列是经济性，对应论文中的目标1
# tune_x = np.array([exam_x[:,0], exam_x[:,2], exam_x[:,1]])
# tune_y = np.array([exam_y[:,0], exam_y[:,2], exam_y[:,1]])
# io.savemat('weights_wPC_'+str(num_sample)+'.mat',{'weights':tune_x})
# io.savemat('objectives_wPC_'+str(num_sample)+'.mat',{'objectives':tune_y})


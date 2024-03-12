from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF,DotProduct, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt
import main_py_fun_GK_OLTC as mfGK
from mpl_toolkits.mplot3d import Axes3D
import joblib
import ParetoCheck3 as PC
import config
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import multiprocessing
import main_py_fun_ws_OLTC as mfws
from keras import models
import random

initial_num = 235
test_prop = 0.2
relax = config.get_relax()
MSE = config.get_MSE()
k = config.get_k()
MAE = config.get_MAE()
diff = config.get_diff()
test_prop = config.get_test_prop()
test_new = config.get_test_new()
num_iterations = config.get_num_iterations()

exam_w = []
num_sample = 10000
for i in range(num_sample):
    w1 = random.uniform(0,1)
    w2 = random.uniform(0,1-w1)
    exam_w.append([w1,w2])
exam_w = np.array(exam_w)



# Objective function 1
########################
########################
########################
f1_w_train = np.load('w1_train_iteration_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
f1_train = np.load('f1_train_iteration_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
# Plot the used training points
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(f1_w_train[:,0], f1_w_train[:,1], f1_train.reshape(len(f1_train),1))
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('F1', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('w2', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('w1', fontdict={'size': 15, 'color': 'red'})
plt.title('V22')
plt.savefig('M1_training_points_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.png')
plt.show()

model1 = models.load_model('M1_GK_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.h5')
exam_f1 = model1.predict(exam_w)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(exam_w[:,0], exam_w[:,1], exam_f1.reshape(len(exam_f1),1))
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('F1', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('w2', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('w1', fontdict={'size': 15, 'color': 'red'})
ax.set_title('Modeled points for objective function 1')
fig.savefig('Modeled_ points_for_objective_function_1_'+str(len(exam_f1)))
plt.show()



# Objective function 2
########################
########################
########################
f2_w_train = np.load('w2_train_iteration_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
f2_train = np.load('f2_train_iteration_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
# Plot the used training points
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(f2_w_train[:,0], f2_w_train[:,1], f2_train.reshape(len(f2_train),1))
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('F2', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('w2', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('w1', fontdict={'size': 15, 'color': 'red'})
plt.title('V22')
plt.savefig('M2_training_points_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.png')
plt.show()

model2 = models.load_model('M2_GK_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.h5')
exam_f2 = model2.predict(exam_w)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(exam_w[:,0], exam_w[:,1], exam_f2.reshape(len(exam_f2),1))
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('F2', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('w2', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('w1', fontdict={'size': 15, 'color': 'red'})
ax.set_title('Modeled points for objective function 2')
fig.savefig('Modeled_ points_for_objective_function_2_'+str(len(exam_f2)))
plt.show()



# Objective function 3
########################
########################
########################
f3_w_train = np.load('w3_train_iteration_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
f3_train = np.load('f3_train_iteration_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
# Plot the used training points
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(f3_w_train[:,0], f3_w_train[:,1], f3_train.reshape(len(f3_train),1))
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('F3', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('w2', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('w1', fontdict={'size': 15, 'color': 'red'})
plt.title('V22')
plt.savefig('M3_training_points_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.png')
plt.show()

model3 = models.load_model('M3_GK_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.h5')
exam_f3 = model3.predict(exam_w)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(exam_w[:,0], exam_w[:,1], exam_f3.reshape(len(exam_f3),1))
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('F3', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('w2', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('w1', fontdict={'size': 15, 'color': 'red'})
ax.set_title('Modeled points for objective function 3')
fig.savefig('Modeled_ points_for_objective_function_3_'+str(len(exam_f3)))
plt.show()



# the final 3-D objective function
########################
########################
########################
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(exam_f1.reshape(len(exam_f3),1), exam_f3.reshape(len(exam_f3),1), exam_f2.reshape(len(exam_f3),1))
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('F2', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('F3', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('F1', fontdict={'size': 15, 'color': 'red'})
ax.set_title('Modeled points for 3-D objective function')
fig.savefig('Modeled points for 3-D objective function '+str(len(exam_f3)))
plt.show()

exam_f1_pc, exam_f3_pc, exam_f2_pc = PC.ParetoCheck(exam_f1,exam_f3,exam_f2)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(exam_f1_pc.reshape(len(exam_f1_pc),1), exam_f3_pc.reshape(len(exam_f3_pc),1), exam_f2_pc.reshape(len(exam_f2_pc),1))
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('F2', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('F3', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('F1', fontdict={'size': 15, 'color': 'red'})
ax.set_title('Modeled points for 3-D objective function')
fig.savefig('Modeled points for 3-D objective function '+str(len(exam_f3_pc)))
plt.show()
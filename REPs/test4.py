from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF,DotProduct, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
import ParetoCheck as PC
import config
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import random
import multiprocessing
import main_py_fun_ws_OLTC as mfws
import tensorflow as tf

# active learning based on uncertainty sampling principle
FU_1 = config.get_FU_1()
FU_2 = config.get_FU_2()
FU_3 = config.get_FU_3()

FN_1 = config.get_FN_1()
FN_2 = config.get_FN_2()
FN_3 = config.get_FN_3()

relax = config.get_relax()
MSE = config.get_MSE()
k = config.get_k()
MAE = config.get_MAE()
diff = config.get_diff()
test_prop = config.get_test_prop()
test_new = config.get_test_new()

initial_num = config.get_initial_num()
# Load the initial training data
f_set = np.load('f_train_ws_woPC_OLTC_'+str(initial_num)+'.npy')
w_set = np.load('w_train_ws_woPC_OLTC_'+str(initial_num)+'.npy') # all the three weights are given in original file


# Split the dataset into train and test sets
w_train, w_test, f_train, f_test = train_test_split(w_set, f_set, test_size=test_prop)

# Define the ANN model for objective functions
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(w_train.shape[1],)))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='linear'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error',metrics=['mae'])

# Train the model
model.fit(w_train, f_train, epochs=100, batch_size=32, verbose=0)
mse_list = []
mae_list = []
diff_list1 = []
diff_list2 = []
diff_list3 = []
# Evaluate the model on test set
mse,mae = model.evaluate(w_test, f_test, verbose=0)
mse_list.append(mse)
mae_list.append(mae)
print('mse is ',mse)
print('mae is ',mae)
mse_list.append(mse)
mae_list.append(mae)
# Evaluate the model on test set
f_trained = model.predict(w_test)
f1_diff = np.max(np.abs((f_trained[:,0] - f_test[:,0]) / f_test[:,0] * 100))
f2_diff = np.max(np.abs((f_trained[:,1] - f_test[:,1]) / f_test[:,1] * 100))
f3_diff = np.max(np.abs((f_trained[:,2] - f_test[:,2]) / f_test[:,2] * 100))
print('The max absolute diff of F1 is',f1_diff)
print('The max absolute diff of F2 is',f2_diff)
print('The max absolute diff of F3 is',f3_diff)
diff_list1.append(f1_diff)
diff_list2.append(f2_diff)
diff_list3.append(f3_diff)
f1_diff_loc = np.argmax(np.abs((f_trained[:,0] - f_test[:,0]) / f_test[:,0] * 100))
f2_diff_loc = np.argmax(np.abs((f_trained[:,1] - f_test[:,1]) / f_test[:,1] * 100))
f3_diff_loc = np.argmax(np.abs((f_trained[:,2] - f_test[:,2]) / f_test[:,2] * 100))

print('f1 diff loc is ',f1_diff_loc)
print('f2 diff loc is ',f2_diff_loc)
print('f3 diff loc is ',f3_diff_loc)



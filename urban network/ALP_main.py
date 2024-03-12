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

if __name__ == '__main__':
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
    diff = config.get_diff_max()
    test_prop = config.get_test_prop()
    test_new = config.get_test_new()
    num_iterations = config.get_num_iterations()
    diff_max = config.get_diff_max()
    initial_num = config.get_initial_num()
    testing_num = config.get_testing_num()
    BatchSize = config.get_BatchSize()
    Epochs = config.get_Epochs()
    LearningRate = config.get_LearningRate()

    # Load the initial training data
    f_train = np.load('3f_Bay_train_ws_woPC_OLTC_'+'.npy')
    w_train = np.load('3w_Bay_train_ws_woPC_OLTC_'+'.npy') # all the three weights are given in original file
    # Load the testing data
    f_testing = np.load('3f_Bay_test_ws_woPC_OLTC_'+'.npy')
    w_testing = np.load('3w_Bay_test_ws_woPC_OLTC_'+'.npy')

    w_test_tol = []
    w_test = []
    f_test = []
    for i in range(test_new):
        testing_index = random.randint(0,len(w_testing)-1)
        w_test.append(w_testing[testing_index,:])
        f_test.append(f_testing[testing_index,:])
        w_testing = np.delete(w_testing,testing_index,axis=0)
        f_testing = np.delete(f_testing,testing_index,axis=0)
    w_test = np.array(w_test)
    f_test = np.array(f_test)
    w_test_tol.append(w_test)


    # Define the ANN model for objective functions
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(127, activation='relu', input_shape=(w_train.shape[1],)))
    model.add(tf.keras.layers.Dense(482, activation='relu'))
    model.add(tf.keras.layers.Dense(33, activation='relu'))
    model.add(tf.keras.layers.Dense(81, activation='relu'))
    model.add(tf.keras.layers.Dense(146, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LearningRate), loss='mean_squared_error',metrics=['mae'])

    # Train the model
    model.fit(w_train, f_train, epochs=Epochs, batch_size=BatchSize, verbose=0)
    mse_list = []
    mae_list = []
    diff_list1 = [100]
    diff_list2 = [100]
    diff_list3 = [100]
    # Evaluate the model on test set
    mse,mae = model.evaluate(w_test, f_test, verbose=0)
    print('mse is ',mse)
    print('mae is ',mae)
    mse_list.append(mse)
    mae_list.append(mae)
    # Evaluate the model on test set
    f_trained = model.predict(w_test)
    f1_diff = np.max(np.abs(f_trained[:,0] - f_test[:,0]))
    f2_diff = np.max(np.abs(f_trained[:,1] - f_test[:,1]))
    f3_diff = np.max(np.abs(f_trained[:,2] - f_test[:,2]))
    print('The max absolute diff of F1 is',f1_diff)
    print('The max absolute diff of F2 is',f2_diff)
    print('The max absolute diff of F3 is',f3_diff)
    diff_list1.append(f1_diff)
    diff_list2.append(f2_diff)
    diff_list3.append(f3_diff)
    f1_diff_loc = np.argmax(np.abs(f_trained[:,0] - f_test[:,0]))
    f2_diff_loc = np.argmax(np.abs(f_trained[:,1] - f_test[:,1]))
    f3_diff_loc = np.argmax(np.abs(f_trained[:,2] - f_test[:,2]))

    # number of parallel cores
    para_num = 3
    if k > para_num:
        point_train = para_num
    else:
        point_train = 3

    if test_new > para_num:
        point_test = para_num
    else:
        point_test = test_new
    
    ite = 0
    # for i in range(num_iterations):
    while (abs(f1_diff-diff_list1[-2]) > diff_max) or (abs(f2_diff-diff_list2[-2]) > diff_max) or (abs(f3_diff-diff_list3[-2]) > diff_max) or (f1_diff > diff_max) or (f2_diff > diff_max) or (f3_diff > diff_max):
    # while ite < num_iterations:
        print('This is the '+str(ite)+'th iteration.')
        # Objective function 1
        ########################
        ########################
        ########################   
        # Get the next training points of objective function 1
        # Firstly, tell where F1 needs more training points
        if (abs(f1_diff-diff_list1[-2]) > diff_max) or (f1_diff > diff_max):
            radius = np.min(np.linalg.norm(w_test[f1_diff_loc,:]-w_train)) # the minimum circle radius around f1_diff_loc, in other words, obtain the minimum circle
            f1_axis_ws = []
            for i in range(k):
                # the i is useless here
                w1_val = random.uniform(np.max([w_test[f1_diff_loc,0]-radius,0]),np.min([w_test[f1_diff_loc,0]+radius,1-w_test[f1_diff_loc,1]]))
                delta_w1 = np.abs(w1_val-w_test[f1_diff_loc,0])
                radius_2 = np.sqrt(radius**2-delta_w1**2)
                w2_val = random.uniform(np.max([w_test[f1_diff_loc,1]-radius_2,0]),np.min([w_test[f1_diff_loc,1]+radius_2,1-w1_val]))
                f1_axis_ws.append([w1_val,w2_val,1-w1_val-w2_val])
            
            p1 = multiprocessing.Pool(point_train)
            f1_result_list = []
            # log result
            def f1_log_result(result):
                f1_result_list.append(result)
                print('diff list 1: ', diff_list1)
                print('mae list is: ', mae_list)
                print('mse list is: ', mse_list)
                return f1_result_list
            
            for i in range(k):
                print('This is for new training points of objective function 1.')
                p1.apply_async(mfws.main_fun, (4,round(f1_axis_ws[i][0],8),round(f1_axis_ws[i][1],8),i,k,ite), callback=f1_log_result)
            p1.close()
            p1.join()

            for i in range(len(f1_result_list)):
                if f1_result_list[i][0] == -1:
                    print('This sample is infeasible!')
                else:
                    w_train = np.vstack((w_train, np.array(f1_axis_ws[i][0:3])))
                    f_train = np.vstack((f_train, np.array(f1_result_list[i][0:3])))
        
        # Objective function 2
        ########################
        ########################
        ########################   
        # Get the next training points of objective function 2
        # Firstly, tell where F2 needs more training points
        if (abs(f2_diff-diff_list2[-2]) > diff_max) or (f2_diff > diff_max):
            radius = np.min(np.linalg.norm(w_test[f2_diff_loc,:]-w_train)) # the minimum circle radius around f1_diff_loc, in other words, obtain the minimum circle
            f2_axis_ws = []
            for i in range(k):
                # the i is useless here
                w1_val = random.uniform(np.max([w_test[f2_diff_loc,0]-radius,0]),np.min([w_test[f2_diff_loc,0]+radius,1-w_test[f2_diff_loc,1]]))
                delta_w1 = np.abs(w1_val-w_test[f2_diff_loc,0])
                radius_2 = np.sqrt(radius**2-delta_w1**2)
                w2_val = random.uniform(np.max([w_test[f2_diff_loc,1]-radius_2,0]),np.min([w_test[f2_diff_loc,1]+radius_2,1-w1_val]))
                f2_axis_ws.append([w1_val,w2_val,1-w1_val-w2_val])

            p1 = multiprocessing.Pool(point_train)
            f2_result_list = []
            # log result
            def f2_log_result(result):
                f2_result_list.append(result)
                print('diff list 2: ', diff_list2)
                print('mae list is: ', mae_list)
                print('mse list is: ', mse_list)
                return f2_result_list
            
            for i in range(k):
                print('This is for new training points of objective function 2.')
                p1.apply_async(mfws.main_fun, (4,round(f2_axis_ws[i][0],8),round(f2_axis_ws[i][1],8),i,k,ite), callback=f2_log_result)
            p1.close()
            p1.join()

            for i in range(len(f2_result_list)):
                if f2_result_list[i][0] == -1:
                    print('This sample is infeasible!')
                else:
                    w_train = np.vstack((w_train, np.array(f2_axis_ws[i][0:3])))
                    f_train = np.vstack((f_train, np.array(f2_result_list[i][0:3])))                

        # Objective function 3 
        ########################
        ########################
        ########################   
        # Get the next training points of objective function 3
        # Firstly, tell where F3 needs more training points
        if (abs(f3_diff-diff_list3[-2]) > diff_max) or (f3_diff > diff_max):
            radius = np.min(np.linalg.norm(w_test[f3_diff_loc,:]-w_train)) # the minimum circle radius around f1_diff_loc, in other words, obtain the minimum circle
            f3_axis_ws = []
            for i in range(k):
                # the i is useless here
                w1_val = random.uniform(np.max([w_test[f3_diff_loc,0]-radius,0]),np.min([w_test[f3_diff_loc,0]+radius,1-w_test[f3_diff_loc,1]]))
                delta_w1 = np.abs(w1_val-w_test[f3_diff_loc,0])
                radius_2 = np.sqrt(radius**2-delta_w1**2)
                w2_val = random.uniform(np.max([w_test[f3_diff_loc,1]-radius_2,0]),np.min([w_test[f3_diff_loc,1]+radius_2,1-w1_val]))
                f3_axis_ws.append([w1_val,w2_val,1-w1_val-w2_val])

            p1 = multiprocessing.Pool(point_train)
            f3_result_list = []
            # log result
            def f3_log_result(result):
                f3_result_list.append(result)
                print('diff list 3: ', diff_list3)
                print('mae list is: ', mae_list)
                print('mse list is: ', mse_list)
                return f3_result_list
            
            for i in range(k):
                print('This is for new training points of objective function 3.')
                p1.apply_async(mfws.main_fun, (4,round(f3_axis_ws[i][0],8),round(f3_axis_ws[i][1],8),i,k,ite), callback=f3_log_result)
            p1.close()
            p1.join()

            for i in range(len(f3_result_list)):
                if f3_result_list[i][0] == -1:
                    print('This sample is infeasible!')
                else:
                    w_train = np.vstack((w_train, np.array(f3_axis_ws[i][0:3])))
                    f_train = np.vstack((f_train, np.array(f3_result_list[i][0:3])))                

        # Get the next testing points
        ##############################
        ##############################
        ##############################
        w_test = []
        f_test = []
        for i in range(test_new):
            testing_index = random.randint(0,len(w_testing)-1)
            w_test.append(w_testing[testing_index,:])
            f_test.append(f_testing[testing_index,:])
            w_testing = np.delete(w_testing,testing_index,axis=0)
            f_testing = np.delete(f_testing,testing_index,axis=0)
        w_test = np.array(w_test)
        f_test = np.array(f_test)

        
        # Train the model
        ####################
        ####################
        ####################
        model.fit(w_train, f_train, epochs=Epochs, batch_size=BatchSize, verbose=0)

        # Evaluate the model on test set
        mse,mae = model.evaluate(w_test, f_test, verbose=0)
        mse_list.append(mse)
        mae_list.append(mae)

        # Calculate the maximum absolute difference
        f_trained = model.predict(w_test)
        f1_diff = np.max(np.abs(f_trained[:,0] - f_test[:,0]))
        f2_diff = np.max(np.abs(f_trained[:,1] - f_test[:,1]))
        f3_diff = np.max(np.abs(f_trained[:,2] - f_test[:,2]))
        diff_list1.append(f1_diff)
        diff_list2.append(f2_diff)
        diff_list3.append(f3_diff)
        print('The max absolute diff list of F1 is',f1_diff)
        print('The max absolute diff list of F2 is',f2_diff)
        print('The max absolute diff list of F3 is',f3_diff)
        f1_diff_loc = np.argmax(np.abs(f_trained[:,0] - f_test[:,0]))
        f2_diff_loc = np.argmax(np.abs(f_trained[:,1] - f_test[:,1]))
        f3_diff_loc = np.argmax(np.abs(f_trained[:,2] - f_test[:,2]))

        ite = ite+1

        print('The number of training points is '+str(len(f_train)))

        np.save('w_train_iteration_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy',w_train)
        np.save('f_train_iteration_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy',f_train)
        np.save('w_test_tol_train_iteration_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy',w_test_tol)

        # Save the model
        model.save('M_GK_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.h5')
        np.save('mse_list_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy',np.array(mse_list))
        np.save('mae_list_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy',np.array(mae_list))
        
        np.save('diff_list1_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy',np.array(diff_list1))
        np.save('diff_list2_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy',np.array(diff_list2))
        np.save('diff_list3_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy',np.array(diff_list3))

    print('Iteration time:', ite)

    # Plot the used training points in F1 view
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(w_train[:,0], w_train[:,1], f_train[:,0])
    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('F1', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('w2', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('w1', fontdict={'size': 15, 'color': 'red'})
    plt.title('V22')
    plt.savefig('M_training_points_F1_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.png')
    plt.show()

    # Plot the used training points in F2 view
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(w_train[:,0], w_train[:,1], f_train[:,1])
    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('F2', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('w2', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('w1', fontdict={'size': 15, 'color': 'red'})
    plt.title('V22')
    plt.savefig('M_training_points_F2_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.png')
    plt.show()

    # Plot the used training points in F3 view
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(w_train[:,0], w_train[:,1], f_train[:,2])
    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('F3', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('w2', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('w1', fontdict={'size': 15, 'color': 'red'})
    plt.title('V22')
    plt.savefig('M_training_points_F3_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.png')
    plt.show()
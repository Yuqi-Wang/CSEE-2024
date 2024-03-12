import multiprocessing
import time
import numpy as np
import random
import config
import easygui
import main_py_fun_ws_OLTC as mfws_OLTC
import scipy.io as io


def main_1():

    w = []
    num_sample = 800
    batch = 1
    for i in range(num_sample):
        w1 = random.uniform(0,1)
        w2 = random.uniform(0,1-w1)
        w.append([w1,w2])
    np.save('random_ws_wc_OLTC_'+str(num_sample)+'_'+str(batch)+'.npy',w)

    print(len(w))
    if len(w) > 3:
        point_num = 3
    else:
        point_num = len(w)

    p1 = multiprocessing.Pool(point_num)

    for i in range(len(w)):
        p1.apply_async(mfws_OLTC.main_fun, (4,round(w[i][0],8),round(w[i][1],8),i,len(w),i))

    p1.close()
    p1.join()

    easygui.msgbox('V45_Python succeeded!')

def main_2():

    w = io.loadmat('w_case.mat')['w_case']

    print(len(w))
    if len(w) > 4:
        point_num = 4
    else:
        point_num = len(w)

    p1 = multiprocessing.Pool(point_num)

    for i in range(len(w)):
        p1.apply_async(mfws_OLTC.main_fun, (4,round(w[i,1],8),round(w[i,0],8),i,len(w),i))

    p1.close()
    p1.join()

    easygui.msgbox('V45_Python succeeded!')

if __name__ == '__main__':

    main_1()

    
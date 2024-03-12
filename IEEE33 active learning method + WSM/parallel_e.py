import multiprocessing
import time
import main_py_fun_e_OLTC as mfe_OLTC
import numpy as np
import random
import config
import easygui
import main_py_fun_ws_OLTC as mfws_OLTC

def main_2():
    # this function is used for e-constraint method
    w = []

    N_random = 100
    for i in range(N_random):
        w1 = random.uniform(0,1)
        w2 = random.uniform(0,1)
        w.append([w1,w2])
    np.save('random_e_OLTC_'+str(N_random)+'.npy',w)

    if len(w) > 3:
        point_num = 3
    else:
        point_num = len(w)

    p1 = multiprocessing.Pool(point_num)

    # 计算中间点
    for i in range(len(w)):
        p1.apply_async(mfe_OLTC.main_fun, (2,round(w[i][0],8),round(w[i][1],8),i))

    p1.close()
    p1.join()

    easygui.msgbox('V45_Python succeeded!')

if __name__ == '__main__':
    # main_1()

    # this function is used for e-constraint method
    main_2()


    
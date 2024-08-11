import multiprocessing
import time
# import main_py_fun_e_OLTC as mfe_OLTC
import numpy as np
import random
import config
import easygui
import main_py_fun_ws_OLTC as mfws_OLTC


# s

# def main_2():
#     # this function is used for e-constraint method
#     start = time.process_time()
#     w = []
#     # for i in range(1,18):
#     #     for j in range(1,20-i-1):
#     #         [w1, w2] = [round(i*0.05,1), round(j*0.05,1)]
#     #         w.append([w1,w2])
#     N_random = 3000
#     for i in range(N_random):
#         w1 = random.uniform(0,1)
#         w2 = random.uniform(0,1)
#         w.append([w1,w2])
#     np.save('random_e_wc_OLTC_'+str(N_random)+'.npy',w)

#     print(len(w))
#     if len(w) > 5:
#         point_num = 5
#     else:
#         point_num = len(w)

#     p1 = multiprocessing.Pool(point_num)

#     # 计算中间点
#     for i in range(len(w)):
#         p1.apply_async(mfe_OLTC.main_fun, (2,round(w[i][0],8),round(w[i][1],8)))
    
#     # results = [p1.apply_async(mfe_OLTC.main_fun, (2,round(ww[0],8),round(ww[1],8))) for ww in w]
#     # print(results)


#     p1.close()
#     p1.join()

#     # func_value = [result.get() for result in results]

#     end = time.process_time()
#     print('The total time is:',end-start)

#     easygui.msgbox('V45_Python succeeded!')

def main_3():
    # this function is used for weighted-sum method
    # start = time.process_time()
    w = []
    # for i in range(1,18):
    #     for j in range(1,20-i-1):
    #         [w1, w2] = [round(i*0.05,1), round(j*0.05,1)]
    #         w.append([w1,w2])
    num_sample = 100
    batch = 3
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

    # 计算中间点
    for i in range(len(w)):
        p1.apply_async(mfws_OLTC.main_fun, (4,round(w[i][0],8),round(w[i][1],8),i,len(w),i))


    p1.close()
    p1.join()

    # func_value = [result.get() for result in results]

    # end = time.process_time()
    # print('The total time is:',end-start)

    # for i in range(len(res)):
    #     res[i] = res[i].get()


    easygui.msgbox('V45_Python succeeded!')

if __name__ == '__main__':
    # main_1()

    # # this function is used for e-constraint method
    # main_2()

    # this function is used for weighted-sum method
    main_3()

    
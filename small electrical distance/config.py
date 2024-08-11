class global_var:
    # 需要定义的全局变量
    beta = 0.5
    ur = 0.05

    FU_1 = -95740016.757514700000 
    FU_2 = -9217428.793990370000 
    FU_3 = 0.299859184810 
	
    FN_1 = 382189345.774046000000 
    FN_2 = 16078795.267308400000 
    FN_3 = 136.367463382228 

    co2_pro = 1

    N_random = 3000
    num_iterations = 10
    relax = 0.001
    max_sigma = 0.01
    MSE = 0.0005
    k = 4
    MAE = 0.001
    diff = 0.03
    test_prop = 0.2
    test_new = 5
    initial_num = 373
    diff_max = diff
    testing_num = 485

    fail_line = 0.0009
    fail_pv = 0.0009
    fail_ess = 0.0009
    fail_ene = 0.1

    BatchSize = 300
    Epochs = 50
    LearningRate = 0.005




def set_beta(beta):
    global_var.beta = beta
def get_beta():
    return global_var.beta

def set_ur(ur):
    global_var.ur = ur
def get_ur():
    return global_var.ur

def set_FU_1(FU_1):
    global_var.FU_1 = FU_1
def get_FU_1():
    return global_var.FU_1

def set_FU_2(FU_2):
    global_var.FU_2 = FU_2
def get_FU_2():
    return global_var.FU_2

def set_FU_3(FU_3):
    global_var.FU_3 = FU_3
def get_FU_3():
    return global_var.FU_3

def set_FN_1(FN_1):
    global_var.FN_1 = FN_1
def get_FN_1():
    return global_var.FN_1

def set_FN_2(FN_2):
    global_var.FN_2 = FN_2
def get_FN_2():
    return global_var.FN_2

def set_FN_3(FN_3):
    global_var.FN_3 = FN_3
def get_FN_3():
    return global_var.FN_3

def set_N_random(N_random):
    global_var.N_random = N_random
def get_N_random():
    return global_var.N_random

def set_num_iterations(num_iterations):
    global_var.num_iterations = num_iterations
def get_num_iterations():
    return global_var.num_iterations

def set_relax(relax):
    global_var.relax = relax
def get_relax():
    return global_var.relax

def set_max_sigma(max_sigma):
    global_var.max_sigma = max_sigma
def get_max_sigma():
    return global_var.max_sigma

def set_MSE(MSE):
    global_var.MSE = MSE
def get_MSE():
    return global_var.MSE

def set_k(k):
    global_var.k = k
def get_k():
    return global_var.k

def set_MAE(MAE):
    global_var.MAE = MAE
def get_MAE():
    return global_var.MAE

def set_diff(diff):
    global_var.diff = diff
def get_diff():
    return global_var.diff

def set_test_prop(test_prop):
    global_var.test_prop = test_prop
def get_test_prop():
    return global_var.test_prop

def set_test_new(test_new):
    global_var.test_new = test_new
def get_test_new():
    return global_var.test_new

def set_fail_line(fail_line):
    global_var.fail_line = fail_line
def get_fail_line():
    return global_var.fail_line

def set_fail_ess(fail_ess):
    global_var.fail_ess = fail_ess
def get_fail_ess():
    return global_var.fail_ess

def set_fail_pv(fail_pv):
    global_var.fail_pv = fail_pv
def get_fail_pv():
    return global_var.fail_pv

def set_fail_ene(fail_ene):
    global_var.fail_ene = fail_ene
def get_fail_ene():
    return global_var.fail_ene

def set_initial_num(initial_num):
    global_var.initial_num = initial_num
def get_initial_num():
    return global_var.initial_num

def set_diff_max(diff_max):
    global_var.diff_max = diff_max
def get_diff_max():
    return global_var.diff_max

def get_testing_num():
    return global_var.testing_num

def get_LearningRate():
    return global_var.LearningRate

def get_Epochs():
    return global_var.Epochs

def get_BatchSize():
    return global_var.BatchSize

def get_co2_pro():
    return global_var.co2_pro
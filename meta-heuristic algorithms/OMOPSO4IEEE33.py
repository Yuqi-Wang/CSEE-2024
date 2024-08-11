from platypus import OMOPSO, Problem, Real, Binary, CompoundOperator, SBX, HUX, PM, BitFlip
import numpy as np
import config
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class IEEE33(Problem):
    
    def __init__(self):
        global upper_limit
        beta = config.get_beta()

        pv_bus = [9,22,29]
        pv_num = len(pv_bus)
        ess_bus = [4,30]
        ess_num = len(ess_bus)

        end_bus = [17,21,24,32]

        PF = 0.9
        night_p = 1 # PV在夜间输出无功的上限占VSC容量的比值
        interest = 0.1 # the annual interest rate
        life = 20 # the life span of PV and ESS
        year = 365 # the days in one year
        yita = 0.9 # the risk factor 
        v_norm = 1 # the normal voltage on each bus
        Busnum = 33 # the bus numbers
        N = 5 # 随机场景的数量
        Linenum = 32 # IEEE33节点算例线路条数
        P_pv_max = 10/100 # 0.1表示光伏的容量，单位是MW，除以100是用100MW对他进行标幺化
        ESS_cap_max = 5/100 # 储能的最大电能容量，1表示储能的电能容量，单位是MW·h，除以100是用100MW对他进行标幺化
        inv_max = 5/100 # 0.01表示储能的充放电功率，单位是MW，除以100是用100MW对他进行标幺化
        fai_max = 0.9 # the maximum SOC of ESS
        fai_min = 0.1 # the minimum SOC of ESS
        lamda_ch = 0.9 # the efficiency of charging
        lamda_dch = 0.9 # the efficiency of discharging
        S_sub = 30/100 # the capacity of substaion and the unit is MVA，10的单位是MVA，除以100是进行标幺化
        basekV = 10
        Vmin = (0.9)**2 # the minimum bus voltage, and the unit is (p.u.)
        Vmax = (1.1)**2 # the maximum bus voltage, and the unit is (p.u.)
        b_con_son = scipy.io.loadmat('b_con_son.mat')['b_con_son'] # the uptril matrix and the i th line contains the son buses of bus i 
        b_con_fa = scipy.io.loadmat('b_con_fa.mat')['b_con_fa'] # the downtril matrix and the i th line contains the father buses of bus i
        z_pu = (basekV)**2/100 # 阻抗的基准值，单位是Ω
        line_r = scipy.io.loadmat('line_r.mat')['line_r']/z_pu # the i th element is the R in the line where bus i is its son bus, 读取出的单位是Ω，除以z_pu化成标幺值
        line_x = scipy.io.loadmat('line_x.mat')['line_x']/z_pu # the i th element is the X in the line where bus i is its son bus, 读取出的单位是Ω，除以z_pu化成标幺值
        i_fa = scipy.io.loadmat('i_fa.mat')['i_fa'] # the father bus of bus i
        c_e_b = scipy.io.loadmat('c_e_t.mat')['c_e_t']*1000*100 # the electricity price for buy at each hour and the unit is $/kWh，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MWh
        c_e_b = c_e_b.reshape((1,1,24))
        c_e_s = np.zeros((1,1,24)) # the electricity price for sell
        for i in range(24):
            c_e_s[0,0,i] = c_e_b[0,0,i] * 0.7

        c_ess_op = 0.00027*1000*100 # the operation cost of ess unit is $/kWh，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MWh
        c_pv_cut = 0.033*1000*100 # the punishment on pv deserting, and the unit is $/kWh，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MWh
        # co2 = 0.779*1000*100 # the co2 emission, and the unit is kg/kWh，乘以1000是将单位换算为kg/MWh，再乘以100是将单位换算为kg/100MWh
        # the co2 emission, and the unit is kg/kWh，乘以1000是将单位换算为kg/MWh，再乘以100是将单位换算为kg/100MWh
        co2 = [0.531914894,	0.717391304,0.844444444,0.837209302,0.833333333,0.86746988,	0.814814815,0.853658537,0.880952381,0.256410256,0.207692308,0.18,0.244444444,0.313953488,0.298850575,0.239583333,0.220883534,0.240480962,0.325581395,0.311111111,0.208333333,0.227743271,0.346938776,0.458333333]
        for i in range(24):
            co2[i] = co2[i] * 1000 * 100

        c_pv = 1343*1000*100 # the installation cost of pv on each bus and it is 1343 $/kW，乘以1000是将单位换算为$/MW，再乘以100是将单位换算为$/100MW
        c_ess = 150*1000*100 # the installation cost of ess. the unit is 150 $/100MWh，或者说150$/p.u.，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MWh
        c_ess_inv = 150*1000*100 # the installation cost of ess. the unit is 150 $/100MW，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MW
        i_pu = 100/basekV # 电流的基准值，单位是kA
        i_max = (20/basekV/i_pu)**2 # the maximum I in a line （分子上），所以分母上也得是电流基准值的平方

        p_load = scipy.io.loadmat('p_load.mat')['p_load'][:,0:N,:] # 标幺值 and represents the fluctuation
        q_load = scipy.io.loadmat('q_load.mat')['q_load'][:,0:N,:] # 标幺值 and represents the fluctuation
        p_pv = scipy.io.loadmat('p_pv.mat')['p_pv'][0:pv_num,0:N,:] # the value is in p.u. value and represents the fluctuation, the power factor of PV is 0.9，滞后
        # q_pv = scipy.io.loadmat('q_pv.mat')['q_pv'][:,0:N,:] # the value is in p.u. value and represents the fluctuation, the power factor of PV is 0.9，滞后，PV的无功输出现在是变量了

        # q_pv = q_pv * P_pv # transform into 标幺值,pv的无功功率也是注入

        zero = np.zeros((1,1,1))

        # the tap for the OLTC
        per_num = 3
        per_value = 0.1/per_num
        tap_opt = np.linspace(1-per_value*per_num, 1+per_value*per_num, 2*per_num+1)
        tap_posi = np.zeros((len(tap_opt),N,24))
        for i in range(len(tap_opt)):
            tap_posi[i,:,:] = tap_opt[i]

        # the big M
        M = 1000000

        # life cycle carbon emission
        # life_PV = 1.63 / 0.26 * (185+20) / 17 * 3 # kg/(台*年), 185是production，20是recycling
        life_PV = 1000000/230*180 * 100 # kg/(100 MW), 先用1MW除以230获得对应的面积，再乘以180（每平方米生产制造时的碳排放），最后乘以100是换算成kg/(100 MW)
        life_ESS = 50 * 100 # kg/(100 MWh)
        life_ESS_p = 50 * 100 # kg/(100 MWh)

        hour_max = 5
        hour_min = 1
        co2_ess_op = 0.36*600*100 # 自己换算的，0.36是储能运行1MWh对应的支撑电量，乘以600是均值碳排放强度，乘以100是换算成系统的标幺功率

        fail_line = config.get_fail_line() # 如果是0.01，则含义是每100小时中有1小时故障
        fail_pv = config.get_fail_pv()
        fail_ess = config.get_fail_ess()
        fail_ene = config.get_fail_ene()

        print('1')
        # variable number
        time_seg = 24
        self.n_omig_ess_ch  = int(ess_num * N * time_seg)
        self.n_omig_ess_dch = int(self.n_omig_ess_ch * 2)

        self.n_p_sub_b = self.n_omig_ess_dch + int(1*N*time_seg)
        self.n_q_sub = self.n_p_sub_b + int(1*N*time_seg)
        self.n_p_ch = self.n_q_sub + int(ess_num*N*time_seg)
        self.n_p_dch = self.n_p_ch + int(ess_num*N*time_seg)
        self.n_u_bus = self.n_p_dch + int(Busnum*N*time_seg)
        self.n_q_pv = self.n_u_bus + int(pv_num*N*time_seg)

        self.n_p_line = self.n_q_pv + int(Linenum*N*time_seg)
        self.n_q_line = self.n_p_line + int(Linenum*N*time_seg)
        self.n_i_line = self.n_q_line + int(Linenum*N*time_seg)
        self.n_ess_w = self.n_i_line + int(ess_num)
        self.n_v_1 = self.n_ess_w + int(Busnum*N*time_seg)
        self.n_v_2 = self.n_v_1 + int(Busnum*N*time_seg)
        self.n_tap_k = self.n_v_2 + int(len(tap_opt)*N*time_seg)

        self.n_E_ess = self.n_tap_k + int(ess_num)
        self.n_p_ess = self.n_E_ess + int(ess_num)
        self.n_ch_ess = self.n_p_ess + int(ess_num*N*time_seg)
        self.n_dis_ess = self.n_ch_ess + int(ess_num*N*time_seg)

        self.n_pv_cap = self.n_dis_ess + int(pv_num)
        self.n_power_pv = self.n_pv_cap + int(pv_num*N*time_seg)
        self.n_q_ess = self.n_power_pv + int(ess_num*N*time_seg)
        self.n_abs_p_line = self.n_q_ess + int(Linenum*N*time_seg)

        self.n_var_total = self.n_abs_p_line

        # constraint number
        c1 = int(ess_num*N*time_seg)
        c2 = c1 + int(ess_num*N*time_seg)
        c3 = c2 + int(ess_num*N*time_seg)
        c4 = c3 + int(ess_num*N*time_seg)
        c5 = c4 + int(ess_num*N*time_seg)
        c6 = c5 + int(ess_num*N*time_seg*2)
        c7 = c6 + int(ess_num*N*time_seg*2)
        c8 = c7 + int(ess_num*N*time_seg*2)
        c9 = c8 + int(ess_num*N*time_seg*2)
        c10 = c9 + int(pv_num*N*14)
        c11 = c10 + int(pv_num*N*(6+4))
        c12 = c11 + int(pv_num*N*(6+4))
        
        c13 = c12 + int(ess_num*N*23)
        c14 = c13 + int(ess_num*N*23)
        c15 = c14 + int(ess_num*N)

        c16 = c15 + int(N*time_seg)
        c17 = c16 + int(N*time_seg*2)
        c18 = c17 + int((Busnum-1)*N*time_seg*2)
        c19 = c18 + int((Busnum-1)*N*time_seg)
        c20 = c19 + int((Busnum-1)*N*time_seg)
        c21 = c20 + int((Busnum-1)*N*time_seg)
        c22 = c21 + int(len(tap_opt)*N*time_seg)
        c23 = c22 + int(len(tap_opt)*N*time_seg)
        c24 = c23 + int(N*time_seg)

        c25 = c24 + int((Busnum-1)*N*time_seg)
        c26 = c25 + int(Linenum*N*time_seg)
        c27 = c26 + int(Linenum*N*time_seg)

        c28 = c27 + int(Busnum*N*time_seg)
        c29 = c28 + int(Busnum*N*time_seg*2)
        c30 = c29 + int(ess_num*2)
        c31 = c30 + int(ess_num*2)
        c32 = c31 + int(pv_num + pv_num*N*time_seg)
        c33 = c32 + int(pv_num + pv_num*N*time_seg)
        c34 = c33 + int(ess_num)
        c35 = c34 + int(ess_num)

        c36 = c35 + int(ess_num*N*time_seg*2)
        c37 = c36 + int(1)
        c38 = c37 + int(Linenum*N*time_seg*2)

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5
        self.c6 = c6
        self.c7 = c7
        self.c8 = c8
        self.c9 = c9
        self.c10 = c10
        self.c11 = c11
        self.c12 = c12
        self.c13 = c13
        self.c14 = c14
        self.c15 = c15
        self.c16 = c16
        self.c17 = c17
        self.c18 = c18
        self.c19 = c19
        self.c20 = c20
        self.c21 = c21
        self.c22 = c22
        self.c23 = c23
        self.c24 = c24
        self.c25 = c25
        self.c26 = c26
        self.c27 = c27
        self.c28 = c28
        self.c29 = c29
        self.c30 = c30
        self.c31 = c31
        self.c32 = c32
        self.c33 = c33
        self.c34 = c34
        self.c35 = c35
        self.c36 = c36
        self.c37 = c37
        self.c38 = c38

        n_const_total = c38

        super().__init__(self.n_var_total, 3, n_const_total)


        self.types[0:self.n_omig_ess_ch] = Binary(1)
        self.types[self.n_omig_ess_ch:self.n_omig_ess_dch] = Binary(1)
        self.types[self.n_omig_ess_dch:self.n_p_sub_b] = Real(-upper_limit,upper_limit)
        self.types[self.n_p_sub_b:self.n_q_sub] = Real(-upper_limit,upper_limit)
        self.types[self.n_q_sub:self.n_p_ch] = Real(0,upper_limit)
        self.types[self.n_p_ch:self.n_p_dch] = Real(0,upper_limit)
        self.types[self.n_p_dch:self.n_u_bus] = Real(Vmin,Vmax) # 
        self.types[self.n_u_bus:self.n_q_pv] = Real(-upper_limit,upper_limit)
        self.types[self.n_q_pv:self.n_p_line] = Real(-upper_limit,upper_limit)
        self.types[self.n_p_line:self.n_q_line] = Real(-upper_limit,upper_limit)
        self.types[self.n_q_line:self.n_i_line] = Real(0,i_max) # 
        self.types[self.n_i_line:self.n_ess_w] = Binary(1)
        self.types[self.n_ess_w:self.n_v_1] = Real(0,upper_limit)
        self.types[self.n_v_1:self.n_v_2] = Real(0,upper_limit)
        self.types[self.n_v_2:self.n_tap_k] = Binary(1)
        self.types[self.n_tap_k:self.n_E_ess] = Real(0,ESS_cap_max) # 
        self.types[self.n_E_ess:self.n_p_ess] = Real(0,inv_max) # 
        self.types[self.n_p_ess:self.n_ch_ess] = Real(0,upper_limit)
        self.types[self.n_ch_ess:self.n_dis_ess] = Real(0,upper_limit)
        self.types[self.n_dis_ess:self.n_pv_cap] = Real(0,P_pv_max) # 
        self.types[self.n_pv_cap:self.n_power_pv] = Real(0,upper_limit)
        self.types[self.n_power_pv:self.n_q_ess] = Real(-upper_limit,upper_limit)
        self.types[self.n_q_ess:self.n_abs_p_line] = Real(-upper_limit,upper_limit)

        self.constraints[0:c1] = "<=0"
        self.constraints[c1:c2] = ">=0"
        self.constraints[c2:c3] = "<=0"
        self.constraints[c3:c4] = ">=0"
        self.constraints[c4:c5] = "<=0"
        self.constraints[c5:c6] = "<=0"
        self.constraints[c6:c7] = "<=0"
        self.constraints[c7:c8] = ">=0"
        self.constraints[c8:c9] = ">=0"
        self.constraints[c9:c10] = "<=0"
        self.constraints[c10:c11] = ">=0"
        self.constraints[c11:c12] = "<=0"
        self.constraints[c12:c13] = ">=0"
        self.constraints[c13:c14] = "<=0"
        self.constraints[c14:c15] = "==0"
        self.constraints[c15:c16] = "<=0"
        self.constraints[c16:c17] = "==0"
        self.constraints[c17:c18] = "==0"
        self.constraints[c18:c19] = "==0"
        self.constraints[c19:c20] = ">=0"
        self.constraints[c20:c21] = "<=0"
        self.constraints[c21:c22] = ">=0"
        self.constraints[c22:c23] = "<=0"
        self.constraints[c23:c24] = "==0"
        self.constraints[c24:c25] = "<=0"
        self.constraints[c25:c26] = ">=0"
        self.constraints[c26:c27] = "<=0"
        self.constraints[c27:c28] = "==0"
        self.constraints[c28:c29] = ">=0"
        self.constraints[c29:c30] = "<=0"
        self.constraints[c30:c31] = ">=0"
        self.constraints[c31:c32] = "<=0"
        self.constraints[c32:c33] = ">=0"
        self.constraints[c33:c34] = ">=0"
        self.constraints[c34:c35] = "<=0"
        self.constraints[c35:c36] = "<=0"
        self.constraints[c36:c37] = "<=0"
        self.constraints[c37:c38] = ">=0"

        self.ess_num = ess_num
        self.N = N
        self.time_seg = time_seg
        self.Busnum = Busnum
        self.pv_num = pv_num
        self.Linenum = Linenum
        self.len_tap_opt = len(tap_opt)


    def evaluate(self, solution):
        pv_bus = [9,22,29]
        pv_num = len(pv_bus)
        ess_bus = [4,30]
        ess_num = len(ess_bus)

        end_bus = [17,21,24,32]

        PF = 0.9
        night_p = 1 # PV在夜间输出无功的上限占VSC容量的比值
        interest = 0.1 # the annual interest rate
        life = 20 # the life span of PV and ESS
        year = 365 # the days in one year
        yita = 0.9 # the risk factor 
        v_norm = 1 # the normal voltage on each bus
        Busnum = 33 # the bus numbers
        N = 5 # 随机场景的数量
        Linenum = 32 # IEEE33节点算例线路条数
        P_pv_max = 10/100 # 0.1表示光伏的容量，单位是MW，除以100是用100MW对他进行标幺化
        ESS_cap_max = 5/100 # 储能的最大电能容量，1表示储能的电能容量，单位是MW·h，除以100是用100MW对他进行标幺化
        inv_max = 5/100 # 0.01表示储能的充放电功率，单位是MW，除以100是用100MW对他进行标幺化
        fai_max = 0.9 # the maximum SOC of ESS
        fai_min = 0.1 # the minimum SOC of ESS
        lamda_ch = 0.9 # the efficiency of charging
        lamda_dch = 0.9 # the efficiency of discharging
        S_sub = 30/100 # the capacity of substaion and the unit is MVA，10的单位是MVA，除以100是进行标幺化
        basekV = 10
        Vmin = (0.9)**2 # the minimum bus voltage, and the unit is (p.u.)
        Vmax = (1.1)**2 # the maximum bus voltage, and the unit is (p.u.)
        b_con_son = scipy.io.loadmat('b_con_son.mat')['b_con_son'] # the uptril matrix and the i th line contains the son buses of bus i 
        b_con_fa = scipy.io.loadmat('b_con_fa.mat')['b_con_fa'] # the downtril matrix and the i th line contains the father buses of bus i
        z_pu = (basekV)**2/100 # 阻抗的基准值，单位是Ω
        line_r = scipy.io.loadmat('line_r.mat')['line_r']/z_pu # the i th element is the R in the line where bus i is its son bus, 读取出的单位是Ω，除以z_pu化成标幺值
        line_x = scipy.io.loadmat('line_x.mat')['line_x']/z_pu # the i th element is the X in the line where bus i is its son bus, 读取出的单位是Ω，除以z_pu化成标幺值
        i_fa = scipy.io.loadmat('i_fa.mat')['i_fa'] # the father bus of bus i
        c_e_b = scipy.io.loadmat('c_e_t.mat')['c_e_t']*1000*100 # the electricity price for buy at each hour and the unit is $/kWh，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MWh
        c_e_b = c_e_b.reshape((1,1,24))
        c_e_s = np.zeros((1,1,24)) # the electricity price for sell
        for i in range(24):
            c_e_s[0,0,i] = c_e_b[0,0,i] * 0.7

        c_ess_op = 0.00027*1000*100 # the operation cost of ess unit is $/kWh，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MWh
        c_pv_cut = 0.033*1000*100 # the punishment on pv deserting, and the unit is $/kWh，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MWh
        # co2 = 0.779*1000*100 # the co2 emission, and the unit is kg/kWh，乘以1000是将单位换算为kg/MWh，再乘以100是将单位换算为kg/100MWh
        # the co2 emission, and the unit is kg/kWh，乘以1000是将单位换算为kg/MWh，再乘以100是将单位换算为kg/100MWh
        co2 = [0.531914894,	0.717391304,0.844444444,0.837209302,0.833333333,0.86746988,	0.814814815,0.853658537,0.880952381,0.256410256,0.207692308,0.18,0.244444444,0.313953488,0.298850575,0.239583333,0.220883534,0.240480962,0.325581395,0.311111111,0.208333333,0.227743271,0.346938776,0.458333333]
        for i in range(24):
            co2[i] = co2[i] * 1000 * 100

        c_pv = 1343*1000*100 # the installation cost of pv on each bus and it is 1343 $/kW，乘以1000是将单位换算为$/MW，再乘以100是将单位换算为$/100MW
        c_ess = 150*1000*100 # the installation cost of ess. the unit is 150 $/100MWh，或者说150$/p.u.，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MWh
        c_ess_inv = 150*1000*100 # the installation cost of ess. the unit is 150 $/100MW，乘以1000是将单位换算为$/MWh，再乘以100是将单位换算为$/100MW
        i_pu = 100/basekV # 电流的基准值，单位是kA
        i_max = (20/basekV/i_pu)**2 # the maximum I in a line （分子上），所以分母上也得是电流基准值的平方

        p_load = scipy.io.loadmat('p_load.mat')['p_load'][:,0:N,:] # 标幺值 and represents the fluctuation
        q_load = scipy.io.loadmat('q_load.mat')['q_load'][:,0:N,:] # 标幺值 and represents the fluctuation
        p_pv = scipy.io.loadmat('p_pv.mat')['p_pv'][0:pv_num,0:N,:] # the value is in p.u. value and represents the fluctuation, the power factor of PV is 0.9，滞后
        # q_pv = scipy.io.loadmat('q_pv.mat')['q_pv'][:,0:N,:] # the value is in p.u. value and represents the fluctuation, the power factor of PV is 0.9，滞后，PV的无功输出现在是变量了

        # q_pv = q_pv * P_pv # transform into 标幺值,pv的无功功率也是注入

        zero = np.zeros((1,1,1))

        # the tap for the OLTC
        per_num = 3
        per_value = 0.1/per_num
        tap_opt = np.linspace(1-per_value*per_num, 1+per_value*per_num, 2*per_num+1)
        tap_posi = np.zeros((len(tap_opt),N,24))
        for i in range(len(tap_opt)):
            tap_posi[i,:,:] = tap_opt[i]

        # the big M
        M = 1000000

        # life cycle carbon emission
        # life_PV = 1.63 / 0.26 * (185+20) / 17 * 3 # kg/(台*年), 185是production，20是recycling
        life_PV = 1000000/230*180 * 100 # kg/(100 MW), 先用1MW除以230获得对应的面积，再乘以180（每平方米生产制造时的碳排放），最后乘以100是换算成kg/(100 MW)
        life_ESS = 50 * 100 # kg/(100 MWh)
        life_ESS_p = 50 * 100 # kg/(100 MWh)

        hour_max = 5
        hour_min = 1
        co2_ess_op = 0.36*600*100 # 自己换算的，0.36是储能运行1MWh对应的支撑电量，乘以600是均值碳排放强度，乘以100是换算成系统的标幺功率

        fail_line = config.get_fail_line() # 如果是0.01，则含义是每100小时中有1小时故障
        fail_pv = config.get_fail_pv()
        fail_ess = config.get_fail_ess()
        fail_ene = config.get_fail_ene()
        omig_ess_ch = np.array(solution.variables[0:self.n_omig_ess_ch]).reshape((self.ess_num,self.N,self.time_seg)).astype(int) 
        omig_ess_dch = np.array(solution.variables[self.n_omig_ess_ch:self.n_omig_ess_dch]).reshape((self.ess_num,self.N,self.time_seg)).astype(int) 
        p_sub_b = np.array(solution.variables[self.n_omig_ess_dch:self.n_p_sub_b]).reshape((1,self.N,self.time_seg))
        q_sub = np.array(solution.variables[self.n_p_sub_b:self.n_q_sub]).reshape((1,self.N,self.time_seg))
        p_ch = np.array(solution.variables[self.n_q_sub:self.n_p_ch]).reshape((self.ess_num,self.N,self.time_seg))
        p_dch = np.array(solution.variables[self.n_p_ch:self.n_p_dch]).reshape((self.ess_num,self.N,self.time_seg))
        u_bus = np.array(solution.variables[self.n_p_dch:self.n_u_bus]).reshape((self.Busnum,self.N,self.time_seg))
        q_pv = np.array(solution.variables[self.n_u_bus:self.n_q_pv]).reshape((self.pv_num,self.N,self.time_seg))
        p_line = np.array(solution.variables[self.n_q_pv:self.n_p_line]).reshape((self.Linenum,self.N,self.time_seg))
        q_line = np.array(solution.variables[self.n_p_line:self.n_q_line]).reshape((self.Linenum,self.N,self.time_seg))
        i_line = np.array(solution.variables[self.n_q_line:self.n_i_line]).reshape((self.Linenum,self.N,self.time_seg)).astype(int) 
        ess_w = np.array(solution.variables[self.n_i_line:self.n_ess_w]).reshape((self.ess_num,1))
        v_1 = np.array(solution.variables[self.n_ess_w:self.n_v_1]).reshape((self.Busnum,self.N,self.time_seg))
        v_2 = np.array(solution.variables[self.n_v_1:self.n_v_2]).reshape((self.Busnum,self.N,self.time_seg))
        tap_k = np.array(solution.variables[self.n_v_2:self.n_tap_k]).reshape((self.len_tap_opt,self.N,self.time_seg)).astype(int) 
        E_ess = np.array(solution.variables[self.n_tap_k:self.n_E_ess]).reshape((self.ess_num,1))
        p_ess = np.array(solution.variables[self.n_E_ess:self.n_p_ess]).reshape((self.ess_num,1))
        ch_ess = np.array(solution.variables[self.n_p_ess:self.n_ch_ess]).reshape((self.ess_num,self.N,self.time_seg))
        dis_ess = np.array(solution.variables[self.n_ch_ess:self.n_dis_ess]).reshape((self.ess_num,self.N,self.time_seg))
        pv_cap = np.array(solution.variables[self.n_dis_ess:self.n_pv_cap]).reshape((self.pv_num,1))
        power_pv = np.array(solution.variables[self.n_pv_cap:self.n_power_pv]).reshape((self.pv_num,self.N,self.time_seg))
        q_ess = np.array(solution.variables[self.n_power_pv:self.n_q_ess]).reshape((self.ess_num,self.N,self.time_seg))
        abs_p_line = np.array(solution.variables[self.n_q_ess:self.n_abs_p_line]).reshape((self.Linenum,self.N,self.time_seg))

        # objectives
        # obj1
        solution.objectives[0] = [life_PV * pv_cap.sum() + life_ESS * E_ess.sum() + life_ESS_p*p_ess.sum() + sum(co2_ess_op*(p_ch[i,j,t]+p_dch[i,j,t]) for i in range(ess_num) for j in range(N) for t in range(24))/N*year*life + sum((p_sub_b[0,:,t].sum())*co2[t] for t in range(24))/N*year*life]

        # obj2
        os_sum = 1/N*sum(sum(p_sub_b[0,j,t]*c_e_b[0,0,t] for t in range(24)) + sum(p_ch[i,j,t]+p_dch[i,j,t] for i in range(ess_num) for t in range(24))*c_ess_op + sum(pv_cap[i,0]*p_pv[i,j,t]-power_pv[i,j,t] for i in range(pv_num) for t in range(24))*c_pv_cut for j in range(N))
        
        in_rate = (interest*(1+interest)**life)/((1+interest)**life-1)
        
        solution.objectives[1] = [(pv_cap.sum()*c_pv + E_ess.sum()*c_ess + p_ess.sum()*c_ess_inv)*in_rate + year*os_sum]

        # obj3
        solution.objectives[2] = [(pv_cap.sum()*c_pv + E_ess.sum()*c_ess + p_ess.sum()*c_ess_inv)*in_rate + year*os_sum]

        # constraints
        solution.constraints[0:self.c1] = [omig_ess_ch[b,i,j] + omig_ess_dch[b,i,j] - ess_w[b,0] for b in range(ess_num) for i in range(N) for j in range(self.time_seg)]
        solution.constraints[self.c1:self.c2] = [p_ch[b,i,j] for b in range(ess_num) for i in range(N) for j in range(self.time_seg)]
        solution.constraints[self.c2:self.c3] = [p_ch[b,i,j] - ch_ess[b,i,j] for b in range(ess_num) for i in range(N) for j in range(self.time_seg)]
        solution.constraints[self.c3:self.c4] = [p_dch[b,i,j] for b in range(ess_num) for i in range(N) for j in range(self.time_seg)]
        solution.constraints[self.c4:self.c5] = [p_dch[b,i,j] - dis_ess[b,i,j] for b in range(ess_num) for i in range(N) for j in range(self.time_seg)]
        solution.constraints[self.c5:self.c6] = [ch_ess[b,i,j] - omig_ess_ch[b,i,j]*M for b in range(ess_num) for i in range(N) for j in range(self.time_seg)] + [dis_ess[b,i,j] - omig_ess_dch[b,i,j]*M for b in range(ess_num) for i in range(N) for j in range(self.time_seg)]
        solution.constraints[self.c6:self.c7] = [ch_ess[b,i,j] - p_ess[b,0] for b in range(ess_num) for i in range(N) for j in range(24)] + [dis_ess[b,i,j] - p_ess[b,0] for b in range(ess_num) for i in range(N) for j in range(24)]
        solution.constraints[self.c7:self.c8] = [ch_ess[b,i,j] for b in range(ess_num) for i in range(N) for j in range(24)] + [dis_ess[b,i,j] for b in range(ess_num) for i in range(N) for j in range(24)]
        solution.constraints[self.c8:self.c9] = [ch_ess[b,i,j] - p_ess[b,0] + (1-omig_ess_ch[b,i,j])*M for b in range(ess_num) for i in range(N) for j in range(24)] + [dis_ess[b,i,j] - p_ess[b,0] + (1-omig_ess_dch[b,i,j])*M for b in range(ess_num) for i in range(N) for j in range(24)]
        solution.constraints[self.c9:self.c10] = [power_pv[b,i,j] * power_pv[b,i,j] + q_pv[b,i,j] * q_pv[b,i,j] - pv_cap[b,0] * pv_cap[b,0] for b in range(pv_num) for i in range(N) for j in range(6,20)]
        solution.constraints[self.c10:self.c11] = [q_pv[b,i,j] + night_p*pv_cap[b,0] for b in range(pv_num) for i in range(N) for j in range(0,6)] + [q_pv[b,i,j] + night_p*pv_cap[b,0] for b in range(pv_num) for i in range(N) for j in range(20,24)]
        solution.constraints[self.c11:self.c12] = [q_pv[b,i,j] - night_p*pv_cap[b,0] for b in range(pv_num) for i in range(N) for j in range(0,6)] + [q_pv[b,i,j] - night_p*pv_cap[b,0] for b in range(pv_num) for i in range(N) for j in range(20,24)]
        solution.constraints[self.c12:self.c13] = [E_ess[i,0]*0.5 + p_ch[i,j,0:t].sum()*lamda_ch - p_dch[i,j,0:t].sum()/lamda_dch - E_ess[i,0]*fai_min for i in range(ess_num) for j in range(N) for t in range(1,24)]
        solution.constraints[self.c13:self.c14] = [E_ess[i,0]*0.5 + p_ch[i,j,0:t].sum()*lamda_ch - p_dch[i,j,0:t].sum()/lamda_dch - E_ess[i,0]*fai_max for i in range(ess_num) for j in range(N) for t in range(1,24)]
        solution.constraints[self.c14:self.c15] = [E_ess[i,0]*0.5 + p_ch[i,j,0:24].sum()*lamda_ch - p_dch[i,j,0:24].sum()/lamda_dch - E_ess[i,0]*0.5 for i in range(ess_num) for j in range(N)]
        solution.constraints[self.c15:self.c16] = [p_sub_b[0,i,j]*p_sub_b[0,i,j] + q_sub[0,i,j]*q_sub[0,i,j] - S_sub**2 for i in range(N) for j in range(24)]

        # the power balance constraint in each bus
        # there is no father bus for bus 1
        pre_son = np.argwhere(b_con_son[0,:] == 1) # 注意索引是从0开始
        son = []
        for i in pre_son:
            son.append(list(i)[0].tolist())
        solution.constraints[self.c16:self.c17] = [sum(p_line[m-1,j,t] for m in son) - p_sub_b[0,j,t] + p_load[0,j,t] for j in range(N) for t in range(24)] + [sum(q_line[m-1,j,t] for m in son) - q_sub[0,j,t] + q_load[0,j,t] for j in range(N) for t in range(24)]

        cons_c18 = []
        # 功率平衡方程，考虑PV和ESS的选择性
        pv_count = 0
        ess_count = 0
        for i in range(1,self.Busnum):
            if i not in end_bus:
                pre_son = np.argwhere(b_con_son[i,:] == 1)
                son = []
                for n in pre_son:
                    son.append(list(n)[0].tolist())
                
                if (i in pv_bus) and (i in ess_bus):
                    cons_c18.extend([sum(p_line[m-1,j,t] for m in son) -    
                power_pv[pv_count,j,t] - p_dch[ess_count,j,t] + p_ch[ess_count,j,t] + p_load[i,j,t] - p_line[i-1,j,t] + i_line[i-1,j,t]*line_r[i] for j in range(N) for t in range(24)])
                    
                    cons_c18.extend([sum(q_line[m-1,j,t] for m in son) - q_ess[ess_count,j,t] - q_pv[pv_count,j,t] + q_load[i,j,t] - q_line[i-1,j,t] + i_line[i-1,j,t]*line_x[i] for j in range(N) for t in range(24)])
                    
                    pv_count = pv_count + 1
                    ess_count = ess_count + 1
                elif (i in pv_bus) and (i not in ess_bus):
                    cons_c18.extend([sum(p_line[m-1,j,t] for m in son) -    
                power_pv[pv_count,j,t] + p_load[i,j,t] - p_line[i-1,j,t] + i_line[i-1,j,t]*line_r[i] for j in range(N) for t in range(24)])
                    
                    cons_c18.extend([sum(q_line[m-1,j,t] for m in son) - q_pv[pv_count,j,t] + q_load[i,j,t] - q_line[i-1,j,t] + i_line[i-1,j,t]*line_x[i] for j in range(N) for t in range(24)])
                    
                    pv_count = pv_count + 1
                elif (i in ess_bus) and (i not in pv_bus):
                    cons_c18.extend([sum(p_line[m-1,j,t] for m in son) -    
                p_dch[ess_count,j,t] + p_ch[ess_count,j,t] + p_load[i,j,t] - p_line[i-1,j,t] + i_line[i-1,j,t]*line_r[i] for j in range(N) for t in range(24)])
                    
                    cons_c18.extend([sum(q_line[m-1,j,t] for m in son) - q_ess[ess_count,j,t] + q_load[i,j,t] - q_line[i-1,j,t] + i_line[i-1,j,t]*line_x[i] for j in range(N) for t in range(24)])

                    ess_count = ess_count + 1
                else:
                    cons_c18.extend([sum(p_line[m-1,j,t] for m in son) +    
                p_load[i,j,t] - p_line[i-1,j,t] + i_line[i-1,j,t]*line_r[i] for j in range(N) for t in range(24)])
                    
                    cons_c18.extend([sum(q_line[m-1,j,t] for m in son) + q_load[i,j,t] - q_line[i-1,j,t] + i_line[i-1,j,t]*line_x[i] for j in range(N) for t in range(24)])
            
            else:
                # there is no son buses for ending buses
                if (i in pv_bus) and (i in ess_bus):
                    cons_c18.extend([power_pv[pv_count,j,t] + p_dch[ess_count,j,t] - p_ch[ess_count,j,t] - p_load[i,j,t] + p_line[i-1,j,t] -i_line[i-1,j,t]*line_r[i] for j in range(N) for t in range(24)])
                    
                    cons_c18.extend([q_ess[ess_count,j,t] + q_pv[pv_count,j,t] - q_load[i,j,t] + q_line[i-1,j,t] -i_line[i-1,j,t]*line_x[i] for j in range(N) for t in range(24)])

                    pv_count = pv_count + 1
                    ess_count = ess_count + 1
                elif (i in pv_bus) and (i not in ess_bus):
                    cons_c18.extend([power_pv[pv_count,j,t] - p_load[i,j,t] + p_line[i-1,j,t] -i_line[i-1,j,t]*line_r[i] for j in range(N) for t in range(24)])

                    cons_c18.extend([q_pv[pv_count,j,t] - q_load[i,j,t] + q_line[i-1,j,t] -i_line[i-1,j,t]*line_x[i] for j in range(N) for t in range(24)])
                    
                    pv_count = pv_count + 1
                elif (i in ess_bus) and (i not in pv_bus):
                    cons_c18.extend([p_dch[ess_count,j,t] - p_ch[ess_count,j,t] - p_load[i,j,t] + p_line[i-1,j,t] -i_line[i-1,j,t]*line_r[i] for j in range(N) for t in range(24)])
                    
                    cons_c18.extend([q_ess[ess_count,j,t] - q_load[i,j,t] + q_line[i-1,j,t] -i_line[i-1,j,t]*line_x[i] for j in range(N) for t in range(24)])

                    ess_count = ess_count + 1
                else:
                    cons_c18.extend([p_load[i,j,t] - p_line[i-1,j,t] + i_line[i-1,j,t]*line_r[i] for j in range(N) for t in range(24)])
                    
                    cons_c18.extend([q_load[i,j,t] - q_line[i-1,j,t] + i_line[i-1,j,t]*line_x[i] for j in range(N) for t in range(24)])
        solution.constraints[self.c17:self.c18] = cons_c18
        
        solution.constraints[self.c18:self.c19] = [u_bus[i,j,t] - u_bus[i_fa[i]-1,j,t] + 2*(p_line[i-1,j,t]*line_r[i] + q_line[i-1,j,t]*line_x[i]) - i_line[i-1,j,t]*(np.square(line_r[i])+np.square(line_x[i])) for i in range(1,Busnum) for j in range(N) for t in range(24)]
        solution.constraints[self.c19:self.c20] = [u_bus[i,j,t] - Vmin for i in range(1,Busnum) for j in range(N) for t in range(24)]
        solution.constraints[self.c20:self.c21] = [u_bus[i,j,t] - Vmax for i in range(1,Busnum) for j in range(N) for t in range(24)]
        solution.constraints[self.c21:self.c22] = [u_bus[0,j,t] - tap_posi[i,j,t]**2 + (1-tap_k[i,j,t])*M for i in range(len(tap_opt)) for j in range(N) for t in range(24)]
        solution.constraints[self.c22:self.c23] = [u_bus[0,j,t] - tap_posi[i,j,t]**2 - (1-tap_k[i,j,t])*M for i in range(len(tap_opt)) for j in range(N) for t in range(24)]
        solution.constraints[self.c23:self.c24] = [tap_k[:,j,t].sum() - 1 for j in range(N) for t in range(24)]

        solution.constraints[self.c24:self.c25] = [p_line[i-1,j,t]*p_line[i-1,j,t] + q_line[i-1,j,t]*q_line[i-1,j,t] - u_bus[i_fa[i]-1,j,t]*i_line[i-1,j,t] for i in range(1,Busnum) for j in range(N) for t in range(24)]
        solution.constraints[self.c25:self.c26] = [i_line[i,j,t] for i in range(Linenum) for j in range(N) for t in range(24)]
        solution.constraints[self.c26:self.c27] = [i_line[i,j,t] - i_max for i in range(Linenum) for j in range(N) for t in range(24)]

        solution.constraints[self.c27:self.c28] = [v_1[i,j,t] - v_2[i,j,t] - u_bus[i,j,t] + v_norm for i in range(Busnum) for j in range(N) for t in range(24)]
        solution.constraints[self.c28:self.c29] = [v_1[i,j,t] for i in range(Busnum) for j in range(N) for t in range(24)] + [v_2[i,j,t] for i in range(Busnum) for j in range(N) for t in range(24)]

        solution.constraints[self.c29:self.c30] = [E_ess[i,0] - ess_w[i,0]*ESS_cap_max for i in range(ess_num)] + [p_ess[i,0] - ess_w[i,0]*inv_max for i in range(ess_num)]
        solution.constraints[self.c30:self.c31] = [E_ess[i,0] for i in range(ess_num)] + [p_ess[i,0] for i in range(ess_num)]
        solution.constraints[self.c31:self.c32] = [pv_cap[i,0] - P_pv_max for i in range(pv_num)] + [power_pv[i,j,t] - pv_cap[i,0]*p_pv[i,j,t] for i in range(pv_num) for j in range(N) for t in range(24)]
        solution.constraints[self.c32:self.c33] = [pv_cap[i,0] for i in range(pv_num)] + [power_pv[i,j,t] for i in range(pv_num) for j in range(N) for t in range(24)]
        solution.constraints[self.c33:self.c34] = [E_ess[i,0] - hour_min * p_ess[i,0] for i in range(ess_num)]
        solution.constraints[self.c34:self.c35] = [E_ess[i,0] - hour_max * p_ess[i,0] for i in range(ess_num)]
        solution.constraints[self.c35:self.c36] = [p_ch[i,j,t]*p_ch[i,j,t] + q_ess[i,j,t]*q_ess[i,j,t] - p_ess[i,0]*p_ess[i,0] for i in range(ess_num) for j in range(N) for t in range(24)] + [p_dch[i,j,t]*p_dch[i,j,t] + q_ess[i,j,t]*q_ess[i,j,t] - p_ess[i,0]*p_ess[i,0] for i in range(ess_num) for j in range(N) for t in range(24)]
        solution.constraints[self.c36:self.c37] = [fail_line/N*abs_p_line[:,:,:].sum() +  fail_ess/N*(p_ch[:,:,:] + p_dch[:,:,:]).sum() + fail_pv/N*power_pv[:,:,:].sum() - fail_ene/N * p_load[:,:,:].sum()]
        solution.constraints[self.c37:self.c38] = [abs_p_line[i,j,t] - p_line[i,j,t] for i in range(Linenum) for j in range(N) for t in range(24)] + [abs_p_line[i,j,t] + p_line[i,j,t] for i in range(Linenum) for j in range(N) for t in range(24)]


FU_1 = config.get_FU_1()
FU_2 = config.get_FU_2()
FU_3 = config.get_FU_3()

FN_1 = config.get_FN_1()
FN_2 = config.get_FN_2()
FN_3 = config.get_FN_3()

generation_num = 100000
population = 4000
upper_limit = 0.01
algorithm = OMOPSO(IEEE33(),swarm_size=4000,leader_size=1000,epsilons=0.5)
algorithm.run(generation_num)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
        [(np.array(s.objectives[0])-FU_1)/(FN_1-FU_1) for s in algorithm.result],
        [(np.array(s.objectives[1])-FU_2)/(FN_2-FU_2) for s in algorithm.result],
        [(np.array(s.objectives[2])-FU_3)/(FN_3-FU_3) for s in algorithm.result]
            )
plt.savefig('gene_'+str(generation_num)+'_pop_'+str(population)+'_upper_limit_'+str(upper_limit)+'_normalize.png')
plt.show()

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(
        [s.objectives[0] for s in algorithm.result],
        [s.objectives[1] for s in algorithm.result],
        [s.objectives[2] for s in algorithm.result]
)
plt.savefig('gene_'+str(generation_num)+'_pop_'+str(population)+'_upper_limit_'+str(upper_limit)+'_real value.png')
plt.show()
import json
import scipy.io
import numpy as np
import config

#根据保存的优化结果计算三个目标函数各自的值

def obj_values(w1,w2):
     obj = 4
     pv_bus = [3,7,14,21,23,26]
     pv_num = len(pv_bus)
     ess_bus = [4,19,29]
     ess_num = len(ess_bus)

     end_bus = [8,11,14,16,17,25,28,30,31,32]

     PF = 0.9
     night_p = 1 # PV在夜间输出无功的上限占VSC容量的比值
     interest = 0.1 # the annual interest rate
     life = 20 # the life span of PV and ESS
     year = 365 # the days in one year
     yita = 0.9 # the risk factor 
     v_norm = 1 # the normal voltage on each bus
     Busnum = 33 # the bus numbers
     N = 20 # 随机场景的数量
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
     per_value = 0.01*1.67
     per_num = 8
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

     # calculate the objective function values
     result_data = np.load('./DataFile/result_obj'+str(obj)+'_w1_'+str(w1)+'_w2_'+str(w2)+'_pv'+str(P_pv_max)+'.npz')
     p_sub_b = result_data['p_sub_b']
     ess_w = result_data['ess_w']
     p_ch = result_data['p_ch']
     p_dch = result_data['p_dch']
     p_ess = result_data['p_ess']
     E_ess = result_data['E_ess']
     v_1 = result_data['v_1']
     v_2 = result_data['v_2']
     pv_cap = result_data['pv_cap']
     power_pv = result_data['power_pv']

     # 碳排放量
     F1 = sum((p_sub_b[0,:,t].sum())*co2[t]/N*year for t in range(24)) + life_PV * pv_cap.sum() + life_ESS * E_ess.sum() + life_ESS_p*p_ess.sum() + sum(co2_ess_op*(p_ch[i,j,t]+p_dch[i,j,t]) for i in range(ess_num) for j in range(N) for t in range(24))/N*year

     sum_1 = F1
 


     # F2：经济性
     os_sum = 1/N*sum(sum(p_sub_b[0,j,t]*c_e_b[0,0,t] for t in range(24)) + sum(p_ch[i,j,t]+p_dch[i,j,t] for i in range(ess_num) for t in range(24))*c_ess_op + sum(pv_cap[i,0]*p_pv[i,j,t]-power_pv[i,j,t] for i in range(pv_num) for t in range(24))*c_pv_cut for j in range(N))
     in_rate = (interest*(1+interest)**life)/((1+interest)**life-1)

     F2 = (pv_cap.sum()*c_pv + E_ess.sum()*c_ess + p_ess.sum()*c_ess_inv)*in_rate + year*os_sum

     sum_2 = F2

     # F3：电压偏移
     sum_3 = (v_1.sum()+v_2.sum())/N

     return sum_1, sum_2, sum_3



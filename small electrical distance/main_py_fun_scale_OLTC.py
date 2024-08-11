from fileinput import filename
import imp
from os import name
from statistics import mode
from numpy.core.defchararray import mod
import scipy.io
import numpy as np
import gurobipy as gpy
import json
import easygui
import config
import os

# def main_fun(obj_index,w1,w2):
obj_index = 2
w1 = 0.1
w2 = 0.1

FU_1 = config.get_FU_1()
FU_2 = config.get_FU_2()
FU_3 = config.get_FU_3()

FN_1 = config.get_FN_1()
FN_2 = config.get_FN_2()
FN_3 = config.get_FN_3()

co2_pro = config.get_co2_pro()

pv_bus = [2,10,19,21]
pv_num = len(pv_bus)
ess_bus = [25,10]
ess_num = len(ess_bus)

end_bus = scipy.io.loadmat('end_bus.mat')['end_bus'] 

PF = 0.9
night_p = 1 # PV在夜间输出无功的上限占VSC容量的比值
interest = 0.1 # the annual interest rate
life = 20 # the life span of PV and ESS
year = 365 # the days in one year
v_norm = 1 # the normal voltage on each bus
Busnum = 28 # the bus numbers
N = 20 # 随机场景的数量
Linenum = 27 # IEEE33节点算例线路条数
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
life_PV = 1000000/230*180 * 100 * co2_pro # kg/(100 MW), 先用1MW除以230获得对应的面积，再乘以180（每平方米生产制造时的碳排放），最后乘以100是换算成kg/(100 MW)
life_ESS = 50 * 100 * co2_pro # kg/(100 MWh)
life_ESS_p = 50 * 100 * co2_pro # kg/(100 MWh)

hour_max = 5
hour_min = 1
co2_ess_op = 0.36*600*100 # 自己换算的，0.36是储能运行1MWh对应的支撑电量，乘以600是均值碳排放强度，乘以100是换算成系统的标幺功率

fail_line = config.get_fail_line() # 如果是0.01，则含义是每100小时中有1小时故障
fail_pv = config.get_fail_pv()
fail_ess = config.get_fail_ess()
fail_ene = config.get_fail_ene()

print('1')

model = gpy.Model() 
# yike = model.addMVar((1,N,1),vtype=gpy.GRB.CONTINUOUS,lb=-gpy.GRB.INFINITY,name='yike')
# kesai = model.addMVar(1,lb=-gpy.GRB.INFINITY,name='kesai')
omig_ess_ch = model.addMVar((ess_num,N,24),vtype=gpy.GRB.BINARY,name='omig_ess_ch') # 储能系统是否充电
omig_ess_dch = model.addMVar((ess_num,N,24),vtype=gpy.GRB.BINARY, name='omig_ess_dch') # 储能系统是否充电

p_sub = model.addMVar((1,N,24),lb=-gpy.GRB.INFINITY, name = 'p_sub') # the power output of substation, only installed at bus 1 
p_sub_b = model.addMVar((1,N,24), lb=-gpy.GRB.INFINITY, name = 'p_sub_b') # the power bought from main grid, only positive
# p_sub_s = model.addMVar((1,N,24), name = 'p_sub_s') # the power sold to main grid, only positive

q_sub = model.addMVar((1,N,24),lb=-gpy.GRB.INFINITY, name = 'q_sub')
p_ch = model.addMVar((ess_num,N,24), name = 'p_ch') # 每个节点在不同场景、不同时刻的储能充电功率
p_dch = model.addMVar((ess_num,N,24), name = 'p_dch') # 每个节点在不同场景、不同时刻的储能放电功率
# gama = model.addMVar((pv_num,N,24), name = 'gama') # 每个节点的PV在不同场景、不同时刻的弃光率
u_bus = model.addMVar((Busnum,N,24), name = 'u_bus') # the square of voltage at each bus
# miu = model.addMVar((pv_num,N,24), name = 'miu') # 光伏的实际输出率
q_pv = model.addMVar((pv_num,N,24),lb=-gpy.GRB.INFINITY, name = 'q_pv')

p_line = model.addMVar((Linenum,N,24),lb=-gpy.GRB.INFINITY, name = 'p_line') # 每条线路在不同场景、不同时刻的有功大小，设行号为i,则第i条线路的子节点为i+1
q_line = model.addMVar((Linenum,N,24),lb=-gpy.GRB.INFINITY, name = 'q_line') # 每条线路在不同场景、不同时刻的无功大小，设行号为i,则第i条线路的子节点为i+1
i_line = model.addMVar((Linenum,N,24), name = 'i_line') # the square of current in each line
# pv_w = model.addMVar((pv_num,1), vtype=gpy.GRB.BINARY, name = 'pv_w')
ess_w = model.addMVar((ess_num,1), vtype=gpy.GRB.BINARY, name = 'ess_w')
v_1 = model.addMVar((Busnum,N,24), name = 'v_1') # the auxiliary variable 1 for absolute voltage deviation value 
v_2 = model.addMVar((Busnum,N,24), name = 'v_2') # the auxiliary variable 2 for absolute voltage deviation value 
# dr_u = model.addMVar((dr_num,N,24), name='dr_u')
# dr_d = model.addMVar((dr_num,N,24), name='dr_d')
tap_k = model.addMVar((len(tap_opt),N,24), vtype=gpy.GRB.BINARY, name = 'tap_k')

# big-M theory for ESS planning
E_ess = model.addMVar((ess_num,1), name='energy capacity')
p_ess = model.addMVar((ess_num,1), name='power capacity')
ch_ess = model.addMVar((ess_num,N,24), name='new charging')
dis_ess = model.addMVar((ess_num,N,24), name='new discharging')

# variables for PV units
pv_cap = model.addMVar((pv_num,1), name = 'pv_cap')
power_pv = model.addMVar((pv_num,N,24), name='power_pv')

# reactive power from ESS
q_ess = model.addMVar((ess_num,N,24),lb=-gpy.GRB.INFINITY, name='q of ESS')

abs_p_line = model.addMVar((Linenum,N,24),lb=-gpy.GRB.INFINITY, name='abs_p_line')

model.update()

print('2')

# 限制某些点不能安装PV和ESS
# 只有4，7，12，17，21，22，24，28，30，32（从0算起）可以安装PV
# model.addConstrs(pv_w[i,0] == 0 for i in [0,1,2,3,5,6,8,9,10,11,13,14,15,16,18,19,20,23,25,26,27,29,31])
# 只有4，9，13，19，29（从0算起）可以安装ESS
# model.addConstrs(ess_w[i,0] == 0 for i in [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32])

model.addConstrs(omig_ess_ch[b,i,j] + omig_ess_dch[b,i,j] <= ess_w[b]\
    for b in range(ess_num) for i in range(N) for j in range(24))

# the charging constraint of ess, if there is no ess installed, the power has to be zero, also the first big-M constraint
model.addConstrs(p_ch[b,i,j] >= 0 for b in range(ess_num) for i in range(N) for j in range(24))
model.addConstrs(p_ch[b,i,j] <= ch_ess[b,i,j] for b in range(ess_num) for i in range(N) for j in range(24))
# the discharging constraint of ess
model.addConstrs(p_dch[b,i,j] >= 0 for b in range(ess_num) for i in range(N) for j in range(24))
model.addConstrs(p_dch[b,i,j] <= dis_ess[b,i,j] for b in range(ess_num) for i in range(N) for j in range(24))

# The second big-M constraint
model.addConstrs(ch_ess[b,i,j] <= omig_ess_ch[b,i,j]*M for b in range(ess_num) for i in range(N) for j in range(24))
model.addConstrs(dis_ess[b,i,j] <= omig_ess_dch[b,i,j]*M for b in range(ess_num) for i in range(N) for j in range(24))

# the third big-M constraint
model.addConstrs(ch_ess[b,i,j] <= p_ess[b,0] for b in range(ess_num) for i in range(N) for j in range(24))
model.addConstrs(ch_ess[b,i,j] >= 0 for b in range(ess_num) for i in range(N) for j in range(24))
model.addConstrs(dis_ess[b,i,j] <= p_ess[b,0] for b in range(ess_num) for i in range(N) for j in range(24))
model.addConstrs(dis_ess[b,i,j] >= 0 for b in range(ess_num) for i in range(N) for j in range(24))

# the fourth big-M constraint
model.addConstrs(ch_ess[b,i,j] >= p_ess[b,0] - (1-omig_ess_ch[b,i,j])*M for b in range(ess_num) for i in range(N) for j in range(24))
model.addConstrs(dis_ess[b,i,j] >= p_ess[b,0] - (1-omig_ess_dch[b,i,j])*M for b in range(ess_num) for i in range(N) for j in range(24))

# #  the constraint on pv cut rate. It has to be 1 when there is no PV installed and it is [0,1] when there is PV
# model.addConstrs(gama[b,i,j] >= 0 for b in range(pv_num)\
#     for i in range(N) for j in range(24))
# model.addConstrs(gama[b,i,j] <= 1 for b in range(pv_num)\
#     for i in range(N) for j in range(24))

# #  the constraint on pv remaining rate.
# model.addConstrs(miu[b,i,j] >= 0 for b in range(pv_num) for i in range(N) for j in range(24))
# model.addConstrs(miu[b,i,j] <= 1 for b in range(pv_num) for i in range(N) for j in range(24))

# 在白天
for b in range(pv_num):
    for i in range(N):
        for j in range(6,20):
            # PV的无功输出功率与实际输出的有功满足功率因数的约束
            # model.addConstr((1-PF**2) * p_pv[b,i,j]**2 * (miu[b,i,j] @ miu[b,i,j]) >= PF**2 * (q_pv[b,i,j] @ q_pv[b,i,j]))
            # PV的有功无功独立控制，只需满足容量约束即可
            model.addConstr(power_pv[b,i,j] * power_pv[b,i,j] + q_pv[b,i,j] * q_pv[b,i,j] <= pv_cap[b,0] * pv_cap[b,0])
# model.addConstrs(q_pv[b,i,j] <= 0.9*miu[b,i,j]*p_pv[b,i,j] for b in range(Busnum) for i in range(N) for j in range(6,20))


# 在夜间，PV的无功输出功率上下限即VSC的容量
model.addConstrs(q_pv[b,i,j] >= -night_p*pv_cap[b,0] for b in range(pv_num) for i in range(N) for j in range(0,6))
model.addConstrs(q_pv[b,i,j] <= night_p*pv_cap[b,0] for b in range(pv_num) for i in range(N) for j in range(0,6))
model.addConstrs(q_pv[b,i,j] >= -night_p*pv_cap[b,0] for b in range(pv_num) for i in range(N) for j in range(20,24))
model.addConstrs(q_pv[b,i,j] <= night_p*pv_cap[b,0] for b in range(pv_num) for i in range(N) for j in range(20,24))



# # the relation between pv cut and pv remaining
# model.addConstrs(miu[b,i,j] + gama[b,i,j] == pv_w[b,0] for b in range(pv_num) for i in range(N) for j in range(24))

print('3')

# the energy constraints on ess
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0]*lamda_ch - p_dch[i,j,0]/lamda_dch >= E_ess[i,0]*fai_min for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0]*lamda_ch - p_dch[i,j,0]/lamda_dch <= E_ess[i,0]*fai_max for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:2].sum()*lamda_ch - p_dch[i,j,0:2].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:2].sum()*lamda_ch - p_dch[i,j,0:2].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:3].sum()*lamda_ch - p_dch[i,j,0:3].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:3].sum()*lamda_ch - p_dch[i,j,0:3].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:4].sum()*lamda_ch - p_dch[i,j,0:4].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:4].sum()*lamda_ch - p_dch[i,j,0:4].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:5].sum()*lamda_ch - p_dch[i,j,0:5].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:5].sum()*lamda_ch - p_dch[i,j,0:5].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:6].sum()*lamda_ch - p_dch[i,j,0:6].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:6].sum()*lamda_ch - p_dch[i,j,0:6].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:7].sum()*lamda_ch - p_dch[i,j,0:7].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:7].sum()*lamda_ch - p_dch[i,j,0:7].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:8].sum()*lamda_ch - p_dch[i,j,0:8].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:8].sum()*lamda_ch - p_dch[i,j,0:8].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))
        
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:9].sum()*lamda_ch - p_dch[i,j,0:9].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:9].sum()*lamda_ch - p_dch[i,j,0:9].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:10].sum()*lamda_ch - p_dch[i,j,0:10].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:10].sum()*lamda_ch - p_dch[i,j,0:10].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:11].sum()*lamda_ch - p_dch[i,j,0:11].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:11].sum()*lamda_ch - p_dch[i,j,0:11].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:12].sum()*lamda_ch - p_dch[i,j,0:12].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:12].sum()*lamda_ch - p_dch[i,j,0:12].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:13].sum()*lamda_ch - p_dch[i,j,0:13].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:13].sum()*lamda_ch - p_dch[i,j,0:13].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:14].sum()*lamda_ch - p_dch[i,j,0:14].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:14].sum()*lamda_ch - p_dch[i,j,0:14].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:15].sum()*lamda_ch - p_dch[i,j,0:15].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:15].sum()*lamda_ch - p_dch[i,j,0:15].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:16].sum()*lamda_ch - p_dch[i,j,0:16].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:16].sum()*lamda_ch - p_dch[i,j,0:16].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:17].sum()*lamda_ch - p_dch[i,j,0:17].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:17].sum()*lamda_ch - p_dch[i,j,0:17].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:18].sum()*lamda_ch - p_dch[i,j,0:18].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:18].sum()*lamda_ch - p_dch[i,j,0:18].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:19].sum()*lamda_ch - p_dch[i,j,0:19].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:19].sum()*lamda_ch - p_dch[i,j,0:19].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:20].sum()*lamda_ch - p_dch[i,j,0:20].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:20].sum()*lamda_ch - p_dch[i,j,0:20].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:21].sum()*lamda_ch - p_dch[i,j,0:21].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:21].sum()*lamda_ch - p_dch[i,j,0:21].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:22].sum()*lamda_ch - p_dch[i,j,0:22].sum()/lamda_dch >= E_ess[i,0]*fai_min\
    for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:22].sum()*lamda_ch - p_dch[i,j,0:22].sum()/lamda_dch <= E_ess[i,0]*fai_max\
    for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:23].sum()*lamda_ch - p_dch[i,j,0:23].sum()/lamda_dch >= E_ess[i,0]*fai_min for i in range(ess_num) for j in range(N))
model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:23].sum()*lamda_ch - p_dch[i,j,0:23].sum()/lamda_dch <= E_ess[i,0]*fai_max for i in range(ess_num) for j in range(N))

model.addConstrs(E_ess[i,0]*0.5 + p_ch[i,j,0:24].sum()*lamda_ch - p_dch[i,j,0:24].sum()/lamda_dch == E_ess[i,0]*0.5 for i in range(ess_num) for j in range(N))

print('4')

# # 使用正八边形线性化变电站功率约束
# model.addConstrs(p_sub_b[0,i,j] >= -S_sub for i in range(N)\
#     for j in range(24))
# model.addConstrs(p_sub_b[0,i,j] <= S_sub for i in range(N)\
#     for j in range(24))
# model.addConstrs(q_sub[0,i,j] >= -S_sub for i in range(N)\
#     for j in range(24))
# model.addConstrs(q_sub[0,i,j] <= S_sub for i in range(N)\
#     for j in range(24))
# model.addConstrs(p_sub_b[0,i,j]-q_sub[0,i,j] >= -np.sqrt(2)*S_sub for i in range(N) for j in range(24))
# model.addConstrs(p_sub_b[0,i,j]-q_sub[0,i,j] <= np.sqrt(2)*S_sub for i in range(N) for j in range(24))
# model.addConstrs(p_sub_b[0,i,j]+q_sub[0,i,j] >= -np.sqrt(2)*S_sub for i in range(N) for j in range(24))
# model.addConstrs(p_sub_b[0,i,j]+q_sub[0,i,j] <= np.sqrt(2)*S_sub for i in range(N) for j in range(24))

# 对变电站使用二次约束
model.addConstrs(p_sub_b[0,i,j]*p_sub_b[0,i,j] + q_sub[0,i,j]*q_sub[0,i,j] <= S_sub**2 for i in range(N) for j in range(24))

print('5')

# the power balance constraint in each bus
# there is no father bus for bus 1
pre_son = np.argwhere(b_con_son[0,:] == 1) # 注意索引是从0开始
son = []
for i in pre_son:
    son.append(list(i)[0].tolist())
model.addConstrs((sum(p_line[m-1,j,t] for m in son) == p_sub_b[0,j,t] - p_load[0,j,t] for j in range(N) for t in range(24)), name = 'c1')  # 由于节点和p_line的索引都是从0开始，抵消了，所以son仍需要减1
model.addConstrs((sum(q_line[m-1,j,t] for m in son) == q_sub[0,j,t] - q_load[0,j,t] for j in range(N) for t in range(24)), name = 'c2')  # 由于索引是从0开始，所以son不再需要减1

print('6')

# PV节点和储能节点重合，他们的功率平衡方程最复杂
# 由于从0开始计数，从b_con_son中获得的son不需要减1；gama_num在设定时已经考虑了减1的问题，这里的索引不需要减1
# p_line和i_line仍然需要减1，因为，所有节点的索引都减了1，那么节点索引i再减1对应的还是相应的线路潮流和电流
""" for i in pv_bus:
    pre_son = np.argwhere(b_con_son[i,:] == 1)
    son = []
    for n in pre_son:
        son.append(list(n)[0].tolist())
    model.addConstrs((sum(p_line[m-1,j,t] for m in son) == (1-gama[int(gama_num[i]),j,t])*p_pv[int(gama_num[i]),j,t]\
        + p_dch[int(gama_num[i]),j,t] - p_ch[int(gama_num[i]),j,t] - p_load[i,j,t] + p_line[i-1,j,t]\
            - i_line[i-1,j,t]*line_r[i] for j in range(N) for t in range(24)), name = 'c3')
    model.addConstrs((sum(q_line[m-1,j,t] for m in son) == (1-gama[int(gama_num[i]),j,t])*q_pv[int(gama_num[i]),j,t]\
        - q_load[i,j,t] + q_line[i-1,j,t] - i_line[i-1,j,t]*line_x[i] for j in range(N)\
            for t in range(24)), name = 'c4') """


print('7')

# 功率平衡方程，考虑PV和ESS的选择性
pv_count = 0
ess_count = 0
for i in range(1,Busnum):
    if i not in end_bus:
        
        pre_son = np.argwhere(b_con_son[i,:] == 1)
        son = []
        for n in pre_son:
            son.append(list(n)[0].tolist())
        
        if (i in pv_bus) and (i in ess_bus):
            model.addConstrs((sum(p_line[m-1,j,t] for m in son) ==    
            power_pv[pv_count,j,t] + p_dch[ess_count,j,t] - p_ch[ess_count,j,t] - p_load[i,j,t] + p_line[i-1,j,t] -i_line[i-1,j,t]*line_r[i] for j in range(N) for t in range(24)), name = 'c5')
            
            model.addConstrs((sum(q_line[m-1,j,t] for m in son) == q_ess[ess_count,j,t] + q_pv[pv_count,j,t] - q_load[i,j,t] + q_line[i-1,j,t] - i_line[i-1,j,t]*line_x[i] for j in range(N) for t in range(24)),name = 'c6')

            pv_count = pv_count + 1
            ess_count = ess_count + 1

        elif (i in pv_bus) and (i not in ess_bus):
            model.addConstrs((sum(p_line[m-1,j,t] for m in son) ==    
            power_pv[pv_count,j,t] - p_load[i,j,t] + p_line[i-1,j,t] -i_line[i-1,j,t]*line_r[i] for j in range(N) for t in range(24)), name = 'c5')
            
            model.addConstrs((sum(q_line[m-1,j,t] for m in son) == q_pv[pv_count,j,t] - q_load[i,j,t] + q_line[i-1,j,t] - i_line[i-1,j,t]*line_x[i] for j in range(N) for t in range(24)),name = 'c6')

            pv_count = pv_count + 1

        elif (i in ess_bus) and (i not in pv_bus):
            model.addConstrs((sum(p_line[m-1,j,t] for m in son) == p_dch[ess_count,j,t] - p_ch[ess_count,j,t] - p_load[i,j,t] + p_line[i-1,j,t] -i_line[i-1,j,t]*line_r[i] for j in range(N) for t in range(24)), name = 'c5')
            
            model.addConstrs((sum(q_line[m-1,j,t] for m in son) == q_ess[ess_count,j,t] - q_load[i,j,t] + q_line[i-1,j,t] - i_line[i-1,j,t]*line_x[i] for j in range(N) for t in range(24)),name = 'c6')

            ess_count = ess_count + 1

        else:
            model.addConstrs((sum(p_line[m-1,j,t] for m in son) == - p_load[i,j,t] + p_line[i-1,j,t] -i_line[i-1,j,t]*line_r[i] for j in range(N) for t in range(24)), name = 'c5')
            
            model.addConstrs((sum(q_line[m-1,j,t] for m in son) == - q_load[i,j,t] + q_line[i-1,j,t] - i_line[i-1,j,t]*line_x[i] for j in range(N) for t in range(24)),name = 'c6')

    else:
        # there is no son buses for ending buses
        if (i in pv_bus) and (i in ess_bus):
            model.addConstrs((power_pv[pv_count,j,t] + p_dch[ess_count,j,t] - p_ch[ess_count,j,t] - p_load[i,j,t] + p_line[i-1,j,t] -i_line[i-1,j,t]*line_r[i] == 0 for j in range(N) for t in range(24)), name = 'c5')
            
            model.addConstrs((q_ess[ess_count,j,t] + q_pv[pv_count,j,t] - q_load[i,j,t] + q_line[i-1,j,t] - i_line[i-1,j,t]*line_x[i] == 0 for j in range(N) for t in range(24)),name = 'c6')

            pv_count = pv_count + 1
            ess_count = ess_count + 1

        elif (i in pv_bus) and (i not in ess_bus):
            model.addConstrs((power_pv[pv_count,j,t] - p_load[i,j,t] + p_line[i-1,j,t] -i_line[i-1,j,t]*line_r[i] == 0 for j in range(N) for t in range(24)), name = 'c5')
            
            model.addConstrs((q_pv[pv_count,j,t] - q_load[i,j,t] + q_line[i-1,j,t] - i_line[i-1,j,t]*line_x[i] == 0 for j in range(N) for t in range(24)),name = 'c6')

            pv_count = pv_count + 1

        elif (i in ess_bus) and (i not in pv_bus):
            model.addConstrs((p_dch[ess_count,j,t] - p_ch[ess_count,j,t] - p_load[i,j,t] + p_line[i-1,j,t] -i_line[i-1,j,t]*line_r[i] == 0 for j in range(N) for t in range(24)), name = 'c5')
            
            model.addConstrs((q_ess[ess_count,j,t] - q_load[i,j,t] + q_line[i-1,j,t] - i_line[i-1,j,t]*line_x[i] == 0 for j in range(N) for t in range(24)),name = 'c6')

            ess_count = ess_count + 1

        else:
            model.addConstrs((- p_load[i,j,t] + p_line[i-1,j,t] -i_line[i-1,j,t]*line_r[i] == 0 for j in range(N) for t in range(24)), name = 'c5')
            
            model.addConstrs((- q_load[i,j,t] + q_line[i-1,j,t] - i_line[i-1,j,t]*line_x[i] == 0 for j in range(N) for t in range(24)),name = 'c6')

print('8')

# the voltage constraints
for i in np.arange(1,Busnum):
    model.addConstrs((u_bus[i,j,t] == u_bus[i_fa[i]-1,j,t] - 2*(p_line[i-1,j,t]*line_r[i] + q_line[i-1,j,t]*line_x[i]) + i_line[i-1,j,t]*(np.square(line_r[i])+np.square(line_x[i])) for j in range(N) for t in range(24)), name = 'c9')

# 节点0是OLTC，其余节点电压约束
model.addConstrs((u_bus[i,j,t] >= Vmin for i in range(1,Busnum) for j in range(N) for t in range(24)), name = 'c10')
model.addConstrs((u_bus[i,j,t] <= Vmax for i in range(1,Busnum) for j in range(N) for t in range(24)), name = 'c11')
# OLTC电压约束
model.addConstrs((u_bus[0,j,t] >= tap_posi[i,j,t]**2 - (1-tap_k[i,j,t])*M for i in range(len(tap_opt)) for j in range(N) for t in range(24)), name = 'c111')
model.addConstrs((u_bus[0,j,t] <= tap_posi[i,j,t]**2 + (1-tap_k[i,j,t])*M for i in range(len(tap_opt)) for j in range(N) for t in range(24)), name = 'c112')

model.addConstrs((tap_k[:,j,t].sum() == 1 for j in range(N) for t in range(24)), name = 'C113')

print('9')

# the constraint on power of current and voltage
# 使用二阶锥形式
for i in np.arange(1,Busnum):
    for j in range(N):
        for t in range(24):
            model.addConstr((p_line[i-1,j,t]*p_line[i-1,j,t] + q_line[i-1,j,t]*q_line[i-1,j,t] - u_bus[i_fa[i]-1,j,t]*i_line[i-1,j,t] <= 0), name = 'c12')

print('10')

print('11')

# the constraint on line current
model.addConstrs(i_line[i,j,t] >= 0 for i in range(Linenum) for j in range(N)\
    for t in range(24))
model.addConstrs(i_line[i,j,t] <= i_max for i in range(Linenum) for j in \
    range(N) for t in range(24))

print('12')

print('13')

# the voltage deviation relaxation
# v_1, v_2 >= 0是gurobi自动满足的，默认值
model.addConstrs(v_1[i,j,t] - v_2[i,j,t] == u_bus[i,j,t] - v_norm for i in range(Busnum) for j in range(N) for t in range(24))
model.addConstrs(v_1[i,j,t] >= 0 for i in range(Busnum) for j in range(N) for t in range(24))
model.addConstrs(v_2[i,j,t] >= 0 for i in range(Busnum) for j in range(N) for t in range(24))

# the constraints on energy and power capacity of ESS
model.addConstrs((E_ess[i,0] <= ess_w[i,0]*ESS_cap_max for i in range(ess_num)),name='c0805 1')
model.addConstrs(E_ess[i,0] >= 0 for i in range(ess_num))

model.addConstrs(p_ess[i,0] <= ess_w[i,0]*inv_max for i in range(ess_num))
model.addConstrs(p_ess[i,0] >= 0 for i in range(ess_num))

# PV capacity and power
model.addConstrs(pv_cap[i,0] <= P_pv_max for i in range(pv_num))
model.addConstrs(pv_cap[i,0] >= 0 for i in range(pv_num))

model.addConstrs(power_pv[i,j,t] <= pv_cap[i,0]*p_pv[i,j,t] for i in range(pv_num) for j in range(N) for t in range(24))
model.addConstrs(power_pv[i,j,t] >= 0 for i in range(pv_num) for j in range(N) for t in range(24))

# power to energy ratio constraints
model.addConstrs(E_ess[i,0] >= hour_min * p_ess[i,0] for i in range(ess_num))
model.addConstrs(E_ess[i,0] <= hour_max * p_ess[i,0] for i in range(ess_num))

# power capacity constraints of ESS
model.addConstrs(p_ch[i,j,t]*p_ch[i,j,t] + q_ess[i,j,t]*q_ess[i,j,t] <= p_ess[i,0]*p_ess[i,0] for i in range(ess_num) for j in range(N) for t in range(24))
model.addConstrs(p_dch[i,j,t]*p_dch[i,j,t] + q_ess[i,j,t]*q_ess[i,j,t] <= p_ess[i,0]*p_ess[i,0] for i in range(ess_num) for j in range(N) for t in range(24))

# EENS
model.addConstr(fail_line/N*abs_p_line[:,:,:].sum() +  fail_ess/N*(p_ch[:,:,:] + p_dch[:,:,:]).sum() + fail_pv/N*power_pv[:,:,:].sum() <= fail_ene/N * p_load[:,:,:].sum())
model.addConstrs(abs_p_line[i,j,t]>=p_line[i,j,t] for i in range(Linenum) for j in range(N) for t in range(24))
model.addConstrs(abs_p_line[i,j,t]>=-p_line[i,j,t] for i in range(Linenum) for j in range(N) for t in range(24))

# 碳排放量
F1 = life_PV * pv_cap.sum() + life_ESS * E_ess.sum() + life_ESS_p*p_ess.sum() + sum(co2_ess_op*(p_ch[i,j,t]+p_dch[i,j,t]) for i in range(ess_num) for j in range(N) for t in range(24))/N*year*life + sum((p_sub_b[0,:,t].sum())*co2[t] for t in range(24))/N*year*life

# 经济成本
os_sum = 1/N*sum(sum(p_sub_b[0,j,t]*c_e_b[0,0,t] for t in range(24)) + sum(p_ch[i,j,t]+p_dch[i,j,t] for i in range(ess_num) for t in range(24))*c_ess_op + sum(pv_cap[i,0]*p_pv[i,j,t]-power_pv[i,j,t] for i in range(pv_num) for t in range(24))*c_pv_cut for j in range(N))
in_rate = (interest*(1+interest)**life)/((1+interest)**life-1)

F2 = (pv_cap.sum()*c_pv + E_ess.sum()*c_ess + p_ess.sum()*c_ess_inv)*in_rate + year*os_sum

# 电压偏移
F3 = (v_1.sum()+v_2.sum())/N

# # the e-constraints
# model.addConstr((F1 <= FU_1 + w1*(FN_1-FU_1)), name='c13')
# model.addConstr((F3 <= FU_3 + w2*(FN_3-FU_3)), name='c14')



if obj_index == 1:
    # obj1: the objective of co2 generation
    model.setObjective(F1)
elif obj_index == 2:
    # obj2: the economic cost
    model.setObjective(F2)
elif obj_index == 3:
    # obj3: the voltage deviation
    model.setObjective(F3)
else:
    # obj1 balance with obj2
    model.setObjective(F2)

#model.Params.DualReductions = 1

model.setParam('MIPGap', 0.05)
# model.setParam('NoRelHeurTime', 5000)

model.optimize()

if model.Status == 2:
    print('Get the sulotion!!!')

    filename = 'result_obj'+str(obj_index)+'_w1_'+str(w1)+'_w2_'+str(w2)+'_pv'+str(P_pv_max)+'.npz'

    np.savez(filename,
             omig_ess_ch = omig_ess_ch.X,
             omig_ess_dch = omig_ess_dch.X,
             p_sub_b = p_sub_b.X,
             q_sub = q_sub.X,
             p_ch = p_ch.X,
             p_dch = p_dch.X,
             u_bus = u_bus.X,
             p_line = p_line.X,
             q_line = q_line.X,
             i_line = i_line.X,
             ess_w = ess_w.X,
             q_pv = q_pv.X,
             v_1 = v_1.X,
             v_2 = v_2.X,
             tap_k = tap_k.X,
             E_ess = E_ess.X,
             p_ess = p_ess.X,
             ch_ess = ch_ess.X,
             dis_ess = dis_ess.X,
             pv_cap = pv_cap.X,
             power_pv = power_pv.X,
             q_ess = q_ess.X,
             abs_p_line = abs_p_line.X,
             objection = model.objval)

    # 碳排放量
    F1 = life_PV * pv_cap.X.sum() + life_ESS * E_ess.X.sum() + life_ESS_p*p_ess.X.sum() + sum(co2_ess_op*(p_ch.X[i,j,t]+p_dch.X[i,j,t]) for i in range(ess_num) for j in range(N) for t in range(24))/N*year*life + sum((p_sub_b.X[0,:,t].sum())*co2[t] for t in range(24))/N*year*life

    # 经济成本
    os_sum = 1/N*sum(sum(p_sub_b.X[0,j,t]*c_e_b[0,0,t] for t in range(24)) + sum(p_ch.X[i,j,t]+p_dch.X[i,j,t] for i in range(ess_num) for t in range(24))*c_ess_op + sum(pv_cap.X[i,0]*p_pv[i,j,t]-power_pv.X[i,j,t] for i in range(pv_num) for t in range(24))*c_pv_cut for j in range(N))
    in_rate = (interest*(1+interest)**life)/((1+interest)**life-1)

    F2 = (pv_cap.X.sum()*c_pv + E_ess.X.sum()*c_ess + p_ess.X.sum()*c_ess_inv)*in_rate + year*os_sum

    # 电压偏移
    F3 = (v_1.X.sum()+v_2.X.sum())/N

    print('F1 is: ', F1)
    print('F2 is: ', F2)
    print('F3 is: ', F3)

    easygui.msgbox('V11_Python succeeded!')
    

if model.Status == 3:
    filename = 'result_obj'+str(obj_index)+'_w1_'+str(w1)+'_w2_'+str(w2)+'_pv'+str(P_pv_max)+'_'
    model.computeIIS()
    model.write(filename+'.ilp')
    print('The result is in :'+'\n' + filename +'.ilp')

#     # easygui.msgbox('V11_Python succeeded!')
        



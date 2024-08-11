% 这个版本里面直接在程序编写的时候就限制好只有有限个节点安装光伏和储能
yita = 0.9; % the risk factor
Busnum = 33; % the bus numbers
N = 3; % 随机场景的数量
Linenum = 32; % IEEE33节点算例线路条数
P_pv = 10; % the capacity of PV and the unit is kW
ESS_cap = 20; % the energy capacity of ess on each bus, and the unit is kWh
Pmax = 3000; % the maximum active power in lines and the unit is kw
p_ess_max = 10; % the maximum charging and discharging power of ess, and the unit is kW
E_ess = [0.95 0.25]*ESS_cap; % the maximum and minimum energy capacity
E_ess_0 = ones(3,N)*ESS_cap*0.3; % the initial energy in the ess in all buses and scenarios
lamda_ch = 0.9; % the efficiency of charging
lamda_dch = 0.9; % the efficiency of discharging
S_sub = 5000; % the capacity of substaion and the unit is kVA. 节点上其他类型的注入功率都是kW，这个也应该是kVA
Vmin = (12.66*0.9)^2; % the minimum bus voltage, and the unit is kV^2
Vmax = (12.66*1.1)^2; % the maximum bus voltage, and the unit is kV^2
b_con_son = load('b_con_son.mat').b_con_son; % the uptril matrix and the i th line contains the son buses of bus i 
b_con_fa = load('b_con_fa.mat').b_con_fa; % the downtril matrix and the i th line contains the father buses of bus i
b_r_fa = load('b_r_fa.mat').b_r_fa * 12.66^2/100; % the R in each line, in downtril matrix and the unit is ohmn
b_x_fa = load('b_x_fa.mat').b_x_fa * 12.66^2/100; % the X in each line, in downtril matrix and the unit is ohmn
line_r = load('line_r.mat').line_r; % the i th element is the R in the line where bus i is its son bus
line_x = load('line_x.mat').line_x; % the i th element is the X in the line where bus i is its son bus
i_fa = load('i_fa.mat').i_fa; % the father bus of bus i
c_e_t = load('c_e_t.mat').c_e_t; % the electricity price at each hour and the unit is $/kWh
c_e_t = reshape(c_e_t,1,1,24);
c_ess_op = 0.00027; % the operation cost of ess unit is $/kWh
c_pv_cut = 0.033; % the punishment on pv deserting, and the unit is $/kWh
co2 = 0.779; % the co2 emission, and the unit is kg/kWh
c_pv = 1343*P_pv; % the installation cost of pv on each bus and it is 1343 $/kWh
c_ess = 304*ESS_cap; % the installation cost of ess. the unit is 304 $/kWh
i_max = 200; % the maximum I in a line and the unit is A^2
pv_bus = [8 20 28]; % 选定的PV节点
ess_bus = [8 20 28]; % 选定的储能节点
gama_num = zeros(Busnum,1); % 用来指示PV节点的位置

% 每个PV节点对应的gama变量分别是哪一个
num = 1;
for i = pv_bus
    gama_num(i) = num;
    num = num + 1;
end

p_load = load('p_load.mat').p_load(:,1:N,:); % 有名值 and represents the fluctuation
q_load = load('q_load.mat').q_load(:,1:N,:); % 有名值 and represents the fluctuation
p_pv = load('p_pv.mat').p_pv(1:3,1:N,:); % the value is in p.u. value and represents the fluctuation, the power factor of PV is 0.9，滞后
q_pv = load('q_pv.mat').q_pv(1:3,1:N,:); % the value is in p.u. value and represents the fluctuation, the power factor of PV is 0.9，滞后

p_pv = p_pv * P_pv; % transform into real value(有名值)
q_pv = q_pv * P_pv; % transform into real value(有名值),pv的无功功率也是注入


yike = sdpvar(N,1); % 辅助变量
kesai = sdpvar(1,1); % 辅助变量
% pv_ins = binvar(Busnum,1,1); % 各节点是否安装PV
% ess_ins = binvar(Busnum,1,1); % 各节点是否安装储能
omig_ess_ch = binvar(3,N,24); % 储能系统是否充电
omig_ess_dch = binvar(3,N,24); % 储能系统是否放电

p_sub = sdpvar(1,N,24); % the power output of substation, only installed at bus 1 
q_sub = sdpvar(1,N,24);
p_ch = sdpvar(3,N,24); % 每个节点在不同场景、不同时刻的储能充电功率
p_dch = sdpvar(3,N,24); % 每个节点在不同场景、不同时刻的储能放电功率
gama = sdpvar(3,N,24); % 每个节点的PV在不同场景、不同时刻的弃光率，考虑对光伏不存在的时刻设置为0
u_bus = sdpvar(Busnum,N,24); % the square of voltage at each bus

p_line = sdpvar(Linenum,N,24); % 每条线路在不同场景、不同时刻的有功大小，设行号为i,则第i条线路的子节点为i+1
q_line = sdpvar(Linenum,N,24); % 每条线路在不同场景、不同时刻的无功大小，设行号为i,则第i条线路的子节点为i+1
i_line = sdpvar(Linenum,N,24); % the square of current in each line


% the constraints
% c = [-Pmax <= p_line <= Pmax]; % the active power constraint in lines according to the thermal stability
% c = [c; -Qmax <= q_line <= Qmax]; % the reactive power constraint in lines
c = [omig_ess_ch + omig_ess_dch <= 1]; % there's only charging or discharging at one monent
% c = [0 <= p_ch <= (ess_ins.*omig_ess_ch)*p_ess_max]; % the charging constraint of ess, if there is no ess installed, the power has to be zero
% c = [c; 0 <= p_dch <= (ess_ins.*omig_ess_dch).*p_ess_max]; % the discharging constraint of ess
% c = [c; omig_ess_ch + omig_ess_dch <= 1]; % there's only charging or discharging at one monent
% c = [c; 0 <= 1-gama <= pv_ins.*ones(Busnum,N,24)]; % the constraint on pv cut rate


for i = 1:N
    for j = 1:24
        c = [c; 0 <= p_ch(:,i,j) <= omig_ess_ch(:,i,j)*p_ess_max]; % the charging constraint of ess, if there is no ess installed, the power has to be zero
        c = [c; 0 <= p_dch(:,i,j) <= omig_ess_dch(:,i,j)*p_ess_max]; % the discharging constraint of ess
        c = [c; 0 <= gama(:,i,j) <= 1]; % the constraint on pv cut rate. It has to be 1 when there is no PV installed and it is [0,1] when there is PV.
    end
end
disp('Mark 1')
% the energy constraints on ess, the middle in the constraint is a 2-d
% matrix
c = [c; E_ess(2) <= E_ess_0 + p_ch(:,:,1)*lamda_ch - p_dch(:,:,1)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:2),3)*lamda_ch - sum(p_dch(:,:,1:2),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:3),3)*lamda_ch - sum(p_dch(:,:,1:3),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:4),3)*lamda_ch - sum(p_dch(:,:,1:4),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:5),3)*lamda_ch - sum(p_dch(:,:,1:5),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:6),3)*lamda_ch - sum(p_dch(:,:,1:6),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:7),3)*lamda_ch - sum(p_dch(:,:,1:7),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:8),3)*lamda_ch - sum(p_dch(:,:,1:8),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:9),3)*lamda_ch - sum(p_dch(:,:,1:9),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:10),3)*lamda_ch - sum(p_dch(:,:,1:10),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:11),3)*lamda_ch - sum(p_dch(:,:,1:11),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:12),3)*lamda_ch - sum(p_dch(:,:,1:12),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:13),3)*lamda_ch - sum(p_dch(:,:,1:13),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:14),3)*lamda_ch - sum(p_dch(:,:,1:14),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:15),3)*lamda_ch - sum(p_dch(:,:,1:15),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:16),3)*lamda_ch - sum(p_dch(:,:,1:16),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:17),3)*lamda_ch - sum(p_dch(:,:,1:17),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:18),3)*lamda_ch - sum(p_dch(:,:,1:18),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:19),3)*lamda_ch - sum(p_dch(:,:,1:19),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:20),3)*lamda_ch - sum(p_dch(:,:,1:20),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:21),3)*lamda_ch - sum(p_dch(:,:,1:21),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:22),3)*lamda_ch - sum(p_dch(:,:,1:22),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess(2) <= E_ess_0 + sum(p_ch(:,:,1:23),3)*lamda_ch - sum(p_dch(:,:,1:23),3)/lamda_dch <= E_ess(1)];
c = [c; E_ess_0 + sum(p_ch(:,:,1:24),3)*lamda_ch - sum(p_dch(:,:,1:24),3)/lamda_dch == E_ess_0];

disp('Mark 2')

% c = [c; p_sub.*p_sub + q_sub.*q_sub <= S_sub^2]; % the power constraint of substation
% 使用正八边形线性化变电站功率约束
c = [c; -S_sub <= p_sub <= S_sub];
c = [c; -S_sub <= q_sub <= S_sub];
c = [c; -2^(0.5)*S_sub <= p_sub-q_sub <= 2^(0.5)*S_sub];
c = [c; -2^(0.5)*S_sub <= p_sub+q_sub <= 2^(0.5)*S_sub];

disp('Mark 3')

% the power balance constraint in each bus
% there is no father bus for bus 1
son = find(b_con_son(1,:) == 1); % get the son buses of bus 1
c = [c; p_line(son-1,:,:) == p_sub - p_load(1,:,:)];
c = [c; q_line(son-1,:,:) == q_sub - q_load(1,:,:)];

disp('Mark 4')

% PV节点和储能节点重合，他们的功率平衡方程最复杂
for i = pv_bus
    fa = find(b_con_fa(i,:) == 1); % get the father buses of bus i
    son = find(b_con_son(i,:) == 1); % get the son buses of bus i
    c = [c; sum(p_line(son-1,:,:),1) == (1-gama(gama_num(i),:,:)).*p_pv(gama_num(i),:,:) + p_dch(gama_num(i),:,:) - p_ch(gama_num(i),:,:) - p_load(i,:,:) + p_line(i-1,:,:) - i_line(i-1,:,:)*line_r(i)];
    c = [c; sum(q_line(son-1,:,:),1) == (1-gama(gama_num(i),:,:)).*q_pv(gama_num(i),:,:) - q_load(i,:,:) + q_line(i-1,:,:) - i_line(i-1,:,:)*line_x(i)];
end

% 排除掉PV、储能节点和PCC节点
for i = [2:7,9:19,21:27,29:33]
    if i ~= 18 && i ~= 22 && i ~= 25 && i ~= 33
        fa = find(b_con_fa(i,:) == 1); % get the father buses of bus i
        son = find(b_con_son(i,:) == 1); % get the son buses of bus i
        c = [c; sum(p_line(son-1,:,:),1) == - p_load(i,:,:) + p_line(i-1,:,:) - i_line(i-1,:,:)*line_r(i)];
        c = [c; sum(q_line(son-1,:,:),1) == - q_load(i,:,:) + q_line(i-1,:,:) - i_line(i-1,:,:)*line_x(i)];
    else
        % there is no son buses for ending buses
        fa = find(b_con_fa(i,:) == 1); % get the father buses of bus i
        c = [c; 0 == - p_load(i,:,:) + p_line(i-1,:,:) - i_line(i-1,:,:)*line_r(i)];
        c = [c; 0 == - q_load(i,:,:) + q_line(i-1,:,:) - i_line(i-1,:,:)*line_x(i)];
    end
end

disp('Mark 5')
% the voltage constraints
for i = 2:Busnum
    c = [c; u_bus(i,:,:) == u_bus(i_fa(i),:,:) - 2*(p_line(i-1,:,:)*line_r(i) + q_line(i-1,:,:)*line_x(i)) + i_line(i-1,:,:)*(line_r(i)^2+line_x(i)^2)];
end
c = [c; Vmin <= u_bus <= Vmax];

disp('Mark 6')

% the constraint on power of current and voltage
% 用等式代替二阶锥的表达式，因为YALMIP会自动将这样的等式转化为二阶锥的形式
% for i = 2:Busnum
%     c = [c; i_line(i-1,:,:).*u_bus(i_fa(i),:,:) == p_line(i-1,:,:).*p_line(i-1,:,:) + q_line(i-1,:,:).*q_line(i-1,:,:)];
% end
% 使用准确的二阶锥的形式
for i = 2:Busnum
    for j = 1:N
        for t = 1:24
%             c = [c;cone([2*p_line(i-1,j,t); 2*q_line(i-1,j,t); i_line(i-1,j,t)-u_bus(i_fa(i),j,t)],i_line(i-1,j,t)+u_bus(i_fa(i),j,t))];
            c = [c;norm([2*p_line(i-1,j,t); 2*q_line(i-1,j,t); i_line(i-1,j,t)-u_bus(i_fa(i),j,t)],2) <= i_line(i-1,j,t)+u_bus(i_fa(i),j,t)];
%             c = [c;(2*p_line(i-1,j,t))^2 + (2*q_line(i-1,j,t))^2 + (i_line(i-1,j,t)-u_bus(i_fa(i),j,t))^2 <= (i_line(i-1,j,t)+u_bus(i_fa(i),j,t))^2];
%             c = [c; (p_line(i-1,j,t)^2 + q_line(i-1,j,t)^2)/u_bus(i_fa(i),j,t) <= i_line(i-1,j,t)];
        end
    end
end

disp('Mark 7')

% the CVaR constraint
% c = [c; yike >= sum(p_sub.*c_e_t,3) + sum(sum(c_ess_op*(p_ch + p_dch),1),3) + sum(sum(c_pv_cut*(pv_ins.*gama.*p_pv),1),3) - kesai];
for i = 1:N
    c = [c; yike(i) >= sum(p_sub(1,i,:).*c_e_t,3) + sum(sum(c_ess_op*(p_ch(:,i,:) + p_dch(:,i,:)),1),3) + sum(sum(gama(:,i,:).*p_pv(:,i,:),3),1)*c_pv_cut - kesai];
end
c = [c; yike >= 0 ];

% c = [c; 0 <= i_line <= i_max];

disp('Mark 8')
% c = [c; gama(:,:,[1:6,20:24]) == 0]; % when there is no pv power, the
% gama can be set to 0


% the objectives
f1 = co2*sum(sum(sum(p_sub,1),2),3)*1/N; % the objective of co2 generation

f2 = kesai + 1/(1-yita)*sum(yike)*1/N;

options = sdpsettings('showprogress',1,'debug',1,'solver','gurobi');

result = optimize(c,f2,options);

% re_pv_ins = value(pv_ins)
% re_ess_ins = value(ess_ins);
re_p_ch = value(p_ch);
re_p_dch = value(p_dch);
re_p_line = value(p_line);
re_q_line = value(q_line);
re_i_line = value(i_line);
re_gama = value(gama);
re_u_bus = value(u_bus);


% 修改思路
% 检查约束条件是否正确
% 将储能的充放电整数约束去掉，改为充电功率可取正负值




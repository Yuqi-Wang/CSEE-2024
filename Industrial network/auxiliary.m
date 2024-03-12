%%
% 这个文件夹下采用标幺值进行计算
% 从case33bw.m文件中取出各节点的负荷数据
case_data = loadcase('case33bw.m'); % 读取case文件
load_data = case_data.bus(:,3:4)/100; % 第三、四列为负荷数据，第三列为有功，第四列为无功，单位为MW，除以100是为了换算为标幺值，100代表100MVA
                                         %
distload = load('load_trend.mat').load_trend; % 第一行为负荷有功的波动曲线，第二行为负荷无功的波动曲线, adapted from residential load in industry
distpv = load('pv_trend.mat').pv_trend; % the fluctuation feature of PV, adapted from summer in industry
Busnum = 33; % the number of buses
N = 100; % the number of scenarios
p_load = zeros(Busnum,N,24); % the active power fluction of load
q_load = zeros(Busnum,N,24); % the reactive power fluction of load
p_pv = zeros(Busnum,N,24); % the active power fluctuation of PV
q_pv = zeros(Busnum,N,24); % the reactive power fluctuation of PV
for i = 1:Busnum
    for s = 1:N
        for t = 1:24
            sta_1 = abs(distload(1,t) + 0.3 * distload(1,t) * randn);
            p_load(i,s,t) = sta_1 * load_data(i,1); % the active power and fluctuation of active power consists the active load
            q_load(i,s,t) = sta_1 * load_data(i,2); % the reactive power and fluctuation of reactive power consists the reactive load
            sta_2 = abs(distpv(1,t) + 0.3 * distpv(1,t) * randn);
            p_pv(i,s,t) = sta_2;
            q_pv(i,s,t) = sta_2 * 0.4843; % the power factor is set as 0.95, so the reactive power fluctuation is the same as the active
        end
    end
end

% the load value is 有名值, and the pv value is in p.u. value
save p_load p_load
save q_load q_load
save p_pv p_pv
save q_pv q_pv

%%
% get the connection matrix
Busnum = 33;
case_data = loadcase('case33bw.m'); % 读取case文件
branch_data = case_data.branch(:,1:4); % 
b_con = zeros(Busnum,Busnum);
b_r = zeros(Busnum,Busnum);
b_x = zeros(Busnum,Busnum);
for i = 1:size(branch_data,1)
    if i ~= 33 && i ~= 35 && i ~= 34 && i ~= 37 && i ~= 36
        b_con(branch_data(i,1),branch_data(i,2)) = 1;
        b_r(branch_data(i,1),branch_data(i,2)) = branch_data(i,3);
        b_x(branch_data(i,1),branch_data(i,2)) = branch_data(i,4);
    end
end
line_r = zeros(Busnum,1);
line_x = zeros(Busnum,1);
for i = 2:Busnum
    line_r(i) = sum(b_r(:,i)) * 12.66^2/100;
    line_x(i) = sum(b_x(:,i)) * 12.66^2/100;
end

%%
% find i_fa
i_fa = zeros(Busnum,1);
for i = 2:Busnum
    i_fa(i) = find(b_con_fa(i,:)==1);
end

%%
valley = 23.8/1000;
flat = 119/1000;
peak = 357/1000;
c_e_t = [ones(1,7)*valley ones(1,3)*flat ones(1,6)*peak ones(1,2)*flat ones(1,4)*peak ones(1,1)*flat ones(1,1)*valley]; % the unit is $/kWh
% for i = 1:24
%     if i == 11 || i == 12 || i == 12 || i == 13 || i == 14 || i == 15 
%         c_e_t(i) = 357/1000;
%     elseif 1<=i<=7 || i == 24
%         c_e_t(i) = 23.8/1000;
%     else
%         c_e_t(i) = 119/1000;
%     end
% end

%%
% 测试二阶锥约束
x = [1 2 3 4 5 6]';
t = (0:0.02:2*pi)';
Atrue = [sin(t) sin(2*t) sin(3*t) sin(4*t) sin(5*t) sin(6*t)];
ytrue = Atrue*x;
A = Atrue;
A(100:210,:)=.1;
y = ytrue+randn(length(ytrue),1);

xhat = sdpvar(6,1);
sdpvar u v

F = [cone(y-A*xhat,u+xhat(1)), cone(xhat,v+xhat(2))];
F = [F; xhat(1) == 2*xhat(2) + 1];
optimize(F,u + v);

% 测试高维矩阵的二阶锥约束规划
x_1 = sdpvar(4,3,2);
x_2 = sdpvar(4,3,2);
x_3 = binvar(4,3,2);
x_4 = binvar(4,3,2);

c = [x_1 + x_2 <= 1];
c = [c; cone([x_1(1,1,1);x_1(2,3,2);x_2(1,2,1)-x_1(3,2,1)], x_2(1,2,1)+x_1(3,2,1))];
c = [c; cone([x_1(4,2,1);x_1(3,1,2);x_2(4,2,1)-x_1(2,2,1)], x_2(4,2,1)+x_1(2,2,1))];
c = [c; cone([x_1(3,2,1);x_1(1,3,2);x_2(3,2,1)-x_1(1,2,1)], x_2(3,2,1)+x_1(1,2,1))];
c = [c; x_3 + x_4 <= 1];

obj = sum(sum(sum(-x_1-x_2+x_3,1),2),3);

result = optimize(c,obj);
re_x_1 = value(x_1);
re_x_1(:,1,1)

%%
% 这一部分用来生成毕业论文中的展示随机场景和电价的图
ex_pv = zeros(24,100);
ex_pload = zeros(24,100);
ex_qload = zeros(24,100);
for i = 1:24
    for j = 1:100
        ex_pv(i,j) = p_pv(3,j,i);
        ex_pload(i,j) = p_load(3,j,i);
        ex_qload(i,j) = q_load(3,j,i);
    end
end








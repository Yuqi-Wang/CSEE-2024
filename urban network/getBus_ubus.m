function bus = getBus_ubus(cnum,j,t,n)
% 计算图片中节点编号,根据节点电压变量的编号
bus = (cnum-24*j-(t+1)-23-24*(n-1))/(24*n)+1 + 1;
function [line,s,t] = getNum(N)
N = N + 1;
line = floor(N/2400)+1;
s = floor((N-(line-1)*2400)/24)+1;
t = N - (line-1)*2400 - (s-1)*24;

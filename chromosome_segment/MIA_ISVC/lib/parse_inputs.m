function [C,T_angle,sig,H,L,Endpoint,Gap_size] = parse_inputs(varargin)
Para=[1.5,162,3,0.35,0,1,1]; %Default experience value;%
C=Para(1); %coefficent c=1 no corner is removed
T_angle=Para(2); % T angel 160:200
sig=Para(3); % sigma
H=Para(4); 
L=Para(5);
Endpoint=Para(6);
Gap_size=Para(7);

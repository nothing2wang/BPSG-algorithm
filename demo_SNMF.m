%       SNMF experiment on ORL data     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
Data = load('ORL64');
A  = Data.fea;
A = A';
[n, d] = size(A);
% normalize the data as suggested
for i = 1:d 
    A(:, i) = A(:, i)./norm(A(:, i));   
end
y = A;
maxNumCompThreads(1);
[n, d] = size(y);
% minibatch subsampling ratio = 1/sr = 1/20 = 5%
sr = 20;
n_epochs = 100; % number of total epochs
tau = round(n/2);% sparsity constraint
% number of basis image to be extracted
r  = 25;
 load('init_snmf_orl_s');
% BPSG-SGD
  [ Aout3, xt3, error3, time3 ] = SNMF_BPSG_SGD_e(y,sr,n_epochs, tau,  r, Ain, xin);
% BPSG-SAGA
  [ Aout4, xt4, error4, time4 ] = SNMF_BPSG_SAGA_e(y,sr,n_epochs, tau, r, Ain, xin);

% BPSG-SARAH
  [ Aout5, xt5, error5, time5 ] = SNMF_BPSG_SARAH_e(y,sr,n_epochs/2, tau,  r, Ain, xin);

bound = 7777;%
%%
linewidth = 1;
axesFontSize = 6;
labelFontSize = 11;
legendFontSize = 8;
resolution = 108; % output resolution
output_size = resolution *[12, 12]; % output size

%%%%%% %%%%%% %%%%%% %%%%%% %%%%%%
figure(101), clf;
set(gca,'DefaultTextFontSize',18);
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.175 -0.0 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[1.15 0.6]);
p3 = plot(0:1:n_epochs,min(bound,log10(error3(1:end))), 's-','LineWidth',1.5,'color', [1,0,1], 'MarkerIndices', 1:20:100,'MarkerSize',10);
hold on
p4 = plot(0:1:n_epochs,min(bound,log10(error4(1:end))), 'o-','LineWidth',1.5,'color', [1,0,0], 'MarkerIndices', 1:20:100,'MarkerSize',10);
hold on
p5 = plot(0:2:n_epochs,min(bound,log10(error5(1:end))), 'd-','LineWidth',1.5,'color', [0,0,1], 'MarkerIndices', 1:10:50,'MarkerSize',8);
hold off
set(gca,'FontSize', 12);
grid on;
lg = legend([p3, p4, p5], 'BPSG-SG','BPSG-SAGA', 'BPSG-SARAH', 'NumColumns',1);
legend('boxoff');
set(lg, 'Location', 'NorthEast');
set(lg, 'FontSize', 12);
ylb = ylabel({'$\mathrm{log}(\Phi(U_k, V_k))$'},'FontAngle', 'normal', 'Interpreter', 'latex', 'FontSize', 16);
set(ylb, 'Units', 'Normalized', 'Position', [-0.08, 0.5, 0]);
xlb = xlabel({'$\#~ of~epochs$'}, 'FontSize', 14,'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.07, 0]);
%%%%%% %%%%%% %%%%%% %%%%%% %%%%%%
figure(102);clf;
set(gca,'DefaultTextFontSize',18);
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.175 -0.0 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[1.15 0.6]);
p3 = plot([0; time3],min(bound,log10(error3(1:end))), 's-','LineWidth',1.5,'color', [1,0,1], 'MarkerIndices', 1:20:100,'MarkerSize',10);
hold on
p4 = plot([0; time4],min(bound,log10(error4(1:end))), 'o-','LineWidth',1.5,'color', [1,0,0], 'MarkerIndices', 1:20:100,'MarkerSize',10);
hold on
p5 = plot([0; time5],min(bound,log10(error5(1:end))), 'd-','LineWidth',1.5,'color', [0,0,1], 'MarkerIndices', 1:10:50,'MarkerSize',8);
set(gca,'FontSize', 12);
grid on;
lg = legend([p3, p4, p5], 'BPSG-SG','BPSG-SAGA', 'BPSG-SARAH', 'NumColumns',1);
legend('boxoff');
set(lg, 'Location', 'NorthEast');
set(lg, 'FontSize', 12);

ylb = ylabel({'$\mathrm{log}(\Phi(U_k, V_k))$'},...
    'FontAngle', 'normal', 'Interpreter', 'latex', 'FontSize', 16);
set(ylb, 'Units', 'Normalized', 'Position', [-0.08, 0.5, 0]);
xlb = xlabel({'$time~(s)$'}, 'FontSize', 14,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.07, 0]);






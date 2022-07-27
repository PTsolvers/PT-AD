clear
fileID1 = fopen('../out_visu/gpu_timing.txt','r');
fileID2 = fopen('../out_visu/cpu_timing.txt','r');
formatSpec = '%d %d %f %f %f %f %f %f %f %f %f %f';
% nx  niter  time_synt  time_fwd  time_adj time/iter_synt time/iter_fwd time/iter_adj Teff_synt Teff_fwd Teff_adj cost_fun_grd_eval_sec
sizeA = [12 Inf];
GPU     = fscanf(fileID1,formatSpec,sizeA);
CPU     = fscanf(fileID2,formatSpec,sizeA);
nx_g        = GPU(1,:);
t_it_synt_g = GPU(6,:);
t_it_fwd_g  = GPU(7,:);
t_it_adj_g  = GPU(8,:);
Teff_synt_g = GPU(9,:);
Teff_fwd_g  = GPU(10,:);
Teff_adj_g  = GPU(11,:);
grd_eval_g  = GPU(12,:);

nx_c        = CPU(1,:);
t_it_synt_c = CPU(6,:);
t_it_fwd_c  = CPU(7,:);
t_it_adj_c  = CPU(8,:);
Teff_synt_c = CPU(9,:);
Teff_fwd_c  = CPU(10,:);
Teff_adj_c  = CPU(11,:);
grd_eval_c  = CPU(12,:);

Teff_fwd_g(5) = [];
Teff_adj_g(5) = [];
nx_g2 = nx_g;
nx_g2(5) = [];

% figure(1),clf
% subplot(121)
% semilogx(nx_g2,Teff_fwd_g,'-o',nx_g2,Teff_adj_g,'-o','Linewidth',1.5);axis square
% xlabel('n_x');ylabel('T_{eff} [GB/s]')
% legend({'forward','adjoint'},'Location','northwest')
% subplot(122)
% loglog(nx_g,t_it_fwd_g,'-o',nx_g,t_it_adj_g,'-o','Linewidth',1.5);axis square
% legend({'forward','adjoint'},'Location','northwest')
% xlabel('n_x');ylabel('time/it [s]')
% exportgraphics(gcf,'Teff_timeit.png','Resolution',300)
% 
% figure(2),clf
% loglog(nx_g,grd_eval_g,'-o','Linewidth',3)
% legend('cost fun')
% 
% figure(3),clf
% subplot(121)
% semilogx(nx_c,Teff_fwd_c,'-o',nx_c,Teff_adj_c,'-o','Linewidth',3)
% legend('T_{eff} forward','T_{eff} adjoint')
% subplot(122)
% loglog(nx_c,t_it_fwd_c,'-o',nx_c,t_it_adj_c,'-o','Linewidth',3)
% legend('time/it forward','time/it adjoint')
% 
% figure(4),clf
% loglog(nx_c,grd_eval_c,'-o','Linewidth',3)
% legend('cost fun')

figure(5),clf
loglog(nx_g,t_it_fwd_g,'-o',nx_c,t_it_fwd_c,'-o','Linewidth',1.5);axis square
ylabel('time/it forward [s]');xlabel('n_x')
legend('GPU','CPU')
exportgraphics(gcf,'timeit_gpu_cpu.png','Resolution',300)

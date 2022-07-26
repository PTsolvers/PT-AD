clear;figure(1);clf;colormap parula
simdir = '../out_visu/run2';
load([simdir '/static.mat'])
tiledlayout(1,2,'TileSpacing','compact','Padding','compact')
S_obs = B + H_obs; S_obs(H_obs==0)=NaN;
for gd_iter = 1:20
    load([simdir '/step_' int2str(gd_iter) '.mat'])
    S = B + H; S(H==0)=NaN;
    nexttile(1);contourf(S'    ,0:0.5:7);axis image;caxis([0 6]);xlim([0 230])
    hold on; contour(H_obs',[0.02 0.02],'LineWidth',1.5,'Color','r','LineStyle','--');hold off
    title('\rm\itH');legend('','observations')
    xticklabels([]);yticklabels([])
    nexttile(2);semilogy(iter_evo,J_evo,'-x','LineWidth',1);grid on
    xlabel('# iter');ylabel('\itJ')
    drawnow
    exportgraphics(gcf,sprintf('%s/anim/frame_%04d.tiff',simdir,gd_iter),'Resolution',300)
end
fps = 4;
system(sprintf('ffmpeg -framerate %d -i %s/anim/frame_%%04d.tiff -c libx264 -pix_fmt yuv420p -r %d %s/anim.mp4',fps,simdir,fps,simdir))
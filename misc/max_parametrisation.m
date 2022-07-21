clear;figure(1);clf
c     = 0.25;
w     = 0.05;
b0    = linspace(-1,1,201);
b_ref = min(b0,c);
transition1 =   -0.5*(tanh((b0-c)/w)-1);
transition2 =  1+0.5*(tanh((b0-c)/w)-1);
b_reg = b0.*transition1 + transition2*c;
plot(b0,b_ref,'b',b0,b_reg,'r--','LineWidth',2)
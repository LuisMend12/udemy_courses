% Same code as IVP_test, different example
% Stiff problem

% Setup: solve y' = f(t,y(t)), y(0) = y0, t <= T, with
f = @(t,y)(-10*t*y);
y0 = 2;
T = 2;                                       % 2 or 10

% Exact solution
yexact = @(t)y0*exp(-5*t.^2);
hfine = 1e-3;
tfine = 0:hfine:T;
yfine = yexact(tfine);
figure(1); clf;
plot(tfine,yfine,'r-','linewidth',2)
axis([0 T -.5 2]);
set(gca,'fontsize',16);

%% Numerical solution: Forward Euler (FE)

for h = [.2 .1 .05]
    t = 0:h:T;
    nsteps = length(t)-1;
    yFE = [y0; zeros(nsteps, 1)];    
    for n = 1:nsteps
        yFE(n+1) = yFE(n) + h*f(t(n),yFE(n));        
    end
    figure(1); hold on;
    plot(t,yFE,'b-x','linewidth',1);
axis([0 T -.5 2]);
end
xlabel('t'); ylabel('y'); title('Forward Euler')
legend('Exact','h = .2', 'h = .1', 'h = .05');

%% Numerical solution: Runge-Kutta 2 (RK)

figure(1); clf;
plot(tfine,yfine,'r-','linewidth',2)
set(gca,'fontsize',16);
for h = [.2 .1 .05]
    t = 0:h:T;
    nsteps = length(t)-1;
    yRK = [y0; zeros(nsteps, 1)];    
    for n = 1:nsteps
        yFE = yRK(n) + h*f(t(n),yRK(n));
        yRK(n+1) = yRK(n) + h/2*...
            ( f(t(n),yRK(n)) + f(t(n+1), yFE) );        
    end
    figure(1); hold on;
    plot(t,yRK,'b-x','linewidth',1);
    axis([0 T -.5 2]);
end
xlabel('t'); ylabel('y'); title('Runge-Kutta 2')
legend('Exact','h = .2', 'h = .1', 'h = .05');

%% Numerical solution: Backward Euler (BE)

figure(1); clf;
plot(tfine,yfine,'r-','linewidth',2)
set(gca,'fontsize',16);
for h = [.2 .1 .05]
    t = 0:h:T;
    nsteps = length(t)-1;
    yBE = [y0; zeros(nsteps, 1)];    
    for n = 1:nsteps
        F = @(x)(x - yBE(n) - h*f(t(n+1),x));        
        yBE(n+1) = fzero(F,yBE(n));
    end
    figure(1); hold on;
    plot(t,yBE,'b-x','linewidth',1);
    axis([0 T -.5 2]);
end
xlabel('t'); ylabel('y'); title('Backward Euler')
legend('Exact','h = .2','h = .1','h = .05');

%% Numerical solution: Midpoint method (MM)

figure(1); clf;
plot(tfine,yfine,'r-','linewidth',2)
set(gca,'fontsize',16);
for h = [.2 .1 .05]
    t = 0:h:T;
    nsteps = length(t)-1;
    yMM = [y0; zeros(nsteps, 1)];    
    for n = 1:nsteps
        F = @(x)(x - yMM(n) - h/2*...
            ( f(t(n),yMM(n)) + f(t(n+1),x)) );        
        yMM(n+1) = fzero(F,yMM(n));
    end
    figure(1); hold on;
    plot(t,yMM,'b-x','linewidth',1);
    axis([0 T -.5 2]);
end
xlabel('t'); ylabel('y'); title('Midpoint method')
legend('Exact','h = .2', 'h = .1','h = .05');


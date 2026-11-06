n = 20;
x = [0, sort(rand(n-1,1))', 1];
f = @(x)real(1./(x-(.3+.1*1i))./(x-(.6-.2*1i)));
figure(1); clf;
plot(x,f(x),'rx','linewidth',2)
set(gca,'fontsize',16)

%%
h = x(2:n+1) - x(1:n);   % h_1 through h_N
inth = h(2:n-1);            % h_2 through h_N-1
sumh = h(1:n-1) + h(2:n);
A = diag(2*sumh) + diag(inth,-1) + diag(inth,1);
ff = f(x);

if (1),     % perturbation test
    ff(end-2) = 10;
    figure(1); clf; plot(x,ff,'rx','linewidth',2)
end

df = (ff(2:n+1) - ff(1:n))./h;       
b = 6*(df(2:n) - df(1:n-1))';
sigma = A \ b;

alpha = ff(2:n+1)./h - h.*[sigma' 0]/6;
beta = ff(1:n)./h - h.*[0 sigma']/6;

dx = 1/1e4;
xx = dx:dx:1-dx;
s = zeros(length(xx),1);
sigma = [0 sigma' 0];   % sigma(j+1) is old sigma(j)
for k = 1:length(xx)
    j = sum(xx(k)>x);
    hj = h(j);
    s(k) = (x(j+1)-xx(k))^3/(6*hj)*sigma(j) + ...
        (xx(k)-x(j))^3/(6*hj)*sigma(j+1) + ...
        alpha(j)*(xx(k)-x(j)) + beta(j)*(x(j+1)-xx(k));
end
figure(1); hold on;
plot(xx,s,'b-','linewidth',1);

%% 
plot(xx,f(xx),'r-','linewidth',1);



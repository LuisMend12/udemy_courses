f =@(x)(1./(1+(4*x).^2));
N = 8;
x = (-1:2/N:1)';
y = f(x);
Nfine = 500;
xfine = (-1.2:2/Nfine:1.2);
yfine = f(xfine);

V = zeros(N+1);
for k = 0:N
    V(:,k+1) = x.^k;
end
a = V \ y;
p = zeros(size(xfine));
for k = 0:N
    p = p + a(k+1)*(xfine.^k);
end
p = min(p,2);
p = max(p,-1);

figure(1); clf;
plot(xfine,yfine,'b-','linewidth',1);
hold on;
plot(x,y,'rx','linewidth',2);
plot(xfine,p,'g-','linewidth',2);
set(gca,'fontsize',16)
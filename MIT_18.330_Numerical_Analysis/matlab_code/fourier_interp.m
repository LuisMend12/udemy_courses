N = 20;         % even
h = 2*pi/N;
x = 0:h:(2*pi-h);
k = -N/2:(N/2-1);
u = @(x)exp(sin(x));
v = u(x);
V = fftshift(fft(v));

Nfine = 1000;
hfine = 2*pi/Nfine;
xfine = 0:hfine:(2*pi-hfine);
P = [V(1)/2 V(2:end) V(1)/2];
P = [zeros(1,(Nfine-N)/2) P zeros(1,(Nfine-N)/2-1)];
p = Nfine/N*ifft(ifftshift(P));

err = u(xfine) - p;
max(abs(err))

figure(1); clf;
plot(xfine,u(xfine),'b-'); hold on;
plot(x,v,'bx');
plot(xfine,p,'r-');


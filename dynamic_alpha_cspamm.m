close all;
clear all;
clc;
dt = 0.1
T1 = 0.85
TR = 1
M0 = 1
MSS = 1
TAG = 1

n = 10
alpha = zeros(1,n);


for i = 0:n-1
    C = exp(-dt/T1);
    j = 0:i;
    denominator = sqrt(sum(C.^(2*j)));
    numerator = C^i;
    alpha(n-i+1)= asin(numerator/denominator);
end

alpha_deg = alpha*180/pi
tk = (1:n)*dt;
cosines = cos(alpha);
products = [];

for k = 1:(n)
    selection = cosines(1:k);
    result = cumprod(selection);
    products(k)= result(k);
end

I_opt = MSS*TAG*(exp(-tk/T1).*products).*sin(alpha(2:(n+1)));
plot(tk,I_opt,"-or")
hold on;

alpha = alpha*0 + alpha(2);
cosines = cos(alpha);
products = []

for k = 1:(n)
    selection = cosines(1:k);
    result = cumprod(selection);
    products(k)= result(k);
end

I_opt = MSS*TAG*(exp(-tk/T1).*products).*sin(alpha(2:(n+1)));
plot(tk,I_opt,"-ob")
hold on;

alpha = alpha*0 + 45*pi/180;
cosines = cos(alpha);
products = [];

for k = 1:(n)
    selection = cosines(1:k);
    result = cumprod(selection);
    products(k)= result(k);
end

I_opt = MSS*TAG*(exp(-tk/T1).*products).*sin(alpha(1:(n)));
plot(tk,I_opt,"-ok")
hold on;


title("Flip Angle Pulse Sequences vs. Time of Successive Images")
legend("Optimized, Varying Flip Angle, \alpha_{1}=9.6^\circ, \alpha_{10}=90^\circ","Constant Flip Angle, \alpha_{1-10}=9.6^\circ","Typical Flip Angle, \alpha_{1-10}=45^\circ")
xlabel("Elapsed Time")
ylabel("Normalized Transverse Magnetization (M_{x,y})")

has context menu
Compose

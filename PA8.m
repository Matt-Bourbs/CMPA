function [] = PA8()

% Part 1
Is = 0.01E-12;
Ib = 0.1E-12;
Vb = 1.3;
Gp = 0.1;

V = linspace(-1.95, 0.7, 200);
I = Is .* (exp((1.2/0.025).*V) - 1) + Gp.*V - Ib.*(exp((-1.2/0.025).*(V + Vb))-1);
noise = (rand(1, 200)-0.5)./5;
noise = I .* noise;
I_noise = I + noise;

figure(1)
plot(V, I_noise)
title('Noisy Plot with Polyfit')
figure(2)
semilogy(V, abs(I_noise))
title('Semilogy With Polyfit')

% Part 2
p4 = polyfit(V, I_noise, 4);
I4 = polyval(p4, V);
p8 = polyfit(V, I_noise, 8);
I8 = polyval(p8, V);

figure(1)
hold on 
plot(V, I4)
plot(V, I8)
figure(2)
hold on 
semilogy(V, abs(I4))
semilogy(V, abs(I8))

% Part 3
fo1 = fittype('A.*(exp(1.2*x/25e-3)-1)+0.1.*x-C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff1 = fit(transpose(V),transpose(I),fo1);
If1 = ff1(V);

figure(3)
plot(V, If1);
title('Non-linear Fit')
hold on
figure(4)
semilogy(V, abs(If1));
title('Semilogy Non-linear Fit')
hold on

fo2 = fittype('A.*(exp(1.2*x/25e-3)-1)+B.*x-C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff2 = fit(transpose(V),transpose(I),fo2);
If2 = ff2(V);

figure(3)
plot(V, If2);
hold on
figure(4)
semilogy(V, abs(If2));
hold on

fo3 = fittype('A.*(exp(1.2*x/25e-3)-1)+B.*x-C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff3 = fit(transpose(V),transpose(I),fo3);
If3 = ff3(V);

figure(3)
plot(V, If3);
hold on
figure(4)
semilogy(V, abs(If3));
hold on

% Part 4
inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net)
Inn = outputs;

figure(5)
plot(V, Inn);
title('Neural Net Fit')
hold on
figure(6)
semilogy(V, abs(Inn));
title('Semilogy Neural Net Fit')
hold on
end
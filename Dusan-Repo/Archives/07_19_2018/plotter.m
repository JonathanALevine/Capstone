x = linspace(1.3e-6, 1.7e-6, 200);
figure; hold on;
plot(x, abs(flipud(ins1)))
plot(x, abs(flipud(ins2)))
legend("TM0", "TE0");
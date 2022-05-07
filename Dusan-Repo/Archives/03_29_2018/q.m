figure; hold on;
for i = 1:500
    gc.train(100);
    plot(i, gc.validate, '.');
end

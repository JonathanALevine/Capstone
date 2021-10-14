% gc = Data; BE CAREFUL HERE
committee_size = 10;
figure; hold on;
for i = 1:committee_size
    gc_models(i) = Model(gc, 10);
    gc_models(i).train(100);
    gc_models(i).
end

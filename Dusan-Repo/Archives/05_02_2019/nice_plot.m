hold on;
plot(s1.train_error, 'b');
plot(s1.validate_error, 'r');
plot(s1.test_error, 'k');
plot(g1.train_error, 'b--');
plot(g1.validate_error, 'r--');
plot(g1.test_error, 'k--');
ylim([0 0.8]);
ylabel('Error');
xlabel('Epochs');
legend('Training Error (ANN_{PSO})','Validation Error (ANN_{PSO})','Test Error (ANN_{PSO})',...
    'Training Error (ANN_{U})','Validation Error (ANN_{U})','Test Error (ANN_{U})');
set(gca,'FontSize',12);
xticks([0 10000 20000 30000])
xticklabels({'0', '10000','20000','30000'})
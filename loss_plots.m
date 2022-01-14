close all; clc;  clear all; %initialization

% training loss and testing loss
% First model
table1 = readtable('losses_mse_069_lr_wd.csv');
training_loss1 = table1.training_loss;
testing_loss1 = table1.validation_loss;

% Second model
table2 = readtable('losses_nn_6_50_100_1.csv');
training_loss2 = table2.training_loss;
testing_loss2 = table2.validation_loss;

% third model
table3 = readtable('losses_nn_50_100_200_100_1.csv');
training_loss3 = table3.training_loss;
testing_loss3 = table3.validation_loss;

% Fourth model
table4 = readtable('losses_adam_lr_001_wd_0005_new.csv');
training_loss4 = table4.training_loss;
testing_loss4 = table4.validation_loss;

figure('Name', 'Plot of Losses');
plot(training_loss1)
xlim([400 10000])
ylim([0.5 1])
hold on;
plot(testing_loss1)
plot(training_loss2)
plot(testing_loss2)
plot(training_loss3)
plot(testing_loss3)
plot(training_loss4)
plot(testing_loss4)
hold off;
grid on;
xlabel('Epoch', 'FontSize', 16);
ylabel('MSE Error', 'FontSize', 16);
legend('Training Loss (Network 1)', 'Testing Loss (Network 1)', 'Training Loss (Network 2)', 'Testing Loss (Network 2)', 'Training Loss (Network 3)', 'Testing Loss (Network 3)');

FN2 = 'Plot of Losses';   
print(gcf, '-dpng', '-r600', FN2);  %Save graph in PNG





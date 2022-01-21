close all; clc;  clear all; %initialization

% training loss and testing loss
% First model
table1 = readtable('losses_mse_069_lr_wd.csv');
table4 = readtable('losses_adam_lr_001_wd_0005_new.csv');
table5 = [table1; table4];
training_loss1 = table5.training_loss;
testing_loss1 = table5.validation_loss;

% Second model
table2 = readtable('losses_nn_6_50_100_1.csv');
training_loss2 = table2.training_loss;
testing_loss2 = table2.validation_loss;

% third model
table3 = readtable('losses_nn_50_100_200_100_1.csv');
table6 = readtable('losses_nn_50_100_200_100_1_second.csv');
table7 = [table3; table6];
training_loss3 = table7.training_loss;
testing_loss3 = table7.validation_loss;

% fourth model
table8 = readtable('training_losses.csv');
training_loss8 = table8.training_loss;
testing_loss8 = table8.testing_loss;

figure('Name', 'Plot of Losses');
plot(training_loss8)
ylim([0 1])
hold on
plot(testing_loss8)
hold off
grid on;
xlabel('Epoch', 'FontSize', 16);
ylabel('MSE Error', 'FontSize', 16);
legend('Training Loss', 'Testing Loss')

% figure('Name', 'Plot of log Losses');
% plot(abs(log(training_loss8)))
% xlim([5 length(training_loss8)])
% hold on
% plot(abs(log(testing_loss8)))
% hold off
% grid on;
% xlabel('Epoch', 'FontSize', 16);
% ylabel('|log(MSE Error)|', 'FontSize', 16);
% legend('|log(Training Loss)|', '|log(Testing Loss)|')

% figure('Name', 'Plot of Losses');
% plot(training_loss1)
% xlim([10 10000])
% ylim([0, 1])
% hold on;
% plot(testing_loss1)
% plot(training_loss2)
% plot(testing_loss2)
% plot(training_loss3)
% plot(testing_loss3)
% plot(training_loss8)
% plot(testing_loss8)
% hold off;
% grid on;
% xlabel('Epoch', 'FontSize', 16);
% ylabel('MSE Error', 'FontSize', 16);
% legend('Training Loss (Network 1)', 'Testing Loss (Network 1)', 'Training Loss (Network 2)', 'Testing Loss (Network 2)', 'Training Loss (Network 3)', 'Testing Loss (Network 3)', 'Training Loss (Network 4)', 'Testing Loss (Network 4)');

FN2 = 'Plot of Losses';   
print(gcf, '-dpng', '-r600', FN2);  %Save graph in PNG





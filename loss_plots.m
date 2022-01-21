close all; clc;  clear all; %initialization

% fourth model
table = readtable('training_losses.csv');
training_loss = table.training_loss;
testing_loss = table.testing_loss;

figure('Name', 'Plot of Losses');
plot(training_loss)
ylim([0 1])
hold on
plot(testing_loss)
hold off
grid on;
xlabel('Epoch', 'FontSize', 16);
ylabel('MSE Error', 'FontSize', 16);
legend('Training Loss', 'Testing Loss')

% FN2 = 'Plot of Losses';   
% print(gcf, '-dpng', '-r600', FN2);  %Save graph in PNG





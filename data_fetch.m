
close all; clc;  clear all; %initialization

data = load("Dusan-Repo/data.mat").gc_data.examples;

number_of_examples = size(data,2);

lambdas = linspace(1.4*10^(-6), 1.7*10^(-6), 50);

data(1)
plot(data(1).labels)

% for i=1:10
% %     plot(lambdas, abs(data(i).labels))
%     TF = islocalmax(abs(data(i).labels));
%     if sum(TF) == 1
%         hold on
% %         plot(lambdas, abs(data(i).labels), lambdas(TF), abs(data(i).labels(TF)), 'r*')
%     plot(lambdas, data(i).labels)
%     end
%     pause(0.02)
% end
% number_of_examples = size(data,2);
% A = [];
% lambdas = linspace(1.4*10^(-6), 1.7*10^(-6), 50);
% 
% for i = 1:number_of_examples
%     [min_val, index] = min(data(i).labels);
%     lambda_val = lambdas(index);
%     transmission_val = min_val;
%     row = [data(i).features, lambda_val, transmission_val];
%     A = [A; row];
% end

% writematrix(A, 'data.csv');
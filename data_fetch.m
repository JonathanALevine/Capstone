
close all; clc;  clear all; %initialization

data = load("Dusan-Repo/data.mat").gc_data.examples;

number_of_examples = size(data,2);
A = [];
lambdas = linspace(1.5*10^(-6), 1.6*10^(-6), 50);

for i = 1:number_of_examples
    [min_val, index] = min(data(i).labels);
    lambda_val = lambdas(index);
    transmission_val = min_val;
    row = [data(i).features, lambda_val, transmission_val];
    A = [A; row];
end

writematrix(A, 'data.csv');
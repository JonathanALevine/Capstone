
data = load("Dusan-Repo/data.mat").gc_data.examples;

number_of_examples = size(data,2);
A = [];

% for i = 1:number_of_examples
%     row = []
%     for j = 1:size(data(1).labels, 2)
%         row(j) = data(i).labels(j);
%     end
%     A = [A; row]; 
% end
% 
% A
% 
% writematrix(A, 'labels.csv')

plot(data(1).labels)

x = categorical(["Network 1" "Network 2" "Network 3"]);
y = [48, 24, 72];

figure('Name', 'Plot of Training Time');
bar(x, y)
grid on;

xlabel('ANN', 'FontSize', 16);
ylabel('Training Time (Hours)', 'FontSize', 16);

FN2 = 'Plot of Training Time';   
print(gcf, '-dpng', '-r600', FN2);  %Save graph in PNG
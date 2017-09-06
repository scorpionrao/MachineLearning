fprintf('Program paused. Press enter to continue.\n');
pause;
%% ========================================================
fprintf("Plotting Data...\n");
data = load('ex1data1.txt');
X = data(:, 1);
y = data(:, 2);
m = length(y);
plotData(X, y);
fprintf('Program paused. Press enter to continue.\n');
pause;
%% ========================================================
X = [ones(m, 1), data(:, 1)];
theta = zeros(2, 1);
%% ========================================================
fprintf('\nRunning Gradient Descent ...\n')
iterations = 10;
alpha = 0.01;
theta = gradientDescent(X, y, theta, alpha, iterations);
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);
%% ========================================================
fprintf('\nPlot the linear fit ...\n')
hold on;
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off
%% ========================================================
fprintf('\nPredict values ...\n')
predict1 = [1, 100] * theta;
fprintf('For X = 100, we predict Y = %f\n', predict1);
%% ========================================================
fprintf('Visualizing J(theta_0, theta_1) ...\n')
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
J_vals = zeros(length(theta0_vals), length(theta1_vals));
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); 
ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

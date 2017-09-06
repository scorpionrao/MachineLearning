function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
m = length(y);
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
    hypothesis = X * theta;
    temp = zeros(size(X,2), 1);
    for row = 1:size(theta, 1)
        temp(row) = theta(row) - alpha / m * sum((hypothesis - y) .* X(:,row));
    end
    theta = temp;  
    J_history(iter) = computeCost(X, y, theta);
end
end
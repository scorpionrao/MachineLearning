function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% Theta1 - 25 x 401 - EVERY COLUMN CORRESPONDS TO A LAYER
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
% Theta2 - 10 x 26
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% FORWARD FEED AND COST FUNCTION

K = num_labels;
Y = eye(K)(y, :);

% Part 1 - Forward Feed and Cost Function
% adding bias vector and representing first layer - a1
a1 = [ones(m, 1), X];
% a1 - [m * 401] matrix; Theta1' - [401 * 25] matrix
z2 = a1 * Theta1';
% z2 and a2 - [m * 25]
a2 = sigmoid(z2);

% adding bias vector and representing second layer - a2
a2 = [ones(size(a2, 1), 1), a2];
% a2 is [m * 26]

%a2 - m * 26 matrix; Theta2' - 26 * 10 matrix
z3 = a2 * Theta2';

% Final layer
a3 = sigmoid(z3);
% a3 - [m * 10] - for each training example --> corresponding hypothesis
% In the 10 values in each row, only one value will be 1, rest will be 0.

% Y(i) - 1 * 10
% Summation - all classes and all training sets without regularization
costSummation = sum((-Y .* log(a3)) - ((1 - Y) .* log(1 - a3)), 2);
J = (1 / m) * sum(costSummation);

% J can be returned here for 'Without regularization scenario'

% Regularized cost function - DO NOT REGULARIZE BIAS UNITS
Theta1WithoutBias = Theta1(:, 2:end);
Theta2WithoutBias = Theta2(:, 2:end);

regularizationSummation = (lambda / (2 * m)) * (sum(sumsq(Theta1WithoutBias)) + sum(sumsq(Theta2WithoutBias)));
J += regularizationSummation;

% J can be returned here for 'With regularization scenario'

% BACK PROPAGATION
delta2 = 0;
delta1 = 0;

for t = 1:m
	% Start with a forward pass 
	% Inputs with bias
	a1 = [1; X(t, :)'];

	z2 = Theta1 * a1;
	% Inputs with bias
	a2 = [1; sigmoid(z2)];

	z3 = Theta2 * a2;
	% Inputs with bias
	a3 = sigmoid(z3);

	% delta (ERROR term) for output layer - abs|activation - true target value|
	% d3 = 10*1 matrix - 1 per activation node
	d3 = a3 - Y(t, :)';
	
	% delta Hidden Layer - weighted average of error terms in layer 3 based on slope.
	d2 = (Theta2WithoutBias' * d3) .* sigmoidGradient(z2);
	% d2 = 25*1 matrix - 1 per activation node

	% Step 4 - Accumulate
	delta2 += (d3 * a2');
	delta1 += (d2 * a1');
end

% normal (non regularized) gradient
Theta1_grad = (1 / m) * delta1;
Theta2_grad = (1 / m) * delta2;

% regularized gradient
% regualarization NOT applied to Bias element
Theta1_grad(:, 2:end) += ((lambda / m) * Theta1WithoutBias);
Theta2_grad(:, 2:end) += ((lambda / m) * Theta2WithoutBias);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

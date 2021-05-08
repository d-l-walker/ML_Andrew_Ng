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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Unregularised
X   = [ones(m,1) X];
z2  = Theta1*X';
a2  = sigmoid(z2);

a2  = [ones(m,1) a2'];
z3  = Theta2*a2';
h   = sigmoid(z3);

y2  = zeros(num_labels, m);

for i = 1:m,
  y2(y(i), i) = 1;
end

% Compute unregularised cost function
J = (1/m) * sum(sum((-y2).*log(h) - (1 - y2).*log(1 - h)));

% Regularisation -- need to skip bias unit:
Theta1_reg = Theta1(:,2:size(Theta1,2));
Theta2_reg = Theta2(:,2:size(Theta2,2));
J_reg      = (lambda/(2*m)) * (sum(sum(Theta1_reg.^ 2)) + sum(sum(Theta2_reg.^ 2)));
J          = J + J_reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
% Implement back-prop ...
for t=1:m
% 1
	a1 = X(t,:);
  a1 = a1';
	z2 = Theta1*a1;
	a2 = sigmoid(z2);

  a2 = [1 ; a2];
	z3 = Theta2*a2;
	a3 = sigmoid(z3);

% 2
	delta3 = a3 - y2(:,t);
	z2     = [1; z2];

% 3
  delta2 = (Theta2'*delta3).*sigmoidGradient(z2);

% 4
	delta2      = delta2(2:end);
	Theta2_grad = Theta2_grad + delta3 * a2';
	Theta1_grad = Theta1_grad + delta2 * a1';

end;

% 5
Theta2_grad = (1/m) * Theta2_grad;
Theta1_grad = (1/m) * Theta1_grad;

% Part 3: Implement regularization with the cost function and gradients.
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda/m)*Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda/m)*Theta2(:,2:end));
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

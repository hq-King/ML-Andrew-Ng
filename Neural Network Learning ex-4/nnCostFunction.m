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

%计算h(x) --a3:先计算a2 再计算a3
X = [ones(m,1) X];
a2 = sigmoid(Theta1*X');
a2 = [ones(1,m);a2];
h = sigmoid(Theta2*a2);
h = h' ;%m*k的向量
%把y变成m*k的向量m每一行是[1 0 0 0 0.....]只有一个1
temp = zeros(m,num_labels);
for i = 1:m
    temp(i,y(i)) = 1;
end
y = temp;
%计算J
%J = (-1/m)* sum(sum(y.*log(h)+(1-y).*log(1-h)));
%加入正规化
t1 = Theta1;
t2 = Theta2;
t1(:,1) = 0;%theta0(第一列)不参与正规化
t2(:,1) = 0;
J = (-1/m)* sum(sum(y.*log(h)+(1-y).*log(1-h)))+lambda/(2*m)*(sum(sum(t1.^2))+sum(sum(t2.^2)));


%为每一组theta1、theta2计算一个梯度值

%一定要放在循环外
Delta1 = zeros(size(Theta1));        %25x401
Delta2 = zeros(size(Theta2));         %10x26
for t = 1:m
    a1 = X(t,:)';
    yt = y(t,:)';
    z2 = Theta1 *a1;
    a2 = sigmoid(z2);%25维向量
    a2 = [1;a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    delta3 = a3-yt;%10维向量
    z2 = [1;z2];
    delta2 = Theta2'*delta3.*sigmoidGradient(z2);%26维向量
    %删除delta2(0)，变成25维向量
    delta2 = delta2(2:end);
    
    Delta2 = Delta2+delta3 *a2';
    Delta1 = Delta1+delta2 *a1';
end
    Theta1_grad = 1/m * Delta1;
    Theta2_grad = 1/m * Delta2;

% Regularized Neural Networks
% Theta2_grad(:, 2 : end) = Theta2_grad(:, 2 : end) + lambda * Theta2(:, 2 : end) / m;
% Theta1_grad(:, 2 : end) = Theta1_grad(:, 2 : end) + lambda * Theta1(:, 2 : end) / m;

temp1 = Theta1;
temp2 = Theta2;
temp1(:,1) = 0;
temp2(:,1) = 0;
 
r1 = lambda/m * temp1;
r2 = lambda/m * temp2;

 Theta1_grad =Theta1_grad+r1;
 Theta2_grad = Theta2_grad+r2;









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

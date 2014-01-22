function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m,
    h = sigmoid(theta'*X(i,:)');
    J += -y(i) * log(h) - (1 - y(i)) * log(1 - h);


    for j = 1:length(theta),    
        grad(j) += (h - y(i)) * X(i,j);
    end;

end;

J = J / m;
grad = grad / m;

reg_sum = 0;
reg_grad = zeros(size(theta));

for j = 2:length(theta),
    reg_sum += lambda / (2 * m ) * theta(j)^2;
    reg_grad(j) = lambda / m * theta(j);
end;

J = J + reg_sum;
grad = grad + reg_grad;









% =============================================================

end

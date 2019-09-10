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

  z = X*theta;
  hypX=sigmoid(z);
 
  X0 = X(:,1);
  theta0 = theta(1);
  XJ = X(:,2:size(X)(2));
  thetaJ = theta(2:length(theta));
  
  J = ((1/m)*sum((-y'*log(hypX))- (1-y)'*log(1-hypX))) + (lambda/(2*m))*sum(thetaJ.^2);

  grad0 = (1/m)*(hypX-y)'*X0;
  gradJ = (1/m)*((hypX-y)'*XJ)'+(thetaJ*(lambda/m));

  grad = [grad0;gradJ];

% size(X) %118x28
  % size(X0) %118x1
  % size(hypX) %118x1
  % size(y) %118x1
  % size(XJ) %118x27
  % (XJ-y)
  % size(XJ)
  % size((hypX-y)')
  % (hypX-y)'*XJ+(thetaJ*(lambda/m))'
  % (hypX-y)'*XJ
  % (hypX-y)'*XJ.+(thetaJ*(lambda/m))

% =============================================================

end

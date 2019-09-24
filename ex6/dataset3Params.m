function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

sigma_vector = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
C_vector = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% pred_train = zeros(length(sigma_vector),  length(C_vector));
pred_error = zeros(length(sigma_vector),1);
err = 1;
% ====================== YOUR CODE HERE ======================

%
  for i = 1:length(sigma_vector)
    sigma_test = sigma_vector(i);
    for j = 1:length(C_vector)
      C_test = C_vector(j);
      model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
      % modelCV = svmTrain(Xval, yval, C, @(x1,x2) gaussianKernel(x1,x2, sigma));

      % pred_train = svmPredict(model, X);
      predictions = svmPredict(model, Xval);
      test_err = mean(double(predictions ~=yval))

      if test_err <= err
        C_cv = C_test;
        sigma_cv = sigma_test;
        err = test_err;
        fprintf('new min. C = %f, sigma= %f with error: %f', C_cv, sigma_cv, err)
      end
    end
  end
  C = C_cv
  sigma = sigma_cv



%
%
  








% =========================================================================

end

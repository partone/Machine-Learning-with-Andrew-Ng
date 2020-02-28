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
sigma = 0.3;

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
% 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30
testValues = [0.01 0.03 0.1 0.3 1 3 10 30];

%Fist column is C, second is sigma, third is the error
predictions = zeros(size(testValues, 2) ^ 2, 3);
testNumber = 1;

for(i = 1:size(testValues, 2))
  for(j = 1:size(testValues, 2))
    model = svmTrain(X, y, testValues(i), @(x1, x2) gaussianKernel(x1, x2, testValues(j)));
    predictions(testNumber, 1) = testValues(i);
    predictions(testNumber, 2) = testValues(j);
    predictions(testNumber, 3) = mean(double(svmPredict(model, Xval) ~= yval));
    testNumber = testNumber + 1;
  end
end

disp(predictions)
[e, smallestError] = min(predictions(:, 3));
disp(smallestError)
C = predictions(smallestError, 1);
sigma = predictions(smallestError, 2);
pause
% =========================================================================

end

%% grad_Lasso - Gradient of smooth part of Lasso problem
%
% Returns the gradient of the smooth part of the Lasso objective:
%   f(x) = (1/(2m)) || A x - b ||_2^2
%
% INPUTS:
%   - x: Point at which to evaluate the gradient
%   - A: Data matrix (size m x n)
%   - b: Observation vector (size m x 1)
%   - lambda: Regularization parameter (not used in gradient computation)
%
% OUTPUT:
%   - grad: Gradient of f at x
%
% (c) 2025 Pablo Krupa
%

function grad = grad_Lasso(x, A, b, lambda)
    arguments
        x (:,1) {mustBeNumeric}
        A (:,:) {mustBeNumeric}
        b (:,1) {mustBeNumeric}
        lambda (1,1) {mustBeNumeric} %#ok<INUSA> % Not used, but kept for consistency
    end
    m = size(A, 1);
    % Compute the gradient
    grad = (1/m) * (A' * (A * x - b));
end
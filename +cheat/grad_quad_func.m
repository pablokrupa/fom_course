%% grad_quad_fund - Gradient of Quadratic Function
%
% Computes the gradient of
%   f(x) = (1/2)x'Hx + q'x
%
% INPUTS:
%   - x: Point at which to evaluate the gradient
%   - H: Cost function matrix
%   - q: Cost function vector
%
% OUTPUT:
%   - grad: Gradient of f at x
%
% (c) 2025 Pablo Krupa
%

function grad = grad_quad_func(x, H, q)
    arguments
        x (:,1) {mustBeNumeric}
        H (:,:) {mustBeNumeric}
        q (:,1) {mustBeNumeric}
    end
    % Compute the gradient
    grad = H * x + q;
end
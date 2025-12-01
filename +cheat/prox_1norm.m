%% prox_1norm - Proximal operator for ell_1 norm
%
% Computes prox_{t g}(x) for g(x) = lambda || x ||_1.
%
% INPUTS:
%   - x: Input vector
%   - t: Step size
%   - lambda
%
% OUTPUT:
%   - x_prox: Solution of the proximal operator
%
% (c) 2025 Pablo Krupa
%

function x_prox = prox_1norm(x, t, lambda)
    arguments
        x (:,1) {mustBeNumeric}
        t (1,1) {mustBeNumeric}
        lambda (1,1) {mustBeNumeric, mustBeNonnegative}
    end
    % Soft-thresholding operator
    x_prox = sign(x) .* max(abs(x) - t * lambda, 0);
end
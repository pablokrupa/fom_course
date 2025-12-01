%% prox_box_constraints.m - Proximal operator for box constraints
%
% Computes prox_{t g}(x) for g(x) = I_{[lb, ub]}(x), where I is the indicator function.
%
% INPUTS:
%   - x: Input vector
%   - t: Step size (not used in box constraints projection)
%   - lb: Lower bound vector
%   - ub: Upper bound vector
%
% OUTPUT:
%   - x_prox: Solution of the proximal operator
%
% (c) 2025 Pablo Krupa
%

function x_prox = prox_box_constraints(x, t, lb, ub)
    arguments
        x (:,1) {mustBeNumeric}
        t (1,1) {mustBeNumeric} = 1 %#ok<INUSA> % Step size (not used)
        lb (:,1) {mustBeNumeric} = -inf(size(x), 1)
        ub (:,1) {mustBeNumeric} = inf(size(x), 1)
    end
    % Project x onto the box [lb, ub]
    x_prox = min(max(x, lb), ub);
end
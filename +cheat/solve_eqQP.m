%% solve_eqQP - Solve equality-constrained QP problem
%
% This function solves the following Quadratic Programming (QP) problem with
% equality constraints:
%   minimize (1/2)x'Hx + q'x
%   subject to Gx = b
% by solving the linear system derived from the KKT conditions.
%
% INPUTS:
%   - H: Positive definite cost function matrix
%   - q: Cost function vector
%   - G: Equality constraints matrix
%   - b: Equality constraints vector
%
% OPTIONAL INPUTS:
%   - v: vector of additional quadratic term
%   - rho: scalar weight for the additional quadratic term
%
% If the additional quadratic term is provided, the problem solved is:
%   minimize (1/2)x'Hx + q'x + (rho/2)||x - v||^2_2
%   subject to Gx = b
%
% OUTPUT:
%   - x_sol: Solution to the QP problem
%
% Note: An efficient implementation would factorize the KKT matrix once and reuse
% it for different values of q, b, v.
%
% (c) 2025 Pablo Krupa
%

function x_sol = solve_eqQP(H, q, G, b, v, rho)
    arguments
        H (:,:) {mustBeNumeric}
        q (:,1) {mustBeNumeric}
        G (:,:) {mustBeNumeric}
        b (:,1) {mustBeNumeric}
        v (:,1) {mustBeNumeric} = []
        rho (1,1) {mustBeNumeric, mustBeNonnegative} = 0
    end
    % Dimensions
    n = size(H, 1);
    m = size(G, 1);

    % Construct the KKT matrix and right-hand side
    if ~isempty(v) && rho > 0
        H = H + rho * eye(n);
        q = q - rho * v;
    end
    KKT_matrix = [H, G'; G, zeros(m, m)];
    rhs = [-q; b];

    % Solve the KKT system
    sol = KKT_matrix \ rhs;

    % Extract the solution for x
    x_sol = sol(1:n);
end
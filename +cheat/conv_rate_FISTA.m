%% conv_rate_FISTA - Computes the theoretical convergence rate of FISTA
%
% INPUTS:
%   - L_f: Lipschitz constant of the gradient of the smooth part of f(x) = h(x) + g(x)
%   - mu_f: Strong convexity parameter of the smooth part of f(x) = h(x) + g(x)
%   - x0: Initial point of FISTA
%   - f0: Initial objective value at x0
%   - x_opt: Optimal solution of the optimization problem
%   - f_opt: Optimal objective value of the optimization problem
%   - L: Lipschitz constant used in the PGM method (0 if backtracking is used)
%   - k:  Length of the convergence rate bound vector to compute
%   - opt: Options structure with fields:
%       - L_0: Initial estimate of Lipschitz constant (for backtracking)
%       - eta_L: Factor to increase L_k during backtracking
%
% OUTPUTS:
%   - sublin: Vector of length k containing the sublinear convergence rate bound
%   - lin: Vector of length k containing the linear convergence rate bound
%
% NOTE: The linear convergence rate is for V-FISTA, as FISTA does not have linear convergence
%       in the strongly convex case.
%
% (c) 2025 Pablo Krupa
%

function [sublin, lin] = conv_rate_FISTA(L_f, mu_f, x0, f0, x_opt, f_opt, L, k, opt)

    % Pick value of alpha based on whether L is known or backtracking is used
    if L > 0.0
        alpha = 1;
        L_conv = L;
    else
        alpha = max(opt.eta_L, opt.L_0/L_f);
        L_conv = L_f;
    end

    sublin_conv_bound_k = @(k) (L_conv * alpha * norm(x0 - x_opt)^2) ./ ((k + 1).^2');
    lin_conv_bound_k = @(k) (1 - sqrt(mu_f / L_conv)).^k *(f0 - f_opt + (mu_f / 2) * norm(x0 - x_opt)^2);

    sublin = sublin_conv_bound_k(1:k);
    lin = lin_conv_bound_k(1:k);

end
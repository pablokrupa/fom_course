%% conv_rate_PGM - Computes the theoretical convergence rate of the PGM
%
% INPUTS:
%   - L_f: Lipschitz constant of the gradient of the smooth part of f(x) = h(x) + g(x)
%   - mu_f: Strong convexity parameter of the smooth part of f(x) = h(x) + g(x)
%   - x0: Initial point of the PGM
%   - x_opt: Optimal solution of the optimization problem
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
% (c) 2025 Pablo Krupa
%

function [sublin, lin] = conv_rate_PGM(L_f, mu_f, x0, x_opt, L, k, opt)

    % Pick value of alpha based on whether L is known or backtracking is used
    if L > 0.0
        alpha = 1;
        L_conv = L;
    else
        alpha = max(opt.eta_L, opt.L_0/L_f);
        L_conv = L_f;
    end

    sublin_conv_bound_k = @(k) (L_conv * alpha * norm(x0 - x_opt)^2) ./ (2 .* k);
    lin_conv_bound_k = @(k) (alpha * L_conv / 2) + (1 - mu_f / (alpha * L_conv)).^k * norm(x0 - x_opt)^2;

    sublin = sublin_conv_bound_k(1:k);
    lin = lin_conv_bound_k(1:k);

end
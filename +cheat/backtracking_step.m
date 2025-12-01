%% backtracking_step - Perform a backtracking procedure for first-order methods
%
% INPUTS:
%   - x_k: Current iterate
%   - grad_k: Gradient of f at x_k
%   - h_eval: Function handle to evaluate the smooth part of f(x) = h(x) + g(x)
%   - prox_op: Proximal operator function handle for g(x)
%   - L_k: Current Liptschitz constant estimate
%   - eta_L: Factor to increase L_k during backtracking
%   - verbose: Verbosity level (0: silent, 1: basic info, 2: detailed info)
%   - k: Current iteration number (for logging purposes)
%
% OUTPUTS:
%   - x_next: Next iterate after backtracking step
%   - t_k: Updated step size after backtracking
%   - j: Number of backtracking iterations performed
%
% NOTE: Divisions could be avoided by taking alpha_k = 1 / L_k as an input of the function and
% returning alpha_k, but we keep L_k here for clarity.
%
% (c) 2025 Pablo Krupa
%

function [x_next, L_k, j] = backtracking_step(x_k, grad_k, h_eval, prox_op, L_k, eta_L, verbose, k)
    h_k = h_eval(x_k);
    j = 0;
    while true
        j = j + 1;
        x_next = prox_op(x_k - (1/L_k) * grad_k, 1/L_k);
        h_temp = h_eval(x_next);
        % Check the sufficient decrease condition
        if h_temp <= h_k + grad_k' * (x_next - x_k) + (L_k/2) * norm(x_next - x_k, 2)^2
            break; % Descent lemma condition satisfied
        else
            L_k = L_k * eta_L; % Increase step size
            if verbose > 1
                fprintf('\tBacktracking: Increaseing L_k to %.6f at iteration %d\n', L_k, k);
            end
        end
    end
end
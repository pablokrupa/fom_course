%% Exercise on solving a Quadratic Programming (QP) problem with box constraints
%
% We will solve the following QP problem:
%   minimize (1/2)x'Hx + q'x
%   subject to lb <= x <= ub
% where H is a positive definite matrix, q is a vector, and lb and ub are
% the lower and upper bounds for x, respectively.
%
% The objective is to solve the problem using the PGM and FISTA (including using constant
% step size and backtracking). We will also implement a restart scheme for FISTA.
% The result can be checked against MATLAB's built-in 'quadprog' function.
%
% Compare the convergence of the different methods by plotting the objective value
% suboptimality (along with the theoretical convergence rates).
% 
% (c) 2025 Pablo Krupa
%
% -------------------------------------------------------------------------
clear; clc; %#ok<*UNRCH>
rng(1); % For reproducibility

%% Part 1: Problem Setup
n = 100; % Dimension of the QP problem
cond_num = 100; % Condition number for H
active_constr = 10; % Desired minimum number of active constraints at the solution
q_max = 1; % Magnitude of the linear term (smaller number tends to provide less active constraint)
[H, q, ~, ~, lb, ub] = cheat.generate_rand_boxQP(n, 0, 'cond_num', cond_num, ...
                                                 'active_constr', active_constr, 'q_mag', q_max);
L_f = eigs(H, 1, 'largestabs'); % Lipschitz constant of $L$-smooth function f(x)
mu_f = eigs(H, 1, 'smallestabs'); % Strong convexity parameter of f(x)

% Solve QP using MATLAB's quadprog for reference
options = optimoptions('quadprog', 'Display', 'off');
x_qp = quadprog(H, q, [], [], [], [], lb, ub, [], options);
f_qp = 0.5 * x_qp' * H * x_qp + q' * x_qp;

% Print some info
fprintf('Condition number of H: %.2f. ', cond(H));
fprintf('Lipschitz constant L_f: %.2f. ', L_f);
fprintf('Strong convexity parameter mu_f: %.2f\n', mu_f);
fprintf('Optimal objective value from quadprog: f(x^*) = %.6f\n', f_qp);

% Check number of active constraints at the solution
active_count = sum(x_qp <= lb + 1e-6) + sum(x_qp >= ub - 1e-6);
fprintf('Number of active constraints at the solution: %d\n', active_count);

% Initial setup for PGM and FISTA
x0 = zeros(n, 1); % Initial guess
L = L_f; % Estimation of Lipschitz constant. If 0, backtracking will be used.

% These will be the default options for the solvers (see below for explanation)
opt.max_iters = 2000;
opt.tol = 1e-4;
opt.verbose = 2;
opt.record = true;
opt.L_0 = L_f / 100;
opt.eta_L = 1.5;
opt.f_eval = @(x) 0.5 * x' * H * x + q' * x;
opt.h_eval = opt.f_eval;
opt.g_eval = 0; % We should have the indicator function for the box constraints here

%% Part 2: Solve using PGM
%
% Task: Implement the Proximal Gradient Method (PGM) to solve the QP with box constraints.
% Signature of the PGM function:
%   [x_sol, f_sol, e_flag, hist] = cheat.PGM(grad_h, prox_op, x0, L_f, opt);
% where:
%   - grad_h: Function handle for the gradient of the smooth part of f(x) = h(x) + g(x).
%             Signature: grad_h(x)
%   - prox_op: Function handle for the proximal operator of g(x).
%              Signature: prox_op(x, t), where t is the step size.
%   - x0: Initial guess
%   - L_f: Lipschitz constant of f(x). If 0, backtracking is used.
%   - opt: Options structure with fields:
%       - max_iters: Maximum number of iterations
%       - tol: Tolerance for stopping criterion
%       - verbose: Integer to control verbosity (0: none, 1: some, 2: detailed)
%       - record: Boolean to record function values (and other metrics)at each iteration
%       - L_0: Initial estimate of Lipschitz constant (for backtracking)
%       - eta_L: Factor to increase L_k during backtracking
%       - f_eval: Function handle to evaluate f(x)
%       - h_eval: Function handle to evaluate the smooth part of f(x) = h(x) + g(x)
%       - g_eval: Function handle to evaluate the non-smooth part of f(x) = h(x) + g(x)
% The outpus are:
%   - x_sol: Solution found
%   - f_sol: Objective value at the solution
%   - e_flag: Exit flag (1: converged, -1: max iterations reached)
%   - hist: Structure with history of the optimization. At least contains:
%       - f_k: Vector of function values at each iteration
%       - k: Number of iterations performed
%

% Function handlers for gradient and proximal operator
% Task: Code them yourself, instead of using the following cheat functions:
grad_h = @(x) cheat.grad_quad_func(x, H, q); % Gradient of objective function
prox_op = @(x, t) cheat.prox_box_constraints(x, t, lb, ub); % Prox operator for box constraints

opt_pgm = opt; % Copy default options for PGM
[x_pgm, f_pgm, e_flag_pgm, hist_pgm] = cheat.PGM(grad_h, prox_op, x0, L, opt_pgm);

% Compare results
fprintf('Difference in f(x): |f(x_pgm) - f(x_qp)| = %.6e\n', abs(f_pgm - f_qp));
fprintf('Difference in solutions: ||x_pgm - x_qp|| = %.6e\n', norm(x_pgm - x_qp));
fprintf('-------------------------------------------\n');

%% Part 3: Solve using FISTA
%
% Task: Implement the FISTA algorithm to solve the QP with box constraints.
% You can use a similar signature as the PGM function, but adapted for FISTA.
% Additionally, implement a restart scheme for FISTA based on function value increase.
% Compare the results with MATLAB's quadprog and the PGM solution.
%
% Additional options for FISTA are:
%  - restart: Boolean to enable/disable restart scheme
%  - v_FISTA: Boolean to use V-FISTA variant for linear convergence in strongly convex case
%  - mu_f: Strong convexity parameter of the smooth part of f(x) = h(x) + g(x) (needed for V-FISTA)
%

opt_fista = opt; % Copy default options for FISTA
opt_fista.restart = false; % NOTE: Try with true/false
opt_fista.v_FISTA = false; % NOTE: Try with true/false
opt_fista.mu_f = mu_f; % Needed for V-FISTA variant
[x_fista, f_fista, e_flag_fista, hist_fista] = cheat.FISTA(grad_h, prox_op, x0, L, opt_fista);

% Compare results
fprintf('Difference in f(x): |f(x_fista) - f(x_qp)| = %.6e\n', abs(f_fista - f_qp));
fprintf('Difference in solutions: ||x_fista - x_qp|| = %.6e\n', norm(x_fista - x_qp));

%% Part 4: Plot convergence results

% Task: compute the theoretical convergence rates for PGM and FISTA
% and plot them along with the actual convergence results from hist_pgm and hist_fista.
%
% Note that we compute the linear convergence rate for V-FISTA, as FISTA does not have linear 
% convergence in the strongly convex case.
max_iter = max(length(hist_pgm.f_k), length(hist_fista.f_k));
[cb_pgm_sublin, cb_pgm_lin] = cheat.conv_rate_PGM(L_f, mu_f, x0, x_qp, L, max_iter, opt_pgm);
[cb_fista_sublin, cb_fista_lin] = cheat.conv_rate_FISTA(L_f, mu_f, x0, opt_fista.f_eval(x0), x_qp, f_qp, L, max_iter, opt_fista);

% Plot objective value suboptimality
figure(1); clf(1); hold on; grid on; box on;
plot(hist_pgm.f_k - f_qp, 'b', 'LineWidth', 1.5);
plot(hist_fista.f_k - f_qp, 'r', 'LineWidth', 1.5);
plot(cb_pgm_lin, '--b', 'LineWidth', 1.5);
plot(cb_fista_lin, '--r', 'LineWidth', 1.5);
plot(cb_pgm_sublin, ':b', 'LineWidth', 2);
plot(cb_fista_sublin, ':r', 'LineWidth', 2);
set(gcf, 'color','w');
set(gca, 'FontSize', 18)
set(gca, 'YScale', 'log')
xlabel('Iteration k');
ylabel('Objective value suboptimality f(x_k) - f(x^*)');
ylim([1e-12, max([hist_pgm.f_k(1) - f_qp, hist_fista.f_k(1) - f_qp, ...
     cb_pgm_lin(1), cb_fista_lin(1), cb_pgm_sublin(1), cb_fista_sublin(1)])]);
legend('PGM', 'FISTA', 'PGM Linear Rate', ...
       'V-FISTA Linear Rate', 'PGM Sublinear Rate', 'FISTA Sublinear Rate', 'Location', 'northeastoutside');
title('Convergence of PGM and FISTA on box-constrained QP');

%% Exercise on solving a Lasso optimization problem
%
% We will solve the following Lasso problem:
%   minimize (1/(2m)) || A x - b ||_2^2 + lambda || x ||_1
% where A is a data matrix (size m x n), b is an observation vector, and lambda is
% a regularization parameter.
% 
% The objective is to solve the problem using the PGM and FISTA (including using constant
% step size and backtracking). We will also implement a restart scheme for FISTA.
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
m = 150; % Number of samples
n = 500; % Number of features
density = 0.2; % Density of the random matrix A
sparsity = 0.9; % Proportion of zero entries in the true solution
noise_level = 0.02; % Standard deviation of the noise added to the measurements b
[A, b, x_true] = cheat.generate_rand_Lasso(m, n, 'density', density, ...
                                           'sparsity', sparsity, 'noise_level', noise_level);
lambda = 0.002; % Regularization parameter
H = (1/m) * (A' * A);
L_f = eigs(H, 1, 'largestabs'); % Lipschitz constant of $L$-smooth function f(x)
mu_f = eigs(H, 1, 'smallestabs'); % This will be zero in most Lasso problems

% Initial setup for PGM and FISTA
x0 = x_true; % Initial guess
L_est = full(max(sum(abs(H), 2))); % Compute upper bound of L using Gershgorin circle theorem
L = L_est; % Estimation of Lipschitz constant. If 0, backtracking will be used.

% These will be the default options for the solvers
opt.max_iters = 2000;
opt.tol = 1e-4;
opt.verbose = 2;
opt.record = true;
opt.L_0 = L_f / 100;
opt.eta_L = 1.5;
opt.f_eval = @(x) (1/(2*m)) * norm(A*x - b, 2)^2 + lambda * norm(x, 1);
opt.h_eval = @(x) (1/(2*m)) * norm(A*x - b, 2)^2;
opt.g_eval = @(x) lambda * norm(x, 1);

% Solve using Matlab's lasso solver
x_m = lasso(full(A), b, 'Lambda', lambda, 'Intercept', false, 'Standardize', false, 'RelTol', 1e-10);
f_m = opt.f_eval(x_m);
res_m = norm(A*x_m - b, 2);
sparsity_m = sum(x_m == 0)/n;

% Print some info
fprintf('Generated Lasso problem with %d samples and %d features. Sparsity x_true = %.2f\n', ...
         m, n, sum(x_true == 0)/n);
fprintf('Lipschitz constant L_f: %.2f\n', L_f);
fprintf('Matlab results: f(x^*) = %.6f, || A x_m - b ||_2 = %.6f, sparsity = %.2f\n', ...
        f_m, res_m, sparsity_m);
fprintf('-------------------------------------------\n');

%% Part 2: Solve using PGM
% Same instructions as in `exercise_QP_PGM.m`

% Function handlers for gradient and proximal operator
% Task: Code them yourself, instead of using the following cheat functions:
grad_h = @(x) cheat.grad_Lasso(x, A, b, lambda);
prox_op = @(x, t) cheat.prox_1norm(x, t, lambda);

opt_pgm = opt; % Copy default options for PGM
[x_pgm, f_pgm, e_flag_pgm, hist_pgm] = cheat.PGM(grad_h, prox_op, x0, L, opt_pgm);
res_pgm = norm(A*x_pgm - b, 2);
sparsity_pgm = sum(x_pgm == 0)/n;

% Compare results
fprintf('PGM Results:\n');
fprintf('Objective value: f(x_pgm) = %.6f, || A x_pgm - b ||_2 = %.6f, sparsity = %.2f\n', ...
        f_pgm, res_pgm, sparsity_pgm);
fprintf('Difference in f(x): |f(x_pgm) - f(x_m)| = %.6e\n', abs(f_pgm - f_m));
fprintf('Difference in solutions: ||x_pgm - x_m|| = %.6e\n', norm(x_pgm - x_m));
fprintf('-------------------------------------------\n');

%% Part 3: Solve using FISTA
% Same instructions as in `exercise_QP_PGM.m`

opt_fista = opt; % Copy default options for FISTA
opt_fista.restart = false; % NOTE: Try with true/false
[x_fista, f_fista, e_flag_fista, hist_fista] = cheat.FISTA(grad_h, prox_op, x0, L, opt_fista);
res_fista = norm(A*x_fista - b, 2);
sparsity_fista = sum(x_fista == 0)/n;

% Compare results
fprintf('FISTA Results:\n');
fprintf('Objective value: f(x_fista) = %.6f, || A x_fista - b ||_2 = %.6f, sparsity = %.2f\n', ...
        f_fista, res_fista, sparsity_fista);
fprintf('Difference in f(x): |f(x_fista) - f(x_m)| = %.6e\n', abs(f_fista - f_m));
fprintf('Difference in solutions: ||x_fista - x_m|| = %.6e\n', norm(x_fista - x_m));
fprintf('-------------------------------------------\n');

%% Part 4: Plot convergence results

% Task: compute the theoretical convergence rates for PGM and FISTA
% See instructions in `exercise_QP_PGM.m`
max_iter = max(length(hist_pgm.f_k), length(hist_fista.f_k));
cb_pgm_sublin = cheat.conv_rate_PGM(L_f, mu_f, x0, x_m, L, max_iter, opt_pgm);
cb_fista_sublin = cheat.conv_rate_FISTA(L_f, mu_f, x0, opt_fista.f_eval(x0), x_m, f_m, L, max_iter, opt_fista);

% Plot objective value suboptimality
figure(1); clf(1); hold on; grid on; box on;
plot(hist_pgm.f_k - f_m, 'b', 'LineWidth', 1.5);
plot(hist_fista.f_k - f_m, 'r', 'LineWidth', 1.5);
plot(cb_pgm_sublin, ':b', 'LineWidth', 2);
plot(cb_fista_sublin, ':r', 'LineWidth', 2);
set(gcf, 'color','w');
set(gca, 'FontSize', 18)
set(gca, 'YScale', 'log')
xlabel('Iteration k');
ylabel('Objective value suboptimality f(x_k) - f(x^*)');
ylim([1e-12, max([hist_pgm.f_k(1) - f_m, hist_fista.f_k(1) - f_m, ...
     cb_pgm_sublin(1), cb_fista_sublin(1)])]);
legend('PGM', 'FISTA', 'PGM Sublinear Rate', 'FISTA Sublinear Rate', 'Location', 'northeastoutside');
title('Convergence of PGM and FISTA on Lasso Problem');

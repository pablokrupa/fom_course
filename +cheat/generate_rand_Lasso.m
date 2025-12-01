%% generate_rand_Lasso - Generate random Lasso problem
%
% This function generates a random Lasso optimization problem of the form:
%   minimize (1/(2m)) || A x - b ||_2^2 + lambda || x ||_1
%
% INPUTS:
%   - m: Number of observations
%   - n: Number of features
% 
% OPTIONAL INPUTS:
%   - density: Density of the random matrix A (size m x n)
%   - sparsity: Proportion of zero entries in the true solution x_true
%   - noise_level: Standard deviation of the noise added to b
%   - seed: Seed for the random number generator for reproducibility
%
% OUTPUTS:
%   - A: Data matrix (size m x n)
%   - b: Observation vector (size m x 1)
%   - x_true: True underlying sparse solution (without noise added to b)
%
% (c) 2025 Pablo Krupa
%

function [A, b, x_true] = generate_rand_Lasso(m, n, opt)
    arguments
        m (1,1) {mustBePositive, mustBeInteger}
        n (1,1) {mustBePositive, mustBeInteger}
        opt.density (1,1) {mustBePositive} = 0.2
        opt.sparsity (1,1) {mustBePositive, mustBeLessThanOrEqual(opt.sparsity, 1)} = 0.5
        opt.noise_level (1,1) {mustBeNonnegative} = 0.01
        opt.seed = []
    end

    % If a seed is provided, set the random number generator for reproducibility
    if ~isempty(opt.seed)
        curr_seed = rng; % Keep current seed to reset later
        rng(opt.seed);
    end

    % Generate random sparse matrix A
    A = sprandn(m, n, opt.density);

    % Generate true sparse solution x_true
    x_true = zeros(n, 1);
    num_nonzero = round((1 - opt.sparsity) * n); % 10% non-zero entries
    indices = randperm(n, num_nonzero);
    x_true(indices) = randn(num_nonzero, 1);

    % Generate observation vector b with noise
    b = A * x_true + opt.noise_level * randn(m, 1);

    % Reset random number generator to previous state
    if ~isempty(opt.seed)
        rng(curr_seed);
    end
end
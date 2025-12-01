%% generate_rand_boxQP - Generate random box-constrained QP problem
%
% This function generates a random Quadratic Programming (QP) problem with box constraints
%   minimize (1/2)x'Hx + q'x
%   subject to  G x = b
%               lb <= x <= ub
%
% INPUTS:
%   - n: Number of decision variables
%   - m: Number of equality constraints
%
% OPTIONAL INPUTS:
%   - cond_num: Condition number of H
%   - q_mag: Magnitude of q
%   - seed: Seed of rng(), for reproducibility
%   - active_constr: Forces active constraints at the solution.
%                    If > 0, tries to set at least that many active constraints.
%
% OUTPUTS:
%   - H, q: Cost function matrix and vector
%   - G, b: Equality constraints matrix and vector
%   - LB, UB: Lower and upper bound of the box constraints
%
% (c) 2025 Pablo Krupa
%

function [H, q, G, b, LB, UB] = generate_rand_boxQP(n, m, opt)
    arguments
        n (1,1) {mustBePositive, mustBeInteger}
        m (1,1) {mustBeNonnegative, mustBeInteger}
        opt.cond_num (1,1) {mustBePositive} = 100
        opt.q_mag (1,1) {mustBePositive} = 1
        opt.seed = []
        opt.active_constr (1,1) {mustBeNonnegative, mustBeInteger} = 0
    end

    % If a seed is provided, set the random number generator for reproducibility
    if ~isempty(opt.seed)
        curr_seed = rng; % Keep current seed to reset later
        rng(opt.seed);
    end

    % Compute H and q
    S=diag(exp(-log(opt.cond_num)/4:log(opt.cond_num)/2/(n-1):log(opt.cond_num)/4));
    [U,~] = qr((rand(n, n)-.5)*200);
    [V,~] = qr((rand(n, n)-.5)*200);
    H = U*S*V';
    H = H'*H;
    q = opt.q_mag*randn(n, 1);
    
    % Compute inequality constraints
    LB = -rand(n, 1) - 0.5;
    UB = rand(n, 1) + 0.5;
    
    % Compute equality constraints
    G = randn(m, n);
    b = rand(m, 1);

    % Compute box constraints
    if opt.active_constr
        % x_sol = cheat.solve_eqQP(H, q, G, b);
        options = optimoptions('quadprog', 'Display', 'off');
        x_sol = quadprog(H, q, [], [], [], [], LB, UB, [], options);
        active_lb = x_sol <= LB + 1e-6;
        active_ub = x_sol >= UB - 1e-6;
        active_count = sum(active_lb) + sum(active_ub);
        num_new = opt.active_constr - active_count;
        selectable_idx = find(~(active_lb | active_ub));
        if num_new > 0
            idx = randperm(length(selectable_idx), min(num_new, length(selectable_idx)));
            idx = selectable_idx(idx);
            for i = 1:length(idx)
                if x_sol(idx(i)) < 0
                    LB(idx(i)) = x_sol(idx(i)) + 0.1*rand();
                else
                    UB(idx(i)) = x_sol(idx(i)) - 0.1*rand();
                end
            end
        end
    end

    % Reset random number generator to previous state
    if ~isempty(opt.seed)
        rng(curr_seed);
    end

end


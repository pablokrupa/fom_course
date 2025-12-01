%% PGM - Implementation of the Proximal Gradient Method
%
% Solves optimization problems of the form:
%   min_x {f(x) = h(x) + g(x)}
% where h is $L_f$-smooth and g has a simple proximal operator.
%
% INPUTS:
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
%       - record: Boolean to record function values (and other metrics) at each iteration
%       - L_0: Initial estimate of Lipschitz constant (for backtracking)
%       - eta_L: Factor to increase L_k during backtracking
%       - f_eval: Function handle to evaluate f(x) (for recording purposes)
% OUTPUTS:
%   - x_sol: Solution found
%   - f_sol: Objective value at the solution (0 if f_eval not provided)
%   - e_flag: Exit flag (1: converged, -1: max iters reached)
%   - hist: Structure with history of the optimization. Contains:
%       - f_k: Vector of function values at each iteration
%       - k: Number of iterations performed
%       - num_prox_eval: Number of proximal operator evaluations
%       - L_k: Vector of estimated Lipschitz constants (if using backtracking)
%       - residual: Final residual value
%
% (c) 2025 Pablo Krupa
%

function [x_sol, f_sol, e_flag, hist] = PGM(grad_h, prox_op, x0, L_f, opt)
    arguments
        grad_h function_handle
        prox_op function_handle
        x0 (:,1) {mustBeNumeric}
        L_f (1,1) {mustBeNonnegative}
        opt struct
    end
    % Give default options if not provided
    if ~isfield(opt, 'max_iters'); opt.max_iters = 2000; end
    if ~isfield(opt, 'tol'); opt.tol = 1e-6; end
    if ~isfield(opt, 'verbose'); opt.verbose = 0; end
    if ~isfield(opt, 'record'); opt.record = false; end
    if ~isfield(opt, 'L_0'); opt.L_0 = 1; end
    if ~isfield(opt, 'eta_L'); opt.eta_L = 1.5; end
    if ~isfield(opt, 'f_eval'); opt.f_eval = []; end
    
    % Initialize variables
    x_k = x0;
    hist.f_k = [];
    hist.num_prox_eval = 0;
    hist.L_k = [];
    e_flag = 0; %#ok<NASGU> % Value 0 should not be returned
    if L_f > 0.0
        alpha = 1 / L_f; % Fixed step size
    else
        L_k = opt.L_0; % Initial estimate for backtracking
        hist.L_k = opt.L_0;
    end

    % Initial info
    if opt.verbose
        fprintf('--- Proximal Gradient Method (PGM) ---\n');
        if L_f > 0.0
            fprintf('Starting PGM with fixed step size (1/L_f = %.6f)...\n', alpha);
        else
            fprintf('Starting PGM with backtracking step size (initial 1/L_0 = %.6f).\n', 1 / opt.L_0);
        end
        fprintf('Problem with %d decision variables. Exit tolerance = %.6e\n', length(x0), opt.tol);
    end

    for k = 1:opt.max_iters
        % Compute gradient at current point
        grad_k = grad_h(x_k);
        
        % Update step
        if L_f > 0.0
            % Fixed step size
            x_next = prox_op(x_k - alpha * grad_k, alpha);
            hist.num_prox_eval = hist.num_prox_eval + 1;
        else
            % Backtracking line search (prox-grad step done inside to avoid additional computations)
            if isempty(opt.f_eval)
                error('For backtracking, opt.f_eval must be provided to evaluate f(x).');
            end
            [x_next, L_k, j_k] = cheat.backtracking_step(x_k, grad_k, opt.h_eval, prox_op, ...
                                                         L_k, opt.eta_L, opt.verbose, k);
            if opt.record
                hist.L_k = [hist.L_k; L_k];
                hist.num_prox_eval = hist.num_prox_eval + j_k;
            end
        end
        
        % Compute function value for recording
        if opt.record && ~isempty(opt.f_eval)
            f_val = opt.f_eval(x_next);
            hist.f_k = [hist.f_k; f_val];
        end
        
        % Check convergence
        residual = norm(x_next - x_k);
        if L_f > 0.0
            residual = residual * L_f;
        else
            residual = residual * L_k;
        end

        if residual < opt.tol
            if opt.verbose
                fprintf('PGM converged in %d iterations.\n', k);
                fprintf('Output: residual = %.6e, ||grad_f(x)|| = %.6f', residual, norm(grad_k));
                if opt.record && ~isempty(opt.f_eval)
                    fprintf(', f(x) = %.6f\n\n', f_val);
                else
                    fprintf('\n\n');
                end
            end
            break;
        end
        
        % Update current point
        x_k = x_next;
        
        % Verbose output
        if opt.verbose && ( mod(k, 100) == 0 || k == 1 )
            fprintf('Iteration %d: residual = %.6f, ||grad_f(x)|| = %.6f', k, residual, norm(grad_k));
            if opt.record && ~isempty(opt.f_eval)
                fprintf(', f(x) = %.6f\n', f_val);
            else
                fprintf('\n');
            end
        end
    end

    if k == opt.max_iters
        if opt.verbose
            fprintf('PGM reached maximum iterations (%d) without convergence.\n', opt.max_iters);
            fprintf('Output: residual = %.6e, ||grad_f(x)|| = %.6f', residual, norm(grad_k));
            if opt.record && ~isempty(opt.f_eval)
                fprintf(', f(x) = %.6f\n\n', f_val);
            else
                fprintf('\n\n');
            end
        end
        e_flag = -1; % Max iterations reached
    else
        e_flag = 1; % Converged
    end
    
    % Outputs
    x_sol = x_k;
    f_sol = opt.f_eval(x_sol);
    hist.k = k;
    hist.residual = residual;
end
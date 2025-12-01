# Corse material for gradient descent and the proximal gradient method

This folder contains exercises and material for the course "An applied course on first-order methods".

In particular, we cover here the following:
1. The part related to gradient descent and the Proximal Gradient Method (PGM), including aspects such as:
- Backtracking
- Acceleration
- Restart schemes
- Application to QP and Lasso problems
2. The part related to ADMM, including:
- Solving a problem with quadratic objective function, affine inequality constraints and restriction onto the ball of radius `r`.
- Solving linear MPC.

This material is intended as an exercise.
The files named `exersize_*.m` in this folder are templates for you to fill in the missing parts by coding the required functions (PGM algorithm, proximal operators, etc.).
Solutions for the missing functions can be found in the `+cheat` folder.
For example, you can run:
```matlab
[x_sol, f_sol, e_flag, hist] = cheat.PGM(grad_h, prox_op, x0, L, opt);
```
to use the provided solution for the Proximal Gradient Method.
The idea is to write you own implementation of all the `+cheat` functions requested in the `exersize_*.m` files (indicated by some `Task:` comment).
In the above example, create your own `PGM.m` file in this directory, implementing the Proximal Gradient Method.
Then run the example script removing the `cheat.` prefix to test your implementation:
```matlab
[x_sol, f_sol, e_flag, hist] = PGM(grad_h, prox_op, x0, L, opt);
```
You can then compare your results with the ones obtained using the `cheat.PGM` implementation.
Check out the documentation of the functions in the `+cheat` folder for details on their inputs, outputs and usage.

Each file contains comments that explain what you need to implement (again, indicated by a `Task:` comment).
You can run the scripts to test your implementations and visualize the results.
Files should run without errors if you use the `+cheat` implementations.

> [!NOTE]
> Many operations can be done more efficiently, but the objective here is to segment the different parts of the algorithms to better understand how they work and what ingredients are needed to implemente these optimization methods.

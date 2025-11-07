from maxsmooth.derivatives import derivative_prefactors
from maxsmooth.utils import is_positive_definite_cholesky
from jaxopt import OSQP, CvxpyQP
from jax import numpy as jnp
from itertools import product
import jax


def qp(x, y, N, pivot_point, function, basis_function, solver='OSQP'):

    # needs some dummy parameters to make basis
    basis = jax.vmap(basis_function, in_axes=(0, None, None, None)) \
        (x, x[pivot_point], y[pivot_point], jnp.ones(N))  
    Q = jnp.dot(basis.T, basis)
    #regularization = 1e-6 * jnp.eye(N)  # Small regularization term
    #Q = Q + regularization

    check = is_positive_definite_cholesky(Q)
    print("Is Q positive definite?", check)

    c = -jnp.dot(basis.T, y)
    G = derivative_prefactors(function, x, x[pivot_point], y[pivot_point], jnp.ones(N), N)
    G = G[2:]
    G_scaled = []
    for i, g in enumerate(G):
        # square root of sum of squares of each row
        g_norm = jnp.linalg.norm(g, axis=1, keepdims=True)
        g_norm = jnp.where(g_norm < 1e-10, 1.0, g_norm)  # Avoid division by zero
        G_scaled.append(g / g_norm)
    G = G_scaled


    print("Q shape:", Q.shape)
    print("c shape:", c.shape)
    print("G[0] shape:", G[0].shape, len(G))

    # All possible sign combinations for the blocks of G
    all_signs = list(product((-1.0, 1.0), repeat=len(G)))
    print(f"Total sign combinations: {len(all_signs)}")

    all_signs = jnp.array(all_signs)

    #init_params = jnp.linalg.solve(Q, -c)

    if solver == 'OSQP':
        qp = OSQP(maxiter=10000, tol=1e-3)
        @jax.jit
        def dcf(signs, c, Q):
            
            Gmat = jnp.vstack([signs[m] * G[m] for m in range(len(G))]) 
            h = jnp.zeros(Gmat.shape[0])
            sol = qp.run(params_obj=(Q, c), params_ineq=(Gmat, h))#.params
            
            return sol

        vmapped_dcf = jax.vmap(dcf, in_axes=(0, None, None))
        # Solve all QPs
        sol = vmapped_dcf(all_signs, c, Q)

    if solver == 'CvxpyQP':
        qp = CvxpyQP()


        def dcf_cvxpyqp(signs, c, Q):
            Gmat = jnp.vstack([signs[m] * G[m] for m in range(len(G))])
            h = jnp.zeros(Gmat.shape[0])
            
            # Convert to numpy for cvxpy
            Q_np = jnp.array(Q)
            c_np = jnp.array(c)
            Gmat_np = jnp.array(Gmat)
            h_np = jnp.array(h)
            
            sol = qp.run(init_params=jnp.ones(N), params_obj=(Q_np, c_np), params_ineq=(Gmat_np, h_np))
            return sol

        # Use a regular loop instead of vmap
        solutions = []
        print(f"Solving {len(all_signs)} QPs...")
        for i, signs in enumerate(all_signs):
            if i % 4 == 0:  # Progress indicator
                print(f"  Progress: {i}/{len(all_signs)}")
            sol = dcf_cvxpyqp(signs, c, Q)
            solutions.append(sol)
        sol = solutions.copy()

    return {'solver': solver, 'state': sol.state, 'params': sol.params.primal, 'sol': sol}
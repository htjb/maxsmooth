from jaxopt import OSQP, CvxpyQP
from jax import numpy as jnp
from maxsmooth.models import (difference_polynomial_basis, difference_polynomial)
from maxsmooth.derivatives import derivative_prefactors, make_derivative_functions
import matplotlib.pyplot as plt
import jax
from itertools import product
from jax.scipy.linalg import cholesky

funciton = difference_polynomial
basis_funciton = difference_polynomial_basis

def is_positive_definite_cholesky(A):
  """
  Checks if a symmetric matrix A is positive definite using Cholesky factorization.
  """
  try:
    # Attempt Cholesky factorization. JAX's cholesky can return NaNs or raise an error.
    # We rely on the fact that an error will be raised for non-positive definite matrices
    # if check_finite=True (default in jax.scipy.linalg.cholesky).
    L = cholesky(A, lower=True, check_finite=True)
    # If it succeeds, the matrix is positive definite.
    return True
  except ValueError:
    # If a ValueError (specifically LinAlgError in numpy, which maps to ValueError in jax)
    # is raised, the matrix is not positive definite.
    return False
  except Exception as e:
    # Catch other potential errors, if necessary.
    print(f"An unexpected error occurred: {e}")
    return False

key = jax.random.PRNGKey(0)
x = jnp.linspace(50, 150, 100)
y = 5e3*x**(-2.5) + 0.5*jax.random.normal(key, x.shape)
#y = x**2
N = 5
pivot_point = len(x)//2

init_params = jnp.ones(N)

basis = jax.vmap(basis_funciton, in_axes=(0, None, None, None)) \
    (x, x[pivot_point], y[pivot_point], init_params)
Q = jnp.dot(basis.T, basis)
#regularization = 1e-6 * jnp.eye(N)  # Small regularization term
#Q = Q + regularization

check = is_positive_definite_cholesky(Q)
print("Is Q positive definite?", check)

c = -jnp.dot(basis.T, y)
G = derivative_prefactors(funciton, x, x[pivot_point], y[pivot_point], init_params, N)
G = G[2:]
G_scaled = []
for i, g in enumerate(G):
    # square root of sum of squares of each row
    g_norm = jnp.linalg.norm(g, axis=1, keepdims=True)
    g_norm = jnp.where(g_norm < 1e-10, 1.0, g_norm)  # Avoid division by zero
    G_scaled.append(g / g_norm)
G = G_scaled

print(G)

print("Q shape:", Q.shape)
print("c shape:", c.shape)
print("G[0] shape:", G[0].shape, len(G))

# All possible sign combinations for the blocks of G
all_signs = list(product((-1.0, 1.0), repeat=len(G)))
print(f"Total sign combinations: {len(all_signs)}")

all_signs = jnp.array(all_signs)

solver = 'OSQP'

#init_params = jnp.linalg.solve(Q, -c)

if solver == 'OSQP':
    qp = OSQP(maxiter=50000, tol=1e-3)
    @jax.jit
    def dcf(signs, c, Q):
        Gmat = jnp.vstack([signs[m] * G[m] for m in range(len(G))])  # shape = (sum_rows_of_G[m], N)
        #Gmat_norm = jnp.mean(Gmat)
        #Gmat = Gmat/Gmat_norm  # Centering to improve numerical stability
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
        
        sol = qp.run(init_params=init_params, params_obj=(Q_np, c_np), params_ineq=(Gmat_np, h_np))
        return sol

    # Use a regular loop instead of vmap
    solutions = []
    print(f"Solving {len(all_signs)} QPs...")
    for i, signs in enumerate(all_signs):
        if i % 4 == 0:  # Progress indicator
            print(f"  Progress: {i}/{len(all_signs)}")
        sol = dcf_cvxpyqp(signs, c, Q)
        solutions.append(sol)
    print(solutions)

"""print("\n=== Detailed Solver Diagnostics ===")
print("Unique status values:", jnp.unique(sol.state.status))
for status_val in jnp.unique(sol.state.status):
    count = jnp.sum(sol.state.status == status_val)
    indices = jnp.where(sol.state.status == status_val)[0]
    print(f"\nStatus {status_val}: {count} cases")
    print(f"  Iteration range: {sol.state.iter_num[indices].min()}-{sol.state.iter_num[indices].max()}")
    if hasattr(sol.state, 'primal_residual'):
        print(f"  Primal residual range: {sol.state.primal_residual[indices].min():.2e}-{sol.state.primal_residual[indices].max():.2e}")
    if hasattr(sol.state, 'dual_residual'):
        print(f"  Dual residual range: {sol.state.dual_residual[indices].min():.2e}-{sol.state.dual_residual[indices].max():.2e}")

# Check solution quality
print("\n=== Solution Quality ===")
for i in range(min(3, len(sol.params.primal))):
    params = sol.params.primal[i]
    obj_val = 0.5 * params @ Q @ params + c @ params
    print(f"Solution {i}: status={sol.state.status[i]}, obj={obj_val:.3e}, |params|={jnp.linalg.norm(params):.3e}")

"""
print(sol.state)

sol = sol.params

objective_values = []
for params in sol.primal:

    #plt.plot(x, y, 'o', label='data')
    fit = jax.vmap(funciton, in_axes=(0, None, None, None)) \
        (x, x[pivot_point], y[pivot_point], params)
    obj_val = jnp.sum((y - fit)**2)
    objective_values.append(obj_val)
    """plt.plot(x, fit, '-', label='fit obj={:.2e}'.format(obj_val))
    plt.legend()
    plt.show()

    # Check that derivatives satisfy constraints
    deriv_funcs = make_derivative_functions(funciton, N)
    derivs = [jax.vmap(df, in_axes=(0, None, None, None))(x, x[pivot_point], y[pivot_point], params)
            for df in deriv_funcs]
    
    derivs = derivs[2:]  # From 2nd derivative onwards

    [plt.plot(x, derivs[m], label=f"{m+2}-th derivative") for m in range(len(derivs))]
    plt.axhline(0, color='k', ls='--')
    plt.legend()
    plt.show()"""

plt.plot(objective_values, 'o-')
plt.xlabel("Solution index")
plt.ylabel("Objective value")
plt.title("Objective values for different constraint sign combinations")
plt.show()
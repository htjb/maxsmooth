from jaxopt import OSQP, CvxpyQP
from jax import numpy as jnp
from maxsmooth.models import (normalised_polynomial, normalised_polynomial_basis)
from maxsmooth.derivatives import derivative_prefactors, make_derivative_functions
import matplotlib.pyplot as plt
import jax
from itertools import product
from maxsmooth.utils import is_positive_definite_cholesky

jax.config.update('jax_enable_x64', True)

function = normalised_polynomial
basis_function = normalised_polynomial_basis

key = jax.random.PRNGKey(0)
x = jnp.linspace(50, 150, 100)
y = 5e6*x**(-2.5) + 0.01*jax.random.normal(key, x.shape)
N = 8
pivot_point = len(x)//2

init_params = jnp.ones(N)

basis = jax.vmap(basis_function, in_axes=(0, None, None, None)) \
    (x, x[pivot_point], y[pivot_point], init_params)
Q = jnp.dot(basis.T, basis)
#regularization = 1e-6 * jnp.eye(N)  # Small regularization term
#Q = Q + regularization

check = is_positive_definite_cholesky(Q)
print("Is Q positive definite?", check)

c = -jnp.dot(basis.T, y)
G = derivative_prefactors(function, x, x[pivot_point], y[pivot_point], init_params, N)
G = G[2:]
"""G_scaled = []
for i, g in enumerate(G):
    # square root of sum of squares of each row
    g_norm = jnp.linalg.norm(g, axis=1, keepdims=True)
    g_norm = jnp.where(g_norm < 1e-10, 1.0, g_norm)  # Avoid division by zero
    G_scaled.append(g / g_norm)
G = G_scaled"""


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
    qp = OSQP(maxiter=10000, tol=1e-3)
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

print(sol.state)

sol = sol.params

objective_values = []
for params in sol.primal:

    #plt.plot(x, y, 'o', label='data')
    fit = jax.vmap(function, in_axes=(0, None, None, None)) \
        (x, x[pivot_point], y[pivot_point], params)
    obj_val = jnp.sum((y - fit)**2)
    objective_values.append(obj_val)


plt.plot(objective_values, 'o-')
plt.xlabel("Solution index")
plt.ylabel("Objective value")
plt.title("Objective values for different constraint sign combinations")
plt.show()

minimum_index = jnp.argmin(jnp.array(objective_values))
best_params = sol.primal[minimum_index]
print(best_params)
plt.plot(x, y, 'o', label='data')
best_fit =  jax.vmap(function, in_axes=(0, None, None, None)) \
        (x, x[pivot_point], y[pivot_point], best_params)
plt.plot(x, best_fit, '-', label='best fit obj={:.2e}'.format(objective_values[minimum_index]))
plt.legend()
plt.show()

plt.plot(x, y - best_fit, 'o', label='residuals')
plt.axhline(0, color='k', ls='--')
plt.legend()
plt.show()

deriv_funcs = make_derivative_functions(function, N)
derivs = [jax.vmap(df, in_axes=(0, None, None, None))(x, x[pivot_point], y[pivot_point], best_params)
        for df in deriv_funcs]

derivs = derivs[2:]  # From 2nd derivative onwards

[plt.plot(x, derivs[m], label=f"{m+2}-th derivative") for m in range(len(derivs))]
plt.axhline(0, color='k', ls='--')
plt.legend()
plt.show()
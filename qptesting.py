from jaxopt import OSQP
from jax import numpy as jnp
from maxsmooth.models import normalised_polynomial_basis, normalised_polynomial
from maxsmooth.derivatives import derivative_prefactors, make_derivative_functions
import matplotlib.pyplot as plt
import jax
from itertools import product

key = jax.random.PRNGKey(0)
x = jnp.linspace(2, 10, 100)
y = 5e3*x**(-2.5) + 5*jax.random.normal(key, x.shape)
N = 8
pivot_point = 25

init_params = jnp.ones(N)

basis = jax.vmap(normalised_polynomial_basis, in_axes=(0, None, None, None)) \
    (x, x[pivot_point], y[pivot_point], init_params)
Q = jnp.dot(basis.T, basis)
c = -jnp.dot(basis.T, y)
G = derivative_prefactors(normalised_polynomial, x, x[pivot_point], y[pivot_point], init_params, N)
G = G[2:]
print("Q shape:", Q.shape)
print("c shape:", c.shape)
print("G[0] shape:", G[0].shape, len(G))

# All possible sign combinations for the blocks of G
all_signs = list(product((-1.0, 1.0), repeat=len(G)))
print(f"Total sign combinations: {len(all_signs)}")

all_signs = jnp.array(all_signs)


qp = OSQP()

@jax.jit
def dcf(signs, c, Q):
    Gmat = jnp.vstack([signs[m] * G[m] for m in range(len(G))])  # shape = (sum_rows_of_G[m], N)
    h = jnp.zeros(Gmat.shape[0])*1e-6
    sol = qp.run(params_obj=(Q, c), params_ineq=(Gmat, h))#.params
    return sol

vmapped_dcf = jax.vmap(dcf, in_axes=(0, None, None))
# Solve all QPs
sol = vmapped_dcf(all_signs, c, Q)

exit()
sol = sol.params
print(sol)

plt.plot(x, y, 'o', label='data')
fit = jax.vmap(normalised_polynomial, in_axes=(0, None, None, None)) \
    (x, x[pivot_point], y[pivot_point], sol.primal)
plt.plot(x, fit, '-', label='fit')
plt.legend()
plt.show()

# Check that derivatives satisfy constraints
deriv_funcs = make_derivative_functions(normalised_polynomial, N)
derivs = [jax.vmap(df, in_axes=(0, None, None, None))(x, x[pivot_point], y[pivot_point], sol.primal)
          for df in deriv_funcs]

[plt.plot(x, derivs[m], label=f"{m}-th derivative") for m in range(len(derivs))]
plt.axhline(0, color='k', ls='--')
plt.legend()
plt.show()
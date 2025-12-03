"""Example use case for the qp solver."""

import time

import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp

from maxsmooth.derivatives import make_derivative_functions
from maxsmooth.models import normalised_polynomial, normalised_polynomial_basis
from maxsmooth.qp import qp

jax.config.update("jax_enable_x64", True)

function = normalised_polynomial
basis_function = normalised_polynomial_basis

key = jax.random.PRNGKey(0)
x = jnp.linspace(50, 150, 100)
y = 5e6 * x ** (-2.5) + 0.01 * jax.random.normal(key, x.shape)
N = 8
pivot_point = len(x) // 2

init_params = jnp.ones(N)

start = time.time()
qp = jax.jit(
    qp, static_argnames=("N", "pivot_point", "function", "basis_function")
)
sol = qp(x, y, N, pivot_point, function, basis_function)
end = time.time()
print(f"QP solved in {end - start:.2f} seconds")

objective_values = []
for params in sol["params"]:
    # plt.plot(x, y, 'o', label='data')
    fit = jax.vmap(function, in_axes=(0, None, None, None))(
        x, x[pivot_point], y[pivot_point], params
    )
    obj_val = jnp.sum((y - fit) ** 2)
    objective_values.append(obj_val)


plt.plot(objective_values, "o-")
plt.xlabel("Solution index")
plt.ylabel("Objective value")
plt.title("Objective values for different constraint sign combinations")
plt.show()

minimum_index = jnp.argmin(jnp.array(objective_values))
best_params = sol["params"][minimum_index]
print(best_params)
plt.plot(x, y, "o", label="data")
best_fit = jax.vmap(function, in_axes=(0, None, None, None))(
    x, x[pivot_point], y[pivot_point], best_params
)
plt.plot(
    x,
    best_fit,
    "-",
    label=f"best fit obj={objective_values[minimum_index]:.2e}",
)
plt.legend()
plt.show()

plt.plot(x, y - best_fit, "o", label="residuals")
plt.axhline(0, color="k", ls="--")
plt.legend()
plt.show()

deriv_funcs = make_derivative_functions(function, N)
derivs = [
    jax.vmap(df, in_axes=(0, None, None, None))(
        x, x[pivot_point], y[pivot_point], best_params
    )
    for df in deriv_funcs
]

derivs = derivs[2:]  # From 2nd derivative onwards

[
    plt.plot(x, derivs[m], label=f"{m + 2}-th derivative")
    for m in range(len(derivs))
]
plt.axhline(0, color="k", ls="--")
plt.legend()
plt.show()

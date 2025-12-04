"""Example use case for the qp solver."""

import time

import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp

from maxsmooth.models import difference_polynomial, difference_polynomial_basis
from maxsmooth.qp import fastqpsearch

jax.config.update("jax_enable_x64", True)

function = difference_polynomial
basis_function = difference_polynomial_basis

key = jax.random.PRNGKey(0)
x = jnp.linspace(50, 150, 100)
y = 5e6 * x ** (-2.5) + 0.01 * jax.random.normal(key, x.shape)
N = 7
pivot_point = len(x) // 2

fastqpsearch = jax.jit(
    fastqpsearch,
    static_argnames=("N", "pivot_point", "function", "basis_function"),
)
start = time.time()
results = fastqpsearch(x, y, N, pivot_point, function, basis_function)
print(results[1])
end = time.time()
print(f"Fast QP Search: QP solved in {end - start:.5f} seconds")

"""qp_jitted = jax.jit(
    qp, static_argnames=("N", "pivot_point", "function", "basis_function")
)
start = time.time()
status, params, error = qp_jitted(
    x, y, N, pivot_point, function, basis_function
)
print(params)
end = time.time()
print(f"First Call: QP solved in {end - start:.5f} seconds")
"""
vmapped_fit = jax.vmap(function, in_axes=(0, None, None, None))
fit = vmapped_fit(x, x[pivot_point], y[pivot_point], results[1])

plt.plot(x, y, "o", label="data")
plt.plot(
    x,
    fit,
    "-",
)
plt.legend()
plt.show()

plt.plot(x, y - fit, "o", label="residuals")
plt.axhline(0, color="k", ls="--")
plt.legend()
plt.show()

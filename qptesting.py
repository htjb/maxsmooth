"""Example use case for the qp solver."""

import time

import jax
import matplotlib.pyplot as plt
import tqdm
from jax import numpy as jnp

from maxsmooth.models import normalised_polynomial, normalised_polynomial_basis
from maxsmooth.qp import fastqpsearch, qp

jax.config.update("jax_enable_x64", True)

function = normalised_polynomial
basis_function = normalised_polynomial_basis

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

qp_jitted = jax.jit(
    qp, static_argnames=("N", "pivot_point", "function", "basis_function")
)
start = time.time()
status, params, error = qp_jitted(
    x, y, N, pivot_point, function, basis_function
)
print(params)
end = time.time()
print(f"First Call: QP solved in {end - start:.5f} seconds")
exit()

start = time.time()
status, params, error = qp_jitted(
    x, y, N, pivot_point, function, basis_function
)
end = time.time()
print(f"Second Call: QP solved in {end - start:.5f} seconds")
print(status)
print(params)
exit()
vmapped_function = jax.vmap(function, in_axes=(0, None, None, None))

objective_values = []
for i in tqdm.tqdm(range(len(params))):
    # plt.plot(x, y, 'o', label='data')
    fit = vmapped_function(x, x[pivot_point], y[pivot_point], params[i])
    print(fit)
    obj_val = jnp.sum((y - fit) ** 2)
    objective_values.append(obj_val)


plt.plot(objective_values, "o-")
plt.xlabel("Solution index")
plt.ylabel("Objective value")
plt.title("Objective values for different constraint sign combinations")
plt.show()

minimum_index = jnp.argmin(jnp.array(objective_values))
best_params = params[minimum_index]
plt.plot(x, y, "o", label="data")
best_fit = vmapped_function(x, x[pivot_point], y[pivot_point], best_params)
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

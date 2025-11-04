from jax import numpy as jnp
import jax

@jax.jit
def normalised_polynomial(x, y, N, pivot_point, params):
    y_sum = y[pivot_point]*jnp.sum(jnp.array([
        params[i]*(x/x[pivot_point])**i for i in range(N)]), axis=0)
    return y_sum

@jax.jit
def polynomial(x, y, N, pivot_point, params):
    y_sum = jnp.sum(jnp.array([
        params[i]*(x)**i for i in range(N)]), axis=0)
    return y_sum

@jax.jit
def loglog_polynomial(x, y, N, pivot_point, params):
    y_sum = 10**(jnp.sum(jnp.array([
        params[i]*jnp.log10(x)**i for i in range(N)]), axis=0))
    return y_sum

@jax.jit
def exponential(x, y, N, pivot_point, params):
    y_sum = y[pivot_point]*jnp.sum(jnp.array([
        params[i]*jnp.exp(-i*x/x[pivot_point]) for i in range(N)]), axis=0)
    return y_sum

@jax.jit
def log_polynomial(x, y, N, pivot_point, params):
    y_sum = jnp.sum(jnp.array([
        params[i]*jnp.log10(x/x[pivot_point])**i for i in range(N)]), axis=0)
    return y_sum

@jax.jit
def difference_polynomial(x, y, N, pivot_point, params):
    y_sum = jnp.sum(jnp.array([
        params[i]*(x - x[pivot_point])**i for i in range(N)]), axis=0)
    return y_sum

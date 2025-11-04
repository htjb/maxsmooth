from jax import numpy as jnp
import jax

@jax.jit
def normalised_polynomial(x, y, pivot_point, params):
    i = jnp.arange(params.shape[0])
    powers = (x / x[pivot_point])[None, :] ** i[:, None]  
    y_sum = y[pivot_point] * jnp.sum(params[:, None] * powers, axis=0)
    return y_sum

@jax.jit
def polynomial(x, y, pivot_point, params):
    i = jnp.arange(params.shape[0])
    powers = x[None, :] ** i[:, None]
    y_sum = jnp.sum(params[:, None] * powers, axis=0)
    return y_sum

@jax.jit
def loglog_polynomial(x, y, pivot_point, params):
    i = jnp.arange(params.shape[0])
    powers = jnp.log10(x)[None, :] ** i[:, None]
    y_sum = 10**jnp.sum(params[:, None] * powers, axis=0)
    return y_sum

@jax.jit
def exponential(x, y, pivot_point, params):
    i = jnp.arange(params.shape[0])
    powers = jnp.exp(-i[:, None] * x[None, :] / x[pivot_point])
    y_sum = y[pivot_point] * jnp.sum(params[:, None] * powers, axis=0)
    return y_sum

@jax.jit
def log_polynomial(x, y, pivot_point, params):
    i = jnp.arange(params.shape[0])
    powers = jnp.log10(x / x[pivot_point])[None, :] ** i[:, None]
    y_sum = jnp.sum(params[:, None] * powers, axis=0)
    return y_sum

@jax.jit
def difference_polynomial(x, y, pivot_point, params):
    i = jnp.arange(params.shape[0])
    powers = (x - x[pivot_point])[None, :] ** i[:, None]
    y_sum = jnp.sum(params[:, None] * powers, axis=0)
    return y_sum

from jax import numpy as jnp
import jax

@jax.jit
def normalised_polynomial(x, norm_x, norm_y, params):
    i = jnp.arange(params.shape[0])
    powers = (x / norm_x) ** i 
    y_sum = norm_y * jnp.sum(params * powers, axis=0)
    return y_sum

@jax.jit
def normalised_polynomial_basis(x, norm_x, norm_y, params):
    i = jnp.arange(params.shape[0])
    powers = norm_y * (x / norm_x) ** i  
    return powers

@jax.jit
def polynomial(x, norm_x, norm_y, params):
    i = jnp.arange(params.shape[0])
    powers = x ** i
    y_sum = jnp.sum(params * powers, axis=0)
    return y_sum

@jax.jit
def polynomial_basis(x, norm_x,  norm_y, params):
    i = jnp.arange(params.shape[0])
    powers = x ** i  
    return powers

@jax.jit
def loglog_polynomial(x, norm_x, norm_y, params):
    i = jnp.arange(params.shape[0])
    powers = jnp.log10(x) ** i
    y_sum = 10**jnp.sum(params * powers, axis=0)
    return y_sum

@jax.jit
def loglog_polynomial_basis(x, norm_x,  norm_y, params):
    i = jnp.arange(params.shape[0])
    powers = jnp.log10(x) ** i  
    return powers

@jax.jit
def exponential(x, norm_x, norm_y, params):
    i = jnp.arange(params.shape[0])
    powers = jnp.exp(-i * x / norm_x)
    y_sum = norm_y * jnp.sum(params * powers, axis=0)
    return y_sum

@jax.jit
def exponential_basis(x, norm_x,  norm_y, params):
    i = jnp.arange(params.shape[0])
    powers = norm_y * jnp.exp(-i * x / norm_x)
    return powers

@jax.jit
def log_polynomial(x, norm_x, norm_y, params):
    i = jnp.arange(params.shape[0])
    powers = jnp.log10(x / norm_x) ** i
    y_sum = jnp.sum(params * powers, axis=0)
    return y_sum

@jax.jit
def log_polynomial_basis(x, norm_x,  norm_y, params):
    i = jnp.arange(params.shape[0])
    powers = jnp.log10(x / norm_x) ** i  
    return powers

@jax.jit
def difference_polynomial(x, norm_x, norm_y, params):
    i = jnp.arange(params.shape[0])
    powers = (x - norm_x + 1e-6) ** i
    y_sum = jnp.sum(params * powers, axis=0)
    return y_sum

@jax.jit
def difference_polynomial_basis(x, norm_x, norm_y, params):
    i = jnp.arange(params.shape[0])
    powers = (x - norm_x + 1e-6) ** i  
    return powers
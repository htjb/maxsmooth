"""Various functional forms for modeling data."""

import jax
from jax import numpy as jnp


@jax.jit
def normalised_polynomial(
    x: jnp.ndarray,
    norm_x: jnp.ndarray,
    norm_y: jnp.ndarray,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a normalised polynomial at x.

    Args:
        x (jnp.ndarray): Input data points.
        norm_x (jnp.ndarray): Normalisation point for x.
        norm_y (jnp.ndarray): Normalisation point for y.
        params (jnp.ndarray): Coefficients of the polynomial.

    Returns:
        jnp.ndarray: Evaluated polynomial at x.
    """
    i = jnp.arange(params.shape[0])
    powers = (x / norm_x) ** i
    y_sum = norm_y * jnp.sum(params * powers, axis=0)
    return y_sum


@jax.jit
def normalised_polynomial_basis(
    x: jnp.ndarray,
    norm_x: jnp.ndarray,
    norm_y: jnp.ndarray,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the basis functions of a normalised polynomial at x.

    Args:
        x (jnp.ndarray): Input data points.
        norm_x (jnp.ndarray): Normalisation point for x.
        norm_y (jnp.ndarray): Normalisation point for y.
        params (jnp.ndarray): Coefficients of the polynomial.

    Returns:
        jnp.ndarray: Evaluated basis functions at x.
    """
    i = jnp.arange(params.shape[0])
    powers = norm_y * (x / norm_x) ** i
    return powers


@jax.jit
def polynomial(
    x: jnp.ndarray,
    norm_x: jnp.ndarray,
    norm_y: jnp.ndarray,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a polynomial at x.

    Args:
        x (jnp.ndarray): Input data points.
        norm_x (jnp.ndarray): Normalisation point for x.
        norm_y (jnp.ndarray): Normalisation point for y.
        params (jnp.ndarray): Coefficients of the polynomial.

    Returns:
        jnp.ndarray: Evaluated polynomial at x.
    """
    i = jnp.arange(params.shape[0])
    powers = x**i
    y_sum = jnp.sum(params * powers, axis=0)
    return y_sum


@jax.jit
def polynomial_basis(
    x: jnp.ndarray,
    norm_x: jnp.ndarray,
    norm_y: jnp.ndarray,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the basis functions of a polynomial at x.

    Args:
        x (jnp.ndarray): Input data points.
        norm_x (jnp.ndarray): Normalisation point for x.
        norm_y (jnp.ndarray): Normalisation point for y.
        params (jnp.ndarray): Coefficients of the polynomial.

    Returns:
        jnp.ndarray: Evaluated basis functions at x.
    """
    i = jnp.arange(params.shape[0])
    powers = x**i
    return powers


@jax.jit
def loglog_polynomial(
    x: jnp.ndarray,
    norm_x: jnp.ndarray,
    norm_y: jnp.ndarray,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a log-log polynomial at x.

    Args:
        x (jnp.ndarray): Input data points.
        norm_x (jnp.ndarray): Normalisation point for x.
        norm_y (jnp.ndarray): Normalisation point for y.
        params (jnp.ndarray): Coefficients of the polynomial.

    Returns:
        jnp.ndarray: Evaluated polynomial at x.
    """
    i = jnp.arange(params.shape[0])
    powers = jnp.log10(x) ** i
    y_sum = 10 ** jnp.sum(params * powers, axis=0)
    return y_sum


@jax.jit
def loglog_polynomial_basis(
    x: jnp.ndarray,
    norm_x: jnp.ndarray,
    norm_y: jnp.ndarray,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the basis functions of a log-log polynomial at x.

    Args:
        x (jnp.ndarray): Input data points.
        norm_x (jnp.ndarray): Normalisation point for x.
        norm_y (jnp.ndarray): Normalisation point for y.
        params (jnp.ndarray): Coefficients of the polynomial.

    Returns:
        jnp.ndarray: Evaluated basis functions at x.
    """
    i = jnp.arange(params.shape[0])
    powers = jnp.log10(x) ** i
    return powers


@jax.jit
def exponential(
    x: jnp.ndarray,
    norm_x: jnp.ndarray,
    norm_y: jnp.ndarray,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate an exponential basis at x.

    Args:
        x (jnp.ndarray): Input data points.
        norm_x (jnp.ndarray): Normalisation point for x.
        norm_y (jnp.ndarray): Normalisation point for y.
        params (jnp.ndarray): Coefficients of the exponential basis.

    Returns:
        jnp.ndarray: Evaluated exponential basis at x.
    """
    i = jnp.arange(params.shape[0])
    powers = jnp.exp(-i * x / norm_x)
    y_sum = norm_y * jnp.sum(params * powers, axis=0)
    return y_sum


@jax.jit
def exponential_basis(
    x: jnp.ndarray,
    norm_x: jnp.ndarray,
    norm_y: jnp.ndarray,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the basis functions of an exponential basis at x.

    Args:
        x (jnp.ndarray): Input data points.
        norm_x (jnp.ndarray): Normalisation point for x.
        norm_y (jnp.ndarray): Normalisation point for y.
        params (jnp.ndarray): Coefficients of the exponential basis.

    Returns:
        jnp.ndarray: Evaluated basis functions at x.
    """
    i = jnp.arange(params.shape[0])
    powers = norm_y * jnp.exp(-i * x / norm_x)
    return powers


@jax.jit
def log_polynomial(
    x: jnp.ndarray,
    norm_x: jnp.ndarray,
    norm_y: jnp.ndarray,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a log polynomial at x.

    Args:
        x (jnp.ndarray): Input data points.
        norm_x (jnp.ndarray): Normalisation point for x.
        norm_y (jnp.ndarray): Normalisation point for y.
        params (jnp.ndarray): Coefficients of the polynomial.

    Returns:
        jnp.ndarray: Evaluated polynomial at x.
    """
    i = jnp.arange(params.shape[0])
    powers = jnp.log10(x / norm_x) ** i
    y_sum = jnp.sum(params * powers, axis=0)
    return y_sum


@jax.jit
def log_polynomial_basis(
    x: jnp.ndarray,
    norm_x: jnp.ndarray,
    norm_y: jnp.ndarray,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the basis functions of a log polynomial at x.

    Args:
        x (jnp.ndarray): Input data points.
        norm_x (jnp.ndarray): Normalisation point for x.
        norm_y (jnp.ndarray): Normalisation point for y.
        params (jnp.ndarray): Coefficients of the polynomial.

    Returns:
        jnp.ndarray: Evaluated basis functions at x.
    """
    i = jnp.arange(params.shape[0])
    powers = jnp.log10(x / norm_x) ** i
    return powers


@jax.jit
def difference_polynomial(
    x: jnp.ndarray,
    norm_x: jnp.ndarray,
    norm_y: jnp.ndarray,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a difference polynomial at x.

    Args:
        x (jnp.ndarray): Input data points.
        norm_x (jnp.ndarray): Normalisation point for x.
        norm_y (jnp.ndarray): Normalisation point for y.
        params (jnp.ndarray): Coefficients of the polynomial.

    Returns:
        jnp.ndarray: Evaluated polynomial at x.
    """
    i = jnp.arange(params.shape[0])
    powers = (x - norm_x + 1e-6) ** i
    y_sum = jnp.sum(params * powers, axis=0)
    return y_sum


@jax.jit
def difference_polynomial_basis(
    x: jnp.ndarray,
    norm_x: jnp.ndarray,
    norm_y: jnp.ndarray,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the basis functions of a difference polynomial at x.

    Args:
        x (jnp.ndarray): Input data points.
        norm_x (jnp.ndarray): Normalisation point for x.
        norm_y (jnp.ndarray): Normalisation point for y.
        params (jnp.ndarray): Coefficients of the polynomial.

    Returns:
        jnp.ndarray: Evaluated basis functions at x.
    """
    i = jnp.arange(params.shape[0])
    powers = (x - norm_x + 1e-6) ** i
    return powers

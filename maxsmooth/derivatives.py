"""Compute derivative prefactors for a given function using JAX."""

from collections.abc import Callable

import jax
import jax.numpy as jnp


# function to generate derivatives
def make_derivative_functions(f: Callable, max_order: int) -> list[Callable]:
    """Return list of functions computing derivatives of f up to max_order.

    Args:
        f (Callable): Function to differentiate.
        max_order (int): Maximum order of derivatives to compute.

    Returns:
        List[Callable]: List of derivative functions.
    """
    derivs = [f]
    for _ in range(1, max_order):
        derivs.append(jax.grad(derivs[-1], argnums=0))
    return derivs


def derivative_prefactors(
    f: Callable,
    x: jnp.ndarray,
    norm_x: jnp.ndarray,
    norm_y: jnp.ndarray,
    params: jnp.ndarray,
    max_order: int,
) -> list[jnp.ndarray]:
    """Return list of derivative matrices G[m].

    G[m] maps params -> m-th derivative at all x.

    Args:
        f (Callable): Function to differentiate.
        x (jnp.ndarray): Input data points.
        norm_x (jnp.ndarray): Normalisation point for x.
        norm_y (jnp.ndarray): Normalisation point for y.
        params (jnp.ndarray): Parameters of the function.
        max_order (int): Maximum order of derivatives to compute.

    Returns:
        List[jnp.ndarray]: List of derivative matrices.
    """
    Gs = []
    df_dx = f  # start from f

    for m in range(max_order):
        # Jacobian of current derivative w.r.t parameters
        Gm = jax.vmap(
            lambda xi: jax.jacobian(df_dx, argnums=3)(
                xi, norm_x, norm_y, params
            )
        )(x)
        Gs.append(Gm)

        # Prepare next derivative wrt x
        df_dx = jax.grad(df_dx, argnums=0)
    return Gs

"""Compute derivative prefactors for a given function using JAX."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

# module level cache that persists for the length of the python process
_G_cache: dict[tuple, list[jnp.ndarray]] = {}


def derivative_prefactors(
    f: Callable,
    x: jnp.ndarray,
    norm_x: jnp.ndarray,
    norm_y: jnp.ndarray,
    params: jnp.ndarray,
    max_order: int,
) -> list[jnp.ndarray]:
    """Return list of derivative matrices G[m].

    G[m] maps params -> m-th derivative at all x. Results are cached by
    (function name, x values, norm_x, norm_y, max_order) so repeated calls
    with the same dataset do not recompute the autodiff chain.

    Args:
        f (Callable): Function to differentiate.
        x (jnp.ndarray): Input data points.
        norm_x (jnp.ndarray): Normalisation point for x.
        norm_y (jnp.ndarray): Normalisation point for y.
        params (jnp.ndarray): Parameters of the function (unused for linear
            models; only shape matters).
        max_order (int): Maximum order of derivatives to compute.

    Returns:
        List[jnp.ndarray]: List of derivative matrices.
    """
    key = (
        f.__name__,
        np.asarray(x).tobytes(),
        float(norm_x),
        float(norm_y),
        max_order,
    )
    if key in _G_cache:
        return _G_cache[key]

    Gs = []
    df_dx = f

    for m in range(max_order):
        Gm = jax.vmap(
            lambda xi: jax.jacobian(df_dx, argnums=3)(
                xi, norm_x, norm_y, params
            )
        )(x)
        Gs.append(Gm)
        df_dx = jax.grad(df_dx, argnums=0)

    _G_cache[key] = Gs
    return Gs

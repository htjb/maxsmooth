"""Quadratic programming call for maxsmooth."""

from collections.abc import Callable
from itertools import product

import jax
import jaxopt
from jax import numpy as jnp
from jaxopt import OSQP

from maxsmooth.derivatives import derivative_prefactors

qpsolver = OSQP(maxiter=10000, tol=1e-3, eq_qp_solve="lu")


@jax.jit
def _dcf(
    signs: jnp.ndarray,
    c: jnp.ndarray,
    Q: jnp.ndarray,
    G: jnp.ndarray,
) -> jaxopt._src.base.OptStep:
    """Run one QP solve for a given sign vector.

    Args:
        signs (jnp.ndarray): Sign combination (n_constrained,).
        c (jnp.ndarray): Linear term in the objective (N,).
        Q (jnp.ndarray): Quadratic term in the objective (N, N).
        G (jnp.ndarray): Derivative constraint matrix
            (n_constrained, n_data, N).

    Returns:
        sol: Solution of the quadratic programming problem.
    """
    Gmat = (signs[:, None, None] * G).reshape(-1, G.shape[2])
    h = jnp.zeros(Gmat.shape[0])
    return qpsolver.run(params_obj=(Q, c), params_ineq=(Gmat, h))


_vmapped_dcf = jax.vmap(_dcf, in_axes=(0, None, None, None))


def _flip_one(i: int, s: jnp.ndarray) -> jnp.ndarray:
    """Flip the i-th element of sign vector s.

    Args:
        i (int): Index to flip.
        s (jnp.ndarray): Sign vector.

    Returns:
        jnp.ndarray: Sign vector with element i negated.
    """
    return s.at[i].set(-s[i])


_flip_sign = jax.vmap(_flip_one, in_axes=(0, None))


def qp(
    x: jnp.ndarray,
    y: jnp.ndarray,
    N: int,
    pivot_point: int,
    function: Callable,
    basis_function: Callable,
    lowest_constrained_derivative: int = 2,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Set up and solve the quadratic programming problem for maxsmooth.

    Args:
        x (jnp.ndarray): Input data points.
        y (jnp.ndarray): Output data points.
        N (int): Number of basis functions.
        pivot_point (int): Index of the pivot point.
        function (Callable): The model funciton from `maxsmooth.models`.
        basis_function (Callable): The basis function to use.
        lowest_constrained_derivative (int): The lowest derivative to
            apply the constraints to.

    Returns:
        jnp.ndarray: state of the solver for each sign combination.
        jnp.ndarray: the parameters of the fits.
        jnp.ndarray: the reported error from jaxopt.
    """
    x_pivot = x[pivot_point]
    y_pivot = y[pivot_point]
    vmapped_basis = jax.vmap(basis_function, in_axes=(0, None, None, None))
    basis = vmapped_basis(x, x_pivot, y_pivot, jnp.ones(N))
    Q = jnp.dot(basis.T, basis)
    c = -jnp.dot(basis.T, y)

    G = derivative_prefactors(function, x, x_pivot, y_pivot, jnp.ones(N), N)[
        lowest_constrained_derivative:
    ]
    G = jnp.array(G)
    g_norm = jnp.linalg.norm(G, axis=2, keepdims=True)
    g_norm = jnp.where(g_norm < 1e-10, 1.0, g_norm)
    G = G / g_norm

    all_signs = jnp.array(list(product((-1.0, 1.0), repeat=len(G))))
    sol = _vmapped_dcf(all_signs, c, Q, G)

    vmapped_function = jax.vmap(function, in_axes=(0, None, None, None))
    objective_values = jax.vmap(
        lambda primal: jnp.sum(
            (y - vmapped_function(x, x_pivot, y_pivot, primal)) ** 2
        )
    )(sol.params.primal)
    best_index = jnp.argmin(objective_values)

    return (
        sol.state.status[best_index],
        sol.params.primal[best_index],
        sol.state.error[best_index],
    )


def qpsignsearch(
    x: jnp.ndarray,
    y: jnp.ndarray,
    N: int,
    pivot_point: int,
    function: Callable,
    basis_function: Callable,
    lowest_constrained_derivative: int = 2,
    key: jnp.ndarray = jax.random.PRNGKey(0),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Set up and solve the quadratic programming problem for maxsmooth.

    slowqpsearch uses some elements of the searching algorithm
    detailed in the maxsmooth paper to reduce the number of QP solves needed.
    However it involves a lot of conditional logic which makes it slower
    in JAX than the brute-fore try everything method for small problems.

    Args:
        x (jnp.ndarray): Input data points.
        y (jnp.ndarray): Output data points.
        N (int): Number of basis functions.
        pivot_point (int): Index of the pivot point.
        function (Callable): The model funciton from `maxsmooth.models`.
        basis_function (Callable): The basis function to use.
        key (jnp.ndarray): JAX random key.
        lowest_constrained_derivative (int): The lowest derivative to
            apply the constraints to.

    Returns:
        jnp.ndarray: state of the solver for each sign combination.
        jnp.ndarray: the parameters of the fits.
        jnp.ndarray: the reported error from jaxopt.
    """
    x_pivot = x[pivot_point]
    y_pivot = y[pivot_point]
    vmapped_basis = jax.vmap(basis_function, in_axes=(0, None, None, None))
    basis = vmapped_basis(x, x_pivot, y_pivot, jnp.ones(N))
    Q = jnp.dot(basis.T, basis)
    c = -jnp.dot(basis.T, y)

    G = derivative_prefactors(function, x, x_pivot, y_pivot, jnp.ones(N), N)[
        lowest_constrained_derivative:
    ]
    G = jnp.array(G)
    g_norm = jnp.linalg.norm(G, axis=2, keepdims=True)
    g_norm = jnp.where(g_norm < 1e-10, 1.0, g_norm)
    G = G / g_norm

    all_signs = jnp.array(list(product((-1.0, 1.0), repeat=len(G))))

    signs = jnp.array(
        [
            jnp.ones(len(G)),
            -jnp.ones(len(G)),
            jnp.array([1 if i % 2 == 0 else -1 for i in range(len(G))]),
            jnp.array([-1 if i % 2 == 0 else 1 for i in range(len(G))]),
        ]
    )

    visited_signs = jnp.zeros(len(all_signs))
    visited_signs = jax.lax.fori_loop(
        0,
        len(signs),
        lambda i, vs: vs.at[
            jnp.where(
                jnp.all(all_signs == signs[i], axis=1),  # type: ignore
                size=1,
                fill_value=0,
            )[0]
        ].set(1),
        visited_signs,
    )

    sol = _vmapped_dcf(signs, c, Q, G)
    error = sol.state.error
    minimum_index = jnp.argmin(error)
    signs = signs[minimum_index]
    error = error[minimum_index]
    best_error = jnp.inf

    initial_state = (
        sol.state.status[minimum_index],
        error,
        best_error,
        signs,
        c,
        Q,
        sol.params.primal[minimum_index],
        visited_signs,
    )

    def condition(state: tuple) -> bool:
        _, error, best_error, _, _, _, _, _ = state
        return error < best_error

    def body(state: tuple) -> tuple:
        status, error, best_error, signs, c, Q, best_params, visited_signs = (
            state
        )
        best_error = error
        flip_signs = _flip_sign(jnp.arange(len(signs)), signs)

        def body_unique_flip(i: int, fs: jnp.ndarray) -> jnp.ndarray:
            """Check if flip sign has been visited already.

            Args:
                i (int): Index in flip_signs.
                fs (jnp.ndarray): Current flip signs.

            Returns:
                jnp.ndarray: Updated flip signs with visited ones zeroed.
            """
            index = jnp.where(
                jnp.all(all_signs == flip_signs[i], axis=1),  # type: ignore
                size=1,
                fill_value=-1,
            )[0]
            fs = jax.lax.cond(
                visited_signs.at[index] == 1,
                lambda f: jnp.zeros_like(f),
                lambda f: f,
                fs,
            )
            return fs

        flip_signs = jax.lax.fori_loop(
            0,
            len(flip_signs),
            body_unique_flip,
            flip_signs,
        )  # type: ignore

        visited_signs = jax.lax.fori_loop(
            0,
            len(flip_signs),
            lambda i, vs: vs.at[
                jnp.where(
                    jnp.all(all_signs == flip_signs[i], axis=1),  # type: ignore
                    size=1,
                    fill_value=0,
                )[0]
            ].set(1),
            visited_signs,
        )

        sol = _vmapped_dcf(flip_signs, c, Q, G)
        minimum_index = jnp.argmin(jnp.array(sol.state.error))
        return (
            sol.state.status[minimum_index],
            sol.state.error[minimum_index],
            best_error,
            flip_signs[minimum_index],
            c,
            Q,
            sol.params.primal[minimum_index],
            visited_signs,
        )

    results = jax.lax.while_loop(condition, body, initial_state)

    return results[0], results[6], jnp.array([])

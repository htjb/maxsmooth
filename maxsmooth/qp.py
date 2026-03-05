"""Quadratic programming call for maxsmooth."""

from collections.abc import Callable
from itertools import product

import jax
import jax.numpy as jnp
import qpax

from maxsmooth.derivatives import derivative_prefactors


@jax.jit
def _dcf(
    signs: jnp.ndarray,
    c: jnp.ndarray,
    Q: jnp.ndarray,
    G: jnp.ndarray,
    max_iters: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve one QP for a given sign vector.
    
    Using a primal-dual interior point method implemented in qpax.

    CVXOPT was used in version 1 of maxsmooth which also implements
    primal-dual interior point method but is not
    jit-compatible and is much slower than qpax.

    Args:
        signs (jnp.ndarray): Sign combination (n_constrained,).
        c (jnp.ndarray): Linear term in the objective (N,).
        Q (jnp.ndarray): Quadratic term in the objective (N, N).
        G (jnp.ndarray): Derivative constraint matrix
            (n_constrained, n_data, N).
        max_iters (int): Maximum iterations for the QP solver.

    Returns:
        jnp.ndarray: Optimal parameters (N,), or NaN if infeasible.
        jnp.ndarray: Convergence flag (1 = converged, 0 = not).
    """
    Gmat = (signs[:, None, None] * G).reshape(-1, G.shape[2])
    h = jnp.zeros(Gmat.shape[0])
    A_eq = jnp.zeros((0, Q.shape[0]))
    b_eq = jnp.zeros(0)
    x_sol, _, _, _, converged, _ = qpax.solve_qp(Q, c, A_eq, b_eq, Gmat, h,
                    max_iters=max_iters
    )
    return x_sol, converged


_vmapped_dcf = jax.vmap(_dcf, in_axes=(0, None, None, None, None))


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
    max_iters: int | jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
    """Set up and solve the quadratic programming problem for maxsmooth.

    Brute-forces all 2^(N - lowest_constrained_derivative) sign combinations
    in a single vmapped call, returning the best feasible fit.

    Args:
        x (jnp.ndarray): Input data points.
        y (jnp.ndarray): Output data points.
        N (int): Number of basis functions.
        pivot_point (int): Index of the pivot point.
        function (Callable): The model function from `maxsmooth.models`.
        basis_function (Callable): The basis function to use.
        lowest_constrained_derivative (int): The lowest derivative to
            apply the constraints to.
        max_iters (int | None): Maximum iterations for the QP solver. If None,
            uses max_iters: int =jnp.where(N**2 > 50, N**2, 50),

    Returns:
        jnp.ndarray: The best-fit parameters (N,).
        jnp.ndarray: Objective (chi-squared) value for the best fit.
        bool: True if the winning QP solve converged within qpax's
            iteration limit.
    """
    max_iters = (jnp.where(N**2 > 50, N**2, 50) 
                 if max_iters is None else max_iters
    )
    x_pivot = x[pivot_point]
    y_pivot = y[pivot_point]
    vmapped_basis = jax.vmap(basis_function, in_axes=(0, None, None, None))
    basis = vmapped_basis(x, x_pivot, y_pivot, jnp.ones(N))
    Q = jnp.dot(basis.T, basis)
    c = -jnp.dot(basis.T, y)

    G = jnp.array(
        derivative_prefactors(function, x, x_pivot, y_pivot, jnp.ones(N), N)[
            lowest_constrained_derivative:
        ]
    )
    g_norm = jnp.linalg.norm(G, axis=2, keepdims=True)
    G = G / jnp.where(g_norm < 1e-10, 1.0, g_norm)

    all_signs = jnp.array(list(product((-1.0, 1.0), repeat=len(G))))
    solutions, converged_flags = _vmapped_dcf(all_signs, c, Q, G, max_iters)

    vmapped_fn = jax.vmap(function, in_axes=(0, None, None, None))

    def chi2(params: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum((y - vmapped_fn(x, x_pivot, y_pivot, params)) ** 2)

    obj = jax.vmap(chi2)(solutions)
    # NaN solutions (infeasible sign combos) are excluded from selection
    obj = jnp.where(jnp.any(jnp.isnan(solutions), axis=1), jnp.inf, obj)
    best_index = jnp.argmin(obj)
    converged = bool(converged_flags[best_index])
    return solutions[best_index], obj[best_index], converged


def qpsignsearch(
    x: jnp.ndarray,
    y: jnp.ndarray,
    N: int,
    pivot_point: int,
    function: Callable,
    basis_function: Callable,
    lowest_constrained_derivative: int = 2,
    max_iters: int | jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
    """Solve the DCF QP using a sign-navigating search.

    Starts from four seed sign vectors and iteratively flips one sign at a
    time toward lower chi-squared, using jax.lax.while_loop.  Described in
    the maxsmooth MNRAS paper.

    Args:
        x (jnp.ndarray): Input data points.
        y (jnp.ndarray): Output data points.
        N (int): Number of basis functions.
        pivot_point (int): Index of the pivot point.
        function (Callable): The model function from `maxsmooth.models`.
        basis_function (Callable): The basis function to use.
        lowest_constrained_derivative (int): The lowest derivative to
            apply the constraints to.
        max_iters (int | None): Maximum iterations for the QP solver. If None,
            uses max_iters: int =jnp.where(N**2 > 50, N**2, 50),
                which is a heuristic that seems to work well in practice.

    Returns:
        jnp.ndarray: The best-fit parameters (N,).
        jnp.ndarray: Objective (chi-squared) value for the best fit.
        bool: True if every QP solve along the search path converged.
    """
    max_iters = (jnp.where(N**2 > 50, N**2, 50) 
                 if max_iters is None else max_iters
    )
    x_pivot = x[pivot_point]
    y_pivot = y[pivot_point]
    vmapped_basis = jax.vmap(basis_function, in_axes=(0, None, None, None))
    basis = vmapped_basis(x, x_pivot, y_pivot, jnp.ones(N))
    Q = jnp.dot(basis.T, basis)
    c = -jnp.dot(basis.T, y)

    G = jnp.array(
        derivative_prefactors(function, x, x_pivot, y_pivot, jnp.ones(N), N)[
            lowest_constrained_derivative:
        ]
    )
    g_norm = jnp.linalg.norm(G, axis=2, keepdims=True)
    G = G / jnp.where(g_norm < 1e-10, 1.0, g_norm)

    all_signs = jnp.array(list(product((-1.0, 1.0), repeat=len(G))))

    seeds = jnp.array(
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
        len(seeds),
        lambda i, vs: vs.at[
            jnp.where(
                jnp.all(all_signs == seeds[i], axis=1),  # type: ignore
                size=1,
                fill_value=0,
            )[0]
        ].set(1),
        visited_signs,
    )

    vmapped_fn = jax.vmap(function, in_axes=(0, None, None, None))

    def chi2(params: jnp.ndarray) -> jnp.ndarray:
        val = jnp.sum((y - vmapped_fn(x, x_pivot, y_pivot, params)) ** 2)
        return jnp.where(jnp.any(jnp.isnan(params)), jnp.inf, val)

    seed_solutions, seed_converged = _vmapped_dcf(seeds, c, Q, G, max_iters)
    seed_chi2 = jax.vmap(chi2)(seed_solutions)
    best_seed = jnp.argmin(seed_chi2)

    # State: (current_chi2, best_chi2, current_signs, best_params,
    #         visited, all_converged)
    initial_state = (
        seed_chi2[best_seed],
        jnp.inf,
        seeds[best_seed],
        seed_solutions[best_seed],
        visited_signs,
        seed_converged[best_seed],
    )

    def condition(state: tuple) -> jnp.ndarray:
        current_chi2, best_chi2, _, _, _, _ = state
        return current_chi2 < best_chi2

    def body(state: tuple) -> tuple:
        current_chi2, best_chi2, signs, best_params, visited_signs, acc_conv = (
            state
        )
        best_chi2 = current_chi2
        flip_signs = _flip_sign(jnp.arange(len(signs)), signs)

        def mask_visited(i: int, fs: jnp.ndarray) -> jnp.ndarray:
            """Zero out any flip that has already been visited.

            Args:
                i (int): Index in flip_signs.
                fs (jnp.ndarray): Current set of candidate flipped signs.

            Returns:
                jnp.ndarray: Updated candidates with visited ones zeroed.
            """
            index = jnp.where(
                jnp.all(all_signs == flip_signs[i], axis=1),  # type: ignore
                size=1,
                fill_value=-1,
            )[0]
            return jax.lax.cond(
                visited_signs.at[index] == 1,
                lambda f: jnp.zeros_like(f),
                lambda f: f,
                fs,
            )

        flip_signs = jax.lax.fori_loop(
            0, len(flip_signs), mask_visited, flip_signs
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

        flip_solutions, flip_converged = _vmapped_dcf(
            flip_signs, c, Q, G, max_iters
        )
        flip_chi2 = jax.vmap(chi2)(flip_solutions)
        best_flip = jnp.argmin(flip_chi2)
        acc_conv = acc_conv & flip_converged[best_flip]
        return (
            flip_chi2[best_flip],
            best_chi2,
            flip_signs[best_flip],
            flip_solutions[best_flip],
            visited_signs,
            acc_conv,
        )

    final_chi2, _, _, best_params, _, all_converged = jax.lax.while_loop(
        condition, body, initial_state
    )
    return best_params, final_chi2, bool(all_converged)

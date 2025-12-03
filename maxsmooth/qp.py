"""Quadratic programming call for maxsmooth."""

from collections.abc import Callable
from itertools import product

import jax
import jaxopt
from jax import numpy as jnp
from jaxopt import OSQP

from maxsmooth.derivatives import derivative_prefactors


def qp(
    x: jnp.ndarray,
    y: jnp.ndarray,
    N: int,
    pivot_point: int,
    function: Callable,
    basis_function: Callable,
) -> dict:
    """Set up and solve the quadratic programming problem for maxsmooth.

    Args:
        x (jnp.ndarray): Input data points.
        y (jnp.ndarray): Output data points.
        N (int): Number of basis functions.
        pivot_point (int): Index of the pivot point.
        function (Callable): The model funciton from `maxsmooth.models`.
        basis_function (Callable): The basis function to use.

    Returns:
        dict: Dictionary containing solver information and solution.
    """
    # needs some dummy parameters to make basis
    basis_function = jax.vmap(basis_function, in_axes=(0, None, None, None))
    basis = basis_function(x, x[pivot_point], y[pivot_point], jnp.ones(N))
    Q = jnp.dot(basis.T, basis)

    c = -jnp.dot(basis.T, y)
    G = derivative_prefactors(
        function, x, x[pivot_point], y[pivot_point], jnp.ones(N), N
    )[2:]

    all_signs = jnp.array(list(product((-1.0, 1.0), repeat=len(G))))

    qp = OSQP(maxiter=5000, tol=1e-3)

    @jax.jit
    def dcf(
        signs: jnp.ndarray, c: jnp.ndarray, Q: jnp.ndarray
    ) -> jaxopt._src.base.OptStep:
        """Run the quadratic programming using jaxopt OSQP.

        Args:
            signs (jnp.ndarray): Sign combination
                for the inequality constraints.
            c (jnp.ndarray): Linear term in the objective function.
            Q (jnp.ndarray): Quadratic term in the objective function.

        Returns:
            sol: Solution of the quadratic programming problem.
        """
        Gmat = jnp.vstack([signs[m] * G[m] for m in range(len(G))])
        h = jnp.zeros(Gmat.shape[0])
        sol = qp.run(params_obj=(Q, c), params_ineq=(Gmat, h))
        return sol

    vmapped_dcf = jax.vmap(dcf, in_axes=(0, None, None))
    # Solve all QPs
    sol = vmapped_dcf(all_signs, c, Q)

    return {
        "status": sol.state.status,
        "params": sol.params.primal,
        "error": sol.state.error,
    }


def fastqpsearch(
    x: jnp.ndarray,
    y: jnp.ndarray,
    N: int,
    pivot_point: int,
    function: Callable,
    basis_function: Callable,
    key: jnp.ndarray = jax.random.PRNGKey(0),
) -> dict:
    """Set up and solve the quadratic programming problem for maxsmooth.

    fastqpsearch uses the searching algorithm detailed in the maxsmooth
    paper to reduce the number of QP solves needed.

    Args:
        x (jnp.ndarray): Input data points.
        y (jnp.ndarray): Output data points.
        N (int): Number of basis functions.
        pivot_point (int): Index of the pivot point.
        function (Callable): The model funciton from `maxsmooth.models`.
        basis_function (Callable): The basis function to use.
        key (jnp.ndarray): JAX random key.

    Returns:
        dict: Dictionary containing solver information and solution.
    """

    @jax.jit
    def dcf(
        signs: jnp.ndarray, c: jnp.ndarray, Q: jnp.ndarray
    ) -> jaxopt._src.base.OptStep:
        """Run the quadratic programming using jaxopt OSQP.

        Args:
            signs (jnp.ndarray): Sign combination
                for the inequality constraints.
            c (jnp.ndarray): Linear term in the objective function.
            Q (jnp.ndarray): Quadratic term in the objective function.

        Returns:
            sol: Solution of the quadratic programming problem.
        """
        Gmat = jnp.vstack([signs[m] * G[m] for m in range(len(G))])
        h = jnp.zeros(Gmat.shape[0])
        sol = qp.run(params_obj=(Q, c), params_ineq=(Gmat, h))
        return sol

    # needs some dummy parameters to make basis
    basis_function = jax.vmap(basis_function, in_axes=(0, None, None, None))
    basis = basis_function(x, x[pivot_point], y[pivot_point], jnp.ones(N))
    Q = jnp.dot(basis.T, basis)

    c = -jnp.dot(basis.T, y)
    G = derivative_prefactors(
        function, x, x[pivot_point], y[pivot_point], jnp.ones(N), N
    )[2:]

    all_signs = jnp.array(list(product((-1.0, 1.0), repeat=len(G))))

    qp = OSQP(maxiter=5000, tol=1e-3)

    key, subkey = jax.random.split(key)
    r = jax.random.choice(subkey, all_signs.shape[0], (1,))
    signs = all_signs[r[0]]

    sol = dcf(signs, c, Q)
    error = sol.state.error
    best_error = jnp.inf

    flip_sign = jax.vmap(lambda i, s: s.at[i].set(-s[i]), in_axes=(0, None))
    vmapped_dcf = jax.vmap(dcf, in_axes=(0, None, None))

    while error < best_error:
        best_error = error
        flip_signs = flip_sign(jnp.arange(len(signs)), signs)
        sol = vmapped_dcf(flip_signs, c, Q)
        minimum_index = jnp.argmin(jnp.array([s.error for s in sol]))
        test_signs = flip_signs[minimum_index]
        new_error = sol[minimum_index].state.error
        if new_error < error:
            error = new_error
            signs = test_signs

    return {
        "state": sol.state,
        "params": sol.params.primal,
        "sol": sol,
    }

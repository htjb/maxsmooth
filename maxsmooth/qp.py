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
    basis = basis_function(
        x, x[pivot_point], y[pivot_point], jnp.ones(N)
    )
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
        "state": sol.state,
        "params": sol.params.primal,
        "sol": sol,
    }

def fastqpsearch(
    x: jnp.ndarray,
    y: jnp.ndarray,
    N: int,
    pivot_point: int,
    function: Callable,
    basis_function: Callable,
    solver: str = "OSQP",
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
        solver (str, optional): Solver to use ('OSQP' or 'CvxpyQP').
            Defaults to 'OSQP'.

    Returns:
        dict: Dictionary containing solver information and solution.
    """
    return NotImplementedError("fastqpsearch is not implemented yet.")
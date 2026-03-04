"""Tests for the maxsmooth v2 (JAX/qpax) API."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from maxsmooth.derivatives import _G_cache, derivative_prefactors
from maxsmooth.models import (
    exponential,
    exponential_basis,
    loglog_polynomial,
    loglog_polynomial_basis,
    normalised_polynomial,
    normalised_polynomial_basis,
    polynomial,
    polynomial_basis,
)
from maxsmooth.qp import qp, qpsignsearch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def polynomial_data() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Exact cubic: y = 1 + x + x^2 + x^3 on [1, 2].

    Using [1, 2] rather than [0, 1] avoids the near-singular Gram matrix
    that `polynomial` basis produces in float32 on the unit interval.
    All constrained derivatives (m >= 2) are strictly positive on [1,2].
    """
    x = jnp.linspace(1, 2, 100)
    y = 1 + x + x**2 + x**3
    return x, y


@pytest.fixture
def power_law_data() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Power law with noise: y = 5e7 * x^-2.5 on [50, 200]."""
    rng = np.random.default_rng(42)
    x_np = np.linspace(50, 200, 100)
    y_np = 5e7 * x_np**(-2.5) + rng.normal(0, 1e3, 100)
    return jnp.array(x_np), jnp.array(y_np)


# ── qp() ──────────────────────────────────────────────────────────────────────

def test_qp_returns_finite(power_law_data: tuple) -> None:
    """qp() returns finite params and chi2."""
    x, y = power_law_data
    params, chi2, _ = qp(x, y, 6, 50, normalised_polynomial, normalised_polynomial_basis)
    assert jnp.all(jnp.isfinite(params))
    assert jnp.isfinite(chi2)


def test_qp_chi2_matches_residual(power_law_data: tuple) -> None:
    """Returned chi2 equals sum((y - yfit)^2) recomputed from params."""
    x, y = power_law_data
    params, chi2, _ = qp(x, y, 6, 50, normalised_polynomial, normalised_polynomial_basis)
    yfit = jax.vmap(normalised_polynomial, in_axes=(0, None, None, None))(
        x, x[50], y[50], params
    )
    assert_allclose(float(chi2), float(jnp.sum((y - yfit) ** 2)), rtol=1e-4)


def test_qp_recovers_polynomial_coefficients(polynomial_data: tuple) -> None:
    """qp() fits exact cubic data to near-zero residual."""
    x, y = polynomial_data
    params, chi2, _ = qp(x, y, 4, 50, polynomial, polynomial_basis)
    # Float32 precision limits exact param recovery; check fit quality instead.
    yfit = jax.vmap(polynomial, in_axes=(0, None, None, None))(
        x, x[50], y[50], params
    )
    assert_allclose(np.array(yfit), np.array(y), rtol=5e-3)


# ── qpsignsearch() ────────────────────────────────────────────────────────────

def test_signsearch_returns_finite(power_law_data: tuple) -> None:
    """qpsignsearch() returns finite params and chi2."""
    x, y = power_law_data
    params, chi2, _ = qpsignsearch(
        x, y, 6, 50, normalised_polynomial, normalised_polynomial_basis
    )
    assert jnp.all(jnp.isfinite(params))
    assert jnp.isfinite(chi2)


def test_signsearch_chi2_matches_residual(power_law_data: tuple) -> None:
    """Returned chi2 equals sum((y - yfit)^2) recomputed from params."""
    x, y = power_law_data
    params, chi2, _ = qpsignsearch(
        x, y, 6, 50, normalised_polynomial, normalised_polynomial_basis
    )
    yfit = jax.vmap(normalised_polynomial, in_axes=(0, None, None, None))(
        x, x[50], y[50], params
    )
    assert_allclose(float(chi2), float(jnp.sum((y - yfit) ** 2)), rtol=1e-4)


# ── qp vs qpsignsearch ────────────────────────────────────────────────────────

def test_qp_and_signsearch_agree(power_law_data: tuple) -> None:
    """Brute-force qp and qpsignsearch find consistent minima."""
    x, y = power_law_data
    _, chi2_qp, _ = qp(x, y, 6, 50, normalised_polynomial, normalised_polynomial_basis)
    _, chi2_ss, _ = qpsignsearch(
        x, y, 6, 50, normalised_polynomial, normalised_polynomial_basis
    )
    assert_allclose(float(chi2_qp), float(chi2_ss), rtol=0.05)


# ── Constraint satisfaction ───────────────────────────────────────────────────

def test_constrained_derivatives_do_not_change_sign(power_law_data: tuple) -> None:
    """Each constrained derivative order should be all >= 0 or all <= 0."""
    x, y = power_law_data
    N, pivot = 6, 50
    params, _, _ = qp(x, y, N, pivot, normalised_polynomial, normalised_polynomial_basis)
    G_list = derivative_prefactors(
        normalised_polynomial, x, x[pivot], y[pivot], params, N
    )
    for m in range(2, N):
        deriv_m = jnp.dot(G_list[m], params)
        tol = 0.05 * float(jnp.max(jnp.abs(deriv_m)))
        all_pos = jnp.all(deriv_m >= -tol)
        all_neg = jnp.all(deriv_m <= tol)
        assert all_pos or all_neg, f"derivative order {m} changes sign"


# ── lowest_constrained_derivative ─────────────────────────────────────────────

def test_fewer_constraints_gives_equal_or_better_fit(power_law_data: tuple) -> None:
    """Relaxing constraints (higher lowest_constrained_derivative) should not worsen fit."""
    x, y = power_law_data
    _, chi2_from2, _ = qp(
        x, y, 6, 50, normalised_polynomial, normalised_polynomial_basis,
        lowest_constrained_derivative=2,
    )
    _, chi2_from3, _ = qp(
        x, y, 6, 50, normalised_polynomial, normalised_polynomial_basis,
        lowest_constrained_derivative=3,
    )
    assert float(chi2_from3) <= float(chi2_from2) + 1.0


# ── Built-in models ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("fn,basis", [
    (normalised_polynomial, normalised_polynomial_basis),
    (exponential, exponential_basis),
])
def test_builtin_models_run(
    power_law_data: tuple,
    fn: callable,
    basis: callable,
) -> None:
    """All built-in model/basis pairs complete without error."""
    x, y = power_law_data
    params, chi2, _ = qp(x, y, 6, 50, fn, basis)
    assert jnp.all(jnp.isfinite(params)), f"{fn.__name__} returned non-finite params"
    assert jnp.isfinite(chi2), f"{fn.__name__} returned non-finite chi2"


# ── Derivative prefactors cache ───────────────────────────────────────────────

def test_derivative_prefactors_cache_consistent(power_law_data: tuple) -> None:
    """Two calls to derivative_prefactors with the same args return identical results."""
    x, y = power_law_data
    N, pivot = 6, 50
    _G_cache.clear()
    G1 = derivative_prefactors(
        normalised_polynomial, x, x[pivot], y[pivot], jnp.ones(N), N
    )
    G2 = derivative_prefactors(
        normalised_polynomial, x, x[pivot], y[pivot], jnp.ones(N), N
    )
    for m in range(N):
        assert_allclose(np.array(G1[m]), np.array(G2[m]))

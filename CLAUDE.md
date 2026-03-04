# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_smooth.py::test_api

# Lint
ruff check maxsmooth/

# Lint and auto-fix
ruff check --fix maxsmooth/

# Build and serve docs locally
pip install ".[docs]"
mkdocs serve
```

## Architecture

`maxsmooth` fits derivative-constrained functions (DCFs) to data using quadratic programming (QP). The core idea: for an N-parameter model, find coefficients **a** minimising `||y - B·a||²` subject to `G·a ≤ 0`, where G encodes sign constraints on selected derivative orders.

Because each constrained derivative can be positive or negative, the constraint is not inherently linear. `maxsmooth` handles this by testing sign combinations `s_m ∈ {-1, +1}` for each constrained derivative order m, solving `(s_m · G_m)·a ≤ 0` for every combination and selecting the best fit.

### Module layout

- **`maxsmooth/qp.py`** — Core QP solvers using `jaxopt.OSQP` (ADMM):
  - `qp()` — brute-force: vmaps over all `2^(N - lowest_constrained_derivative)` sign combinations simultaneously.
  - `qpsignsearch()` — sign-navigating algorithm (starts from 4 seed sign vectors, uses `jax.lax.while_loop` to flip one sign at a time toward lower error). Described in the maxsmooth MNRAS paper. Currently slower than `qp()` for small problems due to JAX's JIT overhead on conditional logic.

- **`maxsmooth/derivatives.py`** — Builds the derivative constraint matrix G via JAX automatic differentiation (`jax.grad` + `jax.jacobian`). `derivative_prefactors()` returns a list of Jacobian matrices G[m] where `G[m] · params = m-th derivative at all x`.

- **`maxsmooth/models.py`** — Built-in functional forms, each with a `<name>` and `<name>_basis` variant. All are `@jax.jit`-decorated and accept `(x, norm_x, norm_y, params)`. Available: `normalised_polynomial`, `polynomial`, `loglog_polynomial`, `exponential`, `log_polynomial`, `difference_polynomial`. The basis variant returns the design matrix row (used to build Q and c for QP).

- **`maxsmooth/utils.py`** — `is_positive_definite_cholesky()` for checking matrix Q.

### Branch context

The current branch (`v2-jax`) is a JAX-based rewrite. The tests in `tests/test_smooth.py` import from `maxsmooth.DCF` (the v1 API) which does not exist yet in this branch — those tests are carried over from v1 and will fail until the new API surface is implemented.

### Key design constraints

- All model functions must accept `(x, norm_x, norm_y, params)` where `norm_x = x[pivot_point]` and `norm_y = y[pivot_point]` are normalisation values.
- G matrices are row-normalised before QP to improve OSQP conditioning.
- `jax.vmap` is used to solve all sign combinations in a single batched QP call.

### Linting conventions

Ruff is configured with Google-style docstrings (`pydocstyle convention = "google"`) and enforces E/F/W, isort (I), docstrings (D), pyupgrade (UP), and type annotations (ANN). Line length is 79 characters.

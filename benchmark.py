#!/usr/bin/env python3
"""Benchmark: maxsmooth v1 vs v2, brute-force vs sign-search.

Four variants:
  v1-qp          CVXOPT, brute-force all sign combos
  v1-sign_flip   CVXOPT, sign-descent heuristic  (default in v1)
  v2-qp          JAX/OSQP, vmap over all sign combos
  v2-signsearch  JAX/OSQP, sign-search with while_loop

Toy problem: power law y = 5e7 * x**-2.5 + Gaussian noise.

Run from the repo root:
    .venv/bin/python benchmark.py
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── Worktree for master (v1) ──────────────────────────────────────────────
REPO = Path(__file__).parent
MASTER_WT = Path("/tmp/maxsmooth-master-bench")

if not MASTER_WT.exists():
    print("Creating git worktree for master branch...")
    subprocess.run(
        ["git", "worktree", "add", str(MASTER_WT), "master"],
        cwd=REPO, check=True, capture_output=True,
    )

# ── V1 imports (master on path first) ────────────────────────────────────
sys.path.insert(0, str(MASTER_WT))
from maxsmooth.DCF import smooth as _smooth_v1  # noqa: E402, I001
sys.path.pop(0)
for _k in list(sys.modules):
    if _k == "maxsmooth" or _k.startswith("maxsmooth."):
        del sys.modules[_k]

# ── V2 imports (installed package) ───────────────────────────────────────
import jax                                               # noqa: E402, I001
import jax.numpy as jnp                                  # noqa: E402, I001

jax.config.update("jax_enable_x64", True)

from maxsmooth.models import (
    difference_polynomial,
    difference_polynomial_basis                           # noqa: E402, I001
)
from maxsmooth.qp import qp as _qp_v2                   # noqa: E402, I001
from maxsmooth.qp import qpsignsearch as _qpsearch_v2   # noqa: E402, I001

# ── Toy data ──────────────────────────────────────────────────────────────
# Similar to toy data for a 21-cm problem.
rng = np.random.default_rng(42)
Ndat = 100
x_np = np.linspace(50, 200, Ndat)
y_true = 5e7 * x_np**(-2.5)
y_np = y_true + rng.normal(0, 0.25, Ndat)

x_jnp = jnp.array(x_np)
y_jnp = jnp.array(y_np)

PIVOT = Ndat // 2
N_VALUES = [4, 6, 8, 12]
REPEATS = 3


# ── V1 timing ─────────────────────────────────────────────────────────────
def time_v1(N: int, fit_type: str) -> float:
    """Return (first_s, avg_warm_s, qp_solve_count)."""

    kwargs = dict(model_type="difference_polynomial",
                  fit_type=fit_type, print_output=0)

    t0 = time.perf_counter()
    _smooth_v1(x_np, y_np, N, **kwargs)
    first = time.perf_counter() - t0

    return first


# ── V2 timing ─────────────────────────────────────────────────────────────
def time_v2(N: int, use_signsearch: bool) -> tuple[float, float]:
    """Return (cold_s, avg_warm_s, avg_deriv_s)."""
    fn = _qpsearch_v2 if use_signsearch else _qp_v2
    args = (x_jnp, y_jnp, N, PIVOT, difference_polynomial,
            difference_polynomial_basis)

    # cold call — includes XLA JIT compilation
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    cold = time.perf_counter() - t0

    warm_times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        warm_times.append(time.perf_counter() - t0)

    return cold, float(np.mean(warm_times))


# ── Run & print ───────────────────────────────────────────────────────────
def fmt(s: float) -> str:
    """Format seconds as ms if <1s, otherwise s with 2 decimal places."""
    return f"{s:.2f}s" if s >= 1.0 else f"{s*1e3:.0f}ms"


W = 115
print("\n" + "=" * W)
print(f"  maxsmooth benchmark  |  y=5e7·x^-2.5 + epsilon  |"
      f"  {Ndat} pts  |  {REPEATS} warm repeats")
print("=" * W)
print(
    f"{'N':>3}  {'combos':>7}  "
    f"{'v1-qp':>9}  {'v1-signflip':>11}  "
    f"{'v2-qp cold':>11}  {'v2-qp warm':>10}  "
    f"{'v2-search cold':>14}  {'v2-search warm':>14}  {'qp conv?':>8}"
)
print("-" * W)

all_results = {}
for N in N_VALUES:
    n_combos = 2 ** (N - 2)
    print(f"  N={N}  ...", end="", flush=True)

    v1_qp_warm = time_v1(N, "qp")
    v1_sf_warm = time_v1(N, "qp-sign_flipping")
    v2_qp_cold, v2_qp_warm = time_v2(N, use_signsearch=False)
    v2_ss_cold, v2_ss_warm = time_v2(N, use_signsearch=True)

    _, _, qp_conv = _qp_v2(
        x_jnp, y_jnp, N, PIVOT, difference_polynomial,
        difference_polynomial_basis
    )

    all_results[N] = dict(
        n_combos=n_combos,
        v1_qp=v1_qp_warm, v1_sf=v1_sf_warm,
        v2_qp_cold=v2_qp_cold, v2_qp=v2_qp_warm,
        v2_ss_cold=v2_ss_cold, v2_ss=v2_ss_warm, qp_conv=qp_conv,
    )
    print(
        f"\r  {N:>3}  {n_combos:>7}  "
        f"{fmt(v1_qp_warm):>9}  {fmt(v1_sf_warm):>11}  "
        f"{fmt(v2_qp_cold):>11}  {fmt(v2_qp_warm):>10}  "
        f"{fmt(v2_ss_cold):>14}  {fmt(v2_ss_warm):>14}"
        f"  {'YES' if qp_conv else 'NO ':>8}"
    )

print("=" * W)
print("""
  combos    = total sign combinations (2^(N-2)); v1-qp and v2-qp test ALL of them
  v1 solves = actual CVXOPT calls made by sign-descent on the last warm run
  cold      = first call, includes XLA JIT compilation (v2 only)
  qp conv?  = did qpax converge (KKT residual < 1e-3) for the winning sign combo
""")

# ── Residuals comparison ───────────────────────────────────────────────────
print("=" * W)
print("  Residuals check — chi-squared from each method (lower = better fit)")
print("=" * W)
print(f"{'N':>3}  {'combos':>7}  {'v1-qp chi2':>14}  {'v1-signflip chi2':>17}"
      f"  {'v2-qp chi2':>14}  {'v2-search chi2':>15}  {'v1/v2 agree?':>13}")
print("-" * W)

vmapped_np = jax.vmap(difference_polynomial, in_axes=(0, None, None, None))

residual_results = {}
for N in N_VALUES:
    # v1 fits
    sol_v1_qp = _smooth_v1(x_np, y_np, N, model_type="difference_polynomial",
                            fit_type="qp", pivot_point=PIVOT, print_output=0)
    sol_v1_sf = _smooth_v1(x_np, y_np, N, model_type="difference_polynomial",
                            fit_type="qp-sign_flipping", pivot_point=PIVOT,
                            print_output=0)

    # v2 fits
    params_v2_qp, chi2_v2_qp, _ = _qp_v2(
        x_jnp, y_jnp, N, PIVOT, difference_polynomial,
        difference_polynomial_basis)
    params_v2_ss, chi2_v2_ss, _ = _qpsearch_v2(
        x_jnp, y_jnp, N, PIVOT, difference_polynomial,
        difference_polynomial_basis)

    yfit_v2_qp = vmapped_np(x_jnp, x_jnp[PIVOT], y_jnp[PIVOT], params_v2_qp)

    chi2_v1_qp = float(sol_v1_qp.optimum_chi)
    chi2_v1_sf = float(sol_v1_sf.optimum_chi)
    chi2_v2_qp = float(chi2_v2_qp)
    chi2_v2_ss = float(chi2_v2_ss)

    agree = abs(chi2_v1_qp - chi2_v2_qp) / chi2_v1_qp < 0.01  # within 1%
    residual_results[N] = dict(
        chi2_v1_qp=chi2_v1_qp, chi2_v1_sf=chi2_v1_sf,
        chi2_v2_qp=chi2_v2_qp, chi2_v2_ss=chi2_v2_ss,
        yfit_v1=sol_v1_qp.y_fit, yfit_v2_qp=np.array(yfit_v2_qp),
    )
    print(
        f"  {N:>3}  {2**(N-2):>7}  "
        f"{chi2_v1_qp:>14.4g}  {chi2_v1_sf:>17.4g}"
        f"  {chi2_v2_qp:>14.4g}  {chi2_v2_ss:>15.4g}"
        f"  {'YES ✓' if agree else 'NO  ✗':>13}"
    )

print("=" * W)

# ── Plot ──────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt  # noqa: E402, I001

Ns = list(all_results.keys())
r = all_results

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0][0]
ax.plot(Ns, [r[N]["v1_qp"] * 1e3 for N in Ns],
        "o-", label="v1-qp  (CVXOPT brute)", color="steelblue", lw=2)
ax.plot(Ns, [r[N]["v1_sf"] * 1e3 for N in Ns],
        "o--", label="v1-signflip  (CVXOPT descent)", color="steelblue",
        lw=2, alpha=0.5)
ax.plot(Ns, [r[N]["v2_qp"] * 1e3 for N in Ns],
        "s-", label="v2-qp  (JAX vmap warm)", color="tomato", lw=2)
ax.plot(Ns, [r[N]["v2_ss"] * 1e3 for N in Ns],
        "s--", label="v2-signsearch  (JAX while_loop warm)", color="tomato",
        lw=2, alpha=0.5)
ax.set_xlabel("Polynomial order N")
ax.set_ylabel("Wall time (ms)")
ax.set_title("Warm-call timing (log scale)")
ax.legend(fontsize=9)
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)

ax = axes[0][1]
ax.plot(Ns, [r[N]["v2_qp_cold"] * 1e3 for N in Ns],
        "s-", label="v2-qp cold (incl. JIT)", color="tomato", lw=2)
ax.plot(Ns, [r[N]["v2_ss_cold"] * 1e3 for N in Ns],
        "s--", label="v2-signsearch cold (incl. JIT)", color="orange", lw=2)
ax.plot(Ns, [r[N]["v2_qp"] * 1e3 for N in Ns],
        "s:", label="v2-qp warm", color="tomato", lw=2, alpha=0.6)
ax.plot(Ns, [r[N]["v2_ss"] * 1e3 for N in Ns],
        "s:", label="v2-signsearch warm", color="orange", lw=2, alpha=0.6)
ax.plot(Ns, [r[N]["v1_sf"] * 1e3 for N in Ns],
        "o-", label="v1-signflip warm (reference)", color="steelblue", lw=2)
ax.set_xlabel("Polynomial order N")
ax.set_ylabel("Wall time (ms)")
ax.set_title("v2 cold vs warm (JIT overhead)")
ax.legend(fontsize=9)
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)

# ── Residuals panels ──────────────────────────────────────────────────────
colors = ["steelblue", "tomato", "seagreen", "orange", "mediumpurple", "goldenrod"]  # cycle through for each N 
N_plot = N_VALUES  # one line per N

ax = axes[1][0]
ax.scatter(x_np, y_np, s=8, color="gray", alpha=0.5, label="data", zorder=1)
for i, N in enumerate(N_plot):
    ax.plot(x_np, residual_results[N]["yfit_v1"], color=colors[i],
            lw=1.5, label=f"v1-qp  N={N}")
    ax.plot(x_np, residual_results[N]["yfit_v2_qp"], color=colors[i],
            lw=1.5, ls="--")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Fits: v1 (solid) vs v2-qp (dashed) — should overlap")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1][1]
for i, N in enumerate(N_plot):
    resid_v1 = y_np - residual_results[N]["yfit_v1"]
    resid_v2 = y_np - residual_results[N]["yfit_v2_qp"]
    ax.plot(x_np, resid_v1, color=colors[i], lw=1.5, label=f"v1-qp  N={N}")
    ax.plot(x_np, resid_v2, color=colors[i], lw=1.5, ls="--")
ax.axhline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel("x")
ax.set_ylabel("residual  (y - fit)")
ax.set_title("Residuals: v1 (solid) vs v2-qp (dashed) — should overlap")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle(
    "maxsmooth: v1 (CVXOPT) vs v2 (JAX/qpax) — brute-force vs sign-search\n"
    f"y = 5×10⁷·x⁻²·⁵ + epsilon, {Ndat} pts",
    fontsize=11,
)
fig.tight_layout()
out = REPO / "benchmark_results.png"
fig.savefig(out, dpi=150)
print(f"Plot saved → {out}")
plt.close()

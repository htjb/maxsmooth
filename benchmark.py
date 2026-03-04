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
from maxsmooth.DCF import smooth as _smooth_v1  # noqa: E402
import maxsmooth.qp as _v1_qp_mod              # noqa: E402
sys.path.pop(0)
for _k in list(sys.modules):
    if _k == "maxsmooth" or _k.startswith("maxsmooth."):
        del sys.modules[_k]

# ── V2 imports (installed package) ───────────────────────────────────────
import jax                                               # noqa: E402
import jax.numpy as jnp                                  # noqa: E402
from maxsmooth.derivatives import derivative_prefactors  # noqa: E402
from maxsmooth.models import (                           # noqa: E402
    normalised_polynomial,
    normalised_polynomial_basis,
)
from maxsmooth.qp import qp as _qp_v2                   # noqa: E402
from maxsmooth.qp import qpsignsearch as _qpsearch_v2   # noqa: E402

# ── Toy data ──────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
Ndat = 100
x_np = np.linspace(50, 200, Ndat)
y_true = 5e7 * x_np**(-2.5)
y_np = y_true + rng.normal(0, 0.01 * y_true.mean(), Ndat)

x_jnp = jnp.array(x_np)
y_jnp = jnp.array(y_np)

PIVOT = Ndat // 2
N_VALUES = [4, 6, 8]
REPEATS = 3


# ── V1 timing ─────────────────────────────────────────────────────────────
def time_v1(N: int, fit_type: str) -> tuple[float, float, int]:
    """Return (first_s, avg_warm_s, qp_solve_count)."""
    call_count = [0]
    orig = _v1_qp_mod.qp_class.__init__

    def _counted(self, *a, **kw):
        call_count[0] += 1
        orig(self, *a, **kw)

    _v1_qp_mod.qp_class.__init__ = _counted

    kwargs = dict(model_type="normalised_polynomial",
                  fit_type=fit_type, print_output=0)

    t0 = time.perf_counter()
    _smooth_v1(x_np, y_np, N, **kwargs)
    first = time.perf_counter() - t0

    warm_times = []
    for _ in range(REPEATS):
        call_count[0] = 0
        t0 = time.perf_counter()
        _smooth_v1(x_np, y_np, N, **kwargs)
        warm_times.append(time.perf_counter() - t0)
    n_solves = call_count[0]

    _v1_qp_mod.qp_class.__init__ = orig
    return first, float(np.mean(warm_times)), n_solves


# ── V2 timing ─────────────────────────────────────────────────────────────
def time_v2(N: int, use_signsearch: bool) -> tuple[float, float, float]:
    """Return (cold_s, avg_warm_s, avg_deriv_s)."""
    fn = _qpsearch_v2 if use_signsearch else _qp_v2
    args = (x_jnp, y_jnp, N, PIVOT, normalised_polynomial,
            normalised_polynomial_basis)

    # cold call — includes XLA JIT compilation
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    cold = time.perf_counter() - t0

    warm_times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        warm_times.append(time.perf_counter() - t0)

    # derivative_prefactors share of warm time
    deriv_times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        G = derivative_prefactors(
            normalised_polynomial, x_jnp,
            x_jnp[PIVOT], y_jnp[PIVOT], jnp.ones(N), N,
        )
        for g in G:
            jax.block_until_ready(g)
        deriv_times.append(time.perf_counter() - t0)

    return cold, float(np.mean(warm_times)), float(np.mean(deriv_times))


# ── Run & print ───────────────────────────────────────────────────────────
def fmt(s: float) -> str:
    return f"{s:.2f}s" if s >= 1.0 else f"{s*1e3:.0f}ms"


W = 105
print("\n" + "=" * W)
print(f"  maxsmooth benchmark  |  y=5e7·x^-2.5 + 1% noise  |"
      f"  {Ndat} pts  |  {REPEATS} warm repeats")
print("=" * W)
print(
    f"{'N':>3}  {'combos':>7}  "
    f"{'v1-qp':>9}  {'v1-signflip':>11}  {'v1 solves':>9}  "
    f"{'v2-qp cold':>11}  {'v2-qp warm':>10}  {'deriv%':>7}  "
    f"{'v2-search cold':>14}  {'v2-search warm':>14}"
)
print("-" * W)

all_results = {}
for N in N_VALUES:
    n_combos = 2 ** (N - 2)
    print(f"  N={N}  ...", end="", flush=True)

    _, v1_qp_warm, _ = time_v1(N, "qp")
    _, v1_sf_warm, v1_solves = time_v1(N, "qp-sign_flipping")
    v2_qp_cold, v2_qp_warm, v2_deriv = time_v2(N, use_signsearch=False)
    v2_ss_cold, v2_ss_warm, _ = time_v2(N, use_signsearch=True)

    deriv_pct = 100.0 * v2_deriv / v2_qp_warm if v2_qp_warm > 0 else 0

    all_results[N] = dict(
        n_combos=n_combos,
        v1_qp=v1_qp_warm, v1_sf=v1_sf_warm, v1_solves=v1_solves,
        v2_qp_cold=v2_qp_cold, v2_qp=v2_qp_warm, deriv_pct=deriv_pct,
        v2_ss_cold=v2_ss_cold, v2_ss=v2_ss_warm,
    )
    print(
        f"\r  {N:>3}  {n_combos:>7}  "
        f"{fmt(v1_qp_warm):>9}  {fmt(v1_sf_warm):>11}  {v1_solves:>9}  "
        f"{fmt(v2_qp_cold):>11}  {fmt(v2_qp_warm):>10}  {deriv_pct:>6.0f}%  "
        f"{fmt(v2_ss_cold):>14}  {fmt(v2_ss_warm):>14}"
    )

print("=" * W)
print("""
  combos       = total sign combinations (2^(N-2)); v1-qp and v2-qp test ALL of them
  v1 solves    = actual CVXOPT calls made by sign-descent on the last warm run
  deriv%       = fraction of v2-qp warm time spent in derivative_prefactors()
  cold         = first call, includes XLA JIT compilation (v2 only)
""")

# ── Plot ──────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt  # noqa: E402

Ns = list(all_results.keys())
r = all_results

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
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

ax = axes[1]
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

fig.suptitle(
    "maxsmooth: v1 (CVXOPT) vs v2 (JAX/OSQP) — brute-force vs sign-search\n"
    f"y = 5×10⁷·x⁻²·⁵ + 1% noise, {Ndat} pts",
    fontsize=11,
)
fig.tight_layout()
out = REPO / "benchmark_results.png"
fig.savefig(out, dpi=150)
print(f"Plot saved → {out}")
plt.show()

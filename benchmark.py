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

# ── Device detection ──────────────────────────────────────────────────────
CPU_DEVICE = jax.devices("cpu")[0]
try:
    GPU_DEVICE = next(
        d for d in jax.devices() if d.platform == "gpu"
    )
    HAS_GPU = True
except StopIteration:
    GPU_DEVICE = None
    HAS_GPU = False

from maxsmooth.models import (
    normalised_polynomial,
    normalised_polynomial_basis                           # noqa: E402, I001
)
from maxsmooth.qp import qp as _qp_v2                   # noqa: E402, I001
from maxsmooth.qp import qpsignsearch as _qpsearch_v2   # noqa: E402, I001

# ── Toy data ──────────────────────────────────────────────────────────────
# Similar to toy data for a 21-cm problem.
rng = np.random.default_rng(42)
Ndat = 100
x_np = np.linspace(50, 200, Ndat)
y_true = 5e7 * x_np**(-2.5)
noise = rng.normal(0, 0.025, size=Ndat)
y_np = y_true + noise

x_jnp = jnp.array(x_np)
y_jnp = jnp.array(y_np)

PIVOT = Ndat // 2
N_VALUES = [4, 6, 8, 12]
REPEATS = 3


# ── V1 timing ─────────────────────────────────────────────────────────────
def time_v1(N: int, fit_type: str) -> float:
    """Return (first_s, avg_warm_s, qp_solve_count)."""

    kwargs = dict(model_type="normalised_polynomial",
                  fit_type=fit_type, print_output=0)

    t0 = time.perf_counter()
    _smooth_v1(x_np, y_np, N, **kwargs)
    first = time.perf_counter() - t0

    return first


# ── V2 timing ─────────────────────────────────────────────────────────────
def time_v2(
    N: int, use_signsearch: bool, device=None
) -> tuple[float, float]:
    """Return (cold_s, avg_warm_s) for the given JAX device.

    The cold call includes XLA JIT compilation for this device.
    """
    if device is None:
        device = CPU_DEVICE
    fn = _qpsearch_v2 if use_signsearch else _qp_v2
    x = jax.device_put(x_jnp, device)
    y = jax.device_put(y_jnp, device)
    args = (x, y, N, PIVOT, normalised_polynomial,
            normalised_polynomial_basis, 2, N**2)

    with jax.default_device(device):
        # cold call — includes XLA JIT compilation for this device
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
header = (
    f"  maxsmooth benchmark  |  y=5e7·x^-2.5 + epsilon  |"
    f"  {Ndat} pts  |  {REPEATS} warm repeats"
)
if HAS_GPU:
    header += f"  |  GPU: {GPU_DEVICE}"

print("\n" + "=" * W)
print(header)
print("=" * W)
print(
    f"{'N':>3}  {'combos':>7}  "
    f"{'v1-qp':>9}  {'v1-signflip':>11}  "
    f"{'v2-cpu-qp cold':>14}  {'v2-cpu-qp warm':>14}  "
    f"{'v2-cpu-ss cold':>14}  {'v2-cpu-ss warm':>14}  {'qp conv?':>8}"
)
print("-" * W)

all_results = {}
for N in N_VALUES:
    n_combos = 2 ** (N - 2)
    print(f"  N={N}  ...", end="", flush=True)

    v1_qp_warm = time_v1(N, "qp")
    v1_sf_warm = time_v1(N, "qp-sign_flipping")
    v2_qp_cold, v2_qp_warm = time_v2(N, use_signsearch=False, device=CPU_DEVICE)
    v2_ss_cold, v2_ss_warm = time_v2(N, use_signsearch=True, device=CPU_DEVICE)

    with jax.default_device(CPU_DEVICE):
        _, _, qp_conv = _qp_v2(
            jax.device_put(x_jnp, CPU_DEVICE),
            jax.device_put(y_jnp, CPU_DEVICE),
            N, PIVOT, normalised_polynomial,
            normalised_polynomial_basis, max_iters=N**2,
        )

    all_results[N] = dict(
        n_combos=n_combos,
        v1_qp=v1_qp_warm, v1_sf=v1_sf_warm,
        v2_cpu_qp_cold=v2_qp_cold, v2_cpu_qp=v2_qp_warm,
        v2_cpu_ss_cold=v2_ss_cold, v2_cpu_ss=v2_ss_warm,
        qp_conv=qp_conv,
    )
    print(
        f"\r  {N:>3}  {n_combos:>7}  "
        f"{fmt(v1_qp_warm):>9}  {fmt(v1_sf_warm):>11}  "
        f"{fmt(v2_qp_cold):>14}  {fmt(v2_qp_warm):>14}  "
        f"{fmt(v2_ss_cold):>14}  {fmt(v2_ss_warm):>14}"
        f"  {'YES' if qp_conv else 'NO ':>8}"
    )

print("=" * W)

# ── GPU timing (if available) ─────────────────────────────────────────────
if HAS_GPU:
    print(f"\n  v2 on GPU ({GPU_DEVICE})")
    print("=" * W)
    print(
        f"{'N':>3}  {'combos':>7}  "
        f"{'v2-gpu-qp cold':>14}  {'v2-gpu-qp warm':>14}  "
        f"{'v2-gpu-ss cold':>14}  {'v2-gpu-ss warm':>14}"
    )
    print("-" * W)

    for N in N_VALUES:
        print(f"  N={N}  ...", end="", flush=True)
        v2_gpu_qp_cold, v2_gpu_qp_warm = time_v2(
            N, use_signsearch=False, device=GPU_DEVICE
        )
        v2_gpu_ss_cold, v2_gpu_ss_warm = time_v2(
            N, use_signsearch=True, device=GPU_DEVICE
        )
        all_results[N].update(
            v2_gpu_qp_cold=v2_gpu_qp_cold, v2_gpu_qp=v2_gpu_qp_warm,
            v2_gpu_ss_cold=v2_gpu_ss_cold, v2_gpu_ss=v2_gpu_ss_warm,
        )
        print(
            f"\r  {N:>3}  {2**(N-2):>7}  "
            f"{fmt(v2_gpu_qp_cold):>14}  {fmt(v2_gpu_qp_warm):>14}  "
            f"{fmt(v2_gpu_ss_cold):>14}  {fmt(v2_gpu_ss_warm):>14}"
        )
    print("=" * W)

print("""
  combos   = total sign combinations (2^(N-2)); v1-qp and v2-qp test ALL of them
  cold     = first call, includes XLA JIT compilation (v2 only; per-device)
  qp conv? = did qpax converge (KKT residual < 1e-3) for the winning sign combo
""")

# ── Residuals comparison ───────────────────────────────────────────────────
print("=" * W)
print("  Residuals check — chi-squared from each method (lower = better fit)")
print("=" * W)
print(f"{'N':>3}  {'combos':>7}  {'v1-qp chi2':>14}  {'v1-signflip chi2':>17}"
      f"  {'v2-qp chi2':>14}  {'v2-search chi2':>15}  {'v1/v2 agree?':>13}")
print("-" * W)

vmapped_np = jax.vmap(normalised_polynomial, in_axes=(0, None, None, None))

residual_results = {}
for N in N_VALUES:
    # v1 fits
    sol_v1_qp = _smooth_v1(x_np, y_np, N, model_type="normalised_polynomial",
                            fit_type="qp", pivot_point=PIVOT, print_output=0)
    sol_v1_sf = _smooth_v1(x_np, y_np, N, model_type="normalised_polynomial",
                            fit_type="qp-sign_flipping", pivot_point=PIVOT,
                            print_output=0)

    # v2 fits
    params_v2_qp, chi2_v2_qp, _ = _qp_v2(
        x_jnp, y_jnp, N, PIVOT, normalised_polynomial,
        normalised_polynomial_basis, max_iters=N**2)
    params_v2_ss, chi2_v2_ss, _ = _qpsearch_v2(
        x_jnp, y_jnp, N, PIVOT, normalised_polynomial,
        normalised_polynomial_basis, max_iters=N**2)

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

fig, (ax_cold, ax_resid) = plt.subplots(2, 1, figsize=(8, 6))


ax = ax_cold
ax.plot(Ns, [r[N]["v2_cpu_qp_cold"] * 1e3 for N in Ns],
        "s-", label="v2-cpu-qp cold (incl. JIT)", color="tomato", lw=2)
ax.plot(Ns, [r[N]["v2_cpu_ss_cold"] * 1e3 for N in Ns],
        "s--", label="v2-cpu-signsearch cold (incl. JIT)", color="orange", lw=2)
ax.plot(Ns, [r[N]["v2_cpu_qp"] * 1e3 for N in Ns],
        "s:", label="v2-cpu-qp warm", color="tomato", lw=2, alpha=0.6)
ax.plot(Ns, [r[N]["v2_cpu_ss"] * 1e3 for N in Ns],
        "s:", label="v2-cpu-signsearch warm", color="orange", lw=2, alpha=0.6)
if HAS_GPU:
    ax.plot(Ns, [r[N]["v2_gpu_qp_cold"] * 1e3 for N in Ns],
            "^-", label="v2-gpu-qp cold (incl. JIT)", color="tomato", lw=2,
            alpha=0.5)
    ax.plot(Ns, [r[N]["v2_gpu_ss_cold"] * 1e3 for N in Ns],
            "^--", label="v2-gpu-signsearch cold (incl. JIT)", color="orange",
            lw=2, alpha=0.5)
    ax.plot(Ns, [r[N]["v2_gpu_qp"] * 1e3 for N in Ns],
            "^:", label="v2-gpu-qp warm", color="tomato", lw=2, alpha=0.35)
    ax.plot(Ns, [r[N]["v2_gpu_ss"] * 1e3 for N in Ns],
            "^:", label="v2-gpu-signsearch warm", color="orange", lw=2,
            alpha=0.35)
ax.plot(Ns, [r[N]["v1_qp"] * 1e3 for N in Ns],
        "o-", label="v1-qp  (CVXOPT brute)", color="steelblue", lw=2)
ax.plot(Ns, [r[N]["v1_sf"] * 1e3 for N in Ns],
        "o-", label="v1-signsearch (CVXOPT)", color="steelblue", lw=2, ls=":")
ax.set_xlabel("Polynomial order N")
ax.set_ylabel("Wall time (ms)")
gpu_info = f" | GPU: {GPU_DEVICE}" if HAS_GPU else " | no GPU"
ax.set_title(f"Timing (squares=CPU, triangles=GPU){gpu_info}")
ax.legend(fontsize=8)
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)

# ── Residuals panels ──────────────────────────────────────────────────────
colors = ["steelblue", "tomato", "seagreen", "orange", "mediumpurple", "goldenrod"]  # cycle through for each N 
N_plot = N_VALUES  # one line per N

ax = ax_resid
for i, N in enumerate(N_plot):
    resid_v1 = y_np - residual_results[N]["yfit_v1"]
    resid_v2 = y_np - residual_results[N]["yfit_v2_qp"]
    ax.plot(x_np, resid_v1, color=colors[i], lw=1.5, label=f"v1-qp  N={N}")
    ax.plot(x_np, resid_v2, color=colors[i], lw=1.5, ls="--")
ax.plot(x_np, noise, color="k", lw=0.8, ls=":", label="data noise")
ax.axhline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel("x")
ax.set_ylabel("residual  (y - fit)")
ax.set_title("Residuals")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-1, 1)

gpu_subtitle = f" | GPU: {GPU_DEVICE}" if HAS_GPU else ""
fig.suptitle(
    "maxsmooth: v1 (CVXOPT) vs v2 (JAX/qpax) — brute-force vs sign-search\n"
    f"y = 5×10⁷·x⁻²·⁵ + epsilon, {Ndat} pts{gpu_subtitle}",
    fontsize=11,
)
fig.tight_layout()
out = REPO / "benchmark_results.png"
fig.savefig(out, dpi=350)
print(f"Plot saved → {out}")
plt.close()

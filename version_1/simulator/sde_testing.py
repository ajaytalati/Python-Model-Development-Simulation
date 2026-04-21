"""
sde_testing.py — Generic Cross-Validation and Verification
============================================================
Date:    15 April 2026
Version: 1.0

Model-agnostic testing: compares solvers, verifies physics.
"""

import time
import numpy as np
from sde_model import SDEModel
from sde_solver_scipy import solve_deterministic, solve_sde


def cross_validate_deterministic(model, params, init_state, t_grid,
                                  exogenous=None, tolerance=1e-5):
    """Compare scipy BDF vs Diffrax Kvaerno5 on the deterministic ODE.

    Requires: sde_solver_diffrax importable (JAX + Diffrax installed).
    """
    from simulator.sde_solver_diffrax import solve_deterministic_jax

    exogenous = exogenous or {}

    print("  scipy BDF...", end="", flush=True)
    t0 = time.time()
    traj_s = solve_deterministic(model, params, init_state, t_grid, exogenous)
    print(f" {time.time()-t0:.1f}s")

    print("  Diffrax Kvaerno5...", end="", flush=True)
    t0 = time.time()
    traj_d = solve_deterministic_jax(model, params, init_state, t_grid, exogenous)
    print(f" {time.time()-t0:.1f}s")

    all_pass = True
    print(f"\n  {'State':<12} {'MaxErr':>12} {'Tol':>12} {'':>8}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*8}")

    for i, state in enumerate(model.states):
        err = float(np.max(np.abs(traj_s[:, i] - traj_d[:, i])))
        ok = err < tolerance
        all_pass = all_pass and ok
        print(f"  {state.name:<12} {err:>12.2e} {tolerance:>12.2e} "
              f"{'PASS' if ok else 'FAIL':>8}")

    tag = "PASS" if all_pass else "FAIL"
    print(f"\n  {tag}: deterministic cross-validation")
    return {'all_pass': all_pass, 'traj_scipy': traj_s, 'traj_diffrax': traj_d}


def verify_physics(model, trajectory, t_grid, params):
    """Run model-specific physics checks.

    Returns dict of {check_name: bool} if model provides verify_physics_fn.
    Returns empty dict otherwise.
    """
    if model.verify_physics_fn is None:
        print("  No physics verification function provided.")
        return {}

    results = model.verify_physics_fn(trajectory, t_grid, params)
    all_pass = all(v for v in results.values() if isinstance(v, bool))

    for name, value in results.items():
        if isinstance(value, bool):
            print(f"  {name:<30} {'PASS' if value else 'FAIL':>8}")
        else:
            print(f"  {name:<30} {value}")

    print(f"\n  {'PASS' if all_pass else 'FAIL'}: physics verification")
    return results

"""verify_swat_state.py — prove the SWAT code is in the V_h-anabolic state.

Run from the version_1 root:
    PYTHONPATH=. python3 models/swat/verify_swat_state.py

It does FIVE things:
    1. Reads the exact text of `_dynamics.entrainment_quality` — the
       output must contain the V_h-anabolic amplitude formula, the
       V_n damper, and the clamped phase quality.
    2. Reads `simulation.drift` — must contain the V_h-anabolic
       inline expressions matching _dynamics.
    3. Checks PARAM_SET_A T_0, PARAM_SET_B V_n, PARAM_SET_D existence.
    4. Runs the four canonical param sets (A/B/C/D) end-to-end on the
       scipy solver, 14 days, seed=42.
    5. Prints the T trajectory end-values side-by-side.

Pass criteria (all four must hold) — recalibrated for the V_h-anabolic
fix:
    • Set A T_end ≈ 0.90 (stable near new T*; healthy V_h drives strong
       entrainment, mu(E)=mu_0+mu_E ≈ 0.5 so A* ≈ sqrt(0.5/0.5) ≈ 1)
    • Set B T_end ≈ 0.11 (V_h-depleted AND V_n-high; entrainment
       collapses, T → 0)
    • Set C T_end ≈ 0.82 (healthy V_h, recovery from T_0=0.05 — even
       stronger basin than A because no IC penalty)
    • Set D T_end ≈ 0.11 (V_c=6 hours > V_c_max=3 hours clamp;
       phase quality = 0 regardless of V_h)
"""
import os
import sys
import re
import shutil
import subprocess


# ─── Step 0: clear any stale bytecode ──────────────────────────────────────

def clear_caches(root):
    """Remove every __pycache__ directory under root."""
    n = 0
    for dirpath, dirnames, _ in os.walk(root):
        for d in list(dirnames):
            if d == '__pycache__':
                shutil.rmtree(os.path.join(dirpath, d))
                dirnames.remove(d)
                n += 1
    return n


# ─── Step 1+2: textual checks on the source files ─────────────────────────

EXPECTED_DYNAMICS_PATTERNS = [
    r"V_c\s*=\s*params\[pi\['V_c'\]\]",
    # V_h enters via amplitude, NOT directly into u_W
    r"A_W\s*=\s*lambda_amp_W\s*\*\s*Vh",
    r"A_Z\s*=\s*lambda_amp_Z\s*\*\s*Vh",
    # Clamped quarter-period amplitude
    r"amp_W\s*=\s*jax\.nn\.sigmoid\(B_W\s*\+\s*A_W\)\s*-\s*jax\.nn\.sigmoid\(B_W\s*-\s*A_W\)",
    # V_n damper
    r"damp\s*=\s*jnp\.exp\(-Vn\s*/\s*V_n_scale\)",
    # Clamped phase
    r"V_c_eff\s*=\s*jnp\.minimum\(jnp\.abs\(V_c\),\s*V_C_MAX_HOURS\)",
    r"return damp\s*\*\s*amp_quality\s*\*\s*phase_quality",
]

EXPECTED_DRIFT_NP_PATTERNS = [
    # V_h enters via amplitude in numpy entrainment_quality + drift
    r"A_W\s*=\s*p\['lambda_amp_W'\]\s*\*\s*Vh",
    # V_n damper present
    r"damp\s*=\s*math\.exp\(-Vn\s*/\s*p\['V_n_scale'\]\)",
    # Clamped phase
    r"V_c_eff\s*=\s*min\(abs\(V_c\),\s*V_C_MAX_HOURS\)",
    r"phase_quality\s*=\s*math\.cos\(math\.pi\s*\*\s*V_c_eff\s*/",
    r"E\s*=\s*damp\s*\*\s*amp_quality\s*\*\s*phase_quality",
]

EXPECTED_PARAM_PATTERNS = [
    (r"'T_T':\s*0\.0001", "PARAM_SET_A T_T = 0.0001 (not 0.01)"),
    (r"'T_0':\s*0\.5", "PARAM_SET_A T_0 = 0.5 (not 1.0)"),
    (r"INIT_STATE_B\['Vn'\]\s*=\s*3\.5", "PARAM_SET_B V_n raised to 3.5"),
    (r"PARAM_SET_D\s*=\s*dict\(PARAM_SET_A\)", "PARAM_SET_D exists"),
    (r"PARAM_SET_D\['V_c'\]\s*=\s*6\.0", "PARAM_SET_D V_c = 6.0"),
]


def check_file(path, patterns, label):
    """Return (ok, missing_patterns)."""
    if not os.path.exists(path):
        return False, [f"FILE MISSING: {path}"]
    text = open(path).read()
    missing = []
    for p in patterns:
        pat, desc = p if isinstance(p, tuple) else (p, p)
        if not re.search(pat, text):
            missing.append(desc)
    ok = not missing
    print(f"  {'✓' if ok else '✗'} {label}: "
          f"{len(patterns) - len(missing)}/{len(patterns)} patterns matched")
    for m in missing:
        print(f"      MISSING: {m}")
    return ok, missing


# ─── Step 3+4: run each param set ─────────────────────────────────────────

def run_one_set(set_name, seed=42):
    """Import the model fresh, run 14 days, return dict of summary stats."""
    # Force fresh import
    for modname in list(sys.modules):
        if modname.startswith('models.swat'):
            del sys.modules[modname]

    import numpy as np
    from models.swat.simulation import SWAT_MODEL
    from simulator.sde_solver_scipy import solve_sde
    from simulator.sde_observations import generate_all_channels

    params = SWAT_MODEL.param_sets[set_name]
    init   = SWAT_MODEL.init_states[set_name]
    exog   = SWAT_MODEL.exogenous_inputs[set_name]
    dt     = params.get('dt_hours', 5.0 / 60.0)
    t_tot  = params.get('t_total_hours', 14 * 24.0)
    n      = int(t_tot / dt)
    t_grid = np.arange(n) * dt

    rng = np.random.default_rng(seed)
    sde_seed = int(rng.integers(0, 2**31))

    traj = solve_sde(SWAT_MODEL, params, init, t_grid, exog,
                     seed=sde_seed, n_substeps=10)

    one_day = int(24 / dt)
    return {
        'name':     set_name,
        'V_c':      params['V_c'],
        'V_h':      init['Vh'],
        'V_n':      init['Vn'],
        'T_0':      init['T_0'],
        'T_end':    float(traj[-one_day:, 3].mean()),
        'T_mean':   float(traj[:, 3].mean()),
        'T_std':    float(traj[:, 3].std()),
        'W_range':  float(traj[:, 0].max() - traj[:, 0].min()),
        'Zt_range': float(traj[:, 1].max() - traj[:, 1].min()),
    }


# ─── Main ─────────────────────────────────────────────────────────────────

# Expected T_end values (seed=42, 14 days, n_substeps=10) — recalibrated
# for the V_h-anabolic structural fix.
EXPECTED = {
    'A': (0.90, 0.08),   # healthy basin, strong entrainment
    'B': (0.11, 0.08),   # V_h-depleted + V_n-high collapse
    'C': (0.82, 0.10),   # recovery from T_0=0.05 under healthy V_h
    'D': (0.11, 0.08),   # V_c=6 > V_c_max=3 clamps phase to 0
}


def main():
    framework_root = os.getcwd()
    model_dir = os.path.join(framework_root, 'models', 'swat')

    print("=" * 65)
    print("  SWAT state verification")
    print("=" * 65)

    # Step 0
    print(f"\nCwd: {framework_root}")
    n_cleared = clear_caches(framework_root)
    print(f"Cleared {n_cleared} __pycache__ directories\n")

    # Step 1+2+3: textual checks
    print("── Source file content checks ──────────────────────────────")
    all_ok = True
    ok, _ = check_file(os.path.join(model_dir, '_dynamics.py'),
                       EXPECTED_DYNAMICS_PATTERNS,
                       "_dynamics.entrainment_quality V_c-aware")
    all_ok &= ok
    ok, _ = check_file(os.path.join(model_dir, 'simulation.py'),
                       EXPECTED_DRIFT_NP_PATTERNS,
                       "simulation.drift (numpy) V_c-aware")
    all_ok &= ok
    ok, _ = check_file(os.path.join(model_dir, 'simulation.py'),
                       EXPECTED_PARAM_PATTERNS,
                       "param-set updates applied")
    all_ok &= ok

    if not all_ok:
        print("\n*** SOURCE FILES DO NOT MATCH EXPECTED V_c-AWARE STATE. ***")
        print("*** The agent is reading a stale copy of the code.      ***")
        sys.exit(1)

    # Step 4: run four param sets
    print("\n── Running all four param sets on scipy solver ─────────────")
    results = {}
    for s in ['A', 'B', 'C', 'D']:
        results[s] = run_one_set(s)
        r = results[s]
        print(f"  Set {s}: V_c={r['V_c']:3.1f}  V_h={r['V_h']}  "
              f"V_n={r['V_n']}  T_0={r['T_0']:.2f}  ->  T_end={r['T_end']:.3f}")

    # Step 5: compare vs expected
    print("\n── Pass/fail vs expected ───────────────────────────────────")
    all_pass = True
    for s, (expected, tol) in EXPECTED.items():
        actual = results[s]['T_end']
        delta = actual - expected
        passed = abs(delta) <= tol
        all_pass &= passed
        tag = "PASS" if passed else "FAIL"
        print(f"  Set {s}: expected T_end ≈ {expected:.2f} ± {tol:.2f}, "
              f"got {actual:.3f} ({delta:+.3f})  [{tag}]")

    # Step 6: discrimination sanity
    print("\n── Discrimination sanity ───────────────────────────────────")
    dA = results['A']['T_end']; dB = results['B']['T_end']
    dC = results['C']['T_end']; dD = results['D']['T_end']
    print(f"  Set A vs Set D (V_c = 0 vs 6):  "
          f"T_end {dA:.3f} vs {dD:.3f}  "
          f"(ratio {dA/dD:.1f}x)  "
          f"{'[V_c-AWARE]' if abs(dA - dD) > 0.2 else '[V_c-BLIND]'}")
    print(f"  Set A vs Set B (healthy vs insomnia): "
          f"T_end {dA:.3f} vs {dB:.3f}  "
          f"(ratio {dA/dB:.1f}x)  "
          f"{'[Set-B WORKING]' if abs(dA - dB) > 0.2 else '[Set-B BROKEN]'}")
    print(f"  Set C recovery: {dC:.3f} (expected > 0.3)  "
          f"{'[RECOVERY WORKING]' if dC > 0.3 else '[RECOVERY BROKEN]'}")

    print("\n" + ("=" * 65))
    if all_pass:
        print("  ALL CHECKS PASS — V_c-aware dynamics confirmed.")
        sys.exit(0)
    else:
        print("  CHECKS FAILED — see individual lines above.")
        sys.exit(2)


if __name__ == '__main__':
    main()

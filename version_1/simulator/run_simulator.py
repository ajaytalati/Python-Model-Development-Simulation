"""
run_simulator.py — Model-Agnostic SDE Simulation CLI
======================================================
Date:    16 April 2026
Version: 2.0

Lives inside simulator/ alongside sde_model.py, sde_observations.py, etc.
Model definitions live in models/<name>/simulation.py (one folder per model).

Usage (from the framework root):
    python simulator/run_simulator.py
    python simulator/run_simulator.py --model models.ou_test.simulation.OU_MODEL --scipy
    python simulator/run_simulator.py --cross-validate
    python simulator/run_simulator.py --verify
"""

import sys
import os
import time
import importlib
import numpy as np

# ── Path setup: add both simulator/ and the framework root to sys.path ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)      # so bare 'from sde_model import' works
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)       # so 'models.sleep_wake.simulation' works

from sde_model import SDEModel
from sde_solver_scipy import solve_deterministic
from sde_observations import generate_all_channels


def _resolve_solver(name, model):
    """Return (solve_sde_fn, description) for the requested solver.

    Falls back to scipy with a warning if JAX/Diffrax is requested but
    not available, or if the model doesn't provide drift_fn_jax.
    """
    if name == 'diffrax':
        if model.drift_fn_jax is None or model.make_aux_fn_jax is None:
            print(f"  WARNING: model '{model.name}' has no JAX drift — "
                  f"falling back to scipy Euler-Maruyama")
            from sde_solver_scipy import solve_sde
            return solve_sde, 'scipy Euler-Maruyama (fallback)'
        try:
            from sde_solver_diffrax import solve_sde_jax
            return solve_sde_jax, 'JAX/Diffrax scan Euler-Maruyama (GPU-JIT)'
        except ImportError as e:
            print(f"  WARNING: JAX/Diffrax not available ({e}) — "
                  f"falling back to scipy")
            from sde_solver_scipy import solve_sde
            return solve_sde, 'scipy Euler-Maruyama (fallback)'
    elif name == 'scipy':
        from sde_solver_scipy import solve_sde
        return solve_sde, 'scipy Euler-Maruyama'
    else:
        raise ValueError(f"Unknown solver: {name}")


def _resolve_model(model_path):
    """Import a model object from a dotted path like 'models.sleep_wake.simulation.SLEEP_WAKE_MODEL'."""
    parts = model_path.rsplit('.', 1)
    if len(parts) != 2:
        raise ValueError(f"Model path must be 'module.OBJECT', got: {model_path}")
    module_path, obj_name = parts
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        print(f"\n  ERROR: Failed to import '{module_path}'")
        print(f"  Root cause: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    if not hasattr(mod, obj_name):
        available = [n for n in dir(mod) if not n.startswith('_')]
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{obj_name}'.\n"
            f"  This usually means an exception occurred while the module\n"
            f"  was being imported — check for warnings above.\n"
            f"  Available top-level names: {available}"
        )
    return getattr(mod, obj_name)


def _parse_args(argv):
    """Simple argument parser (no external dependencies)."""
    args = {
        'model_path': 'models.sleep_wake.simulation.SLEEP_WAKE_MODEL',
        'param_set': 'A',
        'seed': 42,
        'solver': 'diffrax',
        'n_substeps': 10,
        'cross_validate': False,
        'verify': False,
        'out_dir': 'outputs',
    }
    i = 1
    while i < len(argv):
        if argv[i] == '--model' and i + 1 < len(argv):
            args['model_path'] = argv[i + 1]; i += 2
        elif argv[i] == '--param-set' and i + 1 < len(argv):
            args['param_set'] = argv[i + 1].upper(); i += 2
        elif argv[i] == '--seed' and i + 1 < len(argv):
            args['seed'] = int(argv[i + 1]); i += 2
        elif argv[i] == '--substeps' and i + 1 < len(argv):
            args['n_substeps'] = int(argv[i + 1]); i += 2
        elif argv[i] == '--cross-validate':
            args['cross_validate'] = True; i += 1
        elif argv[i] == '--verify':
            args['verify'] = True; i += 1
        elif argv[i] == '--scipy':
            args['solver'] = 'scipy'; i += 1
        elif argv[i] == '--diffrax':
            args['solver'] = 'diffrax'; i += 1
        elif argv[i] == '--out-dir' and i + 1 < len(argv):
            args['out_dir'] = argv[i + 1]; i += 2
        else:
            i += 1
    return args


def main():
    cli = _parse_args(sys.argv)
    model = _resolve_model(cli['model_path'])

    print("=" * 65)
    print("  SDE TESTING DATA FRAMEWORK — SIMULATOR")
    print(f"  Date:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Version: 2.0")
    print(f"  Model:   {model.name} v{model.version}")
    print(f"  States:  {model.n_states} ({', '.join(model.state_names)})")
    print(f"  Channels: {len(model.channels)} ({', '.join(c.name for c in model.channels)})")
    print(f"  Param set: {cli['param_set']}")
    print(f"  Solver:  {cli['solver']}")
    print(f"  Seed:    {cli['seed']}")
    print("=" * 65)

    # Resolve parameter set
    ps = cli['param_set']
    if model.param_sets is None or ps not in model.param_sets:
        available = list(model.param_sets.keys()) if model.param_sets else []
        print(f"\n  ERROR: param set '{ps}' not found. Available: {available}")
        sys.exit(1)

    params = model.param_sets[ps]
    init_state = (model.init_states or {}).get(ps,
                  list(model.init_states.values())[0] if model.init_states else {})
    exogenous = (model.exogenous_inputs or {}).get(ps, {})

    # Build time grid from model parameters or default
    dt = params.get('dt_hours', 5.0 / 60.0)
    t_total = params.get('t_total_hours', 9 * 24.0)
    n_steps = int(t_total / dt)
    t_grid = np.arange(n_steps) * dt

    print(f"  Grid:    {n_steps} steps × {dt*60:.0f}-min = {t_total/24:.0f} days")

    # ── Cross-validation ──
    if cli['cross_validate']:
        from sde_testing import cross_validate_deterministic, verify_physics
        print(f"\n{'='*65}")
        print(f"  DETERMINISTIC CROSS-VALIDATION")
        print(f"{'='*65}")
        p_det = {**params}
        for state in model.states:
            for noise_key in [f'T_{state.name}', f'sigma_{state.name}']:
                if noise_key in p_det:
                    p_det[noise_key] = 0.0
        cross_validate_deterministic(model, p_det, init_state, t_grid, exogenous)
        return

    # ── Verify only ──
    if cli['verify']:
        from sde_testing import verify_physics
        p_det = {**params}
        for state in model.states:
            for noise_key in [f'T_{state.name}', f'sigma_{state.name}']:
                if noise_key in p_det:
                    p_det[noise_key] = 0.0
        traj = solve_deterministic(model, p_det, init_state, t_grid, exogenous)
        print(f"\n  Physics verification:")
        verify_physics(model, traj, t_grid, params)
        return

    # ── Generate synthetic data ──
    print(f"\n  Simulating SDE...")

    solve_sde_fn, solver_desc = _resolve_solver(cli['solver'], model)
    print(f"  Solver: {solver_desc}")

    t0 = time.time()

    master_rng = np.random.default_rng(cli['seed'])
    sde_seed = int(master_rng.integers(0, 2**31))
    obs_seed = int(master_rng.integers(0, 2**31))

    trajectory = solve_sde_fn(model, params, init_state, t_grid, exogenous,
                              seed=sde_seed, n_substeps=cli['n_substeps'])
    t_sde = time.time() - t0
    print(f"  SDE solved in {t_sde:.1f}s")

    # State ranges
    print(f"\n  State ranges:")
    for i, s in enumerate(model.states):
        lo, hi = trajectory[:, i].min(), trajectory[:, i].max()
        print(f"    {s.name:>6s}: [{lo:.4f}, {hi:.4f}]")

    # Generate observations
    print(f"\n  Generating observations...")
    aux = model.make_aux_fn(params, init_state, t_grid, exogenous) if model.make_aux_fn else None
    channel_outputs = generate_all_channels(
        model, trajectory, t_grid, params, aux, seed=obs_seed)

    for ch_name, ch_data in channel_outputs.items():
        n = len(next(iter(ch_data.values())))
        print(f"    {ch_name}: {n} observations")

    # Save
    run_name = f"synthetic_{model.name}_{ps}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(cli['out_dir'], exist_ok=True)
    out_dir = os.path.join(cli['out_dir'], run_name)
    os.makedirs(out_dir, exist_ok=True)

    np.savez(os.path.join(out_dir, "synthetic_truth.npz"),
             true_trajectory=trajectory,
             t_grid=t_grid,
             true_params=np.array(list(params.values())),
             true_param_names=np.array(list(params.keys())),
             true_init_states=np.array(list(init_state.values())),
             true_init_names=np.array(list(init_state.keys())))

    for ch_name, ch_data in channel_outputs.items():
        np.savez(os.path.join(out_dir, f"channel_{ch_name}.npz"), **ch_data)

    # Save exogenous inputs so recovery scripts can use the same values
    # that generated the data (fixes the daily_active_bins zero-fallback bug).
    if exogenous:
        exog_saveable = {}
        for k, v in exogenous.items():
            try:
                exog_saveable[k] = np.asarray(v)
            except Exception:
                pass  # skip anything that can't be serialised to ndarray
        if exog_saveable:
            np.savez(os.path.join(out_dir, "synthetic_exogenous.npz"),
                     **exog_saveable)

    print(f"\n  Saved: {out_dir}/")

    # Physics verification
    if model.verify_physics_fn:
        print(f"\n  Physics verification:")
        from sde_testing import verify_physics
        verify_physics(model, trajectory, t_grid, params)

    # Plot
    if model.plot_fn:
        print(f"\n  Generating plots...")
        model.plot_fn(trajectory, t_grid, channel_outputs, params, out_dir)

    # Write pipeline-compatible CSVs
    if model.csv_writer_fn:
        csv_dir = os.path.join(out_dir, "garmin_csv")
        print(f"\n  Writing Garmin-format CSVs for pipeline testing...")
        sim_meta = {
            'param_set_name': cli['param_set'],
            'sde_seed': int(sde_seed),
            'obs_seed': int(obs_seed),
            'n_substeps': int(cli['n_substeps']),
            'solver': cli['solver'],
            'solver_description': solver_desc,
            'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        try:
            model.csv_writer_fn(trajectory, t_grid, channel_outputs, params,
                                csv_dir, init_state=init_state,
                                exogenous=exogenous, meta=sim_meta)
        except TypeError:
            model.csv_writer_fn(trajectory, t_grid, channel_outputs, params, csv_dir)
        print(f"  CSV dir: {csv_dir}/")

    t_total_wall = time.time() - t0
    print(f"\n  Total wall time: {t_total_wall:.1f}s")
    print(f"  Done.")


if __name__ == "__main__":
    main()

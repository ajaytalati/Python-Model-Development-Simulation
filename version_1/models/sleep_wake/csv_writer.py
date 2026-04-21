"""
models/sleep_wake_csv_writer.py — Garmin-Format CSV Export
============================================================
Date:    15 April 2026
Version: 1.1

Writes synthetic observations as CSV files that are byte-compatible
with the real Garmin export files.  This allows the ENTIRE estimation
pipeline to be tested end-to-end:

    CSV files → loader.py → thinning → ObservationData → particle filter

The loader's coarse-graining / thinning is tested ONCE by this path.
The filter / MCLMC is the focus of testing.

Output files (same names as real Garmin exports):
    sdk_intraday_hr.csv       1-min HR readings
    sdk_intraday_stress.csv   3-min stress readings
    sleep_stages.csv          Variable-length sleep epochs
    sdk_steps_binned.csv      15-min step count bins
    ground_truth.json         Ground-truth parameters + metadata

Format reference: real CSV files in my_intraday_data/
"""

import os
import json
import math
import numpy as np
import pandas as pd


# =========================================================================
# TIMESTAMP HELPERS
# =========================================================================

def _make_data_start(params):
    """Build a realistic UTC start timestamp for the synthetic data.

    Uses 2026-04-05 00:00:00+00:00 to match the real Garmin data window.
    Can be overridden via params['data_start_utc'].
    """
    start_str = params.get('data_start_utc', '2026-04-05 00:00:00+00:00')
    return pd.Timestamp(start_str)


def _hours_to_utc(t_hours, t0):
    """Convert hours-since-t0 to UTC timestamps, rounded to nearest minute."""
    total_minutes = round(t_hours * 60)
    return t0 + pd.Timedelta(minutes=total_minutes)


def _utc_to_bst(ts):
    """Convert UTC timestamp to BST (UTC+1) for step data."""
    return ts.tz_convert('Europe/London')


# =========================================================================
# CHANNEL 1: HEART RATE (1-minute resolution)
# =========================================================================

def _write_hr_csv(trajectory, t_grid, channel_outputs, params, save_dir):
    """Write sdk_intraday_hr.csv from the HR channel data.

    Pure formatter: takes channel_outputs['hr'] (already at native 1-min
    resolution) and emits CSV rows.  No regeneration, no fresh RNG.

    Columns: timestamp_utc, heart_rate_bpm, source_file
    """
    if 'hr' not in channel_outputs:
        raise KeyError("_write_hr_csv requires channel_outputs['hr']; "
                       "gen_hr must have run.")

    t0 = _make_data_start(params)
    hr_data = channel_outputs['hr']
    t_minutes = hr_data['t_minutes']
    bpms = hr_data['hr_bpm']

    timestamps = [t0 + pd.Timedelta(minutes=int(m)) for m in t_minutes]
    source_files = [
        f"synthetic_{424000000000 + (ts - t0).days}_WELLNESS.fit"
        for ts in timestamps
    ]

    df = pd.DataFrame({
        'timestamp_utc': timestamps,
        'heart_rate_bpm': bpms.astype(int),
        'source_file': source_files,
    })

    path = os.path.join(save_dir, 'sdk_intraday_hr.csv')
    df.to_csv(path, index=False)
    return path, len(df)


# =========================================================================
# CHANNEL 2: STRESS (3-minute resolution)
# =========================================================================

def _stress_category(score):
    """Map stress score to Garmin category string."""
    if score < 26:
        return 'rest'
    elif score < 51:
        return 'low'
    elif score < 76:
        return 'medium'
    else:
        return 'high'


def _write_stress_csv(trajectory, t_grid, channel_outputs, params, save_dir):
    """Write sdk_intraday_stress.csv from the stress channel data.

    Pure formatter: takes channel_outputs['stress'] (already at native
    3-min resolution) and emits CSV rows.

    Columns: timestamp_utc, stress_score, stress_category, source_file
    """
    if 'stress' not in channel_outputs:
        raise KeyError("_write_stress_csv requires channel_outputs['stress']; "
                       "gen_stress must have run.")

    t0 = _make_data_start(params)
    stress_data = channel_outputs['stress']
    t_minutes = stress_data['t_minutes']
    scores = stress_data['stress_score'].astype(int)

    timestamps = [t0 + pd.Timedelta(minutes=int(m)) for m in t_minutes]
    categories = [_stress_category(int(s)) for s in scores]
    source_files = [
        f"synthetic_{424000000000 + (ts - t0).days}_WELLNESS.fit"
        for ts in timestamps
    ]

    df = pd.DataFrame({
        'timestamp_utc': timestamps,
        'stress_score': scores,
        'stress_category': categories,
        'source_file': source_files,
    })

    path = os.path.join(save_dir, 'sdk_intraday_stress.csv')
    df.to_csv(path, index=False)
    return path, len(df)


# =========================================================================
# CHANNEL 3: SLEEP STAGES (variable-length epochs)
# =========================================================================

def _write_sleep_csv(trajectory, t_grid, channel_outputs, params, save_dir):
    """Write sleep_stages.csv from the sleep channel data.

    Pure formatter: takes channel_outputs['sleep'] (epoch run-length
    encoded at 1-min resolution) and emits CSV rows.

    Columns: sleep_date, epoch_start, epoch_end, duration_min,
             sleep_level, is_nap, source_file
    """
    if 'sleep' not in channel_outputs:
        raise KeyError("_write_sleep_csv requires channel_outputs['sleep']; "
                       "gen_sleep must have run.")

    t0 = _make_data_start(params)
    sleep_data = channel_outputs['sleep']
    starts_min = sleep_data['epoch_start_min']
    ends_min = sleep_data['epoch_end_min']
    labels = sleep_data['labels']

    level_map = {0: 'awake', 1: 'light', 2: 'rem', 3: 'deep'}
    rows = []
    for sm, em, lab in zip(starts_min, ends_min, labels):
        duration = int(em - sm)
        if duration < 1:
            continue
        epoch_start = t0 + pd.Timedelta(minutes=int(sm))
        epoch_end = t0 + pd.Timedelta(minutes=int(em))
        sleep_date = epoch_end.strftime('%Y-%m-%d')
        day_num = (epoch_start - t0).days
        rows.append({
            'sleep_date': sleep_date,
            'epoch_start': epoch_start,
            'epoch_end': epoch_end,
            'duration_min': float(duration),
            'sleep_level': level_map[int(lab)],
            'is_nap': False,
            'source_file': f"synthetic_{424000000000 + day_num}_SLEEP_DATA.fit",
        })

    df = pd.DataFrame(rows)
    path = os.path.join(save_dir, 'sleep_stages.csv')
    df.to_csv(path, index=False)
    return path, len(df)


# =========================================================================
# CHANNEL 4: STEPS BINNED (15-minute bins, BST)
# =========================================================================

def _write_steps_csv(trajectory, t_grid, channel_outputs, params, save_dir):
    """Generate sdk_steps_binned.csv with 15-minute bins.

    Real Garmin step data has:
    - date_bst: calendar date in BST
    - bin_start_bst / bin_end_bst: BST timestamps (15-min wide)
    - steps: count (0 bins are ABSENT from the real file)
    - activity_type: walking / running
    - n_events: number of raw events aggregated

    Only bins with steps > 0 appear in the real file.

    CRITICAL (Bug 1 fix): consumes channel_outputs['steps'] from gen_steps()
    instead of regenerating with an independent RNG.  This guarantees the
    step counts in the CSV are EXACTLY the same ones that gen_hr() used
    to compute the exercise indicator → HR is consistent with steps.
    """
    if 'steps' not in channel_outputs:
        raise KeyError("_write_steps_csv requires channel_outputs['steps']; "
                       "gen_steps must run before this writer.")

    t0 = _make_data_start(params)
    rng_n_events = np.random.default_rng(params.get('csv_seed', 2026) + 3)

    steps_data = channel_outputs['steps']
    t_idx_arr = steps_data['t_idx']      # bin midpoint grid indices
    counts_arr = steps_data['counts']     # already-generated step counts
    is_running_arr = steps_data['is_running']

    bins_per_grid = 3                     # 15-min bin = 3 × 5-min grid steps
    rows = []

    for b in range(len(counts_arr)):
        count = counts_arr[b]
        if count <= 0:
            continue                      # zero-count bins absent from real CSV

        # Recover bin start from the midpoint index that gen_steps recorded.
        # gen_steps used: mid = gs + bins_per_grid // 2  (so gs = mid - 1)
        mid = int(t_idx_arr[b])
        gs = max(0, mid - bins_per_grid // 2)
        t_start_h = float(t_grid[gs])

        bin_start_utc = _hours_to_utc(t_start_h, t0)
        bin_end_utc = bin_start_utc + pd.Timedelta(minutes=15)
        bin_start_bst = _utc_to_bst(bin_start_utc)
        bin_end_bst = _utc_to_bst(bin_end_utc)
        date_bst = bin_start_bst.strftime('%Y-%m-%d')

        activity_type = 'running' if bool(is_running_arr[b]) else 'walking'

        rows.append({
            'date_bst': date_bst,
            'bin_start_bst': bin_start_bst,
            'bin_end_bst': bin_end_bst,
            'steps': float(count),
            'activity_type': activity_type,
            'n_events': int(rng_n_events.integers(1, 5)),
        })

    df = pd.DataFrame(rows)

    path = os.path.join(save_dir, 'sdk_steps_binned.csv')
    df.to_csv(path, index=False)
    return path, len(df)


# =========================================================================
# GROUND-TRUTH JSON EXPORT
# =========================================================================

# Parameter grouping for the sleep-wake 6-state model.
# Each group corresponds to a distinct modelling concern — the estimator
# can use this grouping for block-wise analysis and diagnostic plots.
_PARAM_GROUPS = {
    'fast_dynamics': [
        'tau_W', 'tau_Z', 'g_w', 'g_z',
        'a_w', 'z_w', 'a_z', 'w_z', 'c_amp',
        'k_in', 'k_out', 'k_glymph',
    ],
    'circadian': ['phi'],
    'process_noise': ['T_W', 'T_Z', 'T_A', 'T_Vh', 'T_Vn'],
    'slow_dynamics': ['tau_Vh', 'tau_Vn', 'beta_h', 'beta_n'],
    'obs_heart_rate': [
        'HR_base', 'alpha_HR', 'beta_exercise', 'sigma_HR',
    ],
    'obs_stress': ['s_base', 's_W', 's_n', 'sigma_S'],
    'obs_sleep_stages': ['alpha_sleep', 'c_d', 'c_r', 'c_l'],
    'obs_steps': ['p_move', 'r_step', 'alpha_run', 'sigma_step'],
    'exogenous_model': ['gamma_0', 'gamma_steps'],
}

_PARAM_DESCRIPTIONS = {
    'tau_W': 'Wakefulness time constant (hours)',
    'tau_Z': 'Sleep depth time constant (hours)',
    'g_w': 'Wakefulness sigmoid gain',
    'g_z': 'Sleep sigmoid gain',
    'a_w': 'Adenosine → wake inhibition',
    'z_w': 'Sleep → wake inhibition',
    'a_z': 'Adenosine → sleep activation',
    'w_z': 'Wake → sleep inhibition',
    'c_amp': 'Circadian amplitude',
    'k_in': 'Adenosine production rate (during wake)',
    'k_out': 'Adenosine baseline clearance rate',
    'k_glymph': 'Adenosine glymphatic clearance rate (during sleep)',
    'phi': 'Circadian phase offset (radians)',
    'T_W': 'W process noise temperature',
    'T_Z': 'Z process noise temperature',
    'T_A': 'A process noise temperature',
    'T_Vh': 'Vh process noise temperature (daily units)',
    'T_Vn': 'Vn process noise temperature (daily units)',
    'tau_Vh': 'Vh slow time constant (days)',
    'tau_Vn': 'Vn slow time constant (days)',
    'beta_h': 'Vitality entrainment coupling',
    'beta_n': 'Stress entrainment coupling',
    'HR_base': 'HR baseline (bpm)',
    'alpha_HR': 'HR wakefulness coefficient (bpm)',
    'beta_exercise': 'HR exercise coefficient (bpm)',
    'sigma_HR': 'HR observation noise (bpm)',
    's_base': 'Stress score baseline',
    's_W': 'Stress wakefulness coefficient',
    's_n': 'Stress Vn coefficient',
    'sigma_S': 'Stress observation noise',
    'alpha_sleep': 'Sleep stage logistic steepness',
    'c_d': 'Deep sleep threshold on Z',
    'c_r': 'REM sleep threshold on Z',
    'c_l': 'Light sleep threshold on Z',
    'p_move': 'Movement probability coefficient',
    'r_step': 'Step count per active walking bin: LogNormal MEDIAN '
              '(NOT mean; mean = r_step * exp(sigma_step^2 / 2))',
    'alpha_run': 'Running log-step multiplier (added to log r_step)',
    'sigma_step': 'Step count LogNormal sigma (log-space SD)',
    'gamma_0': 'Vh target baseline',
    'gamma_steps': 'Vh target daily-activity coefficient',
}


def _to_json_safe(obj):
    """Convert numpy scalars and arrays to JSON-serialisable types."""
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    return obj


def _write_ground_truth_json(trajectory, t_grid, channel_outputs, params,
                              init_state, exogenous, save_dir, meta=None):
    """Write ground_truth.json with all parameters and metadata.

    This JSON sits alongside the CSVs so the estimation test harness
    can load both the synthetic observations and the true values that
    generated them.  Structure:

        {
          "meta": {...simulation metadata...},
          "initial_state": {...},
          "exogenous_inputs": {...},
          "parameters": {
            "by_group": {
              "fast_dynamics": {"tau_W": 2.0, ...},
              "obs_heart_rate": {...},
              ...
            },
            "flat": {...all params, flat dict...},
            "descriptions": {...human-readable descriptions...}
          },
          "parameter_count": {...counts by group...}
        }
    """
    flat = {k: _to_json_safe(v) for k, v in params.items()}

    # Build grouped structure
    by_group = {}
    param_count = {}
    ungrouped = set(params.keys())

    for group_name, param_names in _PARAM_GROUPS.items():
        group_dict = {}
        for name in param_names:
            if name in params:
                group_dict[name] = _to_json_safe(params[name])
                ungrouped.discard(name)
        if group_dict:
            by_group[group_name] = group_dict
            param_count[group_name] = len(group_dict)

    if ungrouped:
        by_group['unclassified'] = {k: _to_json_safe(params[k]) for k in sorted(ungrouped)}
        param_count['unclassified'] = len(ungrouped)

    descriptions = {k: v for k, v in _PARAM_DESCRIPTIONS.items() if k in params}

    payload = {
        'meta': _to_json_safe(meta or {}),
        'initial_state': _to_json_safe(init_state),
        'exogenous_inputs': _to_json_safe(exogenous or {}),
        'parameters': {
            'by_group': by_group,
            'flat': flat,
            'descriptions': descriptions,
        },
        'parameter_count': {
            'total': len(params),
            **param_count,
        },
    }

    path = os.path.join(save_dir, 'ground_truth.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    return path


# =========================================================================
# PUBLIC API
# =========================================================================

def write_garmin_csvs(trajectory, t_grid, channel_outputs, params, save_dir,
                      init_state=None, exogenous=None, meta=None):
    """Write all 4 Garmin-format CSV files + ground_truth.json for
    end-to-end pipeline testing.

    The CSV files are designed to be read by loader.py → ObservationData
    identically to real Garmin data.  The JSON carries the ground-truth
    parameters, initial state, and exogenous inputs so the estimation
    test harness can compare posterior estimates against them.

    Args:
        trajectory: (T, 6) latent state trajectory
        t_grid: (T,) time grid in hours
        channel_outputs: dict from generate_all_channels
        params: parameter dict
        save_dir: output directory
        init_state: initial state dict (optional, included in JSON if given)
        exogenous: exogenous inputs dict (optional, included in JSON if given)
        meta: additional simulation metadata dict (optional)

    Returns:
        dict of {filename: n_rows} (JSON listed as n_rows=None)
    """
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    path, n = _write_hr_csv(trajectory, t_grid, channel_outputs, params, save_dir)
    results['sdk_intraday_hr.csv'] = n
    print(f"    {os.path.basename(path)}: {n} rows (1-min resolution)")

    path, n = _write_stress_csv(trajectory, t_grid, channel_outputs, params, save_dir)
    results['sdk_intraday_stress.csv'] = n
    print(f"    {os.path.basename(path)}: {n} rows (3-min resolution)")

    path, n = _write_sleep_csv(trajectory, t_grid, channel_outputs, params, save_dir)
    results['sleep_stages.csv'] = n
    print(f"    {os.path.basename(path)}: {n} rows (variable epochs)")

    path, n = _write_steps_csv(trajectory, t_grid, channel_outputs, params, save_dir)
    results['sdk_steps_binned.csv'] = n
    print(f"    {os.path.basename(path)}: {n} rows (15-min bins, non-zero only)")

    # Ground-truth JSON
    default_meta = {
        'model_name': 'sleep_wake_6state',
        'model_version': '1.0',
        'n_grid_steps': int(len(t_grid)),
        'dt_hours': float(t_grid[1] - t_grid[0]) if len(t_grid) > 1 else 0.0,
        't_total_hours': float(t_grid[-1]),
        't_total_days': float(t_grid[-1] / 24.0),
        'n_states': int(trajectory.shape[1]),
        'state_names': ['W', 'Z', 'A', 'C', 'Vh', 'Vn'],
    }
    if meta:
        default_meta.update(meta)

    path = _write_ground_truth_json(
        trajectory, t_grid, channel_outputs, params,
        init_state or {}, exogenous or {}, save_dir, default_meta)
    results['ground_truth.json'] = None
    print(f"    {os.path.basename(path)}: {len(params)} parameters in "
          f"{len(_PARAM_GROUPS)} groups")

    return results

"""
models/sleep_wake_model.py — 6-State Sleep-Wake SDE Model Definition
======================================================================
Date:    15 April 2026
Version: 1.0

All model-specific content for the sleep-wake circadian system.
The generic framework imports only the SLEEP_WAKE_MODEL object.

Equations: n1_pilot_bayesian_estimation.md §2 (Eq 1–8)
"""

import math
import numpy as np

from simulator.sde_model import SDEModel, StateSpec, ChannelSpec, DIFFUSION_DIAGONAL_CONSTANT

# Import model-specific plotter (separated per project convention)
from models.sleep_wake.sim_plots import plot_sleep_wake
from models.sleep_wake.csv_writer import write_garmin_csvs


# =========================================================================
# ELEMENTARY HELPERS
# =========================================================================

def sigmoid(x):
    """Numerically stable logistic sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def circadian(t, params):
    """C(t) = sin(2πt/24 + φ).  Exact analytical expression (numpy path)."""
    return math.sin(2.0 * math.pi * t / 24.0 + params['phi'])


def circadian_jax(t, params):
    """JAX version of circadian(): called inside jit-compiled scans.

    Works with traced JAX values; params is expected to be the JAX
    parameter dict (e.g. built by _params_to_jax).
    """
    import jax.numpy as jnp
    return jnp.sin(2.0 * jnp.pi * t / 24.0 + params['phi'])


def entrainment(W, Z, A, Vh, Vn, p):
    """Entrainment quality E ∈ [0,1] (Eq 2.8 of OT spec)."""
    kappa = 2.0
    mu_W = Vh + Vn - p['a_w'] * A - p['z_w'] * Z
    mu_Z = p['a_z'] * A - p['w_z'] * W - Vn
    c2 = p['c_amp'] ** 2
    return float(sigmoid(kappa * (c2 - mu_W**2)) *
                 sigmoid(kappa * (c2 - mu_Z**2)))


def _vh_target_at_time(t, daily_targets):
    """Look up V_h target for the day containing time t."""
    day = int(t / 24.0)
    day = max(0, min(day, len(daily_targets) - 1))
    return daily_targets[day]


# =========================================================================
# DRIFT
# =========================================================================

def drift(t, y, params, aux):
    """6-state drift f(t, y, θ).

    y = [W, Z, A, C, Vh, Vn]
    params = dict of parameter values
    aux = (daily_targets, vn_target)

    Returns: dy/dt as array(6,)
    """
    daily_targets, vn_target = aux
    W, Z, A, C, Vh, Vn = y[0], y[1], y[2], y[3], y[4], y[5]

    W_e = np.clip(W, 0.0, 1.0)
    Z_e = np.clip(Z, 0.0, 1.0)
    A_e = max(A, 0.0)
    Vn_e = max(Vn, 0.0)

    p = params
    C_exact = circadian(t, p)

    X_W = p['c_amp'] * C_exact + Vh + Vn_e - p['a_w'] * A_e - p['z_w'] * Z_e
    X_Z = -p['c_amp'] * C_exact + p['a_z'] * A_e - p['w_z'] * W_e - Vn_e

    dW = (float(sigmoid(p['g_w'] * X_W)) - W) / p['tau_W']
    dZ = (float(sigmoid(p['g_z'] * X_Z)) - Z) / p['tau_Z']
    dA = p['k_in'] * W_e - (p['k_out'] + p['k_glymph'] * Z_e) * A_e
    dC = (2.0 * math.pi / 24.0) * math.cos(2.0 * math.pi * t / 24.0 + p['phi'])

    E = entrainment(W_e, Z_e, A_e, Vh, Vn_e, p)

    Vh_target = _vh_target_at_time(t, daily_targets)
    dVh = (Vh_target - Vh) / (p['tau_Vh'] * 24.0) + (p['beta_h'] / 24.0) * E
    dVn = (vn_target - Vn) / (p['tau_Vn'] * 24.0) - (p['beta_n'] / 24.0) * E

    return np.array([dW, dZ, dA, dC, dVh, dVn])


# =========================================================================
# DIFFUSION
# =========================================================================

def diffusion_diagonal(params):
    """Diagonal noise σ_i. C=0 (deterministic). Daily→hourly for slow states."""
    p = params
    return np.array([
        math.sqrt(2.0 * p['T_W']),
        math.sqrt(2.0 * p['T_Z']),
        math.sqrt(2.0 * p['T_A']),
        0.0,
        math.sqrt(2.0 * p['T_Vh'] / 24.0),
        math.sqrt(2.0 * p['T_Vn'] / 24.0),
    ])


# =========================================================================
# AUXILIARY DATA
# =========================================================================

def make_aux(params, init_state, t_grid, exogenous):
    """Build (daily_targets, vn_target) for the drift function."""
    daily_bins = exogenous.get('daily_active_bins',
                               np.array([8.0] * 9, dtype=np.float64))
    daily_targets = params['gamma_0'] + params['gamma_steps'] * daily_bins
    vn_target = init_state['Vn_0']
    return (daily_targets, vn_target)


# =========================================================================
# INITIAL STATE
# =========================================================================

def make_y0(init_dict, params):
    """Build [W, Z, A, C(0), Vh, Vn] from init parameters."""
    C0 = math.sin(params['phi'])
    return np.array([
        init_dict['W_0'], init_dict['Z_0'], init_dict['A_0'],
        C0, init_dict['Vh_0'], init_dict['Vn_0'],
    ])


# =========================================================================
# OBSERVATION CHANNEL GENERATORS
# =========================================================================

def gen_steps(trajectory, t_grid, params, aux, prior_channels, seed):
    """Channel 4: Zero-inflated log-normal step counts (§5.6).

    For each 15-min bin, with probability  p_active = W_mean * p_move
    we sample a step count from:
        steps ~ LogNormal(mu = log(r_step) + alpha_run * I[running],
                          sigma = sigma_step)

    PARAMETER SEMANTICS (Bug 2 fix — was previously documented as mean):
      r_step    is the MEDIAN steps per active walking bin (LogNormal scale)
      sigma_step is the LogNormal sigma (log-space standard deviation)
      The MEAN of the distribution is r_step * exp(sigma_step**2 / 2),
      which differs from r_step by ~38% at sigma_step=0.8.

    The prior must be elicited on the median (r_step), not the mean.
    """
    rng = np.random.default_rng(seed)
    W = trajectory[:, 0]
    p = params
    steps_per_bin = 3
    n_bins = len(t_grid) // steps_per_bin

    t_idx = np.zeros(n_bins, dtype=np.int32)
    counts = np.zeros(n_bins, dtype=np.float64)
    is_running = np.zeros(n_bins, dtype=bool)

    for b in range(n_bins):
        gs = b * steps_per_bin
        ge = min(gs + steps_per_bin, len(t_grid))
        mid = min(gs + steps_per_bin // 2, len(t_grid) - 1)
        t_idx[b] = mid
        W_mean = np.mean(W[gs:ge])
        p_active = np.clip(W_mean * p['p_move'], 0.0, 1.0)
        if rng.random() >= p_active:
            continue
        t_hour_of_day = t_grid[mid] % 24.0
        p_run = 0.15 if 7.0 <= t_hour_of_day <= 10.0 else 0.03
        running = rng.random() < p_run
        is_running[b] = running
        log_mu = math.log(p['r_step']) + p['alpha_run'] * float(running)
        counts[b] = max(1.0, round(math.exp(rng.normal(log_mu, p['sigma_step']))))

    return {'t_idx': t_idx, 'counts': counts, 'is_running': is_running}


def gen_hr(trajectory, t_grid, params, aux, prior_channels, seed):
    """Channel 2: Gaussian HR observations at NATIVE 1-minute resolution.

    Real Garmin HR is recorded ~1 reading/minute continuously, with
    occasional multi-minute gaps when the watch loses wrist contact.

    Output is at 1-min resolution (NOT the 5-min simulation grid).
    The estimation pipeline's loader/thinning will aggregate these
    readings to the 5-min grid — the same pipeline as for real data.

    Returns dict with:
        t_minutes:  (n_obs,) integer minutes since t0
        t_hours:    (n_obs,) float hours since t0
        hr_bpm:     (n_obs,) integer BPM values
        exercise:   (n_obs,) {0, 1} indicator
    """
    rng = np.random.default_rng(seed)
    p = params

    # Build 1-minute time axis
    total_minutes = int(round(t_grid[-1] * 60.0)) + 1
    t_minutes_axis = np.arange(total_minutes, dtype=np.int64)
    t_hours_1min = t_minutes_axis / 60.0

    # Interpolate W to 1-min resolution
    W_1min = np.interp(t_hours_1min, t_grid, trajectory[:, 0])

    # Watch-off gap model: Poisson(~4/day), each 5-30 min long
    is_observed = np.ones(total_minutes, dtype=bool)
    n_days = max(1.0, t_grid[-1] / 24.0)
    n_gaps = int(rng.poisson(4.0 * n_days))
    if n_gaps > 0:
        gap_starts = rng.integers(0, total_minutes, size=n_gaps)
        for gs in gap_starts:
            gap_len = int(rng.integers(5, 31))
            ge = min(int(gs) + gap_len, total_minutes)
            is_observed[int(gs):ge] = False

    # Exercise indicator from prior steps channel.  Active 15-min bins
    # (steps > 200) flag their entire interval as exercise.
    exercise_1min = np.zeros(total_minutes, dtype=np.float64)
    if 'steps' in prior_channels:
        step_counts = prior_channels['steps']['counts']
        step_t_idx = prior_channels['steps']['t_idx']  # bin midpoint grid indices
        bin_minutes = 15  # each step bin spans 15 minutes
        for b, count in enumerate(step_counts):
            if count <= 200:
                continue
            # gen_steps recorded mid = b*3 + 1 (for bins_per_grid=3, dt=5min)
            # The bin starts at mid-1 grid steps and is 3 grid steps wide.
            mid_grid = int(step_t_idx[b])
            grid_dt_min = (t_grid[1] - t_grid[0]) * 60.0  # 5
            start_min = int(round((mid_grid - 1) * grid_dt_min))
            end_min = start_min + bin_minutes
            start_min = max(0, start_min)
            end_min = min(total_minutes, end_min)
            exercise_1min[start_min:end_min] = 1.0

    # Generate HR values for every minute (vectorised)
    noise = rng.normal(0.0, p['sigma_HR'], total_minutes)
    hr_values = (p['HR_base']
                 + p['alpha_HR'] * W_1min
                 + p['beta_exercise'] * exercise_1min
                 + noise)
    hr_values = np.clip(np.round(hr_values), 30, 220).astype(np.int32)

    # Keep only observed minutes
    keep = np.where(is_observed)[0]
    return {
        't_minutes': t_minutes_axis[keep].astype(np.int64),
        't_hours': t_hours_1min[keep].astype(np.float64),
        'hr_bpm': hr_values[keep],
        'exercise': exercise_1min[keep].astype(np.float64),
    }


def gen_stress(trajectory, t_grid, params, aux, prior_channels, seed):
    """Channel 3: Gaussian stress observations at NATIVE 3-minute resolution.

    Real Garmin stress is recorded every 3 minutes when the watch is worn.
    Output is at 3-min resolution; loader/thinning aligns to 5-min grid.

    Returns dict with:
        t_minutes:    (n_obs,) integer minutes since t0 (multiples of 3)
        t_hours:      (n_obs,) float hours since t0
        stress_score: (n_obs,) integer scores [0, 100]
    """
    rng = np.random.default_rng(seed)
    p = params

    # Build 3-minute time axis
    total_3min = int(round(t_grid[-1] * 60.0 / 3.0)) + 1
    t_minutes_axis = np.arange(total_3min, dtype=np.int64) * 3
    t_hours_3min = t_minutes_axis / 60.0

    # Interpolate W and Vn to 3-min resolution
    W_3min = np.interp(t_hours_3min, t_grid, trajectory[:, 0])
    Vn_3min = np.interp(t_hours_3min, t_grid, trajectory[:, 5])

    # Observation probability ~60% of slots
    is_observed = rng.random(total_3min) < 0.60

    # Generate scores for every slot (vectorised)
    noise = rng.normal(0.0, p['sigma_S'], total_3min)
    scores = (p['s_base']
              + p['s_W'] * W_3min
              + p['s_n'] * Vn_3min
              + noise)
    scores = np.clip(np.round(scores), 0, 100).astype(np.int32)

    keep = np.where(is_observed)[0]
    return {
        't_minutes': t_minutes_axis[keep].astype(np.int64),
        't_hours': t_hours_3min[keep].astype(np.float64),
        'stress_score': scores[keep],
    }


def gen_sleep(trajectory, t_grid, params, aux, prior_channels, seed):
    """Channel 1: Ordered logistic sleep stages, native epoch structure.

    Real Garmin sleep data has variable-length contiguous epochs
    (typically 5-30 min per stage).  We sample at 1-min resolution
    using ordered logistic conditional on Z(t), with stage persistence
    so epochs have realistic durations.  The output is the run-length
    encoded epoch list.

    Returns dict with:
        epoch_start_min: (n_epochs,) integer minutes since t0 (epoch start)
        epoch_end_min:   (n_epochs,) integer minutes since t0 (epoch end)
        labels:          (n_epochs,) int {0, 1, 2, 3} stage labels
        t_hours:         (n_epochs,) float hours of epoch midpoint
                         (preserved for backward compatibility / plotting)
    """
    rng = np.random.default_rng(seed)
    p = params
    alpha = p['alpha_sleep']
    cd, cr, cl = p['c_d'], p['c_r'], p['c_l']

    # 1-min resolution
    total_min = int(round(t_grid[-1] * 60.0)) + 1
    t_hours_1min = np.arange(total_min, dtype=np.float64) / 60.0

    Z_1min = np.interp(t_hours_1min, t_grid, trajectory[:, 1])
    W_1min = np.interp(t_hours_1min, t_grid, trajectory[:, 0])

    in_sleep = (W_1min < 0.5) | (Z_1min > 0.3)

    def _sample_stage(z_val):
        cum_d = float(sigmoid(alpha * (z_val - cd)))
        cum_r = float(sigmoid(alpha * (z_val - cr)))
        cum_l = float(sigmoid(alpha * (z_val - cl)))
        probs = np.array([1.0 - cum_l, cum_l - cum_r, cum_r - cum_d, cum_d])
        probs = np.maximum(probs, 1e-10)
        probs /= probs.sum()
        return int(rng.choice(4, p=probs))

    # Sample stages with persistence (realistic epoch lengths)
    labels_1min = np.full(total_min, -1, dtype=int)
    current_stage = -1
    hold_remaining = 0
    hold_map = {0: (2, 10), 1: (3, 15), 2: (5, 20), 3: (5, 20)}

    for i in range(total_min):
        if not in_sleep[i]:
            current_stage = -1
            hold_remaining = 0
            continue
        if hold_remaining > 0:
            labels_1min[i] = current_stage
            hold_remaining -= 1
        else:
            new_stage = _sample_stage(float(Z_1min[i]))
            current_stage = new_stage
            labels_1min[i] = new_stage
            lo, hi = hold_map.get(new_stage, (3, 10))
            hold_remaining = int(rng.integers(lo, hi + 1)) - 1

    # Run-length encode into epochs
    epoch_starts_min = []
    epoch_ends_min = []
    epoch_labels = []
    cur_label = labels_1min[0]
    cur_start = 0

    for i in range(1, total_min):
        if labels_1min[i] != cur_label:
            if cur_label >= 0:
                epoch_starts_min.append(cur_start)
                epoch_ends_min.append(i)
                epoch_labels.append(cur_label)
            cur_label = int(labels_1min[i])
            cur_start = i
    if cur_label >= 0:
        epoch_starts_min.append(cur_start)
        epoch_ends_min.append(total_min)
        epoch_labels.append(cur_label)

    starts_arr = np.array(epoch_starts_min, dtype=np.int64)
    ends_arr = np.array(epoch_ends_min, dtype=np.int64)
    labels_arr = np.array(epoch_labels, dtype=np.int32)
    midpoints_h = (starts_arr + ends_arr) / 2.0 / 60.0

    return {
        'epoch_start_min': starts_arr,
        'epoch_end_min': ends_arr,
        'labels': labels_arr,
        't_hours': midpoints_h.astype(np.float64),
    }


# =========================================================================
# PHYSICS VERIFICATION
# =========================================================================

def _estimate_period_autocorrelation(signal, dt):
    """Estimate dominant period via autocorrelation (robust to noise).

    The zero-crossing method fails on noisy signals because noise
    causes spurious crossings during transitions.  Autocorrelation
    finds the dominant periodicity regardless of noise amplitude.

    Returns the period in the same units as dt, or nan if no clear peak.
    """
    x = signal - np.mean(signal)
    n = len(x)
    # Normalised autocorrelation via FFT (O(n log n))
    fft_x = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(fft_x * np.conj(fft_x))[:n]
    acf = acf / acf[0] if acf[0] > 0 else acf

    # Find first peak after the initial decay (lag > min_lag)
    min_lag = int(12.0 / dt)  # at least 12 hours
    max_lag = min(int(48.0 / dt), n - 1)  # at most 48 hours

    if max_lag <= min_lag:
        return float('nan')

    acf_window = acf[min_lag:max_lag + 1]
    if len(acf_window) < 3:
        return float('nan')

    peak_idx = np.argmax(acf_window)
    period = (min_lag + peak_idx) * dt
    return float(period)


def verify_physics(trajectory, t_grid, params):
    """Model-specific physics checks for the sleep-wake system.

    Uses autocorrelation for period estimation (robust to SDE noise).
    Uses mean-contrast for adenosine homeostasis (robust to noise
    realisation — unlike per-step correlation which picks up σ_A √dt
    noise directly and varies strongly with random seed).
    """
    skip = min(576, len(t_grid) // 3)
    W = trajectory[skip:, 0]
    A = trajectory[skip:, 2]

    # Period via autocorrelation (robust to noise-induced jitter)
    period = _estimate_period_autocorrelation(W, float(t_grid[1] - t_grid[0]))

    # Adenosine homeostasis: physical claim is "A is higher during wake
    # than during sleep".  This is a mean contrast over many points,
    # which is robust to noise realisation.
    wake_mask = W > 0.5
    sleep_mask = W < 0.5
    mean_A_wake = float(np.mean(A[wake_mask])) if wake_mask.any() else float('nan')
    mean_A_sleep = float(np.mean(A[sleep_mask])) if sleep_mask.any() else float('nan')

    # Effect size: (mean_wake - mean_sleep) / pooled_std
    # Physical expectation: k_in/(k_out+k_glymph) ~ 0.8/2.3 ~ 0.35 → clear effect
    pooled_std = float(np.std(A))
    effect_size = ((mean_A_wake - mean_A_sleep) / pooled_std
                   if pooled_std > 0 else 0.0)

    # Boundedness
    bounded = (np.all(trajectory[:, :3] >= -1e-6) and
               np.all(trajectory[:, :3] <= 1.0 + 1e-6) and
               np.all(trajectory[:, 4:] >= -1e-6) and
               np.all(trajectory[:, 4:] <= 1.0 + 1e-6))

    return {
        'period_h': period,
        'period_ok': 20.0 < period < 28.0 if np.isfinite(period) else False,
        'mean_A_wake': mean_A_wake,
        'mean_A_sleep': mean_A_sleep,
        'A_wake_vs_sleep_effect_size': effect_size,
        'adenosine_builds_during_wake': effect_size > 0.3,
        'all_bounded': bool(bounded),
        'no_nans': bool(np.all(np.isfinite(trajectory))),
    }


# =========================================================================
# PARAMETER SETS (§3)
# =========================================================================

PARAM_SET_A = {
    'tau_W': 2.0, 'tau_Z': 2.0, 'g_w': 8.0, 'g_z': 10.0,
    'a_w': 3.0, 'z_w': 5.0, 'a_z': 4.0, 'w_z': 6.0, 'c_amp': 4.0,
    'k_in': 0.8, 'k_out': 0.3, 'k_glymph': 2.0,
    'phi': -math.pi / 3.0,
    'T_W': 0.01, 'T_Z': 0.01, 'T_A': 0.005, 'T_Vh': 0.05, 'T_Vn': 0.05,
    'tau_Vh': 7.0, 'tau_Vn': 7.0, 'beta_h': 0.1, 'beta_n': 0.1,
    'c_d': 0.85, 'c_r': 0.60, 'c_l': 0.25, 'alpha_sleep': 6.0,
    'HR_base': 50.0, 'alpha_HR': 25.0, 'beta_exercise': 40.0, 'sigma_HR': 8.0,
    's_base': 3.0, 's_W': 40.0, 's_n': 3.0, 'sigma_S': 10.0,
    'p_move': 0.35, 'r_step': 1200.0, 'alpha_run': 0.5, 'sigma_step': 0.8,
    'gamma_0': 0.5, 'gamma_steps': 0.1,
}

INIT_STATE_A = {'W_0': 0.5, 'Z_0': 0.3, 'A_0': 0.5, 'Vh_0': 0.05, 'Vn_0': 0.5}

DAILY_ACTIVE_BINS = np.array([6, 9, 13, 9, 14, 9, 9, 10, 8], dtype=np.float64)


def _make_param_set_b(base=None, seed=123):
    base = base or PARAM_SET_A
    rng = np.random.default_rng(seed)
    frozen = {'tau_Vh', 'tau_Vn', 'beta_h', 'beta_n'}
    out = {}
    for k, v in base.items():
        if k in frozen:
            out[k] = v
        elif k == 'phi':
            out[k] = v + rng.uniform(-0.3, 0.3)
        else:
            out[k] = v * rng.uniform(0.8, 1.2)
    return out


# =========================================================================
# JAX DRIFT + AUX (for Diffrax solver — optional, requires JAX)
# =========================================================================

def _try_import_jax():
    """Lazy import JAX — only needed when Diffrax solver is used."""
    try:
        import jax
        import jax.numpy as jnp
        return jax, jnp
    except ImportError:
        return None, None


def drift_jax(t, y, args):
    """6-state drift in JAX.  Diffrax signature: (t, y, args) -> dy/dt.
    args = (params_dict_jax, daily_targets_jax, vn_target_jax).

    Equations IDENTICAL to drift() above — cross-validated by the framework.
    """
    _, jnp = _try_import_jax()
    jax_mod, _ = _try_import_jax()

    p, daily_targets, vn_target = args
    W, Z, A, C, Vh, Vn = y[0], y[1], y[2], y[3], y[4], y[5]
    W_e = jnp.clip(W, 0.0, 1.0)
    Z_e = jnp.clip(Z, 0.0, 1.0)
    A_e = jnp.maximum(A, 0.0)
    Vn_e = jnp.maximum(Vn, 0.0)

    C_ex = jnp.sin(2.0 * jnp.pi * t / 24.0 + p['phi'])
    X_W = p['c_amp'] * C_ex + Vh + Vn_e - p['a_w'] * A_e - p['z_w'] * Z_e
    X_Z = -p['c_amp'] * C_ex + p['a_z'] * A_e - p['w_z'] * W_e - Vn_e

    dW = (jax_mod.nn.sigmoid(p['g_w'] * X_W) - W) / p['tau_W']
    dZ = (jax_mod.nn.sigmoid(p['g_z'] * X_Z) - Z) / p['tau_Z']
    dA = p['k_in'] * W_e - (p['k_out'] + p['k_glymph'] * Z_e) * A_e
    dC = (2.0 * jnp.pi / 24.0) * jnp.cos(2.0 * jnp.pi * t / 24.0 + p['phi'])

    kappa = 2.0
    mu_W = Vh + Vn_e - p['a_w'] * A_e - p['z_w'] * Z_e
    mu_Z = p['a_z'] * A_e - p['w_z'] * W_e - Vn_e
    c2 = p['c_amp'] ** 2
    E = (jax_mod.nn.sigmoid(kappa * (c2 - mu_W**2)) *
         jax_mod.nn.sigmoid(kappa * (c2 - mu_Z**2)))

    day_idx = jnp.clip((t / 24.0).astype(jnp.int32), 0, daily_targets.shape[0] - 1)
    Vh_tgt = daily_targets[day_idx]

    dVh = (Vh_tgt - Vh) / (p['tau_Vh'] * 24.0) + (p['beta_h'] / 24.0) * E
    dVn = (vn_target - Vn) / (p['tau_Vn'] * 24.0) - (p['beta_n'] / 24.0) * E

    return jnp.array([dW, dZ, dA, dC, dVh, dVn])


def make_aux_jax(params, init_state, t_grid, exogenous):
    """Build JAX-compatible args tuple for drift_jax."""
    _, jnp = _try_import_jax()
    if jnp is None:
        raise ImportError("JAX is required for Diffrax solver")

    p_jax = {k: jnp.float64(v) for k, v in params.items()}
    daily_bins = exogenous.get('daily_active_bins',
                               np.array([8.0] * 9, dtype=np.float64))
    daily_targets = p_jax['gamma_0'] + p_jax['gamma_steps'] * jnp.array(
        daily_bins, dtype=jnp.float64)
    vn_target = jnp.float64(init_state['Vn_0'])
    return (p_jax, daily_targets, vn_target)


# =========================================================================
# THE MODEL OBJECT
# =========================================================================

SLEEP_WAKE_MODEL = SDEModel(
    name="sleep_wake_6state",
    version="1.0",

    states=(
        StateSpec("W", 0.0, 1.0),
        StateSpec("Z", 0.0, 1.0),
        StateSpec("A", 0.0, 1.0),
        StateSpec("C", -1.0, 1.0, is_deterministic=True,
                  analytical_fn=circadian, analytical_fn_jax=circadian_jax),
        StateSpec("Vh", 0.0, 1.0),
        StateSpec("Vn", 0.0, 1.0),
    ),

    drift_fn=drift,
    drift_fn_jax=drift_jax,
    diffusion_type=DIFFUSION_DIAGONAL_CONSTANT,
    diffusion_fn=diffusion_diagonal,
    make_aux_fn=make_aux,
    make_aux_fn_jax=make_aux_jax,
    make_y0_fn=make_y0,

    channels=(
        ChannelSpec("steps",  depends_on=(),          generate_fn=gen_steps),
        ChannelSpec("hr",     depends_on=("steps",),  generate_fn=gen_hr),
        ChannelSpec("stress", depends_on=(),          generate_fn=gen_stress),
        ChannelSpec("sleep",  depends_on=(),          generate_fn=gen_sleep),
    ),

    plot_fn=plot_sleep_wake,
    csv_writer_fn=write_garmin_csvs,
    verify_physics_fn=verify_physics,

    param_sets={'A': PARAM_SET_A, 'B': _make_param_set_b()},
    init_states={'A': INIT_STATE_A},
    exogenous_inputs={'A': {'daily_active_bins': DAILY_ACTIVE_BINS}},
)

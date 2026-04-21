"""
Sleep-wake 6-state SDE estimation model.
==========================================
Date:    15 April 2026
Version: 5.0 (model-agnostic framework)

ALL model-specific code for the differentiable particle filter lives
here: drift equations, guided proposal, observation likelihoods,
phase-conditioned init, noise structure, priors.

The generic framework never imports from this file directly — it
receives the SLEEP_WAKE_ESTIMATION object and calls its functions.
"""

import math
import numpy as np
from collections import OrderedDict

import jax
import jax.numpy as jnp

from estimation_model import EstimationModel
from _likelihood_constants import HALF_LOG_2PI

# ─── Frozen parameters ──────────────────────────────────────────

FROZEN = {
    'tau_Vh': 7.0, 'tau_Vn': 7.0,
    'beta_h': 0.3, 'beta_n': 0.2,
}
KAPPA = 2.0  # entrainment steepness

# ─── Parameter index (built from prior config) ──────────────────

PARAM_PRIOR_CONFIG = OrderedDict([
    ('tau_W',    ('lognormal', (0.693, 0.27))),
    ('tau_Z',    ('lognormal', (0.693, 0.27))),
    ('g_w',      ('lognormal', (2.079, 0.31))),
    ('g_z',      ('lognormal', (2.303, 0.31))),
    ('a_w',      ('lognormal', (1.099, 0.30))),
    ('z_w',      ('lognormal', (1.609, 0.30))),
    ('a_z',      ('lognormal', (1.386, 0.30))),
    ('w_z',      ('lognormal', (1.791, 0.30))),
    ('c_amp',    ('lognormal', (1.386, 0.20))),
    ('k_in',     ('lognormal', (-0.223, 0.20))),
    ('k_out',    ('lognormal', (-1.204, 0.20))),
    ('k_glymph', ('lognormal', (0.693, 0.30))),
    ('phi',      ('vonmises', (-1.047, 4.0))),
    ('T_W',      ('lognormal', (-4.605, 0.30))),
    ('T_Z',      ('lognormal', (-4.605, 0.30))),
    ('T_A',      ('lognormal', (-5.298, 0.30))),
    ('T_Vh',     ('lognormal', (-2.996, 0.20))),
    ('T_Vn',     ('lognormal', (-2.996, 0.20))),
    ('c_d',        ('normal', (0.85, 0.05))),
    ('c_r',        ('normal', (0.55, 0.05))),
    ('c_l',        ('normal', (0.20, 0.05))),
    ('alpha_sleep', ('lognormal', (2.303, 0.25))),
    ('HR_base',    ('normal', (50.0, 5.0))),
    ('alpha_HR',   ('lognormal', (3.219, 0.15))),
    ('beta_exercise', ('lognormal', (3.689, 0.25))),
    ('sigma_HR',   ('lognormal', (2.079, 0.15))),
    ('s_base',     ('normal', (10.0, 3.0))),
    ('s_W',        ('lognormal', (3.689, 0.15))),
    ('s_n',        ('lognormal', (1.099, 0.30))),
    ('sigma_S',    ('lognormal', (2.708, 0.15))),
    ('p_move',     ('beta', (5.0, 12.0))),
    ('r_step',     ('lognormal', (7.090, 0.15))),
    ('alpha_run',  ('lognormal', (-0.693, 0.20))),
    ('sigma_step', ('lognormal', (-0.357, 0.10))),
    ('gamma_0',    ('normal', (0.0, 0.5))),
    ('gamma_steps', ('lognormal', (-2.303, 0.30))),
])

INIT_STATE_PRIOR_CONFIG = OrderedDict([
    ('W_0',  ('beta', (3.0, 3.0))),
    ('Z_0',  ('beta', (3.0, 3.0))),
    ('A_0',  ('lognormal', (0.0, 0.30))),
    ('Vh_0', ('normal', (1.5, 0.7))),
    ('Vn_0', ('lognormal', (1.386, 0.30))),
])

PI = PARAM_PRIOR_CONFIG
_PK = list(PI.keys())
_PI = {k: i for i, k in enumerate(_PK)}


# ─── JAX functions for inside jax.lax.scan ──────────────────────

def _entrainment(W, Z, A, Vh, Vn, a_w, z_w, a_z, w_z, c_amp):
    """Entrainment quality E(t) in [0,1]."""
    mu_W = Vh + Vn - a_w * A - z_w * Z
    mu_Z = a_z * A - w_z * W - Vn
    c2 = c_amp ** 2
    return (jax.nn.sigmoid(KAPPA * (c2 - mu_W ** 2)) *
            jax.nn.sigmoid(KAPPA * (c2 - mu_Z ** 2)))


def _vh_target(t, daily_targets):
    """Look up daily Vh target at time t."""
    day = jnp.clip((t / 24.0).astype(jnp.int32), 0, daily_targets.shape[0] - 1)
    return daily_targets[day]


def _imex_components(t, y, params, daily_targets, Vn_target):
    """IMEX forcing/decay for the 6-state sleep-wake SDE."""
    W  = jnp.clip(y[0], 0.0, 1.0)
    Z  = jnp.clip(y[1], 0.0, 1.0)
    A  = jax.nn.softplus(20.0 * y[2]) / 20.0
    C  = y[3]
    Vh = y[4]
    Vn = jnp.maximum(y[5], 0.0)

    tau_W = params[_PI['tau_W']]; tau_Z = params[_PI['tau_Z']]
    g_w = params[_PI['g_w']]; g_z = params[_PI['g_z']]
    a_w = params[_PI['a_w']]; z_w = params[_PI['z_w']]
    a_z = params[_PI['a_z']]; w_z = params[_PI['w_z']]
    c_amp = params[_PI['c_amp']]
    k_in = params[_PI['k_in']]; k_out = params[_PI['k_out']]
    k_glymph = params[_PI['k_glymph']]

    X_W = c_amp * C + Vh + Vn - a_w * A - z_w * Z
    X_Z = -c_amp * C + a_z * A - w_z * W - Vn

    fW = jax.nn.sigmoid(g_w * X_W) / tau_W; dW = 1.0 / tau_W
    fZ = jax.nn.sigmoid(g_z * X_Z) / tau_Z; dZ = 1.0 / tau_Z
    fA = k_in * W; dA = k_out + k_glymph * Z

    vh_t = _vh_target(t, daily_targets)
    E = _entrainment(W, Z, A, Vh, Vn, a_w, z_w, a_z, w_z, c_amp)
    fVh = vh_t / (FROZEN['tau_Vh'] * 24) + (FROZEN['beta_h'] / 24) * E
    dVh = 1.0 / (FROZEN['tau_Vh'] * 24)
    fVn = jnp.maximum(
        Vn_target / (FROZEN['tau_Vn'] * 24) - (FROZEN['beta_n'] / 24) * E, 0.0)
    dVn = 1.0 / (FROZEN['tau_Vn'] * 24)

    return (jnp.array([fW, fZ, fA, 0.0, fVh, fVn]),
            jnp.array([dW, dZ, dA, 0.0, dVh, dVn]))


def diffusion_fn(params):
    """Diagonal noise coefficients. C=0 (deterministic)."""
    return jnp.array([
        jnp.sqrt(2.0 * params[_PI['T_W']]),
        jnp.sqrt(2.0 * params[_PI['T_Z']]),
        jnp.sqrt(2.0 * params[_PI['T_A']]),
        0.0,
        jnp.sqrt(2.0 * params[_PI['T_Vh']] / 24.0),
        jnp.sqrt(2.0 * params[_PI['T_Vn']] / 24.0),
    ])


def propagate_fn(y, t, dt, params, grid_obs, k,
                 sigma_diag, noise, rng_key):
    """IMEX propagation with Pitt & Shephard guided proposal on W.

    Args:
        y: Current particle [W,Z,A,C,Vh,Vn], shape (6,).
        t: Current time in hours.
        dt: Time step in hours.
        params: Parameter vector, shape (36,).
        grid_obs: Dict of observation arrays.
        k: Current step index within the shard.
        sigma_diag: Noise diagonal, shape (6,).
        noise: Standard normal, shape (6,).
        rng_key: JAX PRNG key (unused here).

    Returns:
        Tuple (x_new, predictive_log_weight).
    """
    daily_targets = params[_PI['gamma_0']] + params[_PI['gamma_steps']] * grid_obs['daily_active_bins']
    Vn_target = y[5]  # current Vn as target proxy
    phi = params[_PI['phi']]

    forcing, decay = _imex_components(t, y, params, daily_targets, Vn_target)
    denom = 1.0 + dt * decay
    mu_prior = (y + dt * forcing) / denom
    var_prior = (sigma_diag ** 2 * dt) / (denom ** 2)

    # ── Guided proposal for W ──
    HR_base = params[_PI['HR_base']]; alpha_HR = params[_PI['alpha_HR']]
    beta_ex = params[_PI['beta_exercise']]; sigma_HR = params[_PI['sigma_HR']]
    s_base = params[_PI['s_base']]; s_W = params[_PI['s_W']]
    s_n = params[_PI['s_n']]; sigma_S = params[_PI['sigma_S']]

    W_prec = 1.0 / jnp.maximum(var_prior[0], 1e-12)
    W_info = W_prec * mu_prior[0]

    hr_obs = grid_obs['hr_present'][k]
    W_info += hr_obs * alpha_HR * (
        grid_obs['hr_value'][k] - HR_base - beta_ex * grid_obs['hr_exercise'][k]
    ) / sigma_HR ** 2
    W_prec += hr_obs * alpha_HR ** 2 / sigma_HR ** 2

    s_obs = grid_obs['stress_present'][k]
    Vn_pr = mu_prior[5]
    W_info += s_obs * s_W * (
        grid_obs['stress_value'][k] - s_base - s_n * Vn_pr
    ) / sigma_S ** 2
    W_prec += s_obs * s_W ** 2 / sigma_S ** 2

    W_var = 1.0 / W_prec
    W_mu = W_var * W_info

    # Sample
    W_new = jnp.clip(W_mu + jnp.sqrt(W_var) * noise[0], 0.0, 1.0)
    Z_new = jnp.clip(mu_prior[1] + jnp.sqrt(jnp.maximum(var_prior[1], 1e-12)) * noise[1], 0.0, 1.0)
    A_new = jnp.clip(mu_prior[2] + jnp.sqrt(jnp.maximum(var_prior[2], 1e-12)) * noise[2], 0.0, 1.0)
    C_new = jnp.sin(2.0 * jnp.pi * (t + dt) / 24.0 + phi)
    Vh_new = jnp.clip(mu_prior[4] + jnp.sqrt(jnp.maximum(var_prior[4], 1e-12)) * noise[3], 0.0, 1.0)
    Vn_new = jnp.clip(mu_prior[5] + jnp.sqrt(jnp.maximum(var_prior[5], 1e-12)) * noise[4], 0.0, 1.0)

    x_new = jnp.array([W_new, Z_new, A_new, C_new, Vh_new, Vn_new])

    # ── Predictive log-weight (HR + stress) ──
    # Includes -HALF_LOG_2PI per Gaussian observation (v6.4 convention).
    lw = 0.0
    pred_var_hr = sigma_HR ** 2 + alpha_HR ** 2 * var_prior[0]
    pred_mu_hr = HR_base + alpha_HR * mu_prior[0] + beta_ex * grid_obs['hr_exercise'][k]
    lw += hr_obs * (-0.5 * (grid_obs['hr_value'][k] - pred_mu_hr) ** 2 / pred_var_hr
                    - 0.5 * jnp.log(pred_var_hr) - HALF_LOG_2PI)

    pred_var_s = sigma_S ** 2 + s_W ** 2 * var_prior[0]
    pred_mu_s = s_base + s_W * mu_prior[0] + s_n * Vn_pr
    lw += s_obs * (-0.5 * (grid_obs['stress_value'][k] - pred_mu_s) ** 2 / pred_var_s
                   - 0.5 * jnp.log(pred_var_s) - HALF_LOG_2PI)

    return x_new, lw


def obs_log_weight_fn(x_new, grid_obs, k, params):
    """Observation log-weight from sleep + step channels.

    HR and stress are handled by the guided proposal (predictive weight).
    """
    Z_new = x_new[1]; W_new = x_new[0]

    # Sleep (ordered logistic)
    alpha_sl = params[_PI['alpha_sleep']]
    c_d = params[_PI['c_d']]; c_r = params[_PI['c_r']]; c_l = params[_PI['c_l']]
    cum_d = jax.nn.sigmoid(alpha_sl * (Z_new - c_d))
    cum_r = jax.nn.sigmoid(alpha_sl * (Z_new - c_r))
    cum_l = jax.nn.sigmoid(alpha_sl * (Z_new - c_l))
    probs = jnp.maximum(jnp.array([1.0 - cum_l, cum_l - cum_r, cum_r - cum_d, cum_d]), 1e-8)
    lw = grid_obs['sleep_present'][k] * jnp.log(probs[grid_obs['sleep_label'][k]])

    # Steps (zero-inflated log-normal)
    # Includes -HALF_LOG_2PI for the LogNormal density (v6.4 convention).
    p_move = params[_PI['p_move']]; r_step = params[_PI['r_step']]
    alpha_run = params[_PI['alpha_run']]; sigma_step = params[_PI['sigma_step']]
    pnz = jnp.clip(W_new * p_move, 1e-8, 1 - 1e-8)
    count = grid_obs['step_count'][k]
    is_zero = (count < 0.5)
    log_mu = jnp.log(r_step + 1e-8) + alpha_run * grid_obs['step_is_running'][k]
    log_s = jnp.log(jnp.maximum(count, 1.0))
    ll_z = jnp.log(1.0 - pnz)
    ll_nz = (jnp.log(pnz) - 0.5 * ((log_s - log_mu) / sigma_step) ** 2
             - jnp.log(sigma_step) - log_s - HALF_LOG_2PI)
    lw += grid_obs['step_present'][k] * jnp.where(is_zero, ll_z, ll_nz)

    return lw


def shard_init_fn(time_offset, params, exogenous, global_init):
    """Phase-conditioned init for one shard."""
    dt_h = 5.0 / 60.0
    t_start = time_offset * dt_h
    phi = params[_PI['phi']]
    C_s = jnp.sin(2.0 * jnp.pi * t_start / 24.0 + phi)

    W_i = jax.nn.sigmoid(2.0 * C_s)
    Z_i = jax.nn.sigmoid(-2.0 * C_s)
    k_in = params[_PI['k_in']]; k_out = params[_PI['k_out']]
    k_gl = params[_PI['k_glymph']]
    A_i = jnp.clip(k_in * W_i / jnp.maximum(k_out + k_gl * Z_i, 1e-6), 0, 1)

    daily_bins = exogenous.get('daily_active_bins', jnp.zeros(9))
    day = jnp.clip((t_start / 24.0).astype(jnp.int32), 0, daily_bins.shape[0] - 1)
    Vh_i = params[_PI['gamma_0']] + params[_PI['gamma_steps']] * daily_bins[day]
    Vn_i = global_init[4]

    return jnp.array([W_i, Z_i, A_i, jnp.sin(2 * jnp.pi * t_start / 24 + phi), Vh_i, Vn_i])


def align_obs_fn(obs_data, t_steps, dt_hours):
    """Convert ObservationData to dict of grid-aligned JAX arrays."""
    T = t_steps
    hr_val = np.zeros(T, dtype=np.float32)
    hr_ex = np.zeros(T, dtype=np.float32)
    hr_pres = np.zeros(T, dtype=np.float32)
    hr_val[obs_data.hr_t_idx] = obs_data.hr_bpm
    hr_ex[obs_data.hr_t_idx] = obs_data.hr_exercise
    hr_pres[obs_data.hr_t_idx] = 1.0

    stress_val = np.zeros(T, dtype=np.float32)
    stress_pres = np.zeros(T, dtype=np.float32)
    stress_val[obs_data.stress_t_idx] = obs_data.stress_score
    stress_pres[obs_data.stress_t_idx] = 1.0

    sleep_lab = np.zeros(T, dtype=np.int32)
    sleep_pres = np.zeros(T, dtype=np.float32)
    sleep_idx = np.round(obs_data.sleep_t_hours / dt_hours).astype(np.int32)
    sleep_idx = np.clip(sleep_idx, 0, T - 1)
    for i, idx in enumerate(sleep_idx):
        sleep_lab[idx] = obs_data.sleep_labels[i]
        sleep_pres[idx] = 1.0

    step_cnt = np.zeros(T, dtype=np.float32)
    step_run = np.zeros(T, dtype=np.float32)
    step_pres = np.zeros(T, dtype=np.float32)
    step_cnt[obs_data.step_t_idx] = obs_data.step_counts
    step_run[obs_data.step_t_idx] = obs_data.step_is_running.astype(np.float32)
    step_pres[obs_data.step_t_idx] = 1.0

    has_any = np.maximum(np.maximum(hr_pres, stress_pres),
                          np.maximum(sleep_pres, step_pres))

    return {
        'hr_value': jnp.array(hr_val), 'hr_exercise': jnp.array(hr_ex),
        'hr_present': jnp.array(hr_pres),
        'stress_value': jnp.array(stress_val),
        'stress_present': jnp.array(stress_pres),
        'sleep_label': jnp.array(sleep_lab),
        'sleep_present': jnp.array(sleep_pres),
        'step_count': jnp.array(step_cnt),
        'step_is_running': jnp.array(step_run),
        'step_present': jnp.array(step_pres),
        'has_any_obs': jnp.array(has_any),
        'daily_active_bins': jnp.array(obs_data.daily_active_bins),
    }


def forward_sde_fn(init_state, params, exogenous, dt, n_steps,
                    rng_key=None):
    """Stochastic IMEX forward integration for MAP trajectory.

    Includes process noise from the diffusion — essential for
    realistic dynamics (sleep-wake transitions, adenosine fluctuations).

    Args:
        init_state: Init estimates, shape (5,) or (6,).
        params: Parameter vector, shape (36,).
        exogenous: Dict with 'daily_active_bins'.
        dt: Time step in hours.
        n_steps: Number of steps.
        rng_key: JAX PRNG key. If None, uses PRNGKey(0).

    Returns:
        Trajectory array, shape (n_steps, 6).
    """
    phi = params[_PI['phi']]
    sigma_d = diffusion_fn(params)
    sqrt_dt = jnp.sqrt(dt)
    daily_bins = exogenous.get('daily_active_bins', jnp.zeros(9))
    daily_targets = params[_PI['gamma_0']] + params[_PI['gamma_steps']] * daily_bins
    Vn_tgt = float(init_state[5]) if init_state.shape[0] > 5 else 0.5

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    def step(carry, i):
        y, key = carry
        key, noise_key = jax.random.split(key)
        t = i * dt

        forcing, decay = _imex_components(t, y, params, daily_targets, Vn_tgt)
        noise = jax.random.normal(noise_key, (6,))

        y_next = (y + dt * forcing + sigma_d * sqrt_dt * noise) / (1.0 + dt * decay)

        t_next = t + dt
        y_next = y_next.at[3].set(jnp.sin(2.0 * jnp.pi * t_next / 24.0 + phi))
        for j in [0, 1, 2, 4, 5]:
            y_next = y_next.at[j].set(jnp.clip(y_next[j], 0.0, 1.0))
        return (y_next, key), y_next

    # Build full 6D init (init_state may be 5D or 6D)
    if init_state.shape[0] == 5:
        y0 = jnp.array([init_state[0], init_state[1], init_state[2],
                         jnp.sin(phi), init_state[3], init_state[4]])
    else:
        y0 = init_state

    (_, _), traj = jax.lax.scan(step, (y0, rng_key), jnp.arange(n_steps))
    return traj


# ── Init estimates (clipped v8 medians) ──

_INIT_EST = {
    'tau_W': 3.34, 'tau_Z': 2.53, 'g_w': 12.7, 'g_z': 8.65,
    'a_w': 3.60, 'z_w': 10.0, 'a_z': 3.64, 'w_z': 6.74,
    'c_amp': 3.10, 'k_in': 0.631, 'k_out': 0.240, 'k_glymph': 3.33,
    'phi': -1.276, 'T_W': 0.020, 'T_Z': 0.017, 'T_A': 0.0029,
    'T_Vh': 0.0506, 'T_Vn': 0.0752,
    'c_d': 0.827, 'c_r': 0.534, 'c_l': 0.180, 'alpha_sleep': 6.02,
    'HR_base': 49.5, 'alpha_HR': 36.0, 'beta_exercise': 22.0, 'sigma_HR': 6.81,
    's_base': 3.01, 's_W': 55.0, 's_n': 2.87, 'sigma_S': 10.5,
    'p_move': 0.55, 'r_step': 900.0, 'alpha_run': 0.687, 'sigma_step': 0.85,
    'gamma_0': 0.627, 'gamma_steps': 0.134,
    'W_0': 0.655, 'Z_0': 0.244, 'A_0': 0.998, 'Vh_0': 0.066, 'Vn_0': 2.77,
}


def _prior_mean(ptype, pargs):
    if ptype == 'lognormal': return math.exp(pargs[0] + pargs[1]**2 / 2)
    elif ptype == 'normal': return pargs[0]
    elif ptype == 'vonmises': return pargs[0]
    elif ptype == 'beta': return pargs[0] / (pargs[0] + pargs[1])
    return 0.0


def get_init_theta():
    """Initial theta from clipped v8 estimates."""
    all_names = list(PARAM_PRIOR_CONFIG.keys()) + list(INIT_STATE_PRIOR_CONFIG.keys())
    all_config = OrderedDict()
    all_config.update(PARAM_PRIOR_CONFIG)
    all_config.update(INIT_STATE_PRIOR_CONFIG)
    means = {n: _prior_mean(pt, pa) for n, (pt, pa) in all_config.items()}
    return np.array([_INIT_EST.get(n, means[n]) for n in all_names], dtype=np.float32)


# ── Import model-specific I/O ──
from models.sleep_wake.data import load_data as _load_data
from models.sleep_wake.plots import plot_trajectory as _plot_traj
from models.sleep_wake.plots import plot_residuals as _plot_resid


# ─── NEW: Direct-scan functions (no particles, no noise) ────────

def imex_step_fn(y, t, dt, params, grid_obs):
    """Deterministic IMEX step for the direct-scan log-density.

    Same equations as propagate_fn but: no noise, no guided proposal,
    no particle dimension.  Just the deterministic dynamics.
    """
    daily_targets = (params[_PI['gamma_0']]
                     + params[_PI['gamma_steps']] * grid_obs['daily_active_bins'])
    Vn_target = y[5]
    phi = params[_PI['phi']]

    forcing, decay = _imex_components(t, y, params, daily_targets, Vn_target)
    y_next = (y + dt * forcing) / (1.0 + dt * decay)

    # Overwrite circadian (deterministic)
    t_next = t + dt
    y_next = y_next.at[3].set(jnp.sin(2.0 * jnp.pi * t_next / 24.0 + phi))

    # Constraint enforcement
    y_next = y_next.at[0].set(jnp.clip(y_next[0], 0.0, 1.0))
    y_next = y_next.at[1].set(jnp.clip(y_next[1], 0.0, 1.0))
    y_next = y_next.at[2].set(jnp.clip(y_next[2], 0.0, 1.0))
    y_next = y_next.at[4].set(jnp.clip(y_next[4], 0.0, 1.0))
    y_next = y_next.at[5].set(jnp.clip(y_next[5], 0.0, 1.0))

    return y_next


def obs_log_prob_fn(y, grid_obs, k, params):
    """Total observation log-probability at step k given state y.

    Combines all 4 channels: HR (Gaussian), stress (Gaussian),
    sleep (ordered logistic), steps (zero-inflated log-normal).
    """
    W = y[0]; Z = y[1]; Vn = y[5]
    ll = jnp.float32(0.0)

    # HR  (Gaussian)
    HR_base = params[_PI['HR_base']]; alpha_HR = params[_PI['alpha_HR']]
    beta_ex = params[_PI['beta_exercise']]; sigma_HR = params[_PI['sigma_HR']]
    hr_pred = HR_base + alpha_HR * W + beta_ex * grid_obs['hr_exercise'][k]
    hr_resid = grid_obs['hr_value'][k] - hr_pred
    ll += grid_obs['hr_present'][k] * (
        -0.5 * (hr_resid / sigma_HR) ** 2 - jnp.log(sigma_HR) - HALF_LOG_2PI)

    # Stress  (Gaussian)
    s_base = params[_PI['s_base']]; s_W = params[_PI['s_W']]
    s_n = params[_PI['s_n']]; sigma_S = params[_PI['sigma_S']]
    s_pred = s_base + s_W * W + s_n * Vn
    s_resid = grid_obs['stress_value'][k] - s_pred
    ll += grid_obs['stress_present'][k] * (
        -0.5 * (s_resid / sigma_S) ** 2 - jnp.log(sigma_S) - HALF_LOG_2PI)

    # Sleep (ordered logistic)
    alpha_sl = params[_PI['alpha_sleep']]
    c_d = params[_PI['c_d']]; c_r = params[_PI['c_r']]; c_l = params[_PI['c_l']]
    cum_d = jax.nn.sigmoid(alpha_sl * (Z - c_d))
    cum_r = jax.nn.sigmoid(alpha_sl * (Z - c_r))
    cum_l = jax.nn.sigmoid(alpha_sl * (Z - c_l))
    probs = jnp.maximum(jnp.array([
        1.0 - cum_l, cum_l - cum_r, cum_r - cum_d, cum_d]), 1e-8)
    ll += grid_obs['sleep_present'][k] * jnp.log(probs[grid_obs['sleep_label'][k]])

    # Steps (zero-inflated log-normal)
    p_move = params[_PI['p_move']]; r_step = params[_PI['r_step']]
    alpha_run = params[_PI['alpha_run']]; sigma_step = params[_PI['sigma_step']]
    pnz = jnp.clip(W * p_move, 1e-8, 1 - 1e-8)
    count = grid_obs['step_count'][k]
    is_zero = (count < 0.5)
    log_mu = jnp.log(r_step + 1e-8) + alpha_run * grid_obs['step_is_running'][k]
    log_s = jnp.log(jnp.maximum(count, 1.0))
    ll_z = jnp.log(1.0 - pnz)
    ll_nz = (jnp.log(pnz) - 0.5 * ((log_s - log_mu) / sigma_step) ** 2
             - jnp.log(sigma_step) - log_s - HALF_LOG_2PI)
    ll += grid_obs['step_present'][k] * jnp.where(is_zero, ll_z, ll_nz)

    return ll


def obs_sample_fn(y, exog, k, params, rng_key):
    """Sample one observation from each channel at step k given state y.

    SAMPLING COUNTERPART of ``obs_log_prob_fn``.  Each channel below
    draws from EXACTLY the same distribution that ``obs_log_prob_fn``
    evaluates a step later — so a synthetic data set produced via this
    function and then scored by ``obs_log_prob_fn`` is statistically
    self-consistent (the log-density is maximised at the true params,
    modulo the SDE noise contribution).

    Args:
        y: 6-vector latent state at step k.
        exog: Dict containing the exogenous arrays
            ``hr_exercise`` (T,) and ``step_is_running`` (T,).
            These are inputs to the observation model — the simulator
            generates them BEFORE calling this function.
        k: Integer step index.
        params: Estimated parameter vector (36,).
        rng_key: JAX PRNG key.

    Returns:
        Dict of scalar JAX arrays:
            ``hr_value``    — float
            ``stress_value``— float
            ``sleep_label`` — int32 in {0, 1, 2, 3}
            ``step_count``  — float (≥ 0; exact 0 when zero-inflation fires)
    """
    W = y[0]; Z = y[1]; Vn = y[5]
    k_hr, k_st, k_sl, k_sp_act, k_sp_log = jax.random.split(rng_key, 5)

    # ── HR ~ N(HR_base + alpha_HR * W + beta_ex * exercise, sigma_HR^2) ──
    HR_base = params[_PI['HR_base']]; alpha_HR = params[_PI['alpha_HR']]
    beta_ex = params[_PI['beta_exercise']]; sigma_HR = params[_PI['sigma_HR']]
    hr_mean = HR_base + alpha_HR * W + beta_ex * exog['hr_exercise'][k]
    hr_value = hr_mean + sigma_HR * jax.random.normal(k_hr, dtype=W.dtype)

    # ── Stress ~ N(s_base + s_W * W + s_n * Vn, sigma_S^2) ──
    s_base = params[_PI['s_base']]; s_W = params[_PI['s_W']]
    s_n = params[_PI['s_n']]; sigma_S = params[_PI['sigma_S']]
    s_mean = s_base + s_W * W + s_n * Vn
    stress_value = s_mean + sigma_S * jax.random.normal(k_st, dtype=W.dtype)

    # ── Sleep: Categorical over the same ordered-logistic probs ──
    alpha_sl = params[_PI['alpha_sleep']]
    c_d = params[_PI['c_d']]; c_r = params[_PI['c_r']]; c_l = params[_PI['c_l']]
    cum_d = jax.nn.sigmoid(alpha_sl * (Z - c_d))
    cum_r = jax.nn.sigmoid(alpha_sl * (Z - c_r))
    cum_l = jax.nn.sigmoid(alpha_sl * (Z - c_l))
    probs = jnp.maximum(jnp.array([
        1.0 - cum_l, cum_l - cum_r, cum_r - cum_d, cum_d]), 1e-8)
    sleep_label = jax.random.categorical(k_sl, jnp.log(probs)).astype(jnp.int32)

    # ── Steps: zero-inflated log-normal ──
    # Active w.p. p_active = clip(W * p_move, 0, 1).
    # If active:  count = max(1, exp( log(r_step) + alpha_run*is_running
    #                                 + sigma_step * N(0,1) ))
    # The max(., 1) matches the floor used inside obs_log_prob_fn's log_s.
    p_move = params[_PI['p_move']]; r_step = params[_PI['r_step']]
    alpha_run = params[_PI['alpha_run']]; sigma_step = params[_PI['sigma_step']]
    pnz = jnp.clip(W * p_move, 0.0, 1.0)
    is_active = jax.random.uniform(k_sp_act, dtype=W.dtype) < pnz
    log_mu = jnp.log(r_step + 1e-8) + alpha_run * exog['step_is_running'][k]
    log_count = log_mu + sigma_step * jax.random.normal(k_sp_log, dtype=W.dtype)
    count = jnp.maximum(jnp.exp(log_count), 1.0)
    step_count = jnp.where(is_active, count, jnp.zeros((), dtype=W.dtype))

    return {
        'hr_value':     hr_value,
        'stress_value': stress_value,
        'sleep_label':  sleep_label,
        'step_count':   step_count,
    }


def make_init_state_fn(init_estimates, params):
    """Build full 6D initial state from estimated init values + params.

    Args:
        init_estimates: [W_0, Z_0, A_0, Vh_0, Vn_0], shape (5,).
        params: Parameter vector, shape (36,).

    Returns:
        y0: [W, Z, A, C, Vh, Vn], shape (6,).
    """
    phi = params[_PI['phi']]
    return jnp.array([
        init_estimates[0],  # W_0
        init_estimates[1],  # Z_0
        init_estimates[2],  # A_0
        jnp.sin(phi),       # C_0 = sin(phi) at t=0
        init_estimates[3],  # Vh_0
        init_estimates[4],  # Vn_0
    ])


def gaussian_obs_fn(y, grid_obs, k, params):
    """Per-step Gaussian-channel info for the EKF (HR + stress).

    Sleep-wake has 4 observation channels but only 2 are Gaussian:

      HR     ~ N(HR_base + alpha_HR*W + beta_ex*hr_exercise[k], sigma_HR^2)
      stress ~ N(s_base + s_W*W + s_n*Vn,                       sigma_S^2)

    The other two channels (sleep ordered logistic, steps zero-inflated
    log-normal) are NOT Gaussian — they are passed back to the caller
    via obs_log_prob_fn in 'ekf_hybrid' mode (handled outside this fn).

    Args:
        y: 6-vector latent state.
        grid_obs: Dict of grid arrays.
        k: Step index.
        params: Estimated parameter vector.

    Returns:
        Dict with mean, value, cov_diag, present each shape (2,):
            entry 0 is HR, entry 1 is stress.
    """
    W = y[0]; Vn = y[5]
    HR_base = params[_PI['HR_base']]; alpha_HR = params[_PI['alpha_HR']]
    beta_ex = params[_PI['beta_exercise']]; sigma_HR = params[_PI['sigma_HR']]
    s_base = params[_PI['s_base']]; s_W = params[_PI['s_W']]
    s_n = params[_PI['s_n']]; sigma_S = params[_PI['sigma_S']]

    hr_mean = HR_base + alpha_HR * W + beta_ex * grid_obs['hr_exercise'][k]
    s_mean = s_base + s_W * W + s_n * Vn

    return {
        'mean':     jnp.array([hr_mean, s_mean]),
        'value':    jnp.array([grid_obs['hr_value'][k],
                               grid_obs['stress_value'][k]]),
        'cov_diag': jnp.array([sigma_HR ** 2, sigma_S ** 2]),
        'present':  jnp.array([grid_obs['hr_present'][k],
                               grid_obs['stress_present'][k]]),
    }


# ─── Marginal-SGR kernel densities (NEW v6.4) ───────────────────

def _imex_prior_moments(y, t, dt, params, grid_obs):
    """Compute the IMEX prior mean and variance for all 6 states.

    This is the shared computation used by both the dynamic kernel
    and the guided proposal.  Factored out so the marginal-SGR scan
    can call it once per ancestor.

    Args:
        y: Previous state [W,Z,A,C,Vh,Vn], shape (6,).
        t: Current time in hours.
        dt: Time step in hours.
        params: Parameter vector.
        grid_obs: Dict of grid arrays.

    Returns:
        (mu_prior, var_prior) — both shape (6,).
        mu_prior: IMEX step mean  = (y + dt*forcing) / (1 + dt*decay).
        var_prior: IMEX step var  = σ² · dt / (1 + dt*decay)².
    """
    daily_targets = (params[_PI['gamma_0']]
                     + params[_PI['gamma_steps']]
                     * grid_obs['daily_active_bins'])
    Vn_target = y[5]

    forcing, decay = _imex_components(t, y, params, daily_targets, Vn_target)
    denom = 1.0 + dt * decay
    mu_prior = (y + dt * forcing) / denom

    sigma_diag = diffusion_fn(params)
    var_prior = (sigma_diag ** 2 * dt) / (denom ** 2)

    return mu_prior, var_prior


def _guided_proposal_W_moments(mu_prior, var_prior, params, grid_obs, k):
    """Compute the Kalman-updated (W_mu, W_var) for the guided proposal.

    Conditions the prior on W on Gaussian observations (HR, stress)
    exactly as ``propagate_fn`` does (lines 179-202 of this file).

    Args:
        mu_prior: shape (6,) — IMEX prior mean.
        var_prior: shape (6,) — IMEX prior variance.
        params: parameter vector.
        grid_obs: dict of grid arrays.
        k: step index.

    Returns:
        (W_mu, W_var) — scalars, the posterior mean and variance of W.
    """
    HR_base = params[_PI['HR_base']]; alpha_HR = params[_PI['alpha_HR']]
    beta_ex = params[_PI['beta_exercise']]; sigma_HR = params[_PI['sigma_HR']]
    s_base = params[_PI['s_base']]; s_W = params[_PI['s_W']]
    s_n = params[_PI['s_n']]; sigma_S = params[_PI['sigma_S']]

    W_prec = 1.0 / jnp.maximum(var_prior[0], 1e-12)
    W_info = W_prec * mu_prior[0]

    hr_obs = grid_obs['hr_present'][k]
    W_info += hr_obs * alpha_HR * (
        grid_obs['hr_value'][k] - HR_base - beta_ex * grid_obs['hr_exercise'][k]
    ) / sigma_HR ** 2
    W_prec += hr_obs * alpha_HR ** 2 / sigma_HR ** 2

    s_obs = grid_obs['stress_present'][k]
    Vn_pr = mu_prior[5]
    W_info += s_obs * s_W * (
        grid_obs['stress_value'][k] - s_base - s_n * Vn_pr
    ) / sigma_S ** 2
    W_prec += s_obs * s_W ** 2 / sigma_S ** 2

    W_var = 1.0 / W_prec
    W_mu = W_var * W_info
    return W_mu, W_var


def dynamic_kernel_log_density_fn(x_new, x_prev, t, dt, params,
                                    grid_obs, sigma_diag):
    """Evaluate log p(x_new | x_prev) under the bootstrap dynamic kernel.

    The kernel is N(mu_prior, diag(var_prior)) over the STOCHASTIC
    dimensions {W, Z, A, Vh, Vn} (indices 0,1,2,4,5).  C (index 3)
    is deterministic and contributes nothing to the density.

    Used by the marginal-SGR scan for the O(K²) kernel matrix.

    Args:
        x_new:  shape (6,) — new particle position.
        x_prev: shape (6,) — ancestor particle position.
        t:      scalar — time in hours.
        dt:     scalar — step size in hours.
        params: parameter vector.
        grid_obs: dict of grid arrays.
        sigma_diag: shape (6,) — diffusion diagonal.

    Returns:
        scalar log-density.
    """
    mu_prior, var_prior = _imex_prior_moments(x_prev, t, dt, params, grid_obs)
    sto = jnp.array([0, 1, 2, 4, 5], dtype=jnp.int32)
    diff = x_new[sto] - mu_prior[sto]
    v = jnp.maximum(var_prior[sto], 1e-30)
    return -0.5 * jnp.sum(diff ** 2 / v + jnp.log(2.0 * jnp.pi * v))


def proposal_log_density_fn(x_new, x_prev, t, dt, params,
                              grid_obs, k, sigma_diag):
    """Evaluate log π(x_new | x_prev, y_k) under the guided proposal.

    Identical to ``dynamic_kernel_log_density_fn`` except that
    dimension 0 (W) uses the Kalman-updated (W_mu, W_var) from
    conditioning on the Gaussian observations (HR + stress) at step k.

    Dimensions Z, A, Vh, Vn use the IMEX prior (π = p for those).

    Used by the marginal-SGR scan for the O(K²) proposal matrix.

    Args:
        x_new:  shape (6,) — new particle position.
        x_prev: shape (6,) — ancestor particle position.
        t, dt:  time / step size.
        params: parameter vector.
        grid_obs: dict of grid arrays.
        k:      step index (for observation lookup).
        sigma_diag: shape (6,) — diffusion diagonal (unused here, for
                     interface consistency).

    Returns:
        scalar log-density.
    """
    mu_prior, var_prior = _imex_prior_moments(x_prev, t, dt, params, grid_obs)
    W_mu, W_var = _guided_proposal_W_moments(
        mu_prior, var_prior, params, grid_obs, k)

    # Build per-dimension mean and variance arrays:
    # W uses guided, everything else uses IMEX prior.
    sto = jnp.array([0, 1, 2, 4, 5], dtype=jnp.int32)
    mu = mu_prior[sto]
    v = jnp.maximum(var_prior[sto], 1e-30)
    # Overwrite dimension 0 with the guided proposal moments
    mu = mu.at[0].set(W_mu)
    v = v.at[0].set(jnp.maximum(W_var, 1e-30))

    diff = x_new[sto] - mu
    return -0.5 * jnp.sum(diff ** 2 / v + jnp.log(2.0 * jnp.pi * v))


# ─── Assemble the EstimationModel ───────────────────────────────

SLEEP_WAKE_ESTIMATION = EstimationModel(
    name="sleep_wake",
    version="6.4",
    n_states=6,
    n_stochastic=5,
    stochastic_indices=(0, 1, 2, 4, 5),
    state_bounds=((0, 1), (0, 1), (0, 1), (-1, 1), (0, 1), (0, 1)),
    param_prior_config=PARAM_PRIOR_CONFIG,
    init_state_prior_config=INIT_STATE_PRIOR_CONFIG,
    frozen_params=FROZEN,
    propagate_fn=propagate_fn,
    diffusion_fn=diffusion_fn,
    obs_log_weight_fn=obs_log_weight_fn,
    align_obs_fn=align_obs_fn,
    shard_init_fn=shard_init_fn,
    forward_sde_fn=forward_sde_fn,
    get_init_theta_fn=get_init_theta,
    load_data_fn=_load_data,
    plot_trajectory_fn=_plot_traj,
    plot_residuals_fn=_plot_resid,
    # NEW in v6.0: direct-scan functions
    imex_step_fn=imex_step_fn,
    obs_log_prob_fn=obs_log_prob_fn,
    make_init_state_fn=make_init_state_fn,
    # NEW in v6.3: simulator integration
    obs_sample_fn=obs_sample_fn,
    # NEW in v6.4: EKF support
    gaussian_obs_fn=gaussian_obs_fn,
    # NEW in v6.4: marginal-SGR kernel densities
    dynamic_kernel_log_density_fn=dynamic_kernel_log_density_fn,
    proposal_log_density_fn=proposal_log_density_fn,
    exogenous_keys=('daily_active_bins',),
)
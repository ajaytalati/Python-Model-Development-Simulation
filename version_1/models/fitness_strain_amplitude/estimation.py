"""models/fitness_strain_amplitude/estimation.py — FSA Estimation.

Date:    18 April 2026
Version: 1.0

Estimation model for the 3-state Fitness-Strain-Amplitude SDE.

MATHEMATICAL SPECIFICATION
--------------------------

Latent SDE (same as simulator):

    dB = (1 + alpha_A A)/tau_B * (T_B(t) - B) dt
         + sigma_B * sqrt(B(1-B)) dW_B

    dF = [Phi(t) - (1 + lambda_B B + lambda_A A)/tau_F * F] dt
         + sigma_F * sqrt(F) dW_F

    dA = [mu(B,F) A - eta A^3] dt
         + sigma_A * sqrt(A + eps_A) dW_A

    mu(B, F) = mu_0 + mu_B B - mu_F F - mu_FF F^2

Observation model:

    y_B(t_k) = B(t_k) + eps_k,    eps_k ~ N(0, sigma_obs^2)
    y_F(t_k) = F(t_k) + eps_k'
    y_A(t_k) = A(t_k) + eps_k''

One shared sigma_obs across all three channels.

Estimated parameters (11 total):
    10 dynamical: tau_B, alpha_A, tau_F, lambda_B, lambda_A,
                  mu_0, mu_B, mu_F, mu_FF, eta
     1 obs noise: sigma_obs
    (Process noise sigma_B, sigma_F, sigma_A are FROZEN at
     physiologically-informed values; see §7.4 of the spec.)

Initial states (3): B_0, F_0, A_0

Frozen constants (module-level, not estimated):
    EPS_A_FROZEN, EPS_B_FROZEN, SIGMA_B_FROZEN, SIGMA_F_FROZEN,
    SIGMA_A_FROZEN

Exogenous inputs (read from grid_obs): T_B, Phi.
"""

import math
import numpy as np
from collections import OrderedDict

import jax
import jax.numpy as jnp

from estimation_model import EstimationModel
from _likelihood_constants import HALF_LOG_2PI


# =========================================================================
# FROZEN CONSTANTS
# =========================================================================

EPS_A_FROZEN    = 1.0e-4
EPS_B_FROZEN    = 1.0e-4
SIGMA_B_FROZEN  = 0.01     # day^{-1/2} — process noise on B
SIGMA_F_FROZEN  = 0.005    #              process noise on F
SIGMA_A_FROZEN  = 0.02     #              process noise on A


# =========================================================================
# PRIORS
# =========================================================================

PARAM_PRIOR_CONFIG = OrderedDict([
    # Timescales — log-normal centred at reference values, moderate width
    ('tau_B',     ('lognormal', (math.log(14.0), 0.3))),
    ('alpha_A',   ('lognormal', (math.log( 1.0), 0.4))),
    ('tau_F',     ('lognormal', (math.log( 7.0), 0.3))),
    ('lambda_B',  ('lognormal', (math.log( 3.0), 0.3))),
    ('lambda_A',  ('lognormal', (math.log( 1.5), 0.3))),

    # Bifurcation parameters — careful with mu_0 (negative).
    # Parameterise |mu_0| log-normally, then negate in propagate_fn.
    ('mu_0_abs',  ('lognormal', (math.log(0.10), 0.4))),
    ('mu_B',      ('lognormal', (math.log(0.30), 0.4))),
    ('mu_F',      ('lognormal', (math.log(0.10), 0.4))),
    ('mu_FF',     ('lognormal', (math.log(0.40), 0.4))),
    ('eta',       ('lognormal', (math.log(0.20), 0.3))),

    # Observation noise
    ('sigma_obs', ('lognormal', (math.log(0.02), 0.4))),
])

INIT_STATE_PRIOR_CONFIG = OrderedDict([
    ('B_0', ('normal', (0.05, 0.05))),
    ('F_0', ('normal', (0.10, 0.10))),
    ('A_0', ('normal', (0.01, 0.01))),
])

_PK = list(PARAM_PRIOR_CONFIG.keys())
_PI = {k: i for i, k in enumerate(_PK)}


# =========================================================================
# SDE DYNAMICS (bootstrap PF Euler-Maruyama step)
# =========================================================================

def propagate_fn(y, t, dt, params, grid_obs, k,
                 sigma_diag, noise, rng_key):
    """Locally-optimal (guided) PF Euler-Maruyama step for 3-state FSA.

    Each of the 3 observed states (B, F, A) has its own guided proposal
    obtained by Gaussian fusion of the Euler prediction and the
    channel's observation:

        q*(x_{k+1} | x_k, y_{k+1}) proportional to p(x_{k+1}|x_k) p(y_{k+1}|x_{k+1})

    State-dependent process variances (Jacobi / CIR / regularised-
    Landau) are evaluated at x_k via the same formulas as the simulator.
    Observation variance is the shared sigma_obs^2.

    Weight correction returns pred_lw such that the filter framework's
    step_lw = pred_lw + obs_lw equals the sum of per-channel PREDICTIVE
    likelihoods  sum_i  log N(y_i; x_pred_i, sigma_proc_i^2 + sigma_obs^2),
    which is sample-independent — the key variance-reduction property.

    When a channel's observation is missing (obs_i_present = 0) that
    channel reverts to bootstrap (no guidance, no weight contribution).
    """
    del t, rng_key, sigma_diag

    # --- Unpack params ---
    tau_B     = params[_PI['tau_B']]
    alpha_A   = params[_PI['alpha_A']]
    tau_F     = params[_PI['tau_F']]
    lambda_B  = params[_PI['lambda_B']]
    lambda_A  = params[_PI['lambda_A']]
    mu_0      = -params[_PI['mu_0_abs']]
    mu_B      = params[_PI['mu_B']]
    mu_F      = params[_PI['mu_F']]
    mu_FF     = params[_PI['mu_FF']]
    eta       = params[_PI['eta']]
    sigma_obs = params[_PI['sigma_obs']]

    B = y[0]; F = y[1]; A = y[2]
    T_B_k = grid_obs['T_B'][k]
    Phi_k = grid_obs['Phi'][k]

    # --- Deterministic Euler predictions ---
    mu_bif  = mu_0 + mu_B * B - mu_F * F - mu_FF * F * F
    drift_B = (1.0 + alpha_A * A) / tau_B * (T_B_k - B)
    drift_F = Phi_k - (1.0 + lambda_B * B + lambda_A * A) / tau_F * F
    drift_A = mu_bif * A - eta * A * A * A

    B_pred = B + dt * drift_B
    F_pred = F + dt * drift_F
    A_pred = A + dt * drift_A

    # --- State-dependent process variances (at x_k) ---
    B_cl = jnp.clip(B, EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F_cl = jnp.maximum(F, 0.0)
    A_cl = jnp.maximum(A, 0.0)

    sp2_B = SIGMA_B_FROZEN * SIGMA_B_FROZEN * B_cl * (1.0 - B_cl) * dt
    sp2_F = SIGMA_F_FROZEN * SIGMA_F_FROZEN * F_cl * dt
    sp2_A = SIGMA_A_FROZEN * SIGMA_A_FROZEN * (A_cl + EPS_A_FROZEN) * dt
    so2   = sigma_obs * sigma_obs

    # --- Gaussian fusion per channel ---
    obs_B_pres = grid_obs['obs_B_present'][k]
    obs_F_pres = grid_obs['obs_F_present'][k]
    obs_A_pres = grid_obs['obs_A_present'][k]
    y_B        = grid_obs['obs_B_value'][k]
    y_F        = grid_obs['obs_F_value'][k]
    y_A        = grid_obs['obs_A_value'][k]

    HALF_LOG_2PI_L = 0.5 * jnp.log(2.0 * jnp.pi)

    def _fuse_channel(x_pred, y_obs, sp2, so2, present):
        """Gaussian fusion: returns (mu_eff, sigma_eff_sq, log_predictive).
        Falls back to bootstrap when present < 0.5.
        """
        sum_var       = sp2 + so2
        sigma_prop_sq = sp2 * so2 / sum_var
        mu_prop       = (x_pred * so2 + y_obs * sp2) / sum_var

        guided       = present > 0.5
        mu_eff       = jnp.where(guided, mu_prop,       x_pred)
        sigma_eff_sq = jnp.where(guided, sigma_prop_sq, sp2)

        log_predictive = jnp.where(
            guided,
            -0.5 * (y_obs - x_pred)**2 / sum_var
            - 0.5 * jnp.log(sum_var) - HALF_LOG_2PI_L,
            0.0)
        return mu_eff, sigma_eff_sq, log_predictive

    mu_B_e, sig_B_sq, lp_B = _fuse_channel(B_pred, y_B, sp2_B, so2, obs_B_pres)
    mu_F_e, sig_F_sq, lp_F = _fuse_channel(F_pred, y_F, sp2_F, so2, obs_F_pres)
    mu_A_e, sig_A_sq, lp_A = _fuse_channel(A_pred, y_A, sp2_A, so2, obs_A_pres)

    # --- Sample from proposals ---
    B_new = mu_B_e + jnp.sqrt(sig_B_sq) * noise[0]
    F_new = mu_F_e + jnp.sqrt(sig_F_sq) * noise[1]
    A_new = mu_A_e + jnp.sqrt(sig_A_sq) * noise[2]

    # --- Enforce physical bounds ---
    B_new = jnp.clip(B_new, EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F_new = jnp.maximum(F_new, 0.0)
    A_new = jnp.maximum(A_new, 0.0)

    # --- Weight correction ---
    # Framework adds obs_lw = sum_i obs_i_present * log N(y_i; x_new_i, so2).
    # We want total step_lw = lp_B + lp_F + lp_A (predictive likelihood sum).
    # So pred_lw = (lp_B + lp_F + lp_A) - (log_obs_B + log_obs_F + log_obs_A).
    def _log_obs_new(y_obs, x_new, so2, present):
        return present * (
            -0.5 * (y_obs - x_new)**2 / so2
            - 0.5 * jnp.log(so2) - HALF_LOG_2PI_L)

    log_obs_B = _log_obs_new(y_B, B_new, so2, obs_B_pres)
    log_obs_F = _log_obs_new(y_F, F_new, so2, obs_F_pres)
    log_obs_A = _log_obs_new(y_A, A_new, so2, obs_A_pres)

    pred_lw = (lp_B + lp_F + lp_A) - (log_obs_B + log_obs_F + log_obs_A)

    return jnp.array([B_new, F_new, A_new]), pred_lw


def diffusion_fn(params):
    """Frozen process-noise amplitudes (state-independent factor)."""
    del params
    return jnp.array([SIGMA_B_FROZEN, SIGMA_F_FROZEN, SIGMA_A_FROZEN])


def imex_step_fn(y, t, dt, params, grid_obs):
    """Deterministic Euler step for EKF-style prediction."""
    del t
    tau_B     = params[_PI['tau_B']]
    alpha_A   = params[_PI['alpha_A']]
    tau_F     = params[_PI['tau_F']]
    lambda_B  = params[_PI['lambda_B']]
    lambda_A  = params[_PI['lambda_A']]
    mu_0      = -params[_PI['mu_0_abs']]
    mu_B      = params[_PI['mu_B']]
    mu_F      = params[_PI['mu_F']]
    mu_FF     = params[_PI['mu_FF']]
    eta       = params[_PI['eta']]
    B = y[0]; F = y[1]; A = y[2]
    T_B_k = grid_obs.get('T_B_k', 0.0)
    Phi_k = grid_obs.get('Phi_k', 0.0)
    mu = mu_0 + mu_B * B - mu_F * F - mu_FF * F * F
    drift_B = (1.0 + alpha_A * A) / tau_B * (T_B_k - B)
    drift_F = Phi_k - (1.0 + lambda_B * B + lambda_A * A) / tau_F * F
    drift_A = mu * A - eta * A * A * A
    return jnp.array([B + dt*drift_B, F + dt*drift_F, A + dt*drift_A])


# =========================================================================
# OBSERVATION MODEL — three independent Gaussian channels, shared sigma_obs
# =========================================================================

def obs_log_prob_fn(y, grid_obs, k, params):
    """Summed Gaussian log-probability over all three obs channels."""
    sigma_obs = params[_PI['sigma_obs']]
    def _comp(y_hat, val, pres):
        resid = val - y_hat
        return pres * (-0.5 * (resid / sigma_obs) ** 2
                       - jnp.log(sigma_obs) - HALF_LOG_2PI)

    lp_B = _comp(y[0], grid_obs['obs_B_value'][k], grid_obs['obs_B_present'][k])
    lp_F = _comp(y[1], grid_obs['obs_F_value'][k], grid_obs['obs_F_present'][k])
    lp_A = _comp(y[2], grid_obs['obs_A_value'][k], grid_obs['obs_A_present'][k])
    return lp_B + lp_F + lp_A


def obs_log_weight_fn(x_new, grid_obs, k, params):
    return obs_log_prob_fn(x_new, grid_obs, k, params)


def gaussian_obs_fn(y, grid_obs, k, params):
    """Per-step Gaussian info for EKF (stack all three channels)."""
    sigma_obs = params[_PI['sigma_obs']]
    return {
        'mean':     jnp.array([y[0], y[1], y[2]]),
        'value':    jnp.array([grid_obs['obs_B_value'][k],
                                grid_obs['obs_F_value'][k],
                                grid_obs['obs_A_value'][k]]),
        'cov_diag': jnp.array([sigma_obs**2, sigma_obs**2, sigma_obs**2]),
        'present':  jnp.array([grid_obs['obs_B_present'][k],
                                grid_obs['obs_F_present'][k],
                                grid_obs['obs_A_present'][k]]),
    }


def obs_sample_fn(y, exog, k, params, rng_key):
    del exog, k
    sigma_obs = params[_PI['sigma_obs']]
    keys = jax.random.split(rng_key, 3)
    eps = jnp.array([sigma_obs * jax.random.normal(keys[i], dtype=y.dtype)
                      for i in range(3)])
    return {
        'obs_B_value': y[0] + eps[0],
        'obs_F_value': y[1] + eps[1],
        'obs_A_value': y[2] + eps[2],
    }


# =========================================================================
# GRID ALIGNMENT — obs (3 channels) + exogenous (T_B, Phi)
# =========================================================================

def _align_channel(obs_data, t_steps, value_key):
    """Align a single obs channel."""
    val  = np.zeros(t_steps, dtype=np.float32)
    pres = np.zeros(t_steps, dtype=np.float32)
    if obs_data is not None and 't_idx' in obs_data and value_key in obs_data:
        idx = np.asarray(obs_data['t_idx']).astype(int)
        val[idx]  = np.asarray(obs_data[value_key]).astype(np.float32)
        pres[idx] = 1.0
    return val, pres


def align_obs_fn(obs_data, t_steps, dt_hours):
    """Align all three obs channels + T_B and Phi exogenous channels."""
    del dt_hours

    # The simulator packs per-channel dicts inside obs_data under channel
    # names.  Access each by name; fall back to flat layout.
    def _get(name):
        return obs_data.get(name) if isinstance(obs_data, dict) else None

    B_ch = _get('obs_B');   F_ch = _get('obs_F');   A_ch = _get('obs_A')
    T_ch = _get('T_B');     P_ch = _get('Phi')

    obsB_val, obsB_pres = _align_channel(B_ch, t_steps, 'obs_value')
    obsF_val, obsF_pres = _align_channel(F_ch, t_steps, 'obs_value')
    obsA_val, obsA_pres = _align_channel(A_ch, t_steps, 'obs_value')
    T_B_val, _          = _align_channel(T_ch, t_steps, 'T_B_value')
    Phi_val, _          = _align_channel(P_ch, t_steps, 'Phi_value')

    has_any = np.maximum.reduce([obsB_pres, obsF_pres, obsA_pres])

    return {
        'obs_B_value': jnp.array(obsB_val),
        'obs_B_present': jnp.array(obsB_pres),
        'obs_F_value': jnp.array(obsF_val),
        'obs_F_present': jnp.array(obsF_pres),
        'obs_A_value': jnp.array(obsA_val),
        'obs_A_present': jnp.array(obsA_pres),
        'has_any_obs':   jnp.array(has_any),
        'T_B':           jnp.array(T_B_val),
        'Phi':           jnp.array(Phi_val),
    }


# =========================================================================
# FORWARD INTEGRATION
# =========================================================================

def forward_sde_stochastic(init_state, params, exogenous, dt, n_steps,
                            rng_key=None):
    """Stochastic Euler-Maruyama forward integrator.

    exogenous dict must contain arrays 'T_B' and 'Phi' of length n_steps.
    """
    tau_B     = params[_PI['tau_B']]
    alpha_A   = params[_PI['alpha_A']]
    tau_F     = params[_PI['tau_F']]
    lambda_B  = params[_PI['lambda_B']]
    lambda_A  = params[_PI['lambda_A']]
    mu_0      = -params[_PI['mu_0_abs']]
    mu_B      = params[_PI['mu_B']]
    mu_F      = params[_PI['mu_F']]
    mu_FF     = params[_PI['mu_FF']]
    eta       = params[_PI['eta']]
    sqrt_dt   = jnp.sqrt(dt)
    T_B_arr = jnp.asarray(exogenous['T_B'])
    Phi_arr = jnp.asarray(exogenous['Phi'])
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    def step(carry, i):
        y, key = carry
        key, nk = jax.random.split(key)
        noise = jax.random.normal(nk, (3,))
        B, F, A = y[0], y[1], y[2]
        mu = mu_0 + mu_B*B - mu_F*F - mu_FF*F*F
        dB = (1.0 + alpha_A*A)/tau_B * (T_B_arr[i] - B)
        dF = Phi_arr[i] - (1.0 + lambda_B*B + lambda_A*A)/tau_F * F
        dA = mu*A - eta*A*A*A
        B_cl = jnp.clip(B, EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
        F_cl = jnp.maximum(F, 0.0); A_cl = jnp.maximum(A, 0.0)
        B_new = B + dt*dB + SIGMA_B_FROZEN*jnp.sqrt(B_cl*(1-B_cl))*sqrt_dt*noise[0]
        F_new = F + dt*dF + SIGMA_F_FROZEN*jnp.sqrt(F_cl)*sqrt_dt*noise[1]
        A_new = A + dt*dA + SIGMA_A_FROZEN*jnp.sqrt(A_cl + EPS_A_FROZEN)*sqrt_dt*noise[2]
        B_new = jnp.clip(B_new, EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
        F_new = jnp.maximum(F_new, 0.0); A_new = jnp.maximum(A_new, 0.0)
        y_new = jnp.array([B_new, F_new, A_new])
        return (y_new, key), y_new

    (_, _), traj = jax.lax.scan(step, (init_state, rng_key),
                                 jnp.arange(n_steps))
    return traj


# =========================================================================
# MISC HELPERS
# =========================================================================

def shard_init_fn(time_offset, params, exogenous, global_init):
    del time_offset, params, exogenous
    return global_init


def make_init_state_fn(init_estimates, params):
    del params
    return init_estimates


def _prior_mean(ptype, pargs):
    if ptype == 'lognormal':
        return math.exp(pargs[0] + pargs[1]**2 / 2)
    elif ptype == 'normal':
        return pargs[0]
    return 0.0


def get_init_theta():
    all_config = OrderedDict()
    all_config.update(PARAM_PRIOR_CONFIG)
    all_config.update(INIT_STATE_PRIOR_CONFIG)
    return np.array([_prior_mean(pt, pa) for _, (pt, pa) in all_config.items()],
                    dtype=np.float32)


# =========================================================================
# ASSEMBLE
# =========================================================================

FSA_ESTIMATION = EstimationModel(
    name="fitness_strain_amplitude",
    version="1.0",
    n_states=3,
    n_stochastic=3,
    stochastic_indices=(0, 1, 2),
    state_bounds=((0.0, 1.0), (0.0, 10.0), (0.0, 5.0)),
    param_prior_config=PARAM_PRIOR_CONFIG,
    init_state_prior_config=INIT_STATE_PRIOR_CONFIG,
    frozen_params={
        'eps_A':   EPS_A_FROZEN,
        'eps_B':   EPS_B_FROZEN,
        'sigma_B': SIGMA_B_FROZEN,
        'sigma_F': SIGMA_F_FROZEN,
        'sigma_A': SIGMA_A_FROZEN,
    },
    propagate_fn=propagate_fn,
    diffusion_fn=diffusion_fn,
    obs_log_weight_fn=obs_log_weight_fn,
    align_obs_fn=align_obs_fn,
    shard_init_fn=shard_init_fn,
    forward_sde_fn=forward_sde_stochastic,
    get_init_theta_fn=get_init_theta,
    imex_step_fn=imex_step_fn,
    obs_log_prob_fn=obs_log_prob_fn,
    make_init_state_fn=make_init_state_fn,
    obs_sample_fn=obs_sample_fn,
    gaussian_obs_fn=gaussian_obs_fn,
    exogenous_keys=('T_B', 'Phi'),
)
"""
models/sleep_wake_20p/estimation.py — 20-Parameter Sleep-Wake-Adenosine SDE.
===============================================================================
Date:    17 April 2026
Version: 1.0

EstimationModel for the 20-parameter SDE derived in
Identifiability_Proof_17_Parameter_Sleep_Wake_Adenosine_Model.md
plus 3 diffusion temperatures {T_W, T_Z, T_a}.

All JAX dynamics live in ._dynamics.  This file wires them into the
callables that log_density/* and runner.py expect.

Parameter vector (length 20):

    kappa lmbda gamma_3 tau_W tau_Z phi HR_base alpha_HR sigma_HR c_tilde
    tau_a beta_Z Vh Vn T_W T_Z T_a

Initial-state vector (length 3):

    W_0 Zt_0 a_0

Note: V_h, V_n are ESTIMATED as parameters (constants in Phase 1) whereas
W_0, Zt_0, a_0 are estimated initial states.  They differ operationally:
init-states enter make_init_state_fn; parameters enter the drift/obs.
"""

from __future__ import annotations

import math
import numpy as np
from collections import OrderedDict

import jax
import jax.numpy as jnp
from jax import Array

from estimation_model import EstimationModel
from _likelihood_constants import HALF_LOG_2PI
from models.sleep_wake_20p import _dynamics as dyn


# ─── Priors  (from proof doc §2.5) ──────────────────────────────

PARAM_PRIOR_CONFIG = OrderedDict([
    # Flip-flop
    ('kappa',    ('lognormal', (math.log(6.67), 0.5))),
    ('lmbda',    ('lognormal', (math.log(32.0), 0.5))),
    # NOTE: gamma_3 was re-centred from 60 → 8 following BUG_REPORT_gamma3_sleep_depth.md.
    # The ODE-calibrated value of 60 is catastrophically wrong for the SDE: with
    # T_W = 0.01 the W noise floor during sleep (mean W ≈ 0.033) suppresses Zt below
    # c_tilde.  The biologically correct SDE value is ~8.  sigma=0.5 gives a 95% CI
    # of [3.0, 21.3] — centred on 8 and wide enough to span the ~4–20 plausible range.
    ('gamma_3',  ('lognormal', (math.log(8.0), 0.5))),
    ('tau_W',    ('lognormal', (math.log(2.0),  0.3))),
    ('tau_Z',    ('lognormal', (math.log(2.0),  0.3))),
    # Circadian
    ('phi',      ('vonmises',  (-math.pi / 3.0, 4.0))),
    # HR
    ('HR_base',  ('normal',    (50.0, 5.0))),
    ('alpha_HR', ('lognormal', (math.log(25.0), 0.3))),
    ('sigma_HR', ('lognormal', (math.log(8.0),  0.3))),
    # Binary sleep
    # NOTE: c_tilde raised from 1.5 → 3.0.  With Zt lower-bounded at 0,
    # c_tilde=1.5 gives σ(-1.5)=18% false sleep labels during wake.
    # c_tilde=3.0 drops this floor to σ(-3.0)=5%.  sigma=0.5 gives 95% CI
    # [1.8, 4.2] — centred on 3.0, enough to span the plausible range.
    ('c_tilde',  ('normal',    (3.0, 0.5))),
    # Adenosine
    ('tau_a',    ('lognormal', (math.log(3.0),  0.3))),
    # NOTE: beta_Z raised from 1.5 → 2.5 to compensate for the higher c_tilde.
    # With beta_Z=2.5 and a≈1 at sleep onset, Zt_sleep≈5.4 >> c_tilde=3.0.
    ('beta_Z',   ('lognormal', (math.log(2.5),  0.4))),
    # Behavioural potentials  (constants in Phase 1)
    ('Vh',       ('normal',    (1.0, 0.7))),
    ('Vn',       ('normal',    (1.0, 0.7))),
    # Diffusion temperatures
    ('T_W',      ('lognormal', (math.log(0.01), 0.5))),
    ('T_Z',      ('lognormal', (math.log(0.05), 0.5))),
    ('T_a',      ('lognormal', (math.log(0.01), 0.5))),
])

INIT_STATE_PRIOR_CONFIG = OrderedDict([
    ('W_0',  ('beta',      (3.0, 3.0))),
    ('Zt_0', ('normal',    (3.5, 0.8))),   # raised from 1.8 to match new c_tilde=3.0
    ('a_0',  ('lognormal', (math.log(0.5), 0.5))),
])

_PK = list(PARAM_PRIOR_CONFIG.keys())
PI = {k: i for i, k in enumerate(_PK)}


# ─── Grid alignment ─────────────────────────────────────────────

def align_obs_fn(obs_data, t_steps, dt_hours):
    """Convert simulator channel dict into grid-aligned JAX arrays.

    Expects obs_data to contain (optionally) 't_idx', 'hr_value',
    'sleep_label'.  Missing channels are filled with zeros and their
    'present' masks left as zero.
    """
    del dt_hours  # grids are dense (simulator output)
    T = t_steps
    hr_val = np.zeros(T, dtype=np.float32)
    hr_pres = np.zeros(T, dtype=np.float32)
    sl_lab = np.zeros(T, dtype=np.int32)
    sl_pres = np.zeros(T, dtype=np.float32)

    if 't_idx' in obs_data:
        idx = np.asarray(obs_data['t_idx']).astype(int)
        if 'hr_value' in obs_data:
            hr_val[idx] = np.asarray(obs_data['hr_value']).astype(np.float32)
            hr_pres[idx] = 1.0
        if 'sleep_label' in obs_data:
            sl_lab[idx] = np.asarray(obs_data['sleep_label']).astype(np.int32)
            sl_pres[idx] = 1.0

    return {
        'hr_value':     jnp.array(hr_val),
        'hr_present':   jnp.array(hr_pres),
        'sleep_label':  jnp.array(sl_lab),
        'sleep_present':jnp.array(sl_pres),
        'has_any_obs':  jnp.array(np.maximum(hr_pres, sl_pres)),
    }


# ─── IMEX step for direct_scan / EKF ────────────────────────────

def imex_step_fn(y, t, dt, params, grid_obs):
    del grid_obs  # no exogenous inputs in this model
    return dyn.imex_step_deterministic(y, t, dt, params, PI)


def diffusion_fn(params):
    return dyn.diffusion(params, PI)


# ─── Observation log-probabilities ──────────────────────────────

def _hr_log_prob(y, grid_obs, k, params):
    sigma_HR = params[PI['sigma_HR']]
    mean = dyn.hr_mean(y, params, PI)
    resid = grid_obs['hr_value'][k] - mean
    return grid_obs['hr_present'][k] * (
        -0.5 * (resid / sigma_HR) ** 2 - jnp.log(sigma_HR) - HALF_LOG_2PI)


def _sleep_log_prob(y, grid_obs, k, params):
    p = dyn.sleep_prob(y, params, PI)
    p_safe = jnp.clip(p, 1e-8, 1.0 - 1e-8)
    s = grid_obs['sleep_label'][k].astype(p.dtype)
    return grid_obs['sleep_present'][k] * (
        s * jnp.log(p_safe) + (1.0 - s) * jnp.log(1.0 - p_safe))


def obs_log_prob_fn(y, grid_obs, k, params):
    """Total observation log-prob = HR (Gaussian) + binary sleep."""
    return _hr_log_prob(y, grid_obs, k, params) + \
           _sleep_log_prob(y, grid_obs, k, params)


def gaussian_obs_fn(y, grid_obs, k, params):
    """HR-only Gaussian info for the EKF (sleep is non-Gaussian)."""
    return {
        'mean':     jnp.array([dyn.hr_mean(y, params, PI)]),
        'value':    jnp.array([grid_obs['hr_value'][k]]),
        'cov_diag': jnp.array([params[PI['sigma_HR']] ** 2]),
        'present':  jnp.array([grid_obs['hr_present'][k]]),
    }


def obs_log_weight_fn(x_new, grid_obs, k, params):
    """PF observation weight: binary sleep only.

    HR is handled inside propagate_fn by the guided proposal (predictive
    weight).  Returning sleep here mirrors the sleep_wake 6-state pattern.
    """
    return _sleep_log_prob(x_new, grid_obs, k, params)


# ─── Guided proposal for W, conditioning on HR ──────────────────

def propagate_fn(y, t, dt, params, grid_obs, k, sigma_diag, noise, rng_key):
    """Stochastic IMEX step with Pitt-Shephard guided proposal on W.

    HR observation tilts the prior on W toward the value consistent with
    the observed heart-rate residual.  Remaining states use bootstrap.
    """
    del rng_key  # noise already drawn outside
    y_next, mu_prior, var_prior = dyn.imex_step_stochastic(
        y, t, dt, params, sigma_diag, noise, PI)

    HR_base  = params[PI['HR_base']]
    alpha_HR = params[PI['alpha_HR']]
    sigma_HR = params[PI['sigma_HR']]

    # Precision / information form Kalman update for the W marginal
    W_prec = 1.0 / jnp.maximum(var_prior[0], 1e-12)
    W_info = W_prec * mu_prior[0]
    hr_pres = grid_obs['hr_present'][k]
    W_info += hr_pres * alpha_HR * (grid_obs['hr_value'][k] - HR_base) \
              / (sigma_HR ** 2)
    W_prec += hr_pres * (alpha_HR ** 2) / (sigma_HR ** 2)
    W_var = 1.0 / W_prec
    W_mu = W_var * W_info

    # Resample W from guided posterior (overrides the earlier bootstrap draw)
    W_new = jnp.clip(W_mu + jnp.sqrt(W_var) * noise[0], 0.0, 1.0)
    y_next = y_next.at[0].set(W_new)
    # Keep Zt in [0, A]
    y_next = y_next.at[1].set(jnp.clip(y_next[1], 0.0, dyn.A_SCALE))
    # Keep a nonneg
    y_next = y_next.at[2].set(jnp.maximum(y_next[2], 0.0))

    # Predictive HR log-weight  (accounts for the tilt)
    pred_var_hr = sigma_HR ** 2 + alpha_HR ** 2 * var_prior[0]
    pred_mu_hr = HR_base + alpha_HR * mu_prior[0]
    lw = hr_pres * (
        -0.5 * (grid_obs['hr_value'][k] - pred_mu_hr) ** 2 / pred_var_hr
        - 0.5 * jnp.log(pred_var_hr) - HALF_LOG_2PI)
    return y_next, lw


# ─── Initial-state assembly and shard init ──────────────────────

def make_init_state_fn(init_estimates, params):
    """Build the full 6D state vector [W, Zt, a, C, Vh, Vn] at t=0."""
    phi = params[PI['phi']]
    return jnp.array([
        init_estimates[0],        # W_0
        init_estimates[1],        # Zt_0
        init_estimates[2],        # a_0
        jnp.sin(phi),             # C(0)
        params[PI['Vh']],         # V_h constant
        params[PI['Vn']],         # V_n constant
    ])


def shard_init_fn(time_offset, params, exogenous, global_init):
    """Phase-conditioned init at arbitrary shard start.  Constants pass through."""
    del exogenous
    dt_h = 5.0 / 60.0
    t_start = time_offset * dt_h
    phi = params[PI['phi']]
    return jnp.array([
        global_init[0],
        global_init[1],
        global_init[2],
        jnp.sin(2.0 * jnp.pi * t_start / 24.0 + phi),
        params[PI['Vh']],
        params[PI['Vn']],
    ])


# ─── Synthetic observation sampling ─────────────────────────────

def obs_sample_fn(y, exog, k, params, rng_key):
    del exog, k  # no exogenous covariates
    k1, k2 = jax.random.split(rng_key)
    hr = dyn.hr_mean(y, params, PI) + \
         params[PI['sigma_HR']] * jax.random.normal(k1, dtype=y.dtype)
    p_sleep = dyn.sleep_prob(y, params, PI)
    label = (jax.random.uniform(k2, dtype=y.dtype) < p_sleep).astype(jnp.int32)
    return {'hr_value': hr, 'sleep_label': label}


# ─── Forward simulation for MAP trajectory ──────────────────────

def forward_sde_fn(init_state, params, exogenous, dt, n_steps, rng_key=None):
    """Stochastic forward Euler-Maruyama (6D state)."""
    del exogenous
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    sigma_d = diffusion_fn(params)
    y0 = make_init_state_fn(init_state, params)

    def step(carry, i):
        y, key = carry
        key, nk = jax.random.split(key)
        noise = jax.random.normal(nk, (6,))
        t = i * dt
        y_next, _, _ = dyn.imex_step_stochastic(
            y, t, dt, params, sigma_d, noise, PI)
        return (y_next, key), y_next

    (_, _), traj = jax.lax.scan(step, (y0, rng_key), jnp.arange(n_steps))
    return traj


# ─── get_init_theta (prior medians, match proof doc §2.5) ───────

def _prior_mean(ptype, pargs):
    if ptype == 'lognormal': return math.exp(pargs[0] + pargs[1] ** 2 / 2)
    if ptype == 'normal':    return pargs[0]
    if ptype == 'vonmises':  return pargs[0]
    if ptype == 'beta':      return pargs[0] / (pargs[0] + pargs[1])
    return 0.0


def get_init_theta():
    all_cfg = OrderedDict()
    all_cfg.update(PARAM_PRIOR_CONFIG)
    all_cfg.update(INIT_STATE_PRIOR_CONFIG)
    means = [_prior_mean(pt, pa) for _, (pt, pa) in all_cfg.items()]
    return np.array(means, dtype=np.float32)


# ─── Assemble ───────────────────────────────────────────────────

SLEEP_WAKE_20P_ESTIMATION = EstimationModel(
    name="sleep_wake_20p",
    version="1.0",
    n_states=6,
    n_stochastic=3,
    stochastic_indices=(0, 1, 2),
    state_bounds=((0.0, 1.0), (0.0, dyn.A_SCALE), (0.0, 5.0),
                  (-1.0, 1.0), (-5.0, 5.0), (-5.0, 5.0)),
    param_prior_config=PARAM_PRIOR_CONFIG,
    init_state_prior_config=INIT_STATE_PRIOR_CONFIG,
    frozen_params={},
    propagate_fn=propagate_fn,
    diffusion_fn=diffusion_fn,
    obs_log_weight_fn=obs_log_weight_fn,
    align_obs_fn=align_obs_fn,
    shard_init_fn=shard_init_fn,
    forward_sde_fn=forward_sde_fn,
    get_init_theta_fn=get_init_theta,
    imex_step_fn=imex_step_fn,
    obs_log_prob_fn=obs_log_prob_fn,
    make_init_state_fn=make_init_state_fn,
    obs_sample_fn=obs_sample_fn,
    gaussian_obs_fn=gaussian_obs_fn,
    exogenous_keys=(),
)

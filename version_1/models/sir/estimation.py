"""models/sir/estimation.py — JAX-side EstimationModel for stochastic SIR.

Inference companion to ``models.sir.simulation``. Pure-JAX dynamics live
in ``models.sir._dynamics``; this file holds the priors, grid alignment,
and the EstimationModel contract that the SMC² framework consumes.

Conventions:

  * State vector y = [S, I]  (R = N - S - I eliminated; conserved)
  * dt is in hours (matches simulation.py's dt_hours = 1.0)
  * Rates β, γ are per hour (Set A truth: β = 0.0692/hr ≈ 1.66/day,
    γ = 0.0208/hr = 0.5/day, R₀ = 3.32 — the Anderson & May 1978
    boarding-school flu outbreak, paper-parity benchmark).
  * Bootstrap PF (no Pitt-Shephard guidance) — both observation
    channels are non-Gaussian-or-sparse and contribute via
    ``obs_log_weight_fn``. Simpler than SWAT's HR-tilted proposal.

Priors are centered on Set A. For Sets B/C/D with different rate
magnitudes, the SMC² driver should override with scenario-specific
priors via ``SirRollingConfig``; or call ``make_sir_estimation`` with
a ``frozen_params`` override to switch the population size N.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

import jax
import jax.numpy as jnp

from estimation_model import EstimationModel
from models.sir import _dynamics as dyn


# =========================================================================
# Priors (Set A-centered)
# =========================================================================

PARAM_PRIOR_CONFIG = OrderedDict([
    # Transmission rate (per hour). Set A truth: 0.0692/hr (1.66/day).
    ('beta',    ('lognormal', (math.log(0.0692), 0.4))),
    # Recovery rate (per hour). Set A truth: 0.0208/hr (0.5/day, 2-day infection).
    ('gamma',   ('lognormal', (math.log(0.0208), 0.3))),
    # Case-detection probability. Beta(8, 2): mean 0.8, mode 0.875.
    # Set A boarding-school truth ρ = 1 sits at the upper edge.
    ('rho',     ('beta',      (8.0, 2.0))),
    # Serology survey noise (prevalence-scale).
    ('sigma_z', ('lognormal', (math.log(0.02),  0.5))),
    # Diffusion temperatures (per-state Itô noise σ²).
    ('T_S',     ('lognormal', (math.log(1.0),   0.5))),
    ('T_I',     ('lognormal', (math.log(1.0),   0.5))),
])

INIT_STATE_PRIOR_CONFIG = OrderedDict([
    # Initial infected count. Set A truth I_0 = 1 (single index case).
    ('I_0',     ('lognormal', (math.log(1.0),   1.0))),
])

# ─── Scenario-specific prior overrides ─────────────────────────────────
# The default `PARAM_PRIOR_CONFIG` above is centered on Set A truth (the
# paper-parity benchmark). For Sets B/C/D — same model, different operating
# regime (lower rates, larger population, different ρ) — the priors need
# to track. Otherwise the SMC² inference fights a 1-2 SD prior-truth gap
# in every dimension and never converges. (See SF Path B-fixed history:
# `outputs/SF_BEST_PRACTICE_2_models.md` for the analogous SWAT story.)
#
# Each entry is a partial override; missing keys fall through to the
# Set-A default. ``make_sir_estimation`` consumes these via the
# ``param_prior_overrides`` kwarg.

PARAM_PRIOR_OVERRIDES_B = {
    # Set B: β = 0.5/day = 0.0208/hr, γ = 0.2/day = 0.00833/hr, R₀ = 2.5,
    #        σ_z = 0.005, T_S = T_I = 1.0 (same diffusion as Set A — sim
    #        keeps temperatures fixed; only rates and population change),
    #        I_0 = 5 (small community seed).
    'beta':    ('lognormal', (math.log(0.0208),  0.4)),
    'gamma':   ('lognormal', (math.log(0.00833), 0.3)),
    'rho':     ('beta',      (5.0, 5.0)),     # mean 0.5 (community reporting)
    'sigma_z': ('lognormal', (math.log(0.005),  0.5)),
    'I_0':     ('lognormal', (math.log(5.0),    1.0)),
    # T_S, T_I retain Set-A defaults (LogNormal(log 1, 0.5)) — match simulator truth.
}

PARAM_PRIOR_OVERRIDES_C = {
    # Set C: β = 0.8/day = 0.0333/hr, γ = 0.2/day = 0.00833/hr, R₀ = 4.0,
    #        σ_z = 0.005, I_0 = 10.
    'beta':    ('lognormal', (math.log(0.0333),  0.4)),
    'gamma':   ('lognormal', (math.log(0.00833), 0.3)),
    'rho':     ('beta',      (5.0, 5.0)),
    'sigma_z': ('lognormal', (math.log(0.005),  0.5)),
    'I_0':     ('lognormal', (math.log(10.0),   1.0)),
}

PARAM_PRIOR_OVERRIDES_D = {
    # Set D: β = 0.6/day = 0.025/hr, γ = 0.2/day = 0.00833/hr, R₀ = 3.0,
    #        σ_z = 0.005, I_0 = 10, v = 0.02/day = 8.3e-4/hr (frozen, not estimated).
    'beta':    ('lognormal', (math.log(0.025),   0.4)),
    'gamma':   ('lognormal', (math.log(0.00833), 0.3)),
    'rho':     ('beta',      (5.0, 5.0)),
    'sigma_z': ('lognormal', (math.log(0.005),  0.5)),
    'I_0':     ('lognormal', (math.log(10.0),   1.0)),
}

# Frozen (non-estimated) parameters — population size + vaccination rate.
# Per-scenario overrides go via ``make_sir_estimation``: Set A N=763 boarding-
# school, Sets B/C/D N=10000 community.
DEFAULT_FROZEN_PARAMS = {
    'N':  763.0,    # Anderson-May boarding-school cohort
    'v':  0.0,      # vaccination rate (0 baseline; >0 in Set D)
}

_PK = list(PARAM_PRIOR_CONFIG.keys())
PI = {k: i for i, k in enumerate(_PK)}


# =========================================================================
# Grid alignment: simulator obs_data → grid-indexed JAX arrays
# =========================================================================

def align_obs_fn(obs_data, t_steps, dt_hours):
    """Convert simulator's per-channel obs dict into dense grid arrays.

    Simulator emits:
        {'cases':    {'t_idx', 'cases', 'bin_hours'},
         'serology': {'t_idx', 'prevalence'}}

    Either channel may be absent (e.g., real-data feeds without serology);
    its ``*_present`` mask stays at zero everywhere.
    """
    del dt_hours
    T = t_steps
    cases_count = np.zeros(T, dtype=np.int32)
    cases_present = np.zeros(T, dtype=np.float32)
    serology_value = np.zeros(T, dtype=np.float32)
    serology_present = np.zeros(T, dtype=np.float32)
    cases_bin_hours = 24.0    # default; overwritten if cases channel present

    if 'cases' in obs_data and 'cases' in obs_data['cases']:
        ch = obs_data['cases']
        idx = np.asarray(ch['t_idx']).astype(int)
        m = (idx >= 0) & (idx < T)
        cases_count[idx[m]] = np.asarray(ch['cases'])[m].astype(np.int32)
        cases_present[idx[m]] = 1.0
        cases_bin_hours = float(ch.get('bin_hours', 24.0))

    if 'serology' in obs_data and 'prevalence' in obs_data['serology']:
        ch = obs_data['serology']
        idx = np.asarray(ch['t_idx']).astype(int)
        m = (idx >= 0) & (idx < T)
        serology_value[idx[m]] = np.asarray(ch['prevalence'])[m].astype(np.float32)
        serology_present[idx[m]] = 1.0

    has_any = np.maximum(cases_present, serology_present)

    return {
        'cases_count':       jnp.array(cases_count),
        'cases_present':     jnp.array(cases_present),
        'cases_bin_hours':   jnp.float32(cases_bin_hours),
        'serology_value':    jnp.array(serology_value),
        'serology_present':  jnp.array(serology_present),
        'has_any_obs':       jnp.array(has_any),
    }


# =========================================================================
# Framework contract: factory closures over `frozen`
# =========================================================================

def _imex_step_fn(frozen):
    def imex_step_fn(y, t, dt, params, grid_obs):
        del grid_obs
        return dyn.imex_step_deterministic(y, t, dt, params, frozen, PI)
    return imex_step_fn


def diffusion_fn(params):
    """Diagonal diffusion magnitudes √T_S, √T_I."""
    return dyn.diffusion(params, PI)


def _obs_log_prob_fn(frozen):
    def obs_log_prob_fn(y, grid_obs, k, params):
        return (dyn.cases_log_prob(y, grid_obs, k, params, frozen, PI)
                + dyn.serology_log_prob(y, grid_obs, k, params, frozen, PI))
    return obs_log_prob_fn


def _obs_log_weight_fn(frozen):
    """Bootstrap PF reweighting: both channels go here (no guided proposal)."""
    def obs_log_weight_fn(x_new, grid_obs, k, params):
        return (dyn.cases_log_prob(x_new, grid_obs, k, params, frozen, PI)
                + dyn.serology_log_prob(x_new, grid_obs, k, params, frozen, PI))
    return obs_log_weight_fn


def _gaussian_obs_fn(frozen):
    """For the EKF: only serology is Gaussian; cases (Poisson) is non-Gaussian
    and is handled via ``obs_log_weight_fn``."""
    def gaussian_obs_fn(y, grid_obs, k, params):
        return {
            'mean':     jnp.array([dyn.serology_mean(y, frozen)]),
            'value':    jnp.array([grid_obs['serology_value'][k]]),
            'cov_diag': jnp.array([params[PI['sigma_z']] ** 2]),
            'present':  jnp.array([grid_obs['serology_present'][k]]),
        }
    return gaussian_obs_fn


def _propagate_fn(frozen):
    """Bootstrap PF propagation: stochastic Euler step, log-weight = 0.

    No HR-tilt-style guided proposal; the sparse Gaussian serology channel
    doesn't motivate it and the Poisson cases channel is non-Gaussian. The
    framework's contract allows ``log_w = 0`` for bootstrap proposals.
    """
    def propagate_fn(y, t, dt, params, grid_obs, k, sigma_diag, noise, rng_key):
        del grid_obs, k, rng_key
        y_next, _, _ = dyn.imex_step_stochastic(
            y, t, dt, params, sigma_diag, noise, frozen, PI)
        # Clip S, I to non-negative (SDE diffusion can briefly go negative).
        y_next = y_next.at[0].set(jnp.maximum(y_next[0], 0.0))
        y_next = y_next.at[1].set(jnp.maximum(y_next[1], 0.0))
        return y_next, jnp.float32(0.0)
    return propagate_fn


def _make_init_state_fn(frozen):
    """Build full state vector y(0) = [S_0, I_0] from estimable init."""
    def make_init_state_fn(init_estimates, params):
        del params
        I_0 = init_estimates[0]
        S_0 = frozen['N'] - I_0
        return jnp.array([S_0, I_0])
    return make_init_state_fn


def _shard_init_fn(frozen):
    """Init state at an arbitrary time-offset (rolling-window mid-trial start).

    For SIR there's no natural circadian phase to align — the rolling driver
    threads the previous-window's posterior mean as ``global_init``.
    """
    def shard_init_fn(time_offset, params, exogenous, global_init):
        del time_offset, params, exogenous
        return jnp.asarray(global_init, dtype=jnp.float64)
    return shard_init_fn


def _forward_sde_fn(frozen):
    """JAX forward Euler-Maruyama trajectory (for posterior MAP plots)."""
    def forward_sde_fn(init_state, params, exogenous, dt, n_steps, rng_key=None):
        del exogenous
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        sigma_d = diffusion_fn(params)
        y0 = jnp.asarray(init_state, dtype=jnp.float64)

        def step(carry, i):
            y, key = carry
            key, nk = jax.random.split(key)
            noise = jax.random.normal(nk, (2,))
            t = i * dt
            y_next, _, _ = dyn.imex_step_stochastic(
                y, t, dt, params, sigma_d, noise, frozen, PI)
            y_next = y_next.at[0].set(jnp.maximum(y_next[0], 0.0))
            y_next = y_next.at[1].set(jnp.maximum(y_next[1], 0.0))
            return (y_next, key), y_next

        (_, _), traj = jax.lax.scan(step, (y0, rng_key), jnp.arange(n_steps))
        return traj
    return forward_sde_fn


def _obs_sample_fn(frozen):
    """Synthetic observation sampler — used for posterior-predictive checks."""
    def obs_sample_fn(y, exog, k, params, rng_key):
        del exog, k
        k1, k2 = jax.random.split(rng_key)
        rate = dyn.cases_rate(y, params, frozen, PI)
        expected = rate * 24.0
        cases = jax.random.poisson(k1, expected).astype(jnp.int32)
        prev = dyn.serology_mean(y, frozen)
        sigma_z = params[PI['sigma_z']]
        prev_obs = prev + sigma_z * jax.random.normal(k2, dtype=y.dtype)
        return {'cases': cases, 'prevalence': prev_obs}
    return obs_sample_fn


# =========================================================================
# Prior-mean initialiser (SMC² particles)
# =========================================================================

def _prior_mean(ptype, pargs):
    if ptype == 'lognormal':
        return math.exp(pargs[0] + pargs[1] ** 2 / 2)
    if ptype == 'normal':
        return pargs[0]
    if ptype == 'beta':
        return pargs[0] / (pargs[0] + pargs[1])
    return 0.0


def get_init_theta():
    """Concatenated prior means for [PARAM_PRIOR_CONFIG, INIT_STATE_PRIOR_CONFIG]."""
    all_cfg = OrderedDict()
    all_cfg.update(PARAM_PRIOR_CONFIG)
    all_cfg.update(INIT_STATE_PRIOR_CONFIG)
    means = [_prior_mean(pt, pa) for _, (pt, pa) in all_cfg.items()]
    return np.array(means, dtype=np.float32)


# =========================================================================
# EstimationModel factory + canonical instance
# =========================================================================

def make_sir_estimation(
    frozen_params: dict | None = None,
    param_prior_overrides: dict | None = None,
) -> EstimationModel:
    """Build an EstimationModel with scenario-specific overrides.

    Args:
      frozen_params: override ``DEFAULT_FROZEN_PARAMS`` (e.g. switch N=763
        to N=10000 for Sets B/C/D, or activate vaccination v > 0).
      param_prior_overrides: override individual prior entries in
        ``PARAM_PRIOR_CONFIG`` (and / or ``INIT_STATE_PRIOR_CONFIG``) by
        key. Use ``PARAM_PRIOR_OVERRIDES_B/C/D`` for canonical Sets B/C/D.

    Defaults to Set A (boarding school).
    """
    frozen = dict(DEFAULT_FROZEN_PARAMS)
    if frozen_params:
        frozen.update(frozen_params)

    # Merge prior overrides (supports both PARAM_* and INIT_STATE_* keys).
    param_cfg = OrderedDict(PARAM_PRIOR_CONFIG)
    init_cfg = OrderedDict(INIT_STATE_PRIOR_CONFIG)
    if param_prior_overrides:
        for k, v in param_prior_overrides.items():
            if k in param_cfg:
                param_cfg[k] = v
            elif k in init_cfg:
                init_cfg[k] = v
            else:
                raise KeyError(
                    f"prior-override key {k!r} matches neither "
                    f"PARAM_PRIOR_CONFIG nor INIT_STATE_PRIOR_CONFIG")

    return EstimationModel(
        name="sir",
        version="1.0",
        n_states=2,
        n_stochastic=2,
        stochastic_indices=(0, 1),
        state_bounds=(
            (0.0, 1e9),    # S
            (0.0, 1e9),    # I
        ),
        param_prior_config=param_cfg,
        init_state_prior_config=init_cfg,
        frozen_params=frozen,
        propagate_fn=_propagate_fn(frozen),
        diffusion_fn=diffusion_fn,
        obs_log_weight_fn=_obs_log_weight_fn(frozen),
        align_obs_fn=align_obs_fn,
        shard_init_fn=_shard_init_fn(frozen),
        forward_sde_fn=_forward_sde_fn(frozen),
        get_init_theta_fn=get_init_theta,
        imex_step_fn=_imex_step_fn(frozen),
        obs_log_prob_fn=_obs_log_prob_fn(frozen),
        make_init_state_fn=_make_init_state_fn(frozen),
        obs_sample_fn=_obs_sample_fn(frozen),
        gaussian_obs_fn=_gaussian_obs_fn(frozen),
        exogenous_keys=(),
    )


# Default canonical instance (Set A: boarding-school flu, N=763).
SIR_ESTIMATION = make_sir_estimation()

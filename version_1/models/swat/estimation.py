"""
models/swat/estimation.py — Sleep-Wake-Adenosine-Testosterone SDE.
===================================================================
Date:    20 April 2026
Version: 1.0

EstimationModel for the 24-parameter SWAT SDE (spec-level count; see
Spec_24_Parameter_Sleep_Wake_Adenosine_Testosterone_Model.md).

Code-level accounting mirrors models.sleep_wake_20p:

  PARAM_PRIOR_CONFIG      (length 21): 17 old drift/obs + 3 old diffusion +
                                        4 new T-block drift + 1 new coupling +
                                        1 new T-diffusion = 26

  Wait — that's not right.  Let me count carefully:

    Old 20p PARAM_PRIOR_CONFIG = 17 entries:
       kappa, lmbda, gamma_3, tau_W, tau_Z, phi, HR_base, alpha_HR,
       sigma_HR, c_tilde, tau_a, beta_Z, Vh, Vn, T_W, T_Z, T_a
    New SWAT PARAM_PRIOR_CONFIG adds 6 entries:
       mu_0, mu_E, eta, tau_T, alpha_T, T_T
    Total: 23 entries.

  INIT_STATE_PRIOR_CONFIG (length 4): 3 old (W_0, Zt_0, a_0) + 1 new (T_0).

  Total estimable scalars: 23 + 4 = 27.

The spec's "24-parameter" count comes from taking the 17-param identifiability
proof's count (14 drift + 3 IC + Vh + Vn, which makes 17... no wait, the spec
says 17 drift+IC block + 5 T block + 1 coupling + 1 diffusion = 24).  The code
differs because it estimates T_W, T_Z, T_a (three extra fast-noise temperatures
not counted in the spec's 24).  Per §7.1 of the spec: "if T_W, T_Z, T_a are to
be estimated, the count rises to 27."  That 27 matches our code's total.

All JAX dynamics live in ._dynamics (extends sleep_wake_20p._dynamics).
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
from models.swat import _dynamics as dyn


# ─── Priors ──────────────────────────────────────────────────────
# 17-parameter block: identical to sleep_wake_20p priors.

PARAM_PRIOR_CONFIG = OrderedDict([
    # ── Inherited 17-block (flip-flop, circadian, HR, sleep, adenosine, V_h, V_n)
    ('kappa',    ('lognormal', (math.log(6.67), 0.5))),
    ('lmbda',    ('lognormal', (math.log(32.0), 0.5))),
    ('gamma_3',  ('lognormal', (math.log(8.0),  0.5))),
    ('tau_W',    ('lognormal', (math.log(2.0),  0.3))),
    ('tau_Z',    ('lognormal', (math.log(2.0),  0.3))),
    # V_c: phase shift (hours) from the frozen morning-type baseline
    # (PHI_MORNING_TYPE = -pi/3, peak wake ~10am solar).  Healthy: V_c ~ 0.
    # Positive V_c = subject's rhythm delayed relative to external light
    # (evening chronotype, delayed sleep phase, westbound jet lag, late
    # shift work).  Normal(0, 3) prior gives 95% CI roughly ±6 hours —
    # covers full plausible range including chronic shift-work patterns.
    # V_c serves dual purpose: (1) estimable parameter during inference,
    # (2) controllable intervention target for forward-simulated treatment
    # scenarios (light therapy, sleep scheduling, melatonin timing).
    ('V_c',      ('normal',    (0.0, 3.0))),
    ('HR_base',  ('normal',    (50.0, 5.0))),
    ('alpha_HR', ('lognormal', (math.log(25.0), 0.3))),
    ('sigma_HR', ('lognormal', (math.log(8.0),  0.3))),
    ('c_tilde',  ('normal',    (3.0, 0.5))),
    ('tau_a',    ('lognormal', (math.log(3.0),  0.3))),
    ('beta_Z',   ('lognormal', (math.log(2.5),  0.4))),
    ('Vh',       ('normal',    (1.0, 0.7))),
    ('Vn',       ('normal',    (1.0, 0.7))),
    # ── Inherited diffusion temperatures (fast-subsystem)
    ('T_W',      ('lognormal', (math.log(0.01), 0.5))),
    ('T_Z',      ('lognormal', (math.log(0.05), 0.5))),
    ('T_a',      ('lognormal', (math.log(0.01), 0.5))),

    # ── New Stuart-Landau testosterone block (see spec §10)
    # mu_0: signed; expected < 0.  Normal prior centred on -0.5.
    # 95% CI [-1.1, +0.1] — soft prior, allows mu_0 slightly > 0 if the
    # data demands it.
    ('mu_0',     ('normal',    (-0.5, 0.3))),
    # mu_E: expected > 0 with mu_0 + mu_E > 0.  LogNormal centred on 1.0
    # gives 95% CI roughly [0.55, 1.82] — includes values well above |mu_0|.
    ('mu_E',     ('lognormal', (math.log(1.0),  0.3))),
    # eta: Landau cubic coefficient.  Set T* = sqrt(mu/eta) = 1 at
    # healthy (mu = 0.5), giving eta = 0.5.
    ('eta',      ('lognormal', (math.log(0.5),  0.3))),
    # tau_T: slow timescale of T dynamics.  48 h = intermediate between
    # the 2-h flip-flop and the ~7-day vitality scale.
    ('tau_T',    ('lognormal', (math.log(48.0), 0.3))),
    # alpha_T: small-to-moderate coupling into u_W.  At T ~ 1, the
    # contribution alpha_T * T ~ 0.3 is comparable to the V_h contribution.
    ('alpha_T',  ('lognormal', (math.log(0.3),  0.3))),
    # T_T: small process-noise temperature.  Matches the order of the
    # other diffusion temps.  Reduced from 0.01 to 0.0001 so T doesn't
    # drift wildly on the 48h timescale.
    ('T_T',      ('lognormal', (math.log(0.0001), 0.5))),

    # ── Entrainment-amplitude block (V_h-anabolic structural fix) ──────
    # V_h modulates entrainment amplitude rather than entering u_W
    # directly.  A_W = lambda_amp_W * V_h, A_Z = lambda_amp_Z * V_h.
    # Calibrated against the validation pipeline's gating tests so the
    # healthy corner (V_h-healthy, V_n-low) gives strong entrainment
    # E ~ 1, and the depleted-V_h corner gives E ~ 0.
    ('lambda_amp_W', ('lognormal', (math.log(5.0), 0.3))),
    ('lambda_amp_Z', ('lognormal', (math.log(8.0), 0.3))),
    # V_n damper scale: damp(V_n) = exp(-V_n / V_n_scale).  Smaller =>
    # V_n bites entrainment faster.  Pinned at 2.0 in the validation
    # work; LogNormal here for consistency with the other positive
    # scale parameters.
    ('V_n_scale',    ('lognormal', (math.log(2.0), 0.3))),

    # ── 3-level ordinal sleep channel (Phase 1 addition) ──────────────
    # delta_c: gap between light/deep thresholds on Zt; c2 = c_tilde + delta_c.
    # Must be positive (ordering constraint satisfied automatically by LogNormal).
    # Prior centred on 1.5: with c_tilde ~ 3 and Zt_peak ~ 5, the deep stage
    # (Zt > 4.5) occupies the top ~30% of sleep time — matches typical sleep
    # architecture where deep sleep is ~20-25% of total sleep.
    ('delta_c',  ('lognormal', (math.log(1.5),  0.3))),

    # ── Steps Poisson channel (Phase 1 addition) ──────────────────────
    # lambda_base: step rate during sleep (true zero + small sensor noise).
    # Expected ~0.5 steps/hour; LogNormal with wide prior to allow near-zero.
    ('lambda_base', ('lognormal', (math.log(0.5),  0.7))),
    # lambda_step: peak step rate during wake.  ~200 steps/hour sustained
    # activity corresponds to a normal active day; LogNormal prior covers
    # sedentary-awake (50/h) to very-active (800/h).
    ('lambda_step', ('lognormal', (math.log(200.0), 0.5))),
    # W_thresh: wakefulness threshold above which step rate activates.
    # Normal centred on 0.6 covers drowsy-to-alert transition.
    ('W_thresh',    ('normal',    (0.6, 0.1))),

    # ── Garmin stress channel (Phase 1 addition) ──────────────────────
    # s_base: baseline stress score (W=0, V_n=0).  Normal on 0-100 scale.
    ('s_base',   ('normal',    (30.0, 10.0))),
    # alpha_s: wake modulation of stress.  Normal centred on 40 (W=0->30,
    # W=1 -> 70) — typical Garmin stress range for sedentary-wake.
    ('alpha_s',  ('normal',    (40.0, 10.0))),
    # beta_s: coupling of stress to nuisance-load V_n.  LogNormal centred
    # on 10 — provides the cross-channel constraint that disambiguates
    # V_n from V_h (both appear in u_W; only V_n is in the stress model).
    ('beta_s',   ('lognormal', (math.log(10.0), 0.5))),
    # sigma_s: stress observation noise (0-100 scale).  LogNormal prior.
    ('sigma_s',  ('lognormal', (math.log(15.0), 0.3))),
])

# 3 old ICs + 1 new (T_0).
INIT_STATE_PRIOR_CONFIG = OrderedDict([
    ('W_0',  ('beta',      (3.0, 3.0))),
    ('Zt_0', ('normal',    (3.5, 0.8))),
    ('a_0',  ('lognormal', (math.log(0.5), 0.5))),
    ('T_0',  ('lognormal', (math.log(1.0), 0.5))),   # new: testosterone IC
])

_PK = list(PARAM_PRIOR_CONFIG.keys())
PI = {k: i for i, k in enumerate(_PK)}


# ─── Grid alignment (identical to 20p) ───────────────────────────

def align_obs_fn(obs_data, t_steps, dt_hours):
    """Convert multi-channel simulator output into grid-aligned arrays.

    ``obs_data`` is the per-channel dict-of-dicts the SDEModel produces:

        {'hr':     {'t_idx': ..., 'hr_value': ...},
         'sleep':  {'t_idx': ..., 'sleep_level': ...},      # 3-level ordinal
         'steps':  {'t_idx': ..., 'steps': ..., 'bin_hours': float},
         'stress': {'t_idx': ..., 'stress_score': ...}}

    Each channel may be absent (e.g. real wearable data without stress);
    its ``*_present`` mask stays at zero everywhere.

    Backward-compatibility: if ``obs_data`` is a single flat dict
    containing 't_idx' at the top level (the legacy 20p / sleep_wake
    convention), it is interpreted as the HR + binary-sleep channel
    pair, with the binary sleep promoted to ordinal level 0/1
    (level 1 = "any sleep").
    """
    del dt_hours   # grids are dense (simulator output)
    T = t_steps

    def _zeros(dtype=np.float32):
        return np.zeros(T, dtype=dtype)

    hr_val   = _zeros(); hr_pres   = _zeros()
    sl_lev   = _zeros(np.int32); sl_pres = _zeros()
    st_count = _zeros(np.int32); st_pres = _zeros(); bin_hours_val = 0.25
    str_val  = _zeros(); str_pres  = _zeros()

    def _ingest_channel(ch_data, mask_into, value_key, value_into,
                        cast=lambda a: a.astype(np.float32)):
        idx = np.asarray(ch_data['t_idx']).astype(int)
        m = (idx >= 0) & (idx < T)
        value_into[idx[m]] = cast(np.asarray(ch_data[value_key])[m])
        mask_into[idx[m]] = 1.0

    if 't_idx' in obs_data:
        # Legacy flat-dict input (20p convention). Promote binary sleep
        # label to ordinal levels 0/1.
        if 'hr_value' in obs_data:
            _ingest_channel(obs_data, hr_pres, 'hr_value', hr_val)
        if 'sleep_label' in obs_data:
            _ingest_channel(obs_data, sl_pres, 'sleep_label', sl_lev,
                            cast=lambda a: a.astype(np.int32))
    else:
        # Multi-channel input. Each per-channel dict was already
        # window-extracted upstream.
        hr_ch  = obs_data.get('hr')     or obs_data.get('obs_hr')
        sl_ch  = obs_data.get('sleep')  or obs_data.get('obs_sleep')
        st_ch  = obs_data.get('steps')  or obs_data.get('obs_steps')
        str_ch = obs_data.get('stress') or obs_data.get('obs_stress')

        if hr_ch is not None and 'hr_value' in hr_ch:
            _ingest_channel(hr_ch, hr_pres, 'hr_value', hr_val)

        if sl_ch is not None:
            if 'sleep_level' in sl_ch:
                _ingest_channel(sl_ch, sl_pres, 'sleep_level', sl_lev,
                                cast=lambda a: a.astype(np.int32))
            elif 'sleep_label' in sl_ch:
                # Binary fallback (20p-era data); treat as ordinal level
                _ingest_channel(sl_ch, sl_pres, 'sleep_label', sl_lev,
                                cast=lambda a: a.astype(np.int32))

        if st_ch is not None and 'steps' in st_ch:
            _ingest_channel(st_ch, st_pres, 'steps', st_count,
                            cast=lambda a: a.astype(np.int32))
            bin_hours_val = float(st_ch.get('bin_hours', 0.25))

        if str_ch is not None and 'stress_score' in str_ch:
            _ingest_channel(str_ch, str_pres, 'stress_score', str_val)

    has_any = np.maximum.reduce([hr_pres, sl_pres, st_pres, str_pres])

    return {
        'hr_value':       jnp.array(hr_val),
        'hr_present':     jnp.array(hr_pres),
        'sleep_level':    jnp.array(sl_lev),
        'sleep_present':  jnp.array(sl_pres),
        'steps_count':    jnp.array(st_count),
        'steps_present':  jnp.array(st_pres),
        'bin_hours':      jnp.float32(bin_hours_val),
        'stress_value':   jnp.array(str_val),
        'stress_present': jnp.array(str_pres),
        'has_any_obs':    jnp.array(has_any),
    }


# ─── IMEX step for direct_scan / EKF ─────────────────────────────

def imex_step_fn(y, t, dt, params, grid_obs):
    del grid_obs
    return dyn.imex_step_deterministic(y, t, dt, params, PI)


def diffusion_fn(params):
    return dyn.diffusion(params, PI)


# ─── Observation log-probabilities (identical to 20p) ────────────

def _hr_log_prob(y, grid_obs, k, params):
    sigma_HR = params[PI['sigma_HR']]
    mean = dyn.hr_mean(y, params, PI)
    resid = grid_obs['hr_value'][k] - mean
    return grid_obs['hr_present'][k] * (
        -0.5 * (resid / sigma_HR) ** 2 - jnp.log(sigma_HR) - HALF_LOG_2PI)


def _sleep_log_prob(y, grid_obs, k, params):
    """3-level ordinal sleep log-prob.

    grid_obs['sleep_level'][k] is an int in {0, 1, 2}; gather the
    corresponding log-probability from
    ``dyn.sleep_level_log_probs(y, params, PI)``.
    """
    log_pmf = dyn.sleep_level_log_probs(y, params, PI)   # shape (3,)
    level_k = grid_obs['sleep_level'][k]
    ll = log_pmf[level_k]
    return grid_obs['sleep_present'][k] * ll


def _steps_log_prob(y, grid_obs, k, params):
    """Poisson log-prob: log P(N = n | rate * bin_hours).

    The simulator emits step counts only on 15-min bin starts, so
    grid_obs['steps_present'][k] is sparse. Uses lgamma for the log
    factorial term.
    """
    rate = dyn.steps_rate(y, params, PI)
    expected = rate * grid_obs['bin_hours']
    n = grid_obs['steps_count'][k].astype(expected.dtype)
    # Log Poisson pmf: n*log(λ) - λ - lgamma(n+1)
    log_pmf = n * jnp.log(jnp.maximum(expected, 1e-12)) \
              - expected - jax.lax.lgamma(n + 1.0)
    return grid_obs['steps_present'][k] * log_pmf


def _stress_log_prob(y, grid_obs, k, params):
    """Gaussian log-prob for the stress channel."""
    sigma_s = params[PI['sigma_s']]
    mean = dyn.stress_mean(y, params, PI)
    resid = grid_obs['stress_value'][k] - mean
    return grid_obs['stress_present'][k] * (
        -0.5 * (resid / sigma_s) ** 2 - jnp.log(sigma_s) - HALF_LOG_2PI)


def obs_log_prob_fn(y, grid_obs, k, params):
    """Joint log-likelihood across all 4 channels at step k."""
    return (_hr_log_prob(y, grid_obs, k, params)
            + _sleep_log_prob(y, grid_obs, k, params)
            + _steps_log_prob(y, grid_obs, k, params)
            + _stress_log_prob(y, grid_obs, k, params))


def gaussian_obs_fn(y, grid_obs, k, params):
    """Gaussian-channel info for the EKF (HR + stress; sleep + steps
    are non-Gaussian and handled via obs_log_weight_fn)."""
    return {
        'mean':     jnp.array([dyn.hr_mean(y, params, PI),
                                dyn.stress_mean(y, params, PI)]),
        'value':    jnp.array([grid_obs['hr_value'][k],
                                grid_obs['stress_value'][k]]),
        'cov_diag': jnp.array([params[PI['sigma_HR']] ** 2,
                                params[PI['sigma_s']] ** 2]),
        'present':  jnp.array([grid_obs['hr_present'][k],
                                grid_obs['stress_present'][k]]),
    }


def obs_log_weight_fn(x_new, grid_obs, k, params):
    """PF observation weight: non-Gaussian channels (sleep + steps)
    plus stress. HR is absorbed into ``propagate_fn``'s Pitt-Shephard
    guidance, so it must NOT be re-counted here.
    """
    return (_sleep_log_prob(x_new, grid_obs, k, params)
            + _steps_log_prob(x_new, grid_obs, k, params)
            + _stress_log_prob(x_new, grid_obs, k, params))


# ─── Guided proposal for W (identical pattern to 20p) ────────────
#
# NOTE: For the SWAT model we could also guide T through the HR tilt,
# since T enters u_W with coefficient alpha_T.  However, the guidance on
# W is far more informative (coefficient 1 via u_W vs coefficient alpha_T
# ~ 0.3, plus tau_W ~ 2 h vs tau_T ~ 48 h).  We keep the 20p pattern:
# guide only W, let T track via the bootstrap + (eventually) a dedicated
# proposal in future work.

def propagate_fn(y, t, dt, params, grid_obs, k, sigma_diag, noise, rng_key):
    """Stochastic IMEX step with Pitt-Shephard guided proposal on W.

    HR observation tilts the prior on W toward the value consistent with
    the observed heart-rate residual.  T uses bootstrap (no HR guidance).
    """
    del rng_key
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

    # Resample W from guided posterior
    W_new = jnp.clip(W_mu + jnp.sqrt(W_var) * noise[0], 0.0, 1.0)
    y_next = y_next.at[0].set(W_new)
    # Bound Zt, a, T
    y_next = y_next.at[1].set(jnp.clip(y_next[1], 0.0, dyn.A_SCALE))
    y_next = y_next.at[2].set(jnp.maximum(y_next[2], 0.0))
    y_next = y_next.at[3].set(jnp.maximum(y_next[3], 0.0))

    # Predictive HR log-weight
    pred_var_hr = sigma_HR ** 2 + alpha_HR ** 2 * var_prior[0]
    pred_mu_hr = HR_base + alpha_HR * mu_prior[0]
    lw = hr_pres * (
        -0.5 * (grid_obs['hr_value'][k] - pred_mu_hr) ** 2 / pred_var_hr
        - 0.5 * jnp.log(pred_var_hr) - HALF_LOG_2PI)
    return y_next, lw


# ─── Initial-state assembly and shard init ───────────────────────

def make_init_state_fn(init_estimates, params):
    """Build the full 7D state vector [W, Zt, a, T, C, Vh, Vn] at t=0.

    C(0) is the EXTERNAL light cycle at t=0 (frozen morning-type phase);
    it does not depend on V_c.  V_c only shifts the subject's internal
    drive entering u_W inside the dynamics.
    """
    return jnp.array([
        init_estimates[0],                # W_0
        init_estimates[1],                # Zt_0
        init_estimates[2],                # a_0
        init_estimates[3],                # T_0 (new)
        jnp.sin(dyn.PHI_MORNING_TYPE),    # C(0): external light cycle
        params[PI['Vh']],
        params[PI['Vn']],
    ])


def shard_init_fn(time_offset, params, exogenous, global_init):
    """Phase-conditioned init at arbitrary shard start.

    C-state initialised to the EXTERNAL light cycle at the shard start
    time (frozen morning-type phase; no V_c shift here).
    """
    del exogenous
    dt_h = 5.0 / 60.0
    t_start = time_offset * dt_h
    return jnp.array([
        global_init[0],
        global_init[1],
        global_init[2],
        global_init[3],              # T_0 (new)
        jnp.sin(2.0 * jnp.pi * t_start / 24.0 + dyn.PHI_MORNING_TYPE),
        params[PI['Vh']],
        params[PI['Vn']],
    ])


# ─── Synthetic observation sampling ──────────────────────────────

def obs_sample_fn(y, exog, k, params, rng_key):
    del exog, k
    k1, k2 = jax.random.split(rng_key)
    hr = dyn.hr_mean(y, params, PI) + \
         params[PI['sigma_HR']] * jax.random.normal(k1, dtype=y.dtype)
    p_sleep = dyn.sleep_prob(y, params, PI)
    label = (jax.random.uniform(k2, dtype=y.dtype) < p_sleep).astype(jnp.int32)
    return {'hr_value': hr, 'sleep_label': label}


# ─── Forward simulation for MAP trajectory ───────────────────────

def forward_sde_fn(init_state, params, exogenous, dt, n_steps, rng_key=None):
    """Stochastic forward Euler-Maruyama (7D state)."""
    del exogenous
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    sigma_d = diffusion_fn(params)
    y0 = make_init_state_fn(init_state, params)

    def step(carry, i):
        y, key = carry
        key, nk = jax.random.split(key)
        noise = jax.random.normal(nk, (7,))
        t = i * dt
        y_next, _, _ = dyn.imex_step_stochastic(
            y, t, dt, params, sigma_d, noise, PI)
        return (y_next, key), y_next

    (_, _), traj = jax.lax.scan(step, (y0, rng_key), jnp.arange(n_steps))
    return traj


# ─── get_init_theta (prior medians) ──────────────────────────────

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


# ─── Assemble ────────────────────────────────────────────────────

SWAT_ESTIMATION = EstimationModel(
    name="swat",
    version="1.0",
    n_states=7,
    n_stochastic=4,
    stochastic_indices=(0, 1, 2, 3),
    state_bounds=(
        (0.0, 1.0),         # W
        (0.0, dyn.A_SCALE), # Zt
        (0.0, 5.0),         # a
        (0.0, 5.0),         # T (new)
        (-1.0, 1.0),        # C
        (-5.0, 5.0),        # Vh
        (-5.0, 5.0),        # Vn
    ),
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

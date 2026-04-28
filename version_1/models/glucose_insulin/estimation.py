"""models/glucose_insulin/estimation.py — Bergman model EstimationModel.

Inference companion to ``models.glucose_insulin.simulation``. Pure-JAX
dynamics live in ``models.glucose_insulin._dynamics``; this file holds
the priors, grid alignment, and the EstimationModel contract that the
SMC² framework consumes.

Conventions:

  * State y = [G, X, I]  in [mg/dL, 1/hr, μU/mL]
  * dt is in hours (5/60 = 5 minutes per bin)
  * Pitt-Shephard guided proposal via the Gaussian CGM channel — Kalman
    update on G at every CGM obs (every 5-min bin), mirroring SWAT's
    HR-tilted W proposal. The dense CGM observation provides per-step
    guidance which is structurally what SIR Sets B/C/D were missing.

The meal & insulin schedules are SCENARIO-SPECIFIC exogenous inputs (not
estimated). They are baked into the EstimationModel via the
``make_glucose_insulin_estimation`` factory, which closes over the
schedules and passes them through ``align_obs_fn`` into ``grid_obs`` for
JAX evaluation.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

import jax
import jax.numpy as jnp

from estimation_model import EstimationModel
from _likelihood_constants import HALF_LOG_2PI
from models.glucose_insulin import _dynamics as dyn


_TAU_GASTRIC = 0.5     # hours; matches simulation.py
_TAU_BOLUS = 0.5       # hours; matches simulation.py


# =========================================================================
# Priors (Set A-centered; Bergman 1979 healthy-cohort means)
# =========================================================================

PARAM_PRIOR_CONFIG = OrderedDict([
    # Glucose effectiveness (non-insulin-mediated disposal). /hr
    ('p1',        ('lognormal', (math.log(1.8),    0.3))),
    # Remote insulin decay. /hr
    ('p2',        ('lognormal', (math.log(1.5),    0.3))),
    # Insulin sensitivity coefficient. /(hr²·μU/mL)
    ('p3',        ('lognormal', (math.log(4.68e-2), 0.5))),
    # Plasma insulin clearance. /hr
    ('k',         ('lognormal', (math.log(18.0),   0.3))),
    # Basal glucose. mg/dL — biologically tight.
    ('Gb',        ('normal',    (90.0, 8.0))),
    # CGM observation noise. mg/dL
    ('sigma_cgm', ('lognormal', (math.log(8.0),    0.3))),
    # Glucose diffusion temperature.
    ('T_G',       ('lognormal', (math.log(1.0),    0.5))),
])

INIT_STATE_PRIOR_CONFIG = OrderedDict([
    ('G_0',  ('normal',    (90.0, 10.0))),
    ('I_0',  ('lognormal', (math.log(7.0), 0.3))),
])

# Frozen non-estimated parameters — physiology constants + scenario-specific
# β-cell secretion / basal-insulin / population-volume settings.
DEFAULT_FROZEN_PARAMS = {
    'Ib':       7.0,        # μU/mL basal insulin (healthy)
    'V_G':      1.6,        # dL/kg glucose volume of distribution
    'V_I':      1.2,        # dL/kg insulin volume of distribution
    'BW':       70.0,       # kg body weight
    'T_X':      1e-6,       # X process noise (essentially deterministic)
    'T_I':      0.5,        # I process noise (μU/mL)²/hr
    'n_beta':   8.0,        # β-cell secretion rate (Bergman 1981 ext.)
    'h_beta':   90.0,       # secretion threshold
}

_PK = list(PARAM_PRIOR_CONFIG.keys())
PI = {k: i for i, k in enumerate(_PK)}


# =========================================================================
# Scenario-specific prior overrides (Sets B/C/D)
# =========================================================================

PARAM_PRIOR_OVERRIDES_B = {
    # Insulin resistance — SI = p₃/p₂ halved relative to A.
    'p3': ('lognormal', (math.log(2.34e-2), 0.5)),  # half of Set A
}

PARAM_PRIOR_OVERRIDES_C = {
    # T1D no-control — basal insulin is zero, so prior on Ib is moot
    # (Ib is frozen, not estimated). The estimable params are otherwise
    # similar; no scenario-specific override needed beyond Ib in frozen.
}

PARAM_PRIOR_OVERRIDES_D = {
    # T1D with open-loop insulin — same as Set C from the inference side
    # (the insulin schedule is a known exogenous input, not estimated).
}


# =========================================================================
# Grid alignment + meal/insulin schedule arrays
# =========================================================================

def _build_rate_arrays(meal_schedule, insulin_schedule, t_steps, dt_h,
                        V_G, BW):
    """Pre-compute D_rate[k] and I_input_rate[k] arrays for the trial.

    Computed once at align_obs time so the JAX inner PF can index them
    directly without summing over meals every step.
    """
    D_rate = np.zeros(t_steps, dtype=np.float32)
    I_rate = np.zeros(t_steps, dtype=np.float32)

    if meal_schedule:
        for k in range(t_steps):
            t = k * dt_h
            r = 0.0
            for t_meal, carbs in meal_schedule:
                dt_m = t - t_meal
                if dt_m >= 0:
                    r += (carbs / _TAU_GASTRIC ** 2) * dt_m \
                         * np.exp(-dt_m / _TAU_GASTRIC)
            D_rate[k] = r * 1000.0 / (V_G * BW)

    if insulin_schedule:
        V_I_mL = insulin_schedule['V_I_BW'] * 100.0
        boluses = insulin_schedule['boluses']
        basal = insulin_schedule['basal_rate_U_hr']
        for k in range(t_steps):
            t = k * dt_h
            r = basal
            for t_bolus, units in boluses:
                dt_b = t - t_bolus
                if dt_b >= 0:
                    r += (units / _TAU_BOLUS ** 2) * dt_b \
                         * np.exp(-dt_b / _TAU_BOLUS)
            I_rate[k] = r * 1e6 / V_I_mL

    return D_rate, I_rate


def _make_align_obs_fn(meal_schedule, insulin_schedule, V_G, BW):
    """Closure that returns an align_obs_fn baked with this scenario's schedules."""
    def align_obs_fn(obs_data, t_steps, dt_hours):
        T = t_steps
        cgm_value = np.zeros(T, dtype=np.float32)
        cgm_present = np.zeros(T, dtype=np.float32)
        carbs_g = np.zeros(T, dtype=np.float32)
        carbs_present = np.zeros(T, dtype=np.float32)
        carb_truth = np.zeros(T, dtype=np.float32)

        if 'cgm' in obs_data and 'cgm_value' in obs_data['cgm']:
            ch = obs_data['cgm']
            idx = np.asarray(ch['t_idx']).astype(int)
            m = (idx >= 0) & (idx < T)
            cgm_value[idx[m]] = np.asarray(ch['cgm_value'])[m].astype(np.float32)
            cgm_present[idx[m]] = 1.0

        if 'meal_carbs' in obs_data and 'carbs_g' in obs_data['meal_carbs']:
            ch = obs_data['meal_carbs']
            idx = np.asarray(ch['t_idx']).astype(int)
            m = (idx >= 0) & (idx < T)
            carbs_g[idx[m]] = np.asarray(ch['carbs_g'])[m].astype(np.float32)
            carbs_present[idx[m]] = 1.0

        # carb_truth from the meal schedule (the Poisson rate parameter)
        for t_meal, c_truth in (meal_schedule or []):
            idx = int(round(t_meal / dt_hours))
            if 0 <= idx < T:
                carb_truth[idx] = float(c_truth)

        D_rate, I_rate = _build_rate_arrays(
            meal_schedule, insulin_schedule, T, dt_hours, V_G, BW)

        has_any = np.maximum(cgm_present, carbs_present)

        return {
            'cgm_value':       jnp.array(cgm_value),
            'cgm_present':     jnp.array(cgm_present),
            'carbs_g':         jnp.array(carbs_g),
            'carbs_present':   jnp.array(carbs_present),
            'carb_truth':      jnp.array(carb_truth),
            'D_rate':          jnp.array(D_rate),
            'I_input_rate':    jnp.array(I_rate),
            'has_any_obs':     jnp.array(has_any),
        }
    return align_obs_fn


# =========================================================================
# Framework contract: factory closures over `frozen` and aux
# =========================================================================

def diffusion_fn(params, frozen):
    return dyn.diffusion(params, frozen, PI)


def _make_aux_at_step(grid_obs, k):
    """Pull D_rate / I_rate for step k out of grid_obs into a tiny aux dict."""
    return {
        'D_rate_at_t':         grid_obs['D_rate'][k],
        'I_input_rate_at_t':   grid_obs['I_input_rate'][k],
    }


def _imex_step_fn(frozen):
    def imex_step_fn(y, t, dt, params, grid_obs):
        aux_k = _make_aux_at_step(grid_obs, jnp.int32(t / dt))
        return dyn.imex_step_deterministic(y, t, dt, params, frozen, aux_k, PI)
    return imex_step_fn


def _obs_log_prob_fn(frozen):
    def obs_log_prob_fn(y, grid_obs, k, params):
        return (dyn.cgm_log_prob(y, grid_obs, k, params, PI)
                + dyn.carb_log_prob(grid_obs, k))
    return obs_log_prob_fn


def _obs_log_weight_fn(frozen):
    """PF reweight for non-guided channel (carbs only — CGM is in propagate_fn)."""
    def obs_log_weight_fn(x_new, grid_obs, k, params):
        return dyn.carb_log_prob(grid_obs, k)
    return obs_log_weight_fn


def _gaussian_obs_fn(frozen):
    def gaussian_obs_fn(y, grid_obs, k, params):
        return {
            'mean':     jnp.array([y[0]]),    # CGM observes G directly
            'value':    jnp.array([grid_obs['cgm_value'][k]]),
            'cov_diag': jnp.array([params[PI['sigma_cgm']] ** 2]),
            'present':  jnp.array([grid_obs['cgm_present'][k]]),
        }
    return gaussian_obs_fn


def _propagate_fn(frozen):
    """Pitt-Shephard guided proposal via CGM Gaussian channel.

    Mirrors the SWAT pattern (HR-tilt on W). CGM observes G directly with
    σ_CGM ~ 8 mg/dL → Kalman update on G (state index 0) at every 5-min bin
    keeps particles tied to truth. Returns the predictive log-weight, the
    standard Pitt-Shephard correction.
    """
    def propagate_fn(y, t, dt, params, grid_obs, k, sigma_diag, noise, rng_key):
        del rng_key

        aux_k = _make_aux_at_step(grid_obs, k)

        # Stochastic Euler step: deterministic mean + diffusion noise.
        y_next, mu_prior, var_prior = dyn.imex_step_stochastic(
            y, t, dt, params, sigma_diag, noise, frozen, aux_k, PI)

        # Pitt-Shephard guidance on G via CGM (Kalman update, precision form).
        sigma_cgm = params[PI['sigma_cgm']]
        cgm_pres = grid_obs['cgm_present'][k]
        cgm_val = grid_obs['cgm_value'][k]

        G_prec = 1.0 / jnp.maximum(var_prior[0], 1e-12)
        G_info = G_prec * mu_prior[0]
        G_info += cgm_pres * cgm_val / (sigma_cgm ** 2)
        G_prec += cgm_pres / (sigma_cgm ** 2)
        G_var = 1.0 / G_prec
        G_mu = G_var * G_info

        # Sample G from guided posterior; clip to physiological range.
        G_new = jnp.clip(G_mu + jnp.sqrt(G_var) * noise[0], 0.0, 600.0)
        y_next = y_next.at[0].set(G_new)
        # X, I retain their stochastic Euler values (clipped non-negative).
        y_next = y_next.at[1].set(jnp.maximum(y_next[1], 0.0))
        y_next = y_next.at[2].set(jnp.maximum(y_next[2], 0.0))

        # Predictive log-weight for CGM (mirror SWAT's HR predictive).
        pred_var_cgm = sigma_cgm ** 2 + var_prior[0]
        pred_mu_cgm = mu_prior[0]
        lw = cgm_pres * (
            -0.5 * (cgm_val - pred_mu_cgm) ** 2 / pred_var_cgm
            - 0.5 * jnp.log(pred_var_cgm) - HALF_LOG_2PI)
        return y_next, lw
    return propagate_fn


def _make_init_state_fn(frozen):
    def make_init_state_fn(init_estimates, params):
        del params
        G_0 = init_estimates[0]
        I_0 = init_estimates[1]
        X_0 = jnp.float64(0.0)
        return jnp.array([G_0, X_0, I_0])
    return make_init_state_fn


def _shard_init_fn(frozen):
    def shard_init_fn(time_offset, params, exogenous, global_init):
        del time_offset, params, exogenous
        return jnp.asarray(global_init, dtype=jnp.float64)
    return shard_init_fn


def _forward_sde_fn(frozen, meal_schedule, insulin_schedule, V_G, BW):
    """JAX forward Euler-Maruyama trajectory (for posterior MAP plots)."""
    def forward_sde_fn(init_state, params, exogenous, dt, n_steps, rng_key=None):
        del exogenous
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        sigma_d = diffusion_fn(params, frozen)
        y0 = jnp.asarray(init_state, dtype=jnp.float64)

        # Pre-compute D_rate / I_rate arrays for the n_steps grid.
        D_rate_np, I_rate_np = _build_rate_arrays(
            meal_schedule, insulin_schedule, n_steps, float(dt), V_G, BW)
        D_rate = jnp.asarray(D_rate_np, dtype=jnp.float64)
        I_rate = jnp.asarray(I_rate_np, dtype=jnp.float64)

        def step(carry, i):
            y, key = carry
            key, nk = jax.random.split(key)
            noise = jax.random.normal(nk, (3,))
            t = i * dt
            aux_k = {'D_rate_at_t': D_rate[i],
                     'I_input_rate_at_t': I_rate[i]}
            y_next, _, _ = dyn.imex_step_stochastic(
                y, t, dt, params, sigma_d, noise, frozen, aux_k, PI)
            y_next = y_next.at[0].set(jnp.clip(y_next[0], 0.0, 600.0))
            y_next = y_next.at[1].set(jnp.maximum(y_next[1], 0.0))
            y_next = y_next.at[2].set(jnp.maximum(y_next[2], 0.0))
            return (y_next, key), y_next

        (_, _), traj = jax.lax.scan(step, (y0, rng_key), jnp.arange(n_steps))
        return traj
    return forward_sde_fn


def _obs_sample_fn(frozen):
    def obs_sample_fn(y, exog, k, params, rng_key):
        del exog, k
        sigma_cgm = params[PI['sigma_cgm']]
        cgm = y[0] + sigma_cgm * jax.random.normal(rng_key, dtype=y.dtype)
        return {'cgm_value': cgm}
    return obs_sample_fn


# =========================================================================
# Prior-mean initialiser
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
    all_cfg = OrderedDict()
    all_cfg.update(PARAM_PRIOR_CONFIG)
    all_cfg.update(INIT_STATE_PRIOR_CONFIG)
    means = [_prior_mean(pt, pa) for _, (pt, pa) in all_cfg.items()]
    return np.array(means, dtype=np.float32)


# =========================================================================
# EstimationModel factory + canonical instance
# =========================================================================

def make_glucose_insulin_estimation(
    meal_schedule=None,
    insulin_schedule=None,
    frozen_params: dict | None = None,
    param_prior_overrides: dict | None = None,
) -> EstimationModel:
    """Build an EstimationModel for a specific scenario.

    Args:
      meal_schedule: list[(t_hr, carbs_g)] of meal events for the trial.
        None or empty for fasting trials.
      insulin_schedule: dict {boluses, basal_rate_U_hr, V_I_BW} or None.
      frozen_params: override DEFAULT_FROZEN_PARAMS (e.g., switch
        ``Ib=0`` and ``n_beta=0`` for T1D scenarios).
      param_prior_overrides: per-set prior overrides (Sets B/C/D).

    Defaults to Set A (healthy adult, Bergman 1979 cohort).
    """
    frozen = dict(DEFAULT_FROZEN_PARAMS)
    if frozen_params:
        frozen.update(frozen_params)

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

    V_G = frozen['V_G']
    BW = frozen['BW']
    align_obs_fn = _make_align_obs_fn(meal_schedule, insulin_schedule, V_G, BW)

    return EstimationModel(
        name="glucose_insulin",
        version="1.0",
        n_states=3,
        n_stochastic=3,
        stochastic_indices=(0, 1, 2),
        state_bounds=(
            (0.0, 600.0),    # G mg/dL
            (0.0,   2.0),    # X 1/hr
            (0.0, 500.0),    # I μU/mL
        ),
        param_prior_config=param_cfg,
        init_state_prior_config=init_cfg,
        frozen_params=frozen,
        propagate_fn=_propagate_fn(frozen),
        diffusion_fn=lambda p: diffusion_fn(p, frozen),
        obs_log_weight_fn=_obs_log_weight_fn(frozen),
        align_obs_fn=align_obs_fn,
        shard_init_fn=_shard_init_fn(frozen),
        forward_sde_fn=_forward_sde_fn(
            frozen, meal_schedule, insulin_schedule, V_G, BW),
        get_init_theta_fn=get_init_theta,
        imex_step_fn=_imex_step_fn(frozen),
        obs_log_prob_fn=_obs_log_prob_fn(frozen),
        make_init_state_fn=_make_init_state_fn(frozen),
        obs_sample_fn=_obs_sample_fn(frozen),
        gaussian_obs_fn=_gaussian_obs_fn(frozen),
        exogenous_keys=(),
    )


# Default canonical instance (Set A: healthy adult; meal schedule built
# lazily for one canonical day so the import-time cost is tiny). The SMC²
# driver re-builds with a scenario-specific schedule via the factory.
SIR_DEFAULT_MEAL_SCHEDULE = [(8.0, 40.0), (13.0, 40.0), (19.0, 40.0)]

GLUCOSE_INSULIN_ESTIMATION = make_glucose_insulin_estimation(
    meal_schedule=SIR_DEFAULT_MEAL_SCHEDULE,
    insulin_schedule=None,
)

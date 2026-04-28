"""models/glucose_insulin/_dynamics.py — JAX dynamics for the Bergman model.

Pure-JAX functions implementing the Bergman 1981 extended-minimal-model
SDE drift / diffusion / IMEX step / per-channel observation log-probs.
Shared between simulation.py (numpy) and estimation.py (JAX) — the
drift parity test ensures the two agree to machine precision.

State vector layout:
    y[0] = G    (plasma glucose, mg/dL,    stochastic)
    y[1] = X    (remote insulin action, 1/hr, stochastic)
    y[2] = I    (plasma insulin, μU/mL,   stochastic)

The ``pi`` dict maps parameter names to positions in the parameter vector;
built once in ``estimation.py``. ``frozen`` is a small dict of constants
(``Ib``, ``V_G``, ``V_I``, ``BW``, ``T_X``, ``T_I``, ``meal_carbs_g``,
``insulin_schedule_active``, etc.) that are not estimated and pass through
unchanged. ``aux`` carries the meal/insulin schedules for the trial.

Math (frequency-dependent, /hour units):

    dG/dt = -p₁(G - Gb) - X·G + D(t) / (V_G·BW)
    dX/dt = -p₂·X + p₃·max(I - Ib, 0)
    dI/dt = -k(I - Ib) + n_β·max(G - h_β, 0) + I_input(t) / (V_I·BW·100)

Channel means:
    cgm_mean(y) = G                                 (CGM Gaussian)
    carb_mean(y, aux, t) = meal_carbs_truth(t, aux) (Poisson)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from _likelihood_constants import HALF_LOG_2PI


_STATE_NAMES = ('G', 'X', 'I')


# =========================================================================
# DRIFT
# =========================================================================

def drift_jax(y, t, params, frozen, aux, pi):
    """Bergman 1981 extended-minimal-model drift.

    Signature matches the canonical SWAT pattern ``(y, t, params, ...)``;
    the additional ``aux`` arg carries the meal / insulin schedules
    pre-built by ``simulation.make_aux``.
    """
    G, X, I = y[0], y[1], y[2]
    Gb = params[pi['Gb']]
    p1 = params[pi['p1']]
    p2 = params[pi['p2']]
    p3 = params[pi['p3']]
    k = params[pi['k']]

    Ib = frozen['Ib']
    n_beta = frozen['n_beta']
    h_beta = frozen['h_beta']

    D_rate = aux['D_rate_at_t']        # mg/dL/hr (precomputed at scan time)
    I_rate = aux['I_input_rate_at_t']  # μU/mL/hr

    secretion = n_beta * jnp.maximum(G - h_beta, 0.0)
    dG = -p1 * (G - Gb) - X * G + D_rate
    dX = -p2 * X + p3 * jnp.maximum(I - Ib, 0.0)
    dI = -k * (I - Ib) + secretion + I_rate
    return jnp.array([dG, dX, dI])


# =========================================================================
# DIFFUSION
# =========================================================================

def diffusion(params, frozen, pi):
    """Diagonal diffusion magnitudes √T_G, √T_X, √T_I."""
    return jnp.array([
        jnp.sqrt(params[pi['T_G']]),
        jnp.sqrt(frozen['T_X']),
        jnp.sqrt(frozen['T_I']),
    ])


# =========================================================================
# IMEX STEPS
# =========================================================================

def imex_step_deterministic(y, t, dt, params, frozen, aux, pi):
    """Explicit Euler step on the drift."""
    return y + dt * drift_jax(y, t, params, frozen, aux, pi)


def imex_step_stochastic(y, t, dt, params, sigma_diag, noise, frozen, aux, pi):
    """Euler-Maruyama step. Returns (y_next, mu_prior, var_prior)."""
    y_det = imex_step_deterministic(y, t, dt, params, frozen, aux, pi)
    y_next = y_det + sigma_diag * jnp.sqrt(dt) * noise
    mu_prior = y_det
    var_prior = (sigma_diag ** 2) * dt
    return y_next, mu_prior, var_prior


# =========================================================================
# OBSERVATION-CHANNEL LOG-PROBABILITIES
# =========================================================================

def cgm_log_prob(y, grid_obs, k, params, pi):
    """Gaussian log-pdf for CGM observation at step k (5-min cadence)."""
    sigma_cgm = params[pi['sigma_cgm']]
    G_pred = y[0]
    resid = grid_obs['cgm_value'][k] - G_pred
    return grid_obs['cgm_present'][k] * (
        -0.5 * (resid / sigma_cgm) ** 2 - jnp.log(sigma_cgm) - HALF_LOG_2PI)


def carb_log_prob(grid_obs, k):
    """Poisson log-pmf for meal carb-count observation at step k.

    The Poisson rate equals the *truth* meal carb count at the meal time
    (encoded in ``grid_obs['carb_truth']``). The observation is the patient-
    logged count which is itself a noisy Poisson realisation. Since the
    truth carb count is part of the SCENARIO INPUT (not an estimable
    parameter), this channel's likelihood is independent of the latent
    state — it informs the meal-schedule estimation only if we extend the
    model later.
    """
    # Use carb_truth as the Poisson rate (deterministic given meal schedule).
    rate = grid_obs['carb_truth'][k]
    n = grid_obs['carbs_g'][k].astype(rate.dtype)
    safe_rate = jnp.maximum(rate, 1e-12)
    log_pmf = (n * jnp.log(safe_rate) - safe_rate - jax.lax.lgamma(n + 1.0))
    return grid_obs['carbs_present'][k] * log_pmf

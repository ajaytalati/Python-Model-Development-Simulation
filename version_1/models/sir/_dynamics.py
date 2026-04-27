"""models/sir/_dynamics.py — JAX dynamics helpers for SIR.

Pure-JAX functions implementing the SIR SDE drift/diffusion, the
2-state Euler-Maruyama step, and the per-channel observation
log-probabilities.

State vector layout:
    y[0] = S    (susceptible, stochastic)
    y[1] = I    (infected,    stochastic)

R = N - S - I is eliminated; conserved by construction (mass conservation
modulo SDE diffusion noise).

The ``pi`` dict maps parameter names to positions in the parameter vector;
built once in ``estimation.py`` and passed through to every helper here.
``frozen`` is a small dict of constants (``N``, ``v``) that are not
estimated and are passed through unchanged.

Math:

    dS = (-β S I / N - v S) dt + √T_S dW_1
    dI = ( β S I / N - γ I) dt + √T_I dW_2

    cases_rate(y, params)    = ρ β S I / N            (cases per hour)
    serology_mean(y, frozen) = I / N                   (prevalence)

References:
    Anderson & May 1991, "Infectious Diseases of Humans" §6
    Endo, van Leeuwen, Baguelin 2019, Epidemics 29 (PMCMC tutorial benchmark)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from _likelihood_constants import HALF_LOG_2PI


_STATE_NAMES = ('S', 'I')


# =========================================================================
# DRIFT
# =========================================================================

def drift_jax(y, params, frozen, pi):
    """Frequency-dependent SIR drift.

    With optional vaccination control input ``frozen['v']`` (∈ ℝ⁺,
    units 1/hr); zero in baseline scenarios.
    """
    S, I = y[0], y[1]
    N = frozen['N']
    v = frozen['v']
    beta = params[pi['beta']]
    gamma = params[pi['gamma']]

    incidence = beta * S * I / N
    dS = -incidence - v * S
    dI =  incidence - gamma * I
    return jnp.array([dS, dI])


# =========================================================================
# DIFFUSION  (per-state diagonal, constant in y)
# =========================================================================

def diffusion(params, pi):
    """Diagonal diffusion magnitudes √T_S, √T_I."""
    return jnp.array([
        jnp.sqrt(params[pi['T_S']]),
        jnp.sqrt(params[pi['T_I']]),
    ])


# =========================================================================
# IMEX STEPS  (explicit Euler — drift is non-stiff at sane rates)
# =========================================================================

def imex_step_deterministic(y, t, dt, params, frozen, pi):
    """One explicit Euler step on the drift.

    For SIR with hourly dt, explicit Euler is stable up to R_0 ≈ 8-10. Higher
    R_0 or smaller-dt regimes would warrant a semi-implicit splitting.
    """
    del t
    return y + dt * drift_jax(y, params, frozen, pi)


def imex_step_stochastic(y, t, dt, params, sigma_diag, noise, frozen, pi):
    """One Euler-Maruyama step. Returns (y_next, mu_prior, var_prior).

    mu_prior / var_prior describe the predictive Gaussian over y_next under the
    explicit-Euler approximation; provided for guided proposals (unused in SIR's
    bootstrap PF, but required by the framework contract).
    """
    y_det = imex_step_deterministic(y, t, dt, params, frozen, pi)
    y_next = y_det + sigma_diag * jnp.sqrt(dt) * noise
    mu_prior = y_det
    var_prior = (sigma_diag ** 2) * dt
    return y_next, mu_prior, var_prior


# =========================================================================
# OBSERVATION-CHANNEL MEANS
# =========================================================================

def cases_rate(y, params, frozen, pi):
    """Expected case-detection rate per hour: ρ β S I / N."""
    S, I = y[0], y[1]
    N = frozen['N']
    return params[pi['rho']] * params[pi['beta']] * S * I / N


def serology_mean(y, frozen):
    """Mean serology = prevalence I / N."""
    return y[1] / frozen['N']


# =========================================================================
# OBSERVATION-CHANNEL LOG-PROBABILITIES
# =========================================================================

def cases_log_prob(y, grid_obs, k, params, frozen, pi):
    """Poisson log-pmf for daily case-count observation at step k.

    Expected count = rate × bin_hours where bin_hours = 24 (daily aggregation).
    Uses ``lgamma(n+1)`` for the log-factorial term to support fractional
    n in tempered settings (downstream tempering can pass non-integer n).
    """
    rate = cases_rate(y, params, frozen, pi)
    expected = rate * grid_obs['cases_bin_hours']
    n = grid_obs['cases_count'][k].astype(expected.dtype)
    log_pmf = (n * jnp.log(jnp.maximum(expected, 1e-12))
               - expected
               - jax.lax.lgamma(n + 1.0))
    return grid_obs['cases_present'][k] * log_pmf


def serology_log_prob(y, grid_obs, k, params, frozen, pi):
    """Gaussian log-pdf for serology survey observation at step k.

    Survey is sparse (weekly); ``serology_present[k]`` masks non-survey steps.
    """
    sigma_z = params[pi['sigma_z']]
    mean = serology_mean(y, frozen)
    resid = grid_obs['serology_value'][k] - mean
    return grid_obs['serology_present'][k] * (
        -0.5 * (resid / sigma_z) ** 2 - jnp.log(sigma_z) - HALF_LOG_2PI)

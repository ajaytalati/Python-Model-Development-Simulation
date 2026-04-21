"""models/swat/_dynamics.py — JAX dynamics helpers for SWAT.

Date:    20 April 2026
Version: 1.0

Pure JAX functions implementing the SWAT SDE drift/diffusion and the
7-state IMEX step.  Extends models.sleep_wake_20p._dynamics by:

  - adding the testosterone state T as state index 3;
  - adding entrainment quality E(t) computation (instantaneous-DC);
  - adding alpha_T * T loading into u_W;
  - adding the Stuart-Landau drift (mu(E) T - eta T^3) / tau_T for state T;
  - shifting C, Vh, Vn to indices 4, 5, 6.

State vector layout:
    y[0] = W       (wakefulness, stochastic)
    y[1] = Zt      (sleep depth, stochastic)
    y[2] = a       (adenosine, stochastic)
    y[3] = T       (testosterone pulsatility, stochastic)
    y[4] = C       (circadian pacemaker, analytical-deterministic)
    y[5] = Vh      (vitality potential, constant in Phase 1)
    y[6] = Vn      (chronic-load potential, constant in Phase 1)

The param_index dict pi maps parameter names to positions in the
parameter vector; built once in estimation.py.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import Array
from typing import Dict, Tuple


A_SCALE = 6.0      # fixed scale constant for tilde-Z (same as 20p)

# Frozen morning-type circadian baseline (matches simulation.py).  The
# chronotype phi is NOT a parameter; V_c (parameter) shifts the subject's
# effective drive by V_c hours relative to this baseline.
import math as _math
PHI_MORNING_TYPE = -_math.pi / 3.0

_STATE_NAMES = ('W', 'Zt', 'a', 'T', 'C', 'Vh', 'Vn')


# =========================================================================
# ENTRAINMENT QUALITY  (amplitude × phase, V_c-aware)
# =========================================================================
#
# E combines two clinically distinct failure modes:
#   (1) amplitude failure — slow pressures so imbalanced that the sigmoid
#       is saturated and the sleep/wake flip-flop cannot alternate cleanly.
#       Captured by the two 4*sigma(mu)*(1-sigma(mu)) factors.
#   (2) phase-alignment failure — subject's rhythm shifted relative to the
#       external light cycle (shift worker, jet lag).  Captured by the
#       max(C_ext * C_eff, 0) factor: it swings daily but its TIME AVERAGE
#       is cos(2π V_c / 24)_+ which is 1 at V_c=0, 0 at V_c=±6h, and
#       negative (clipped to 0) beyond that.
#
# Multiplying the two yields the final E in [0, 1].  T evolves on
# tau_T ≈ 48 h so it naturally low-passes the daily ripple from the phase
# factor; the mean behaviour is what drives the bifurcation.
#
# =========================================================================

def entrainment_quality(y: Array, params: Array,
                        pi: Dict[str, int]) -> Array:
    """Entrainment quality E(t) in [0, 1] — V_c-aware.

    Combines amplitude quality (slow-backdrop balance) with phase
    alignment (subject's V_c shift vs external light cycle).  Phase
    quality depends on V_c ONLY, not on t — no daily ripple.
    """
    a, T = y[2], y[3]
    Vh, Vn = y[5], y[6]

    beta_Z  = params[pi['beta_Z']]
    alpha_T = params[pi['alpha_T']]
    V_c     = params[pi['V_c']]

    # --- amplitude quality: can the flip-flop engage at all? ---
    mu_W_slow = Vh + Vn - a + alpha_T * T
    mu_Z_slow = -Vn + beta_Z * a
    sW = jax.nn.sigmoid(mu_W_slow)
    sZ = jax.nn.sigmoid(mu_Z_slow)
    E_W = 4.0 * sW * (1.0 - sW)
    E_Z = 4.0 * sZ * (1.0 - sZ)
    amp_quality = E_W * E_Z

    # --- phase alignment: is subject's rhythm in sync with external light? ---
    # Function of V_c ONLY (no daily ripple).  max(cos(2π V_c/24), 0):
    #   V_c = 0h  -> 1.0 (aligned)
    #   V_c = ±3h -> 0.707
    #   V_c = ±6h -> 0.0  (shift worker)
    #   V_c = ±12h -> 0.0 (full inversion; clipped by max)
    V_c_rad = 2.0 * jnp.pi * V_c / 24.0
    phase_quality = jnp.maximum(jnp.cos(V_c_rad), 0.0)

    return amp_quality * phase_quality


# =========================================================================
# SIGMOID ARGUMENTS (include circadian + alpha_T * T)
# =========================================================================

def compute_sigmoid_args(y: Array, t: Array, params: Array,
                         pi: Dict[str, int]) -> Tuple[Array, Array]:
    """Compute u_W, u_Z at state y, time t.

    u_W uses the subject's SHIFTED circadian drive C_eff (shifted by V_c
    hours from the external morning-type baseline).  u_Z has no direct
    circadian term (unchanged from 20p); it inherits the shift through
    the -gamma_3 * W feedback.
    """
    W, Zt, a, T = y[0], y[1], y[2], y[3]
    Vh, Vn = y[5], y[6]

    kappa   = params[pi['kappa']]
    lmbda   = params[pi['lmbda']]
    gamma_3 = params[pi['gamma_3']]
    beta_Z  = params[pi['beta_Z']]
    alpha_T = params[pi['alpha_T']]
    V_c     = params[pi['V_c']]

    # Subject's shifted circadian drive
    C_eff = jnp.sin(2.0 * jnp.pi * (t - V_c) / 24.0 + PHI_MORNING_TYPE)

    u_W = -kappa * Zt + lmbda * C_eff + Vh + Vn - a + alpha_T * T
    u_Z = -gamma_3 * W - Vn + beta_Z * a
    return u_W, u_Z


# =========================================================================
# DRIFT (stochastic + deterministic components)
# =========================================================================

def drift(y: Array, t: Array, params: Array,
          pi: Dict[str, int]) -> Array:
    """Drift vector for the SWAT SDE (7D; 4 stochastic components)."""
    W, Zt, a, T = y[0], y[1], y[2], y[3]
    u_W, u_Z = compute_sigmoid_args(y, t, params, pi)

    tau_W = params[pi['tau_W']]
    tau_Z = params[pi['tau_Z']]
    tau_a = params[pi['tau_a']]
    tau_T = params[pi['tau_T']]
    mu_0  = params[pi['mu_0']]
    mu_E  = params[pi['mu_E']]
    eta   = params[pi['eta']]

    E = entrainment_quality(y, params, pi)
    mu_bifurc = mu_0 + mu_E * E

    dW  = (jax.nn.sigmoid(u_W) - W) / tau_W
    dZt = (A_SCALE * jax.nn.sigmoid(u_Z) - Zt) / tau_Z
    da  = (W - a) / tau_a
    dT  = (mu_bifurc * T - eta * T ** 3) / tau_T

    return jnp.array([dW, dZt, da, dT, 0.0, 0.0, 0.0])


# =========================================================================
# DIFFUSION
# =========================================================================

def diffusion(params: Array, pi: Dict[str, int]) -> Array:
    """Diagonal diffusion coefficients sqrt(2 * T_i)  —  shape (7,)."""
    return jnp.array([
        jnp.sqrt(2.0 * params[pi['T_W']]),
        jnp.sqrt(2.0 * params[pi['T_Z']]),
        jnp.sqrt(2.0 * params[pi['T_a']]),
        jnp.sqrt(2.0 * params[pi['T_T']]),
        0.0,  # C
        0.0,  # Vh
        0.0,  # Vn
    ])


# =========================================================================
# IMEX SPLIT
# =========================================================================
#
# The 20p IMEX split exploited the fact that each stochastic state's drift
# has the form (forcing - decay * y).  For the T-SDE this is not exactly
# true: dT/dt = (mu T - eta T^3) / tau_T has no constant forcing and a
# state-dependent decay ((eta T^2 - mu)/tau_T if we interpret mu > 0 as a
# negative effective decay at small T).  A clean split:
#
#   If mu(E) <= 0:   forcing = 0, decay = (|mu| + eta T^2) / tau_T >= 0
#   If mu(E) >  0:   forcing = mu T / tau_T, decay = eta T^2 / tau_T
#
# We use the second form unconditionally — this makes the decay term
# non-negative (eta, T^2 >= 0) and absorbs the possibly-signed mu*T into
# the explicit forcing.  The net effect on the ODE is identical.
#
# =========================================================================

def imex_components(y: Array, t: Array, params: Array,
                    pi: Dict[str, int]) -> Tuple[Array, Array]:
    """IMEX forcing/decay for the four stochastic states.

    For dW = (s_W - W)/tau_W, 20p-style split: forcing = s_W/tau_W,
    decay = 1/tau_W.

    For dT = (mu T - eta T^3)/tau_T, the split is:
      forcing = mu * T / tau_T             (positive if mu > 0, can be negative)
      decay   = eta * T^2 / tau_T           (always >= 0)

    C, Vh, Vn contribute zero.
    """
    u_W, u_Z = compute_sigmoid_args(y, t, params, pi)
    W, T_amp = y[0], y[3]

    tau_W = params[pi['tau_W']]
    tau_Z = params[pi['tau_Z']]
    tau_a = params[pi['tau_a']]
    tau_T = params[pi['tau_T']]
    mu_0  = params[pi['mu_0']]
    mu_E  = params[pi['mu_E']]
    eta   = params[pi['eta']]

    E = entrainment_quality(y, params, pi)
    mu_bifurc = mu_0 + mu_E * E

    fW  = jax.nn.sigmoid(u_W) / tau_W
    fZt = A_SCALE * jax.nn.sigmoid(u_Z) / tau_Z
    fa  = W / tau_a
    fT  = mu_bifurc * T_amp / tau_T     # explicit forcing (signed)

    dW_decay  = 1.0 / tau_W
    dZt_decay = 1.0 / tau_Z
    da_decay  = 1.0 / tau_a
    dT_decay  = eta * T_amp ** 2 / tau_T  # implicit decay (>= 0)

    forcing = jnp.array([fW, fZt, fa, fT, 0.0, 0.0, 0.0])
    decay   = jnp.array([dW_decay, dZt_decay, da_decay, dT_decay,
                         0.0, 0.0, 0.0])
    return forcing, decay


def imex_step_deterministic(y: Array, t: Array, dt: Array, params: Array,
                            pi: Dict[str, int]) -> Array:
    """One deterministic IMEX step (no noise).  Used by the EKF."""
    forcing, decay = imex_components(y, t, params, pi)
    y_next = (y + dt * forcing) / (1.0 + dt * decay)
    # Reset deterministic C state to the EXTERNAL light cycle
    y_next = y_next.at[4].set(
        jnp.sin(2.0 * jnp.pi * (t + dt) / 24.0 + PHI_MORNING_TYPE))
    y_next = y_next.at[5].set(y[5])  # Vh constant
    y_next = y_next.at[6].set(y[6])  # Vn constant
    # Positivity clip on T
    y_next = y_next.at[3].set(jnp.maximum(y_next[3], 0.0))
    return y_next


def imex_step_stochastic(y: Array, t: Array, dt: Array, params: Array,
                         sigma_diag: Array, noise: Array,
                         pi: Dict[str, int]) -> Tuple[Array, Array, Array]:
    """One stochastic IMEX step — used inside the PF scan body.

    Returns:
        (y_next, mu_prior, var_prior).  mu_prior, var_prior per-state
        used by the guided proposal that conditions W on HR.
    """
    forcing, decay = imex_components(y, t, params, pi)
    denom = 1.0 + dt * decay
    mu_prior = (y + dt * forcing) / denom
    var_prior = (sigma_diag ** 2 * dt) / (denom ** 2)

    sqrt_dt = jnp.sqrt(dt)
    y_next = mu_prior + sigma_diag * sqrt_dt * noise / denom

    # Reset deterministic C state to the EXTERNAL light cycle
    y_next = y_next.at[4].set(
        jnp.sin(2.0 * jnp.pi * (t + dt) / 24.0 + PHI_MORNING_TYPE))
    y_next = y_next.at[5].set(y[5])
    y_next = y_next.at[6].set(y[6])
    # Positivity clip on T (reflecting-boundary approximation)
    y_next = y_next.at[3].set(jnp.maximum(y_next[3], 0.0))
    return y_next, mu_prior, var_prior


# =========================================================================
# OBSERVATION HELPERS (same as 20p — HR depends on W, sleep on Zt)
# =========================================================================

def hr_mean(y: Array, params: Array, pi: Dict[str, int]) -> Array:
    """Predicted HR given state y: HR_base + alpha_HR * W."""
    return params[pi['HR_base']] + params[pi['alpha_HR']] * y[0]


def sleep_prob(y: Array, params: Array, pi: Dict[str, int]) -> Array:
    """Prob(sleep = 1 | y) = sigmoid(Zt - c_tilde)."""
    return jax.nn.sigmoid(y[1] - params[pi['c_tilde']])
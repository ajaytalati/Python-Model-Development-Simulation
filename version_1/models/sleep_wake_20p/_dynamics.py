"""models/sleep_wake_20p/_dynamics.py — JAX dynamics helpers.

Date:    17 April 2026
Version: 1.0

Pure JAX functions implementing the 20-parameter sleep-wake-adenosine
SDE drift/diffusion and the 5-state IMEX step.  Kept separate from
estimation.py so each file stays under 200 lines.

The ``param_index`` dict passed to every function maps parameter names
to positions in the parameter vector; it is built once in estimation.py.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import Array
from typing import Dict, Tuple


A_SCALE = 6.0  # fixed scale constant (see simulation.py)

_STATE_NAMES = ('W', 'Zt', 'a', 'C', 'Vh', 'Vn')


def compute_sigmoid_args(y: Array, t: Array, params: Array,
                         pi: Dict[str, int]) -> Tuple[Array, Array]:
    """Compute the two sigmoid arguments u_W, u_Z at state y, time t.

    Args:
        y: state vector shape (6,): [W, Zt, a, C, Vh, Vn].
        t: scalar time in hours.
        params: parameter vector.
        pi: name -> index mapping for params.

    Returns:
        (u_W, u_Z) — scalars.
    """
    W, Zt, a = y[0], y[1], y[2]
    Vh, Vn = y[4], y[5]
    kappa = params[pi['kappa']]
    lmbda = params[pi['lmbda']]
    gamma_3 = params[pi['gamma_3']]
    beta_Z = params[pi['beta_Z']]
    phi = params[pi['phi']]
    C = jnp.sin(2.0 * jnp.pi * t / 24.0 + phi)
    u_W = -kappa * Zt + lmbda * C + Vh + Vn - a
    u_Z = -gamma_3 * W - Vn + beta_Z * a
    return u_W, u_Z


def drift(y: Array, t: Array, params: Array,
          pi: Dict[str, int]) -> Array:
    """Drift vector for the 20p SDE (6D, with 3 non-trivial components).

    States 3 (C), 4 (Vh), 5 (Vn) have zero drift here.  The circadian C
    is reset analytically after each IMEX step (see imex_step_fn_impl).
    V_h, V_n are constants in Phase 1.
    """
    W, Zt, a = y[0], y[1], y[2]
    u_W, u_Z = compute_sigmoid_args(y, t, params, pi)
    tau_W = params[pi['tau_W']]
    tau_Z = params[pi['tau_Z']]
    tau_a = params[pi['tau_a']]

    dW  = (jax.nn.sigmoid(u_W) - W) / tau_W
    dZt = (A_SCALE * jax.nn.sigmoid(u_Z) - Zt) / tau_Z
    da  = (W - a) / tau_a
    return jnp.array([dW, dZt, da, 0.0, 0.0, 0.0])


def diffusion(params: Array, pi: Dict[str, int]) -> Array:
    """Diagonal diffusion coefficients sqrt(2 * T_i)  —  shape (6,)."""
    return jnp.array([
        jnp.sqrt(2.0 * params[pi['T_W']]),
        jnp.sqrt(2.0 * params[pi['T_Z']]),
        jnp.sqrt(2.0 * params[pi['T_a']]),
        0.0,  # C
        0.0,  # Vh
        0.0,  # Vn
    ])


def imex_components(y: Array, t: Array, params: Array,
                    pi: Dict[str, int]) -> Tuple[Array, Array]:
    """IMEX split forcing/decay for the three stochastic states.

    For dW = (s_W - W)/tau_W etc, the IMEX split is:
        forcing = s_W / tau_W    (or equivalent for Zt, a)
        decay   = 1 / tau_W

    C, Vh, Vn contribute zero forcing and zero decay — circadian is
    re-computed analytically after the step.

    Returns:
        (forcing, decay) each shape (6,).
    """
    u_W, u_Z = compute_sigmoid_args(y, t, params, pi)
    tau_W = params[pi['tau_W']]
    tau_Z = params[pi['tau_Z']]
    tau_a = params[pi['tau_a']]
    W = y[0]

    fW  = jax.nn.sigmoid(u_W) / tau_W
    fZt = A_SCALE * jax.nn.sigmoid(u_Z) / tau_Z
    fa  = W / tau_a

    dW_decay  = 1.0 / tau_W
    dZt_decay = 1.0 / tau_Z
    da_decay  = 1.0 / tau_a

    forcing = jnp.array([fW, fZt, fa, 0.0, 0.0, 0.0])
    decay = jnp.array([dW_decay, dZt_decay, da_decay, 0.0, 0.0, 0.0])
    return forcing, decay


def imex_step_deterministic(y: Array, t: Array, dt: Array, params: Array,
                            pi: Dict[str, int]) -> Array:
    """One deterministic IMEX step (no noise).  Used by the EKF.

    Circadian C(t+dt) is reset analytically.  V_h, V_n are constants.
    """
    forcing, decay = imex_components(y, t, params, pi)
    y_next = (y + dt * forcing) / (1.0 + dt * decay)
    phi = params[pi['phi']]
    y_next = y_next.at[3].set(jnp.sin(2.0 * jnp.pi * (t + dt) / 24.0 + phi))
    y_next = y_next.at[4].set(y[4])  # Vh constant
    y_next = y_next.at[5].set(y[5])  # Vn constant
    return y_next


def imex_step_stochastic(y: Array, t: Array, dt: Array, params: Array,
                         sigma_diag: Array, noise: Array,
                         pi: Dict[str, int]) -> Tuple[Array, Array]:
    """One stochastic IMEX step — used inside the PF scan body.

    Returns:
        (y_next, mu_prior, var_prior).  mu_prior, var_prior are used by
        the guided proposal that conditions W on HR.

        mu_prior = (y + dt * forcing) / (1 + dt * decay)
        var_prior = sigma^2 dt / (1 + dt * decay)^2
    """
    forcing, decay = imex_components(y, t, params, pi)
    denom = 1.0 + dt * decay
    mu_prior = (y + dt * forcing) / denom
    var_prior = (sigma_diag ** 2 * dt) / (denom ** 2)

    sqrt_dt = jnp.sqrt(dt)
    y_next = mu_prior + sigma_diag * sqrt_dt * noise / denom
    # Reset deterministic components
    phi = params[pi['phi']]
    y_next = y_next.at[3].set(jnp.sin(2.0 * jnp.pi * (t + dt) / 24.0 + phi))
    y_next = y_next.at[4].set(y[4])
    y_next = y_next.at[5].set(y[5])
    return y_next, mu_prior, var_prior


def hr_mean(y: Array, params: Array, pi: Dict[str, int]) -> Array:
    """Predicted HR given state y: HR_base + alpha_HR * W."""
    return params[pi['HR_base']] + params[pi['alpha_HR']] * y[0]


def sleep_prob(y: Array, params: Array, pi: Dict[str, int]) -> Array:
    """Prob(sleep = 1 | y) = sigmoid(Zt - c_tilde)."""
    return jax.nn.sigmoid(y[1] - params[pi['c_tilde']])

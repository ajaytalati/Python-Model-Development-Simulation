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


A_SCALE = 6.0      # legacy scale constant; superseded by Zt ∈ [0, 1] rescale.
                   # Kept for backwards-compat for any consumer that imports
                   # A_SCALE directly. The drift no longer multiplies sigma(u_Z)
                   # by A_SCALE (Zt now saturates at 1, matching W and FSA-v2's
                   # B); diffusion uses Jacobi sqrt(Z(1-Z)). See PARAM_SET_A
                   # for the rescaled c_tilde / delta_c values.

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
    """Entrainment quality E(t) in [0, 1] — V_h-anabolic fix formula.

    Direct JAX port of the reference implementation in
    `swat_entrainment_docs/entrainment_model.py` (the "PR #11" V_h-
    anabolic structural fix). See `swat_entrainment_docs/01_formula.md`
    § "The full formula" for the end-to-end derivation.

        A_W   = lambda_amp_W · V_h            (band half-width on W)
        A_Z   = lambda_amp_Z · V_h            (band half-width on Z)
        B_W   = V_n − a + alpha_T · T          (band centre — NO V_h)
        B_Z   = -V_n + beta_Z · a              (band centre — NO V_h)
        amp_W = sigmoid(B_W + A_W) − sigmoid(B_W − A_W)
        amp_Z = sigmoid(B_Z + A_Z) − sigmoid(B_Z − A_Z)
        damp  = exp(-V_n / V_n_scale)
        phase = cos(π · min(|V_c|, V_c_max) / (2 · V_c_max))
        E_dyn = damp · amp_W · amp_Z · phase

    The formula depends ONLY on slow states (a, T) and controls
    (V_h, V_n, V_c) — NOT on instantaneous W, Zt, or C(t). So E_dyn
    is structurally non-oscillating within a day; a healthy patient
    sits sustainedly at ≈ 0.85 (V_n=0.3) or ≈ 0.98 (V_n=0).

    Sanity values (a=0.5, T=0.85; see swat_entrainment_docs/02_components.md):
        Healthy V_h=1, V_n=0.3, V_c=0:    E ≈ 0.8476
        V_h depleted V_h=0.2, V_n=0.3:    E ≈ 0.1747
        V_n high V_h=1, V_n=3.5:          E ≈ 0.1477
        Phase shift V_c=1h:               E ≈ 0.7340
        Phase shift V_c=2h:               E ≈ 0.4238
        Phase shift V_c≥3h (clamp):       E = 0.0000
    """
    a, T = y[2], y[3]
    Vh, Vn = y[5], y[6]

    alpha_T      = params[pi['alpha_T']]
    beta_Z       = params[pi['beta_Z']]
    lambda_amp_W = params[pi['lambda_amp_W']]
    lambda_amp_Z = params[pi['lambda_amp_Z']]
    V_n_scale    = params[pi['V_n_scale']]
    V_c_max      = params[pi['V_c_max']]
    V_c          = params[pi['V_c']]

    A_W = lambda_amp_W * Vh
    A_Z = lambda_amp_Z * Vh
    B_W = Vn - a + alpha_T * T
    B_Z = -Vn + beta_Z * a

    amp_W = jax.nn.sigmoid(B_W + A_W) - jax.nn.sigmoid(B_W - A_W)
    amp_Z = jax.nn.sigmoid(B_Z + A_Z) - jax.nn.sigmoid(B_Z - A_Z)
    damp  = jnp.exp(-Vn / V_n_scale)

    V_c_eff = jnp.minimum(jnp.abs(V_c), V_c_max)
    phase   = jnp.cos(jnp.pi * V_c_eff / (2.0 * V_c_max))

    return damp * amp_W * amp_Z * phase


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
    """Drift vector for the SWAT SDE (7D; 4 stochastic components).

    The C state's drift is the analytical d/dt of the external light
    cycle so that ``drift`` matches ``simulation.py:drift`` and
    ``simulation.py:drift_jax`` exactly. The IMEX step also explicitly
    resets C to its analytical value at each substep, making this
    redundant at integration time — but not at sim/est consistency-
    check time, where the drift values must agree per the §1.4
    discipline (Python-Model-Scenario-Simulation §1.4).
    """
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
    dZt = (jax.nn.sigmoid(u_Z) - Zt) / tau_Z   # Zt ∈ [0, 1]: saturates at 1
    da  = (W - a) / tau_a
    dT  = (mu_bifurc * T - eta * T ** 3) / tau_T

    # External light cycle (deterministic). Matches simulation.py.
    dC = (2.0 * jnp.pi / 24.0) * jnp.cos(2.0 * jnp.pi * t / 24.0
                                           + PHI_MORNING_TYPE)

    return jnp.array([dW, dZt, da, dT, dC, 0.0, 0.0])


# =========================================================================
# DIFFUSION
# =========================================================================

def diffusion(params: Array, pi: Dict[str, int]) -> Array:
    """Diagonal diffusion coefficients sqrt(2 * T_i)  —  shape (7,).

    These are the CONSTANT base coefficients sigma_i; the per-step
    increment is sigma_i * g_i(y) * sqrt(dt) * xi_i, where g_i is the
    state-dependent multiplier returned by ``noise_scale_jax`` (Jacobi
    for W, Zt, a; identity for T).
    """
    return jnp.array([
        jnp.sqrt(2.0 * params[pi['T_W']]),
        jnp.sqrt(2.0 * params[pi['T_Z']]),
        jnp.sqrt(2.0 * params[pi['T_a']]),
        jnp.sqrt(2.0 * params[pi['T_T']]),
        0.0,  # C
        0.0,  # Vh
        0.0,  # Vn
    ])


def noise_scale_jax(y: Array, params: Array) -> Array:
    """Diagonal Jacobi multipliers for the [0, 1]-bounded states.

    For Z, a (and W) ∈ [0, 1] use Jacobi: g(x) = sqrt(x*(1-x)) so the
    diffusion vanishes at the boundaries and the SDE stays in [0, 1]
    intrinsically. For T the noise is state-INDEPENDENT (g = 1) — the
    Stuart-Landau drift (mu T - eta T^3) vanishes at T=0, so additive
    Gaussian kicks are needed to escape the absorbing boundary at T=0.
    The deterministic states C, Vh, Vn are zeroed by ``diffusion`` so
    their multipliers are irrelevant; we set them to 1 for symmetry.
    """
    del params
    W, Zt, a = y[0], y[1], y[2]
    sqrt = jnp.sqrt
    return jnp.array([
        sqrt(jnp.maximum(W * (1.0 - W), 0.0)),
        sqrt(jnp.maximum(Zt * (1.0 - Zt), 0.0)),
        sqrt(jnp.maximum(a * (1.0 - a), 0.0)),
        1.0,    # T (state-INDEP additive noise)
        1.0,    # C  (no noise via diffusion=0)
        1.0,    # Vh (no noise via diffusion=0)
        1.0,    # Vn (no noise via diffusion=0)
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
    fZt = jax.nn.sigmoid(u_Z) / tau_Z   # Zt ∈ [0, 1]: saturates at 1
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
    # Bound clips on the now-[0,1] states W, Zt, a (Jacobi diffusion vanishes
    # at boundaries but Euler-Maruyama can drift slightly outside).
    y_next = y_next.at[0].set(jnp.clip(y_next[0], 0.0, 1.0))
    y_next = y_next.at[1].set(jnp.clip(y_next[1], 0.0, 1.0))
    y_next = y_next.at[2].set(jnp.clip(y_next[2], 0.0, 1.0))
    # Positivity clip on T (Stuart-Landau, no upper bound)
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
    # Bound clips on the [0, 1] states W, Zt, a + positivity clip on T.
    y_next = y_next.at[0].set(jnp.clip(y_next[0], 0.0, 1.0))
    y_next = y_next.at[1].set(jnp.clip(y_next[1], 0.0, 1.0))
    y_next = y_next.at[2].set(jnp.clip(y_next[2], 0.0, 1.0))
    y_next = y_next.at[3].set(jnp.maximum(y_next[3], 0.0))
    return y_next, mu_prior, var_prior


# =========================================================================
# OBSERVATION HELPERS (same as 20p — HR depends on W, sleep on Zt)
# =========================================================================

def hr_mean(y: Array, params: Array, pi: Dict[str, int]) -> Array:
    """Predicted HR given state y: HR_base + alpha_HR * W."""
    return params[pi['HR_base']] + params[pi['alpha_HR']] * y[0]


def sleep_prob(y: Array, params: Array, pi: Dict[str, int]) -> Array:
    """Prob(sleep ≥ 1 | y) = sigmoid(sleep_sharpness * (Zt - c_tilde)).

    Backward-compatible binary helper retained for callers that only
    need the wake-vs-sleep cut. For 3-level inference, use
    ``sleep_level_log_probs``.
    """
    sharp = params[pi['sleep_sharpness']] if 'sleep_sharpness' in pi else 1.0
    return jax.nn.sigmoid(sharp * (y[1] - params[pi['c_tilde']]))


def sleep_level_log_probs(y: Array, params: Array,
                           pi: Dict[str, int]) -> Array:
    """Log-probabilities of the 3-level ordinal sleep stages.

    Mirrors ``simulation.py:gen_sleep`` exactly:
        c1 = c_tilde,  c2 = c_tilde + delta_c,  k = sleep_sharpness
        s1 = sigmoid(k * (Zt - c1))         # P(sleep, any)
        s2 = sigmoid(k * (Zt - c2))         # P(deep)
        P(level=0 = wake)      = 1 - s1
        P(level=1 = light+REM) = s1 - s2
        P(level=2 = deep)      = s2

    Returns array of shape (3,) with log P(level=k | y).
    """
    Zt = y[1]
    c1 = params[pi['c_tilde']]
    c2 = c1 + params[pi['delta_c']]
    sharp = params[pi['sleep_sharpness']] if 'sleep_sharpness' in pi else 1.0
    s1 = jax.nn.sigmoid(sharp * (Zt - c1))
    s2 = jax.nn.sigmoid(sharp * (Zt - c2))

    p0 = 1.0 - s1
    p1 = s1 - s2
    p2 = s2
    safe = lambda p: jnp.clip(p, 1e-12, 1.0)
    return jnp.array([jnp.log(safe(p0)), jnp.log(safe(p1)), jnp.log(safe(p2))])


def steps_rate(y: Array, params: Array, pi: Dict[str, int]) -> Array:
    """Poisson rate (counts per hour) given state y.

    Mirrors ``simulation.py:gen_steps``:
        rate(W) = lambda_base + lambda_step * sigmoid(10 * (W - W_thresh))
    """
    W = y[0]
    return (params[pi['lambda_base']]
            + params[pi['lambda_step']]
            * jax.nn.sigmoid(10.0 * (W - params[pi['W_thresh']])))


def stress_mean(y: Array, params: Array, pi: Dict[str, int]) -> Array:
    """Predicted Garmin stress score given state y.

    Mirrors ``simulation.py:gen_stress``:
        mean = s_base + alpha_s * W + beta_s * Vn
    (no clip in the predictor — clipping is sim-side noise modelling).
    """
    W, Vn = y[0], y[6]
    return (params[pi['s_base']]
            + params[pi['alpha_s']] * W
            + params[pi['beta_s']] * Vn)

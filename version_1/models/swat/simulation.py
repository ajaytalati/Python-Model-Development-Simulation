"""
models/swat/simulation.py — 7-State Sleep-Wake-Adenosine-Testosterone SDE.
===========================================================================
Date:    20 April 2026
Version: 1.0

SWAT = Sleep-Wake-Adenosine-Testosterone model.

Minimal extension of the 20-parameter sleep-wake-adenosine SDE by adding a
single additional slow latent state T (testosterone pulsatility amplitude)
following the Stuart-Landau normal form.  See:
    Spec_24_Parameter_Sleep_Wake_Adenosine_Testosterone_Model.md
    Identifiability_and_Lyapunov_Proof_24_Parameter_Model.md

Latent SDE (7 states; C is deterministic; V_h, V_n are carried as constants):

    dW   = (1/tau_W) [sigma(u_W) - W] dt   + sqrt(2 T_W) dB_W
    dZt  = (1/tau_Z) [A sigma(u_Z) - Zt] dt+ sqrt(2 T_Z) dB_Z
    da   = (1/tau_a) (W - a) dt            + sqrt(2 T_a) dB_a
    dT   = (1/tau_T) [mu(E) T - eta T^3] dt+ sqrt(2 T_T) dB_T   [NEW]
    C    = sin(2 pi t / 24 + phi)          (deterministic)
    V_h, V_n: constants (Phase 1 — reverted from Phase-2 OU-process treatment)

with

    u_W = -kappa * Zt + lambda * C(t) + V_h + V_n - a + alpha_T * T   [MODIFIED]
    u_Z = -gamma_3 * W - V_n + beta_Z * a
    A   = 6  (fixed scale constant, NOT a parameter)

    mu(E) = mu_0 + mu_E * E                                          [NEW]

    E(t) = [4 * sigma(mu_W_slow) * (1 - sigma(mu_W_slow))] *         [NEW]
           [4 * sigma(mu_Z_slow) * (1 - sigma(mu_Z_slow))]

    mu_W_slow = V_h + V_n - a + alpha_T * T    (slow backdrop only)
    mu_Z_slow = -V_n + beta_Z * a              (slow backdrop only)

Observation channels:

    hr    ~ N(HR_base + alpha_HR * W, sigma_HR^2)       (continuous)
    sleep ~ Bernoulli(sigma(Zt - c_tilde))              (binary: 0=wake, 1=sleep)

    T is a LATENT state (no direct observation).  Identifiability flows
    via T -> u_W -> W -> HR.

Parameter count:
  - Code PARAM_PRIOR_CONFIG: 20 old (14 drift + 3 diffusion + Vh + Vn + ...)
    plus 5 new drift (mu_0, mu_E, eta, tau_T, alpha_T) plus 1 new diffusion
    (T_T) = 26.  Wait — old has 17 PARAM entries; +5 drift +1 diffusion = 23
    code parameters plus 4 ICs (W_0, Zt_0, a_0, T_0) = 27 estimable scalars.
  - Spec-level "24 parameters": 17 block + 5 T-block + 1 coupling + 1 T_T
    diffusion = 24, with T_W, T_Z, T_a frozen per the identifiability proof.

Notes on the cycle-averaged entrainment approximation:
  The spec's E(t) uses cycle-averaged DC offsets of the sigmoid arguments.
  For computational simplicity we use the INSTANTANEOUS non-circadian parts
  here (mu_W_dc drops the lambda*C term from u_W; mu_Z_dc equals u_Z since
  u_Z contains no circadian term).  Since T evolves on tau_T ~ 48h while the
  fast states oscillate at 24h, the T SDE naturally low-passes the residual
  24h fluctuations in E.  The approximation can be refined later by adding
  explicit running-average states if needed.
"""

import math
import numpy as np

from simulator.sde_model import (
    SDEModel, StateSpec, ChannelSpec,
    DIFFUSION_DIAGONAL_CONSTANT, DIFFUSION_DIAGONAL_STATE)
from models.swat.sim_plots import plot_swat


# =========================================================================
# CONSTANTS
# =========================================================================

A_SCALE = 6.0   # fixed rescaling constant for tilde-Z; NOT a parameter

# Frozen morning-type circadian phase: healthy wake peak at ~10am solar time.
# The chronotype parameter phi from the 17-param model has been REPLACED by
# the phase-shift parameter V_c (estimable + controllable).  Interpretation:
#   * External light cycle: C(t)     = sin(2*pi*t/24 + PHI_MORNING_TYPE)
#   * Subject's circadian drive: C_eff(t) = sin(2*pi*(t - V_c)/24 + PHI_MORNING_TYPE)
# V_c is in HOURS, positive = subject's rhythm shifted later than external light.
PHI_MORNING_TYPE = -math.pi / 3.0


# =========================================================================
# ELEMENTARY HELPERS
# =========================================================================

def _sigmoid(x):
    """Numerically stable logistic sigmoid (numpy)."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def circadian(t, params):
    """External light cycle C(t) — the OBJECTIVE sun/dark signal.

    Uses the frozen morning-type phase PHI_MORNING_TYPE.  This is the
    external reference; the subject's internal drive C_eff is shifted
    by V_c hours (see drift() below).
    """
    del params  # phi is no longer a parameter
    return math.sin(2.0 * math.pi * t / 24.0 + PHI_MORNING_TYPE)


def circadian_jax(t, params):
    """JAX variant of circadian() for the Diffrax solver."""
    import jax.numpy as jnp
    del params
    return jnp.sin(2.0 * jnp.pi * t / 24.0 + PHI_MORNING_TYPE)


def entrainment_quality(W, Zt, a, T, Vh, Vn, params):
    """Amplitude × phase entrainment quality E in [0, 1] — V_c-aware.

    Phase quality is a function of V_c ONLY (no daily ripple).  W, Zt
    retained for API compatibility but unused here.
    """
    del W, Zt
    p = params

    mu_W_slow = Vh + Vn - a + p['alpha_T'] * T
    mu_Z_slow = -Vn + p['beta_Z'] * a
    sW = float(_sigmoid(mu_W_slow))
    sZ = float(_sigmoid(mu_Z_slow))
    amp_quality = (4.0 * sW * (1.0 - sW)) * (4.0 * sZ * (1.0 - sZ))

    V_c_rad = 2.0 * math.pi * p['V_c'] / 24.0
    phase_quality = max(math.cos(V_c_rad), 0.0)

    return amp_quality * phase_quality


# =========================================================================
# DRIFT
# =========================================================================

def drift(t, y, params, aux):
    """7-state drift f(t, y, theta) for the scipy solver.

    y = [W, Zt, a, T, C, Vh, Vn].  The first four are stochastic;
    C, Vh, Vn are deterministic / constant carrying states.
    """
    del aux
    W, Zt, a, T, C, Vh, Vn = y[0], y[1], y[2], y[3], y[4], y[5], y[6]

    p = params
    kappa = p['kappa']; lam = p['lmbda']; gamma_3 = p['gamma_3']
    beta_Z = p['beta_Z']; alpha_T = p['alpha_T']
    tau_W = p['tau_W']; tau_Z = p['tau_Z']; tau_a = p['tau_a']
    tau_T = p['tau_T']
    mu_0 = p['mu_0']; mu_E = p['mu_E']; eta = p['eta']

    C_exact = circadian(t, p)          # external light cycle (state 4)

    # Subject's internal circadian drive: shifted by V_c hours
    V_c = p['V_c']
    C_eff = math.sin(2.0 * math.pi * (t - V_c) / 24.0 + PHI_MORNING_TYPE)

    # Full sigmoid arguments — u_W uses the SHIFTED drive; u_Z is driven only
    # through -gamma_3 * W (which inherits the shift via W's response).
    u_W = -kappa * Zt + lam * C_eff + Vh + Vn - a + alpha_T * T
    u_Z = -gamma_3 * W - Vn + beta_Z * a

    # Entrainment quality: amplitude × phase (V_c-aware).
    mu_W_slow = Vh + Vn - a + alpha_T * T
    mu_Z_slow = -Vn + beta_Z * a
    sW = float(_sigmoid(mu_W_slow))
    sZ = float(_sigmoid(mu_Z_slow))
    amp_quality = (4.0 * sW * (1.0 - sW)) * (4.0 * sZ * (1.0 - sZ))
    V_c_rad = 2.0 * math.pi * V_c / 24.0
    phase_quality = max(math.cos(V_c_rad), 0.0)
    E = amp_quality * phase_quality

    mu_bifurc = mu_0 + mu_E * E  # Stuart-Landau bifurcation parameter

    dW  = (float(_sigmoid(u_W)) - W) / tau_W
    dZt = (float(_sigmoid(u_Z)) - Zt) / tau_Z   # Zt ∈ [0, 1]: saturates at 1
    da  = (W - a) / tau_a
    dT  = (mu_bifurc * T - eta * T ** 3) / tau_T

    # C state tracks the EXTERNAL light cycle (objective reference)
    dC = (2.0 * math.pi / 24.0) * math.cos(2.0 * math.pi * t / 24.0
                                             + PHI_MORNING_TYPE)

    # V_h, V_n are constants -> zero drift
    dVh = 0.0
    dVn = 0.0

    return np.array([dW, dZt, da, dT, dC, dVh, dVn])


def drift_jax(t, y, args):
    """7-state drift in JAX (for the Diffrax solver).  Matches drift()."""
    import jax.numpy as jnp
    import jax
    (p,) = args
    W, Zt, a, T, C, Vh, Vn = y[0], y[1], y[2], y[3], y[4], y[5], y[6]

    # External light cycle (for the C state's drift) and subject's shifted
    # drive (for u_W).
    V_c = p['V_c']
    C_ex = jnp.sin(2.0 * jnp.pi * t / 24.0 + PHI_MORNING_TYPE)       # external
    C_eff = jnp.sin(2.0 * jnp.pi * (t - V_c) / 24.0 + PHI_MORNING_TYPE)  # subject's

    u_W = -p['kappa'] * Zt + p['lmbda'] * C_eff + Vh + Vn - a + p['alpha_T'] * T
    u_Z = -p['gamma_3'] * W - Vn + p['beta_Z'] * a

    # Entrainment quality: amplitude × phase (V_c-aware).
    mu_W_slow = Vh + Vn - a + p['alpha_T'] * T
    mu_Z_slow = -Vn + p['beta_Z'] * a
    sW = jax.nn.sigmoid(mu_W_slow)
    sZ = jax.nn.sigmoid(mu_Z_slow)
    amp_quality = (4.0 * sW * (1.0 - sW)) * (4.0 * sZ * (1.0 - sZ))
    V_c_rad = 2.0 * jnp.pi * V_c / 24.0
    phase_quality = jnp.maximum(jnp.cos(V_c_rad), 0.0)
    E = amp_quality * phase_quality

    mu_bifurc = p['mu_0'] + p['mu_E'] * E

    dW  = (jax.nn.sigmoid(u_W) - W) / p['tau_W']
    dZt = (jax.nn.sigmoid(u_Z) - Zt) / p['tau_Z']    # Zt ∈ [0, 1]
    da  = (W - a) / p['tau_a']
    dT  = (mu_bifurc * T - p['eta'] * T ** 3) / p['tau_T']
    # C state tracks the external light cycle
    dC = (2.0 * jnp.pi / 24.0) * jnp.cos(2.0 * jnp.pi * t / 24.0
                                           + PHI_MORNING_TYPE)

    return jnp.array([dW, dZt, da, dT, dC, 0.0, 0.0])


# =========================================================================
# DIFFUSION
# =========================================================================

def diffusion_diagonal(params):
    """Constant diffusion magnitudes sigma_i = sqrt(2 * T_i).

    The full per-step increment is sigma_i * g_i(y) * sqrt(dt) * xi_i,
    where g_i comes from ``noise_scale_fn`` (Jacobi for the [0, 1]
    states W, Zt, a; identity for T; arbitrary for the deterministic
    states C, Vh, Vn since their sigma_i is 0 anyway).

    States 4 (C), 5 (Vh), 6 (Vn) are deterministic -> sigma = 0.
    """
    p = params
    return np.array([
        math.sqrt(2.0 * p['T_W']),
        math.sqrt(2.0 * p['T_Z']),
        math.sqrt(2.0 * p['T_a']),
        math.sqrt(2.0 * p['T_T']),   # state-INDEPENDENT for T (Stuart-Landau
                                     # absorbing-boundary fix; see noise_scale_fn)
        0.0,   # C (deterministic)
        0.0,   # Vh (constant)
        0.0,   # Vn (constant)
    ])


def noise_scale_fn(y, params):
    """State-dependent diagonal noise multipliers (numpy).

    For W, Zt, a (all in [0, 1]) use Jacobi: g(x) = sqrt(x*(1-x)) so
    the SDE stays in [0, 1] intrinsically (diffusion vanishes at
    boundaries). For T use g = 1 (additive noise) — the Stuart-Landau
    drift (mu T - eta T^3) vanishes at T=0, so a state-dependent
    multiplier sqrt(T) would make T=0 absorbing. The deterministic
    states (C, Vh, Vn) have sigma=0 in diffusion_diagonal so their
    multiplier doesn't matter; we set 1.0 for symmetry.
    """
    del params
    W = max(0.0, min(1.0, float(y[0])))
    Zt = max(0.0, min(1.0, float(y[1])))
    a = max(0.0, min(1.0, float(y[2])))
    return np.array([
        math.sqrt(W * (1.0 - W)),
        math.sqrt(Zt * (1.0 - Zt)),
        math.sqrt(a * (1.0 - a)),
        1.0,   # T (state-INDEP; preserves additive Gaussian kicks at T=0)
        1.0,
        1.0,
        1.0,
    ])


def noise_scale_fn_jax(y, params):
    """JAX state-dependent noise multipliers — matches noise_scale_fn."""
    import jax.numpy as jnp
    del params
    W = jnp.clip(y[0], 0.0, 1.0)
    Zt = jnp.clip(y[1], 0.0, 1.0)
    a = jnp.clip(y[2], 0.0, 1.0)
    return jnp.array([
        jnp.sqrt(W * (1.0 - W)),
        jnp.sqrt(Zt * (1.0 - Zt)),
        jnp.sqrt(a * (1.0 - a)),
        1.0,   # T
        1.0,
        1.0,
        1.0,
    ])


# =========================================================================
# AUXILIARY / INITIAL STATE
# =========================================================================

def make_aux(params, init_state, t_grid, exogenous):
    """No auxiliary state needed."""
    del params, init_state, t_grid, exogenous
    return None


def make_aux_jax(params, init_state, t_grid, exogenous):
    """JAX auxiliary builder for the Diffrax solver."""
    import jax.numpy as jnp
    del init_state, t_grid, exogenous
    p_jax = {k: jnp.float64(v) for k, v in params.items()}
    return (p_jax,)


def make_y0(init_dict, params):
    """Build [W, Zt, a, T, C, Vh, Vn] at t=0 from init parameters.

    C state initial value is the EXTERNAL light cycle at t=0; V_c does not
    enter the stored C trajectory (that's the subject's internal drive,
    which enters u_W analytically inside drift()).
    """
    del params
    C0 = math.sin(PHI_MORNING_TYPE)
    return np.array([
        init_dict['W_0'],
        init_dict['Zt_0'],
        init_dict['a_0'],
        init_dict['T_0'],
        C0,
        init_dict['Vh'],
        init_dict['Vn'],
    ])


# =========================================================================
# OBSERVATION CHANNELS
# =========================================================================

def gen_hr(trajectory, t_grid, params, aux, prior_channels, seed):
    """Gaussian HR channel: hr = HR_base + alpha_HR * W + N(0, sigma_HR^2)."""
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    W = trajectory[:, 0]
    p = params
    T_len = len(t_grid)

    hr_mean = p['HR_base'] + p['alpha_HR'] * W
    hr = hr_mean + rng.normal(0.0, p['sigma_HR'], size=T_len)

    return {
        't_idx': np.arange(T_len, dtype=np.int32),
        'hr_value': hr.astype(np.float32),
    }


def gen_sleep(trajectory, t_grid, params, aux, prior_channels, seed):
    """3-level ordinal sleep channel with sticky-HMM persistence.

    Per-bin marginal P(label | Zt) uses the same sigmoid model as
    before:
        P(wake)      = 1 - sigma(Zt - c1)
        P(light+rem) = sigma(Zt - c1) - sigma(Zt - c2)
        P(deep)      = sigma(Zt - c2)

    Parameterised as (c_tilde, delta_c > 0) with c1 = c_tilde,
    c2 = c_tilde + delta_c.

    Labels are NOT drawn independently. With persistence
    P_stay = exp(-dt_h / tau_sleep_persist_h), the per-bin
    transition kernel mixes the sticky and the marginal:
        P_eff(new=k) = P_stay * 1[prev=k] + (1 - P_stay) * P_marg(k)

    This gives multi-bin coherence: at dt_h = 5/60 and tau = 0.5h,
    P_stay ≈ 0.85, mean run length ≈ 7 bins (~35 min). Healthy sleep
    cycles last 30-90 min per stage, so the resulting trace shows
    realistic block structure rather than the per-bin flicker that
    independent sampling produces.

    The estimator's sleep likelihood (`_ordinal_log_lik`) continues
    to treat labels as conditionally independent given Zt. This
    sim/est asymmetry is deliberate (analogous to the dropout
    asymmetry already in the channel DAG) — the persistence is
    primarily a generative-process correctness fix.
    """
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    Zt = trajectory[:, 1]
    p = params
    T_len = len(t_grid)

    c1 = p['c_tilde']
    c2 = c1 + p['delta_c']
    # Sharpness ≈ A_SCALE=6 by default; recovers pre-rescale threshold
    # separability in the [0, 1] Zt domain.
    sharp = float(p.get('sleep_sharpness', 1.0))

    s1 = _sigmoid(sharp * (Zt - c1)).astype(np.float64)   # P(sleep, any) = 1 - P(wake)
    s2 = _sigmoid(sharp * (Zt - c2)).astype(np.float64)   # P(deep)

    # Per-bin marginal probabilities: shape (T_len, 3) over (wake, light, deep).
    p_marg = np.stack([1.0 - s1, s1 - s2, s2], axis=1)
    # Numerical clip: floating-point can give tiny negative entries
    # at the threshold extremes (e.g. p_light = s1 - s2 ≈ -1e-17).
    p_marg = np.clip(p_marg, 0.0, 1.0)
    p_marg = p_marg / p_marg.sum(axis=1, keepdims=True)

    # Persistence factor — geometric run-length with mean 1/(1-P_stay)
    # bins. tau_sleep_persist_h falls back to 0.5 for older param sets
    # that predate the sticky channel.
    dt_h = float(t_grid[1] - t_grid[0])
    tau_h = float(p.get('tau_sleep_persist_h', 0.5))
    p_stay = np.exp(-dt_h / max(tau_h, 1e-6))

    labels = np.empty(T_len, dtype=np.int32)
    # Initialise from the marginal at t=0 (no previous label to stick to).
    cum0 = np.cumsum(p_marg[0])
    labels[0] = int(np.searchsorted(cum0, rng.random()))

    for t in range(1, T_len):
        sticky = np.zeros(3, dtype=np.float64)
        sticky[labels[t - 1]] = 1.0
        p_eff = p_stay * sticky + (1.0 - p_stay) * p_marg[t]
        cum = np.cumsum(p_eff)
        labels[t] = int(np.searchsorted(cum, rng.random()))

    return {
        't_idx': np.arange(T_len, dtype=np.int32),
        'sleep_level': labels,   # 0=wake, 1=light+rem, 2=deep
    }


def gen_steps(trajectory, t_grid, params, aux, prior_channels, seed):
    """Steps channel — log-Gaussian, WAKE-GATED.

    Replaces the older Poisson channel as of 2026-05-01 to match
    FSA-v2's pattern (ports cleanly into the bench's SMC²-MPC obs
    likelihood).

    For each per-step bin (t_grid resolution, e.g. 5 min):
      log_value = log(steps + 1) ~ N(mu_step0 + beta_W_steps * W,
                                      sigma_step^2)
    Only "observed" when sleep_label == 0 (wake) — the sticky-HMM
    sleep generator's `sleep_level` channel runs first (see
    SWAT_MODEL ChannelSpec ordering), and gen_steps reads it from
    prior_channels.

    Returns a per-bin array of log_value plus a present_mask flag
    (1.0 in wake bins, 0.0 elsewhere). Downstream consumers (psim's
    plot, the bench's `align_obs_fn`) decide how to surface the
    masked bins (typically scatter only the present ones).
    """
    del aux
    rng = np.random.default_rng(seed)
    W = trajectory[:, 0]
    p = params
    T_len = len(t_grid)

    # Wake gate: read sleep_level from the upstream channel. Default to
    # all-wake if absent (e.g. when steps is run without sleep — keeps
    # the channel useful for diagnostic / unit-test purposes).
    sleep_chan = prior_channels.get('sleep') if prior_channels else None
    if sleep_chan is not None and 'sleep_level' in sleep_chan:
        sleep_level = np.asarray(sleep_chan['sleep_level'])
        present_mask = (sleep_level == 0).astype(np.float64)
    else:
        present_mask = np.ones(T_len, dtype=np.float64)

    log_mean = p['mu_step0'] + p['beta_W_steps'] * W
    log_value = log_mean + rng.normal(0.0, p['sigma_step'], size=T_len)
    log_value = log_value.astype(np.float32)

    return {
        't_idx':        np.arange(T_len, dtype=np.int32),
        'log_value':    log_value,        # log(steps + 1)
        'present_mask': present_mask.astype(np.float32),
    }


def gen_stress(trajectory, t_grid, params, aux, prior_channels, seed):
    """Gaussian stress-score channel: stress = s_base + alpha_s * W + beta_s * Vn + N(0, sigma_s^2).

    Note: V_n is constant in Phase 1 so the stress channel's time variation
    comes solely from W.  The coupling to V_n acts as a subject-level offset
    that helps disambiguate V_n from V_h (which both feed u_W).
    """
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    W  = trajectory[:, 0]
    Vn = trajectory[:, 6]
    p = params
    T_len = len(t_grid)

    mean = p['s_base'] + p['alpha_s'] * W + p['beta_s'] * Vn
    stress = mean + rng.normal(0.0, p['sigma_s'], size=T_len)
    stress = np.clip(stress, 0.0, 100.0)

    return {
        't_idx':        np.arange(T_len, dtype=np.int32),
        'stress_score': stress.astype(np.float32),
    }


# =========================================================================
# PHYSICS VERIFICATION
# =========================================================================

def verify_physics(trajectory, t_grid, params):
    """Range / finiteness checks PLUS biological-realism metrics.

    Two classes of return value:

      - **Gating booleans** (must be True for ``PASS: physics
        verification`` to fire): ``W_in_0_1``, ``Zt_in_0_A``,
        ``a_nonneg``, ``T_nonneg``, ``T_bounded``, ``all_finite``.
        These are the necessary range / finiteness checks.

      - **Informational metrics + ``*_realistic`` flags**: numeric
        values plus string ``"yes"/"no"`` flags that indicate whether
        the simulated trajectory exhibits realistic sleep architecture.
        These do NOT gate (CLI's PASS line ignores non-bool fields and
        ignores string-typed flags) but ARE clearly visible in the
        run output so a modeller can spot tuning issues at simulation
        time, before a downstream consumer (psim, SMC²) wastes effort
        trying to fit data with wrong-shape dynamics.

    See public-dev #6 / public-dev #5 for the motivating story
    (SWAT Set A's Zt amplitude limit was found by an SMC² downstream
    diagnostic plot; should have been caught here).
    """
    del t_grid
    W, Zt, a, T = (trajectory[:, 0], trajectory[:, 1],
                    trajectory[:, 2], trajectory[:, 3])

    # T should settle to approximately sqrt(mu/eta) if mu > 0, else 0.
    # We can't know E a priori, so just check T is non-negative and finite.
    mu_0 = params['mu_0']; mu_E = params['mu_E']; eta = params['eta']
    mu_max = mu_0 + mu_E  # E <= 1
    T_max_expected = math.sqrt(max(mu_max / eta, 0.0)) if mu_max > 0 else 0.0

    # ── Sleep-architecture realism (informational) ────────────────────
    c_tilde = params['c_tilde']
    delta_c = params['delta_c']
    c2      = c_tilde + delta_c
    sleep_frac      = float((Zt > c_tilde).mean())   # fraction of bins with Zt > c1
    deep_sleep_frac = float((Zt > c2).mean())        # fraction of bins with Zt > c2
    Zt_max          = float(Zt.max())
    Zt_p99          = float(np.percentile(Zt, 99))

    # Healthy-adult reference ranges:
    #   total sleep:  25-40% of 24h (i.e. 6-10h sleep per day)
    #   deep sleep:   3-12% of 24h (typically ~5-8% in middle-aged adults)
    #   Zt should reach c2 deterministically during overnight peaks
    sleep_realistic       = "yes" if 0.25 <= sleep_frac <= 0.40 else "no"
    deep_sleep_realistic  = "yes" if 0.03 <= deep_sleep_frac <= 0.15 else "no"
    Zt_reaches_c2         = "yes" if Zt_p99 > c2 else "no"

    return {
        # ── Gating booleans (range + finiteness) ──────────────────────
        # Zt and a are now in [0, 1] (Phase 3.6 rescale + Jacobi diffusion).
        'W_in_0_1':       bool((W.min() > -0.05) and (W.max() < 1.05)),
        'Zt_in_0_1':      bool((Zt.min() > -0.05) and (Zt.max() < 1.05)),
        'a_in_0_1':       bool((a.min() > -0.05) and (a.max() < 1.05)),
        'T_nonneg':       bool(T.min() > -0.1),
        'T_bounded':      bool(T.max() < 4.0 * max(T_max_expected, 1.0) + 1.0),
        'all_finite':     bool(np.all(np.isfinite(trajectory))),
        # ── Informational state-range metrics ─────────────────────────
        'W_range':        float(W.max() - W.min()),
        'Zt_range':       float(Zt.max() - Zt.min()),
        'T_range':        float(T.max() - T.min()),
        'T_mean':         float(T.mean()),
        # ── Sleep-architecture realism (informational; see docstring) ─
        'sleep_fraction':                         sleep_frac,
        'deep_sleep_fraction':                    deep_sleep_frac,
        'Zt_max':                                 Zt_max,
        'Zt_p99':                                 Zt_p99,
        'c2_threshold':                           float(c2),
        'sleep_fraction_realistic':               sleep_realistic,
        'deep_sleep_fraction_realistic':          deep_sleep_realistic,
        'Zt_reaches_deep_threshold_realistic':    Zt_reaches_c2,
    }


# =========================================================================
# PARAMETER SETS
# =========================================================================

# Parameter set A = healthy basin.  Inherits sleep-wake-adenosine values
# from sleep_wake_20p PARAM_SET_A.  T-block priors from spec §10:
#   mu_0 = -0.5 (no pulsatility at E=0),  mu_E = 1.0 (mu > 0 at E~1),
#   eta  = 0.5 (T* = sqrt(mu/eta) ~ 1 at healthy),  tau_T = 48 h,
#   T_0  = 1.0 (start near healthy steady state),  alpha_T = 0.3
#   T_T  = 0.01  (mild noise)
PARAM_SET_A = {
    # ── Inherited 17-block + diffusion ─────────────────
    'kappa':      6.67,
    'lmbda':     32.0,
    'gamma_3':    8.0,
    'tau_W':      2.0,
    'tau_Z':      2.0,
    'V_c':        0.0,    # phase-shift (hours); 0 = morning type, healthy
    'HR_base':   50.0,
    'alpha_HR':  25.0,
    'sigma_HR':   8.0,
    # Sleep-stage thresholds rescaled 2026-05-01 from the old [0, A_SCALE=6]
    # domain to the new [0, 1] domain (drift no longer multiplies by A_SCALE):
    #   c_tilde  (wake → light): 2.5 / 6 ≈ 0.417
    #   delta_c  (light → deep): 1.5 / 6 = 0.25  → c2 = 0.667
    'c_tilde':    0.417,
    'tau_a':      3.0,
    # beta_Z 4.0 unchanged; the relative coupling strength (a → u_Z → sigma)
    # is identical post-rescale because a is also in [0, 1].
    'beta_Z':     4.0,
    'T_W':        0.01,
    # T_Z 0.05 unchanged in raw value — the rescale of Zt to [0, 1] makes
    # the per-bin Jacobi-noise step smaller automatically (sqrt(Z*(1-Z))
    # peaks at 0.5 vs the old constant 1.0, so the effective step is now
    # ~0.5x smaller for the same T_Z). Sleep blocks rely on the sticky-
    # HMM kernel in gen_sleep (see `tau_sleep_persist_h`).
    'T_Z':        0.05,
    'T_a':        0.01,

    # ── Stuart-Landau testosterone block ───────────
    # Tuning 2026-05-01 to make Set C visibly recover within the 14-day
    # horizon and Set A robustly sit in the healthy basin:
    #   mu_0 -0.5 → -0.3 — the structural tradeoff between mu_W_slow≈0
    #     (which wants a≈1) and mu_Z_slow≈0 (which wants a≈0) caps the
    #     time-averaged E_dyn at ~0.4 with V_h=1, V_n=0. With the old
    #     mu_0=-0.5 (E_crit=0.5), avg mu was negative → T decayed.
    #     With mu_0=-0.3 (E_crit=0.3), avg mu is positive → T grows.
    #   tau_T 48h → 24h — halves the relaxation timescale so Set C
    #     reaches the new healthy equilibrium T* = sqrt((mu_0+E_avg)/eta)
    #     within the 14-day horizon instead of needing 30+ days.
    'mu_0':      -0.3,
    'mu_E':       1.0,
    'eta':        0.5,
    'tau_T':     24.0,
    'alpha_T':    0.3,
    'T_T':        0.0001,

    # ── 3-level ordinal sleep channel (Zt ∈ [0, 1] post-rescale) ──
    'delta_c':    0.25,    # c2 = c_tilde + delta_c = 0.417 + 0.25 = 0.667 (deep)
    # Sleep-sharpness multiplier — restores the original [0, A_SCALE=6] domain
    # threshold sharpness after rescaling Zt to [0, 1]. Without this factor
    # the sigmoid in the unit interval is too soft (delta_c=0.25 spans only
    # ~6% of the 0-1 sigmoid's transition region), so even Set B (Zt mostly
    # below c_tilde) would yield P_sleep ≈ 0.4 per bin. Setting sharpness
    # equal to the legacy A_SCALE recovers the pre-rescale separability.
    'sleep_sharpness': 10.0,
    # Sleep-stage persistence: P_stay = exp(-dt_h / tau_sleep_persist_h).
    # At dt_h=5/60 and tau=1.0h, P_stay ≈ 0.92 → mean run length ~12 bins
    # (~60 min). With sharp sigmoid above this gives clean wake/sleep blocks
    # without over-persisting in Set B (insomnia, P_marg(sleep) ≈ 0.05).
    'tau_sleep_persist_h': 1.0,

    # ── Steps log-Gaussian channel, wake-gated ───────────────────
    # log(steps + 1) ~ N(mu_step0 + beta_W_steps * W, sigma_step^2),
    # observed only when sleep_label == 0 (wake). Per 15-min wake bin:
    # at W=0.5, log_mean = 4.4 → ~80 steps/bin ≈ 320 steps/h (typical
    # light-activity wake bin). Replaces the older Poisson channel.
    'mu_step0':       4.0,
    'beta_W_steps':   0.8,
    'sigma_step':     0.5,

    # ── New Garmin stress channel ────────────────────────────
    's_base':     30.0,    # baseline stress score
    'alpha_s':    40.0,    # wake modulation (W=0 -> 30, W=1 -> 70)
    'beta_s':     10.0,    # nuisance-load modulation (V_n=0.3 -> +3, V_n=3.5 -> +35)
    'sigma_s':    15.0,    # observation noise on 0-100 scale
}

INIT_STATE_A = {
    'W_0':    0.5,
    'Zt_0':   0.583,    # rescaled from 3.5 / A_SCALE=6 = 0.583 (Zt now ∈ [0, 1])
    'a_0':    0.5,
    'T_0':    0.5,      # start near healthy equilibrium T* ~ 0.5 (physically plausible)
    'Vh':     1.0,      # healthy basin
    'Vn':     0.0,      # 2026-05-01: was 0.3; per user, V_n=0 in A/C/D gives a
                         # clean baseline diagnostic (no chronic load). Set B
                         # explicitly overrides Vn=3.5 below.
}

# Time-grid controls
PARAM_SET_A['dt_hours'] = 5.0 / 60.0       # 5-minute resolution
PARAM_SET_A['t_total_hours'] = 14 * 24.0   # 14-day trial (vs 7d in 20p)
#                                             lengthened so that tau_T = 48h
#                                             is seen several times (R3' of proof)


# Parameter set B = pathological basin (low V_h, high V_n).  Should collapse
# to E ~ 0 -> mu(E) = mu_0 < 0 -> T -> 0 (hypogonadal flatline).
PARAM_SET_B = dict(PARAM_SET_A)
INIT_STATE_B = dict(INIT_STATE_A)
INIT_STATE_B['Vh'] = 0.2
INIT_STATE_B['Vn'] = 3.5  # raised from 2.0 — gives mu(E) ~ -0.45 (Set-D-strength collapse)
# T_0 inherits 0.5 from INIT_STATE_A — start near healthy equilibrium, observe collapse


# Parameter set C = recovery scenario: pathological priors but healthy
# V_h, V_n to exhibit rise of T from small initial value.
PARAM_SET_C = dict(PARAM_SET_A)
INIT_STATE_C = dict(INIT_STATE_A)
INIT_STATE_C['T_0'] = 0.05   # start near the 0-flatline; expect rise to T*


# Parameter set D = phase-shift pathology (shift worker / chronic jet lag).
# Healthy potentials (V_h=1.0, V_n=0.3) but V_c = 6 hours — subject's rhythm
# is 6h delayed relative to external light.  Both W and Zt swing with full
# amplitude, but their peak timing is mis-aligned with C(t).  The entrainment
# quality's phase-correlation term drops toward 0, driving E below E_crit and
# collapsing T.
#
# This demonstrates the fourth failure mode the V_h, V_n potentials alone
# cannot produce: phase misalignment with healthy amplitude.
PARAM_SET_D = dict(PARAM_SET_A)
PARAM_SET_D['V_c'] = 6.0    # 6-hour phase shift (shift worker)
INIT_STATE_D = dict(INIT_STATE_A)
# T_0 inherits 0.5 from INIT_STATE_A — start near healthy equilibrium, observe collapse


# =========================================================================
# THE MODEL OBJECT
# =========================================================================

SWAT_MODEL = SDEModel(
    name="swat",
    version="1.1",        # Phase 3.6: Jacobi diffusion + Zt, a ∈ [0, 1] +
                          # log-Gaussian wake-gated steps + sticky sleep.

    states=(
        StateSpec("W",  0.0, 1.0),
        StateSpec("Zt", 0.0, 1.0),    # rescaled from [0, A_SCALE=6]
        StateSpec("a",  0.0, 1.0),    # rescaled from [0, 5] (a is a low-pass
                                       # of W ∈ [0,1], naturally in [0, 1])
        StateSpec("T",  0.0, 5.0),    # testosterone pulsatility amplitude
        StateSpec("C", -1.0, 1.0, is_deterministic=True,
                  analytical_fn=circadian, analytical_fn_jax=circadian_jax),
        StateSpec("Vh", -5.0, 5.0, is_deterministic=True),
        StateSpec("Vn", -5.0, 5.0, is_deterministic=True),
    ),

    drift_fn=drift,
    drift_fn_jax=drift_jax,
    diffusion_type=DIFFUSION_DIAGONAL_STATE,    # Jacobi for W/Zt/a, identity for T
    diffusion_fn=diffusion_diagonal,
    noise_scale_fn=noise_scale_fn,
    noise_scale_fn_jax=noise_scale_fn_jax,
    make_aux_fn=make_aux,
    make_aux_fn_jax=make_aux_jax,
    make_y0_fn=make_y0,

    channels=(
        # gen_sleep MUST run before gen_steps so that gen_steps can read the
        # wake-gate (sleep_label==0) for its log-Gaussian wake-gated channel.
        ChannelSpec("hr",     depends_on=(), generate_fn=gen_hr),
        ChannelSpec("sleep",  depends_on=(), generate_fn=gen_sleep),
        ChannelSpec("steps",  depends_on=("sleep",), generate_fn=gen_steps),
        ChannelSpec("stress", depends_on=(), generate_fn=gen_stress),
    ),

    plot_fn=plot_swat,
    verify_physics_fn=verify_physics,

    param_sets={'A': PARAM_SET_A, 'B': PARAM_SET_B,
                'C': PARAM_SET_C, 'D': PARAM_SET_D},
    init_states={'A': INIT_STATE_A, 'B': INIT_STATE_B,
                 'C': INIT_STATE_C, 'D': INIT_STATE_D},
    exogenous_inputs={'A': {}, 'B': {}, 'C': {}, 'D': {}},
)

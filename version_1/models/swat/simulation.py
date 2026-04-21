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
    SDEModel, StateSpec, ChannelSpec, DIFFUSION_DIAGONAL_CONSTANT)
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
    dZt = (A_SCALE * float(_sigmoid(u_Z)) - Zt) / tau_Z
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
    dZt = (A_SCALE * jax.nn.sigmoid(u_Z) - Zt) / p['tau_Z']
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
    """Diagonal SDE coefficients.

    States 4 (C), 5 (Vh), 6 (Vn) are deterministic in Phase 1 -> zero.
    """
    p = params
    return np.array([
        math.sqrt(2.0 * p['T_W']),
        math.sqrt(2.0 * p['T_Z']),
        math.sqrt(2.0 * p['T_a']),
        math.sqrt(2.0 * p['T_T']),   # testosterone noise
        0.0,   # C (deterministic)
        0.0,   # Vh (constant)
        0.0,   # Vn (constant)
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
    """3-level ordinal sleep channel: sleep_level in {0=wake, 1=light+rem, 2=deep}.

    Two thresholds c1 < c2 on the Zt state:
        P(wake)      = 1 - sigma(Zt - c1)
        P(light+rem) = sigma(Zt - c1) - sigma(Zt - c2)
        P(deep)      = sigma(Zt - c2)

    Parameterised as (c_tilde, delta_c > 0) with c1 = c_tilde, c2 = c_tilde + delta_c.
    """
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    Zt = trajectory[:, 1]
    p = params
    T_len = len(t_grid)

    c1 = p['c_tilde']
    c2 = c1 + p['delta_c']

    s1 = _sigmoid(Zt - c1).astype(np.float64)   # P(sleep, any) = 1 - P(wake)
    s2 = _sigmoid(Zt - c2).astype(np.float64)   # P(deep)

    # Cumulative probabilities: cum0 = P(level <= 0) = 1 - s1
    #                          cum1 = P(level <= 1) = 1 - s2
    draws = rng.random(size=T_len)
    labels = np.where(draws < 1.0 - s1, 0,
             np.where(draws < 1.0 - s2, 1, 2)).astype(np.int32)

    return {
        't_idx': np.arange(T_len, dtype=np.int32),
        'sleep_level': labels,   # 0=wake, 1=light+rem, 2=deep
    }


def gen_steps(trajectory, t_grid, params, aux, prior_channels, seed):
    """Poisson step-count channel on 15-minute bins.

    Rate:
        r(W) = lambda_base + lambda_step * sigma(10 * (W - W_thresh))
    Bin count:
        k_bin ~ Poisson(r(W_bin) * bin_hours)

    where W_bin is the mean wakefulness over the 15-min bin.
    """
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    W = trajectory[:, 0]
    p = params
    dt_h = float(t_grid[1] - t_grid[0])

    # 15-min bin size (in sample count)
    bin_hours = 0.25
    bin_size = max(int(round(bin_hours / dt_h)), 1)
    n_bins = len(t_grid) // bin_size

    # Mean W per bin
    W_trunc = W[: n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)

    rate = p['lambda_base'] + p['lambda_step'] * _sigmoid(
        10.0 * (W_trunc - p['W_thresh']))
    expected = rate * bin_hours
    k = rng.poisson(expected).astype(np.int32)

    # Timestamp for each bin = bin start
    bin_t_idx = (np.arange(n_bins) * bin_size).astype(np.int32)

    return {
        't_idx':      bin_t_idx,
        'steps':      k,
        'bin_hours':  np.float32(bin_hours),
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
    """Minimal checks: states in expected ranges, T positivity, transitions observed."""
    del t_grid
    W, Zt, a, T = (trajectory[:, 0], trajectory[:, 1],
                    trajectory[:, 2], trajectory[:, 3])

    # T should settle to approximately sqrt(mu/eta) if mu > 0, else 0.
    # We can't know E a priori, so just check T is non-negative and finite.
    mu_0 = params['mu_0']; mu_E = params['mu_E']; eta = params['eta']
    mu_max = mu_0 + mu_E  # E <= 1
    T_max_expected = math.sqrt(max(mu_max / eta, 0.0)) if mu_max > 0 else 0.0

    return {
        'W_in_0_1':       bool((W.min() > -0.05) and (W.max() < 1.05)),
        'Zt_in_0_A':      bool((Zt.min() > -0.5) and (Zt.max() < A_SCALE + 0.5)),
        'a_nonneg':       bool(a.min() > -0.5),
        'T_nonneg':       bool(T.min() > -0.1),
        'T_bounded':      bool(T.max() < 4.0 * max(T_max_expected, 1.0) + 1.0),
        'W_range':        float(W.max() - W.min()),
        'Zt_range':       float(Zt.max() - Zt.min()),
        'T_range':        float(T.max() - T.min()),
        'T_mean':         float(T.mean()),
        'all_finite':     bool(np.all(np.isfinite(trajectory))),
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
    'c_tilde':    3.0,
    'tau_a':      3.0,
    'beta_Z':     2.5,
    'T_W':        0.01,
    'T_Z':        0.05,
    'T_a':        0.01,

    # ── New Stuart-Landau testosterone block ───────────
    'mu_0':      -0.5,
    'mu_E':       1.0,
    'eta':        0.5,
    'tau_T':     48.0,
    'alpha_T':    0.3,
    'T_T':        0.0001,

    # ── New 3-level ordinal sleep channel ────────────────────
    'delta_c':    1.5,    # c2 = c_tilde + delta_c; c1=3.0 -> c2=4.5 (deep threshold)

    # ── New steps Poisson channel ────────────────────────────
    'lambda_base': 0.5,    # rare steps during sleep (~0.5 steps/h)
    'lambda_step': 200.0,  # peak rate during wake (~50 steps/15min bin)
    'W_thresh':    0.6,    # threshold above which step rate activates

    # ── New Garmin stress channel ────────────────────────────
    's_base':     30.0,    # baseline stress score
    'alpha_s':    40.0,    # wake modulation (W=0 -> 30, W=1 -> 70)
    'beta_s':     10.0,    # nuisance-load modulation (V_n=0.3 -> +3, V_n=3.5 -> +35)
    'sigma_s':    15.0,    # observation noise on 0-100 scale
}

INIT_STATE_A = {
    'W_0':    0.5,
    'Zt_0':   3.5,
    'a_0':    0.5,
    'T_0':    0.5,      # start near healthy equilibrium T* ~ 0.5 (physically plausible)
    'Vh':     1.0,      # healthy basin
    'Vn':     0.3,
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
    version="1.0",

    states=(
        StateSpec("W",  0.0, 1.0),
        StateSpec("Zt", 0.0, A_SCALE),
        StateSpec("a",  0.0, 5.0),
        StateSpec("T",  0.0, 5.0),   # testosterone pulsatility amplitude
        StateSpec("C", -1.0, 1.0, is_deterministic=True,
                  analytical_fn=circadian, analytical_fn_jax=circadian_jax),
        StateSpec("Vh", -5.0, 5.0, is_deterministic=True),
        StateSpec("Vn", -5.0, 5.0, is_deterministic=True),
    ),

    drift_fn=drift,
    drift_fn_jax=drift_jax,
    diffusion_type=DIFFUSION_DIAGONAL_CONSTANT,
    diffusion_fn=diffusion_diagonal,
    make_aux_fn=make_aux,
    make_aux_fn_jax=make_aux_jax,
    make_y0_fn=make_y0,

    channels=(
        ChannelSpec("hr",     depends_on=(), generate_fn=gen_hr),
        ChannelSpec("sleep",  depends_on=(), generate_fn=gen_sleep),
        ChannelSpec("steps",  depends_on=(), generate_fn=gen_steps),
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

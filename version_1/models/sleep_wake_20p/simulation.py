"""
models/sleep_wake_20p/simulation.py — 5-State Sleep-Wake-Adenosine SDE.
=========================================================================
Date:    17 April 2026
Version: 1.0

20-parameter SDE derived from the 17-parameter identifiability proof
(Identifiability_Proof_17_Parameter_Sleep_Wake_Adenosine_Model.md) plus
three diffusion temperatures {T_W, T_Z, T_a} making the latent layer a
true SDE rather than an ODE (see user decision 2026-04-17).

Latent SDE (5 states; C is deterministic):

    dW     = (1/tau_W) [sigma(u_W) - W] dt   + sqrt(2 T_W) dB_W
    dZt    = (1/tau_Z) [A sigma(u_Z) - Zt] dt+ sqrt(2 T_Z) dB_Z
    da     = (1/tau_a) (W - a) dt            + sqrt(2 T_a) dB_a
    C      = sin(2 pi t / 24 + phi)          (deterministic)
    V_h, V_n: constants (Phase 1)

with

    u_W = -kappa * Zt + lambda * C(t) + V_h + V_n - a
    u_Z = -gamma_3 * W - V_n + beta_Z * a
    A   = 6  (fixed scale constant, NOT a parameter)

Observation channels:

    hr    ~ N(HR_base + alpha_HR * W, sigma_HR^2)       (continuous)
    sleep ~ Bernoulli(sigma(Zt - c_tilde))              (binary: 0=wake, 1=sleep)

Parameter count: 20 = 17 deterministic + 3 diffusion.

The generic simulator framework imports only the SLEEP_WAKE_20P_MODEL object.
"""

import math
import numpy as np

from simulator.sde_model import (
    SDEModel, StateSpec, ChannelSpec, DIFFUSION_DIAGONAL_CONSTANT)
from models.sleep_wake_20p.sim_plots import plot_sleep_wake_20p


# =========================================================================
# CONSTANTS
# =========================================================================

A_SCALE = 6.0   # fixed rescaling constant for tilde-Z; NOT a parameter


# =========================================================================
# ELEMENTARY HELPERS
# =========================================================================

def _sigmoid(x):
    """Numerically stable logistic sigmoid (numpy)."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def circadian(t, params):
    """C(t) = sin(2 pi t / 24 + phi)  — analytical deterministic state."""
    return math.sin(2.0 * math.pi * t / 24.0 + params['phi'])


def circadian_jax(t, params):
    """JAX variant of circadian() for the Diffrax solver."""
    import jax.numpy as jnp
    return jnp.sin(2.0 * jnp.pi * t / 24.0 + params['phi'])


# =========================================================================
# DRIFT
# =========================================================================

def drift(t, y, params, aux):
    """5-state drift f(t, y, theta) for the scipy solver.

    Args:
        t: scalar time in hours.
        y: state vector [W, Zt, a, C, placeholder_for_Vh, placeholder_for_Vn].
           We carry 6 components so the simulator's solver handles the
           deterministic circadian state uniformly with the other models.
           V_h and V_n are TREATED AS CONSTANTS in Phase 1 — they are
           carried as constant state components with zero drift and zero
           diffusion so forward integration is identical to the paper's
           formulation.
        params: dict of parameter values.
        aux: unused in this model (None returned by make_aux).

    Returns:
        dy/dt as ndarray(6,).
    """
    del aux  # unused
    W, Zt, a, C, Vh, Vn = y[0], y[1], y[2], y[3], y[4], y[5]

    p = params
    kappa = p['kappa']; lam = p['lmbda']; gamma_3 = p['gamma_3']
    beta_Z = p['beta_Z']
    tau_W = p['tau_W']; tau_Z = p['tau_Z']; tau_a = p['tau_a']

    C_exact = circadian(t, p)

    u_W = -kappa * Zt + lam * C_exact + Vh + Vn - a
    u_Z = -gamma_3 * W - Vn + beta_Z * a

    dW = (float(_sigmoid(u_W)) - W) / tau_W
    dZt = (A_SCALE * float(_sigmoid(u_Z)) - Zt) / tau_Z
    da = (W - a) / tau_a

    dC = (2.0 * math.pi / 24.0) * math.cos(2.0 * math.pi * t / 24.0 + p['phi'])

    # V_h, V_n are constants in Phase 1 -> zero drift
    dVh = 0.0
    dVn = 0.0

    return np.array([dW, dZt, da, dC, dVh, dVn])


def drift_jax(t, y, args):
    """5-state drift in JAX (for the Diffrax solver).  Matches drift()."""
    import jax.numpy as jnp
    import jax
    (p,) = args
    W, Zt, a, C, Vh, Vn = y[0], y[1], y[2], y[3], y[4], y[5]

    C_ex = jnp.sin(2.0 * jnp.pi * t / 24.0 + p['phi'])
    u_W = -p['kappa'] * Zt + p['lmbda'] * C_ex + Vh + Vn - a
    u_Z = -p['gamma_3'] * W - Vn + p['beta_Z'] * a

    dW = (jax.nn.sigmoid(u_W) - W) / p['tau_W']
    dZt = (A_SCALE * jax.nn.sigmoid(u_Z) - Zt) / p['tau_Z']
    da = (W - a) / p['tau_a']
    dC = (2.0 * jnp.pi / 24.0) * jnp.cos(2.0 * jnp.pi * t / 24.0 + p['phi'])

    return jnp.array([dW, dZt, da, dC, 0.0, 0.0])


# =========================================================================
# DIFFUSION
# =========================================================================

def diffusion_diagonal(params):
    """Diagonal SDE coefficients.

    States 3 (C), 4 (Vh), 5 (Vn) are deterministic in Phase 1 -> zero.
    """
    p = params
    return np.array([
        math.sqrt(2.0 * p['T_W']),
        math.sqrt(2.0 * p['T_Z']),
        math.sqrt(2.0 * p['T_a']),
        0.0,   # C (deterministic)
        0.0,   # Vh (constant)
        0.0,   # Vn (constant)
    ])


# =========================================================================
# AUXILIARY / INITIAL STATE
# =========================================================================

def make_aux(params, init_state, t_grid, exogenous):
    """This model needs no auxiliary state."""
    del params, init_state, t_grid, exogenous
    return None


def make_aux_jax(params, init_state, t_grid, exogenous):
    """JAX auxiliary builder for the Diffrax solver."""
    import jax.numpy as jnp
    del init_state, t_grid, exogenous
    p_jax = {k: jnp.float64(v) for k, v in params.items()}
    return (p_jax,)


def make_y0(init_dict, params):
    """Build [W, Zt, a, C, Vh, Vn] at t=0 from init parameters."""
    C0 = math.sin(params['phi'])
    return np.array([
        init_dict['W_0'],
        init_dict['Zt_0'],
        init_dict['a_0'],
        C0,
        init_dict['Vh'],
        init_dict['Vn'],
    ])


# =========================================================================
# OBSERVATION CHANNELS
# =========================================================================

def gen_hr(trajectory, t_grid, params, aux, prior_channels, seed):
    """Gaussian HR channel: hr = HR_base + alpha_HR * W + N(0, sigma_HR^2).

    Emitted at the full simulation grid (5-min resolution).
    """
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    W = trajectory[:, 0]
    p = params
    T = len(t_grid)

    hr_mean = p['HR_base'] + p['alpha_HR'] * W
    hr = hr_mean + rng.normal(0.0, p['sigma_HR'], size=T)

    return {
        't_idx': np.arange(T, dtype=np.int32),
        'hr_value': hr.astype(np.float32),
    }


def gen_sleep(trajectory, t_grid, params, aux, prior_channels, seed):
    """Binary sleep channel: sleep_label ~ Bernoulli(sigma(Zt - c_tilde)).

    Emitted at the full simulation grid.
    """
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    Zt = trajectory[:, 1]
    p = params
    T = len(t_grid)

    prob_sleep = _sigmoid(Zt - p['c_tilde']).astype(np.float64)
    draws = rng.random(size=T)
    labels = (draws < prob_sleep).astype(np.int32)

    return {
        't_idx': np.arange(T, dtype=np.int32),
        'sleep_label': labels,
    }


# =========================================================================
# PHYSICS VERIFICATION
# =========================================================================

def verify_physics(trajectory, t_grid, params):
    """Minimal checks: states in expected ranges, transitions observed."""
    del t_grid, params
    W, Zt, a = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    return {
        'W_in_0_1':       bool((W.min() > -0.05) and (W.max() < 1.05)),
        'Zt_in_0_A':      bool((Zt.min() > -0.5) and (Zt.max() < A_SCALE + 0.5)),
        'a_nonneg':       bool(a.min() > -0.5),
        'W_range':        float(W.max() - W.min()),
        'Zt_range':       float(Zt.max() - Zt.min()),
        'all_finite':     bool(np.all(np.isfinite(trajectory))),
    }


# =========================================================================
# PARAMETER SETS
# =========================================================================

# Parameter set A = prior medians from the identifiability proof doc (§2.5).
# V_h high / V_n low -> healthy basin.
PARAM_SET_A = {
    # Flip-flop dynamics
    # NOTE on gamma_3: the identifiability proof used 60 for the *deterministic*
    # ODE.  In the SDE version, T_W = 0.01 keeps a residual noise floor of
    # mean W ≈ 0.033 during sleep.  With gamma_3 = 60, u_Z = -60×0.033 - 0.3
    # + 0.75 = -1.55, giving Zt_eq ≈ 1.05 — BELOW the sleep threshold c_tilde
    # = 1.5, so sleep labels are 50% noise.  Reducing gamma_3 to 8 gives
    # u_Z = -8×0.033 + 0.45 = 0.19, Zt_eq ≈ 3.3 >> 1.5 → clear flip-flop.
    # Bistability is preserved: kappa × gamma_3 = 6.67 × 8 = 53 (vs 400 before).
    'kappa':      6.67,
    'lmbda':     32.0,   # 'lambda' is a Python keyword; use lmbda
    'gamma_3':    8.0,   # reduced from 60 — see note above
    'tau_W':      2.0,
    'tau_Z':      2.0,
    # Circadian
    'phi':       -math.pi / 3.0,
    # HR observation
    'HR_base':   50.0,
    'alpha_HR':  25.0,
    'sigma_HR':   8.0,
    # Binary sleep observation
    # NOTE on c_tilde: with Zt clipped at 0, the daytime false-sleep floor is
    # σ(0 − c_tilde) regardless of dynamics.  c_tilde=1.5 gives σ(-1.5)=18%,
    # producing ~35 false sleep labels per day.  Raising to 3.0 drops this to
    # σ(-3.0)=5%.  beta_Z is raised in tandem so Zt_sleep stays well above 3.0.
    'c_tilde':    3.0,   # raised from 1.5 — see note above
    # Adenosine
    'tau_a':      3.0,
    # NOTE on beta_Z: raised from 1.5 → 2.5 to compensate for the higher
    # c_tilde threshold.  With beta_Z=2.5 and a≈1 at sleep onset:
    #   u_Z = -0.3 + 2.5 = 2.2 → Zt_sleep = 6×σ(2.2) = 5.4 >> c_tilde=3.0
    # At end of sleep (a→0, adenosine cleared): Zt→2.55, prob_sleep≈39% —
    # physiologically correct: sleep pressure falls as adenosine clears.
    'beta_Z':     2.5,   # raised from 1.5 — see note above
    # Potentials (Phase 1: constants)
    # NOTE: in simulation they are held in the state vector; in estimation
    # they are parameters.  The values here go BOTH to sim params (for
    # diffusion / aux look-up) AND to init_state (for the initial state
    # vector) -- see INIT_STATE_A below.
    # Diffusion temperatures (SDE)
    'T_W':        0.01,
    'T_Z':        0.05,
    'T_a':        0.01,
}

# Initial state vector at t=0 for parameter set A.  Note:
#   * W_0, Zt_0, a_0 are latent-state initial conditions (3 of the 20 params)
#   * Vh, Vn are constants (2 of the 20 params) carried in the state vector
INIT_STATE_A = {
    'W_0':    0.5,
    'Zt_0':   3.5,   # raised from 1.8 to sit above new c_tilde=3.0 at t=0
    'a_0':    0.5,
    'Vh':     1.0,   # healthy basin: V_h high
    'Vn':     0.3,   # healthy basin: V_n low
}

# Time-grid controls (read by simulator/run_simulator.py via param dict):
PARAM_SET_A['dt_hours'] = 5.0 / 60.0       # 5-minute resolution
PARAM_SET_A['t_total_hours'] = 7 * 24.0    # 7-day trial


# Parameter set B = pathological basin (low V_h, high V_n).  Everything
# else identical; used to confirm the model exhibits basin-dependent
# dynamics as required by the identifiability proof doc §2.6.
PARAM_SET_B = dict(PARAM_SET_A)
INIT_STATE_B = dict(INIT_STATE_A)
INIT_STATE_B['Vh'] = 0.2
INIT_STATE_B['Vn'] = 2.0


# =========================================================================
# THE MODEL OBJECT
# =========================================================================

SLEEP_WAKE_20P_MODEL = SDEModel(
    name="sleep_wake_20p",
    version="1.0",

    states=(
        StateSpec("W",  0.0, 1.0),
        StateSpec("Zt", 0.0, A_SCALE),
        StateSpec("a",  0.0, 5.0),
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
        ChannelSpec("hr",    depends_on=(), generate_fn=gen_hr),
        ChannelSpec("sleep", depends_on=(), generate_fn=gen_sleep),
    ),

    plot_fn=plot_sleep_wake_20p,
    verify_physics_fn=verify_physics,

    param_sets={'A': PARAM_SET_A, 'B': PARAM_SET_B},
    init_states={'A': INIT_STATE_A, 'B': INIT_STATE_B},
    exogenous_inputs={'A': {}, 'B': {}},
)

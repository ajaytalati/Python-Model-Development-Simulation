"""models/fitness_strain_amplitude/simulation.py — 3-State FSA SDE.

Date:    18 April 2026
Version: 1.0

Three-state Fitness-Strain-Amplitude (FSA) model.  Time unit is DAYS.

SDE
---

    dB = [(1 + alpha_A * A)/tau_B * (T_B(t) - B)] dt
         + sigma_B * sqrt(B(1-B)) dW_B                            (Jacobi)

    dF = [Phi(t) - (1 + lambda_B B + lambda_A A)/tau_F * F] dt
         + sigma_F * sqrt(F) dW_F                                 (CIR)

    dA = [mu(B,F) * A - eta * A^3] dt
         + sigma_A * sqrt(A + eps_A) dW_A           (regularised Landau)

    mu(B, F) = mu_0 + mu_B * B - mu_F * F - mu_FF * F^2

The eps_A > 0 regularisation makes A = 0 non-absorbing (see §5.1 of the
specification document).

Inputs: piecewise-constant schedules T_B(t) (one segment) and Phi(t)
(two segments with jump time).  Both emitted as channels.

Observation: direct observation of (B, F, A) with shared scalar
sigma_obs, emitted as three separate channels (obs_B, obs_F, obs_A).
"""

import math
import numpy as np

from simulator.sde_model import (
    SDEModel, StateSpec, ChannelSpec,
    DIFFUSION_DIAGONAL_CONSTANT, DIFFUSION_DIAGONAL_STATE)
from models.fitness_strain_amplitude.sim_plots import plot_fsa


# =========================================================================
# FROZEN CONSTANTS
# =========================================================================

EPS_A_FROZEN = 1.0e-4    # Boundary regularisation for A-diffusion
EPS_B_FROZEN = 1.0e-4    # Numerical clip for B-diffusion


# =========================================================================
# INPUT SCHEDULES (piecewise constant)
# =========================================================================

def _T_B(t, T_B_const):
    """Single-segment constant schedule for T_B."""
    return T_B_const


def _Phi(t, Phi_1, Phi_2, T_jump):
    """Two-segment piecewise-constant schedule for Phi with jump."""
    return Phi_1 if t < T_jump else Phi_2


# =========================================================================
# DRIFT
# =========================================================================

def drift(t, y, params, aux):
    """FSA drift.  aux = (T_B_const, Phi_1, Phi_2, T_jump)."""
    T_B_const, Phi_1, Phi_2, T_jump = aux
    p = params
    B = y[0]; F = y[1]; A = y[2]

    T_B_t = _T_B(t, T_B_const)
    Phi_t = _Phi(t, Phi_1, Phi_2, T_jump)

    mu = (p['mu_0'] + p['mu_B'] * B
          - p['mu_F'] * F - p['mu_FF'] * F * F)

    dB = (1.0 + p['alpha_A'] * A) / p['tau_B'] * (T_B_t - B)
    dF = Phi_t - (1.0 + p['lambda_B'] * B
                  + p['lambda_A'] * A) / p['tau_F'] * F
    dA = mu * A - p['eta'] * A * A * A

    return np.array([dB, dF, dA])


def drift_jax(t, y, args):
    """JAX variant of drift."""
    import jax.numpy as jnp
    p, T_B_const, Phi_1, Phi_2, T_jump = args
    B = y[0]; F = y[1]; A = y[2]

    T_B_t = T_B_const                       # single-segment
    Phi_t = jnp.where(t < T_jump, Phi_1, Phi_2)

    mu = (p['mu_0'] + p['mu_B'] * B
          - p['mu_F'] * F - p['mu_FF'] * F * F)

    dB = (1.0 + p['alpha_A'] * A) / p['tau_B'] * (T_B_t - B)
    dF = Phi_t - (1.0 + p['lambda_B'] * B
                  + p['lambda_A'] * A) / p['tau_F'] * F
    dA = mu * A - p['eta'] * A * A * A

    return jnp.array([dB, dF, dA])


# =========================================================================
# DIFFUSION (state-dependent — handled by the SDE solver as a callable)
# =========================================================================

def diffusion_diagonal(params):
    """Return diagonal noise *amplitudes* (state-dependent multiplier
    is applied by the Jacobi/CIR forms; here we supply sigma_i only).
    """
    return np.array([params['sigma_B'],
                     params['sigma_F'],
                     params['sigma_A']])


# Factored diffusion: sigma_i(x) = sigma_i * g_i(x), with sigma_i from
# diffusion_diagonal(params) (per-state scalar magnitudes) and g_i from
# noise_scale_fn(y, params) (state-dependent multipliers).  The framework
# supports this via DIFFUSION_DIAGONAL_STATE (sde_solver_scipy.py v1.1+
# and sde_solver_diffrax.py v1.1+).
def noise_scale_fn(y, params):
    """State-dependent noise scaler (numpy).

    Returns [sqrt(B(1-B)), sqrt(F), sqrt(A + eps_A)].  The per-step SDE
    increment is  sigma_i * noise_scale_i * sqrt(dt) * xi_i.
    """
    del params
    B = np.clip(y[0], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F = max(y[1], 0.0)
    A = max(y[2], 0.0)
    return np.array([math.sqrt(B * (1.0 - B)),
                     math.sqrt(F),
                     math.sqrt(A + EPS_A_FROZEN)])


def noise_scale_fn_jax(y, params):
    """State-dependent noise scaler (JAX).  Same math as noise_scale_fn."""
    import jax.numpy as jnp
    del params
    B = jnp.clip(y[0], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F = jnp.maximum(y[1], 0.0)
    A = jnp.maximum(y[2], 0.0)
    return jnp.array([jnp.sqrt(B * (1.0 - B)),
                      jnp.sqrt(F),
                      jnp.sqrt(A + EPS_A_FROZEN)])


# =========================================================================
# AUXILIARY / INITIAL STATE
# =========================================================================

def make_aux(params, init_state, t_grid, exogenous):
    del params, init_state, t_grid
    return (exogenous['T_B_const'],
            exogenous['Phi_1'],
            exogenous['Phi_2'],
            exogenous['T_jump'])


def make_aux_jax(params, init_state, t_grid, exogenous):
    import jax.numpy as jnp
    del init_state, t_grid
    p_jax = {k: jnp.float64(v) for k, v in params.items()}
    return (p_jax,
            jnp.float64(exogenous['T_B_const']),
            jnp.float64(exogenous['Phi_1']),
            jnp.float64(exogenous['Phi_2']),
            jnp.float64(exogenous['T_jump']))


def make_y0(init_dict, params):
    del params
    return np.array([init_dict['B_0'], init_dict['F_0'], init_dict['A_0']])


# =========================================================================
# OBSERVATION CHANNELS — three separate channels, one per latent state
# =========================================================================

def _gen_obs_component(trajectory, t_grid, params, component_idx, seed):
    """Shared helper for Gaussian obs on a single latent component."""
    rng = np.random.default_rng(seed)
    T = len(t_grid)
    x = trajectory[:, component_idx]
    noise = rng.normal(0.0, params['sigma_obs'], size=T)
    return {
        't_idx':     np.arange(T, dtype=np.int32),
        'obs_value': (x + noise).astype(np.float32),
    }


def gen_obs_B(trajectory, t_grid, params, aux, prior_channels, seed):
    del aux, prior_channels
    return _gen_obs_component(trajectory, t_grid, params, 0, seed)


def gen_obs_F(trajectory, t_grid, params, aux, prior_channels, seed):
    del aux, prior_channels
    return _gen_obs_component(trajectory, t_grid, params, 1, seed + 1)


def gen_obs_A(trajectory, t_grid, params, aux, prior_channels, seed):
    del aux, prior_channels
    return _gen_obs_component(trajectory, t_grid, params, 2, seed + 2)


def gen_T_B_channel(trajectory, t_grid, params, aux, prior_channels, seed):
    """Emit T_B(t) schedule as a channel (exogenous input record)."""
    del trajectory, params, prior_channels, seed
    T_B_const, _Phi_1, _Phi_2, _T_jump = aux
    val = np.full(len(t_grid), T_B_const, dtype=np.float32)
    return {'t_idx':    np.arange(len(t_grid), dtype=np.int32),
            'T_B_value': val}


def gen_Phi_channel(trajectory, t_grid, params, aux, prior_channels, seed):
    """Emit Phi(t) schedule as a channel."""
    del trajectory, params, prior_channels, seed
    _T_B_const, Phi_1, Phi_2, T_jump = aux
    t = t_grid
    val = np.where(t < T_jump, Phi_1, Phi_2).astype(np.float32)
    return {'t_idx':     np.arange(len(t), dtype=np.int32),
            'Phi_value': val}


# =========================================================================
# PHYSICS VERIFICATION
# =========================================================================

def _mu_of(B, F, p):
    return p['mu_0'] + p['mu_B']*B - p['mu_F']*F - p['mu_FF']*F*F


def verify_physics(trajectory, t_grid, params):
    """Descriptive statistics and landmark values."""
    B = trajectory[:, 0]; F = trajectory[:, 1]; A = trajectory[:, 2]
    p = params

    mu_traj = _mu_of(B, F, p)
    mu_min, mu_max = float(np.min(mu_traj)), float(np.max(mu_traj))

    # Landmark: B_crit(0), F_max(1)
    B_crit_0 = abs(p['mu_0']) / p['mu_B']
    disc = p['mu_F']**2 + 4.0 * p['mu_FF'] * (p['mu_B'] * 1.0 + p['mu_0'])
    F_max_1 = ((-p['mu_F'] + math.sqrt(disc)) / (2.0 * p['mu_FF'])
               if disc >= 0 else float('nan'))

    return {
        'B_min': float(B.min()), 'B_max': float(B.max()),
        'F_min': float(F.min()), 'F_max': float(F.max()),
        'A_min': float(A.min()), 'A_max': float(A.max()),
        'B_final': float(B[-1]), 'F_final': float(F[-1]),
        'A_final': float(A[-1]),
        'mu_min': mu_min, 'mu_max': mu_max,
        'mu_crosses_zero': bool(mu_min < 0 < mu_max),
        'landmark_B_crit_0': float(B_crit_0),
        'landmark_F_max_1':  float(F_max_1),
        'A_activated':       bool(A.max() > 0.3),
        'all_finite':        bool(np.all(np.isfinite(trajectory))),
    }


# =========================================================================
# SINGLE PARAMETER SET — all three scenarios share these biology parameters
# =========================================================================

# Reference values from Simulation_Briefing_Potential_Landscape.md.
# Day-unit convention: tau in days; sigma in day^{-1/2}; mu coefficients
# in 1/day or 1/(day * strain^k).
DEFAULT_PARAMS = {
    'tau_B':    14.0,
    'alpha_A':   1.0,
    'tau_F':     7.0,
    'lambda_B':  3.0,
    'lambda_A':  1.5,
    'mu_0':     -0.10,
    'mu_B':      0.30,
    'mu_F':      0.10,
    'mu_FF':     0.40,
    'eta':       0.20,
    'sigma_B':   0.01,
    'sigma_F':   0.005,
    'sigma_A':   0.02,
    'sigma_obs': 0.02,
}

# Shared initial condition — deconditioned baseline with mild residual
# strain.  Deliberately non-zero to avoid numerical corner cases; the
# ball sits deep in the pathological basin regardless.
DEFAULT_INIT = {'B_0': 0.05, 'F_0': 0.10, 'A_0': 0.01}


# =========================================================================
# THREE EXOGENOUS INPUT SCHEDULES
# =========================================================================

# S1 — Sedentary: no intervention.  Trajectory decays to origin.
EXO_SEDENTARY = {
    'T_B_const': 0.0,
    'Phi_1':     0.0,
    'Phi_2':     0.0,
    'T_jump':  1000.0,          # never triggers (> T_end)
    'T_end':    120.0,          # days
}

# S2 — Guided recovery: moderate constant intervention.  Runs long
# enough (200 d) for A to reach healthy plateau given reference mu_B=0.3.
EXO_RECOVERY = {
    'T_B_const': 0.6,
    'Phi_1':     0.03,
    'Phi_2':     0.03,          # no jump
    'T_jump':   1000.0,
    'T_end':    200.0,
}

# S3 — Overtraining cliff: S2 schedule until t=150 d (A has started
# activating), then Phi jumps to 0.20.  Horizon extended to 240 d so
# the post-jump collapse is fully visible.
EXO_OVERTRAINING = {
    'T_B_const': 0.6,
    'Phi_1':     0.03,
    'Phi_2':     0.20,          # catabolic surge
    'T_jump':    150.0,
    'T_end':     240.0,
}


# =========================================================================
# THE MODEL OBJECT
# =========================================================================

FSA_MODEL = SDEModel(
    name="fitness_strain_amplitude",
    version="1.1",

    states=(
        StateSpec("B", 0.0, 1.0),
        StateSpec("F", 0.0, 10.0),       # F >= 0 in principle; cap generous
        StateSpec("A", 0.0, 5.0),        # A >= 0; physical range well below 5
    ),

    drift_fn=drift,
    drift_fn_jax=drift_jax,
    diffusion_type=DIFFUSION_DIAGONAL_STATE,
    diffusion_fn=diffusion_diagonal,
    noise_scale_fn=noise_scale_fn,
    noise_scale_fn_jax=noise_scale_fn_jax,
    make_aux_fn=make_aux,
    make_aux_fn_jax=make_aux_jax,
    make_y0_fn=make_y0,

    channels=(
        ChannelSpec("obs_B", depends_on=(), generate_fn=gen_obs_B),
        ChannelSpec("obs_F", depends_on=(), generate_fn=gen_obs_F),
        ChannelSpec("obs_A", depends_on=(), generate_fn=gen_obs_A),
        ChannelSpec("T_B",   depends_on=(), generate_fn=gen_T_B_channel),
        ChannelSpec("Phi",   depends_on=(), generate_fn=gen_Phi_channel),
    ),

    plot_fn=plot_fsa,
    verify_physics_fn=verify_physics,

    # ONE parameter set, ONE initial state, THREE input schedules.
    param_sets={
        'sedentary':    DEFAULT_PARAMS,
        'recovery':     DEFAULT_PARAMS,
        'overtraining': DEFAULT_PARAMS,
    },
    init_states={
        'sedentary':    DEFAULT_INIT,
        'recovery':     DEFAULT_INIT,
        'overtraining': DEFAULT_INIT,
    },
    exogenous_inputs={
        'sedentary':    EXO_SEDENTARY,
        'recovery':     EXO_RECOVERY,
        'overtraining': EXO_OVERTRAINING,
    },
)
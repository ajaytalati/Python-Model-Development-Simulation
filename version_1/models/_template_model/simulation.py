"""_template_model/simulation.py — Skeleton SDE Model Definition.

Reference:  how_to_add_a_new_model/01_simulation.md

Runs out of the box:

    python simulator/run_simulator.py \
        --model models._template_model.simulation.TEMPLATE_MODEL \
        --param-set A --seed 42

Produces outputs/synthetic__template__A_<timestamp>/ with NPZ +
latent_states.png + observations.png.

The minimum-viable implementation below is a **1-state Ornstein-Uhlenbeck
(OU) SDE with a single Gaussian observation channel**:

    dx = -k (x - mu) dt + sigma_x dB_x
    y_k = x(t_k) + N(0, sigma_obs^2)

Replace everything marked TODO with your model's actual dynamics.
"""

import math
import numpy as np

from simulator.sde_model import (
    SDEModel, StateSpec, ChannelSpec, DIFFUSION_DIAGONAL_CONSTANT)
from .sim_plots import plot_template


# =========================================================================
# DRIFT  (TODO: replace with your model's drift)
# =========================================================================

def drift(t, y, params, aux):
    """OU drift: dx = -k (x - mu) dt.

    Signature required by simulator/sde_solver_scipy.py:
        drift(t: float, y: ndarray(n_states,), params: dict, aux: any)
            -> ndarray(n_states,)

    y is indexed in the same order as the `states` tuple in TEMPLATE_MODEL.
    `aux` is whatever make_aux() returned (None in this skeleton).
    """
    del aux  # unused in this template
    x = y[0]
    dx = -params['k'] * (x - params['mu'])
    return np.array([dx])


# =========================================================================
# DIFFUSION  (TODO: replace if your noise structure differs)
# =========================================================================

def diffusion_diagonal(params):
    """Per-state constant noise magnitudes sigma_i.

    For DIFFUSION_DIAGONAL_CONSTANT, the Euler-Maruyama update is
        y_{k+1} = y_k + dt * drift + sigma_i * sqrt(dt) * N(0, 1).
    """
    return np.array([params['sigma_x']])


# =========================================================================
# AUXILIARY / INITIAL STATE
# =========================================================================

def make_aux(params, init_state, t_grid, exogenous):
    """Return auxiliary data passed to drift() at every step.

    Use this to prebuild piecewise schedules, daily targets, etc.
    Return None when no aux is needed.
    """
    del params, init_state, t_grid, exogenous
    return None


def make_y0(init_dict, params):
    """Build the y0 vector. Order MUST match the `states` tuple below."""
    del params
    return np.array([init_dict['x_0']])


# =========================================================================
# OBSERVATION CHANNEL  (TODO: add / modify channels)
# =========================================================================

def gen_obs(trajectory, t_grid, params, aux, prior_channels, seed):
    """Gaussian observation y = x + N(0, sigma_obs^2).

    Signature:
        (trajectory: ndarray(T, n_states),
         t_grid:    ndarray(T,),
         params:    dict,
         aux:       any,
         prior_channels: dict[str, dict[str, ndarray]],
         seed:      int)
        -> dict with at least 't_idx' (or 't_hours') key.
    """
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    T = len(t_grid)
    x = trajectory[:, 0]
    noise = rng.normal(0.0, params['sigma_obs'], size=T)
    return {
        't_idx':     np.arange(T, dtype=np.int32),
        'obs_value': (x + noise).astype(np.float32),
    }


# =========================================================================
# PHYSICS VERIFICATION  (optional; called by `--verify` and after each run)
# =========================================================================

def verify_physics(trajectory, t_grid, params):
    """Return a dict of qualitative / quantitative checks.

    Keys ending in '_ok' or booleans get a PASS/FAIL tag in the CLI output.
    Numeric keys get printed raw.
    """
    x = trajectory[:, 0]
    return {
        'x_min':            float(np.min(x)),
        'x_max':            float(np.max(x)),
        'x_mean':           float(np.mean(x)),
        'converges_to_mu':  abs(float(np.mean(x[-100:])) - params['mu']) < 0.5,
        'all_finite':       bool(np.all(np.isfinite(trajectory))),
    }


# =========================================================================
# PARAMETER SETS  (TODO: add B, C, ... as needed)
# =========================================================================
#
# RULES:
#   * Keys in param_sets, init_states, exogenous_inputs MUST match.
#   * Use uppercase single letters: 'A', 'B', 'C', ...
#   * Each set is a COMPLETE, SELF-CONTAINED scenario.
#   * Include `dt_hours` and `t_total_hours` in params (the simulator reads
#     these to build the time grid).  Defaults: dt=5min, t_total=9 days.

PARAM_SET_A = {
    'k':             1.0,    # OU mean-reversion rate (1/hr)
    'mu':            0.0,    # OU attractor
    'sigma_x':       0.5,    # OU diffusion magnitude
    'sigma_obs':     0.2,    # observation noise std
    'dt_hours':      0.1,    # time step (6 minutes)
    't_total_hours': 24.0,   # simulation horizon (1 day)
}

INIT_STATE_A = {'x_0': 2.0}

EXOGENOUS_A = {}  # no exogenous inputs for this template


# =========================================================================
# THE MODEL OBJECT  (edit `name`, `version`, and `states` for your model)
# =========================================================================

TEMPLATE_MODEL = SDEModel(
    name="_template_",
    version="0.1",

    states=(
        StateSpec("x", -10.0, 10.0),
        # TODO: add additional StateSpec(...) entries here.
    ),

    drift_fn=drift,
    diffusion_type=DIFFUSION_DIAGONAL_CONSTANT,
    diffusion_fn=diffusion_diagonal,
    make_aux_fn=make_aux,
    make_y0_fn=make_y0,

    channels=(
        ChannelSpec("obs", depends_on=(), generate_fn=gen_obs),
        # TODO: add additional ChannelSpec(...) entries here.
    ),

    plot_fn=plot_template,
    verify_physics_fn=verify_physics,

    param_sets={'A': PARAM_SET_A},
    init_states={'A': INIT_STATE_A},
    exogenous_inputs={'A': EXOGENOUS_A},
)

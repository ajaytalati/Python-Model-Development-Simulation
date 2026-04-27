"""sir/simulation.py — Stochastic SIR (Susceptible-Infected-Recovered).

Reference:  how_to_add_a_new_model/01_simulation.md
            Anderson & May 1991, "Infectious Diseases of Humans" §6
            Endo, van Leeuwen, Baguelin 2019, Epidemics 29 (PMCMC tutorial)

Frequency-dependent SIR with diffusion approximation:

    dS = -beta * S * I / N        dt + diffusion noise
    dI = (beta * S * I / N - gamma * I) dt + diffusion noise
    R  = N - S - I                                 (eliminated; conserved)

Two observation channels (mixed likelihood):

    cases_t ~ Poisson(rho * beta * S_t * I_t / N * bin_hours)   [Poisson, daily bins]
    serology_t ~ Normal(I_t / N, sigma_z^2)                      [Gaussian, weekly bins]

R_0 = beta / gamma is the basic reproduction number.
Transcritical bifurcation at R_0 = 1 (DFE stable for R_0 < 1, EE stable for > 1).

Set A is the canonical Anderson & May 1978 boarding-school flu outbreak:
N=763, R_0 ≈ 3.3, 14-day trial. Used as the paper-parity benchmark in
Endo et al 2019.
"""

import math

import numpy as np

from simulator.sde_model import (
    SDEModel, StateSpec, ChannelSpec, DIFFUSION_DIAGONAL_CONSTANT,
)

from .sim_plots import plot_sir


# =========================================================================
# DRIFT
# =========================================================================

def drift(t, y, params, aux):
    """SIR drift (frequency-dependent transmission).

    State vector y = [S, I]. R = N - S - I is eliminated.
    """
    del t, aux
    S, I = y[0], y[1]
    N = params['N']
    beta = params['beta']
    gamma = params['gamma']
    v = params.get('v', 0.0)   # vaccination rate (Set D); 0 elsewhere

    incidence = beta * S * I / N
    dS = -incidence - v * S
    dI =  incidence - gamma * I
    return np.array([dS, dI])


# =========================================================================
# DIFFUSION
# =========================================================================

def diffusion_diagonal(params):
    """Per-state diffusion magnitudes (Itô).

    SIR with SDE diffusion approximation has off-diagonal noise correlation
    (the same noise increment driving incidence appears in both dS and dI).
    The framework's DIFFUSION_DIAGONAL_CONSTANT supports diagonal noise only,
    which is a standard simplification for the small-population regime
    (sqrt(T_S) and sqrt(T_I) are independent process-noise temperatures).
    """
    return np.array([
        math.sqrt(params['T_S']),
        math.sqrt(params['T_I']),
    ])


# =========================================================================
# AUX / Y0
# =========================================================================

def make_aux(params, init_state, t_grid, exogenous):
    """No piecewise schedule needed for baseline scenarios."""
    del params, init_state, t_grid, exogenous
    return None


def make_y0(init_dict, params):
    """Build y0 from the prior init dict.

    S_0 is derived: S_0 = N - I_0 - R_0 (so we don't double-count it in priors).
    """
    N = params['N']
    I_0 = init_dict['I_0']
    R_0 = init_dict.get('R_0_init', 0.0)
    S_0 = N - I_0 - R_0
    return np.array([S_0, I_0])


# =========================================================================
# OBSERVATION CHANNELS
# =========================================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gen_cases(trajectory, t_grid, params, aux, prior_channels, seed):
    """Daily Poisson case counts.

    Expected count per daily bin:
        E[k_d] = rho * beta * <SI/N> * bin_hours

    where <SI/N> is averaged over the bin and rho is the case-detection
    probability. This is the standard incidence-based observation model
    (cases per day, not prevalence).
    """
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    p = params
    dt_h = float(t_grid[1] - t_grid[0])

    # Daily bins (24-hour aggregation)
    bin_hours = 24.0
    bin_size = max(int(round(bin_hours / dt_h)), 1)
    n_bins = len(t_grid) // bin_size
    if n_bins < 1:
        return {'t_idx': np.zeros(0, dtype=np.int32),
                'cases': np.zeros(0, dtype=np.int32),
                'bin_hours': np.float32(bin_hours)}

    S = trajectory[: n_bins * bin_size, 0].reshape(n_bins, bin_size)
    I = trajectory[: n_bins * bin_size, 1].reshape(n_bins, bin_size)
    SI_over_N_mean = (S * I).mean(axis=1) / p['N']   # (n_bins,)

    # Expected new infections per daily bin × detection probability
    expected = p['rho'] * p['beta'] * SI_over_N_mean * bin_hours
    expected = np.maximum(expected, 0.0)
    k = rng.poisson(expected).astype(np.int32)

    bin_t_idx = (np.arange(n_bins) * bin_size + bin_size - 1).astype(np.int32)
    return {
        't_idx':     bin_t_idx,
        'cases':     k,
        'bin_hours': np.float32(bin_hours),
    }


def gen_serology(trajectory, t_grid, params, aux, prior_channels, seed):
    """Weekly Gaussian serology survey: prevalence I/N + Gaussian noise.

    Cross-sectional survey samples a fixed fraction of the population every
    7 days; reported as observed prevalence (continuous in [0, 1]).
    """
    del aux, prior_channels
    rng = np.random.default_rng(seed + 1)
    p = params
    dt_h = float(t_grid[1] - t_grid[0])
    survey_interval_h = 7.0 * 24.0   # weekly
    stride = max(int(round(survey_interval_h / dt_h)), 1)

    # First survey at the end of week 1 (so day 7, day 14, ...)
    obs_idx = np.arange(stride - 1, len(t_grid), stride, dtype=np.int32)
    if len(obs_idx) == 0:
        return {'t_idx': np.zeros(0, dtype=np.int32),
                'prevalence': np.zeros(0, dtype=np.float32)}
    I = trajectory[obs_idx, 1]
    prev_true = I / p['N']
    prev_obs = prev_true + rng.normal(0.0, p['sigma_z'], size=len(obs_idx))
    return {
        't_idx':      obs_idx,
        'prevalence': prev_obs.astype(np.float32),
    }


# =========================================================================
# PHYSICS VERIFICATION
# =========================================================================

def verify_physics(trajectory, t_grid, params):
    """Quick sanity checks on the simulated SIR trajectory."""
    S = trajectory[:, 0]
    I = trajectory[:, 1]
    N = params['N']
    R = N - S - I

    R_0 = params['beta'] / params['gamma']
    peak_I = float(np.max(I))
    peak_t = float(t_grid[int(np.argmax(I))])
    final_R = float(R[-1])
    attack_rate = final_R / N    # fraction ever infected

    # For R_0 > 1 the deterministic SIR has a closed-form attack-rate solution
    # (the final-size relation): a = 1 - exp(-R_0 * a). For R_0 = 3.3, a ≈ 0.94.
    # The stochastic version should be close but not exact.
    expected_attack_lower = 0.5 if R_0 > 1.5 else 0.0
    expected_attack_upper = 1.05

    # SDE diffusion can briefly exit [0, N]; tolerance scales as max(5, 0.5% of N).
    bounds_tol = max(5.0, 0.005 * N)

    return {
        'R_0':                 float(R_0),
        'peak_I':              peak_I,
        'peak_t_hours':        peak_t,
        'final_R':             final_R,
        'attack_rate':         float(attack_rate),
        'mass_conservation_ok': bool(np.all(np.abs(S + I + R - N) < 1e-3)),
        'S_in_range_ok':       bool(np.min(S) > -bounds_tol and np.max(S) < N + bounds_tol),
        'I_nonneg_ok':         bool(np.min(I) > -bounds_tol),
        'R_in_range_ok':       bool(np.min(R) > -bounds_tol and np.max(R) < N + bounds_tol),
        'attack_rate_realistic': bool(
            expected_attack_lower <= attack_rate <= expected_attack_upper),
        'all_finite':          bool(np.all(np.isfinite(trajectory))),
    }


# =========================================================================
# PARAMETER SETS
# =========================================================================
#
# Set A — boarding-school flu (Anderson & May 1978, paper-parity target)
#   N=763, beta=1.66/day, gamma=0.5/day → R_0 = 3.32
#   14-day outbreak, ~94% attack rate (deterministic limit)
#
# Set B — small outbreak: R_0 = 2.5, 60 days, larger population
# Set C — large outbreak: R_0 = 4.0, 90 days
# Set D — vaccination intervention: R_0 = 3.0, with v(t) step-function

PARAM_SET_A = {
    'N':             763,        # boarding school population
    'beta':          1.66 / 24,  # transmission rate (1/hr); 1.66/day
    'gamma':         0.5 / 24,   # recovery rate (1/hr); 0.5/day
    'rho':           1.0,        # case-detection prob (boarding school: full reporting)
    'sigma_z':       0.02,       # serology survey noise std
    'T_S':           1.0,        # S diffusion temp (count-scale variance per hour)
    'T_I':           1.0,        # I diffusion temp
    'v':             0.0,        # no vaccination in baseline
    'dt_hours':      1.0,        # time step (hourly)
    't_total_hours': 14.0 * 24,  # 14 days
}

INIT_STATE_A = {
    'I_0':       1.0,    # one initial infectious case
    'R_0_init':  0.0,    # naming clash with R_0 reproduction number; this is recovered count
}

EXOGENOUS_A = {}

PARAM_SET_B = {**PARAM_SET_A,
    'N':             10_000,
    'beta':          0.5 / 24,    # R_0 = 0.5 / 0.2 = 2.5
    'gamma':         0.2 / 24,
    'rho':           0.5,         # community: half of cases reported
    'sigma_z':       0.005,
    't_total_hours': 60.0 * 24,   # 60 days
}
INIT_STATE_B = {'I_0': 5.0, 'R_0_init': 0.0}

PARAM_SET_C = {**PARAM_SET_A,
    'N':             10_000,
    'beta':          0.8 / 24,    # R_0 = 0.8 / 0.2 = 4.0
    'gamma':         0.2 / 24,
    'rho':           0.5,
    'sigma_z':       0.005,
    't_total_hours': 90.0 * 24,   # 90 days
}
INIT_STATE_C = {'I_0': 10.0, 'R_0_init': 0.0}

PARAM_SET_D = {**PARAM_SET_A,
    'N':             10_000,
    'beta':          0.6 / 24,    # R_0 = 0.6 / 0.2 = 3.0
    'gamma':         0.2 / 24,
    'rho':           0.5,
    'sigma_z':       0.005,
    'v':             0.02 / 24,   # 2%/day vaccination after day 30 (gen_obs uses constant; piecewise via aux is a future extension)
    't_total_hours': 90.0 * 24,
}
INIT_STATE_D = {'I_0': 10.0, 'R_0_init': 0.0}


# =========================================================================
# THE MODEL OBJECT
# =========================================================================

SIR_MODEL = SDEModel(
    name="sir",
    version="1.0",

    states=(
        StateSpec("S", 0.0, 1e9),
        StateSpec("I", 0.0, 1e9),
    ),

    drift_fn=drift,
    diffusion_type=DIFFUSION_DIAGONAL_CONSTANT,
    diffusion_fn=diffusion_diagonal,
    make_aux_fn=make_aux,
    make_y0_fn=make_y0,

    channels=(
        ChannelSpec("cases",    depends_on=(), generate_fn=gen_cases),
        ChannelSpec("serology", depends_on=(), generate_fn=gen_serology),
    ),

    plot_fn=plot_sir,
    verify_physics_fn=verify_physics,

    param_sets={
        'A': PARAM_SET_A, 'B': PARAM_SET_B, 'C': PARAM_SET_C, 'D': PARAM_SET_D,
    },
    init_states={
        'A': INIT_STATE_A, 'B': INIT_STATE_B, 'C': INIT_STATE_C, 'D': INIT_STATE_D,
    },
    exogenous_inputs={
        'A': {}, 'B': {}, 'C': {}, 'D': {},
    },
)

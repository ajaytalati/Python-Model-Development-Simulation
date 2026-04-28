"""glucose_insulin/simulation.py — Bergman minimal model (3-state SDE).

Reference:
    Bergman, R. N., Ider, Y. Z., Bowden, C. R., & Cobelli, C. (1979).
    Quantitative estimation of insulin sensitivity. *American Journal of
    Physiology — Endocrinology and Metabolism* 236, E667-E677.

The three-compartment glucose / remote-insulin / plasma-insulin model
that has anchored physiological filtering and control of T1D / T2D
since 1979. Used as the dynamical core of the Tandem Control-IQ /
Medtronic 780G / OpenAPS closed-loop pumps and the FDA-approved
UVA/Padova T1DMS simulator (in extended form).

Math (frequency-dependent, in /hour units throughout):

    dG/dt = -p₁(G - Gb) - X·G + D(t)/(V_G·BW)         [glucose, mg/dL]
    dX/dt = -p₂·X + p₃·(I - Ib)                       [remote insulin action, 1/hr]
    dI/dt = -k(I - Ib) + I_input(t)/V_I                [plasma insulin, μU/mL]

Observations (mixed likelihood — exercises the same code path SWAT uses):

    cgm_t  ~ Normal(G_t, σ_CGM²)          every 5 minutes (Dexcom-class cadence)
    carb_m ~ Poisson(μ_carb_m)            at each meal time (3 / day)

Known exogenous inputs (not estimated):
    Meal schedule:    3 meals/day at 08:00, 13:00, 19:00 (with day-to-day jitter)
    Meal absorption:  D(t) — gamma-2 gastric-emptying profile per meal
    Insulin schedule: I_input(t) — fixed open-loop bolus (Set D only); zero on A/B/C

Insulin sensitivity index SI = p₃ / p₂ — the *clinically* identifiable
combination, the canonical inference target since 1979.

Parameter sets:
    A — Healthy adult (paper-parity to Bergman 1979 healthy cohort means)
    B — Insulin-resistance / pre-T2D (p₃ halved → SI drops 50%)
    C — T1D no-control (Ib=0, no exogenous insulin → glucose climbs through DKA range)
    D — T1D with open-loop insulin (Ib=0, fixed bolus + basal at carb-counted ratio)
"""

import math

import numpy as np

from simulator.sde_model import (
    SDEModel, StateSpec, ChannelSpec, DIFFUSION_DIAGONAL_CONSTANT,
)

from .sim_plots import plot_glucose_insulin


# =========================================================================
# MEAL SCHEDULE & ABSORPTION
# =========================================================================
#
# Standard 3-meal day: 08:00, 13:00, 19:00 (2.0% of 24h jitter via the seed).
# Each meal absorbs via a gamma-2 (Erlang) profile with τ_g = 0.5 hr peak time
# — the "single-pool" simplification of Dalla Man 2006 used widely in Bergman-
# minimal-model studies.

_MEAL_HOURS_BASE = (8.0, 13.0, 19.0)
_TAU_GASTRIC = 0.5     # hours, half-life of gastric emptying


def _meal_schedule(seed, n_days, meal_carbs_g):
    """Return list of (meal_time_hr, meal_carbs_g) for the trial."""
    rng = np.random.default_rng(seed + 100)
    schedule = []
    for d in range(n_days):
        for base_h in _MEAL_HOURS_BASE:
            jitter = rng.uniform(-0.5, 0.5)   # ±30 min jitter
            t_h = d * 24.0 + base_h + jitter
            carbs = float(rng.normal(meal_carbs_g, 0.15 * meal_carbs_g))
            carbs = max(carbs, 5.0)            # floor for sanity
            schedule.append((t_h, carbs))
    return schedule


def _meal_absorption_rate(t_hr, schedule, V_G, BW):
    """D(t) / (V_G · BW) summed over all meals in the schedule.

    Gamma-2 profile per meal:
        D_m(t) = (carbs_m / τ²) · (t - t_meal) · exp(-(t-t_meal)/τ)

    Units: carbs in g, t in hr, τ in hr → D_m has units g/hr.
    Divided by (V_G [dL/kg] · BW [kg]) gives mg/(dL·hr) when carbs are
    converted to mg (× 1000) — done outside.
    """
    rate = 0.0
    for t_meal, carbs in schedule:
        dt = t_hr - t_meal
        if dt < 0:
            continue
        rate += (carbs / _TAU_GASTRIC ** 2) * dt * math.exp(-dt / _TAU_GASTRIC)
    return rate * 1000.0 / (V_G * BW)   # convert g→mg, divide by V_G·BW


# =========================================================================
# INSULIN SCHEDULE (Set D only)
# =========================================================================

def _insulin_schedule(seed, n_days, meal_schedule, insulin_carb_ratio,
                      basal_rate_U_hr, V_I_BW):
    """Open-loop insulin schedule: bolus at each meal + constant basal.

    Returns list of (t_hr, I_input_per_hr_at_t) for evaluation. Units of
    I_input are μU/mL/hr (= rate of plasma-insulin appearance after dividing
    by V_I).
    """
    if not meal_schedule:
        return []
    boluses = []
    for t_meal, carbs in meal_schedule:
        # Insulin units to dose: carbs / insulin-to-carb ratio.
        # Bolus delivered as a 0.5-hr exponential.
        units = carbs / insulin_carb_ratio
        boluses.append((t_meal, units))
    # We'll evaluate insulin_rate(t) on demand.
    return {
        'boluses': boluses,
        'basal_rate_U_hr': basal_rate_U_hr,
        'V_I_BW': V_I_BW,
    }


def _insulin_input_rate(t_hr, sched):
    """μU/mL/hr at time t."""
    if not sched:
        return 0.0
    rate = sched['basal_rate_U_hr']     # constant basal
    tau_b = 0.5                          # bolus absorption τ (hours)
    for t_bolus, units in sched['boluses']:
        dt = t_hr - t_bolus
        if dt < 0:
            continue
        rate += (units / tau_b ** 2) * dt * math.exp(-dt / tau_b)
    # Convert U/hr → μU/hr (×1e6) → μU/mL/hr by dividing by V_I·BW (mL).
    # V_I_BW is in dL: multiply by 100 to get mL.
    V_I_mL = sched['V_I_BW'] * 100.0
    return rate * 1e6 / V_I_mL


# =========================================================================
# DRIFT
# =========================================================================

def drift(t, y, params, aux):
    """Bergman minimal model drift with endogenous β-cell secretion.

    The original Bergman 1979 minimal model is calibrated against IV-
    glucose / IV-insulin tolerance tests where endogenous insulin
    secretion is bypassed. For daily-life meal-tolerance scenarios we
    use the Bergman 1981 extended form which adds a β-cell secretion
    term ``n · max(G - h, 0)``: insulin secretion rate is linear in the
    glucose excursion above a threshold ``h``. T1D scenarios set ``n = 0``.

    All quantities in /hour units.
    """
    G, X, I = y[0], y[1], y[2]
    p = params
    # aux carries pre-built schedules for this trial
    meal_sched = aux['meal_schedule']
    insulin_sched = aux['insulin_schedule']
    V_G = p['V_G']
    BW = p['BW']

    D_rate = _meal_absorption_rate(t, meal_sched, V_G, BW)
    I_rate = _insulin_input_rate(t, insulin_sched)
    secretion = p.get('n_beta', 0.0) * max(G - p.get('h_beta', p['Gb']), 0.0)

    dG = -p['p1'] * (G - p['Gb']) - X * G + D_rate
    dX = -p['p2'] * X + p['p3'] * max(I - p['Ib'], 0.0)
    dI = -p['k'] * (I - p['Ib']) + secretion + I_rate
    return np.array([dG, dX, dI])


# =========================================================================
# DIFFUSION
# =========================================================================

def diffusion_diagonal(params):
    """Itô diagonal: σ_G, σ_X, σ_I — small process-noise temperatures.

    For Bergman minimal at the typical glucose scale (~100 mg/dL), σ_G ~ 1
    mg/dL/√hr is tiny (signal/noise > 100 at peak excursion). The model is
    near-deterministic; the SDE form is mostly to allow the framework's
    diffusion-aware inference machinery to work.
    """
    return np.array([
        math.sqrt(params['T_G']),
        math.sqrt(params['T_X']),
        math.sqrt(params['T_I']),
    ])


# =========================================================================
# AUX / Y0
# =========================================================================

def make_aux(params, init_state, t_grid, exogenous):
    """Pre-build the meal + insulin schedules for the full trial."""
    del exogenous
    p = params
    n_days = int(round(p['t_total_hours'] / 24.0))
    meal_carbs_g = p.get('meal_carbs_g', 50.0)
    seed = int(p.get('schedule_seed', 0))

    meal_sched = _meal_schedule(seed, n_days, meal_carbs_g)

    insulin_sched = None
    if p.get('insulin_schedule_active', False):
        V_I_BW = p['V_I'] * p['BW']         # dL
        insulin_sched = _insulin_schedule(
            seed, n_days, meal_sched,
            insulin_carb_ratio=p.get('insulin_carb_ratio', 10.0),
            basal_rate_U_hr=p.get('basal_rate_U_hr', 0.5),
            V_I_BW=V_I_BW,
        )

    return {'meal_schedule': meal_sched,
            'insulin_schedule': insulin_sched}


def make_y0(init_dict, params):
    """y0 = [G_0, X_0, I_0]."""
    del params
    return np.array([
        init_dict['G_0'],
        init_dict.get('X_0', 0.0),
        init_dict['I_0'],
    ])


# =========================================================================
# OBSERVATION CHANNELS
# =========================================================================

def gen_cgm(trajectory, t_grid, params, aux, prior_channels, seed):
    """CGM Gaussian obs every 5-minute bin (matches Dexcom G6 cadence)."""
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    G = trajectory[:, 0]
    sigma_cgm = params['sigma_cgm']
    cgm = G + rng.normal(0.0, sigma_cgm, size=len(t_grid))
    return {
        't_idx':     np.arange(len(t_grid), dtype=np.int32),
        'cgm_value': cgm.astype(np.float32),
    }


def gen_meal_carbs(trajectory, t_grid, params, aux, prior_channels, seed):
    """Poisson observation of patient-logged carb count at each meal time.

    The simulator's meal schedule has TRUTH carb counts (e.g., 50g normal).
    The observation is a noisy Poisson(truth) realisation — patient logs are
    notoriously imprecise. This is the Poisson channel exercising the
    framework's mixed-likelihood code path.
    """
    del trajectory, prior_channels
    rng = np.random.default_rng(seed + 1)
    meal_sched = aux['meal_schedule']
    if not meal_sched:
        return {'t_idx': np.zeros(0, dtype=np.int32),
                'carbs_g': np.zeros(0, dtype=np.int32)}

    dt_h = float(t_grid[1] - t_grid[0])
    t_idx = []
    carbs = []
    for t_meal, c_truth in meal_sched:
        idx = int(round(t_meal / dt_h))
        if 0 <= idx < len(t_grid):
            t_idx.append(idx)
            # Poisson observation of integer carb count
            carbs.append(int(rng.poisson(max(c_truth, 1.0))))

    return {
        't_idx':   np.array(t_idx, dtype=np.int32),
        'carbs_g': np.array(carbs, dtype=np.int32),
    }


# =========================================================================
# PHYSICS VERIFICATION
# =========================================================================

def verify_physics(trajectory, t_grid, params):
    """Sanity checks: physiological ranges, peak excursion, return-to-basal."""
    G = trajectory[:, 0]
    I = trajectory[:, 2]

    G_min = float(np.min(G))
    G_max = float(np.max(G))
    G_mean = float(np.mean(G))
    G_final = float(G[-1])
    Gb = params['Gb']
    Ib = params['Ib']

    # Set-A-style healthy expectations: peak ≤ 200 mg/dL post-meal,
    # mean ~ basal, no DKA-range excursions.
    is_t1d_no_control = (Ib == 0.0
                         and not params.get('insulin_schedule_active', False))

    return {
        'G_min':                float(G_min),
        'G_max':                float(G_max),
        'G_mean':               float(G_mean),
        'G_final':              float(G_final),
        'I_max':                float(np.max(I)),
        'in_physiological_range_ok': bool(
            (40.0 if not is_t1d_no_control else 40.0)
            <= G_min and G_max <= (450.0 if is_t1d_no_control else 250.0)),
        'peak_post_meal_realistic': bool(
            (G_max <= 200.0) if not is_t1d_no_control else (G_max >= 200.0)),
        'all_finite':           bool(np.all(np.isfinite(trajectory))),
    }


# =========================================================================
# PARAMETER SETS
# =========================================================================
#
# Bergman 1979 healthy-cohort means (converted to /hour units):
#   p₁ ≈ 0.030/min × 60 = 1.8/hr
#   p₂ ≈ 0.025/min × 60 = 1.5/hr
#   p₃ ≈ 1.3e-5 /(min²·μU/mL) × 3600 = 4.68e-2 /(hr²·μU/mL)
#   k  ≈ 0.30/min × 60 = 18/hr     (insulin clearance)
#   Gb ≈ 90 mg/dL,  Ib ≈ 7 μU/mL
#   V_G ≈ 1.6 dL/kg,  V_I ≈ 1.2 dL/kg,  BW ≈ 70 kg

PARAM_SET_A = {
    # Estimable
    'p1':            1.8,        # /hr (glucose effectiveness)
    'p2':            1.5,        # /hr (remote insulin decay)
    'p3':            4.68e-2,    # /(hr²·μU/mL)
    'k':             18.0,       # /hr (plasma insulin clearance)
    'Gb':            90.0,       # mg/dL
    'sigma_cgm':     8.0,        # mg/dL (Dexcom G6 spec)
    'T_G':           1.0,        # diffusion temp (mg/dL)²/hr — small
    # Frozen
    'Ib':            7.0,        # μU/mL (basal insulin)
    'V_G':           1.6,        # dL/kg
    'V_I':           1.2,        # dL/kg
    'BW':            70.0,       # kg
    'T_X':           1e-6,       # near-zero process noise on X
    'T_I':           0.5,        # μU²/mL²/hr
    'n_beta':        8.0,        # β-cell secretion rate (Bergman 1981 ext.)
    'h_beta':        90.0,       # secretion threshold (= Gb for healthy)
    'meal_carbs_g':  40.0,       # truth meal size (typical mixed meal)
    'schedule_seed': 0,
    # Insulin schedule (Set D activates this)
    'insulin_schedule_active': False,
    'dt_hours':      5.0 / 60.0, # 5-min bins
    't_total_hours': 24.0,       # 1 day
}

INIT_STATE_A = {'G_0': 90.0, 'X_0': 0.0, 'I_0': 7.0}

EXOGENOUS_A = {}


PARAM_SET_B = {**PARAM_SET_A,
    'p3':  4.68e-2 * 0.5,        # SI halved (insulin resistance)
}
INIT_STATE_B = INIT_STATE_A


PARAM_SET_C = {**PARAM_SET_A,
    # T1D no-control: zero basal insulin, no endogenous secretion
    'Ib':     0.0,
    'n_beta': 0.0,               # β-cells destroyed
    'k':      18.0,
    'insulin_schedule_active': False,
}
INIT_STATE_C = {'G_0': 110.0, 'X_0': 0.0, 'I_0': 0.0}


PARAM_SET_D = {**PARAM_SET_C,
    'insulin_schedule_active': True,
    'insulin_carb_ratio':      10.0,    # 1 U per 10 g carbs (typical)
    'basal_rate_U_hr':         0.5,     # 0.5 U/hr basal
}
INIT_STATE_D = INIT_STATE_C


# =========================================================================
# THE MODEL OBJECT
# =========================================================================

GLUCOSE_INSULIN_MODEL = SDEModel(
    name="glucose_insulin",
    version="1.0",

    states=(
        StateSpec("G",  0.0, 600.0),    # mg/dL — physiological
        StateSpec("X",  0.0,   2.0),    # 1/hr — remote insulin action
        StateSpec("I",  0.0, 500.0),    # μU/mL
    ),

    drift_fn=drift,
    diffusion_type=DIFFUSION_DIAGONAL_CONSTANT,
    diffusion_fn=diffusion_diagonal,
    make_aux_fn=make_aux,
    make_y0_fn=make_y0,

    channels=(
        ChannelSpec("cgm",         depends_on=(), generate_fn=gen_cgm),
        ChannelSpec("meal_carbs",  depends_on=(), generate_fn=gen_meal_carbs),
    ),

    plot_fn=plot_glucose_insulin,
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

"""models/swat/sim_plots.py — SWAT diagnostic plots.

Date:    20 April 2026
Version: 1.0

Produces:
  1. latent_states.png  - W, Zt, a, T, C, V_h/V_n panels
  2. observations.png   - HR trace and binary sleep strip
  3. entrainment.png    - Entrainment quality E(t) and bifurcation parameter
                          mu(E) over time, plus the computed T* equilibrium
                          and actual T trajectory for comparison
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sigmoid(x):
    """Numerically stable sigmoid for numpy arrays."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


# =========================================================================
# PUBLIC ENTRY POINT
# =========================================================================

def plot_swat(trajectory, t_grid, channel_outputs, params, save_dir):
    """Generate diagnostic plots for the SWAT SDE."""
    os.makedirs(save_dir, exist_ok=True)

    p1 = os.path.join(save_dir, "latent_states.png")
    _plot_latent(trajectory, t_grid, params, p1)
    print(f"  Plot: {p1}")

    p2 = os.path.join(save_dir, "observations.png")
    _plot_observations(trajectory, t_grid, channel_outputs, params, p2)
    print(f"  Plot: {p2}")

    p3 = os.path.join(save_dir, "entrainment.png")
    _plot_entrainment(trajectory, t_grid, params, p3)
    print(f"  Plot: {p3}")


# =========================================================================
# ENTRAINMENT COMPUTATION  —  amplitude × phase form (PLOT-SIDE PROTOTYPE)
# =========================================================================
#
# IMPORTANT: this plot-side entrainment formula is CURRENTLY DIFFERENT FROM
# the one used inside the SDE dynamics (_dynamics.entrainment_quality,
# simulation.drift, simulation.drift_jax).  The dynamics still use the old
# instantaneous "slow backdrop" formula; the plot here shows the new
# amplitude × phase measurement.
#
# This is a deliberate prototype-stage inconsistency.  Once the new formula
# is confirmed to give clean basin discrimination on the plots, the dynamics
# will be refactored to carry 24h running statistics so the same amplitude ×
# phase quantity can drive mu(E) online.
#
# Until then, interpret entrainment.png panel 1 (E) as a diagnostic view of
# the TRUE entrainment the patient is experiencing; panels 2-3 (mu, T)
# still reflect the dynamics-side formula and may not exactly match the
# plotted E.  Set-to-set discrimination should still be visible.
#
# =========================================================================

def _compute_E(trajectory, t_grid, params):
    """Entrainment quality E(t) from rhythm amplitude and phase alignment.

    For each time t, computes over the preceding 24-hour window:
        amp_W  = (W_max - W_min) / 1                in [0, 1]
        amp_Z  = (Zt_max - Zt_min) / A_SCALE        in [0, 1]
        phase_W = max(corr(W, C(t)),  0)            in [0, 1]
        phase_Z = max(corr(Zt, -C(t)), 0)           in [0, 1]

    Returns
        E = (amp_W * phase_W) * (amp_Z * phase_Z)   in [0, 1]

    Detects two pathologies independently:
      - flat swing (one state not alternating): amp -> 0
      - phase shift (rhythm out of sync with body clock): phase -> 0
    Either failure alone pulls E down; both failing drives E to ~0.

    For t < 24 h the window is partial; E is computed from whatever
    history is available (including the synthetic initial-condition
    transient). Expect the first day's E to be unreliable; all
    downstream checks use day >= 1.
    """
    W  = trajectory[:, 0]
    Zt = trajectory[:, 1]
    # Reference is the EXTERNAL light cycle (objective sun).  A subject with
    # V_c != 0 has their wake rhythm shifted relative to this reference, so
    # the phase-correlation term correctly registers phase-shift pathology
    # as lost entrainment.  We DO NOT use V_c here.
    PHI_MORNING_TYPE = -np.pi / 3.0
    del params  # no longer needs phi
    C = np.sin(2.0 * np.pi * t_grid / 24.0 + PHI_MORNING_TYPE)

    A = 6.0  # A_SCALE — matches simulation.py
    n = len(t_grid)
    # Window size in samples, assuming uniform grid (5-min -> 288 samples/day)
    dt_hours = float(t_grid[1] - t_grid[0])
    win = max(int(round(24.0 / dt_hours)), 3)

    E = np.zeros(n)
    for i in range(n):
        lo = max(i - win + 1, 0)
        W_w  = W[lo:i + 1]
        Z_w  = Zt[lo:i + 1]
        C_w  = C[lo:i + 1]

        if len(W_w) < 3:
            E[i] = 0.0
            continue

        amp_W = (W_w.max() - W_w.min()) / 1.0
        amp_Z = (Z_w.max() - Z_w.min()) / A

        phase_W = max(_safe_corr(W_w, C_w),   0.0)
        phase_Z = max(_safe_corr(Z_w, -C_w),  0.0)

        E_W = amp_W * phase_W
        E_Z = amp_Z * phase_Z
        E[i] = E_W * E_Z

    # Clip to [0, 1] for numerical safety
    return np.clip(E, 0.0, 1.0)


def _compute_E_dynamics(trajectory, params):
    """Dynamics-side E(t) — the V_h-anabolic formula actually driving
    mu(E) inside the SDE.

    Direct numpy port of `_dynamics.entrainment_quality` and
    `simulation.entrainment_quality`. See
    `swat_entrainment_docs/01_formula.md` for the full derivation.

        A_W   = lambda_amp_W · V_h
        A_Z   = lambda_amp_Z · V_h
        B_W   = V_n − a + alpha_T · T
        B_Z   = -V_n + beta_Z · a
        amp_W = sigma(B_W + A_W) − sigma(B_W − A_W)
        amp_Z = sigma(B_Z + A_Z) − sigma(B_Z − A_Z)
        damp  = exp(-V_n / V_n_scale)
        phase = cos(π · min(|V_c|, V_c_max) / (2 · V_c_max))
        E_dyn = damp · amp_W · amp_Z · phase
    """
    a  = trajectory[:, 2]
    T  = trajectory[:, 3]
    Vh = trajectory[:, 5]
    Vn = trajectory[:, 6]
    p = params

    A_W = p['lambda_amp_W'] * Vh
    A_Z = p['lambda_amp_Z'] * Vh
    B_W = Vn - a + p['alpha_T'] * T
    B_Z = -Vn + p['beta_Z'] * a
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    amp_W = sig(B_W + A_W) - sig(B_W - A_W)
    amp_Z = sig(B_Z + A_Z) - sig(B_Z - A_Z)
    damp  = np.exp(-Vn / p['V_n_scale'])

    V_c_max = p['V_c_max']
    V_c_eff = min(abs(p['V_c']), V_c_max)
    phase = math.cos(math.pi * V_c_eff / (2.0 * V_c_max))

    return damp * amp_W * amp_Z * phase


def _safe_corr(x, y):
    """Pearson correlation, returning 0 if either series is constant."""
    sx = x.std()
    sy = y.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


# =========================================================================
# LATENT STATES (6-panel)
# =========================================================================

def _plot_latent(trajectory, t_grid, params, save_path):
    fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
    t_days = t_grid / 24.0

    # W
    axes[0].plot(t_days, trajectory[:, 0], lw=0.6, color='steelblue')
    axes[0].set_ylabel('W (wakefulness)')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    # Zt
    axes[1].plot(t_days, trajectory[:, 1], lw=0.6, color='indigo')
    axes[1].axhline(params['c_tilde'], ls='--', color='red', alpha=0.6,
                    label=f"c_tilde = {params['c_tilde']:.2f}")
    axes[1].set_ylabel('Zt (sleep depth)')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # adenosine
    axes[2].plot(t_days, trajectory[:, 2], lw=0.6, color='darkorange')
    axes[2].set_ylabel('a (adenosine)')
    axes[2].grid(True, alpha=0.3)

    # TESTOSTERONE — the new state
    T_amp = trajectory[:, 3]
    axes[3].plot(t_days, T_amp, lw=0.8, color='crimson')
    # Overlay the healthy-regime equilibrium T* = sqrt((mu_0 + mu_E)/eta) if mu_0+mu_E > 0
    mu_max = params['mu_0'] + params['mu_E']
    if mu_max > 0:
        T_star_max = math.sqrt(mu_max / params['eta'])
        axes[3].axhline(T_star_max, ls=':', color='green', alpha=0.5,
                        label=f"T*(E=1) = {T_star_max:.2f}")
    axes[3].axhline(0, ls=':', color='gray', alpha=0.5, label="T*=0 (flatline)")
    axes[3].set_ylabel('T (testosterone pulsatility)')
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].grid(True, alpha=0.3)

    # Circadian
    axes[4].plot(t_days, trajectory[:, 4], lw=0.6, color='seagreen')
    axes[4].set_ylabel('C(t)')
    axes[4].set_ylim(-1.1, 1.1)
    axes[4].grid(True, alpha=0.3)

    # Vh / Vn constants
    Vh = trajectory[:, 5]
    Vn = trajectory[:, 6]
    axes[5].plot(t_days, Vh, lw=1.0, color='forestgreen', label='V_h')
    axes[5].plot(t_days, Vn, lw=1.0, color='firebrick',    label='V_n')
    axes[5].set_ylabel('Potentials')
    axes[5].set_xlabel('Time (days)')
    axes[5].legend(loc='upper right', fontsize=8)
    axes[5].grid(True, alpha=0.3)

    basin = _basin_label(Vh[0], Vn[0], params.get('V_c', 0.0))
    fig.suptitle(
        f"SWAT SDE  —  latent states  "
        f"(V_h={Vh[0]:.2f}, V_n={Vn[0]:.2f}  ->  {basin};  "
        f"T_0={T_amp[0]:.2f})",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =========================================================================
# OBSERVATIONS (unchanged from sleep_wake_20p)
# =========================================================================

def _plot_observations(trajectory, t_grid, channel_outputs, params, save_path):
    """Four-panel observations plot: HR, sleep (3-level), steps, stress."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 2, 2]})
    t_days = t_grid / 24.0

    # ── Panel 1: HR ──
    hr_chan = channel_outputs.get('hr', {})
    hr_t_idx = hr_chan.get('t_idx', np.arange(len(t_grid)))
    hr_val = hr_chan.get('hr_value', np.zeros(len(t_grid)))
    hr_pred = params['HR_base'] + params['alpha_HR'] * trajectory[:, 0]

    axes[0].plot(t_days, hr_pred, color='crimson', lw=0.6, alpha=0.6,
                 label='HR mean (from W)')
    axes[0].scatter(t_grid[hr_t_idx] / 24.0, hr_val, s=2, alpha=0.35,
                    color='navy', label=f"HR obs (sigma_HR={params['sigma_HR']:.1f})")
    axes[0].set_ylabel('HR (bpm)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # ── Panel 2: 3-level sleep ──
    sl_chan = channel_outputs.get('sleep', {})
    sl_t_idx = sl_chan.get('t_idx', np.arange(len(t_grid)))
    sl_lvl = sl_chan.get('sleep_level', np.zeros(len(t_grid), dtype=int))
    t_days_s = t_grid[sl_t_idx] / 24.0
    axes[1].fill_between(t_days_s, 0, sl_lvl, step='mid',
                         color='midnightblue', alpha=0.7,
                         label='sleep level')
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(['wake', 'light+rem', 'deep'])
    axes[1].set_ylim(-0.3, 2.3)
    axes[1].set_ylabel('Sleep stage')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # ── Panel 3: steps (log-Gaussian, wake-gated, per-bin) ──
    st_chan = channel_outputs.get('steps', {})
    if 'log_value' in st_chan:
        st_t_idx = st_chan['t_idx']
        st_log = np.asarray(st_chan['log_value'])
        st_present = np.asarray(st_chan['present_mask'])
        # Recover step count from log(steps + 1).
        st_counts = np.expm1(st_log)
        st_days = t_grid[st_t_idx] / 24.0
        # Scatter only the wake-gated (present) observations.
        wake = st_present > 0.5
        axes[2].scatter(st_days[wake], st_counts[wake], s=2.5, alpha=0.5,
                        color='seagreen',
                        label='steps obs (wake bins, log-Gaussian)')
        # Overlay the deterministic mean E[steps | W] = exp(μ + σ²/2) − 1
        W = trajectory[:, 0]
        log_mean = (params['mu_step0']
                    + params['beta_W_steps'] * W)
        sigma = params['sigma_step']
        mean_count = np.exp(log_mean + 0.5 * sigma ** 2) - 1.0
        axes[2].plot(t_days, mean_count, lw=0.6, color='darkgreen', alpha=0.6,
                     label='E[steps | W]')
    axes[2].set_ylabel('Steps per bin (wake-gated)')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # ── Panel 4: Garmin stress ──
    sr_chan = channel_outputs.get('stress', {})
    if 'stress_score' in sr_chan:
        sr_t_idx = sr_chan['t_idx']
        sr_val = sr_chan['stress_score']
        sr_days = t_grid[sr_t_idx] / 24.0
        # Predicted stress (deterministic part)
        W = trajectory[:, 0]
        Vn = trajectory[:, 6]
        sr_pred = (params['s_base'] + params['alpha_s'] * W +
                   params['beta_s'] * Vn)
        axes[3].plot(t_days, sr_pred, color='purple', lw=0.6, alpha=0.6,
                     label='stress mean (from W, V_n)')
        axes[3].scatter(sr_days, sr_val, s=1.5, alpha=0.35, color='darkviolet',
                        label=f"stress obs (sigma_s={params['sigma_s']:.1f})")
    axes[3].set_ylabel('Stress (0-100)')
    axes[3].set_xlabel('Time (days)')
    axes[3].set_ylim(-5, 105)
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].grid(True, alpha=0.3)

    fig.suptitle('SWAT SDE  —  Observations (HR, sleep 3-level, steps, stress)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =========================================================================
# ENTRAINMENT DIAGNOSTIC PLOT
# =========================================================================

def _plot_entrainment(trajectory, t_grid, params, save_path):
    """Three-panel plot showing BOTH entrainment measures:

      Panel 1: E_dyn (solid) = what drives mu(E) inside the SDE dynamics
               E_obs (dashed) = windowed amp × phase-correlation diagnostic
               E_crit horizontal reference line

      Panel 2: mu(E_dyn) — the actual bifurcation parameter driving T.
               mu = 0 pitchfork line, shaded regions for pulsatile vs flatline basin.

      Panel 3: T(t) actual vs T* = sqrt(mu(E_dyn)/eta) (when mu > 0).
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t_days = t_grid / 24.0

    # Compute BOTH E quantities
    E_dyn = _compute_E_dynamics(trajectory, params)
    E_obs = _compute_E(trajectory, t_grid, params)
    mu = params['mu_0'] + params['mu_E'] * E_dyn
    E_crit = -params['mu_0'] / params['mu_E']

    T_star = np.where(mu > 0,
                       np.sqrt(np.maximum(mu, 0.0) / params['eta']),
                       0.0)
    T_actual = trajectory[:, 3]

    # ── Panel 1: BOTH E curves ──
    axes[0].plot(t_days, E_dyn, lw=1.0, color='darkviolet',
                 label='E_dyn (drives μ in SDE)')
    axes[0].plot(t_days, E_obs, lw=0.8, ls='--', color='darkorange',
                 label='E_obs (24h windowed, diagnostic)')
    axes[0].axhline(E_crit, ls=':', color='red', alpha=0.7,
                    label=f"E_crit = {E_crit:.2f}")
    axes[0].set_ylabel('E(t)  (entrainment quality)')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # ── Panel 2: mu(E_dyn) ──
    axes[1].plot(t_days, mu, lw=0.8, color='darkgreen')
    axes[1].axhline(0, ls='--', color='red', alpha=0.6,
                    label='mu = 0 (pitchfork)')
    axes[1].fill_between(t_days, 0, mu, where=(mu > 0),
                         color='green', alpha=0.15,
                         label='mu > 0 (pulsatile basin)')
    axes[1].fill_between(t_days, 0, mu, where=(mu < 0),
                         color='red', alpha=0.15,
                         label='mu < 0 (flatline basin)')
    axes[1].set_ylabel('mu(E_dyn)  (bifurcation param)')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # ── Panel 3: T actual vs T* expected ──
    axes[2].plot(t_days, T_actual, lw=0.8, color='crimson',
                 label='T (actual)')
    axes[2].plot(t_days, T_star, lw=0.8, ls='--', color='green',
                 label='T* = sqrt(mu/eta) when mu>0 else 0')
    axes[2].set_ylabel('T  (pulsatility amplitude)')
    axes[2].set_xlabel('Time (days)')
    axes[2].set_ylim(bottom=-0.1)
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.3)

    basin = _basin_label(trajectory[0, 5], trajectory[0, 6],
                         params.get('V_c', 0.0))
    fig.suptitle(
        f"SWAT — Entrainment → Bifurcation → Testosterone  "
        f"({basin};  tau_T = {params['tau_T']:.1f}h)",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =========================================================================
# BASIN LABEL (inherited from sleep_wake_20p)
# =========================================================================

def _basin_label(Vh, Vn, V_c=0.0):
    """Map (V_h, V_n, V_c) to a scenario label.

    V_c phase-shift pathology takes priority over V_h/V_n — a subject
    with healthy potentials but a misaligned rhythm is still pathological.
    """
    # Phase-shift pathology takes priority (|V_c| > 1h is clinically meaningful)
    if abs(V_c) >= 2.0:
        if abs(V_c) >= 8.0:
            return f"phase-inverted (V_c={V_c:+.1f}h)"
        return f"phase-shifted (V_c={V_c:+.1f}h)"

    Vh_high = Vh >= 0.6
    Vn_high = Vn >= 1.0
    if Vh_high and not Vn_high:
        return "healthy"
    if not Vh_high and Vn_high:
        return "hyperarousal-insomnia"
    if not Vh_high and not Vn_high:
        return "hypoarousal-hypersomnia"
    return "allostatic overload"

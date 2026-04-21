"""
models/sleep_wake_plots.py — Sleep-Wake Model Diagnostic Plots
===============================================================
Date:    15 April 2026
Version: 1.1

Produces two separate plots:
  1. latent_states.png    — the 6 hidden SDE states (3 panels)
  2. observations.png     — the 4 observation channels with fitted lines (4 panels)

Called by the generic CLI after data generation.
Not imported by any solver module.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_sleep_wake(trajectory, t_grid, channel_outputs, params, save_dir):
    """Generate all diagnostic plots for the sleep-wake model."""
    os.makedirs(save_dir, exist_ok=True)

    p1 = os.path.join(save_dir, "latent_states.png")
    _plot_latent_states(trajectory, t_grid, params, p1)
    print(f"  Plot: {p1}")

    p2 = os.path.join(save_dir, "observations.png")
    _plot_observations(trajectory, t_grid, channel_outputs, params, p2)
    print(f"  Plot: {p2}")


# =========================================================================
# PLOT 1: LATENT SDE STATES
# =========================================================================

def _plot_latent_states(trajectory, t_grid, params, save_path):
    """3-panel plot of the 6 hidden SDE state variables."""
    t_days = t_grid / 24.0
    W, Z = trajectory[:, 0], trajectory[:, 1]
    A, C = trajectory[:, 2], trajectory[:, 3]
    Vh, Vn = trajectory[:, 4], trajectory[:, 5]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # ── Panel 1: W + Z (fast flip-flop) ──
    ax = axes[0]
    ax.plot(t_days, W, color='#f59e0b', lw=1.2, label='$W$ (wakefulness)')
    ax.plot(t_days, Z, color='#3b82f6', lw=1.2, label='$Z$ (sleep depth)')
    ax.axhline(0.5, color='grey', ls='--', lw=0.6, alpha=0.4)
    ax.fill_between(t_days, 0, 1, where=(W > 0.5),
                     alpha=0.06, color='#f59e0b')
    ax.fill_between(t_days, 0, 1, where=(Z > 0.5),
                     alpha=0.06, color='#3b82f6')
    ax.set_ylabel('State value')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Fast Subsystem: Wakefulness & Sleep Depth')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15)

    # ── Panel 2: A + C (homeostatic + circadian) ──
    ax = axes[1]
    ax.plot(t_days, A, color='#ef4444', lw=1.2, label='$A$ (adenosine)')
    ax2 = ax.twinx()
    ax2.plot(t_days, C, color='#9ca3af', lw=0.8, alpha=0.5,
             label='$C$ (circadian)')
    ax2.set_ylabel('$C(t)$', color='#9ca3af', fontsize=9)
    ax2.set_ylim(-1.3, 1.3)
    ax2.tick_params(axis='y', labelcolor='#9ca3af')
    ax.set_ylabel('$A(t)$')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Homeostatic Pressure & Circadian Drive')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15)

    # ── Panel 3: Vh + Vn (slow potentials) ──
    ax = axes[2]
    ax.plot(t_days, Vh, color='#10b981', lw=1.8, label='$V_h$ (vitality)')
    ax.plot(t_days, Vn, color='#8b5cf6', lw=1.8, label='$V_n$ (stress)')
    ax.set_ylabel('Potential')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Time (days)')
    ax.set_title('Slow Subsystem: Vitality & Allostatic Load')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15)

    fig.suptitle('Latent SDE States: Sleep-Wake 6-State Model',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    plt.savefig(save_path, dpi=150)
    plt.close()


# =========================================================================
# PLOT 2: OBSERVATION CHANNELS
# =========================================================================

def _smooth_fit(t, y, window_hours=6.0):
    """Gaussian-windowed moving average for observation trend line.

    More robust than polynomial fitting — no conditioning issues,
    handles gaps and irregular spacing gracefully.

    Parameters
    ----------
    t : ndarray (in days)
    y : ndarray (observation values)
    window_hours : float
        Gaussian kernel standard deviation in hours.
    """
    if len(t) < 5:
        return y.copy()

    sigma_days = window_hours / 24.0
    t_out = np.sort(t)  # ensure sorted
    y_sorted = y[np.argsort(t)]
    fitted = np.zeros_like(y_sorted)

    for i in range(len(t_out)):
        weights = np.exp(-0.5 * ((t_out - t_out[i]) / sigma_days) ** 2)
        weights /= weights.sum()
        fitted[i] = np.dot(weights, y_sorted)

    # Restore original order
    inv_order = np.argsort(np.argsort(t))
    return fitted[inv_order]


def _plot_observations(trajectory, t_grid, ch, params, save_path):
    """4-panel observation channels, each with a fitted trend line."""
    dt = float(t_grid[1] - t_grid[0])

    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True)

    # ── Panel 1: Heart Rate ──
    ax = axes[0]
    if 'hr' in ch and len(ch['hr'].get('t_hours', [])) > 0:
        hr_t = ch['hr']['t_hours'] / 24.0  # already in hours, convert to days
        hr_v = ch['hr']['hr_bpm']
        ax.scatter(hr_t, hr_v, s=3, alpha=0.25, color='#3b82f6',
                   label=f'HR observations ($n={len(hr_v)}$)', zorder=2)
        fit = _smooth_fit(hr_t, hr_v.astype(float))
        ax.plot(hr_t, fit, color='#1d4ed8', lw=2.0,
                label='Fitted trend', zorder=3)
    ax.set_ylabel('Heart rate (bpm)')
    ax.set_title('Channel 1: Heart Rate (1-min native)')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15)

    # ── Panel 2: Stress Score ──
    ax = axes[1]
    if 'stress' in ch and len(ch['stress'].get('t_hours', [])) > 0:
        st_t = ch['stress']['t_hours'] / 24.0
        st_v = ch['stress']['stress_score']
        ax.scatter(st_t, st_v, s=3, alpha=0.2, color='#8b5cf6',
                   label=f'Stress observations ($n={len(st_v)}$)', zorder=2)
        fit = _smooth_fit(st_t, st_v.astype(float))
        ax.plot(st_t, fit, color='#6d28d9', lw=2.0,
                label='Fitted trend', zorder=3)
    ax.set_ylabel('Stress score (0–100)')
    ax.set_ylim(-5, 105)
    ax.set_title('Channel 2: Garmin Stress (3-min native)')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15)

    # ── Panel 3: Sleep Stages ──
    ax = axes[2]
    if 'sleep' in ch and len(ch['sleep'].get('labels', [])) > 0:
        sl_t = ch['sleep']['t_hours'] / 24.0  # epoch midpoints in days
        sl_l = ch['sleep']['labels']
        stage_names = {0: 'Awake', 1: 'Light', 2: 'REM', 3: 'Deep'}
        stage_colors = {'Awake': '#ef4444', 'Light': '#60a5fa',
                        'REM': '#a78bfa', 'Deep': '#10b981'}

        for label_val, name in stage_names.items():
            mask = sl_l == label_val
            if mask.any():
                ax.scatter(sl_t[mask],
                           np.full(mask.sum(), label_val),
                           s=8, alpha=0.4, color=stage_colors[name],
                           label=f'{name} ($n={mask.sum()}$)', zorder=2)

        # Fitted trend: treat labels as ordinal
        if len(sl_t) > 10:
            fit = _smooth_fit(sl_t, sl_l.astype(float), window_hours=4.0)
            fit = np.clip(fit, -0.3, 3.3)
            ax.plot(sl_t, fit, color='#374151', lw=1.8, ls='--',
                    label='Smoothed depth', zorder=3)

    ax.set_ylabel('Sleep stage')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Awake', 'Light', 'REM', 'Deep'])
    ax.set_ylim(-0.5, 3.5)
    ax.set_title('Channel 3: Sleep Stages (epoch run-length encoded)')
    ax.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.9)
    ax.grid(True, alpha=0.15)

    # ── Panel 4: Step Counts ──
    ax = axes[3]
    if 'steps' in ch and len(ch['steps'].get('t_idx', [])) > 0:
        # Steps are still grid-aligned (15-min bins = 3 grid steps)
        dt_grid = float(t_grid[1] - t_grid[0])
        step_t = ch['steps']['t_idx'] * dt_grid / 24.0
        step_v = ch['steps']['counts']
        nonzero = step_v > 0
        ax.bar(step_t[nonzero], step_v[nonzero],
               width=dt_grid / 24.0 * 2.8,
               color='#f59e0b', alpha=0.5,
               label=f'Step counts ($n_{{>0}}={nonzero.sum()}$)', zorder=2)

        # Show running bins
        running = ch['steps']['is_running']
        if np.any(running):
            ax.scatter(step_t[running], step_v[running],
                       s=15, color='#dc2626', marker='^', zorder=4,
                       label=f'Running ($n={running.sum()}$)')

        # Fitted trend on non-zero bins
        if nonzero.sum() > 10:
            fit = _smooth_fit(step_t[nonzero], step_v[nonzero])
            fit = np.maximum(fit, 0)
            ax.plot(step_t[nonzero], fit, color='#b45309', lw=2.0,
                    label='Fitted trend (non-zero)', zorder=3)

    ax.set_ylabel('Steps per bin')
    ax.set_xlabel('Time (days)')
    ax.set_title('Channel 4: Step Counts (Zero-Inflated Log-Normal)')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.15)

    fig.suptitle('Observation Processes: Synthetic Ground-Truth',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    plt.savefig(save_path, dpi=150)
    plt.close()

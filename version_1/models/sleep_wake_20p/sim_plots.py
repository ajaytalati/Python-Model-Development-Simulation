"""models/sleep_wake_20p/sim_plots.py — 20-Parameter Sleep-Wake Plots.

Date:    17 April 2026
Version: 1.0

Produces:
  1. latent_states.png  - W, Zt, a, C trajectories (plus V_h, V_n labels)
  2. observations.png   - HR trace and binary sleep strip
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_sleep_wake_20p(trajectory, t_grid, channel_outputs, params, save_dir):
    """Generate diagnostic plots for the 20-parameter sleep-wake SDE."""
    os.makedirs(save_dir, exist_ok=True)

    p1 = os.path.join(save_dir, "latent_states.png")
    _plot_latent(trajectory, t_grid, params, p1)
    print(f"  Plot: {p1}")

    p2 = os.path.join(save_dir, "observations.png")
    _plot_observations(trajectory, t_grid, channel_outputs, params, p2)
    print(f"  Plot: {p2}")


def _plot_latent(trajectory, t_grid, params, save_path):
    """5-panel plot of W, Zt, a, C with V_h, V_n in a text panel."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

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

    # Circadian
    axes[3].plot(t_days, trajectory[:, 3], lw=0.6, color='seagreen')
    axes[3].set_ylabel('C(t)')
    axes[3].set_ylim(-1.1, 1.1)
    axes[3].grid(True, alpha=0.3)

    # Vh / Vn constants
    Vh = trajectory[:, 4]
    Vn = trajectory[:, 5]
    axes[4].plot(t_days, Vh, lw=1.0, color='forestgreen', label='V_h')
    axes[4].plot(t_days, Vn, lw=1.0, color='firebrick',    label='V_n')
    axes[4].set_ylabel('Potentials')
    axes[4].set_xlabel('Time (days)')
    axes[4].legend(loc='upper right', fontsize=8)
    axes[4].grid(True, alpha=0.3)

    basin = _basin_label(Vh[0], Vn[0])
    fig.suptitle(
        f"20-Parameter Sleep-Wake-Adenosine SDE  -  latent states  "
        f"(V_h={Vh[0]:.2f}, V_n={Vn[0]:.2f}  -> {basin})",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_observations(trajectory, t_grid, channel_outputs, params, save_path):
    """2-panel plot: HR trace and binary sleep strip."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    t_days = t_grid / 24.0

    # HR
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

    # Binary sleep
    sl_chan = channel_outputs.get('sleep', {})
    sl_t_idx = sl_chan.get('t_idx', np.arange(len(t_grid)))
    sl_lab = sl_chan.get('sleep_label', np.zeros(len(t_grid), dtype=int))
    t_days_s = t_grid[sl_t_idx] / 24.0
    axes[1].fill_between(t_days_s, 0, sl_lab, step='mid',
                         color='midnightblue', alpha=0.7, label='asleep = 1')
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['wake', 'sleep'])
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('20-Parameter Sleep-Wake-Adenosine SDE  -  Observations',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _basin_label(Vh, Vn):
    """Map (V_h, V_n) to the 2x2 basin label from proof doc Section 2.6."""
    # thresholds chosen to match the identifiability proof doc prior medians
    Vh_high = Vh >= 0.6
    Vn_high = Vn >= 1.0
    if Vh_high and not Vn_high:
        return "healthy"
    if not Vh_high and Vn_high:
        return "hyperarousal-insomnia"
    if not Vh_high and not Vn_high:
        return "hypoarousal-hypersomnia"
    return "allostatic overload"

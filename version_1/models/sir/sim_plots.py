"""sir/sim_plots.py — diagnostic plots for the SIR model.

Two figures:
  - latent_states.png:  S(t), I(t), R(t) curves with R_0 = beta/gamma annotated.
  - observations.png:   daily Poisson case counts as bars, weekly serology
                        survey as scatter, both overlaid on truth I(t)/N.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_sir(trajectory, t_grid, channel_outputs, params, save_dir):
    """Two-panel diagnostic plot."""
    os.makedirs(save_dir, exist_ok=True)
    t_days = t_grid / 24.0
    S = trajectory[:, 0]
    I = trajectory[:, 1]
    N = params['N']
    R = N - S - I
    R0 = params['beta'] / params['gamma']

    # --- latent_states.png ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_days, S, label='S (susceptible)', color='steelblue')
    ax.plot(t_days, I, label='I (infected)',    color='firebrick')
    ax.plot(t_days, R, label='R (recovered)',   color='seagreen')
    ax.set_xlabel('time (days)')
    ax.set_ylabel('population')
    ax.set_title(f"SIR latent states  ·  N={N}, R_0={R0:.2f}, "
                 f"attack rate ≈ {R[-1]/N:.2f}")
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'latent_states.png'), dpi=120)
    plt.close(fig)
    print(f"  Plot: {os.path.join(save_dir, 'latent_states.png')}")

    # --- observations.png ---
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)

    cases = channel_outputs.get('cases')
    if cases is not None and len(cases.get('cases', [])) > 0:
        t_cases = t_days[cases['t_idx']]
        # Truth incidence per day for overlay
        beta = params['beta']
        rho = params.get('rho', 1.0)
        bin_h = float(cases['bin_hours'])
        # mean SI/N over the bin (simple per-time-step proxy: index value)
        truth_expected = rho * beta * (S * I / N)
        # smooth to daily for overlay
        dt_h = float(t_grid[1] - t_grid[0])
        bin_size = max(int(round(bin_h / dt_h)), 1)
        n_bins = len(t_grid) // bin_size
        truth_daily = (truth_expected[: n_bins * bin_size]
                       .reshape(n_bins, bin_size).mean(axis=1)) * bin_h
        axes[0].bar(t_cases, cases['cases'], width=0.8, color='firebrick',
                    alpha=0.6, label='observed cases (Poisson)')
        axes[0].plot(t_cases[:len(truth_daily)], truth_daily,
                     color='black', linewidth=1.0, label='E[cases | truth]')
        axes[0].set_ylabel('cases / day')
        axes[0].set_title('Observation channel 1 — Poisson daily case counts')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    sero = channel_outputs.get('serology')
    if sero is not None and len(sero.get('prevalence', [])) > 0:
        t_sero = t_days[sero['t_idx']]
        prev_obs = sero['prevalence']
        axes[1].plot(t_days, I / N, color='firebrick', linewidth=1.0,
                     label='truth I(t)/N', alpha=0.7)
        axes[1].errorbar(t_sero, prev_obs,
                         yerr=2.0 * params['sigma_z'],
                         fmt='o', color='steelblue',
                         label='serology survey (Gaussian, ±2σ)',
                         capsize=3)
        axes[1].set_xlabel('time (days)')
        axes[1].set_ylabel('prevalence')
        axes[1].set_title('Observation channel 2 — Gaussian weekly serology')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'observations.png'), dpi=120)
    plt.close(fig)
    print(f"  Plot: {os.path.join(save_dir, 'observations.png')}")

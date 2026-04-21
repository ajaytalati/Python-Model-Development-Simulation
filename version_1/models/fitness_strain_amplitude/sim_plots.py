"""models/fitness_strain_amplitude/sim_plots.py — FSA Diagnostic Plots.

Date:    18 April 2026
Version: 1.0

Produces:
  1. latent_states.png  — 4-panel plot
        top-left:  B(t), F(t)
        top-right: A(t) with A* = sqrt(mu/eta) overlay where mu > 0
        bot-left:  mu(t) with zero crossing, plus Phi-jump marker if S3
        bot-right: Inputs T_B(t), Phi(t)
  2. landscape_evolution.png — V_A(A) at 5 evenly-spaced snapshots
  3. observations.png — observed y_B, y_F, y_A vs true latent states
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_fsa(trajectory, t_grid, channel_outputs, params, save_dir):
    """Entry point called by the simulator."""
    os.makedirs(save_dir, exist_ok=True)

    T_B_arr = _channel_arr(channel_outputs, 'T_B',  'T_B_value',  len(t_grid))
    Phi_arr = _channel_arr(channel_outputs, 'Phi',  'Phi_value',  len(t_grid))

    p1 = os.path.join(save_dir, "latent_states.png")
    _plot_latent(trajectory, t_grid, T_B_arr, Phi_arr, params, p1)
    print(f"  Plot: {p1}")

    p2 = os.path.join(save_dir, "landscape_evolution.png")
    _plot_landscape(trajectory, t_grid, params, p2)
    print(f"  Plot: {p2}")

    p3 = os.path.join(save_dir, "observations.png")
    _plot_observations(trajectory, t_grid, channel_outputs, params, p3)
    print(f"  Plot: {p3}")


# -------------------------------------------------------------------------
# Panel A — latent trajectories
# -------------------------------------------------------------------------

def _plot_latent(trajectory, t_grid, T_B_arr, Phi_arr, params, save_path):
    B = trajectory[:, 0]; F = trajectory[:, 1]; A = trajectory[:, 2]
    mu = _mu(B, F, params)
    A_star = np.where(mu > 0, np.sqrt(np.maximum(mu, 0) / params['eta']), np.nan)

    fig, ((ax_BF, ax_A), (ax_mu, ax_in)) = plt.subplots(
        2, 2, figsize=(13, 8), sharex=True)

    # --- Top-left: B and F on twin axes -----------------------------------
    ax_BF.plot(t_grid, B, color='steelblue', label='B (fitness)', linewidth=1.2)
    ax_BF.set_ylabel('B', color='steelblue')
    ax_BF.set_ylim(-0.05, 1.05)
    ax_BF.tick_params(axis='y', labelcolor='steelblue')
    ax_BF_r = ax_BF.twinx()
    ax_BF_r.plot(t_grid, F, color='firebrick', label='F (strain)', linewidth=1.2)
    ax_BF_r.set_ylabel('F', color='firebrick')
    ax_BF_r.tick_params(axis='y', labelcolor='firebrick')
    ax_BF.set_title('Fitness and Strain')
    ax_BF.grid(True, alpha=0.3)

    # --- Top-right: A with A* overlay ------------------------------------
    ax_A.plot(t_grid, A, color='darkgreen', linewidth=1.0, label='A (pulsatility)')
    ax_A.plot(t_grid, A_star, color='darkorange', linestyle='--', linewidth=1.0,
              alpha=0.7, label=r'$A^* = \sqrt{\mu/\eta}$')
    ax_A.axhline(0, color='grey', alpha=0.4)
    ax_A.set_ylabel('A')
    ax_A.set_title('HPG Pulsatility Amplitude')
    ax_A.legend(loc='upper left', fontsize=9)
    ax_A.grid(True, alpha=0.3)

    # --- Bottom-left: mu(t) with zero-crossing ---------------------------
    ax_mu.plot(t_grid, mu, color='purple', linewidth=1.2, label=r'$\mu(B,F)$')
    ax_mu.axhline(0, color='black', linestyle='--', alpha=0.6,
                  label='bifurcation ($\\mu=0$)')
    ax_mu.fill_between(t_grid, 0, mu, where=(mu < 0),
                       color='firebrick', alpha=0.15, label='pathological')
    ax_mu.fill_between(t_grid, 0, mu, where=(mu >= 0),
                       color='forestgreen', alpha=0.15, label='healthy')
    ax_mu.set_ylabel(r'$\mu(B,F)$')
    ax_mu.set_xlabel('Time (days)')
    ax_mu.set_title('Bifurcation Parameter')
    ax_mu.legend(loc='best', fontsize=8)
    ax_mu.grid(True, alpha=0.3)

    # --- Bottom-right: inputs --------------------------------------------
    ax_in.plot(t_grid, T_B_arr, color='navy', linewidth=1.3, drawstyle='steps-post',
               label=r'$T_B(t)$ (adaptation target)')
    ax_in_r = ax_in.twinx()
    ax_in_r.plot(t_grid, Phi_arr, color='darkorange', linewidth=1.3,
                 drawstyle='steps-post', label=r'$\Phi(t)$ (strain production)')
    ax_in.set_ylabel(r'$T_B$', color='navy')
    ax_in_r.set_ylabel(r'$\Phi$', color='darkorange')
    ax_in.tick_params(axis='y', labelcolor='navy')
    ax_in_r.tick_params(axis='y', labelcolor='darkorange')
    ax_in.set_xlabel('Time (days)')
    ax_in.set_title('Clinician Inputs')
    ax_in.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------------
# Panel B — landscape V_A(A) snapshots
# -------------------------------------------------------------------------

def _plot_landscape(trajectory, t_grid, params, save_path):
    """V_A(A) at 5 evenly-spaced times along the trajectory."""
    B = trajectory[:, 0]; F = trajectory[:, 1]
    n_snap = 5
    idxs = np.linspace(0, len(t_grid) - 1, n_snap).astype(int)
    A_range = np.linspace(0.0, 1.2, 400)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    cmap = plt.cm.viridis
    for i, idx in enumerate(idxs):
        mu_i = _mu_scalar(B[idx], F[idx], params)
        V = -0.5 * mu_i * A_range**2 + 0.25 * params['eta'] * A_range**4
        color = cmap(i / max(n_snap - 1, 1))
        ax.plot(A_range, V, color=color, linewidth=1.3,
                label=f't={t_grid[idx]:.1f}d  $\\mu$={mu_i:+.3f}')
    ax.axhline(0, color='grey', alpha=0.3)
    ax.set_xlabel('A')
    ax.set_ylabel(r'$V_A(A)$')
    ax.set_title('Landau Potential Evolution  '
                 r'$V_A(A) = -\frac{\mu(B,F)}{2}A^2 + \frac{\eta}{4}A^4$')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------------
# Panel C — observations
# -------------------------------------------------------------------------

def _plot_observations(trajectory, t_grid, channel_outputs, params, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ['B (fitness)', 'F (strain)', 'A (pulsatility)']
    colors = ['steelblue', 'firebrick', 'darkgreen']
    for i, (ch_name, ax, lab, col) in enumerate(zip(
            ['obs_B', 'obs_F', 'obs_A'], axes, labels, colors)):
        ax.plot(t_grid, trajectory[:, i], color=col, linewidth=0.9, label='true')
        ch = channel_outputs.get(ch_name, {})
        if 't_idx' in ch and 'obs_value' in ch:
            idx = np.asarray(ch['t_idx'])
            y   = np.asarray(ch['obs_value'])
            ax.scatter(t_grid[idx], y, s=3, alpha=0.35, color='black',
                       label=f'obs ($\\sigma$={params["sigma_obs"]:.2f})')
        ax.set_ylabel(lab)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (days)')
    fig.suptitle('FSA Observations', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _mu(B, F, p):
    return p['mu_0'] + p['mu_B'] * B - p['mu_F'] * F - p['mu_FF'] * F * F


def _mu_scalar(B, F, p):
    return float(p['mu_0'] + p['mu_B'] * B - p['mu_F'] * F - p['mu_FF'] * F * F)


def _channel_arr(channel_outputs, ch_name, value_key, default_len):
    ch = channel_outputs.get(ch_name, {})
    if value_key in ch:
        return np.asarray(ch[value_key])
    return np.zeros(default_len, dtype=np.float32)

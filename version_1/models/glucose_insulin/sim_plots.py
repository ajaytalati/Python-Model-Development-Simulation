"""glucose_insulin/sim_plots.py — diagnostic plots for the Bergman model.

Two figures:
  - latent_states.png:  G(t), X(t), I(t) — the 3-state SDE trajectory.
  - observations.png:   CGM dots overlaid on G truth + meal carb-count bars
                        + meal timing markers (vertical lines).
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_glucose_insulin(trajectory, t_grid, channel_outputs, params, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    t_h = t_grid
    G = trajectory[:, 0]
    X = trajectory[:, 1]
    I = trajectory[:, 2]

    # --- latent_states.png ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t_h, G, color='firebrick', lw=1.0)
    axes[0].axhline(params['Gb'], color='k', ls='--', alpha=0.4,
                    label=f"Gb = {params['Gb']:.0f}")
    axes[0].axhspan(70, 180, alpha=0.08, color='green', label='target range')
    axes[0].set_ylabel('G (mg/dL)')
    axes[0].set_title(f"glucose_insulin — Bergman minimal model "
                      f"(SI = p₃/p₂ = {params['p3']/params['p2']:.3g} /(hr·μU/mL))")
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_h, X, color='steelblue', lw=1.0)
    axes[1].set_ylabel('X (1/hr)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_h, I, color='seagreen', lw=1.0)
    axes[2].axhline(params['Ib'], color='k', ls='--', alpha=0.4,
                    label=f"Ib = {params['Ib']:.1f}")
    axes[2].set_ylabel('I (μU/mL)')
    axes[2].set_xlabel('time (hours)')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'latent_states.png'), dpi=120)
    plt.close(fig)
    print(f"  Plot: {os.path.join(save_dir, 'latent_states.png')}")

    # --- observations.png ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

    # Panel 0: CGM (Gaussian on G)
    cgm = channel_outputs.get('cgm')
    axes[0].plot(t_h, G, color='firebrick', lw=1.0, alpha=0.7,
                 label='truth G(t)')
    if cgm is not None and len(cgm.get('cgm_value', [])) > 0:
        t_cgm = t_h[cgm['t_idx']]
        axes[0].scatter(t_cgm, cgm['cgm_value'], s=4, color='steelblue',
                        alpha=0.5, label='CGM (Gaussian, 5-min)')
    axes[0].axhspan(70, 180, alpha=0.08, color='green')
    axes[0].set_ylabel('G (mg/dL)')
    axes[0].set_title('Observation channel 1 — CGM (Gaussian, every 5 min)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 1: Meal carb counts (Poisson)
    meals = channel_outputs.get('meal_carbs')
    if meals is not None and len(meals.get('carbs_g', [])) > 0:
        t_meals = t_h[meals['t_idx']]
        axes[1].bar(t_meals, meals['carbs_g'], width=0.3, color='darkorange',
                    alpha=0.7, label='observed meal carbs (Poisson, g)')
        # Mark meal-time vertical lines
        for tm in t_meals:
            axes[1].axvline(tm, color='k', alpha=0.2, lw=0.5)
    axes[1].set_xlabel('time (hours)')
    axes[1].set_ylabel('carbs (g)')
    axes[1].set_title('Observation channel 2 — meal carb counts (Poisson, 3/day)')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'observations.png'), dpi=120)
    plt.close(fig)
    print(f"  Plot: {os.path.join(save_dir, 'observations.png')}")

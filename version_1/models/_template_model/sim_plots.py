"""_template_model/sim_plots.py — Plot function stub.

Reference:  how_to_add_a_new_model/01_simulation.md §2

The plotter receives the full trajectory, time grid, channel outputs, and
params, and writes one or more PNGs to `save_dir`.  Produce at minimum a
`latent_states.png` with one panel per state, and an `observations.png`
overlaying each channel on the latent it tracks.
"""

import os
import matplotlib.pyplot as plt


def plot_template(trajectory, t_grid, channel_outputs, params, save_dir):
    """Two-panel plot: latent x(t) + observation channel overlay."""
    os.makedirs(save_dir, exist_ok=True)

    # --- latent_states.png ---
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t_grid, trajectory[:, 0], label='x(t)', color='steelblue')
    ax.axhline(params['mu'], color='k', linestyle='--',
               alpha=0.5, label=f"mu = {params['mu']}")
    ax.set_xlabel('time (h)')
    ax.set_ylabel('x')
    ax.set_title('_template_ — latent state')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'latent_states.png'), dpi=120)
    plt.close(fig)
    print(f"  Plot: {os.path.join(save_dir, 'latent_states.png')}")

    # --- observations.png ---
    obs = channel_outputs.get('obs')
    if obs is not None:
        t_obs = t_grid[obs['t_idx']]
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(t_grid, trajectory[:, 0], label='x(t) truth',
                color='steelblue', alpha=0.6)
        ax.scatter(t_obs, obs['obs_value'], s=5, color='firebrick',
                   alpha=0.5, label='observations')
        ax.set_xlabel('time (h)')
        ax.set_ylabel('x / y')
        ax.set_title('_template_ — observations vs truth')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'observations.png'), dpi=120)
        plt.close(fig)
        print(f"  Plot: {os.path.join(save_dir, 'observations.png')}")

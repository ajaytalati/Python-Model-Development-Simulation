"""Sleep-wake-specific diagnostic plots.

I/O MODULE: writes PNG files via matplotlib.

Date:    15 April 2026
Version: 5.0 (model-agnostic framework)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_trajectory(t_hours, trajectory, obs_data, params_dict,
                    save_path="map_trajectory.png"):
    """4-panel MAP trajectory with Garmin data overlay.

    Args:
        t_hours: Time grid, shape (T,).
        trajectory: States [W,Z,A,C,Vh,Vn], shape (T, 6).
        obs_data: SleepWakeObsData.
        params_dict: Dict of posterior mean parameter values.
        save_path: Output file.
    """
    dt_h = 5.0 / 60.0
    W, Z, A, C = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3]
    Vh, Vn = trajectory[:, 4], trajectory[:, 5]
    td = t_hours / 24.0

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)

    # Panel 1: W + HR
    ax = axes[0]
    ax.plot(td, W, '#f59e0b', lw=1.5, label='W(t)')
    HRb = params_dict.get('HR_base', 50); aHR = params_dict.get('alpha_HR', 25)
    bex = params_dict.get('beta_exercise', 40)
    ht = obs_data.hr_t_idx * dt_h / 24.0
    hw = (obs_data.hr_bpm - HRb - bex * obs_data.hr_exercise) / max(aHR, 1)
    ax.scatter(ht, hw, c='grey', s=1, alpha=0.15, label='HR->W')
    ax.set_ylabel('W'); ax.set_ylim(-0.2, 1.3); ax.legend(fontsize=7); ax.grid(alpha=0.2)
    ax.set_title('Wakefulness', fontweight='bold')

    # Panel 2: Z + sleep labels
    ax = axes[1]
    ax.plot(td, Z, '#3b82f6', lw=1.5, label='Z(t)')
    sc = {0: '#ef4444', 1: '#60a5fa', 2: '#a78bfa', 3: '#10b981'}
    sn = {0: 'Awake', 1: 'Light', 2: 'REM', 3: 'Deep'}
    zd = {0: 0.0, 1: 0.2, 2: 0.55, 3: 0.85}
    st = obs_data.sleep_t_hours / 24.0
    for lv in [0, 1, 2, 3]:
        m = obs_data.sleep_labels == lv
        if m.any():
            ax.scatter(st[m], np.full(m.sum(), zd[lv]), c=sc[lv], s=8, alpha=0.5, label=sn[lv])
    ax.set_ylabel('Z'); ax.set_ylim(-0.2, 1.2); ax.legend(fontsize=7, ncol=5); ax.grid(alpha=0.2)
    ax.set_title('Sleep Depth', fontweight='bold')

    # Panel 3: A + C
    ax = axes[2]
    ax.plot(td, A, '#ef4444', lw=1.5, label='A(t)')
    ax2 = ax.twinx()
    ax2.plot(td, C, 'grey', lw=0.8, alpha=0.4, label='C(t)')
    ax2.set_ylabel('C', color='grey')
    ax.set_ylabel('A'); ax.legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7); ax.grid(alpha=0.2)
    ax.set_title('Adenosine + Circadian', fontweight='bold')

    # Panel 4: Vh + Vn
    ax = axes[3]
    ax.plot(td, Vh, '#10b981', lw=2, label='Vh')
    ax.plot(td, Vn, '#8b5cf6', lw=2, label='Vn')
    ax.set_ylabel('Potential'); ax.set_xlabel('Days')
    ax.legend(fontsize=7); ax.grid(alpha=0.2)
    ax.set_title('Slow Potentials', fontweight='bold')

    fig.suptitle('MAP Trajectory vs Garmin Data', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0.02, 1, 0.97))
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"  -> {save_path}")


def plot_residuals(t_hours, trajectory, obs_data, params_dict,
                   save_path="residuals.png"):
    """HR and stress residual plots.

    Args:
        t_hours: Time grid, shape (T,).
        trajectory: States, shape (T, 6).
        obs_data: SleepWakeObsData.
        params_dict: Posterior means.
        save_path: Output file.
    """
    dt_h = 5.0 / 60.0
    W, Vn = trajectory[:, 0], trajectory[:, 5]
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

    HRb = params_dict.get('HR_base', 50); aHR = params_dict.get('alpha_HR', 25)
    bex = params_dict.get('beta_exercise', 40); sHR = params_dict.get('sigma_HR', 8)
    pred = HRb + aHR * W[obs_data.hr_t_idx] + bex * obs_data.hr_exercise
    res = obs_data.hr_bpm - pred
    axes[0].scatter(obs_data.hr_t_idx * dt_h / 24, res, c='steelblue', s=2, alpha=0.3)
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_ylabel('HR residual'); axes[0].set_title('HR Residuals', fontweight='bold')
    axes[0].grid(alpha=0.2)

    sb = params_dict.get('s_base', 10); sW = params_dict.get('s_W', 40)
    sn = params_dict.get('s_n', 3)
    sp = sb + sW * W[obs_data.stress_t_idx] + sn * Vn[obs_data.stress_t_idx]
    sr = obs_data.stress_score - sp
    axes[1].scatter(obs_data.stress_t_idx * dt_h / 24, sr, c='#8b5cf6', s=2, alpha=0.3)
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_ylabel('Stress residual'); axes[1].set_xlabel('Days')
    axes[1].set_title('Stress Residuals', fontweight='bold'); axes[1].grid(alpha=0.2)

    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  -> {save_path}")

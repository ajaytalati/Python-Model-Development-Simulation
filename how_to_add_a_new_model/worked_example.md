# Worked example — build a 2-state coupled OU model in 15 minutes

Target: a 2-state coupled Ornstein-Uhlenbeck (OU) SDE with two Gaussian observation channels. We'll call it `ou_pair`.

$$
dx_1 = -k_1 (x_1 - \mu_1) \,dt + \sigma_1 \,dB_1
$$

$$
dx_2 = \bigl[-k_2 (x_2 - \mu_2) + c \cdot x_1\bigr] \,dt + \sigma_2 \,dB_2
$$

$x_1$ is a free mean-reverting OU process; $x_2$ is also mean-reverting but additionally coupled to $x_1$ with strength $c$. Two Gaussian observation channels — one per state.

By the end of this walkthrough you will have a running model, a plot, and a physics check. Elapsed time on a fresh shell: **~15 minutes**, most of which is typing.

---

## 0. Prerequisites check

From the repo root:

```bash
cd version_1
ls simulator/   # must show: sde_model.py, run_simulator.py, sde_solver_scipy.py, sde_solver_diffrax.py, sde_observations.py
python -c "import numpy, scipy, matplotlib; print('ok')"
```

Expected output: `ok`.

---

## 1. Copy the template

From `version_1/`:

```bash
cp -r models/_template_model models/ou_pair
ls models/ou_pair/
```

Expected: `README.md  TESTING.md  __init__.py  _dynamics.py  estimation.py  sim_plots.py  simulation.py`.

Confirm the template runs unchanged:

```bash
python simulator/run_simulator.py \
    --model models.ou_pair.simulation.TEMPLATE_MODEL \
    --param-set A --seed 42
```

Expected tail:

```
  Physics verification:
  converges_to_mu                    PASS
  all_finite                         PASS
  PASS: physics verification
  Plot: outputs/synthetic__template__A_<ts>/latent_states.png
  Plot: outputs/synthetic__template__A_<ts>/observations.png
```

If that worked, the wiring is fine — now edit.

---

## 2. Rename identifiers

Three files need renaming. Open each in your editor (or use `sed`) and change `TEMPLATE` → `OU_PAIR`, `_template_` → `ou_pair`, `plot_template` → `plot_ou_pair`:

```bash
# macOS/BSD sed: use -i ''
# Order matters: replace the longer `_template_model` substring first so the
# `_template_` rule doesn't strip the trailing `_` in directory-path strings.
sed -i \
    -e 's/_template_model/ou_pair/g' \
    -e 's/TEMPLATE/OU_PAIR/g' \
    -e 's/_template_/ou_pair/g' \
    -e 's/plot_template/plot_ou_pair/g' \
    models/ou_pair/__init__.py \
    models/ou_pair/simulation.py \
    models/ou_pair/sim_plots.py
rm models/ou_pair/README.md   # template-specific; not needed
```

Smoke test again — should still work:

```bash
python simulator/run_simulator.py \
    --model models.ou_pair.simulation.OU_PAIR_MODEL \
    --param-set A --seed 42 2>&1 | tail -6
```

Still produces `PASS`. Now the actual edits.

---

## 3. Define two states

Edit `models/ou_pair/simulation.py`. Replace the `states` tuple:

```python
states=(
    StateSpec("x1", -10.0, 10.0),
    StateSpec("x2", -10.0, 10.0),
),
```

---

## 4. Rewrite the drift

```python
def drift(t, y, params, aux):
    """Coupled OU drift: x1 is free, x2 is coupled to x1."""
    del aux
    x1, x2 = y[0], y[1]
    dx1 = -params['k1'] * (x1 - params['mu1'])
    dx2 = -params['k2'] * (x2 - params['mu2']) + params['c'] * x1
    return np.array([dx1, dx2])
```

---

## 5. Rewrite the diffusion

```python
def diffusion_diagonal(params):
    """Per-state constant noise magnitudes."""
    return np.array([params['sigma1'], params['sigma2']])
```

---

## 6. Rewrite `make_y0`

```python
def make_y0(init_dict, params):
    del params
    return np.array([init_dict['x1_0'], init_dict['x2_0']])
```

---

## 7. Add a second observation channel

Replace the single `gen_obs` with two channel generators, one per state:

```python
def gen_obs_x1(trajectory, t_grid, params, aux, prior_channels, seed):
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    x1 = trajectory[:, 0]
    noise = rng.normal(0.0, params['sigma_obs'], size=len(t_grid))
    return {
        't_idx':     np.arange(len(t_grid), dtype=np.int32),
        'obs_value': (x1 + noise).astype(np.float32),
    }


def gen_obs_x2(trajectory, t_grid, params, aux, prior_channels, seed):
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    x2 = trajectory[:, 1]
    noise = rng.normal(0.0, params['sigma_obs'], size=len(t_grid))
    return {
        't_idx':     np.arange(len(t_grid), dtype=np.int32),
        'obs_value': (x2 + noise).astype(np.float32),
    }
```

Update the `channels` tuple on the `SDEModel`:

```python
channels=(
    ChannelSpec("obs_x1", depends_on=(), generate_fn=gen_obs_x1),
    ChannelSpec("obs_x2", depends_on=(), generate_fn=gen_obs_x2),
),
```

---

## 8. Fix the verify_physics and plotter for two states

Edit `verify_physics`:

```python
def verify_physics(trajectory, t_grid, params):
    x1 = trajectory[:, 0]
    x2 = trajectory[:, 1]
    # After many mean-reversion timescales the mean should sit near the coupled equilibrium.
    # Equilibrium: x1_eq = mu1,  x2_eq = mu2 + (c/k2) * mu1.
    x2_eq = params['mu2'] + (params['c'] / params['k2']) * params['mu1']
    return {
        'x1_range':        float(x1.max() - x1.min()),
        'x2_range':        float(x2.max() - x2.min()),
        'x1_near_mu1':     abs(float(x1[-100:].mean()) - params['mu1']) < 0.5,
        'x2_near_x2_eq':   abs(float(x2[-100:].mean()) - x2_eq) < 0.5,
        'all_finite':      bool(np.all(np.isfinite(trajectory))),
    }
```

Edit `models/ou_pair/sim_plots.py` (already renamed to `plot_ou_pair` in step 2):

```python
import os
import matplotlib.pyplot as plt


def plot_ou_pair(trajectory, t_grid, channel_outputs, params, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t_grid, trajectory[:, 0], color='steelblue', label='x1(t)')
    axes[0].axhline(params['mu1'], color='k', linestyle='--', alpha=0.4,
                    label=f"mu1 = {params['mu1']}")
    axes[0].set_ylabel('x1'); axes[0].legend()
    axes[1].plot(t_grid, trajectory[:, 1], color='firebrick', label='x2(t)')
    x2_eq = params['mu2'] + (params['c'] / params['k2']) * params['mu1']
    axes[1].axhline(x2_eq, color='k', linestyle='--', alpha=0.4,
                    label=f"x2_eq = {x2_eq:.2f}")
    axes[1].set_xlabel('time (h)'); axes[1].set_ylabel('x2'); axes[1].legend()
    fig.suptitle('ou_pair — latent states')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'latent_states.png'), dpi=120)
    plt.close(fig)
    print(f"  Plot: {os.path.join(save_dir, 'latent_states.png')}")
```

---

## 9. Rewrite the parameter set

Back in `simulation.py`:

```python
PARAM_SET_A = {
    'k1':            1.0,   # 1/hr  (x1 timescale 1 h)
    'k2':            0.5,   # 1/hr  (x2 timescale 2 h)
    'mu1':           0.0,
    'mu2':           0.0,
    'c':             0.8,   # x1 -> x2 coupling
    'sigma1':        0.3,
    'sigma2':        0.3,
    'sigma_obs':     0.1,
    'dt_hours':      0.1,
    't_total_hours': 48.0,
}

INIT_STATE_A = {'x1_0': 2.0, 'x2_0': -1.0}

EXOGENOUS_A = {}
```

Also add a Set B to demonstrate a different coupling regime:

```python
PARAM_SET_B = {**PARAM_SET_A, 'c': 0.0}  # decoupled case
INIT_STATE_B = INIT_STATE_A
EXOGENOUS_B = {}
```

Update the `SDEModel(...)` kwargs to include both sets:

```python
param_sets={'A': PARAM_SET_A, 'B': PARAM_SET_B},
init_states={'A': INIT_STATE_A, 'B': INIT_STATE_B},
exogenous_inputs={'A': EXOGENOUS_A, 'B': EXOGENOUS_B},
```

---

## 10. Run the model

```bash
python simulator/run_simulator.py \
    --model models.ou_pair.simulation.OU_PAIR_MODEL \
    --param-set A --seed 42
```

Expected output (key lines):

```
  Model:   ou_pair v0.1
  States:  2 (x1, x2)
  Channels: 2 (obs_x1, obs_x2)
  Grid:    480 steps × 6-min = 2 days
  ...
  State ranges:
        x1: [~-0.8, 2.0]
        x2: [~-1.0, ~0.5]
  ...
  Physics verification:
  x1_near_mu1                        PASS
  x2_near_x2_eq                      PASS
  all_finite                         PASS
```

Plots land in `outputs/synthetic_ou_pair_A_<ts>/`.

Try Set B (decoupled):

```bash
python simulator/run_simulator.py \
    --model models.ou_pair.simulation.OU_PAIR_MODEL \
    --param-set B --seed 42
```

$x_2$ should mean-revert to $\mu_2 = 0$ instead of being pulled toward $x_1$; `x2_eq = 0` in `x2_near_x2_eq`.

---

## 11. Verify-only mode

```bash
python simulator/run_simulator.py \
    --model models.ou_pair.simulation.OU_PAIR_MODEL \
    --verify 2>&1 | tail -10
```

Runs the deterministic skeleton (noise zeroed via the `sigma_<state>` pattern match) and prints `verify_physics` results. Both booleans should be `PASS`.

---

## 12. What you have now

Every file in `models/ou_pair/`:

- `simulation.py` — coupled 2-state OU with two Gaussian channels, Set A + Set B.
- `sim_plots.py` — `plot_ou_pair` producing `latent_states.png`.
- `__init__.py` — exports `OU_PAIR_MODEL`.
- `estimation.py`, `_dynamics.py`, `TESTING.md` — stubs, ready to fill in.

You can now:

- **Extend to JAX**: write `drift_jax` following [01_simulation.md §2.2](01_simulation.md#22-drift_fn_jax-optional--enables-diffrax), then `--cross-validate` to confirm they agree.
- **Fill in `TESTING.md`**: [03_testing_and_docs.md §2–§3](03_testing_and_docs.md#2-testingmd-section-template).
- **Write `estimation.py`**: [02_estimation.md](02_estimation.md).
- **Add a user-facing doc**: [03_testing_and_docs.md §4](03_testing_and_docs.md#4-the-per-model-user-facing-doc).

Elapsed time: ~15 minutes of typing plus one or two minutes of test runs. The framework did all the plumbing.

---

**Back to:** [README](README.md) · [01_simulation](01_simulation.md) · [02_estimation](02_estimation.md) · [03_testing_and_docs](03_testing_and_docs.md) · [04_reference](04_reference.md)

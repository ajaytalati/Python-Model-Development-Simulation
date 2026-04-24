# 02 — Writing `estimation.py`

This file is **optional**. Skip if your model is simulation-only. Write it when you want to run posterior inference (particle filter, SMC², EKF).

The estimation contract is defined in [`version_1/estimation_model.py`](../version_1/estimation_model.py) as a frozen dataclass `EstimationModel`. The canonical reference implementation is [`version_1/models/sleep_wake_20p/estimation.py`](../version_1/models/sleep_wake_20p/estimation.py) (+ [`_dynamics.py`](../version_1/models/sleep_wake_20p/_dynamics.py)). Copy that structure.

---

## 1. When to write `estimation.py`

Skip it for:
- Pedagogical / demonstration models.
- Models where you only need synthetic data for downstream pipelines.
- Early-stage model development (wait until the simulation is stable).

Write it when you need:
- Posterior parameter inference (MCMC, SMC², MCLMC).
- Particle-filter likelihoods.
- Extended Kalman filter approximations.
- Marginal-SGR kernel methods.

The estimation object is imported by the sampler/runner, not by the simulator CLI. So the simulator can work end-to-end without it.

---

## 2. The `EstimationModel` contract

Required fields:

| Field | Type | Meaning |
|:---|:---|:---|
| `name`, `version` | `str` | Metadata. |
| `n_states` | `int` | Total latent states including deterministic ones. |
| `n_stochastic` | `int` | Particle dimension (= number of stochastic states). |
| `stochastic_indices` | `tuple[int, ...]` | Indices of stochastic states in the full state vector. |
| `state_bounds` | `tuple[(float, float), ...]` | `(lo, hi)` per stochastic state. |
| `param_prior_config` | `OrderedDict` | Name → `(dist_type, args)` for every estimated parameter. |
| `init_state_prior_config` | `OrderedDict` | Name → `(dist_type, args)` for every estimated initial state. |
| `frozen_params` | `dict` | Simulator parameters **not** estimated (e.g. `dt_hours`, frozen constants). |
| `propagate_fn` | JAX callable | Single-step PF propagation. |
| `diffusion_fn` | JAX callable | `(params_jax) -> jnp.ndarray(n_states,)` — noise diagonal. |
| `obs_log_weight_fn` | JAX callable | `(x_new, grid_obs, step_k, params) -> scalar log-weight`. |
| `align_obs_fn` | numpy callable | Converts simulator channel dict into JAX arrays on the time grid. |
| `shard_init_fn` | JAX callable | Phase-conditioned initial-state sampler. |

Optional fields (use as needed):

| Field | Purpose |
|:---|:---|
| `imex_step_fn`, `obs_log_prob_fn` | Direct-scan log-density path (v6.0+). |
| `make_init_state_fn` | Build full y0 from estimated init values. |
| `obs_sample_fn` | Sampling counterpart of `obs_log_prob_fn`. |
| `gaussian_obs_fn`, `init_cov_fn` | EKF likelihood (v6.4+). |
| `dynamic_kernel_log_density_fn`, `proposal_log_density_fn` | Marginal-SGR kernel. |
| `plot_trajectory_fn`, `plot_residuals_fn` | Posterior visualisation. |
| `forward_sde_fn` | MAP-trajectory integrator. |
| `get_init_theta_fn` | Initial θ vector for the sampler. |
| `load_data_fn` | Custom I/O for non-simulator data. |
| `exogenous_keys` | Tuple of `grid_obs` keys that are broadcast (not time-indexed). |

Derived properties (computed): `n_params`, `n_init_states`, `n_dim`, `all_names`, `param_keys`, `param_idx`.

**The import-time sanity check.**  After you build your `EstimationModel`, confirm `n_dim == len(PARAM_PRIOR_CONFIG) + len(INIT_STATE_PRIOR_CONFIG)`. A mismatch means an ordering bug.

---

## 3. `PARAM_PRIOR_CONFIG` in detail

```python
from collections import OrderedDict
import math

PARAM_PRIOR_CONFIG = OrderedDict([
    # Each entry: (name, (dist_type, args))
    ('kappa',    ('lognormal', (math.log(6.67), 0.5))),   # median 6.67, sigma 0.5 in log-space
    ('lmbda',    ('lognormal', (math.log(32.0), 0.5))),
    ('phi',      ('vonmises',  (-math.pi / 3.0, 4.0))),   # peak phase, concentration
    ('HR_base',  ('normal',    (50.0, 5.0))),             # mean, std
    ('W_0_mix',  ('beta',      (3.0, 3.0))),              # alpha, beta
])
```

Supported distributions:

| `dist_type` | `args` | Support | Notes |
|:---|:---|:---:|:---|
| `'lognormal'` | `(mu, sigma)` of underlying normal | $(0, \infty)$ | Parameterised by the **log-space mean and std**, not the raw mean. Good for scales, timescales, gain coefficients. |
| `'normal'` | `(mean, std)` | $\mathbb{R}$ | For unconstrained quantities. |
| `'vonmises'` | `(mean, concentration)` | $[-\pi, \pi]$ | For phase parameters. Concentration $\to \infty$ approaches a Dirac. |
| `'beta'` | `(alpha, beta)` | $[0, 1]$ | For bounded probabilities and [0,1]-valued states. |

**Insertion order is the parameter-vector order.** The sampler reads keys in dict order to build $\theta \in \mathbb{R}^{n_\mathrm{params}}$. Don't reorder entries after results are reported — it'll scramble everything.

**Frozen params go elsewhere.** Non-estimated simulator keys (`dt_hours`, `t_total_hours`, hard-coded constants) live in `frozen_params = {...}` and are merged into the params dict at runtime.

---

## 4. `INIT_STATE_PRIOR_CONFIG`

```python
INIT_STATE_PRIOR_CONFIG = OrderedDict([
    ('W_0',  ('beta',      (3.0, 3.0))),      # W in [0, 1]
    ('Zt_0', ('normal',    (3.5, 0.8))),
    ('a_0',  ('lognormal', (math.log(0.5), 0.5))),
])
```

Same distribution vocabulary as `PARAM_PRIOR_CONFIG`. Only **estimated** initial states belong here. Constants / Phase-1-frozen states (e.g. $V_h$, $V_n$ in the 20p model where they're estimated as parameters) stay in `PARAM_PRIOR_CONFIG` or `frozen_params`.

---

## 5. The JAX callbacks

All must be pure JAX (no `numpy`, no `math.*`, no `float()` casts) because they're called inside a `jax.lax.scan`.

### 5.1 `propagate_fn`

```python
def propagate_fn(y, t, dt, params, grid_obs, step_k, sigma_diag, noise, rng_key):
    # y: jnp.ndarray(n_states,) — state at time t
    # t, dt: scalars
    # params: jnp-dict with PARAM_PRIOR_CONFIG + frozen_params merged
    # grid_obs: dict of jnp arrays (align_obs_fn output), time-indexed or broadcast
    # step_k: scalar — current grid step index
    # sigma_diag: jnp.ndarray(n_states,) — from diffusion_fn(params)
    # noise: jnp.ndarray(n_states,) — unit-variance noise for this step
    # rng_key: unused key (kept for signature compatibility)
    # RETURN: (x_new: jnp.ndarray(n_states,), pred_lw: scalar)
    ...
```

Typical implementation: Euler-Maruyama or IMEX step using your `_dynamics.drift_jax`. `pred_lw` is the log-weight contribution from a proposal-distribution mismatch (often `0.0` for bootstrap).

### 5.2 `diffusion_fn`

Same shape as the simulator-side one, but **JAX**:

```python
def diffusion_fn(params):
    return jnp.array([
        jnp.sqrt(2.0 * params['T_W']),
        jnp.sqrt(2.0 * params['T_Z']),
        jnp.sqrt(2.0 * params['T_a']),
    ])
```

### 5.3 `obs_log_weight_fn`

```python
def obs_log_weight_fn(x_new, grid_obs, step_k, params):
    # x_new: jnp.ndarray(n_states,) — proposed next state
    # grid_obs: dict — must contain channel values + 'present' masks at step_k
    # RETURN: scalar log p(y_k | x_new)
    hr_pred = params['HR_base'] + params['alpha_HR'] * x_new[0]
    hr_obs  = grid_obs['hr_value'][step_k]
    present = grid_obs['hr_present'][step_k]
    ll_hr = -0.5 * ((hr_obs - hr_pred) / params['sigma_HR'])**2 - jnp.log(params['sigma_HR']) - HALF_LOG_2PI
    return present * ll_hr   # mask out absent observations
```

### 5.4 `align_obs_fn` (numpy)

This one is numpy, called once at setup (not inside `scan`):

```python
def align_obs_fn(obs_data, t_steps, dt_hours):
    # obs_data: dict of simulator channel outputs (the dicts returned by generate_fn)
    # t_steps: int — number of grid steps
    # dt_hours: float — grid step
    # RETURN: dict of ndarrays, each shape (t_steps,) or (t_steps, k), keyed by grid_obs name.
    hr_value   = np.zeros(t_steps, dtype=np.float32)
    hr_present = np.zeros(t_steps, dtype=np.float32)
    if 'hr' in obs_data:
        idx = obs_data['hr']['t_idx']
        hr_value[idx] = obs_data['hr']['hr_value']
        hr_present[idx] = 1.0
    return {'hr_value': hr_value, 'hr_present': hr_present}
```

Rule of thumb: emit one `<channel>_value` array plus one `<channel>_present` mask per channel.

### 5.5 `shard_init_fn`

Samples phase-conditioned initial states for parallel shards (used by rolling SMC²). Signature:

```python
def shard_init_fn(time_offset, params, exogenous, global_init):
    # time_offset: scalar — where this shard starts in the time axis
    # params: jnp-dict
    # exogenous: dict of frozen exogenous inputs
    # global_init: jnp.ndarray — shared starting point (typically prior mean)
    # RETURN: jnp.ndarray(n_stochastic,) — initial particle state
    ...
```

For models without phase / shard structure, return `global_init` unchanged.

---

## 6. The `_dynamics.py` split

When `estimation.py` grows past ~200 lines, move the JAX helpers into `_dynamics.py`. Pattern from `sleep_wake_20p` and `swat`:

- `_dynamics.py` — pure JAX functions: `sigmoid`, `drift`, `diffusion`, `imex_step_deterministic`, `imex_step_stochastic`, `hr_mean`, `sleep_prob`, etc.
- `estimation.py` — priors, the `EstimationModel` object, and thin wrappers that call into `_dynamics`.

The estimator calls `estimation.py`; `estimation.py` imports `_dynamics` and wires the JAX helpers into `propagate_fn`, `obs_log_weight_fn`, etc.

Don't pre-split. Only refactor when `estimation.py` stops being readable in one screen.

---

## 7. Verifying your estimation model

Three cheap sanity checks before running an expensive sampler:

```python
# 1. Dimensionality
from models.your_model.estimation import YOUR_ESTIMATION as E
print(f"n_dim = {E.n_dim} ({E.n_params} params + {E.n_init_states} init)")
# Expected: matches the sum of your two OrderedDict lengths.

# 2. Single-step propagate is stable
import jax.numpy as jnp
y0 = jnp.zeros(E.n_stochastic)
params = {**dict(zip(E.param_keys, jnp.ones(E.n_params))), **E.frozen_params}
sigma = E.diffusion_fn(params)
y1, _ = E.propagate_fn(y0, 0.0, 0.1, params, {}, 0, sigma,
                       jnp.zeros(E.n_stochastic), jax.random.PRNGKey(0))
assert jnp.all(jnp.isfinite(y1)), "propagate produced non-finite state"

# 3. Log-weight is finite on a synthetic obs
lw = E.obs_log_weight_fn(y1, synthetic_grid_obs, 0, params)
assert jnp.isfinite(lw), "obs_log_weight is non-finite"
```

If any of these fail, fix before wiring in the sampler.

---

## 8. Pitfalls

| Symptom | Cause | Fix |
|:---|:---|:---|
| `n_dim` mismatch on import | Extra entry in one of the two OrderedDicts | Count entries carefully; look for duplicates. |
| `NaN` propagating inside the particle filter | `jnp.sqrt(negative)` or `jnp.log(0)` in drift | Clamp with `jnp.maximum` / add small regularisation. |
| `TracerArrayConversionError` from numpy | Using `np.` or `math.` inside a JAX-compiled function | Rewrite with `jnp.*`. |
| Parameter-vector scramble between runs | Changed `PARAM_PRIOR_CONFIG` key order | Only append at the end; never reorder. |
| `lognormal` prior on a negative-valued parameter | Support mismatch | Use `normal` instead (lognormal is strictly positive). |
| `PARAM_SET_A` has extra keys not in `PARAM_PRIOR_CONFIG` | That's fine for the simulator; put them in `frozen_params` for the estimator. |
| Estimator says "missing frozen_params['dt_hours']" | Forgot to move `dt_hours` / `t_total_hours` from simulation to `frozen_params` | Copy them across. |

---

**Next:** [03_testing_and_docs.md](03_testing_and_docs.md) for `TESTING.md` conventions and the user-facing doc page.

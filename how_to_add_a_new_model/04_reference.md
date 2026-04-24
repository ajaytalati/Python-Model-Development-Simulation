# 04 — Reference appendix

Lookup tables only. No narrative. Use when the other guide files say "see reference."

Contents:
1. Every `SDEModel` field.
2. Every `StateSpec` and `ChannelSpec` field.
3. Every callback signature.
4. Every CLI flag.
5. Output file layout.
6. Diffusion-type cheat sheet.
7. State-ordering invariants.
8. Known gotchas / FAQ.

---

## 1. Every `SDEModel` field

Source: [`version_1/simulator/sde_model.py`](../version_1/simulator/sde_model.py).

| Field | Type | Required | Default | Used by |
|:---|:---|:---:|:---|:---|
| `name` | `str` | ✓ | — | CLI output directory, plot titles |
| `version` | `str` | ✓ | — | CLI metadata |
| `states` | `tuple[StateSpec, ...]` | ✓ | — | solver (state bounds, deterministic indices) |
| `drift_fn` | callable | ✓ | — | scipy solver |
| `diffusion_type` | `str` | | `'diagonal_constant'` | both solvers |
| `diffusion_fn` | callable | — (see §6) | `None` | both solvers |
| `noise_scale_fn` | callable | req. if `diffusion_type == 'diagonal_state'` | `None` | scipy solver |
| `noise_scale_fn_jax` | callable | same as above for Diffrax | `None` | Diffrax solver |
| `drift_fn_jax` | callable | | `None` | Diffrax solver (scipy falls back if absent) |
| `make_aux_fn` | callable | | `None` | scipy solver; `None` aux passed to drift |
| `make_aux_fn_jax` | callable | | `None` | Diffrax solver |
| `make_y0_fn` | callable | | `None` | both solvers |
| `channels` | `tuple[ChannelSpec, ...]` | | `()` | `sde_observations.generate_all_channels` |
| `plot_fn` | callable | | `None` | CLI |
| `csv_writer_fn` | callable | | `None` | CLI |
| `verify_physics_fn` | callable | | `None` | CLI `--verify` + post-run |
| `param_sets` | `dict[str, dict]` | ✓ | `None` | CLI |
| `init_states` | `dict[str, dict]` | ✓ | `None` | CLI |
| `exogenous_inputs` | `dict[str, dict]` | | `None` | CLI; passed to `make_aux_fn` |

Derived properties (computed, do not set): `n_states`, `state_names`, `bounds`, `deterministic_indices`, `stochastic_indices`.

---

## 2. Every `StateSpec` and `ChannelSpec` field

### `StateSpec`

| Field | Type | Required | Meaning |
|:---|:---|:---:|:---|
| `name` | `str` | ✓ | State name (appears in CLI output and plots). |
| `lower_bound` | `float` | ✓ | Hard clip applied every substep. |
| `upper_bound` | `float` | ✓ | Hard clip applied every substep. |
| `is_deterministic` | `bool` | | Default `False`; if `True` the state is overwritten each substep by `analytical_fn`. |
| `analytical_fn` | callable | (if deterministic) | `(t: float, params: dict) -> float`. |
| `analytical_fn_jax` | callable | (deterministic + Diffrax) | `(t: jnp.ndarray, params_jax: dict) -> jnp.ndarray`. |

### `ChannelSpec`

| Field | Type | Required | Meaning |
|:---|:---|:---:|:---|
| `name` | `str` | ✓ | Channel identifier; becomes `channel_<name>.npz`. |
| `depends_on` | `tuple[str, ...]` | | Default `()`. Names of channels whose outputs this channel reads. |
| `generate_fn` | callable | ✓ | `(trajectory, t_grid, params, aux, prior_channels, seed) -> dict`. |

---

## 3. Every callback signature

Copy-pasteable.

```python
# Drift (numpy)
def drift_fn(t, y, params, aux) -> np.ndarray: ...
#   t: float
#   y: ndarray(n_states,)
#   params: dict
#   aux: any | None
#   returns: ndarray(n_states,)

# Drift (JAX)
def drift_fn_jax(t, y, args) -> jnp.ndarray: ...
#   t: scalar
#   y: jnp.ndarray(n_states,)
#   args: whatever make_aux_fn_jax returned (tuple; args[0] is params dict by convention)
#   returns: jnp.ndarray(n_states,)

# Diffusion
def diffusion_fn(params) -> np.ndarray: ...
#   returns: ndarray(n_states,)   — per-state sigma

# Noise scale (state-dependent diffusion only)
def noise_scale_fn(y, params) -> np.ndarray: ...
def noise_scale_fn_jax(y, params_jax) -> jnp.ndarray: ...
#   returns: ndarray(n_states,)   — per-state g_i(y, params)

# Aux builder
def make_aux_fn(params, init_state, t_grid, exogenous) -> Any: ...
def make_aux_fn_jax(params, init_state, t_grid, exogenous) -> Any: ...

# Initial state
def make_y0_fn(init_state_dict, params) -> np.ndarray: ...
#   returns: ndarray(n_states,)   — in states-tuple order

# Analytical (deterministic states)
def analytical_fn(t, params) -> float: ...
def analytical_fn_jax(t, params_jax) -> jnp.ndarray: ...

# Channel generation
def generate_fn(trajectory, t_grid, params, aux, prior_channels, seed) -> dict: ...
#   trajectory: ndarray(T, n_states)
#   t_grid: ndarray(T,)
#   prior_channels: dict[str, dict[str, ndarray]]
#   seed: int
#   returns: dict with at least 't_idx' or 't_hours'

# Physics
def verify_physics_fn(trajectory, t_grid, params) -> dict: ...

# Plot
def plot_fn(trajectory, t_grid, channel_outputs, params, save_dir) -> None: ...

# CSV export (flexible)
def csv_writer_fn(trajectory, t_grid, channel_outputs, params, save_dir,
                  init_state=None, exogenous=None, meta=None) -> None: ...
```

Estimation-side callables (see [02_estimation.md](02_estimation.md) §5):

```python
def propagate_fn(y, t, dt, params, grid_obs, step_k, sigma_diag, noise, rng_key)
    -> (x_new: jnp.ndarray(n_states,), pred_lw: scalar)

def diffusion_fn(params) -> jnp.ndarray(n_states,)           # JAX

def obs_log_weight_fn(x_new, grid_obs, step_k, params) -> scalar

def align_obs_fn(obs_data, t_steps, dt_hours) -> dict[str, np.ndarray]  # numpy

def shard_init_fn(time_offset, params, exogenous, global_init) -> jnp.ndarray
```

---

## 4. Every CLI flag

Source: [`version_1/simulator/run_simulator.py`](../version_1/simulator/run_simulator.py) `_parse_args`.

| Flag | Default | Meaning |
|:---|:---|:---|
| `--model <dotted.path>` | `models.sleep_wake.simulation.SLEEP_WAKE_MODEL` | Dotted path to the `SDEModel` object to simulate. |
| `--param-set <letter>` | `'A'` | Which named set to run. Uppercased. |
| `--seed <int>` | `42` | Master RNG. Split into `sde_seed` and `obs_seed`. |
| `--substeps <int>` | `10` | Euler-Maruyama substeps per grid interval. |
| `--cross-validate` | off | Zeroize noise, run scipy + Diffrax, compare. Early-exit. |
| `--verify` | off | Zeroize noise, integrate deterministic skeleton, run `verify_physics_fn`. Early-exit. |
| `--scipy` | off (diffrax default) | Force scipy Euler-Maruyama. |
| `--diffrax` | default | Use Diffrax (auto-falls-back to scipy if unavailable). |
| `--out-dir <path>` | `'outputs'` | Output root directory. |

The CLI reads `dt_hours` and `t_total_hours` from `params` to build the time grid. Defaults: `dt_hours=5/60`, `t_total_hours=9*24`.

Noise-zeroing pattern (used by `--cross-validate` and `--verify`): for every state with `.name == X`, any key in `params` matching `T_X` or `sigma_X` is set to `0.0`.

---

## 5. Output file layout

Directory: `outputs/synthetic_<model.name>_<param_set>_<YYYYMMDD_HHMMSS>/`.

| File | Contents (NPZ keys) |
|:---|:---|
| `synthetic_truth.npz` | `true_trajectory` (T, n_states) · `t_grid` (T,) · `true_params` (values) · `true_param_names` (keys) · `true_init_states` (values) · `true_init_names` (keys) |
| `channel_<name>.npz` | One file per channel, keys are whatever the channel's `generate_fn` returned (at minimum `t_idx` or `t_hours`). |
| `synthetic_exogenous.npz` | Only present when `exogenous` is non-empty and its values are array-serialisable. Keys match `exogenous` dict. |
| `latent_states.png`, `observations.png`, … | Produced by `plot_fn`. |
| `garmin_csv/` | Produced by `csv_writer_fn` when present. |

---

## 6. Diffusion type cheat sheet

| I want… | Use `diffusion_type` | `diffusion_fn` returns | `noise_scale_fn` returns |
|:---|:---|:---|:---|
| Constant Gaussian noise per state | `DIFFUSION_DIAGONAL_CONSTANT` | `ndarray(n_states,)` of sigmas | — (not called) |
| Jacobi diffusion on $[0, 1]$ | `DIFFUSION_DIAGONAL_STATE` | scalar sigmas | `sqrt(y * (1 - y))` |
| CIR / square-root diffusion on $[0, \infty)$ | `DIFFUSION_DIAGONAL_STATE` | scalar sigmas | `sqrt(max(y, 0))` |
| Regularised Landau near reflecting boundary | `DIFFUSION_DIAGONAL_STATE` | scalar sigmas | `sqrt(max(y, 0) + eps)` |
| No noise at all (ODE) | `DIFFUSION_DIAGONAL_CONSTANT` | zeros | — |

Per-step update: $y_{k+1}^{(i)} = y_k^{(i)} + \Delta t \cdot f_i + \sigma_i \cdot g_i(y_k) \cdot \sqrt{\Delta t} \cdot \xi_i$ where $g_i \equiv 1$ for CONSTANT and $g_i$ comes from `noise_scale_fn` for STATE.

---

## 7. State-ordering invariants

Two non-negotiable rules:

1. **`drift_fn` indexing matches `states` tuple order.** If `states = (StateSpec('W', ...), StateSpec('Z', ...))`, then `y[0]` is W and `y[1]` is Z *everywhere* — in drift, diffusion, plots, make_y0, saved NPZ columns.
2. **`make_y0_fn` produces a vector in that same order.** Read from the init-state dict by name, pack into the array in the order of `states`.

Breaking either invariant produces silent wrong answers. Convention: define `states` once at the top of `simulation.py` and refer to the state vector by index *only* in drift/diffusion; use named dict access elsewhere.

---

## 8. Known gotchas / FAQ

Flat list keyed by symptom.

| Symptom | Resolution |
|:---|:---|
| `AttributeError: module ... has no attribute '<NAME>_MODEL'` | Typo in `--model` dotted path, or missed rename in `__init__.py`. |
| `KeyError: 'dt_hours'` | Add `dt_hours` and `t_total_hours` to your `PARAM_SET_A`. |
| `ValueError: param set 'X' not found` | Keys in `param_sets`, `init_states`, `exogenous_inputs` must match. |
| NaN trajectories | Cast all numeric params to `float` in `PARAM_SET_A`; avoid `int`. |
| `TracerArrayConversionError` from JAX | `numpy` or `math` used inside a JAX-compiled function; rewrite with `jnp.*`. |
| scipy vs JAX drift disagreement | `drift_jax` uses wrong namespace; verify with the cross-validation snippet in [01_simulation.md §2.2](01_simulation.md#22-drift_fn_jax-optional--enables-diffrax). |
| `synthetic_exogenous.npz` missing despite `EXOGENOUS_A` entries | Values aren't array-serialisable; stick to floats/ints/ndarrays. |
| `ValueError: circular dependency in channels` | `depends_on` forms a cycle; fix the graph. |
| Template run succeeds but no plot appears | `plot_fn` was forgotten on the `SDEModel` kwargs. |
| CLI `--cross-validate` errors with module-not-found | `diffrax` not installed; use the manual drift-agreement snippet instead. |
| `--verify` says noise is non-zero | Noise-key naming doesn't match `T_<state>` or `sigma_<state>` pattern; rename the noise params. |
| State range always saturates to bounds | Drift blow-up; check first-step output; widen bounds or reduce timestep. |
| Estimation: `n_dim` mismatch | Count entries in both OrderedDicts; look for duplicates. |
| Estimation: parameter ordering scrambled between runs | Never reorder `PARAM_PRIOR_CONFIG` — append only. |
| Estimation: `lognormal` prior fails on a parameter that can be negative | Use `normal` (lognormal support is strictly positive). |

---

**Back to:** [README](README.md) · [01_simulation](01_simulation.md) · [02_estimation](02_estimation.md) · [03_testing_and_docs](03_testing_and_docs.md) · [worked_example](worked_example.md)

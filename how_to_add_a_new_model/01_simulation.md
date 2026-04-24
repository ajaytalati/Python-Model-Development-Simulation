# 01 — Building a simulation

The central file of the guide. Four parts:
1. **§1** — the `SDEModel` contract (what fields exist, what's required vs optional).
2. **§2** — every required and optional callback, with signature + realistic snippet.
3. **§3** — step-by-step walk-through of building a simulation from `_template_model/`.
4. **§4–§6** — parameter sets, what the CLI does, and the top ten pitfalls.

---

## 1. The SDEModel contract

The contract lives in [`version_1/simulator/sde_model.py`](../version_1/simulator/sde_model.py) as three frozen dataclasses: `StateSpec`, `ChannelSpec`, `SDEModel`. Your model is done when you have one `SDEModel` object populated with the fields below.

### 1.1 `StateSpec` — one per state variable

| Field | Type | Required | Meaning |
|:---|:---|:---:|:---|
| `name` | `str` | ✓ | Human-readable state name; appears in CLI output and plots. |
| `lower_bound` | `float` | ✓ | Hard clip applied after every substep. |
| `upper_bound` | `float` | ✓ | Hard clip applied after every substep. |
| `is_deterministic` | `bool` | | If `True`, the state is overwritten analytically each substep instead of integrated. |
| `analytical_fn` | `callable` | | Numpy-side overwrite: `(t: float, params: dict) -> float`. |
| `analytical_fn_jax` | `callable` | | JAX-side overwrite for Diffrax: `(t: jnp.ndarray, params_jax: dict) -> jnp.ndarray`. Use only `jnp.*`. |

**State ordering is load-bearing.** The order of `StateSpec`s in the `states` tuple defines `y[0]`, `y[1]`, `y[2]`, ... everywhere — in `drift_fn`, `diffusion_fn`, `make_y0_fn`, plots, `synthetic_truth.npz`. Pick an order and keep it.

### 1.2 `ChannelSpec` — one per observation channel

| Field | Type | Required | Meaning |
|:---|:---|:---:|:---|
| `name` | `str` | ✓ | Channel identifier; becomes `channel_<name>.npz` in output. |
| `depends_on` | `tuple[str, ...]` | | Names of prior channels this channel's `generate_fn` reads. Default `()`. |
| `generate_fn` | `callable` | ✓ | `(trajectory, t_grid, params, aux, prior_channels, seed) -> dict`. |

Channels form a **DAG** via `depends_on`. The framework topologically sorts them, then generates in order, passing already-generated channels as `prior_channels`. Good for exercise-indicator-feeds-heart-rate-scenarios.

### 1.3 `SDEModel` — the full specification

| Field | Type | Required | Meaning |
|:---|:---|:---:|:---|
| `name` | `str` | ✓ | Model name; used as `synthetic_<name>_...` in output directory names. |
| `version` | `str` | ✓ | Free-form version string. |
| `states` | `tuple[StateSpec, ...]` | ✓ | State-variable definitions, **order-sensitive**. |
| `drift_fn` | `callable` | ✓ | Numpy drift: `(t, y, params, aux) -> ndarray(n_states,)`. |
| `diffusion_type` | `str` | | Either `DIFFUSION_DIAGONAL_CONSTANT` (default) or `DIFFUSION_DIAGONAL_STATE`. |
| `diffusion_fn` | `callable` | (see below) | `(params) -> ndarray(n_states,)` per-state σ. |
| `noise_scale_fn` | `callable` | (see below) | `(y, params) -> ndarray(n_states,)`. Required when `diffusion_type == DIFFUSION_DIAGONAL_STATE`. |
| `noise_scale_fn_jax` | `callable` | | JAX variant of the above. |
| `drift_fn_jax` | `callable` | | JAX drift for Diffrax. If absent, solver falls back to scipy. |
| `make_aux_fn` | `callable` | | `(params, init_state, t_grid, exogenous) -> aux`. Defaults to `None` aux. |
| `make_aux_fn_jax` | `callable` | | JAX variant; must be provided alongside `drift_fn_jax`. |
| `make_y0_fn` | `callable` | ✓ (if `init_states` is non-trivial) | `(init_state_dict, params) -> ndarray(n_states,)`. |
| `channels` | `tuple[ChannelSpec, ...]` | | Default `()`. Most models have at least one channel. |
| `plot_fn` | `callable` | | `(trajectory, t_grid, channel_outputs, params, save_dir) -> None`. |
| `csv_writer_fn` | `callable` | | CSV export for pipeline testing. Flexible signature — see [version_1/models/sleep_wake/csv_writer.py](../version_1/models/sleep_wake/csv_writer.py). |
| `verify_physics_fn` | `callable` | | `(trajectory, t_grid, params) -> dict`. Called by `--verify` and after each simulation. |
| `param_sets` | `dict[str, dict]` | ✓ | Keyed by single uppercase letter: `{'A': {...}, 'B': {...}, ...}`. |
| `init_states` | `dict[str, dict]` | ✓ | Same keys as `param_sets`. |
| `exogenous_inputs` | `dict[str, dict]` | | Same keys; empty dict per set if unused. |

Derived properties (computed; don't set): `n_states`, `state_names`, `bounds`, `deterministic_indices`, `stochastic_indices`.

### 1.4 Diffusion types

The SDE update rule is

$$
y_{k+1}^{(i)} = y_k^{(i)} + \Delta t \cdot f_i(y_k) + \sigma_i(\text{params}) \cdot g_i(y_k) \cdot \sqrt{\Delta t} \cdot \xi_i, \qquad \xi_i \sim \mathcal{N}(0, 1).
$$

- **`DIFFUSION_DIAGONAL_CONSTANT`** (default) — $g_i \equiv 1$. `diffusion_fn(params)` returns the full per-state σ vector. Use this for constant-amplitude Gaussian noise.
- **`DIFFUSION_DIAGONAL_STATE`** — $g_i(y)$ is state-dependent. `diffusion_fn(params)` returns the scalar σ multipliers, and `noise_scale_fn(y, params)` returns the state-dependent $g_i(y)$. Typical choices:

| SDE family | `noise_scale_fn` returns | Example |
|:---|:---|:---|
| Jacobi (on $[0, 1]$) | `sqrt(y * (1 - y))` | Fitness state in FSA |
| CIR (on $[0, \infty)$) | `sqrt(y)` | Strain state in FSA |
| Regularised Landau (reflecting at 0) | `sqrt(y + eps)` | Endocrine amplitude |

### 1.5 Deterministic states

Set `is_deterministic=True` for states whose value is given by a closed-form expression of $t$ (e.g. the 24-hour external light cycle $C(t) = \sin(2\pi t / 24 + \phi)$). Provide `analytical_fn` and, if you want Diffrax support, `analytical_fn_jax`. The solver still calls `drift_fn` and `diffusion_fn` for these states, but **overwrites their values with the analytical result after the step**. This means `drift_fn` can return whatever it wants for deterministic-state slots — `0.0` is conventional.

---

## 2. Required and optional callbacks

### 2.1 `drift_fn`

```python
def drift(t, y, params, aux):
    # t: scalar time (hours or your chosen unit)
    # y: ndarray(n_states,) — state at time t
    # params: dict (the active PARAM_SET_A / B / ...)
    # aux: whatever make_aux_fn returned (None if make_aux_fn is absent)
    # RETURN: ndarray(n_states,) of dy/dt
    ...
```

Pure function, **numpy only**, no random state, no side effects. Called at every substep (typically 10 per grid interval). Must be fast.

Example (from [bistable_controlled](../version_1/models/bistable_controlled/simulation.py)):

```python
def drift(t, y, params, aux):
    T_i, u_on = aux
    x, u = y[0], y[1]
    u_target = u_on if t >= T_i else 0.0
    dx = params['alpha'] * x * (params['a']**2 - x**2) + u
    du = -params['gamma'] * (u - u_target)
    return np.array([dx, du])
```

**Gotcha:** if a parameter is accidentally typed as `int` (e.g. `'alpha': 1` not `'alpha': 1.0`), the first step can overflow. Cast params to float in `PARAM_SET_A`, not in the drift.

### 2.2 `drift_fn_jax` (optional — enables Diffrax)

```python
def drift_jax(t, y, args):
    # args: whatever make_aux_fn_jax returned; convention args[0] is the params dict
    # Use only jnp.*; no math.sin, no float() casts, no numpy.
    import jax.numpy as jnp
    p, T_i, u_on = args
    x, u = y[0], y[1]
    u_target = jnp.where(t < T_i, 0.0, u_on)
    dx = p['alpha'] * x * (p['a']**2 - x**2) + u
    du = -p['gamma'] * (u - u_target)
    return jnp.array([dx, du])
```

**Must be line-for-line equivalent to `drift`.** A cheap end-to-end sanity check:

```python
import math, numpy as np, jax.numpy as jnp
from models.your_model.simulation import drift, drift_jax, PARAM_SET_A
p = PARAM_SET_A
p_jax = {k: jnp.float64(v) for k, v in p.items() if not isinstance(v, str)}
for t, y in [(0.0, [0.5, 0.0]), (10.0, [1.0, 0.5])]:
    d_np = drift(t, np.array(y), p, None)
    d_jx = drift_jax(t, jnp.array(y), (p_jax, ...))  # adjust args to match your make_aux_fn_jax
    diff = float(jnp.max(jnp.abs(jnp.array(d_np) - d_jx)))
    print(f"t={t}: max diff {diff:.2e}")
# expected: < 1e-10
```

### 2.3 `diffusion_fn`

```python
def diffusion_diagonal(params):
    # RETURN: ndarray(n_states,) — per-state sigma
    return np.array([
        math.sqrt(2.0 * params['T_W']),
        math.sqrt(2.0 * params['T_Z']),
        # one entry per state, in the same order as the states tuple
    ])
```

For `DIFFUSION_DIAGONAL_CONSTANT` this is the full per-state amplitude. For `DIFFUSION_DIAGONAL_STATE` it is multiplied element-wise by `noise_scale_fn(y, params)` at each step.

**Convention**: follow the literature for noise-temperature notation. Wiener/OU: `sqrt(2 * T)`. CIR: plain `sigma`. Depends on your SDE's canonical form.

### 2.4 `noise_scale_fn` / `noise_scale_fn_jax`

Required when `diffusion_type == DIFFUSION_DIAGONAL_STATE`.

```python
def noise_scale_fn(y, params):
    B, F, A = y
    return np.array([
        math.sqrt(B * (1.0 - B)),           # Jacobi on [0, 1]
        math.sqrt(max(F, 0.0)),              # CIR
        math.sqrt(max(A, 0.0) + 1.0e-4),     # regularised Landau
    ])
```

JAX variant uses `jnp.sqrt`, `jnp.maximum`.

### 2.5 `make_aux_fn` / `make_aux_fn_jax`

```python
def make_aux(params, init_state, t_grid, exogenous):
    # Prebuild anything the drift needs that doesn't fit in params.
    # Common uses: daily schedules, lookup arrays, exogenous constants.
    return (exogenous['T_intervention'], exogenous['u_on'])

def make_aux_jax(params, init_state, t_grid, exogenous):
    import jax.numpy as jnp
    p_jax = {k: jnp.float64(v) for k, v in params.items() if not isinstance(v, str)}
    return (p_jax, jnp.float64(exogenous['T_intervention']), jnp.float64(exogenous['u_on']))
```

**Convention**: when you pair `_jax` and non-`_jax`, the first tuple element of the JAX return value is the params dict. The solver uses this convention to overwrite deterministic-state slots after each step.

Return `None` from `make_aux_fn` when you don't need aux data; your drift must handle `aux=None`.

### 2.6 `make_y0_fn`

```python
def make_y0(init_dict, params):
    # Map named initial-state dict to a vector in `states` order.
    # Deterministic states should be initialised from the analytical function
    # at t=0.
    C0 = math.sin(params['phi'])   # for a deterministic circadian state
    return np.array([
        init_dict['W_0'], init_dict['Z_0'], init_dict['A_0'],
        C0,
        init_dict['Vh_0'], init_dict['Vn_0'],
    ])
```

### 2.7 `verify_physics_fn`

```python
def verify_physics(trajectory, t_grid, params):
    W = trajectory[:, 0]
    A = trajectory[:, 2]
    wake = W > 0.5
    sleep = W < 0.5
    return {
        'W_range':                       float(W.max() - W.min()),
        'A_mean_wake':                   float(A[wake].mean()) if wake.any() else float('nan'),
        'A_mean_sleep':                  float(A[sleep].mean()) if sleep.any() else float('nan'),
        'adenosine_builds_during_wake':  bool(A[wake].mean() > A[sleep].mean()),
        'all_finite':                    bool(np.all(np.isfinite(trajectory))),
    }
```

Return a dict. Keys named `<thing>_ok` or returning bool get a `PASS`/`FAIL` tag in CLI output; other values are printed raw. Use this to encode qualitative physics expectations (period, effect sizes, bounds).

### 2.8 `plot_fn`

```python
def plot_<model>(trajectory, t_grid, channel_outputs, params, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Produce one or more PNGs in save_dir.
    # Convention: latent_states.png (one panel per state), observations.png
    # (each channel vs its source state).
```

Plots are optional but strongly recommended — they're the first thing anyone checks when something looks off.

### 2.9 `csv_writer_fn`

Optional. Used by [`models/sleep_wake/csv_writer.py`](../version_1/models/sleep_wake/csv_writer.py) to export Garmin-format CSVs for end-to-end pipeline testing. Signature is flexible:

```python
def write_csvs(trajectory, t_grid, channel_outputs, params, save_dir,
               init_state=None, exogenous=None, meta=None):
    # CLI will try with kwargs first, then fall back to positional.
    ...
```

Skip unless you specifically need CSV pipeline integration.

### 2.10 Channel `generate_fn`

```python
def gen_hr(trajectory, t_grid, params, aux, prior_channels, seed):
    # prior_channels: dict[str, dict[str, ndarray]] — outputs of earlier channels.
    # seed: int — unique per channel, use it to make observations reproducible.
    # RETURN: dict with at least 't_idx' or 't_hours' key.
    rng = np.random.default_rng(seed)
    W = trajectory[:, 0]
    hr = params['HR_base'] + params['alpha_HR'] * W + rng.normal(0, params['sigma_HR'], len(t_grid))
    return {
        't_idx':   np.arange(len(t_grid), dtype=np.int32),
        'hr_bpm':  hr.astype(np.int32),
    }
```

A channel can:
- Sample at a different cadence than the simulation grid (e.g. 1-minute HR on a 5-minute grid via interpolation).
- Read another channel's output via `prior_channels` — declare the dependency in the `ChannelSpec`'s `depends_on`.
- Return arbitrary extra keys (the framework only requires an index key for alignment).

---

## 3. Walk-through: build a simulation from the template

After `cp -r version_1/models/_template_model version_1/models/<your_name>`:

1. **Rename identifiers.** Inside your new directory:
   - `__init__.py`: change `TEMPLATE_MODEL` → `<YOUR_NAME>_MODEL` and update the import path.
   - `simulation.py`: change `TEMPLATE_MODEL` → `<YOUR_NAME>_MODEL`, `name="_template_"` → `name="<your_name>"`, and the plotter import.
   - `sim_plots.py`: rename `plot_template` → `plot_<your_name>`.
   - `README.md`: delete it.

2. **Confirm the smoke test still runs.**
   ```bash
   python simulator/run_simulator.py --model models.<your_name>.simulation.<YOUR_NAME>_MODEL --param-set A
   ```
   You should get NPZ + two plots and `converges_to_mu = PASS`. This tells you the wiring is correct before you touch dynamics.

3. **Define your states.** Edit the `states` tuple in `simulation.py`. Write the SDEs on paper first, then transcribe state-by-state. For each state decide: bounds, stochastic vs deterministic, timescale.

4. **Write `drift` (numpy).** One array entry per state, same order as `states`. Ignore the JAX side for now — the scipy solver handles everything if `drift_fn_jax` is absent.

5. **Write `diffusion_diagonal`.** Decide DIAGONAL_CONSTANT (constant σ per state) vs DIAGONAL_STATE (state-dependent multiplier). For DIAGONAL_STATE, also write `noise_scale_fn`.

6. **Fill `PARAM_SET_A` and `INIT_STATE_A`.** Include `dt_hours` and `t_total_hours` in params (the CLI reads them to build the time grid). Leave `EXOGENOUS_A = {}` until you need schedules.

7. **Update `make_y0_fn`.** Must emit entries in `states` order. If you have deterministic states, initialise them from the analytical function at `t=0`.

8. **Adjust the observation channel.** Edit `gen_obs` to generate the observation your model actually produces. Add additional `ChannelSpec`s for extra channels.

9. **Run again.** Check state ranges in the CLI output. They should be physical — e.g. a wake state $\in [0, 1]$, not saturating. If not, adjust gain parameters (sigmoid slopes, coupling strengths).

10. **Write `verify_physics`.** Encode the qualitative checks you'd do by eye. Re-run with `--verify` to confirm the deterministic skeleton satisfies them.

11. **Add a second parameter set** (`B`, typically a pathological variant) so you can compare. Keys in `param_sets`, `init_states`, `exogenous_inputs` must all include `'B'`.

12. **(Later, optional) Add JAX drift** for the Diffrax solver. Write `drift_jax` and `make_aux_jax` alongside the numpy versions. Cross-validate with the snippet in §2.2.

See [worked_example.md](worked_example.md) for a complete transcript — every file contents, every shell command, every expected output.

---

## 4. Parameter sets, initial states, exogenous inputs

Three dicts on the `SDEModel` object — one entry per scenario — **with matching keys**:

```python
TEMPLATE_MODEL = SDEModel(
    ...
    param_sets={'A': PARAM_SET_A, 'B': PARAM_SET_B},
    init_states={'A': INIT_STATE_A, 'B': INIT_STATE_B},
    exogenous_inputs={'A': EXOGENOUS_A, 'B': EXOGENOUS_B},
)
```

Rules:

- **Use uppercase single letters.** `'A'`, `'B'`, `'C'`, `'D'`. Convention — `A` = healthy baseline, `B` = primary pathology, `C` = recovery / alternative, `D` = second pathology.
- **Every key in one dict must exist in the others.** The CLI will error otherwise.
- **`EXOGENOUS_A = {}` is valid.** Use when you have no schedules.
- **`params` must include `dt_hours` and `t_total_hours`.** The CLI builds `t_grid = np.arange(n_steps) * dt` from them. Defaults: `dt_hours=5/60` (5 min), `t_total_hours=216` (9 days).
- **Each set is self-contained.** A reader picking up Set B should understand everything they need from the PARAM_SET_B dict alone.

---

## 5. What the CLI does end-to-end

When you run

```bash
python simulator/run_simulator.py --model models.<m>.simulation.<M>_MODEL --param-set A --seed 42
```

the pipeline in [`run_simulator.py`](../version_1/simulator/run_simulator.py) does:

1. **Parse args** — see [04_reference.md §4](04_reference.md#4-every-cli-flag) for the full list.
2. **Resolve the model** via `importlib.import_module` + `getattr`. If the dotted path is wrong, you get an `AttributeError` listing available objects.
3. **Look up the parameter set.** `model.param_sets['A']` → `params` dict. Missing sets error out with a list of available keys.
4. **Build the time grid** from `params['dt_hours']` and `params['t_total_hours']` (with defaults).
5. **Special modes:**
   - `--cross-validate`: zeroize all `T_*` and `sigma_*` keys in `params` whose suffix matches a state name, then cross-compare scipy vs Diffrax deterministic trajectories. Returns early.
   - `--verify`: same zeroize, integrate the deterministic skeleton once, run `verify_physics_fn`. Returns early.
6. **Split the RNG** into `sde_seed` and `obs_seed`.
7. **Resolve the solver.** Default: Diffrax (if `drift_fn_jax` and `make_aux_fn_jax` are present and JAX is installed); otherwise scipy Euler-Maruyama fallback.
8. **Integrate the SDE.** Trajectory shape is `(n_steps, n_states)`.
9. **Generate channels.** `generate_all_channels` topologically sorts by `depends_on` and runs each `generate_fn` in order, passing previously-generated outputs.
10. **Save outputs** to `outputs/synthetic_<name>_<set>_<timestamp>/`:
    - `synthetic_truth.npz` — full trajectory, t_grid, param names/values, init-state names/values.
    - `channel_<name>.npz` — one file per channel with the dict returned by `generate_fn`.
    - `synthetic_exogenous.npz` — if `exogenous` is non-empty and serialisable.
11. **Verify physics.** `verify_physics_fn` → printed table.
12. **Plot.** `plot_fn(trajectory, t_grid, channel_outputs, params, out_dir)`.
13. **CSV export.** `csv_writer_fn`, if provided; CLI tries kwargs then positional.

Total wall time for a 9-day, 5-min-grid model with scipy + 4 stochastic states + 2 channels: ~1.5 s.

---

## 6. Common pitfalls

| Symptom | Cause | Fix |
|:---|:---|:---|
| `AttributeError: module ... has no attribute '<NAME>_MODEL'` | Typo in `--model` dotted path | Echo `print(dir(module))` to see what's actually exported. |
| NaN trajectories after step 1 | Drift sees an integer parameter | Cast to float in `PARAM_SET_A` (`'alpha': 1.0`, not `'alpha': 1`). |
| scipy vs JAX drift disagreement | `math.sin` / `numpy.sin` used in `drift_jax` | Use only `jnp.*` inside `drift_fn_jax` and `analytical_fn_jax`. |
| State range saturates to its bound every step | Bounds too tight, or drift blows up | Widen bounds; lower diffusion temperature; inspect the first few steps of `trajectory`. |
| `--cross-validate` errors with shape mismatch | `drift_jax` returns a list or scalar | Wrap in `jnp.array(...)` and ensure shape is `(n_states,)`. |
| `ValueError: circular dependency` when generating channels | `depends_on` references a channel that references back | Break the cycle; channels form a DAG. |
| `KeyError: 'dt_hours'` | `PARAM_SET_A` missing the grid keys | Add `'dt_hours'` and `'t_total_hours'` to your params dict. |
| Deterministic state drift doesn't match analytical | `analytical_fn` and `drift_fn` disagree at boundaries | Set `drift_fn` to return `0.0` for deterministic-state slots; the solver overwrites them anyway. |
| JAX complains about params dict | `make_aux_fn_jax` returns raw Python floats | Convert: `{k: jnp.float64(v) for k, v in params.items() if not isinstance(v, str)}`. |
| `synthetic_exogenous.npz` missing | Something in `exogenous` isn't array-serialisable | Stick to floats, ints, and ndarrays in `EXOGENOUS_A`. |
| Plots not produced | `plot_fn` field is `None` on the model object | Check you passed `plot_fn=plot_<your>` to `SDEModel(...)`. |

See [04_reference.md §8](04_reference.md#8-known-gotchas--faq) for the full FAQ list.

---

**Next:** for Bayesian inference, see [02_estimation.md](02_estimation.md). For tests and docs, see [03_testing_and_docs.md](03_testing_and_docs.md).

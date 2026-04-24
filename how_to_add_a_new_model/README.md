# How to add a new model

This guide is the **first and definitive resource** for adding a new SDE model to the repo. Follow it end-to-end and you'll produce: a working simulation, a reproducible test suite, an estimation module (optional), and a user-facing doc page — all matching the conventions of the six existing models.

Assume you have a mathematical specification (SDEs, observation model, priors) in hand. The guide walks you from that specification to a running, tested, documented model.

---

## Who this is for

- A **human engineer** adding a new model to the framework.
- A **coding agent** doing the same.

You need Python / numpy familiarity and a mathematical spec. You do **not** need prior knowledge of this repo — the guide assumes you've seen only `version_1/simulator/`.

## Prerequisites

- Python ≥ 3.10.
- Required: `numpy`, `scipy`, `matplotlib`.
- Optional: `jax`, `jaxlib` (for the Diffrax solver path and drift cross-validation). The scipy path works without them.
- Read access to [`version_1/simulator/`](../version_1/simulator/) — the five files in there define the entire model contract. Nothing else in the repo is needed to build a simulation.

## The 7-step workflow

1. **Copy the template.** `cp -r version_1/models/_template_model version_1/models/<your_name>`, then rename `TEMPLATE` / `_template_` everywhere inside (three files touch it: `simulation.py`, `__init__.py`, `sim_plots.py`). Smoke-test immediately: `python simulator/run_simulator.py --model models.<your_name>.simulation.<YOUR_NAME>_MODEL --param-set A`. It should produce NPZ + plots out of the box.
2. **Fill in states, drift, diffusion, parameters in `simulation.py`.** → [01_simulation.md §3](01_simulation.md#3-walk-through-build-a-simulation-from-the-template)
3. **Wire up observation channels.** → [01_simulation.md §2.10](01_simulation.md#210-channel-generate_fn)
4. **Run iteratively until trajectories look physical.** Use `--verify` and plot inspection to shape your parameter sets. → [01_simulation.md §5](01_simulation.md#5-what-the-cli-does-end-to-end)
5. **Add `verify_physics_fn` and write `TESTING.md`.** → [03_testing_and_docs.md](03_testing_and_docs.md)
6. **(Optional) Write `estimation.py` for Bayesian inference.** → [02_estimation.md](02_estimation.md)
7. **Add a user-facing doc at `version_1/model_documentation/<your_name>/<Your_Name>_Documentation.md`.** → [03_testing_and_docs.md §4](03_testing_and_docs.md#4-the-per-model-user-facing-doc)

If you want the full end-to-end walkthrough on a concrete example, see [worked_example.md](worked_example.md) — it builds a 2-state Ornstein-Uhlenbeck model in ~15 minutes with every file's contents shown verbatim.

## Big-picture orientation of the simulator

The simulator directory contains five files and nothing else matters for adding a model:

| File | Role |
|:---|:---|
| [`sde_model.py`](../version_1/simulator/sde_model.py) | The **contract**: frozen dataclasses (`SDEModel`, `StateSpec`, `ChannelSpec`) you must populate. |
| [`run_simulator.py`](../version_1/simulator/run_simulator.py) | The **CLI**: argument parsing, model resolution, solver selection, saving outputs. |
| [`sde_solver_scipy.py`](../version_1/simulator/sde_solver_scipy.py) | Default **Euler-Maruyama solver** (numpy). Works with every model. |
| [`sde_solver_diffrax.py`](../version_1/simulator/sde_solver_diffrax.py) | Optional **JAX/Diffrax solver** for JIT/GPU. Requires `drift_fn_jax`. |
| [`sde_observations.py`](../version_1/simulator/sde_observations.py) | **Channel-generation pipeline**: builds a DAG from `depends_on` and runs each `generate_fn` in topological order. |

Everything else in the repo is model code, documentation, or tests. The simulator is completely model-agnostic — all it needs is a populated `SDEModel` object.

## Naming conventions at a glance

| Artefact | Convention | Example |
|:---|:---|:---|
| Directory | `version_1/models/<model_name>/` (snake_case) | `sleep_wake_20p/` |
| Exported simulation object | `<MODEL_NAME>_MODEL` (uppercase) | `SLEEP_WAKE_20P_MODEL` |
| Exported estimation object | `<MODEL_NAME>_ESTIMATION` | `SLEEP_WAKE_20P_ESTIMATION` |
| Parameter sets | `PARAM_SET_A`, `PARAM_SET_B`, ... | `'A'`, `'B'`, `'C'`, `'D'` |
| Initial-state dicts | `INIT_STATE_A`, `INIT_STATE_B`, ... | keys match `param_sets` |
| Exogenous-input dicts | `EXOGENOUS_A`, `EXOGENOUS_B`, ... | keys match `param_sets` |
| Prior configs | `PARAM_PRIOR_CONFIG`, `INIT_STATE_PRIOR_CONFIG` | `OrderedDict` in `estimation.py` |
| Frozen (non-estimated) params | `FROZEN_PARAMS` | plain `dict` |
| Helper-module for estimation | `_dynamics.py` | used when `estimation.py` > ~200 lines |
| Tests + live spec | `TESTING.md` | section structure in [03_testing_and_docs.md §2](03_testing_and_docs.md#2-testingmd-section-template) |
| User-facing doc | `<Model_Name>_Documentation.md` under `version_1/model_documentation/<model_name>/` | `SWAT_Basic_Documentation.md` |

## Checklist

Tick these off as you go. Every item maps to a section of the guide.

- [ ] Directory copied from `_template_model/` and renamed (including all `TEMPLATE` strings inside files).
- [ ] Smoke test passes: `python simulator/run_simulator.py --model models.<you>.simulation.<YOUR>_MODEL --param-set A` produces NPZ + plots.
- [ ] `simulation.py` defines `<YOUR>_MODEL` with all required callbacks (`drift_fn`, `diffusion_fn`, `make_y0_fn`, at least one channel).
- [ ] `PARAM_SET_A`, `INIT_STATE_A`, `EXOGENOUS_A` keys match — and `param_sets`, `init_states`, `exogenous_inputs` dicts on the model object use the same set of keys (`'A'`, …).
- [ ] `verify_physics_fn` returns physics checks and every expected `_ok` key is `True` for Set A.
- [ ] State ranges on the CLI output are physical (no NaNs, no saturation of bounds indicating runaway dynamics).
- [ ] (Optional) `drift_fn_jax` + `make_aux_fn_jax` provided; drift cross-validation shows max diff `< 1e-10` on a few test states (manual snippet in [01_simulation.md §6](01_simulation.md#6-common-pitfalls)).
- [ ] (Optional) `estimation.py` with `PARAM_PRIOR_CONFIG`, `INIT_STATE_PRIOR_CONFIG`, `FROZEN_PARAMS`, and the JAX callbacks.
- [ ] `TESTING.md` filled in with at least §1 Purpose, §2 Math spec, §4 Parameter sets, §5 Tests, §7 Exit criteria.
- [ ] User-facing doc added at `version_1/model_documentation/<your_name>/<Your_Name>_Documentation.md`.
- [ ] Row added to [`version_1/model_documentation/README.md`](../version_1/model_documentation/README.md) index table.
- [ ] One-line mention in the repo-root [`README.md`](../README.md) Models table.

## The guide files

| File | Contents |
|:---|:---|
| [README.md](README.md) | You are here — orientation, workflow, checklist. |
| [01_simulation.md](01_simulation.md) | `SDEModel` contract, every callback signature, step-by-step simulation build, pitfalls. |
| [02_estimation.md](02_estimation.md) | `EstimationModel` contract, prior configs, JAX callbacks, the `_dynamics.py` split. |
| [03_testing_and_docs.md](03_testing_and_docs.md) | `TESTING.md` section-by-section guide, per-model doc page template, discoverability. |
| [04_reference.md](04_reference.md) | Exhaustive field reference, CLI flags, output layout, FAQ. |
| [worked_example.md](worked_example.md) | End-to-end: build a 2-state OU model in 15 minutes. Every file's contents shown. |

Read `README.md` → `worked_example.md` first. Then use `01`–`04` as reference docs while you build your own model.

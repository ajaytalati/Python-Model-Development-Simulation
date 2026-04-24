# Python-Model-Development-Simulation

A framework for developing and testing continuous-time stochastic models of human physiology, designed for **N-of-1 Bayesian inference** from wearable-sensor data.

Every model is a set of coupled Itô stochastic differential equations, a diffusion specification, an observation model, and one or more parameter-set scenarios that exercise healthy and pathological regimes. The generic simulator drives all of them through the same CLI and produces synthetic datasets that can be round-tripped through the estimation pipeline.

## Models

Six models live under [`version_1/models/`](version_1/models/). See [`version_1/model_documentation/README.md`](version_1/model_documentation/README.md) for the full per-model doc index; one-line summaries below:

| Model | Description |
|:---|:---|
| [`bistable_controlled`](version_1/model_documentation/bistable_controlled/bistable_controlled_Documentation.md) | Controlled double-well SDE — pedagogical intervention testbed. |
| [`fitness_strain_amplitude`](version_1/model_documentation/fitness_strain_amplitude/FSA_Basic_Documentation.md) | 3-state Fitness-Strain-Amplitude SDE with direct state observation. |
| [`fsa_real_obs`](version_1/model_documentation/fsa_real_obs/fsa_real_obs_Documentation.md) | FSA with six physiological observation channels (RHR, intensity, duration, stress, sleep, timing). |
| [`sleep_wake`](version_1/model_documentation/sleep_wake/sleep_wake_Documentation.md) | 6-state sleep-wake-adenosine SDE with four Garmin-style channels. |
| [`sleep_wake_20p`](version_1/model_documentation/sleep_wake_20p/sleep_wake_20p_Documentation.md) | Minimal 20-parameter sleep-wake-adenosine — identifiability-proof-driven. |
| [`swat`](version_1/model_documentation/swat/SWAT_Basic_Documentation.md) | Sleep-Wake-Adenosine-Testosterone — extends `sleep_wake_20p` with a Stuart-Landau HPG amplitude. |

## Adding a new model

See [`how_to_add_a_new_model/`](how_to_add_a_new_model/) — a self-contained guide that walks a human or agent from "I have a mathematical spec" to "I have a working, tested, documented model" in ~15 minutes for simple cases. Starts with a copy-pasteable template at [`version_1/models/_template_model/`](version_1/models/_template_model/).

## Running a simulation

From the `version_1/` directory:

```bash
# SWAT, healthy parameter set, reproducible seed
python simulator/run_simulator.py \
    --model models.swat.simulation.SWAT_MODEL \
    --param-set A --seed 42

# All parameter sets for a model produce outputs under outputs/
# in a per-run timestamped subdirectory with state trajectories (NPZ),
# observation channels, plots (latent_states.png, observations.png, ...).
```

Command-line options:

- `--model <dotted.path.MODEL_OBJECT>` — model to simulate.
- `--param-set <name>` — named parameter set (A, B, C, ... — varies by model).
- `--seed <int>` — RNG seed (reproducible).
- `--out-dir <path>` — output root directory (default `outputs/`).
- `--verify` — run the model's `verify_physics_fn` instead of simulating.
- `--cross-validate` — scipy-vs-Diffrax deterministic cross-validation.
- `--scipy` — force scipy Euler-Maruyama even if JAX/Diffrax is available.

## Test-suite outputs

The SWAT model ships a self-contained test suite in [`version_1/models/swat/TESTING.md`](version_1/models/swat/TESTING.md). Running it produces one output directory per test (Sets A / B / D / C, cross-validate, physics-verify, reproducibility, Stuart-Landau equilibrium, Lyapunov decrease) under [`version_1/outputs/`](version_1/outputs/). Each test directory contains test-name-prefixed NPZ files, observation channels, and the three analysis plots.

## Repository layout

```
Python-Model-Development-Simulation/
├── README.md                              # you are here
├── how_to_add_a_new_model/                # guide for adding new models
└── version_1/
    ├── simulator/                         # generic CLI + scipy/Diffrax solvers
    ├── models/                            # one subdirectory per model
    │   ├── _template_model/               # copy-me skeleton for new models
    │   ├── bistable_controlled/
    │   ├── fitness_strain_amplitude/
    │   ├── fsa_real_obs/
    │   ├── sleep_wake/
    │   ├── sleep_wake_20p/
    │   └── swat/
    ├── model_documentation/               # audience-facing specs per model
    │   └── README.md                      # docs index
    └── outputs/                           # simulation output directories
```

## Requirements

- Python ≥ 3.10
- `numpy`, `scipy`, `matplotlib` (required)
- `jax`, `jaxlib` (optional — enables Diffrax GPU path and drift cross-validation)
- `diffrax` (optional — enables `--cross-validate` end-to-end; a manual drift-agreement snippet in each `TESTING.md` is used when Diffrax is unavailable)

## License

Private research repository.

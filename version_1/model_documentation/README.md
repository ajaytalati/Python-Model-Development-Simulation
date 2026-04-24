# Model documentation

Audience-facing specifications for the six SDE models in [`version_1/models/`](../models/). Each subdirectory contains a `<Model>_Documentation.md` with the mathematical spec, state/parameter tables, observation model, and parameter-set scenarios.

## Models

| Model | States | Parameters | Purpose | Doc |
|:---|:---:|:---:|:---|:---|
| `bistable_controlled` | 2 | 6 + 2 ICs | Controlled double-well SDE; pedagogical intervention testbed. | [bistable_controlled_Documentation.md](bistable_controlled/bistable_controlled_Documentation.md) |
| `fitness_strain_amplitude` | 3 | 13 (shared SDE) + obs | Athlete fitness/strain/amplitude dynamics — base model with direct state observation. | [FSA_Basic_Documentation.md](fitness_strain_amplitude/FSA_Basic_Documentation.md) |
| `fsa_real_obs` | 3 | 13 + 24 obs | FSA with six physiological observation channels (RHR, intensity, duration, stress, sleep, timing). | [fsa_real_obs_Documentation.md](fsa_real_obs/fsa_real_obs_Documentation.md) |
| `sleep_wake` | 6 | ~40 + 5 ICs | Sleep-wake-adenosine-circadian with four Garmin-style channels. | [sleep_wake_Documentation.md](sleep_wake/sleep_wake_Documentation.md) |
| `sleep_wake_20p` | 5 | 20 + 3 ICs | Minimal identifiability-proof-driven sleep-wake-adenosine SDE. | [sleep_wake_20p_Documentation.md](sleep_wake_20p/sleep_wake_20p_Documentation.md) |
| `swat` | 7 | 31 + 4 ICs | Sleep-Wake-Adenosine-Testosterone — SWAT. Extends `sleep_wake_20p` with a Stuart-Landau $T$-state and $V_c$ phase shift. | [SWAT_Basic_Documentation.md](swat/SWAT_Basic_Documentation.md) |

## Companion docs

- [`swat/SWAT_Clinical_Specification.md`](swat/SWAT_Clinical_Specification.md) — clinical interpretation, pathology modes, and intervention targets for SWAT.
- [`swat/SWAT_Identifiability_Extension.md`](swat/SWAT_Identifiability_Extension.md) — formal Fisher-rank identifiability analysis.

## How these docs relate to `TESTING.md`

Models under active development (currently `sleep_wake_20p` and `swat`) carry a `TESTING.md` alongside their code. That file contains the **live, code-coupled spec** — the mathematical definition used by the simulator, the numerical parameter sets, and the regression test suite with thresholds. When the implementation and the spec disagree, `TESTING.md` is the source of truth. These per-model docs are audience-facing summaries that link back to the TESTING.md for exact thresholds.

## Running a simulation

From `version_1/`:

```bash
python simulator/run_simulator.py \
    --model models.swat.simulation.SWAT_MODEL \
    --param-set A --seed 42
```

Outputs are written to `outputs/synthetic_<model>_<param-set>_<timestamp>/` by default.

See the repo-root [README.md](../../README.md) for setup and the full test-suite walkthrough.

# 03 — Testing and documentation

Two deliverables: a `TESTING.md` that lives with the code and a user-facing doc under `version_1/model_documentation/`. Both are optional in the strict technical sense (the simulator doesn't read them) but **non-optional for maintainability**. If someone else (or you, in six months) wants to understand your model, these are what they'll read.

---

## 1. Why a `TESTING.md`

A `TESTING.md` is a **live specification** that travels with the model code. Its three audiences:

1. **You, six months later.** Remember which parameter set was healthy, what physics checks you expected, and what "passing" means.
2. **A collaborator or agent** running regression tests after a refactor.
3. **A skeptical reviewer** who wants to verify the implementation matches the math.

When the implementation and a paper specification disagree, `TESTING.md` is the **source of truth** for the code. Update it when you fix bugs or re-calibrate thresholds.

Canonical examples: [`models/sleep_wake_20p/TESTING.md`](../version_1/models/sleep_wake_20p/TESTING.md) and [`models/swat/TESTING.md`](../version_1/models/swat/TESTING.md). Your `TESTING.md` should mirror their section structure.

---

## 2. `TESTING.md` section template

Copy this structure verbatim. The `_template_model/TESTING.md` in the template directory is this skeleton pre-filled with TODO markers.

```markdown
# Testing the <MODEL_NAME> Model

**Version:** <x.y>
**Date:** <YYYY-MM-DD>
**Model:** `models/<your_name>/`

**Self-contained.** A reader verifying the implementation against the
specification should need nothing else.

## 1. Purpose
## 2. Mathematical specification
  ### 2.1 State variables
  ### 2.2 The SDE system
  ### 2.3 Observation model
  ### 2.4 Deterministic components (optional)
## 3. Parameter definitions
## 4. Parameter sets for testing
  ### 4.1 Set A — <name / role>
  ### 4.2 Set B — <name / role>
## 5. Tests
  ### Test 0 — Import smoke
  ### Test 1 — Parameter set A
  ### Test 2 — Parameter set B
  ### Test 3 — Cross-validation (scipy vs Diffrax, optional)
  ### Test 4 — Physics verification
  ### Test 5 — Reproducibility
  ### Test 6 — Observation consistency
## 6. Troubleshooting
## 7. Exit criteria
## 8. Calibration results
```

---

## 3. Filling in each section

### §1 Purpose

Two or three sentences. What question does this model answer? Why is the observation model what it is? Example: *"Verify that the Python implementation in `models/swat/` matches the mathematical specification in §2, and that Parameter Set A (healthy basin) produces a stable testosterone equilibrium."*

### §2 Mathematical specification

The complete SDEs. Reproduce them verbatim — a reader should not need another document. Include:

- **§2.1 State variables table.** Columns: Symbol | Name | Domain | Timescale | Role. Example: $W$ | wakefulness | $[0, 1]$ | $\tau_W \approx 2$ h | stochastic.
- **§2.2 The SDE system.** One LaTeX block per state. Write out the drift and diffusion explicitly.
- **§2.3 Observation model.** One equation per channel with likelihood (Gaussian, Bernoulli, Poisson, lognormal, …).
- **§2.4 Deterministic components** (if any). Analytical expressions for deterministic states; exogenous schedules.

### §3 Parameter definitions

Prose, one line per parameter. Group by block (fast subsystem, slow subsystem, observation, noise). Tell the reader what each parameter means physically — "$\kappa$ is the sleep-depth-to-wakefulness inhibition strength; larger values make the flip-flop sharper."

### §4 Parameter sets

One subsection per set. Each subsection contains:

- **Table** with two columns: Parameter | Value. Include every entry of `PARAM_SET_X` (and `INIT_STATE_X` if relevant).
- **Simulation length and grid.** E.g. "7 days, $dt = 5$ minutes, 2016 steps".
- **Expected behaviour.** Qualitative description of what each state should do. What's healthy vs pathological.
- **Quantitative thresholds** the implementation must satisfy. These become the `verify_physics_fn` checks.

Example threshold prose: *"Set A: $T \in [0.45, 0.70]$ throughout the trajectory after day 1; mean $T \approx 0.57 \pm 0.10$; $E \approx 0.66$ from day 1 onward."*

### §5 Tests — the command-line regression suite

One subsection per test, with:

- The **exact shell command** to run it.
- The **expected terminal output** (qualitative or a key line).
- **Qualitative checks** (what plots should look like) and **quantitative checks** (table of thresholds with tolerances).

Recommended minimum tests for every model:

| Test | Command | Expected |
|:---|:---|:---|
| 0. Import smoke | `python -c "from models.X.simulation import X_MODEL; print(X_MODEL.name)"` | Name prints cleanly |
| 1. Set A | `python simulator/run_simulator.py --model models.X.simulation.X_MODEL --param-set A --seed 42` | NPZ + plots + physics checks |
| 2. Set B | same with `--param-set B` | Differs from A in documented ways |
| 3. Cross-validate | `--cross-validate` | Max trajectory diff $< 1\mathrm{e}{-4}$ (requires `drift_fn_jax`) |
| 4. Physics verify | `--verify` | All booleans `True` |
| 5. Reproducibility | Run test 1 twice; compare `synthetic_truth.npz` | Byte-identical trajectories |
| 6. Observation consistency | Manual check that channel values match the spec equations | HR residuals within noise; sleep labels follow $\sigma(\tilde Z - \tilde c)$ |

If you add JAX / Stuart-Landau / Lyapunov structure to your model, add tests for those too (see SWAT's Tests 7–9).

### §6 Troubleshooting

Symptom / cause / fix table. Populate as you encounter failure modes. Keep it concise — each row is one paragraph max. Agents and engineers use this to triage.

### §7 Exit criteria

A checklist of conditions under which the model is "ready for downstream work". Example:

- [ ] Test 0 passes.
- [ ] Set A matches qualitative and quantitative specifications.
- [ ] `verify_physics_fn` all booleans `True` for Set A.
- [ ] Cross-validate max diff < 1e-10 (if JAX provided).
- [ ] Reproducibility test passes.

### §8 Calibration results

Left as a stub until after the first test run. Then fill in:

- **Date, seed, solver, grid** of the calibration runs.
- **Per-set summary table** (e.g. `T` at end, mean $E$, physics-check results).
- **Test outcomes table** (PASS / FAIL with notes).
- **Any spec revisions** the calibration prompted (threshold looseness, parameter-value adjustments).

This is the paper trail that lets the next reader see how the model was validated.

---

## 4. The per-model user-facing doc

A separate, **audience-facing** document at `version_1/model_documentation/<your_name>/<Your_Name>_Documentation.md`. Structurally different from `TESTING.md`:

- `TESTING.md` is code-coupled, minute, and mutable. It's for developers and regression tests.
- The model-documentation page is narrative, concise, and stable. It's for collaborators reading about the model for the first time.

### 4.1 Template

Copy from [`SWAT_Basic_Documentation.md`](../version_1/model_documentation/swat/SWAT_Basic_Documentation.md). Section pattern:

```markdown
# The <Model> Model — Specification
## <Subtitle: what it models in one sentence>

**Version:** <x.y>
**Source:** link to simulation.py

## 1. What this model is for
## 2. States and parameters at a glance
  ### 2.1 The <N>-state dynamical system   (table)
  ### 2.2 The parameter list                (grouped tables)
## 3. The SDE system                        (LaTeX equations)
## 4. Observation model                     (per-channel equations)
## 5. Parameter sets for simulation testing (tables)
## 6. Pointers                              (links to TESTING.md, related models, CLI command)
```

### 4.2 How the two docs relate

Include a one-paragraph note in your doc like:

> *The canonical, actively-maintained specification and test suite lives at [TESTING.md](../../models/your_name/TESTING.md). This document is an audience-facing summary.*

When they drift out of sync, `TESTING.md` wins.

---

## 5. Discoverability

Two one-line edits make your model discoverable:

1. **Append a row** to the table in [`version_1/model_documentation/README.md`](../version_1/model_documentation/README.md). Columns: model name, states, parameters, purpose, link to your doc.
2. **Append a row** to the Models table in the repo-root [`README.md`](../README.md) with a one-line description and link to the doc.

Check that both links resolve locally before pushing.

---

**Next:** [04_reference.md](04_reference.md) for lookup tables; [worked_example.md](worked_example.md) for the end-to-end example.

# Testing the _template_ Model

**Version:** 0.1
**Date:** TODO
**Model:** `models/_template_/`

**TODO:** rename this file's title and references once you have copied the
template. Full section skeleton follows; fill each section with your model's
actual content. See `how_to_add_a_new_model/03_testing_and_docs.md` for guidance.

---

## 1. Purpose

TODO: why this document exists. Usually: "Verify that the Python
implementation in `models/<name>/` matches the mathematical spec in §2, and
that Parameter Set A (healthy) produces the expected qualitative behaviour."

---

## 2. Mathematical specification

### 2.1 State variables

TODO — table of state, meaning, range, timescale.

### 2.2 The SDE system

TODO — drift and diffusion equations.

### 2.3 Observation model

TODO — per-channel likelihoods.

### 2.4 Deterministic components

TODO — analytical-deterministic states (if any), exogenous schedules.

---

## 3. Parameter definitions

TODO — prose description of every estimated parameter, grouped by block.

---

## 4. Parameter sets for testing

### 4.1 Set A — TODO (healthy / default)

TODO — full parameter table.

### 4.2 Set B — TODO

TODO.

---

## 5. Tests

### Test 0 — Import smoke

```bash
python -c "from models._template_.simulation import TEMPLATE_MODEL; print(TEMPLATE_MODEL.name, 'v' + TEMPLATE_MODEL.version)"
```

**Expected:** `_template_ v0.1`

### Test 1 — Parameter set A

```bash
python simulator/run_simulator.py --model models._template_.simulation.TEMPLATE_MODEL --param-set A --seed 42
```

**Expected:** TODO — qualitative behaviour + quantitative thresholds (e.g. "x stays within [-5, 5]").

### Test 2 — Parameter set B

TODO.

### Test 3 — Cross-validation (scipy vs Diffrax)

```bash
python simulator/run_simulator.py --model models._template_.simulation.TEMPLATE_MODEL --cross-validate
```

**Expected:** max trajectory difference `< 1e-4`. Skip if you have not written `drift_jax`.

### Test 4 — Physics verification

```bash
python simulator/run_simulator.py --model models._template_.simulation.TEMPLATE_MODEL --verify
```

**Expected:** all booleans in `verify_physics_fn` output are `True` (or within the tolerance you set).

### Test 5 — Reproducibility

Run Test 1 twice with the same seed; trajectories must be byte-identical.

---

## 6. Troubleshooting

TODO — symptom / cause / fix table as new failure modes are discovered.

---

## 7. Exit criteria

The model is ready for downstream work once:

- [ ] Test 0 (import smoke) passes
- [ ] Test 1 (Set A) matches qualitative + quantitative expectations
- [ ] Test 2 (Set B) TODO
- [ ] Test 4 (physics) all checks `True`
- [ ] Test 5 (reproducibility) passes
- [ ] (Optional) Test 3 (cross-validate) max diff `< 1e-4`

---

## 8. Calibration results

TODO — fill in after the first complete test run. Include observed values
vs spec predictions, any thresholds that needed revision, and the dates/commit
hashes of the calibration runs.

---

*End of document.*

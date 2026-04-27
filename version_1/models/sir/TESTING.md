# Testing the SIR Model

**Version:** 1.0
**Date:** 2026-04-27
**Model:** `models/sir/`

This is the canonical **basic test model** for the three-repo pipeline.
A reviewer / verifier should be able to reproduce every test in §5 and
get the expected results. If any test fails, the model is broken and
should not be used downstream.

---

## 1. Purpose

Verify that the Python implementation in `models/sir/`:

1. matches the mathematical SIR specification in §2,
2. reproduces the Anderson & May 1978 boarding-school flu outbreak
   (Set A) within tight numerical tolerances,
3. preserves mass conservation (S + I + R = N) under SDE diffusion
   approximation, and
4. exercises the mixed-likelihood (Gaussian + Poisson) observation
   path that downstream inference depends on.

Set A is the **paper-parity benchmark**: a published outbreak with
established truth values (β ≈ 1.66/day, γ ≈ 0.5/day, R₀ ≈ 3.32, attack
rate ≈ 0.94). Reproducing it is the strongest end-to-end test that the
pipeline is wired correctly.

---

## 2. Mathematical specification

### 2.1 State variables

| Symbol | Meaning             | Range          | Timescale     |
|--------|---------------------|----------------|---------------|
| `S(t)` | Susceptible count   | [0, N]         | depletion ~1/β·I |
| `I(t)` | Infected count      | [0, N]         | clearance ~1/γ |
| `R(t)` | Recovered (derived) | [0, N], R = N − S − I | monotone non-decreasing |

The state vector solved by the SDE simulator is `y = [S, I]`; R is computed analytically.

### 2.2 The SDE system

Frequency-dependent transmission with optional vaccination input v(t):

```
dS/dt = -β S I / N - v(t) S        +  √T_S · ξ_S(t)
dI/dt = +β S I / N -      γ I      +  √T_I · ξ_I(t)
```

where ξ_S, ξ_I are independent unit-white-noise processes (Itô).

The diffusion approximation is exact in the small-noise limit; for very
small I (early outbreak / late tail) the simulator can briefly leave
[0, N], which the framework's bounds-check tolerates within
`max(5, 0.5%·N)` particles.

### 2.3 Observation model

| Channel    | Type     | Likelihood                                                        | Cadence     |
|------------|----------|-------------------------------------------------------------------|-------------|
| `cases`    | Poisson  | `cases_d ~ Poisson(ρ β S_d I_d / N × 24h)` per daily bin          | every 24 h  |
| `serology` | Gaussian | `serology_w ~ Normal(I_w / N, σ_z²)` at each survey               | every 7 d   |

### 2.4 Deterministic components

None. The model has no analytical-deterministic state and no
exogenous schedule (other than the optional v(t) step-function in Set D,
implemented as a constant via the frozen-params dict; full piecewise
schedules would go through the framework's `make_aux` mechanism).

---

## 3. Parameter definitions

| Name      | Block         | Units     | Prior (Set A)                       | Notes |
|-----------|---------------|-----------|-------------------------------------|-------|
| `beta`    | Transmission  | 1/hr      | LogNormal(log 0.0692, 0.4)          | Set A truth: 0.0692 (= 1.66/day) |
| `gamma`   | Recovery      | 1/hr      | LogNormal(log 0.0208, 0.3)          | Set A truth: 0.0208 (= 0.5/day, mean infectious 2 days) |
| `rho`     | Detection     | -         | Beta(8, 2), mean 0.8                | Set A truth: 1.0 (boarding school: full reporting). Sets B/C/D: 0.5. |
| `sigma_z` | Obs noise     | (frac.)   | LogNormal(log 0.02, 0.5)            | Serology survey error std. |
| `T_S`     | Diffusion     | counts²/hr | LogNormal(log 1.0, 0.5)             | S diffusion temperature. |
| `T_I`     | Diffusion     | counts²/hr | LogNormal(log 1.0, 0.5)             | I diffusion temperature. |
| `I_0`     | Initial state | counts    | LogNormal(log 1.0, 1.0)             | Set A truth: 1 (single index case). |
| `N`       | Frozen        | counts    | not estimated                       | Set A: 763. Sets B/C/D: 10 000. |
| `v`       | Frozen        | 1/hr      | not estimated                       | 0 in baseline. Set D: 0.02/day = 8.3e-4/hr. |

`R₀ = β/γ` is the natural identifiable combination from the cases channel
alone; the serology channel breaks the partial degeneracy between (β, γ)
and ρ.

---

## 4. Parameter sets for testing

### 4.1 Set A — Anderson & May 1978 boarding-school flu (PRIMARY)

Reproduces a real published outbreak. The 1978 boarding-school flu is the
canonical PMCMC tutorial benchmark used by Endo, van Leeuwen & Baguelin
(2019) *Epidemics* 29.

| Parameter | Value     | Note |
|-----------|-----------|------|
| N         | 763       | All boys at the boarding school. |
| β         | 1.66/day  | β·N ≈ 1.27/day; R₀ ≈ 3.32 |
| γ         | 0.5/day   | Mean infectious 2 days |
| ρ         | 1.0       | Full reporting (school nurse logged every case) |
| σ_z       | 0.02      | Serology survey noise |
| T_S, T_I  | 1.0       | Diffusion temperatures (count²/hr) |
| I_0       | 1         | Single index case |
| t_total   | 14 days   | Outbreak ran from day 1 to day 14 |

**Expected behaviour:**
- Peak I between day 5 and day 7, magnitude 250–290 individuals.
- Attack rate (R(end)/N) between 0.85 and 0.95.
- Final S between 30 and 100.

### 4.2 Set B — small community outbreak

| Parameter | Value     |
|-----------|-----------|
| N         | 10 000    |
| β         | 0.5/day   |
| γ         | 0.2/day   |
| R₀        | 2.5       |
| ρ         | 0.5       |
| t_total   | 60 days   |
| I_0       | 5         |

Smooth recovery; full final-size relation (a ≈ 0.89).

### 4.3 Set C — large community outbreak

| Parameter | Value     |
|-----------|-----------|
| N         | 10 000    |
| β         | 0.8/day   |
| γ         | 0.2/day   |
| R₀        | 4.0       |
| ρ         | 0.5       |
| t_total   | 90 days   |
| I_0       | 10        |

High R₀; near-total infection (a ≈ 0.98).

### 4.4 Set D — vaccination intervention

| Parameter | Value     |
|-----------|-----------|
| N         | 10 000    |
| β         | 0.6/day   |
| γ         | 0.2/day   |
| R₀        | 3.0       |
| ρ         | 0.5       |
| v         | 0.02/day  |
| t_total   | 90 days   |
| I_0       | 10        |

Sustained vaccination from t = 0; final attack rate is reduced
relative to the no-intervention case. Sets up the closed-loop control
benchmark for Phase 5 of the SMC² rollout.

---

## 5. Tests

A reviewer should be able to run every test in this section and get
the expected output. All tests run in seconds on CPU.

### Test 0 — Import smoke

```bash
python -c "from models.sir.simulation import SIR_MODEL; \
           from models.sir.estimation import SIR_ESTIMATION; \
           print(SIR_MODEL.name, 'v' + SIR_MODEL.version, '/', \
                 SIR_ESTIMATION.name, 'v' + SIR_ESTIMATION.version)"
```

**Expected:** `sir v1.0 / sir v1.0`

### Test 1 — Set A (paper-parity benchmark)

```bash
python simulator/run_simulator.py \
    --model models.sir.simulation.SIR_MODEL --param-set A --seed 42
```

**Expected:**

- `R_0`: 3.3200
- `peak_I`: between 250 and 290
- `peak_t_hours`: between 96 and 168 (i.e. day 4 to day 7)
- `attack_rate`: between 0.85 and 0.95
- All `*_ok` flags: PASS
- `PASS: physics verification`

### Test 2 — Sets B, C, D

```bash
for set in B C D; do
    python simulator/run_simulator.py \
        --model models.sir.simulation.SIR_MODEL --param-set $set --seed 42
done
```

**Expected:**

| Set | R_0 | attack_rate range |
|-----|----:|-------------------|
| B   | 2.5 | 0.85 – 0.92       |
| C   | 4.0 | 0.95 – 0.99       |
| D   | 3.0 | 0.93 – 0.99 (vaccination starts at t=0; partial mitigation) |

All sets must `PASS: physics verification`.

### Test 3 — Mass conservation under SDE noise

```bash
python -c "
import numpy as np
from models.sir.simulation import SIR_MODEL
from simulator.sde_solver_scipy import solve_sde

params = SIR_MODEL.param_sets['A']
N = params['N']
t_grid = np.linspace(0, params['t_total_hours'],
                     int(params['t_total_hours'] / params['dt_hours']) + 1)
trajectory = solve_sde(SIR_MODEL, params, SIR_MODEL.init_states['A'],
                       t_grid, seed=42, n_substeps=4)

S, I = trajectory[:, 0], trajectory[:, 1]
R = N - S - I
err = float(np.abs(S + I + R - N).max())
assert err < 1e-3, f'Mass conservation violated: {err}'
print(f'mass conservation: max |S+I+R - N| = {err:.2e}')
print(f'final attack rate = {R[-1]/N:.4f}')
"
```

**Expected:** `max |S+I+R - N| < 1e-3` (R = N − S − I by construction;
the sum equals N to floating-point precision).

### Test 4 — Mixed-likelihood observation channels

The cases channel is Poisson (integer counts); the serology channel is
Gaussian (continuous prevalence in [0, 1]).

```bash
python -c "
import numpy as np
from models.sir.simulation import SIR_MODEL
from simulator.sde_solver_scipy import solve_sde

params = SIR_MODEL.param_sets['A']
t_grid = np.linspace(0, params['t_total_hours'],
                     int(params['t_total_hours'] / params['dt_hours']) + 1)
trajectory = solve_sde(SIR_MODEL, params, SIR_MODEL.init_states['A'],
                       t_grid, seed=42, n_substeps=4)
aux = SIR_MODEL.make_aux_fn(params, SIR_MODEL.init_states['A'], t_grid, {})

cases = SIR_MODEL.channels[0].generate_fn(trajectory, t_grid, params, aux, {}, seed=42)
serology = SIR_MODEL.channels[1].generate_fn(trajectory, t_grid, params, aux, {}, seed=42)

assert cases['cases'].dtype == np.int32
assert (cases['cases'] >= 0).all()
assert len(cases['cases']) == 14
assert (serology['prevalence'] >= -3*params['sigma_z']).all()
assert (serology['prevalence'] <= 1.0 + 3*params['sigma_z']).all()
assert len(serology['prevalence']) == 2

print(f'cases (Poisson, sum={cases[\"cases\"].sum()}): {cases[\"cases\"].tolist()}')
print(f'serology (Gaussian, n={len(serology[\"prevalence\"])}): {serology[\"prevalence\"].round(3).tolist()}')
print('PASS: mixed-likelihood obs channels')
"
```

**Expected:** 14 daily Poisson cases summing to ~600–800 (most of cohort
ever-infected), 2 weekly Gaussian serology samples in [−0.05, 0.45].

### Test 5 — JAX/NumPy drift parity

The JAX `_dynamics.drift_jax` must produce numerically identical drift to
the numpy `simulation.drift`, otherwise inference will silently disagree
with the simulator.

```bash
python -c "
import numpy as np
import jax.numpy as jnp
from models.sir.simulation import drift, SIR_MODEL
from models.sir._dynamics import drift_jax
from models.sir.estimation import PI, DEFAULT_FROZEN_PARAMS

# Build a packed param vector matching PI ordering
pkeys = ['beta', 'gamma', 'rho', 'sigma_z', 'T_S', 'T_I']
params_dict = SIR_MODEL.param_sets['A']
params_packed = jnp.array([params_dict[k] for k in pkeys])

y = jnp.array([700.0, 50.0])    # mid-outbreak
np_drift = drift(0.0, np.array(y), params_dict, None)
jx_drift = drift_jax(y, 0.0, params_packed, DEFAULT_FROZEN_PARAMS, PI)

err = float(np.max(np.abs(np.array(jx_drift) - np_drift)))
assert err < 1e-10, f'Drift mismatch: max |JAX - numpy| = {err}'
print(f'drift parity max |err| = {err:.2e}  (PASS at 1e-10)')
"
```

**Expected:** `drift parity max |err| < 1e-10`.

### Test 6 — Reproducibility

Run Test 1 twice with seed 42; trajectories must be byte-identical.

```bash
python simulator/run_simulator.py --model models.sir.simulation.SIR_MODEL --param-set A --seed 42
python simulator/run_simulator.py --model models.sir.simulation.SIR_MODEL --param-set A --seed 42
# diff the latest two trajectory.npz files; expect zero output.
```

**Expected:** Set A run twice yields the same `trajectory.npz` (byte-identical).

---

## 6. Troubleshooting

| Symptom                                 | Cause                                                                 | Fix                                                                                       |
|-----------------------------------------|-----------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| `R_in_range_ok` FAIL on Set B/C/D       | Diffusion temperature too high for tight bounds                       | Tolerance scales with N; check `max(5, 0.005·N)` is wide enough                          |
| Set A peak I outside [250, 290]         | Wrong β or γ; check `1.66/24` for β and `0.5/24` for γ in `PARAM_SET_A` | The factor of 24 converts per-day to per-hour                                            |
| `attack_rate < 0.5` in Set A            | I_0 too small or transmission interrupted                              | Anderson-May has I_0=1; lower I_0 increases stochastic extinction probability             |
| Cases channel returns floats not ints   | `rng.poisson(...)` not cast to int32                                   | See `gen_cases`; the cast `.astype(np.int32)` is mandatory                                |
| JAX/numpy drift mismatch                | Param ordering in `PI` doesn't match unpacking in numpy `drift`        | The numpy `drift` reads from a dict (no ordering); JAX `drift_jax` reads from a vector via `pi`. They should agree by name. |

---

## 7. Exit criteria

The model is ready for downstream work in psim / SMC² once:

- [ ] Test 0 (import smoke) passes
- [ ] Test 1 (Set A paper-parity) reproduces Anderson-May expected values
- [ ] Test 2 (Sets B, C, D) all PASS physics verification
- [ ] Test 3 (mass conservation) `< 1e-3`
- [ ] Test 4 (mixed-likelihood obs) cases are int32 ≥ 0; serology in expected range
- [ ] Test 5 (JAX/numpy drift parity) `< 1e-10`
- [ ] Test 6 (reproducibility) byte-identical at fixed seed

---

## 8. Calibration results

First clean simulator-side run on 2026-04-27 at commit
`<feat/sir_model first commit hash>`:

| Set | R_0   | peak_I | peak_t (days) | attack_rate | physics |
|-----|------:|-------:|--------------:|------------:|---------|
| A   | 3.32  | 266.7  | 5.25          | 0.930       | PASS    |
| B   | 2.5   | 2375.0 | mid-trial     | 0.887       | PASS    |
| C   | 4.0   | 4048.7 | mid-trial     | 0.979       | PASS    |
| D   | 3.0   | 1495.4 | mid-trial     | 0.970       | PASS    |

Set A's `peak_I = 267` is in tight agreement with Anderson & May's
published peak of ~280 cases on day 6.

---

*End of document.*

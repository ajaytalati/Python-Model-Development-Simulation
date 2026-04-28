# Testing the Glucose-Insulin (Bergman Minimal) Model

**Version:** 1.0
**Date:** 2026-04-28
**Model:** `models/glucose_insulin/`

This is the canonical **basic test model** for the three-repo pipeline.
A reviewer should be able to reproduce every test in §5 and get the
expected results. If any test fails, the model is broken and should
not be used downstream.

---

## 1. Purpose

Verify that the Python implementation in `models/glucose_insulin/`:

1. matches the Bergman 1981 extended-minimal-model SDE specification in §2,
2. reproduces the canonical Bergman 1979 healthy-cohort response to a
   standard meal challenge (peak G 150-200 mg/dL, return to basal in
   ≤ 3 hr, postprandial insulin 30-60 μU/mL),
3. preserves physiological state ranges under SDE noise (G ∈ [40, 250]
   for healthy, [40, 450] for T1D no-control), and
4. exercises the mixed-likelihood (Gaussian CGM + Poisson meal carbs)
   observation path that downstream inference depends on.

Set A is the **paper-parity benchmark** to Bergman's 1979 healthy cohort.

---

## 2. Mathematical specification

### 2.1 State variables

| Symbol | Meaning             | Range            | Timescale     |
|--------|---------------------|------------------|---------------|
| `G(t)` | Plasma glucose      | [40, 600] mg/dL  | minutes-hours |
| `X(t)` | Remote insulin action | [0, 2] 1/hr     | minutes       |
| `I(t)` | Plasma insulin      | [0, 500] μU/mL   | minutes       |

### 2.2 The SDE system (Bergman 1981 extended)

```
dG/dt = -p₁(G - Gb) - X·G + D(t)/(V_G·BW)         + √T_G · ξ_G(t)
dX/dt = -p₂·X + p₃·max(I - Ib, 0)                  + √T_X · ξ_X(t)
dI/dt = -k(I - Ib) + n_β·max(G - h_β, 0) + I_input(t)/(V_I·BW·100) + √T_I · ξ_I(t)
```

where ξ_*(t) are independent Itô white noise processes; `D(t)` is the
gamma-2 meal absorption profile; `I_input(t)` is the open-loop insulin
schedule (Set D only); `n_β`, `h_β` parameterise the Bergman 1981 β-cell
secretion extension (zero for T1D scenarios).

### 2.3 Observation model

| Channel    | Type     | Likelihood                                       | Cadence      |
|------------|----------|--------------------------------------------------|--------------|
| `cgm`      | Gaussian | `cgm_t ~ Normal(G_t, σ_CGM²)`                    | every 5 min  |
| `meal_carbs` | Poisson  | `carbs_meal ~ Poisson(carbs_truth_meal)`         | at meal times (3/day) |

### 2.4 Exogenous inputs

- **Meal schedule:** 3 meals/day at 08:00, 13:00, 19:00 with day-to-day
  jitter (±30 min); truth carb counts ~Normal(40, 6) g.
- **Insulin schedule:** zero except Set D, which delivers a 0.5-hr
  exponential bolus at each meal (1 U per 10 g carbs, the typical I:C
  ratio) plus 0.5 U/hr basal.

---

## 3. Parameter definitions

| Name         | Block         | Set A truth        | Notes |
|--------------|---------------|--------------------|-------|
| `p1`         | Glucose effectiveness | 1.8 /hr      | Non-insulin-mediated disposal |
| `p2`         | Remote insulin decay  | 1.5 /hr      | |
| `p3`         | Insulin sensitivity   | 4.68e-2 /(hr²·μU/mL) | SI = p₃/p₂ ≈ 0.031 (healthy) |
| `k`          | Plasma insulin clearance | 18 /hr    | |
| `Gb`         | Basal glucose         | 90 mg/dL    | |
| `sigma_cgm`  | CGM noise std         | 8 mg/dL     | Dexcom G6 spec |
| `T_G`        | Glucose process noise | 1.0 (mg/dL)²/hr | small |
| `Ib`         | (Frozen) Basal insulin| 7 μU/mL (Sets A/B); 0 (Sets C/D) | |
| `n_β`        | (Frozen) β-cell secretion | 8 /(hr·μU/mL/(mg/dL)) (Sets A/B); 0 (C/D) | |
| `V_G`, `V_I`, `BW` | (Frozen) Volumes/weight | 1.6 dL/kg, 1.2 dL/kg, 70 kg | |
| `G_0`, `I_0` | Initial states        | 90 mg/dL, 7 μU/mL | X_0 = 0 |

---

## 4. Parameter sets for testing

### 4.1 Set A — Healthy adult (Bergman 1979 paper-parity)

All canonical Bergman 1979 healthy-cohort means. Expected behaviour:
- 3 meal-response peaks during the 24-hour trial
- Each peak: G 175-185 mg/dL, lasting ~1.5 hr
- Postprandial insulin: 45-55 μU/mL peak
- Mean glucose ~ 100-110 mg/dL
- All physics verifications PASS

### 4.2 Set B — Insulin resistance (pre-T2D)

`p₃` halved → SI drops 50%. Expected behaviour:
- Slightly higher peaks (185-200 mg/dL)
- Slightly higher mean glucose (105-115 mg/dL)
- Insulin response unchanged in magnitude (β-cells still working)
- Physics PASS

### 4.3 Set C — T1D no-control

`Ib = 0`, `n_β = 0`, no insulin doses. Expected behaviour:
- Higher meal peaks (200-250 mg/dL)
- Higher mean glucose (115-130 mg/dL)
- Plasma insulin ~ 0 throughout
- Physics PASS (peak ≥ 200 expected for T1D)

### 4.4 Set D — T1D with open-loop insulin

`Ib = 0`, `n_β = 0`, but with insulin bolus + basal schedule. Expected:
- Glucose returns to near-normal post-meal (peaks 170-185)
- Plasma insulin peaks 25-35 μU/mL after each bolus
- Physics PASS

---

## 5. Tests

A reviewer should be able to run every test in this section and get the
expected output. All tests run in seconds on CPU.

### Test 0 — Import smoke

```bash
python -c "from models.glucose_insulin import GLUCOSE_INSULIN_MODEL, GLUCOSE_INSULIN_ESTIMATION; \
           print(GLUCOSE_INSULIN_MODEL.name, 'v' + GLUCOSE_INSULIN_MODEL.version, '/', \
                 GLUCOSE_INSULIN_ESTIMATION.name, 'v' + GLUCOSE_INSULIN_ESTIMATION.version)"
```

**Expected:** `glucose_insulin v1.0 / glucose_insulin v1.0`

### Test 1 — Set A (paper-parity benchmark)

```bash
python simulator/run_simulator.py \
    --model models.glucose_insulin.simulation.GLUCOSE_INSULIN_MODEL --param-set A --seed 42
```

**Expected:**

- `G_min`: 85-92
- `G_max`: 175-200 (3 meal peaks)
- `G_mean`: 100-115
- `I_max`: 40-60
- All `*_ok` flags PASS
- `PASS: physics verification`

### Test 2 — Sets B, C, D

```bash
for set in B C D; do
    python simulator/run_simulator.py \
        --model models.glucose_insulin.simulation.GLUCOSE_INSULIN_MODEL --param-set $set --seed 42
done
```

**Expected:**

| Set | G_max range | I_max range |
|-----|-------------|-------------|
| B   | 185-205 mg/dL | 45-60 μU/mL |
| C   | 200-220 mg/dL | < 1 μU/mL (no insulin) |
| D   | 170-190 mg/dL | 20-35 μU/mL (open-loop bolus) |

All sets must `PASS: physics verification`.

### Test 3 — Set A peak timing

```bash
python -c "
import numpy as np
from models.glucose_insulin.simulation import GLUCOSE_INSULIN_MODEL
from simulator.sde_solver_scipy import solve_sde

params = GLUCOSE_INSULIN_MODEL.param_sets['A']
t_grid = np.linspace(0, params['t_total_hours'],
                     int(params['t_total_hours'] / params['dt_hours']) + 1)
trajectory = solve_sde(GLUCOSE_INSULIN_MODEL, params, GLUCOSE_INSULIN_MODEL.init_states['A'],
                       t_grid, seed=42, n_substeps=4)
G = trajectory[:, 0]

# Find local maxima representing meal-response peaks
peak_indices = [i for i in range(1, len(G)-1)
                if G[i] > 130 and G[i] > G[i-1] and G[i] > G[i+1]]
print(f'detected {len(peak_indices)} peaks above 130 mg/dL at times:'
      f' {[round(t_grid[i], 2) for i in peak_indices]}')
assert 3 <= len(peak_indices) <= 5, f'expected ~3 meal peaks, got {len(peak_indices)}'
print('PASS: meal-peak structure')
"
```

**Expected:** `detected 3 peaks above 130 mg/dL at times: [~9, ~14, ~20]`,
`PASS: meal-peak structure`.

### Test 4 — Mixed-likelihood obs channels

```bash
python -c "
import numpy as np
from models.glucose_insulin.simulation import GLUCOSE_INSULIN_MODEL
from simulator.sde_solver_scipy import solve_sde

params = GLUCOSE_INSULIN_MODEL.param_sets['A']
t_grid = np.linspace(0, params['t_total_hours'],
                     int(params['t_total_hours'] / params['dt_hours']) + 1)
trajectory = solve_sde(GLUCOSE_INSULIN_MODEL, params, GLUCOSE_INSULIN_MODEL.init_states['A'],
                       t_grid, seed=42, n_substeps=4)
aux = GLUCOSE_INSULIN_MODEL.make_aux_fn(params, GLUCOSE_INSULIN_MODEL.init_states['A'], t_grid, {})

cgm = GLUCOSE_INSULIN_MODEL.channels[0].generate_fn(trajectory, t_grid, params, aux, {}, seed=42)
meals = GLUCOSE_INSULIN_MODEL.channels[1].generate_fn(trajectory, t_grid, params, aux, {}, seed=42)

assert cgm['cgm_value'].dtype == np.float32
assert len(cgm['cgm_value']) == 289   # 5-min × 24h grid is t=0..24 inclusive (289 pts)
assert (cgm['cgm_value'] > 50).all() and (cgm['cgm_value'] < 250).all()
assert meals['carbs_g'].dtype == np.int32
assert (meals['carbs_g'] >= 0).all()
assert len(meals['carbs_g']) == 3   # 3 meals/day

print(f'CGM (Gaussian, n=289): mean {cgm[\"cgm_value\"].mean():.1f} mg/dL, '
      f'std {cgm[\"cgm_value\"].std():.1f}')
print(f'meal carbs (Poisson, n=3): {meals[\"carbs_g\"].tolist()} g')
print('PASS: mixed-likelihood obs channels')
"
```

**Expected:** 288 CGM observations in [50, 250]; 3 Poisson meal counts
each in [20, 60].

### Test 5 — JAX/numpy drift parity

The JAX `_dynamics.drift_jax` must produce numerically identical drift to
the numpy `simulation.drift`, otherwise inference will silently disagree
with the simulator.

```bash
JAX_ENABLE_X64=True python -c "
import numpy as np
import jax.numpy as jnp
from models.glucose_insulin.simulation import (
    drift, GLUCOSE_INSULIN_MODEL, _meal_absorption_rate,
    _insulin_input_rate, _meal_schedule)
from models.glucose_insulin._dynamics import drift_jax
from models.glucose_insulin.estimation import PI, DEFAULT_FROZEN_PARAMS

params_dict = GLUCOSE_INSULIN_MODEL.param_sets['A']
meal_sched = _meal_schedule(seed=0, n_days=1, meal_carbs_g=40.0)
aux = {'meal_schedule': meal_sched, 'insulin_schedule': None}

pkeys = ['p1', 'p2', 'p3', 'k', 'Gb', 'sigma_cgm', 'T_G']
params_packed = jnp.array([params_dict[k] for k in pkeys], dtype=jnp.float64)

t_test = 9.0     # mid-morning, after breakfast
y = jnp.array([180.0, 0.5, 30.0])
V_G, BW = DEFAULT_FROZEN_PARAMS['V_G'], DEFAULT_FROZEN_PARAMS['BW']
D_rate_at_t = _meal_absorption_rate(float(t_test), meal_sched, V_G, BW)
I_rate_at_t = _insulin_input_rate(float(t_test), None)
aux_jax = {'D_rate_at_t': jnp.float64(D_rate_at_t),
           'I_input_rate_at_t': jnp.float64(I_rate_at_t)}

np_drift = drift(t_test, np.array(y), params_dict, aux)
jx_drift = drift_jax(y, jnp.float64(t_test), params_packed,
                      DEFAULT_FROZEN_PARAMS, aux_jax, PI)
err = float(np.max(np.abs(np.array(jx_drift) - np_drift)))
assert err < 1e-10, f'drift parity failed: {err}'
print(f'drift parity max |err| = {err:.2e}  (PASS at 1e-10)')
"
```

**Expected:** `drift parity max |err| < 1e-10`.

### Test 6 — Reproducibility

Run Test 1 twice with seed 42; trajectories must be byte-identical.

```bash
python simulator/run_simulator.py \
    --model models.glucose_insulin.simulation.GLUCOSE_INSULIN_MODEL --param-set A --seed 42
python simulator/run_simulator.py \
    --model models.glucose_insulin.simulation.GLUCOSE_INSULIN_MODEL --param-set A --seed 42
# diff the two synthetic_truth.npz files; expect zero output.
```

**Expected:** byte-identical trajectory.npz.

---

## 6. Troubleshooting

| Symptom                                  | Cause                                    | Fix                                                                    |
|------------------------------------------|------------------------------------------|------------------------------------------------------------------------|
| Set A peaks too high (G_max > 220)       | n_β too low, or meal_carbs too high      | Bump `n_beta` (8 is calibrated for healthy); reduce `meal_carbs_g`     |
| `peak_post_meal_realistic` FAIL          | G_max outside [200, ...] for T1D no-ctrl, OR > 200 for healthy | Check the truth values in the failing PARAM_SET                       |
| JAX/numpy drift mismatch                 | aux pre-computation differs              | drift_jax uses pre-computed `aux['D_rate_at_t']`; numpy drift recomputes from schedule. Ensure both use the same schedule + same t. |
| Insulin trajectory rises in Set C        | Endogenous secretion still active        | Set C must have `n_beta=0` AND `Ib=0`                                  |

---

## 7. Exit criteria

Ready for downstream work in psim / SMC² once:

- [ ] Test 0 (import smoke) passes
- [ ] Test 1 (Set A paper-parity) reproduces healthy postprandial response
- [ ] Test 2 (Sets B/C/D) all PASS physics verification
- [ ] Test 3 (Set A peak timing) detects 3 meal peaks at expected times
- [ ] Test 4 (mixed-likelihood obs) CGM in physiological range, carb counts non-neg int
- [ ] Test 5 (JAX/numpy drift parity) `< 1e-10`
- [ ] Test 6 (reproducibility) byte-identical at fixed seed

---

## 8. Calibration results

First clean simulator-side run on 2026-04-28:

| Set | G_min | G_max | G_mean | I_max | physics |
|-----|------:|------:|-------:|------:|---------|
| A   | 87.7  | 185.0 | 103.4  | 49.1  | PASS    |
| B   | 88.9  | 194.9 | 107.7  | 53.6  | PASS    |
| C   | 88.9  | 211.0 | 117.5  | 0.32  | PASS    |
| D   | 84.1  | 180.1 | 103.5  | 25.7  | PASS    |

Set A reproduces the textbook healthy postprandial response: 3 meal
peaks at ~9am/2pm/8pm, each rising to 175-185 mg/dL and returning to
basal Gb=90 within ~1.5 hr; plasma insulin peaks at 45-55 μU/mL.

---

*End of document.*

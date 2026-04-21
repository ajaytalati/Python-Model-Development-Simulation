# Bug Report: Excess Daytime Sleep Labels — c_tilde Too Small, beta_Z Too Small

**Date:** 2026-04-18  
**Status:** Fixed  
**File affected:** `models/sleep_wake_20p/simulation.py`, `models/sleep_wake_20p/estimation.py`  
**Parameters changed:**
- `c_tilde` from `1.5` → `3.0`
- `beta_Z` from `1.5` → `2.5`
- `Zt_0` from `1.8` → `3.5`

---

## Symptom

After the `gamma_3` fix (see `BUG_REPORT_gamma3_sleep_depth.md`), the sleep-depth
state `Zt` correctly flipped between sleep and wake attractors.  However the
**binary sleep observation channel still showed too many sleep events during
daytime** (high-HR) periods.  The observation panel plot showed sleep labels
(`sleep_label = 1`) scattered through every day, even when the HR mean was
clearly at its daytime peak (~75 bpm).

**Quantitative signature (pre-fix):**

| Quantity | Expected | Observed |
|---|---|---|
| `sleep_label = 1` during daytime | < 5% of steps | ~18–22% of steps |
| Appearance | Clean night blocks | Scattered throughout |

This is a **distinct bug from the `gamma_3` issue**.  The previous fix ensured
Zt reached the correct equilibrium during sleep.  This bug concerns the
observation function threshold, not the latent dynamics.

---

## Root Cause Analysis

### Observation model

```
sleep_label ~ Bernoulli( σ(Zt − c_tilde) )
```

For the sleep label to be reliable, two conditions must hold simultaneously:

1. During **wake** (W ≈ 1): `Zt` should be far *below* `c_tilde` so that
   `σ(Zt − c_tilde) ≪ 1`.
2. During **sleep** (W ≈ 0): `Zt` should be far *above* `c_tilde` so that
   `σ(Zt − c_tilde) ≈ 1`.

### The lower-bound floor problem

The state `Zt` is clipped at its physical lower bound of **0** (see
`StateSpec("Zt", 0.0, A_SCALE)`).  During wakefulness, `gamma_3` suppresses
Zt strongly toward 0, but it cannot go below 0.  Therefore, even with perfect
dynamics, the minimum achievable daytime sleep probability is:

```
P(sleep_label = 1 | W = 1, Zt = 0) = σ(0 − c_tilde) = σ(0 − 1.5) = 0.182
```

**This 18% floor is structural — it does not depend on gamma_3, T_W, T_Z,
or any other dynamics parameter.**  No amount of latent-state tuning can push
it lower as long as `c_tilde = 1.5`.

### SDE noise raises the effective floor further

In the SDE, `Zt` does not sit exactly at 0 during wake — the diffusion term
`sqrt(2 T_Z) dB_Z` keeps it positive via reflection at the lower boundary.
For a reflected OU process, the stationary mean above 0 is:

```
E[Zt | wake]  ≈  sqrt(T_Z · tau_Z) · sqrt(2/π)
              =  sqrt(0.05 · 2) · sqrt(2/π)
              ≈  0.316 · 0.798  ≈  0.25
```

This raises the effective floor to:

```
P(sleep_label = 1 | W = 1) ≈ σ(0.25 − 1.5) = σ(−1.25) ≈ 0.22
```

### Quantified impact

With 22% false sleep rate during wake, a typical 7-day simulation
(5-min grid → 2016 steps, ~60% wake time) produces:

```
False sleep labels ≈ 0.22 × 1210 wake steps ≈ 266 false positives
```

These appear as scattered sleep labels throughout every high-HR period,
which is exactly the observed pathology.

### Why this was not caught alongside the gamma_3 bug

The `gamma_3` bug *masked* this issue: with `gamma_3 = 60`, Zt was suppressed
**below** `c_tilde` even during sleep, so the sleep labels were already
near-random (39%).  Fixing `gamma_3` to 8 restored correct Zt dynamics and
made the sleep labels predominantly correct during sleep — but simultaneously
revealed that the 18% daytime floor had always been present.  The two bugs
were **sequentially masked**: fixing the first exposed the second.

---

## Fix

### Part 1 — Raise `c_tilde` from 1.5 to 3.0

This directly reduces the structural floor:

```
P(sleep | W=1, Zt=0) = σ(0 − 3.0) = σ(−3.0) ≈ 0.05   (was 18%)
```

A 5% daytime false positive rate is acceptable for a noisy wearable-derived
sleep label.

### Part 2 — Raise `beta_Z` from 1.5 to 2.5

With `c_tilde` raised to 3.0, the old `Zt_sleep ≈ 3.28` (from the `gamma_3`
fix) now sits too close to the new threshold, giving insufficient separation.
Raising `beta_Z` (the adenosine→Zt coupling) increases `Zt` during sleep:

```
u_Z during sleep (W=0)  =  −gamma_3 · W − Vn + beta_Z · a
                         =  −8·0 − 0.3 + beta_Z · a
                         =  −0.3 + beta_Z · a
```

At sleep **onset** (adenosine at peak, a ≈ 1):

| `beta_Z` | `u_Z` | `Zt_sleep` | `prob_sleep` |
|---|---|---|---|
| 1.5 | 1.2 | 4.61 | σ(4.61−3.0) = σ(1.61) ≈ 95% |
| 2.5 | 2.2 | 5.40 | σ(5.40−3.0) = σ(2.40) ≈ 99% |

At sleep **end** (adenosine cleared, a → 0):

| `beta_Z` | `u_Z` | `Zt_sleep` | `prob_sleep` |
|---|---|---|---|
| 1.5 | −0.3 | 2.55 | σ(2.55−3.0) = σ(−0.45) ≈ 39% |
| 2.5 | −0.3 | 2.55 | σ(2.55−3.0) = σ(−0.45) ≈ 39% |

Note: at the end of sleep, `a ≈ 0` so `beta_Z` does not matter — the 39%
probability at sleep end is a **correct physiological feature**, not a bug.
It models the fact that sleep pressure falls as adenosine clears toward
morning, which is what triggers waking.

### Part 3 — Raise `Zt_0` from 1.8 to 3.5

The old initial condition `Zt_0 = 1.8` is now below the new `c_tilde = 3.0`,
meaning the simulation starts with an ambiguous sleep signal.  Raising
`Zt_0 = 3.5` places the initial condition above the threshold, consistent with
the model starting in a moderately sleep-like state (W_0 = 0.5).

---

## Verification

**Expected observation statistics (post-fix):**

| Quantity | Expected |
|---|---|
| `P(sleep_label=1)` during daytime (W≈1) | ~5% |
| `P(sleep_label=1)` at sleep onset (a≈1, W=0) | ~99% |
| `P(sleep_label=1)` during stable sleep (a≈0.5) | ~81% |
| `P(sleep_label=1)` near end of sleep (a≈0) | ~39% (correct — adenosine cleared) |
| Sleep labels | Clean night blocks, minimal daytime scatter |

**Bistability check:**  
`kappa × gamma_3 = 6.67 × 8 = 53` — unchanged.  `beta_Z` and `c_tilde` do
not affect the flip-flop mechanism (that depends only on `kappa × gamma_3`).

---

## Code changes

**File:** `models/sleep_wake_20p/simulation.py` — `PARAM_SET_A`

```python
# Before
'c_tilde':    1.5,
'beta_Z':     1.5,
# (INIT_STATE_A)
'Zt_0':       1.8,

# After
'c_tilde':    3.0,   # raised from 1.5 — σ(0−1.5)=18% floor → σ(0−3.0)=5%
'beta_Z':     2.5,   # raised from 1.5 — Zt_sleep at onset: 4.6 → 5.4 >> c_tilde=3.0
# (INIT_STATE_A)
'Zt_0':       3.5,   # raised from 1.8 to sit above new c_tilde=3.0 at t=0
```

**File:** `models/sleep_wake_20p/estimation.py` — `PARAM_PRIOR_CONFIG` and
`INIT_STATE_PRIOR_CONFIG`

```python
# Before
('c_tilde',  ('normal',    (1.5, 0.5))),
('beta_Z',   ('lognormal', (math.log(1.5), 0.4))),
('Zt_0',     ('normal',    (1.8, 0.8))),

# After
('c_tilde',  ('normal',    (3.0, 0.5))),          # 95% CI: [2.0, 4.0]
('beta_Z',   ('lognormal', (math.log(2.5), 0.4))),# 95% CI: [1.1, 5.5]
('Zt_0',     ('normal',    (3.5, 0.8))),           # matches new sim initial condition
```

---

## Impact on downstream files

| File | Impact |
|---|---|
| `simulation.py` — `PARAM_SET_A` | Changed (this fix) |
| `simulation.py` — `PARAM_SET_B` | Inherits via `dict(PARAM_SET_A)` → also fixed |
| `estimation.py` — priors | Changed (this fix) |
| `estimation.py` — `INIT_STATE_PRIOR_CONFIG` | Changed (`Zt_0` prior) |
| Proof-of-principle scripts | No changes needed; read `PARAM_SET_A` at runtime |

---

## Lessons learned

1. **The observation threshold must be calibrated relative to the state's
   reachable range in the SDE, not its theoretical ODE range.**  `Zt` has a
   hard lower bound at 0; any threshold set within `σ`-units of 0 will
   produce a non-negligible baseline false positive rate regardless of dynamics.

2. **Sequential masking of bugs is common in nonlinear SDEs.**  The `gamma_3`
   bug suppressed Zt below `c_tilde` during sleep, masking the floor problem.
   Once the first bug was fixed, the second became visible.  Always re-verify
   the full observation channel after fixing latent-state bugs.

3. **`beta_Z` and `c_tilde` are coupled parameters.**  They should be chosen
   jointly: `c_tilde` sets the discrimination boundary; `beta_Z` sets the
   height of the sleep attractor above that boundary.  Tuning one without the
   other will always leave the model in a suboptimal regime.

4. **The adenosine clearance dynamic has a correct physiological interpretation
   at the boundary.**  The 39% sleep probability when `a → 0` (full clearance)
   is not a model failure — it represents the reduction in sleep pressure that
   triggers waking after a full night's sleep.  It would be wrong to "fix"
   this by raising `beta_Z` further, as that would prevent natural waking.

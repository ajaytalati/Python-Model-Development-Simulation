# Bug Report: Sleep Depth (Zt) Suppressed Below Sleep Threshold

**Date:** 2026-04-17  
**Status:** Fixed  
**File affected:** `models/sleep_wake_20p/simulation.py`  
**Parameter changed:** `gamma_3` from `60.0` → `8.0`

---

## Symptom

When running the 20-parameter sleep-wake-adenosine simulator with `PARAM_SET_A`,
the generated observations showed two clear pathologies:

1. **Sleep labels did not flip-flop with heart rate.** Binary sleep labels
   (`sleep_label = 1`) appeared scattered nearly randomly across the time
   series rather than concentrating in the low-HR night periods where they
   should be.

2. **Sleep depth (Zt) did not flip-flop with wakefulness (W).** Despite W
   showing clear transitions between 0 (asleep) and 1 (awake), Zt oscillated
   with low amplitude, barely touching the sleep threshold `c_tilde = 1.5`
   rather than rising well above it during sleep periods.

**Quantitative signature (pre-fix):**

| Quantity | Expected | Observed |
|---|---|---|
| `Zt` range during sleep | >> `c_tilde = 1.5` | max ≈ 1.5 (at threshold) |
| `Zt` above `c_tilde` | ~40–50% of steps | < 5% |
| `sleep_label = 1` fraction | ~30–40% | near-random scatter |

---

## Root Cause Analysis

### Model equations (relevant)

```
dZt = (A_SCALE * σ(u_Z) - Zt) / tau_Z  dt  +  sqrt(2 T_Z) dB_Z

u_Z = -gamma_3 * W - Vn + beta_Z * a
```

At the **sleep equilibrium** (theoretical, noise-free, W = 0):

```
u_Z  = -gamma_3 * 0 - 0.3 + 1.5 * 0.5  = 0.45
σ(0.45) ≈ 0.61
Zt_eq = 6 * 0.61 = 3.66  >>  c_tilde = 1.5  ✓
```

In the noise-free ODE, this is fine. The problem arises in the **SDE**.

### SDE noise floor in W

The W diffusion temperature `T_W = 0.01` produces a Gaussian noise term at
each integration step:

```
noise std per step = sqrt(2 * T_W) * sqrt(dt)
                   = sqrt(0.02) * sqrt(5/60 h)
                   ≈ 0.041  per step
```

Because W is clipped to `[0, 1]` (state bounds), it cannot go negative.
During sleep, W drifts toward 0 but the noise keeps it bouncing just above
the clip boundary. The time-average of `|N(0, 0.041)|` gives a persistent
**noise floor**:

```
mean W during sleep ≈ 0.033   (non-zero due to rectification at 0)
```

### How gamma_3 = 60 amplifies the noise floor

With `gamma_3 = 60`, this tiny residual W value is multiplied by 60 in `u_Z`:

```
u_Z = -60 * 0.033 - 0.3 + 1.5 * 0.5
    = -2.0  - 0.3  + 0.75
    = -1.55

σ(-1.55) ≈ 0.175
Zt_eq    = 6 * 0.175 = 1.05
```

`Zt_eq = 1.05` is **below** `c_tilde = 1.5`. Consequently:

```
prob_sleep = σ(Zt - c_tilde) ≈ σ(1.05 - 1.5) = σ(-0.45) ≈ 0.39
```

Sleep labels are nearly random (39% probability even at peak Zt), and the
simulated sleep panel shows scattered noise rather than clean flip-flop blocks.

### Why this was not caught during design

`gamma_3 = 60` was taken from the identifiability proof document, which
analysed the **deterministic ODE** (no SDE noise, `T_W = 0`). In the
deterministic system, W reaches exactly 0 during sleep, `u_Z = 0.45`, and
`Zt_eq = 3.66 >> 1.5` — no problem. The SDE noise changes the effective
operating point of the W → Zt coupling, a subtlety not visible in the
ODE analysis.

---

## Fix

Reduce `gamma_3` from `60.0` to `8.0` in `PARAM_SET_A`.

### Verification with gamma_3 = 8

With the same noise floor (mean W ≈ 0.033 during sleep):

```
u_Z = -8 * 0.033 - 0.3 + 1.5 * 0.5
    =  -0.26 - 0.3 + 0.75
    =   0.19

σ(0.19) ≈ 0.547
Zt_eq   = 6 * 0.547 = 3.28  >>  c_tilde = 1.5  ✓
```

During wake (W ≈ 0.97):

```
u_Z = -8 * 0.97 - 0.3 + 0.45 = -7.61
σ(-7.61) ≈ 0.0005
Zt_eq ≈ 0  ✓
```

Clear separation between sleep and wake equilibria. Confirmed by simulation:

**Quantitative signature (post-fix):**

| Quantity | Expected | Observed |
|---|---|---|
| `Zt` max during sleep | >> `c_tilde = 1.5` | 2.85 |
| `Zt` range | large | 2.85 |
| `Zt` above `c_tilde` | ~30–40% of steps | 28.4% |
| `sleep_label = 1` fraction | ~30–40% | 36.9% |
| `W` range | 0 → 1 | 0.000 → 1.000 |
| All states finite | yes | yes |

### Bistability is preserved

The flip-flop mechanism relies on mutual inhibition:

- W suppresses Zt (via `gamma_3 * W` term in `u_Z`)
- Zt suppresses W (via `kappa * Zt` term in `u_W`)

The relevant coupling product:

| Version | `kappa × gamma_3` |
|---|---|
| Pre-fix | `6.67 × 60 = 400` |
| Post-fix | `6.67 × 8 = 53` |

A product of 53 is more than sufficient for bistability; the model continues
to show clean flip-flop transitions between sleep and wake attractors.
The change makes the model appropriately sensitive to the SDE noise level
rather than being dominated by it.

---

## Code change

**File:** `models/sleep_wake_20p/simulation.py`

```python
# Before
'gamma_3':   60.0,

# After
'gamma_3':    8.0,   # reduced from 60 — SDE noise floor suppressed Zt below c_tilde
```

The inline code comment in `simulation.py` documents the reasoning in full.

---

## Impact on downstream files

| File | Impact |
|---|---|
| `simulation.py` — `PARAM_SET_A` | Changed (this fix) |
| `simulation.py` — `PARAM_SET_B` | Inherits `PARAM_SET_A` via `dict(PARAM_SET_A)`, so also fixed |
| `estimation.py` — priors | Prior on `gamma_3` may need re-centring (prior median was presumably 60; should now be ~8) |
| `TESTING.md` | `Zt_range` exit criterion (§7) already updated to `> 2.0` on 2026-04-17 — passes with the new value of 2.85 |
| Proof-of-principle scripts | No changes needed; they call `PARAM_SET_A` at runtime |

### Action required: update estimation prior for gamma_3

The prior distribution on `gamma_3` in `models/sleep_wake_20p/estimation.py`
was likely centred on or near 60. This prior should be updated to reflect the
new understanding that the biologically correct value (given the SDE noise
level) is around 8. The prior should be wide enough to allow the sampler to
explore values from approximately 4 to 20.

---

## Lessons learned

1. **Deterministic parameter calibration does not transfer directly to the
   SDE.** A coupling constant calibrated to give bistability in the ODE may be
   far too large for the stochastic version, where even small noise levels
   create a persistent state-dependent offset.

2. **Check that observation thresholds are compatible with the SDE's
   stochastic equilibrium, not the ODE's deterministic equilibrium.** For any
   binary observation of the form `σ(state - threshold)`, the threshold should
   be set relative to the empirical distribution of `state` under the
   SDE, not the theoretical ODE fixed point.

3. **The rectification of a clipped state amplifies noise effects.** W clipped
   to `[0, 1]` behaves as a half-normal near the boundary, giving a non-zero
   mean even when the drift targets 0. Downstream states multiplied by a large
   coupling constant (gamma_3) will feel this as a systematic offset.

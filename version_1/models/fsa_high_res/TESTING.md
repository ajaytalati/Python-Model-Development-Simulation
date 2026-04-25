# Testing the fsa_high_res Model

**Version:** 0.1
**Date:** 2026-04-25
**Model:** `models/fsa_high_res/`

**Self-contained.** A reader verifying the implementation against the
specification should need nothing else.

## 1. Purpose

Verify that the Python implementation in `models/fsa_high_res/` matches
the mathematical specification in §2 and that Parameter Set A (the
recovery scenario) produces a stable trajectory whose B / F / A states
remain in physical range and whose four observation channels are
generated cleanly through the channel DAG.

This is a **15-min-bin variant** of `fsa_real_obs` with a SWAT-style
4-channel mixed-likelihood observation model and a deterministic
circadian forcing C(t) entering each obs link. The latent SDE is
identical to `fsa_real_obs`; the novelty is the sub-daily resolution
and the diurnal observation structure.

## 2. Mathematical specification

### 2.1 State variables

| Symbol | Name | Domain | Timescale | Role |
|--------|------|--------|-----------|------|
| $B$ | fitness (Banister-style) | $[0, 1]$ | $\tau_B \approx 14$ d | Jacobi-noise |
| $F$ | fatigue | $[0, \infty)$ | $\tau_F \approx 7$ d | CIR-noise |
| $A$ | amplitude (Stuart-Landau) | $[0, \infty)$ | minutes-hours | Landau, regularised |

Time is in days. The bin width is $\Delta t = 1/96$ day = 15 min.

### 2.2 The SDE system

$$
\begin{aligned}
dB &= \frac{1 + \alpha_A A}{\tau_B}\,(T_B(t) - B)\,dt \;+\; \sigma_B \sqrt{B(1-B)}\,dW_B \\
dF &= \left[\Phi(t) - \frac{1 + \lambda_B B + \lambda_A A}{\tau_F}\,F\right]\,dt \;+\; \sigma_F \sqrt{F}\,dW_F \\
dA &= \left[\mu(B,F)\,A - \eta\,A^3\right]\,dt \;+\; \sigma_A \sqrt{A + \varepsilon_A}\,dW_A
\end{aligned}
$$

with $\mu(B,F) = \mu_0 + \mu_B B - \mu_F F - \mu_{FF} F^2$.

### 2.3 Observation model — 4 channels

Deterministic circadian forcing $C(t) = \cos(2\pi t + \phi)$ enters
each obs link. $\phi = 0$ is frozen (healthy morning chronotype).

$$
\begin{aligned}
\text{HR} &\sim \mathcal{N}\!\left(\text{HR}_{\text{base}} - \kappa_B B + \alpha^{HR}_A A + \beta^{HR}_C C(t),\;\sigma^2_{\text{HR}}\right) &\text{(sleep-gated)} \\
\text{sleep} &\sim \text{Bernoulli}\!\left(\sigma\!\left(k_C C(t) + k_A A - \tilde c\right)\right) &\text{(always observed)} \\
\text{stress} &\sim \mathcal{N}\!\left(S_{\text{base}} + k_F F - k^S_A A + \beta^S_C C(t),\;\sigma^2_S\right) &\text{(wake-gated)} \\
\log(\text{steps}+1) &\sim \mathcal{N}\!\left(\mu_{\text{step}} + \beta^{\text{st}}_B B - \beta^{\text{st}}_F F + \beta^{\text{st}}_A A + \beta^{\text{st}}_C C(t),\;\sigma^2_{\text{st}}\right) &\text{(wake-gated)}
\end{aligned}
$$

Sleep gating: HR is observed iff `sleep_label == 1`. Wake gating:
stress and steps are observed iff `sleep_label == 0`.

### 2.4 Deterministic components

- **Circadian C(t)**: emitted as an exogenous channel on the global
  time grid (see §6 — this is *not* recomputed window-locally; doing
  so introduces the C-phase bug).
- **Phi(t)**: morning-loaded burst pattern, Gamma(k=2) shape
  $t\,e^{-t/\tau}$ over wake hours, zero overnight. See
  `generate_phi_sub_daily` in `simulation.py`.
- **T_B(t)**: piecewise-constant per-day target fitness. Driver
  pre-sizes the per-bin array.

## 3. Parameter definitions

29 estimable parameters (10 dynamical + 5 HR + 3 sleep + 5 stress +
6 steps).

**Dynamics (10).**
- $\tau_B, \tau_F$: relaxation timescales of the fast subsystems.
- $\alpha_A$: A-amplification of B-relaxation.
- $\lambda_B, \lambda_A$: B-/A-amplification of F-relaxation.
- $\mu_0, \mu_B, \mu_F, \mu_{FF}$: bifurcation parameters of the
  Landau term. $\mu_0 > 0$ in this regime (Stuart-Landau fixed point
  $A^* = \sqrt{\mu/\eta}$).
- $\eta$: cubic damping.

**HR (5):** $\text{HR}_{\text{base}}$, $\kappa_B$ (vagal tone),
$\alpha^{HR}_A$, $\beta^{HR}_C$, $\sigma_{\text{HR}}$.

**Sleep (3):** $k_C$, $k_A$, $\tilde c$.

**Stress (5):** $S_{\text{base}}$, $k_F$, $k^S_A$, $\beta^S_C$,
$\sigma_S$.

**Steps (6):** $\mu_{\text{step}}$, $\beta^{\text{st}}_B$,
$\beta^{\text{st}}_F$, $\beta^{\text{st}}_A$, $\beta^{\text{st}}_C$,
$\sigma_{\text{st}}$.

**Frozen:** $\sigma_B = 0.01$, $\sigma_F = 0.005$, $\sigma_A = 0.02$,
$\varepsilon_A = \varepsilon_B = 10^{-4}$, $\phi = 0$.

## 4. Parameter sets for testing

### 4.1 Set A — recovery (the only set in v0.1)

`DEFAULT_PARAMS` in `simulation.py`. Tuned so that mu(B,F) is positive
across the recovery trajectory (A sits at a stable Stuart-Landau fixed
point rather than being activated by a bifurcation crossing).

Initial state: `B_0=0.05, F_0=0.10, A_0=0.55`.

A_0 is raised from the daily-FSA convention (0.01) to start near
$A^* = \sqrt{\mu/\eta} \approx 0.69$ at the recovery-trajectory midpoint,
giving some convergence to watch but staying above the quasi-absorbing
boundary.

## 5. Tests

### Test 0 — Import smoke

```python
from models.fsa_high_res import HIGH_RES_FSA_MODEL, HIGH_RES_FSA_ESTIMATION
assert HIGH_RES_FSA_MODEL.name == "fsa_high_res"
assert HIGH_RES_FSA_ESTIMATION.n_dim == 29
```

### Test 1 — CLI smoke (Set A)

```bash
cd version_1
python simulator/run_simulator.py \
  --model models.fsa_high_res.simulation.HIGH_RES_FSA_MODEL \
  --param-set A --seed 42
```

Expected: `PASS: physics verification`, plot at
`outputs/synthetic_fsa_high_res_A_<ts>/fsa_high_res_sim.png`,
wall time ≈ 2 s.

### Test 2 — Physics verification

`verify_physics_fn` returns:
- `all_finite: True` (the only gated boolean)
- `mu_crosses_zero: "no"` (informational; mu_0 > 0 keeps mu positive)
- B/F/A min/max within state bounds.

State bounds: B ∈ [0, 1], F ∈ [0, 10], A ∈ [0, 5].

### Test 3 — Reproducibility

Same seed → bit-identical NPZ trajectory. Verified via:

```bash
md5sum outputs/synthetic_fsa_high_res_A_<ts1>/sde_synthetic_data.npz
md5sum outputs/synthetic_fsa_high_res_A_<ts2>/sde_synthetic_data.npz
# should match for same seed
```

### Test 4 — Sim/est consistency (the §1.4 discipline)

This model is the reference implementation for the
[Python-Model-Scenario-Simulation](https://github.com/ajaytalati/Python-Model-Scenario-Simulation)
sim-est consistency discipline. Three checks must pass before any
SMC² estimation run:

- **Drift parity**: `model_sim.drift_fn` matches the deterministic
  part of `model_est.propagate_fn` at the truth parameters.
- **Obs-prediction parity**: per-Gaussian channel, the simulator's
  noiseless prediction matches the estimator's at every $(state, t, k)$.
- **Round-trip**: re-integrating `propagate_fn` with zero noise from
  the truth initial state recovers the simulator's trajectory.

These are exercised by
`tests/test_consistency_fsa_high_res.py` and
`tests/test_round_trip_fsa_high_res.py` in the psim repo.

## 6. Troubleshooting

### The C-phase bug class

`gen_C_channel` emits C(t) on the **global time grid**. Estimators
(and any rolling-window driver) MUST consume this from the artifact
or aux rather than recomputing C from window-local time inside
`align_obs_fn`. Doing the latter restarts C at $\cos(0)=+1$ at every
window boundary, producing a phase mismatch that biases all
$\beta^{HR}_C, \beta^S_C, \beta^{\text{st}}_C$ posterior means toward
zero with narrow credible intervals locked off-truth.

The fix in this codebase: `align_obs_fn` reads `obs_data['C']`
(global, sliced into the window by `extract_window`) rather than
recomputing.

Full case study:
[POSTMORTEM_three_bugs](https://github.com/ajaytalati/smc2-blackjax-rolling/blob/main/outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md).

### Phi profile shape

The morning-loaded shape is Gamma(k=2): $t\,e^{-t/\tau}$, which peaks
at $t = \tau$ post-wake. Default `tau_hours = 3.0` → peak at 10 am
if waking at 7 am. To shift the peak (e.g. shift-worker model),
adjust `peak_hours_post_wake` and `tau_hours` in
`generate_phi_sub_daily`.

## 7. Exit criteria

A v0.2 release would extend Set B (overtraining), Set C (sedentary)
following `fsa_real_obs`. As of v0.1 the only required exit criterion
is **Test 1 (CLI smoke) passes** — covering import-clean,
physics-finite, and plot-clean in 2 s.

## 8. Calibration results

The reference SMC² estimation run (27 rolling windows, 14-day C0
recovery scenario, `--seed 42`, N_SMC=256) yields **96.8% mean
coverage / 27 of 27 PASS** at the 70% threshold. See
`outputs/fsa_high_res_rolling/C_phase_fix_result.md` in the SMC²
repo.

This calibration is the regression target: any breaking change to
`simulation.py` or `estimation.py` must reproduce it within
stochastic noise.

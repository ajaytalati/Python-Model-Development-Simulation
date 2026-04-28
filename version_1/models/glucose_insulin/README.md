# glucose_insulin — Bergman Minimal Model (3-state SDE)

The canonical **basic test model** for the three-repo pipeline. Replaces
the earlier SIR test model with a more directly-physiological,
better-bridge-tractable system.

## Why this is the basic test model

- **Most-studied physiological filtering AND control problem.** Bergman
  1979 (model), Bergman 1981 (β-cell extension), Hovorka 2004 (in-silico),
  Magni & Cobelli 2009 (MPC), Tandem Control-IQ / Medtronic 780G / OpenAPS
  (current clinical closed-loop pumps).
- **CGMs are wearables** — the dynamical core that lives in the rolling-
  window inference + MPC running in production T1D pumps right now.
- **Mixed-likelihood naturally**: CGM (Gaussian, every 5 min) + meal carb
  counts (Poisson, 3/day) — exercises the same code path SWAT uses.
- **No rolling-window phase mismatch**: every 6-hour window contains a
  full meal-response cycle (rise + return), so the SF Path B-fixed bridge
  doesn't hit the SIR-style multi-window cascade collapse.
- **Natural control input** for the next-stage closed-loop demo
  (Phase 5 follow-up).

## Summary

- **Dynamics:** Bergman 1981 extended-minimal-model SDE (3 states):
  `G` (plasma glucose, mg/dL), `X` (remote insulin action, 1/hr),
  `I` (plasma insulin, μU/mL).
- **Mixed-likelihood observations:**
  - **Gaussian** CGM: `cgm_t ~ Normal(G_t, σ_CGM²)` every 5 min
  - **Poisson** meal carb counts: `carbs_meal ~ Poisson(λ)` at meal times
- **Estimable parameters (7):** p₁, p₂, p₃, k, Gb, σ_CGM, T_G.
- **Estimable initial state (2):** G_0, I_0  (X_0 = 0).
- **Frozen:** Ib, V_G, V_I, BW, T_X, T_I, n_β (β-cell secretion rate),
  h_β (secretion threshold).
- **Known exogenous inputs:** meal schedule (timing + truth carbs),
  insulin schedule (Set D only).

`SI = p₃ / p₂` — the *clinically* identifiable insulin sensitivity
combination, the canonical Bergman 1979 inference target.

## Math

```
dG/dt = -p₁(G - Gb) - X·G + D(t) / (V_G·BW)         [glucose, mg/dL]
dX/dt = -p₂·X + p₃·max(I - Ib, 0)                   [remote insulin action]
dI/dt = -k(I - Ib) + n_β·max(G - h_β, 0) + I_input(t) / (V_I·BW·100)
```

`D(t)`: meal absorption (gamma-2 gastric-emptying profile, τ = 0.5 hr).
`I_input(t)`: insulin appearance from open-loop bolus + basal schedule
(Set D only).

## Files

- `simulation.py` — SDE drift + diffusion + 4 PARAM_SETs + obs generators (CGM Gaussian + meal-carb Poisson) + verify_physics.
- `_dynamics.py` — pure-JAX drift, diffusion, IMEX step, channel log-probs.
  Shared between simulator and estimator; drift parity verified to ≤ 1e-10.
- `estimation.py` — `PARAM_PRIOR_CONFIG`, `INIT_STATE_PRIOR_CONFIG`,
  `make_glucose_insulin_estimation()` factory + canonical
  `GLUCOSE_INSULIN_ESTIMATION` instance, scenario-specific
  `PARAM_PRIOR_OVERRIDES_B/C/D`. Pitt-Shephard guided proposal via the
  Gaussian CGM channel (Kalman update on G at every 5-min bin) — mirrors
  SWAT's HR-tilted W proposal.
- `sim_plots.py` — diagnostic plots: G/X/I latent + CGM scatter + meal carb bars.
- `__init__.py` — exports `GLUCOSE_INSULIN_MODEL`, `GLUCOSE_INSULIN_ESTIMATION`.
- `TESTING.md` — full §1-8 testing protocol with reviewer-checkable tests.

## Parameter sets

| Set | Scenario | Distinguishing | Days | Notes |
|---|---|---|---:|---|
| A | Healthy adult — Bergman 1979 paper-parity | All canonical (p₃=4.68e-2, n_β=8) | 1 | Healthy, all canonical priors centred on truth |
| B | Insulin resistance (pre-T2D) | p₃ halved → SI drops 50% | 1 | Same physiology, reduced tissue response |
| C | T1D no-control (DKA risk) | Ib=0, n_β=0, no exogenous insulin | 1 | β-cells destroyed, no replacement |
| D | T1D with open-loop insulin schedule | Ib=0, n_β=0, fixed bolus + basal | 1 | Carb-counted insulin doses; closed-loop is next-stage |

## Smoke test

From `version_1/`:

```bash
python simulator/run_simulator.py \
    --model models.glucose_insulin.simulation.GLUCOSE_INSULIN_MODEL \
    --param-set A --seed 42
```

Should produce a 1-day trajectory with three meal peaks at ~9am/2pm/8pm,
peaks 175-185 mg/dL, return to basal Gb=90 within ~1.5 hr each, plasma
insulin peaks 45-55 μU/mL. Set A passes physics verification; the diagnostic
plot (`latent_states.png`) is the textbook healthy postprandial response.

## References

- Bergman, R. N., Ider, Y. Z., Bowden, C. R. & Cobelli, C. (1979).
  Quantitative estimation of insulin sensitivity. *American Journal of
  Physiology* 236, E667-E677. — original minimal model + FSIGT calibration.
- Bergman, R. N. (1981). β-cell secretion extension to the minimal model.
- Hovorka, R. et al. (2004). Nonlinear model predictive control of glucose
  in type 1 diabetes. *Physiological Measurement* 25, 905-920.
- Dalla Man, C., Rizza, R. A. & Cobelli, C. (2007). Meal simulation model
  of the glucose-insulin system. *IEEE Transactions on Biomedical
  Engineering* 54, 1740-1749. — the UVA/Padova T1DMS basis.
- Magni, L. & Cobelli, C. (2009). Model-predictive control of T1D using
  the artificial pancreas. *Diabetes Technology & Therapeutics* 11.

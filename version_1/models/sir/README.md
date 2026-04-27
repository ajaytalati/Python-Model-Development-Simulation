# sir — Stochastic SIR (Susceptible–Infected–Recovered)

The canonical **basic test model** for the three-repo pipeline
(public dev → psim → SMC²). Designed to exercise the full mixed-
likelihood inference path on a small, well-known reference system in
a fraction of the wall-clock time of the production models (FSA,
SWAT).

## Summary

- **Dynamics:** frequency-dependent stochastic SIR with diffusion approximation.
  2 latent states (S, I); R = N − S − I eliminated, conserved.
- **Mixed-likelihood observations:**
  - **Poisson** daily case counts: `cases_t ~ Poisson(ρ β S_t I_t / N × 24h)`
  - **Gaussian** weekly serology survey: `serology_t ~ Normal(I_t / N, σ_z²)`
- **Estimable parameters (6):** β, γ, ρ, σ_z, T_S, T_I.
- **Estimable initial state (1):** I_0  (S_0 = N − I_0 derived).
- **Frozen parameters:** N (population), v (vaccination rate).
- **Control input (Set D):** vaccination rate v(t).

`R₀ = β/γ` is the basic reproduction number — natural identifiable
combination that gives the model its bifurcation structure (transcritical
at R₀ = 1).

## Why SIR is the basic test model

1. **Well-known in filtering AND control literature.** Endo/Funk/Held PMCMC
   tutorials, Behncke 2000 + Lenhart-Workman optimal-control textbook, recent
   COVID-19 work — saturated benchmark targets in both areas.
2. **Mixed Gaussian + Poisson observations** exercising the same code path
   that SWAT uses, on a much smaller problem.
3. **Small (2 states, 6 + 1 = 7 estimable scalars)** — full SMC² rolling
   run fits in 10–20 minutes of GPU time.
4. **Natural control input** for the next-stage closed-loop work.
5. **Domain-aligned with the project** (health / population biology).

## Files

- `simulation.py` — SDE + 4 parameter sets (A: boarding-school, B/C/D
  community + vax). Set A reproduces the Anderson & May 1978 flu data.
- `_dynamics.py` — pure-JAX drift, diffusion, IMEX step, per-channel
  observation log-probabilities. Shared between simulation and estimation.
- `estimation.py` — `PARAM_PRIOR_CONFIG`, `INIT_STATE_PRIOR_CONFIG`,
  `make_sir_estimation()` factory + canonical `SIR_ESTIMATION` instance.
- `sim_plots.py` — diagnostic plots: S/I/R latent trajectories +
  Poisson cases bars + Gaussian serology error-bars.
- `TESTING.md` — full testing protocol; reviewers should run every
  test in §5 to verify the model end-to-end.
- `__init__.py` — exports `SIR_MODEL`, `SIR_ESTIMATION`.

## Parameter sets

| Set | Scenario                        | N        | β/day | γ/day | R₀   | Days | Notes |
|-----|---------------------------------|---------:|------:|------:|-----:|-----:|-------|
| A   | Anderson & May 1978 (paper-parity) | 763   | 1.66 | 0.5  | 3.32 | 14   | Boarding-school flu. ρ = 1.0 (full reporting). |
| B   | Small community outbreak        | 10 000   | 0.5  | 0.2  | 2.5  | 60   | ρ = 0.5. |
| C   | Large community outbreak        | 10 000   | 0.8  | 0.2  | 4.0  | 90   | ρ = 0.5. |
| D   | Vaccination intervention        | 10 000   | 0.6  | 0.2  | 3.0  | 90   | v = 0.02/day, ρ = 0.5. Sets up control benchmark. |

Sets B/C/D have priors centered off-truth in the canonical
`PARAM_PRIOR_CONFIG` (which is Set-A-centered for paper-parity); the
SMC² driver overrides them on a per-scenario basis via
`SirRollingConfig`.

## Smoke test

From `version_1/`:

```bash
python simulator/run_simulator.py \
    --model models.sir.simulation.SIR_MODEL --param-set A --seed 42
```

Writes `outputs/synthetic_sir_A_<timestamp>/` with NPZ + two PNG
diagnostic plots. Set A should reproduce the canonical Anderson-May
curve: peak I ≈ 270 around day 5–6, attack rate ≈ 93%.

## References

- Anderson, R. M. & May, R. M. (1991). *Infectious Diseases of Humans*. OUP.
- Endo, A., van Leeuwen, E. & Baguelin, M. (2019). Introduction to
  particle Markov-chain Monte Carlo for disease dynamics modellers.
  *Epidemics* 29, 100363.
- Britton, T. (2010). Stochastic epidemic models: a survey.
  *Mathematical Biosciences* 225, 24–35.
- Lenhart, S. & Workman, J. (2007). *Optimal Control Applied to
  Biological Models*. Chapman & Hall/CRC.

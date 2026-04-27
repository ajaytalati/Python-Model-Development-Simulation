"""sir — Stochastic SIR (Susceptible-Infected-Recovered).

The canonical **basic test model** for the three-repo pipeline
(public dev → psim → SMC²). Compartmental epidemic model with
mixed-likelihood observations: Poisson daily case counts +
Gaussian weekly serology survey. Optional vaccination control input.

Math:
    dS = (-β S I / N - v(t) S) dt + √T_S dW_S
    dI = ( β S I / N - γ I)    dt + √T_I dW_I
    R  = N - S - I       (eliminated; conserved)

Set A is the canonical Anderson & May 1978 boarding-school flu outbreak
(N=763, R₀ ≈ 3.3, 14 days) — the paper-parity benchmark used as the
canonical PMCMC tutorial example in Endo et al 2019 (Epidemics 29).

See:
  - simulation.py — SDE drift + diffusion + per-channel obs generators
  - estimation.py — priors + JAX callbacks + EstimationModel
  - _dynamics.py  — pure-JAX dynamics shared between sim and est
  - sim_plots.py  — matplotlib diagnostic plots
  - TESTING.md    — full testing protocol with reviewer-checkable tests
  - README.md     — model overview + quick-start
"""

from .simulation import SIR_MODEL  # noqa: F401
from .estimation import SIR_ESTIMATION  # noqa: F401

"""sir — Stochastic SIR (Susceptible-Infected-Recovered).

Compartmental epidemic model. Frequency-dependent transmission with
diffusion approximation. Mixed Gaussian (serology survey) + Poisson
(daily case counts) observations. Optional vaccination control input.

Set A is the canonical Anderson & May 1978 boarding-school flu outbreak
(N=763, R_0 ≈ 3.3) — the paper-parity benchmark used in Endo et al 2019
PMCMC tutorial.

See `models/sir/simulation.py` for math + parameter sets.
"""

from .simulation import SIR_MODEL  # noqa: F401

# Uncomment once estimation.py is fleshed out:
# from .estimation import SIR_ESTIMATION  # noqa: F401

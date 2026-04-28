"""glucose_insulin — Bergman minimal model (3-state SDE).

The canonical **basic test model** for the three-repo pipeline. Replaces
the earlier SIR test model. Glucose-insulin dynamics is the most-studied
physiological filtering AND control problem; this is the dynamical core
of the Tandem Control-IQ / Medtronic 780G / OpenAPS closed-loop pumps
and the FDA-approved UVA/Padova T1DMS simulator (in extended form).

Math: 3-state SDE in [G (glucose), X (remote insulin action), I (plasma
insulin)]; mixed-likelihood obs (CGM Gaussian every 5 min + meal carb
counts Poisson at meal times); meal absorption + insulin schedule as
known exogenous inputs.

Set A: Bergman 1979 healthy-cohort means (paper-parity benchmark).
Sets B/C/D: insulin resistance, T1D no-control, T1D open-loop insulin.

See:
  - simulation.py — SDE drift, diffusion, obs generators, 4 PARAM_SETs
  - estimation.py — priors + JAX callbacks + EstimationModel (deferred)
  - _dynamics.py  — pure-JAX dynamics shared between sim and est (deferred)
  - sim_plots.py  — matplotlib diagnostic plots
  - TESTING.md    — full reviewer-checkable testing protocol
  - README.md     — model overview + quick-start
"""

from .simulation import GLUCOSE_INSULIN_MODEL  # noqa: F401
from .estimation import GLUCOSE_INSULIN_ESTIMATION  # noqa: F401

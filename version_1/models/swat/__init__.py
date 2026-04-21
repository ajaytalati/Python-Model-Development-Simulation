"""models/swat/__init__.py — SWAT (Sleep-Wake-Adenosine-Testosterone) package.

Date:    20 April 2026
Version: 1.0

A 24-parameter SDE model extending the 17-parameter sleep-wake-adenosine
model by adding a Stuart-Landau testosterone pulsatility amplitude T with
entrainment quality E(t) as the bifurcation parameter.

See:
  - Spec_24_Parameter_Sleep_Wake_Adenosine_Testosterone_Model.md
  - Identifiability_and_Lyapunov_Proof_24_Parameter_Model.md

Public objects:
  - SWAT_MODEL       : simulation-side SDEModel (for simulator framework)
  - SWAT_ESTIMATION  : estimation-side EstimationModel (for particle-filter /
                       EKF / HMC inference)
"""

from models.swat.simulation import SWAT_MODEL
from models.swat.estimation import SWAT_ESTIMATION

__all__ = ['SWAT_MODEL', 'SWAT_ESTIMATION']

"""20-parameter sleep-wake-adenosine SDE model with behavioural potentials.

Derived from the 17-parameter identifiability proof plus 3 process-noise
temperatures (T_W, T_Z, T_a) that make the latent dynamics an SDE rather
than an ODE.  See:
  * Identifiability_Proof_17_Parameter_Sleep_Wake_Adenosine_Model.md
  * Likelihood_Geometry_and_Filter_Selection_17_Parameter_Model.md

Date:    17 April 2026
Version: 1.0
"""

from models.sleep_wake_20p.estimation import SLEEP_WAKE_20P_ESTIMATION
from models.sleep_wake_20p.simulation import SLEEP_WAKE_20P_MODEL

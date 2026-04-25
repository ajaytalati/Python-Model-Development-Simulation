"""High-resolution (15-min) FSA variant — sub-daily SWAT-style 4-channel mixed-likelihood obs.

Same latent SDE as ``fsa_real_obs`` (3-state B/F/A), but observations
are produced at 15-min bins instead of daily, with a deterministic
circadian forcing C(t) = cos(2 pi t + phi) entering each obs link.

Observation channels (4):
    HR     ~ N(HR_base - kappa_B B + alpha_A_HR A + beta_C_HR C(t), sigma_HR^2)   [Gaussian, sleep-gated]
    sleep  ~ Bernoulli(sigmoid(k_C C(t) + k_A A - c_tilde))                       [Bernoulli, always observed]
    stress ~ N(S_base + k_F F - k_A_S A + beta_C_S C(t), sigma_S^2)               [Gaussian, wake-gated]
    steps  ~ log-Normal(mu_step0 + beta_B_st B - beta_F_st F + beta_A_st A
                        + beta_C_st C(t), sigma_st^2)                             [log-Gaussian, wake-gated]

Date:    25 April 2026
Version: 0.1
"""

from models.fsa_high_res.simulation import HIGH_RES_FSA_MODEL
from models.fsa_high_res.estimation import HIGH_RES_FSA_ESTIMATION

"""2-state controlled-bistable model for mode-preservation + control tests.

    dx = [alpha * x * (a^2 - x^2) + u] dt + sqrt(2 sigma_x) dB_x
    du = -gamma * (u - u_target(t)) dt + sqrt(2 sigma_u) dB_u
    y_k = x(t_k) + N(0, sigma_obs^2)

u is a tilt/barrier process that biases x toward the preferred well;
u_target(t) is a fixed (not estimated) piecewise-constant schedule
expressing the intervention plan.

Critical tilt for alpha = a = 1:  u_c = 2/(3 sqrt(3)) ~= 0.385.
Defaults: u_on = 0.5 (supercritical), u_maint = 0.2 (subcritical).

Parameters:  alpha, a, sigma_x, gamma, sigma_u, sigma_obs        (6)
Init state:  x_0, u_0                                            (2)
Total:       8

Date:    18 April 2026
Version: 1.0
"""

from models.bistable_controlled.estimation import BISTABLE_CTRL_ESTIMATION
from models.bistable_controlled.simulation import BISTABLE_CTRL_MODEL

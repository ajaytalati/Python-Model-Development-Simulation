"""3-State Fitness-Strain-Amplitude Model.

A minimal dynamical model of HPA-HPG axis interaction during fitness
intervention, reduced from the full 6-ODE endocrine model via
Stuart-Landau amplitude reduction.

State variables (all in day-scale dynamics):
    B : base fitness                in [0, 1]          (cardio-metabolic)
    F : accumulated strain          in R_>=0           (allostatic load)
    A : HPG pulsatility amplitude   in R_>=0           (testosterone pulses)

Exogenous inputs (clinician controls):
    T_B(t) : adaptation target     in [0, 1]
    Phi(t) : strain production     in R_>=0

Parameters (13 dynamics + 1 observation noise = 14 total):
    tau_B, alpha_A, tau_F, lambda_B, lambda_A,
    mu_0, mu_B, mu_F, mu_FF, eta,
    sigma_B, sigma_F, sigma_A,
    sigma_obs

The model exhibits three canonical phenomena under a SINGLE parameter
set, differing only in the exogenous input schedule:
    - S1 (sedentary)    : no intervention -> stuck in pathological basin
    - S2 (recovery)     : moderate intervention -> cross bifurcation -> healthy
    - S3 (overtraining) : recovery then Phi jump -> cliff collapse

See: 3_State_HPA_HPG_Model_Specification_and_Analysis.md

Date:    18 April 2026
Version: 1.0
"""

from models.fitness_strain_amplitude.estimation import FSA_ESTIMATION
from models.fitness_strain_amplitude.simulation import FSA_MODEL

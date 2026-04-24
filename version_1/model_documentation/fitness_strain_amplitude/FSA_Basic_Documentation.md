# The FSA Model — Specification
## Fitness-Strain-Amplitude 3-State SDE with Six-Channel Observation Model (v4.1)

**Version:** 4.1 (Real-Obs, Rolling SMC²)
**Author:** Gemini Gem (chat)
**Date:** 22 April 2026
**State dimension:** 3 latent states ($B, F, A$)
**Observation dimension:** 6 physiological channels
**Parameter dimension:** 34 estimated parameters
**Sources:** [fitness_strain_amplitude/simulation.py](../../models/fitness_strain_amplitude/simulation.py), [fsa_real_obs/simulation.py](../../models/fsa_real_obs/simulation.py)

---

## 1. What this model is for

The Fitness-Strain-Amplitude (FSA) model is a 3-state continuous-time stochastic model of an athlete's **daily-timescale** physiological state. The latent triple $(B, F, A)$ represents:

- $B$ — **Fitness**: cumulative positive adaptation, on $[0, 1]$, evolving on ~14-day timescale.
- $F$ — **Strain**: acute training load / inflammation, $\geq 0$, evolving on ~7-day timescale.
- $A$ — **Endocrine amplitude**: HPG/HPA pulsatility amplitude, $\geq 0$, governed by a supercritical Landau-pitchfork bifurcation driven by $(B, F)$.

The clinical/performance question the model is built around is:

> *Given several weeks of training load, resting heart rate, subjective stress, sleep and exercise timing, is this athlete adapting positively (rising $B$) or heading toward overtraining (collapsed $A$)? And which training-load adjustment restores amplitude?*

Two observation-model variants coexist in the codebase and share the SDE specified below:

1. **Direct-observation variant** — [`fitness_strain_amplitude`](../../models/fitness_strain_amplitude/): direct Gaussian observations of $B, F, A$ with a shared $\sigma_\mathrm{obs}$. Used for estimator development and identifiability studies.
2. **Real-observation variant** — [`fsa_real_obs`](../../models/fsa_real_obs/): the six physiological channels described in §3 of this document (RHR, intensity, duration, stress, sleep, timing). This is the production-facing model; the [fsa_real_obs_Documentation](../fsa_real_obs/fsa_real_obs_Documentation.md) gives channel-by-channel tables and parameter-set scenarios.

This document contains the full mathematical structure, the v4.1 orthogonalised RHR reparameterisation, and the concrete priors for the 34 parameters tracked by the rolling SMC² estimator.

---

## 2. Latent Dynamical System (SDEs)

The core biology is governed by a 3-dimensional system of Itô stochastic differential equations. Time $t$ is measured in **days**.

### 1.1 Exogenous Inputs

- $T_B(t) \in [0, 1]$: Adaptation Target (Training Load)
    
- $\Phi(t) \in \mathbb{R}_{\geq 0}$: Strain Production (Training Volume/Intensity)
    

### 1.2 The State Equations

**1. Fitness (**$B$**) — Jacobi Diffusion:**

$$dB_t = \frac{1 + \alpha_A A_t}{\tau_B} \Big(T_B(t) - B_t\Big) dt + \sigma_B \sqrt{B_t(1-B_t)} dW_{B,t}$$

**2. Strain (**$F$**) — CIR Diffusion:**

$$dF_t = \left[ \Phi(t) - \frac{1 + \lambda_B B_t + \lambda_A A_t}{\tau_F} F_t \right] dt + \sigma_F \sqrt{F_t} dW_{F,t}$$

**3. Endocrine Amplitude (**$A$**) — Regularized Landau:**

$$dA_t = \Big(\mu(B_t, F_t) A_t - \eta A_t^3\Big) dt + \sigma_A \sqrt{A_t + \epsilon_A} dW_{A,t}$$

Where the bifurcation parameter $\mu$ drives the phase transition:

$$\mu(B_t, F_t) = \mu_0 + \mu_B B_t - \mu_F F_t - \mu_{FF} F_t^2$$

## 3. Fixed Constants & Initial States (Not Estimated)

To guarantee structural identifiability, the following variables are strictly frozen and removed from the SMC² parameter block.

**Process Noise Scales:**

- $\sigma_B = 0.01$  
    
- $\sigma_F = 0.005$  
    
- $\sigma_A = 0.02$  
    
- $\epsilon_A = 10^{-4}$ (Non-absorbing boundary regularization)
    

**Latent Initial States (**$B_0, F_0, A_0$**):**

- _Window_ 1 _(Cold Start):_ Fixed to $[0.05, 0.10, 0.01]$.
    
- _Windows 2+ (Warm Bridge):_ Hardcoded to the PF-extracted, smoothed posterior state from the previous window at $t = \text{STRIDE\_DAYS}$.
    

## 4. Observation Model (6 Independent Gaussian Channels)

### 3.1 The Orthogonalized RHR Channel (Channel 1)

To break the $(B, F)$ collinearity, RHR is mean-centered (removing $R_{base}$ from estimation) and the sensitivities are reparameterized into a ratio.

**Internal Transformation:**

- $\kappa_{chronic} = \frac{\kappa_{total}}{1 + \kappa_{ratio}}$  
    
- $\kappa_{vagal} = \kappa_{ratio} \cdot \kappa_{chronic}$  
    

**Observation Equation:**

$$RHR_{centered}(t) \sim \mathcal{N}\Big( - \kappa_{vagal} B(t) + \kappa_{chronic} F(t), \, \sigma_{obs,R}^2 \Big)$$

_(Note:_ $RHR_{centered}$ _is computed dynamically per-window during preprocessing by subtracting the rolling mean)._

### 3.2 Performance & Behavioral Channels (Channels 2-6)

**Channel 2: Global Performance Intensity**

$$I_{norm}(t) \sim \mathcal{N}\Big(I_{base} + c_B B(t) - c_F F(t), \, \sigma_{obs,I}^2\Big)$$

**Channel 3: Global Performance Duration**

$$D_{norm}(t) \sim \mathcal{N}\Big(D_{base} + d_B B(t) - d_F F(t), \, \sigma_{obs,D}^2\Big)$$

**Channel 4: Daily Stress**

$$S_{obs}(t) \sim \mathcal{N}\Big(S_{base} - s_A A(t) + s_F F(t), \, \sigma_{obs,S}^2\Big)$$

**Channel** 5: Sleep Quality

$$Sleep_{norm}(t) \sim \mathcal{N}\Big(Sleep_{base} + sl_B B(t) - sl_F F(t) + sl_A A(t), \, \sigma_{obs,Sleep}^2\Big)$$

**Channel 6: Circadian Exercise Timing**

_(Note: Bounded circadian scores are transformed via logit_ $\ln(\frac{y}{1-y})$ _to map to_ $(-\infty, \infty)$ _support)._

$$Time_{logit}(t) \sim \mathcal{N}\Big(Time_{base} + t_A A(t) - t_F F(t), \, \sigma_{obs,Time}^2\Big)$$

## 5. The Parameter Vector & Concrete Priors ($\theta \in \mathbb{R}^{34}$)

The SMC² algorithm tracks exactly 34 parameters. Lognormal distributions are parameterized by the mean and standard deviation of the underlying normal: $\text{Lognormal}(\ln(\mu), \sigma)$.

### 4.1 Dynamical Parameters (10)

|   |   |   |
|---|---|---|
|**Parameter**|**Description**|**Concrete Prior**|
|$\tau_B$|Fitness time constant|$\text{Lognormal}(\ln(14.0), 0.08)$|
|$\alpha_A$|Amplitude enhancement of fitness|$\text{Lognormal}(\ln(1.0), 0.4)$|
|$\tau_F$|Strain time constant|$\text{Lognormal}(\ln(7.0), 0.3)$|
|$\lambda_B$|Fitness enhancement of recovery|$\text{Lognormal}(\ln(3.0), 0.3)$|
|$\lambda_A$|Amplitude enhancement of recovery|$\text{Lognormal}(\ln(1.5), 0.3)$|
|$\mu_0$ (abs)|Absolute baseline bifurcation|$\text{Lognormal}(\ln(0.10), 0.4)$|
|$\mu_B$|Fitness protection of bifurcation|$\text{Lognormal}(\ln(0.30), 0.4)$|
|$\mu_F$|Linear strain penalty|$\text{Lognormal}(\ln(0.10), 0.4)$|
|$\mu_{FF}$|Quadratic strain penalty|$\text{Lognormal}(\ln(0.40), 0.4)$|
|$\eta$|Landau restoring force|$\text{Lognormal}(\ln(0.20), 0.3)$|

### 4.2 Observation Parameters (24)

**RHR (3 Params):**

|   |   |   |
|---|---|---|
|**Parameter**|**Description**|**Concrete Prior**|
|$\kappa_{ratio}$|Ratio of vagal vs. chronic tone|$\text{Lognormal}(\ln(1.2), 0.20)$|
|$\kappa_{total}$|Total RHR sensitivity scale|$\text{Lognormal}(\ln(22.0), 0.30)$|
|$\sigma_{obs,R}$|Measurement noise|$\text{Lognormal}(\ln(1.5), 0.4)$|

**Intensity (4 Params):**

|   |   |   |
|---|---|---|
|**Parameter**|**Description**|**Concrete Prior**|
|$I_{base}$|Baseline capacity|$\text{Normal}(0.5, 0.1)$|
|$c_B$|Fitness sensitivity|$\text{Lognormal}(\ln(0.2), 0.5)$|
|$c_F$|Strain penalty|$\text{Lognormal}(\ln(0.1), 0.5)$|
|$\sigma_{obs,I}$|Measurement noise|$\text{Lognormal}(\ln(0.05), 0.4)$|

**Duration (4 Params):**

|   |   |   |
|---|---|---|
|**Parameter**|**Description**|**Concrete Prior**|
|$D_{base}$|Baseline capacity|$\text{Normal}(0.5, 0.1)$|
|$d_B$|Fitness sensitivity|$\text{Lognormal}(\ln(0.3), 0.5)$|
|$d_F$|Strain penalty|$\text{Lognormal}(\ln(0.2), 0.5)$|
|$\sigma_{obs,D}$|Measurement noise|$\text{Lognormal}(\ln(0.08), 0.4)$|

**Stress (4 Params):**

|   |   |   |
|---|---|---|
|**Parameter**|**Description**|**Concrete Prior**|
|$S_{base}$|Baseline stress|$\text{Normal}(30.0, 10.0)$|
|$s_A$|Vitality suppression|$\text{Lognormal}(\ln(15.0), 0.5)$|
|$s_F$|Strain increase|$\text{Lognormal}(\ln(20.0), 0.5)$|
|$\sigma_{obs,S}$|Measurement noise|$\text{Lognormal}(\ln(5.0), 0.4)$|

**Sleep (5 Params):**

|   |   |   |
|---|---|---|
|**Parameter**|**Description**|**Concrete Prior**|
|$Sleep_{base}$|Baseline quality|$\text{Normal}(0.5, 0.1)$|
|$sl_A$|Amplitude enhancement|$\text{Lognormal}(\ln(0.2), 0.5)$|
|$sl_B$|Fitness enhancement|$\text{Lognormal}(\ln(0.1), 0.5)$|
|$sl_F$|Strain disruption|$\text{Lognormal}(\ln(0.2), 0.5)$|
|$\sigma_{obs,Sleep}$|Measurement noise|$\text{Lognormal}(\ln(0.1), 0.4)$|

**Timing (4 Params):**

|                     |                        |                                   |
| ------------------- | ---------------------- | --------------------------------- |
| **Parameter**       | **Description**        | **Concrete Prior**                |
| $Time_{base}$       | Baseline tendency      | $\text{Normal}(0.0, 1.0)$         |
| $t_A$               | Amplitude morning pull | $\text{Lognormal}(\ln(1.0), 0.5)$ |
| $t_F$               | Strain disruption      | $\text{Lognormal}(\ln(0.5), 0.5)$ |
| $\sigma_{obs,Time}$ | Measurement noise      | $\text{Lognormal}(\ln(0.5), 0.4)$ |

---

## 6. Parameter sets and exogenous input scenarios

The `fsa_real_obs` implementation ships three named exogenous-input scenarios, all sharing the latent parameters in §5 and initial conditions $(B_0, F_0, A_0) = (0.05, 0.10, 0.01)$:

| Scenario | $T_B$ | $\Phi_1$ | $\Phi_2$ | $T_\mathrm{jump}$ | $T_\mathrm{end}$ | Interpretation |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `sedentary` | 0.0 | 0.0 | 0.0 | 1000 d | 120 d | no training |
| `recovery` | 0.6 | 0.03 | 0.03 | 1000 d | 200 d | moderate constant load |
| `overtraining` | 0.6 | 0.03 | 0.20 | 150 d | 240 d | moderate load jumping to overtraining at day 150 |

The `overtraining` scenario is the bifurcation test: post-jump $\Phi_2 = 0.20$ drives $F$ high enough that $\mu(B, F) < 0$, collapsing the endocrine amplitude $A$.

The base `fitness_strain_amplitude` model uses the same three scenarios with direct Gaussian observation of $(B, F, A)$ instead of the six-channel model in §4.

---

## 7. Pointers

- **Implementation (base, direct obs):** [version_1/models/fitness_strain_amplitude/](../../models/fitness_strain_amplitude/)
- **Implementation (real-obs, 6 channels):** [version_1/models/fsa_real_obs/](../../models/fsa_real_obs/)
- **Simulator CLI (base):** `python simulator/run_simulator.py --model models.fitness_strain_amplitude.simulation.FSA_MODEL --param-set recovery`
- **Simulator CLI (real-obs):** `python simulator/run_simulator.py --model models.fsa_real_obs.simulation.FSA_REAL_OBS_MODEL --param-set recovery`
- **Real-obs channel-by-channel doc:** [fsa_real_obs_Documentation.md](../fsa_real_obs/fsa_real_obs_Documentation.md)

---

*End of document.*
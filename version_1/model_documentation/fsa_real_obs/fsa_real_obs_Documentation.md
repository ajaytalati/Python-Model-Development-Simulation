# The FSA Real-Obs Model — Specification
## 3-State FSA SDE with Six Physiological Observation Channels

**Version:** 1.0
**Source:** [version_1/models/fsa_real_obs/simulation.py](../../models/fsa_real_obs/simulation.py)

---

## 1. What this model is for

`fsa_real_obs` is the observation-model-rich variant of the [Fitness-Strain-Amplitude](../fitness_strain_amplitude/FSA_Basic_Documentation.md) 3-state SDE. The latent dynamics are identical to the base FSA model — a Jacobi/CIR/Landau triple representing fitness $B$, strain $F$, and endocrine amplitude $A$ — but the observation model replaces direct state observation with **six independent Gaussian channels** representing quantities that a wearable or training log can actually measure: resting heart rate, training intensity, duration, perceived stress, sleep quality, and circadian timing.

This is the model that downstream estimation pipelines (SMC² in v4.1) are calibrated against.

---

## 2. States and parameters at a glance

### 2.1 The three-state latent dynamical system

| Symbol | Meaning | Role | Range | Timescale |
|:---:|:---|:---|:---:|:---:|
| $B$ | fitness | stochastic (Jacobi) | $[0, 1]$ | $\tau_B \approx 14$ d |
| $F$ | strain | stochastic (CIR) | $\geq 0$ | $\tau_F \approx 7$ d |
| $A$ | endocrine amplitude | stochastic (Landau) | $\geq 0$ | determined by $\mu, \eta$ |

Time is measured in **days**.

### 2.2 The parameter list — 13 latent drift + 24 observation = 37

#### Latent-dynamics block (13 parameters — same as base FSA)

| Parameter | Typical value | Role |
|:---:|:---:|:---|
| `tau_B` | 14.0 | fitness timescale (days) |
| `alpha_A` | 1.0 | amplitude enhancement of fitness recovery |
| `tau_F` | 7.0 | strain timescale (days) |
| `lambda_B` | 3.0 | fitness-driven strain clearance |
| `lambda_A` | 1.5 | amplitude-driven strain clearance |
| `mu_0` | $-0.10$ | baseline bifurcation parameter |
| `mu_B` | 0.30 | fitness protection of bifurcation |
| `mu_F` | 0.10 | linear strain penalty |
| `mu_FF` | 0.40 | quadratic strain penalty |
| `eta` | 0.20 | Landau cubic saturation |
| `sigma_B`, `sigma_F`, `sigma_A` | 0.01, 0.005, 0.02 | diffusion scales (frozen) |

#### Observation block (24 parameters, 4 per channel × 6 channels)

Each channel contributes a baseline, one or more state-sensitivity coefficients, and an observation-noise scale. See §4 for the channel-by-channel breakdown.

#### Initial-condition block (3)

$B_0, F_0, A_0$ with defaults $[0.05, 0.10, 0.01]$.

---

## 3. The latent SDE system

**Fitness** — Jacobi diffusion on $[0, 1]$:

$$
dB = \frac{1 + \alpha_A A}{\tau_B}\,\bigl(T_B(t) - B\bigr)\,dt + \sigma_B\,\sqrt{B(1 - B)}\,dW_B.
$$

**Strain** — CIR diffusion:

$$
dF = \Bigl[\Phi(t) - \frac{1 + \lambda_B B + \lambda_A A}{\tau_F}\,F\Bigr]\,dt + \sigma_F\,\sqrt{F}\,dW_F.
$$

**Endocrine amplitude** — regularised Landau (supercritical pitchfork):

$$
dA = \bigl[\mu(B, F)\,A - \eta\,A^3\bigr]\,dt + \sigma_A\,\sqrt{A + \varepsilon_A}\,dW_A,
$$

with

$$
\mu(B, F) = \mu_0 + \mu_B\,B - \mu_F\,F - \mu_{FF}\,F^2.
$$

The regularisation $\varepsilon_A = 10^{-4}$ makes $A = 0$ non-absorbing.

**Exogenous inputs** (piecewise-constant, emitted as channels):

- $T_B(t)$ — adaptation target / training load (single segment, constant).
- $\Phi(t)$ — strain production / training volume-intensity (two segments with jump at $T_\mathrm{jump}$).

---

## 4. Observation model (6 Gaussian channels)

### 4.1 RHR — resting heart rate

$$
\mathrm{RHR}(t) \sim \mathcal{N}\bigl(R_\mathrm{base} - \kappa_\mathrm{vagal}\,B(t) + \kappa_\mathrm{chronic}\,F(t), \sigma_{\mathrm{obs},R}^2\bigr).
$$

### 4.2 Intensity — global performance intensity

$$
I(t) \sim \mathcal{N}\bigl(I_\mathrm{base} + c_B\,B(t) - c_F\,F(t), \sigma_{\mathrm{obs},I}^2\bigr).
$$

### 4.3 Duration — training session duration

$$
D(t) \sim \mathcal{N}\bigl(D_\mathrm{base} + d_B\,B(t) - d_F\,F(t), \sigma_{\mathrm{obs},D}^2\bigr).
$$

### 4.4 Stress — daily subjective stress

$$
S(t) \sim \mathcal{N}\bigl(S_\mathrm{base} - s_A\,A(t) + s_F\,F(t), \sigma_{\mathrm{obs},S}^2\bigr).
$$

### 4.5 Sleep — sleep quality score

$$
\mathrm{Sleep}(t) \sim \mathcal{N}\bigl(\mathrm{Sleep}_\mathrm{base} + sl_A\,A(t) + sl_B\,B(t) - sl_F\,F(t), \sigma_{\mathrm{obs,Sleep}}^2\bigr).
$$

### 4.6 Timing — circadian exercise timing (logit-transformed)

$$
\mathrm{Time}_\mathrm{logit}(t) \sim \mathcal{N}\bigl(\mathrm{Time}_\mathrm{base} + t_A\,A(t) - t_F\,F(t), \sigma_{\mathrm{obs,Time}}^2\bigr).
$$

Each channel is observed densely on the simulation grid (time unit = days). A companion document ([FSA Mathematical Specification v4.1](../fitness_strain_amplitude/FSA_Basic_Documentation.md)) describes the orthogonalised $(\kappa_\mathrm{ratio}, \kappa_\mathrm{total})$ reparameterisation of RHR used in the inference pipeline.

---

## 5. Parameter sets for simulation testing

Three exogenous-input scenarios are defined, all sharing the same latent parameters and initial conditions:

| Scenario | $T_B$ | $\Phi_1$ | $\Phi_2$ | $T_\mathrm{jump}$ | $T_\mathrm{end}$ | Interpretation |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `sedentary` | 0.0 | 0.0 | 0.0 | 1000 d | 120 d | no training at all |
| `recovery` | 0.6 | 0.03 | 0.03 | 1000 d | 200 d | moderate constant training load |
| `overtraining` | 0.6 | 0.03 | 0.20 | 150 d | 240 d | moderate load that jumps to overtraining at day 150 |

The `overtraining` scenario is the one that stresses the model's bifurcation structure: the post-jump $\Phi_2 = 0.20$ pushes $F$ high enough that $\mu(B, F) < 0$ and the endocrine amplitude $A$ collapses.

---

## 6. Pointers

- **Implementation:** [version_1/models/fsa_real_obs/](../../models/fsa_real_obs/)
- **Simulator CLI:** `python simulator/run_simulator.py --model models.fsa_real_obs.simulation.FSA_REAL_OBS_MODEL --param-set recovery`
- **Base model (direct state observation):** [fitness_strain_amplitude](../fitness_strain_amplitude/FSA_Basic_Documentation.md) — same SDE, simpler observation model for estimator development.
- **Mathematical spec with priors and SMC² parameterisation:** [FSA_Basic_Documentation.md](../fitness_strain_amplitude/FSA_Basic_Documentation.md) — contains the v4.1 orthogonalisation details and concrete priors for all 34 parameters tracked by SMC².

---

*End of document.*

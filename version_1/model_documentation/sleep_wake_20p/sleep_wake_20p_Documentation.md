# The Sleep-Wake 20p Model — Specification
## Minimal 20-Parameter Sleep-Wake-Adenosine SDE

**Version:** 1.0
**Source:** [version_1/models/sleep_wake_20p/simulation.py](../../models/sleep_wake_20p/simulation.py)
**Live spec:** [version_1/models/sleep_wake_20p/TESTING.md](../../models/sleep_wake_20p/TESTING.md) (§2)

---

## 1. What this model is for

`sleep_wake_20p` is the **identifiability-proof-driven** minimal sleep-wake-adenosine SDE — a 5-state latent system derived from `Identifiability_Proof_17_Parameter_Sleep_Wake_Adenosine_Model.md`, plus three diffusion temperatures $\{T_W, T_Z, T_a\}$ that promote the latent layer from an ODE to a true SDE. It is the direct predecessor of [SWAT](../swat/SWAT_Basic_Documentation.md): SWAT's 17-parameter fast subsystem is lifted from this model verbatim.

The observation model is deliberately minimal — only two channels (continuous HR, binary sleep) — so that estimator properties can be studied without the complications of the full Garmin observation suite.

The canonical, actively-maintained specification and test suite lives at [TESTING.md §2–§9](../../models/sleep_wake_20p/TESTING.md). This document is an audience-facing summary.

---

## 2. States and parameters at a glance

### 2.1 The six state components (5 stochastic + constants)

| Symbol | Meaning | Role | Range | Timescale |
|:---:|:---|:---|:---:|:---:|
| $W$ | wakefulness | stochastic | $[0, 1]$ | $\tau_W \approx 2$ h |
| $\tilde Z$ | sleep depth (rescaled) | stochastic | $[0, A = 6]$ | $\tau_Z \approx 2$ h |
| $a$ | adenosine / sleep pressure | stochastic | $\geq 0$ | $\tau_a \approx 3$ h |
| $C$ | external light cycle | analytical-deterministic | $[-1, 1]$ | 24 h |
| $V_h$ | vitality reserve | constant (Phase 1) | $\mathbb{R}$ | — |
| $V_n$ | chronic load | constant (Phase 1) | $\mathbb{R}$ | — |

The rescaling constant $A = 6$ is hard-coded (`A_SCALE`), not a parameter. $V_h, V_n$ are constants in Phase 1 but estimated as per-subject scalars.

### 2.2 The 20-parameter block

#### Fast subsystem (8 parameters)

| Parameter | Symbol | Role |
|:---:|:---:|:---|
| `kappa` | $\kappa$ | $\tilde Z \to u_W$ inhibition |
| `lmbda` | $\lambda$ | circadian drive amplitude in $u_W$ |
| `gamma_3` | $\gamma_3$ | $W \to u_Z$ inhibition |
| `tau_W`, `tau_Z`, `tau_a` | timescales | — |
| `beta_Z` | $\beta_Z$ | $a \to u_Z$ coupling |
| `phi` | $\phi$ | circadian phase |

#### Observation (4 parameters)

| Parameter | Symbol | Role |
|:---:|:---:|:---|
| `HR_base`, `alpha_HR`, `sigma_HR` | HR channel | intercept, gain, noise |
| `c_tilde` | $\tilde c$ | sleep-detection threshold on $\tilde Z$ |

#### Constants as states (2)

| Parameter | Symbol |
|:---:|:---:|
| `Vh` | $V_h$ |
| `Vn` | $V_n$ |

#### Fast-state diffusion (3)

| Parameter | Symbol |
|:---:|:---:|
| `T_W`, `T_Z`, `T_a` | diffusion temperatures |

#### Initial conditions (3)

$W_0, \tilde Z_0, a_0$.

**Count audit:** 8 + 4 + 2 + 3 = 17 deterministic + 3 diffusion = **20 parameters**, plus 3 initial conditions.

---

## 3. The SDE system

**Wakefulness**

$$
dW = \frac{1}{\tau_W}\bigl[\sigma(u_W) - W\bigr]\,dt + \sqrt{2 T_W}\,dB_W, \qquad u_W = -\kappa\,\tilde Z + \lambda\,C(t) + V_h + V_n - a.
$$

**Sleep depth**

$$
d\tilde Z = \frac{1}{\tau_Z}\bigl[A\,\sigma(u_Z) - \tilde Z\bigr]\,dt + \sqrt{2 T_Z}\,dB_Z, \qquad u_Z = -\gamma_3\,W - V_n + \beta_Z\,a.
$$

**Adenosine**

$$
da = \frac{1}{\tau_a}\,(W - a)\,dt + \sqrt{2 T_a}\,dB_a.
$$

**Light cycle** (analytical):

$$
C(t) = \sin\!\bigl(2\pi t / 24 + \phi\bigr).
$$

**Vitality / chronic load** (Phase 1 — constants):

$$
dV_h = 0, \qquad dV_n = 0.
$$

---

## 4. Observation model

**Heart rate** (continuous Gaussian):

$$
\mathrm{HR}(t_k) \sim \mathcal{N}\bigl(\mathrm{HR}_\mathrm{base} + \alpha_\mathrm{HR}\,W(t_k), \sigma_\mathrm{HR}^2\bigr).
$$

**Binary sleep** (Bernoulli on rescaled sleep depth):

$$
\Pr\bigl(S_k = 1 \mid \tilde Z(t_k)\bigr) = \sigma\!\bigl(\tilde Z(t_k) - \tilde c\bigr).
$$

---

## 5. Parameter sets for simulation testing

Per [TESTING.md](../../models/sleep_wake_20p/TESTING.md):

- **Set A** — healthy basin. Canonical parameter values with $V_h = 1.0, V_n = 0.3$.
- **Set B** — hyperarousal / insomnia. Same as A except $V_h = 0.2, V_n = 2.0$. The high $V_n$ suppresses sleep depth; $\tilde Z$ peaks around 1.3 rather than the ~5 of Set A.

Simulation horizon: 7 days, $dt = 5$ minutes. Refer to the TESTING.md for full numeric tables, quantitative thresholds, and the `verify_physics_fn` expectations.

---

## 6. Known issues and fixes

The model has two inherited bug fixes documented under [BUG_REPORT_gamma3_sleep_depth.md](../../models/sleep_wake_20p/BUG_REPORT_gamma3_sleep_depth.md) and [BUG_REPORT_ctilde_beta_z_sleep_false_positives.md](../../models/sleep_wake_20p/BUG_REPORT_ctilde_beta_z_sleep_false_positives.md):

1. `gamma_3` reduced from 60 to 8 so that the SDE noise floor does not suppress $\tilde Z$ below the sleep threshold.
2. `c_tilde = 3.0, beta_Z = 2.5, Zt_0 = 3.5` (were 1.5, 1.5, 1.8) to eliminate an 18 % daytime false-sleep rate.

Both fixes are inherited by SWAT's Set A parameter block.

---

## 7. Pointers

- **Implementation:** [version_1/models/sleep_wake_20p/](../../models/sleep_wake_20p/)
- **Simulator CLI:** `python simulator/run_simulator.py --model models.sleep_wake_20p.simulation.SLEEP_WAKE_20P_MODEL --param-set A`
- **Canonical spec + test suite:** [TESTING.md](../../models/sleep_wake_20p/TESTING.md)
- **Derived extension:** [SWAT](../swat/SWAT_Basic_Documentation.md) — adds a Stuart-Landau testosterone state and the $V_c$ phase-shift parameter.

---

*End of document.*

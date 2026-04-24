# The Sleep-Wake Model — Specification
## 6-State Sleep-Wake SDE with Four Wearable-Sensor Channels

**Version:** 1.0
**Source:** [version_1/models/sleep_wake/simulation.py](../../models/sleep_wake/simulation.py)

---

## 1. What this model is for

`sleep_wake` is a 6-state continuous-time stochastic model of sleep-wake-circadian dynamics with a slow vitality/chronic-load layer. It is the "full observation-model" sibling of [`sleep_wake_20p`](../sleep_wake_20p/sleep_wake_20p_Documentation.md): same fast subsystem, but with four wearable-sensor channels (HR, stress, steps, sleep-stages) generated at realistic Garmin-like cadences. Intended target is **N-of-1 Bayesian inference on real wrist-worn sensor data**.

The model outputs are written in the Garmin CSV format ([csv_writer.py](../../models/sleep_wake/csv_writer.py)) so the same estimation pipeline consumes synthetic and real data interchangeably.

---

## 2. States and parameters at a glance

### 2.1 The six-state dynamical system

| Symbol | Meaning | Role | Range | Timescale |
|:---:|:---|:---|:---:|:---:|
| $W$ | wakefulness | stochastic | $[0, 1]$ | $\tau_W \approx 2$ h |
| $Z$ | sleep depth | stochastic | $[0, 1]$ | $\tau_Z \approx 2$ h |
| $A$ | adenosine / sleep pressure | stochastic | $\geq 0$ | $\tau_A = 1/(k_\mathrm{out} + k_\mathrm{glymph})$ |
| $C$ | external light cycle | analytical-deterministic | $[-1, 1]$ | 24 h |
| $V_h$ | vitality reserve | stochastic (slow) | $[0, 1]$ | $\tau_{V_h} \approx 7$ d |
| $V_n$ | chronic load | stochastic (slow) | $[0, 1]$ | $\tau_{V_n} \approx 7$ d |

### 2.2 The parameter list — grouped by block

#### Block F — Fast sleep-wake subsystem (9 parameters)

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `tau_W`, `tau_Z` | $\tau_W, \tau_Z$ | 2 h each | sigmoid response times |
| `g_w`, `g_z` | $g_W, g_Z$ | 8, 10 | sigmoid gains |
| `a_w`, `z_w` | $a_W, z_W$ | 3, 5 | adenosine/sleep-depth inhibition of $W$ |
| `a_z`, `w_z` | $a_Z, w_Z$ | 4, 6 | adenosine/wakefulness inhibition of $Z$ |
| `c_amp` | $c_\mathrm{amp}$ | 4.0 | circadian drive amplitude |

#### Block C — Circadian (1 parameter)

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `phi` | $\phi$ | $-\pi/3$ | circadian phase (morning-type) |

#### Block A — Adenosine homeostasis (3 parameters)

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `k_in` | $k_\mathrm{in}$ | 0.8 | production rate during wake |
| `k_out` | $k_\mathrm{out}$ | 0.3 | basal clearance rate |
| `k_glymph` | $k_\mathrm{glymph}$ | 2.0 | glymphatic clearance during sleep |

#### Block V — Vitality & chronic load (8 parameters)

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `tau_Vh`, `tau_Vn` | — | 7 d each | $V_h, V_n$ relaxation timescales |
| `beta_h`, `beta_n` | — | 0.1 each | entrainment coupling into $V_h, V_n$ |
| `gamma_0`, `gamma_steps` | — | 0.5, 0.1 | daily-activity → $V_h$ target |
| `T_Vh`, `T_Vn` | — | 0.05 each | diffusion temperatures (per day) |

#### Block N — Fast-state diffusion (3 parameters)

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `T_W`, `T_Z`, `T_A` | — | $\sim 10^{-2}$ | diffusion temperatures for $W, Z, A$ |

#### Block S — Ordered-logistic sleep channel (4 parameters)

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `alpha_sleep` | — | 6.0 | ordered-logistic slope |
| `c_d`, `c_r`, `c_l` | — | 0.85, 0.60, 0.25 | Deep / REM / Light thresholds on $Z$ |

#### Block HR — Heart-rate channel (4 parameters)

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `HR_base` | $\mathrm{HR}_\mathrm{base}$ | 50 bpm | HR intercept |
| `alpha_HR` | $\alpha_\mathrm{HR}$ | 25 bpm | HR gain on $W$ |
| `beta_exercise` | $\beta_\mathrm{ex}$ | 40 bpm | exercise increment |
| `sigma_HR` | $\sigma_\mathrm{HR}$ | 8 bpm | HR observation noise |

#### Block St — Stress channel (4 parameters)

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `s_base` | $s_0$ | 3 | baseline stress |
| `s_W` | $s_W$ | 40 | wake modulation |
| `s_n` | $s_n$ | 3 | $V_n$ coupling |
| `sigma_S` | $\sigma_S$ | 10 | stress observation noise |

#### Block P — Steps zero-inflated lognormal (4 parameters)

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `p_move` | — | 0.35 | probability of movement in a wake bin |
| `r_step` | — | 1200 | median steps per active 15-min bin |
| `alpha_run` | — | 0.5 | running multiplier (log-scale) |
| `sigma_step` | — | 0.8 | lognormal log-space std |

#### Initial-condition block (5 parameters)

$W(0), Z(0), A(0), V_h(0), V_n(0)$.

---

## 3. The SDE system

**Wakefulness**

$$
dW = \frac{1}{\tau_W}\Bigl[\sigma(g_W X_W) - W\Bigr]\,dt + \sqrt{2 T_W}\,dB_W
$$

with sigmoid argument

$$
X_W = c_\mathrm{amp}\,C(t) + V_h + V_n - a_W\,A - z_W\,Z.
$$

**Sleep depth**

$$
dZ = \frac{1}{\tau_Z}\Bigl[\sigma(g_Z X_Z) - Z\Bigr]\,dt + \sqrt{2 T_Z}\,dB_Z, \qquad X_Z = -c_\mathrm{amp}\,C(t) + a_Z\,A - w_Z\,W - V_n.
$$

**Adenosine**

$$
dA = \bigl[k_\mathrm{in}\,W - (k_\mathrm{out} + k_\mathrm{glymph}\,Z)\,A\bigr]\,dt + \sqrt{2 T_A}\,dB_A.
$$

**Light cycle** (analytical):

$$
C(t) = \sin\!\bigl(2\pi t / 24 + \phi\bigr).
$$

**Vitality / chronic load** (slow, with entrainment coupling)

$$
dV_h = \frac{V_{h,\mathrm{target}}(t) - V_h}{24\,\tau_{V_h}}\,dt + \frac{\beta_h}{24}\,E\,dt + \sqrt{2 T_{V_h}/24}\,dB_{V_h}
$$

$$
dV_n = \frac{V_{n,\mathrm{target}} - V_n}{24\,\tau_{V_n}}\,dt - \frac{\beta_n}{24}\,E\,dt + \sqrt{2 T_{V_n}/24}\,dB_{V_n}
$$

where $V_{h,\mathrm{target}}(t) = \gamma_0 + \gamma_\mathrm{steps} \cdot \mathrm{daily\_bins}(t)$ is driven by a daily activity schedule (exogenous input), and the entrainment quality

$$
E = \sigma\!\bigl(\kappa(c_\mathrm{amp}^2 - \mu_W^2)\bigr)\,\sigma\!\bigl(\kappa(c_\mathrm{amp}^2 - \mu_Z^2)\bigr), \quad \kappa = 2,
$$

with $\mu_W, \mu_Z$ the slow-backdrop approximations of $X_W, X_Z$ (ignoring the fast circadian term).

Here $\sigma(u) = 1/(1 + e^{-u})$ and $B_\ast$ are independent standard Brownian motions.

---

## 4. Observation model

Four Garmin-style channels, each at its native cadence, with realistic watch-off gaps.

### 4.1 Heart rate (1-minute resolution)

$$
\mathrm{HR}(t) \sim \mathcal{N}\bigl(\mathrm{HR}_\mathrm{base} + \alpha_\mathrm{HR}\,W(t) + \beta_\mathrm{ex}\,\mathbb{1}\{\mathrm{exercise}\}, \sigma_\mathrm{HR}^2\bigr),
$$

clipped to $[30, 220]$ bpm, with Poisson(~4/day) watch-off gaps of 5–30 minutes.

### 4.2 Stress (3-minute resolution)

$$
S(t) \sim \mathcal{N}\bigl(s_0 + s_W\,W(t) + s_n\,V_n(t), \sigma_S^2\bigr),
$$

clipped to $[0, 100]$, observed in roughly 60 % of 3-minute slots.

### 4.3 Steps (15-minute zero-inflated lognormal)

For each bin, with probability $p_\mathrm{active} = \bar W \cdot p_\mathrm{move}$, sample

$$
\mathrm{steps} \sim \mathrm{LogNormal}\bigl(\log r_\mathrm{step} + \alpha_\mathrm{run}\,\mathbb{1}\{\mathrm{run}\}, \sigma_\mathrm{step}^2\bigr).
$$

The "run" indicator fires with probability 0.15 during 07:00–10:00 solar time, else 0.03. Bins with step counts above 200 are flagged as exercise for the HR channel (back-coupling).

### 4.4 Sleep stages (native variable-length epochs)

Ordered logistic on $Z(t)$ with three thresholds (light, REM, deep), sampled at 1-minute resolution with stage-persistent holds, then run-length encoded into epochs. Stages $\in \{0 = \mathrm{Light}, 1 = \mathrm{REM}, 2 = \mathrm{Deep}, 3 = \mathrm{Awake}\}$.

---

## 5. Parameter sets for simulation testing

| Parameter group | Set A | Set B |
|:---|:---:|:---:|
| All drift/obs params | defaults (see Block tables above) | Set A × Uniform(0.8, 1.2) per-parameter (seed 123), except $\phi$ perturbed $\pm 0.3$ rad and slow-layer $\tau_{V_\ast}, \beta_\ast$ frozen |
| Initial state | $W_0 = 0.5, Z_0 = 0.3, A_0 = 0.5, V_{h,0} = 0.05, V_{n,0} = 0.5$ | same |
| Daily activity | `[6, 9, 13, 9, 14, 9, 9, 10, 8]` active bins/day (9-day schedule) | same |

Set A is the canonical healthy baseline. Set B is a perturbed parameter set for robustness testing.

---

## 6. Pointers

- **Implementation:** [version_1/models/sleep_wake/](../../models/sleep_wake/)
- **Simulator CLI:** `python simulator/run_simulator.py --model models.sleep_wake.simulation.SLEEP_WAKE_MODEL --param-set A`
- **Related model:** [sleep_wake_20p](../sleep_wake_20p/sleep_wake_20p_Documentation.md) — minimal 20-parameter variant with only HR + binary-sleep observations.
- **Garmin CSV exporter:** [csv_writer.py](../../models/sleep_wake/csv_writer.py).

---

*End of document.*

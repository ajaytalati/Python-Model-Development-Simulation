# The SWAT Model — Specification
## Sleep-Wake-Adenosine-Testosterone Dynamics with a Four-Channel Observation Model

**Version:** 2.0
**Date:** 22 April 2026
**Status:** Working specification — reflects the implementation under the accompanying Python package.
**Companion document:** `SWAT_Identifiability_Extension.md` — formal Fisher-rank analysis.

---

## 1. What this model is for

SWAT is a continuous-time stochastic model of the joint sleep-wake-hormonal state of a single subject, designed for **N-of-1 Bayesian inference** from wearable-sensor data. Given a week or two of heart-rate, sleep-stage, step-count, and Garmin-stress time series, the goal is to infer the subject's latent physiological state and the parameters of their personal dynamics.

The clinical question the model is built around is:

> *Why does this subject have low testosterone? Is it amplitude failure (chronic stress / insomnia), phase misalignment (shift work / jet lag), low vitality reserve, or recovery still incomplete? And which intervention would restore pulsatility?*

The answer takes the form of a joint posterior over four physiologically-interpretable parameters — $V_h$ (vitality reserve), $V_n$ (chronic load), $V_c$ (phase shift), and $T$ (pulsatility amplitude) — together with the subject-specific sleep/wake/circadian timescales and HPG parameters.

---

## 2. States and parameters at a glance

### 2.1 The seven-state dynamical system

The model has **seven state variables**, four of which are stochastic (diffused by independent Wiener processes) and three of which are deterministic (either analytical or constant-in-time):

| Symbol | Meaning | Role | Range | Timescale |
|:---:|:---|:---|:---:|:---:|
| $W$ | wakefulness | stochastic | $[0, 1]$ | $\tau_W \approx 2$ h |
| $\tilde Z$ | sleep depth (rescaled) | stochastic | $[0, A=6]$ | $\tau_Z \approx 2$ h |
| $a$ | adenosine / sleep pressure | stochastic | $\geq 0$ | $\tau_a \approx 3$ h |
| $T$ | testosterone pulsatility amplitude | stochastic | $\geq 0$ | $\tau_T \approx 48$ h |
| $C$ | external light cycle | analytical-deterministic | $[-1, 1]$ | 24 h |
| $V_h$ | vitality reserve | constant (Phase 1) | $\mathbb{R}$ | — |
| $V_n$ | chronic load | constant (Phase 1) | $\mathbb{R}$ | — |

The external light cycle is **frozen at a morning-type baseline** ($\phi_0 = -\pi/3$): healthy wake peak occurs at approximately 10am solar time. This is a clinical premise; subjects are not allowed to have "evening-type chronotypes" as a healthy variant. Misalignment from the morning baseline is parameterised separately (see $V_c$ in §3) and treated as pathology.

### 2.2 The parameter list — 31 drift/obs parameters + 4 initial conditions

The total is **35 estimable scalars**. The table groups them by block:

#### Block F — Fast subsystem (17 slots)

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `kappa` | $\kappa$ | 6.67 | $\tilde Z \to u_W$ inhibition strength |
| `lmbda` | $\lambda$ | 32.0 | circadian drive amplitude in $u_W$ |
| `gamma_3` | $\gamma_3$ | 8.0 | $W \to u_Z$ inhibition strength |
| `tau_W` | $\tau_W$ | 2.0 h | wakefulness response time |
| `tau_Z` | $\tau_Z$ | 2.0 h | sleep-depth response time |
| `V_c` | $V_c$ | 0 h | **phase-shift (hours)** — see §3 |
| `HR_base` | $\mathrm{HR}_\mathrm{base}$ | 50 bpm | HR intercept |
| `alpha_HR` | $\alpha_\mathrm{HR}$ | 25 bpm | HR gain on $W$ |
| `sigma_HR` | $\sigma_\mathrm{HR}$ | 8 bpm | HR observation noise |
| `c_tilde` | $\tilde c$ | 3.0 | sleep-detection threshold on $\tilde Z$ |
| `tau_a` | $\tau_a$ | 3.0 h | adenosine timescale |
| `beta_Z` | $\beta_Z$ | 2.5 | adenosine $\to u_Z$ gain |
| `Vh` | $V_h$ | 1.0 | vitality reserve (constant state) |
| `Vn` | $V_n$ | 0.3 | chronic load (constant state) |
| `T_W`, `T_Z`, `T_a` | — | $10^{-2}$ | diffusion temperatures (fast states) |

$V_h$ and $V_n$ are constant states, but their single "value per subject" is estimated as if it were a parameter — so they count in the parameter total.

#### Block T — Stuart-Landau testosterone (6 parameters)

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `mu_0` | $\mu_0$ | $-0.5$ | baseline bifurcation parameter ($< 0$) |
| `mu_E` | $\mu_E$ | $1.0$ | entrainment-coupling strength ($> 0$, $\mu_0 + \mu_E > 0$) |
| `eta` | $\eta$ | $0.5$ | Landau cubic saturation |
| `tau_T` | $\tau_T$ | $48$ h | slow HPG timescale |
| `alpha_T` | $\alpha_T$ | $0.3$ | $T \to u_W$ loading coefficient |
| `T_T` | — | $10^{-4}$ | diffusion temperature (T state) |

#### Block S — Ordinal sleep channel (1 parameter)

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `delta_c` | $\Delta_c$ | $1.5$ | gap between light and deep thresholds on $\tilde Z$ |

#### Block P — Steps Poisson channel (3 parameters)

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `lambda_base` | $\lambda_b$ | $0.5$ /h | background step rate (sensor noise) |
| `lambda_step` | $\lambda_s$ | $200$ /h | peak step rate during wake |
| `W_thresh` | $W_\ast$ | $0.6$ | wakefulness activation threshold |

#### Block R — Stress channel (4 parameters)

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `s_base` | $s_0$ | $30$ | baseline stress score |
| `alpha_s` | $\alpha_s$ | $40$ | wake modulation of stress |
| `beta_s` | $\beta_s$ | $10$ | $V_n$ coupling into stress |
| `sigma_s` | $\sigma_s$ | $15$ | stress observation noise |

#### Initial-condition block (4 parameters)

$W(0)$, $\tilde Z(0)$, $a(0)$, $T(0)$ — standard dynamical ICs.

**Count audit:** 17 + 6 + 1 + 3 + 4 = 31 non-IC parameters, plus 4 ICs = **35 estimable scalars**.

---

## 3. The phase-shift parameter $V_c$

All healthy subjects are assumed to have a morning-type circadian phase. The external light cycle is fixed at

$$
C(t) = \sin\!\bigl(2\pi t / 24 + \phi_0\bigr), \qquad \phi_0 = -\pi/3 \quad \text{(peak at 10am solar time)}.
$$

The subject's **internal** circadian drive — the signal entering the wakefulness-SDE's sigmoid argument — is a time-shifted version:

$$
C_\mathrm{eff}(t) = \sin\!\bigl(2\pi (t - V_c) / 24 + \phi_0\bigr),
$$

where $V_c \in \mathbb{R}$ is the phase shift in hours. Prior: $V_c \sim \mathcal{N}(0, 3)$.

Interpretation:

| $V_c$ | Meaning |
|:---:|:---|
| $0$ h | Subject aligned with external light (healthy) |
| $+3$ h | Rhythm running 3 h late (delayed-sleep-phase, late evening type) |
| $+6$ h | Shift worker / chronic westbound jet lag |
| $-3$ h | Advanced sleep phase (very early riser) |

$V_c$ serves a dual role in the model:

1. **Estimable parameter** during diagnostic inference. From a subject's HR + sleep + activity data, the posterior over $V_c$ quantifies how phase-shifted they are relative to solar time.
2. **Controllable intervention target** in forward simulation. Given a diagnosed $V_c \neq 0$, one can forward-simulate the same subject (everything else fixed) with $V_c$ overridden to zero and check whether testosterone recovers — if yes, the indicated intervention is phase correction (light therapy, sleep scheduling, melatonin timing).

**Clinical premise replacing chronotype.** "Evening type" is not a normal variant; it is a phase-shift pathology with measurable $V_c > 0$ and measurable downstream consequences (collapsed entrainment, testosterone flatline). This is a falsifiable claim and is the clinical stance of the model.

---

## 4. The SDE system

The driving equations. Dropping $\alpha_T = 0$ from $u_W$ and removing the $T$ SDE recovers a pure sleep-wake-adenosine baseline (the "fast subsystem").

### 4.1 Fast subsystem (three stochastic states)

**Wakefulness:**

$$
dW = \frac{1}{\tau_W}\!\left[\sigma(u_W(t)) - W\right]dt + \sqrt{2\,T_W}\,dB_W,
$$

$$
u_W(t) = \lambda\,C_\mathrm{eff}(t) + V_h + V_n - a - \kappa\,\tilde Z + \alpha_T\,T.
$$

**Sleep depth:**

$$
d\tilde Z = \frac{1}{\tau_Z}\!\left[A\,\sigma(u_Z(t)) - \tilde Z\right]dt + \sqrt{2\,T_Z}\,dB_{\tilde Z},
$$

$$
u_Z(t) = -\gamma_3\,W - V_n + \beta_Z\,a.
$$

Here $A = 6$ is a frozen scale constant and $\sigma(u) = 1/(1 + e^{-u})$ is the logistic sigmoid.

**Adenosine:**

$$
da = \frac{1}{\tau_a}(W - a)\,dt + \sqrt{2\,T_a}\,dB_a.
$$

### 4.2 Slow subsystem — Stuart-Landau testosterone

$$
dT = \frac{1}{\tau_T}\!\left[\,\mu(E)\,T - \eta\,T^3\,\right]dt + \sqrt{2\,T_T}\,dB_T,
$$

$$
\mu(E) = \mu_0 + \mu_E\,E(t).
$$

The drift is the Stuart-Landau normal form for a supercritical pitchfork at $\mu = 0$: for $\mu < 0$ the only stable equilibrium is $T = 0$; for $\mu > 0$ the stable equilibrium is $T^\star = \sqrt{\mu/\eta}$.

$T \geq 0$ is preserved by the cubic saturation together with the non-negative initial condition. In the numerical IMEX scheme, a positivity clip is applied as a safety net for large noise realisations near $T = 0$.

### 4.3 Deterministic-state updates

$C(t) = \sin(2\pi t/24 + \phi_0)$ is analytic and reset at each integration step. $V_h$ and $V_n$ are held constant at their initial values throughout the trajectory (Phase 1 convention — these states are candidates for dynamical treatment in a future Phase 2).

### 4.4 Noise conventions

The four independent Wiener processes have temperatures $T_W, T_Z, T_a, T_T > 0$. Values are small (typically $\sim 10^{-2}$ for the fast states, $\sim 10^{-4}$ for $T$) so noise is a perturbation to the drift rather than dominating it. In the current implementation noise temperatures are either estimated or fixed at small values depending on the experimental context.

---

## 5. Entrainment quality — the dual formulation

The entrainment quality $E(t) \in [0, 1]$ measures two *clinically distinct* failure modes of the sleep-wake rhythm:

- **Amplitude failure**: the subject is stuck in one state (can't sleep, or can't wake). $W$ and $\tilde Z$ don't alternate cleanly.
- **Phase-shift failure**: the subject's rhythm is mis-aligned with external light. $W$ and $\tilde Z$ alternate with full amplitude but at the wrong time.

The model carries **two formulas** for $E$. Both are computed and displayed, but **only one drives the dynamics**.

### 5.1 $E_\mathrm{dyn}(t)$ — the dynamics-side formula

This is what drives $\mu(E) = \mu_0 + \mu_E E$ inside the $T$ SDE. It is cheap (instantaneous; no running statistics) and thus can be evaluated at every integration step:

$$
\mu_W^\mathrm{slow} = V_h + V_n - a + \alpha_T\,T, \qquad
\mu_Z^\mathrm{slow} = -V_n + \beta_Z\,a,
$$

$$
\mathrm{amp}_W = 4\sigma(\mu_W^\mathrm{slow})(1 - \sigma(\mu_W^\mathrm{slow})), \qquad
\mathrm{amp}_Z = 4\sigma(\mu_Z^\mathrm{slow})(1 - \sigma(\mu_Z^\mathrm{slow})),
$$

$$
\mathrm{phase}(V_c) = \max\!\bigl(\cos(2\pi V_c/24),\; 0\bigr),
$$

$$
\boxed{\;E_\mathrm{dyn}(t) = \mathrm{amp}_W \cdot \mathrm{amp}_Z \cdot \mathrm{phase}(V_c)\;}
$$

Each of $\mathrm{amp}_W, \mathrm{amp}_Z \in [0, 1]$ is maximised at $\sigma = 1/2$ (balanced plateaus, clean alternation) and vanishes when the sigmoid saturates (patient stuck in one state). The phase factor is $1$ at $V_c = 0$, falls to $0$ at $|V_c| = 6$ h, and is clipped to $0$ beyond. It depends on $V_c$ only — no time dependence, no daily ripple.

### 5.2 $E_\mathrm{obs}(t)$ — the windowed diagnostic formula

This is the honest clinical measurement: what a clinician would compute from a week of HR + sleep data. For each time $t$, using a 24-hour sliding window of $W(\cdot)$ and $\tilde Z(\cdot)$:

$$
\mathrm{amp}_W^\mathrm{obs} = \frac{W_\max - W_\min}{1}, \qquad \mathrm{amp}_Z^\mathrm{obs} = \frac{\tilde Z_\max - \tilde Z_\min}{A},
$$

$$
\mathrm{phase}_W^\mathrm{obs} = \max\bigl(\mathrm{corr}(W, C),\; 0\bigr), \qquad \mathrm{phase}_Z^\mathrm{obs} = \max\bigl(\mathrm{corr}(\tilde Z, -C),\; 0\bigr)
$$

(the correlation is against the **external** light cycle $C$, not the shifted $C_\mathrm{eff}$ — this is what makes $E_\mathrm{obs}$ able to detect phase-shift pathology),

$$
E_\mathrm{obs}(t) = \mathrm{amp}_W^\mathrm{obs}\cdot\mathrm{phase}_W^\mathrm{obs} \cdot \mathrm{amp}_Z^\mathrm{obs}\cdot\mathrm{phase}_Z^\mathrm{obs}.
$$

### 5.3 Why two formulas

$E_\mathrm{dyn}$ is a cheap proxy suitable for evaluation inside the SDE drift at every integration step. $E_\mathrm{obs}$ is a genuine rhythm measurement requiring 24 h of history — too expensive to compute at every drift evaluation. In the diagnostic plot both are overlaid so the user sees:

- What the *model* thinks is driving testosterone ($E_\mathrm{dyn}$)
- What the *data* actually reveals about rhythm ($E_\mathrm{obs}$)

Sustained disagreement between the two over many days is a signal that the model is mis-calibrated. Agreement validates the cheap proxy. Both collapse to zero in pathological scenarios (shallow sleep, phase-shift), and both sit well above zero in healthy scenarios.

**Future extension** (not in this version): move the windowed statistics into the dynamics by carrying 24-h running means/variances as auxiliary deterministic states. This would unify $E_\mathrm{dyn}$ and $E_\mathrm{obs}$ at the cost of adding ~6 deterministic states.

### 5.4 Bifurcation threshold

The critical entrainment for pulsatility is

$$
E_\mathrm{crit} = -\mu_0/\mu_E.
$$

With $\mu_0 = -0.5$, $\mu_E = 1.0$ this gives $E_\mathrm{crit} = 0.5$. For $E_\mathrm{dyn} > 0.5$, $\mu > 0$ and $T$ settles at $T^\star = \sqrt{\mu/\eta}$. For $E_\mathrm{dyn} < 0.5$, $\mu < 0$ and $T$ collapses to $0$.

### 5.5 Why $E_\mathrm{dyn}$ discriminates when cycle-averaged formulas do not

An earlier version of this model used the "band-condition" formula $E = \sigma(\kappa_E(\lambda^2 - \mu_W^2))\cdot\sigma(\kappa_E(\lambda^2 - \mu_Z^2))$ with $\kappa_E = 2$ and $\lambda = 32$. In the actual physiological parameter regime $\lambda^2 = 1024 \gg \mu_W^2, \mu_Z^2 \sim 1$, both sigmoids saturate at 1 and $E \approx 1$ regardless of parameter choice — so the formula cannot distinguish healthy from pathological. The amplitude × phase formula replaces this saturating construction with a non-saturating one: amplitude factors peak at $\mu^\mathrm{slow} = 0$ and monotonically decay as $|\mu^\mathrm{slow}|$ grows, tracking the actual sleep-wake balance rather than comparing unbounded quadratic forms.

---

## 6. Observation model — four channels

### 6.1 Heart rate (Gaussian)

Sampled at the integration grid or subsampled:

$$
y^\mathrm{HR}_k = \mathrm{HR}_\mathrm{base} + \alpha_\mathrm{HR}\,W(t_k) + \varepsilon^\mathrm{HR}_k, \qquad \varepsilon^\mathrm{HR}_k \sim \mathcal{N}(0, \sigma_\mathrm{HR}^2).
$$

Typical ranges: $50 + 25 \cdot W$, giving $W = 0.1 \to 52$ bpm, $W = 0.9 \to 72$ bpm, plus noise $\sigma_\mathrm{HR} \approx 8$ bpm.

### 6.2 Sleep stage (ordinal, 3-level)

The sleep state is one of three ordered categories $\{0 = \text{wake}, 1 = \text{light+rem}, 2 = \text{deep}\}$, with probabilities determined by $\tilde Z$ through two thresholds $\tilde c_1 < \tilde c_2$ (parameterised as $\tilde c_1 = \tilde c$, $\tilde c_2 = \tilde c + \Delta_c$ to enforce ordering automatically via the positivity prior on $\Delta_c$):

$$
\begin{aligned}
P(\mathrm{wake}) &= 1 - \sigma(\tilde Z - \tilde c_1), \\
P(\mathrm{light}+\mathrm{rem}) &= \sigma(\tilde Z - \tilde c_1) - \sigma(\tilde Z - \tilde c_2), \\
P(\mathrm{deep}) &= \sigma(\tilde Z - \tilde c_2).
\end{aligned}
$$

Typical thresholds $\tilde c_1 = 3.0$, $\Delta_c = 1.5$ (so $\tilde c_2 = 4.5$). With $\tilde Z_\max \approx 5$ in healthy sleep, the deep stage (top ~30% of $\tilde Z$ time) occupies ~20–25% of total sleep, matching typical sleep architecture.

**Rationale for ordinal rather than 4-stage.** REM is a distinct neural state — not a depth of sleep. Separating it properly would require a second latent state (a REM propensity), doubling stochastic state count. Phase 1 bundles REM into the light stage; a 4-state extension is reserved for a future phase.

### 6.3 Steps (Poisson, 15-min bins)

Step counts are accumulated into 15-minute bins, with rate depending on wakefulness through a sigmoid threshold:

$$
r(W) = \lambda_b + \lambda_s \cdot \sigma\bigl(10 \cdot (W - W_\ast)\bigr),
$$

$$
k_\mathrm{bin} \sim \mathrm{Poisson}\bigl(r(W_\mathrm{bin}) \cdot \Delta t\bigr), \qquad \Delta t = 0.25 \text{ h}.
$$

Here $W_\mathrm{bin}$ is the mean wakefulness over the bin. The sharpness coefficient $10$ is frozen (not estimated). The parameters are $(\lambda_b, \lambda_s, W_\ast)$:

- $\lambda_b \approx 0.5$ /h handles background / sensor noise during sleep (a few spurious counts per hour)
- $\lambda_s \approx 200$ /h is the peak *rate* during alert wake (≈50 steps per 15-min bin)
- $W_\ast \approx 0.6$ is the wakefulness threshold at which activity switches on

The Poisson model is physically correct: step events are discrete arrivals with memoryless inter-arrival times at each given rate. The 15-minute bin aggregation is computationally convenient and matches typical wearable-data export granularity.

**Identifiability role.** Steps pin down the wake plateau of $W$ very tightly — far more tightly than HR. Dozens to hundreds of events per bin give low-variance estimates of $r(W)$, and hence of $W$ itself. This indirectly constrains $V_h + V_n$ (which together determine the wake value of $u_W$) and the sleep-wake transition parameters $(\tau_W, \kappa, \lambda)$.

### 6.4 Garmin stress score (Gaussian)

The stress score is a 0–100 scalar. It is modelled as a linear combination of wakefulness (phasic, daily) and chronic load (tonic, subject-specific):

$$
y^\mathrm{stress}_k = s_0 + \alpha_s\,W(t_k) + \beta_s\,V_n + \varepsilon^\mathrm{stress}_k, \qquad \varepsilon^\mathrm{stress}_k \sim \mathcal{N}(0, \sigma_s^2).
$$

The output is clipped to $[0, 100]$ at simulation time only (not during inference — that would break Gaussianity of the likelihood).

**Identifiability role.** In Phase 1, $V_n$ is constant. So the stress channel's *time variation* comes solely from $W$ — but that's what HR tells us too. The stress channel's unique contribution is through the **additive offset** $\beta_s V_n$: two subjects with identical $W$-dynamics but different $V_n$ have the same HR traces but different stress baselines. This is the cross-channel constraint that disambiguates $V_n$ from $V_h$ (both feed $u_W$ together, but only $V_n$ feeds the stress channel separately). This is the main identifiability payoff of adding this channel.

**Caveat.** The Garmin stress score is a firmware-derived quantity with proprietary smoothing. The linear-in-$W$-and-$V_n$ model is a physiologically-motivated working approximation; the residuals on real data should be examined for systematic misfit before accepting the fit. If the model is mis-specified for a particular subject's Garmin firmware, removing this channel from the likelihood reduces the estimate back to the three-channel model with no other changes.

### 6.5 Channel timing

All channels are sampled on the integration grid except steps (binned to 15-min). The 5-minute integration resolution handles the $\tau_W = 2$ h timescale with ~24 samples per e-fold, comfortably resolving the flip-flop transitions.

---

## 7. Parameter sets for simulation testing

Four canonical scenarios for end-to-end model validation. All start from $T_0 = 0.5$ (physically plausible — near the healthy equilibrium $T^\star \approx 0.55$) except Set C, which starts from the pathological flatline $T_0 = 0.05$ to test recovery.

| Set | Scenario | $V_h$ | $V_n$ | $V_c$ | $T_0$ | $T$ end (day 14) | Expected behaviour |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| A | Healthy | 1.0 | 0.3 | 0 | 0.5 | 0.56 | Stable near $T^\star$ |
| B | Severe insomnia | 0.2 | **3.5** | 0 | 0.5 | **0.12** | Amplitude collapse |
| C | Recovery from flatline | 1.0 | 0.3 | 0 | **0.05** | 0.42 | Rising toward $T^\star$ |
| D | Shift worker / jet lag | 1.0 | 0.3 | **6.0** | 0.5 | **0.11** | Phase-shift collapse |

The symmetry between Sets B/D (healthy → pathology) and Set C (pathology → healthy) is the key validation: same bifurcation model, opposite trajectory directions. Sets B and D test two independent failure modes (amplitude and phase). Set D's collapse cannot be produced by any choice of $(V_h, V_n)$ alone — it requires a phase-shift mechanism — which is why $V_c$ is a necessary parameter in the model.

Per-set diagnostic checks (expected E values, μ values, trajectory shape, HR swings, sleep fractions) are in the accompanying test specification file in the same package.

### 7.1 Decoupling limits

Setting $\alpha_T = 0$ decouples $T$ from the wakefulness dynamics; the fast subsystem becomes a pure sleep-wake-adenosine model with no testosterone feedback. Further setting $\mu_E = 0$ removes the entrainment-dependence of $T$'s bifurcation parameter; $T$ relaxes to $T^\star = 0$ (if $\mu_0 < 0$) or $T^\star = \sqrt{\mu_0/\eta}$ (if $\mu_0 > 0$), independent of sleep-side dynamics. These are the decoupling limits used in the identifiability analysis (companion document §8).

Dropping any of the three newer observation channels (ordinal sleep → binary sleep; no steps; no stress) reduces the likelihood but does not change the state dynamics; the remaining model is still a valid SDE with a narrower observation window.

---

## 8. What this document does not cover

- **Identifiability proofs.** The Fisher-rank argument for the 31-parameter model is in the companion document `SWAT_Identifiability_Extension.md`.
- **Estimation methodology.** The SMC² inference pipeline (particle filters nested inside an SMC sampler over parameters) is a separate concern and is not specified here. The model is set up to be compatible with any likelihood-free or likelihood-based SDE inference method; all observation channels above are given in closed form so that particle-filter likelihood evaluation is straightforward.
- **Real-data calibration.** Has not been done at the time of writing. The four-scenario synthetic test is the current validation standard.

---

*End of specification.*

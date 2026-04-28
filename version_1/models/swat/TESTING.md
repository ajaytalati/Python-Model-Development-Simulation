# Testing the SWAT Sleep-Wake-Adenosine-Testosterone SDE Model

**Version:** 1.0.0
**Date:** 2026-04-20
**Model:** `models/swat/`
**Spec basis:** `Spec_24_Parameter_Sleep_Wake_Adenosine_Testosterone_Model.md`,
                 `Identifiability_and_Lyapunov_Proof_24_Parameter_Model.md`

**Self-contained.** This document contains the complete mathematical specification of the SWAT model. A tester verifying the Python implementation against the specification should need nothing else.

> **v1.0.0 — initial release.** §8 (calibration results) is left as a stub
> for the first complete test run.  All quantitative thresholds in §5 are
> **predictions from the spec** — they may need revision after the first
> calibration run, exactly as happened with the 20p model (see its TESTING.md
> §8 for the precedent).

---

## Table of contents

1. Purpose of this document
2. Mathematical specification of the model
3. Parameter definitions (plain language)
4. Parameter sets A, B, and C
5. Tests to run
6. Troubleshooting
7. Exit criteria
8. Calibration results (to be filled in after first run)

---

## 1. Purpose

Verify that the Python implementation in `models/swat/` matches the mathematical specification given in §2. All tests run against `simulator/run_simulator.py`. The goal is to confirm, before any estimator work begins on the new T state, that:

- The model imports cleanly into both the simulator and estimator frameworks.
- Parameter set A (healthy basin) produces a stable testosterone equilibrium $T^\star = \sqrt{\mu(E)/\eta} \approx 1.0$ alongside the inherited sleep-wake flip-flop.
- Parameter set B (pathological basin) collapses entrainment $E \to 0$, drives $\mu(E) < 0$, and produces a testosterone flatline $T \to 0$ over $\sim 4\tau_T$.
- Parameter set C (recovery scenario) starts with $T_0 \approx 0.05$ in the healthy V-basin and recovers to $T^\star \approx 1.0$ over $\sim 4\tau_T$.
- The deterministic skeleton cross-validates between scipy and Diffrax solvers for the new 7-state vector — i.e. the numpy and JAX implementations agree, including the new T equation and entrainment computation.

The 17-parameter sleep-wake-adenosine substructure is inherited from `sleep_wake_20p` unchanged. Tests 0–6 of the 20p test suite still apply with `swat` substituted for `sleep_wake_20p`. Tests 7–9 are new and target the SWAT-specific dynamics.

---

## 2. Mathematical specification

### 2.1 State variables

The model has 7 state components. Four are stochastic; one is deterministic-analytical; two are constants in Phase 1.

| Symbol | Name | Domain | Timescale | Role |
|:---:|:---|:---:|:---:|:---|
| $W(t)$ | Wakefulness | $[0, 1]$ | $\tau_W \sim 2$ h | Activity of wake-promoting neurons |
| $\tilde Z(t)$ | Sleep depth (rescaled) | $[0, A]$ | $\tau_Z \sim 2$ h | Activity of sleep-promoting neurons |
| $a(t)$ | Adenosine (rescaled) | $\mathbb{R}_{\geq 0}$ | $\tau_a \sim 3$ h | Homeostatic sleep pressure |
| $T(t)$ | **Testosterone pulsatility amplitude** | $\mathbb{R}_{\geq 0}$ | $\tau_T \sim 48$ h | **Stuart-Landau HPG amplitude** |
| $C(t)$ | External light cycle | $[-1, 1]$ | 24 h, deterministic | $C(t) = \sin(2\pi t / 24 + \phi_0)$ with frozen morning-type $\phi_0$ |
| $V_h$ | Healthy potential | $\mathbb{R}$ | constant (Phase 1) | Vitality / fitness reserve |
| $V_n$ | Nuisance potential | $\mathbb{R}$ | constant (Phase 1) | Chronic stress / inflammation load |

The fixed constant $A := 6$ sets the rescaling of $\tilde Z$. It is **not** a parameter — it is a scale convention inherited from an earlier reparameterisation and is hard-coded in both `simulation.py` and `_dynamics.py` as `A_SCALE = 6.0`.

The fixed constant $\phi_0 := -\pi/3$ (`PHI_MORNING_TYPE`) is the frozen morning-type circadian baseline. The old chronotype parameter $\phi$ is **removed** from this model — the clinical premise is that every healthy rhythm should peak in the morning (wake peak at ~10am solar time with $\phi_0 = -\pi/3$). Misalignment from this baseline is pathological, and is captured by the new parameter:

| Symbol | Role | Status |
|:---|:---|:---|
| $\phi_0 = -\pi/3$ | Frozen morning-type phase; external light cycle $C(t) = \sin(2\pi t/24 + \phi_0)$ | Hard-coded constant |
| $V_c$ | Phase shift (hours) of the subject's internal drive from the external cycle | Estimable parameter **and** controllable intervention target |

The subject's internal circadian drive entering $u_W$ is the *shifted* signal:

$$
C_{\text{eff}}(t) = \sin\!\Bigl(\tfrac{2\pi (t - V_c)}{24} + \phi_0\Bigr).
$$

Healthy: $V_c \approx 0$ (internal drive = external light). Pathological: $|V_c| > 0$ (shift workers, chronic jet lag, delayed/advanced sleep phase). The state $C(t)$ stored in the trajectory tracks the *external* reference (no $V_c$); the shift only enters the dynamics through $u_W$.

### 2.2 The four coupled latent SDEs

**Wakefulness** (modified — adds $+\alpha_T T$ to $u_W$ and uses the subject's *shifted* circadian drive $C_{\text{eff}}$):

$$
dW = \frac{1}{\tau_W}\Bigl[\,\sigma\!\bigl(u_W(t)\bigr) - W\,\Bigr]\,dt + \sqrt{2\,T_W}\,dB_W
$$

with sigmoid argument

$$
u_W(t) = \lambda\,C_{\text{eff}}(t) + V_n - a(t) - \kappa\,\tilde Z(t) + \alpha_T\,T(t).
$$

Note that $V_h$ does **not** appear in $u_W$. It enters the model through the entrainment-amplitude factor in $E_{\text{dyn}}$ (see §2.3) — anabolic on $T$ via amplitude rather than via the wake-promoting drive. This is the V_h-anabolic structural fix; the prior version of the spec had $V_h$ as a $+V_h$ term inside $u_W$ which made $V_h$ structurally catabolic on $T$.

**Sleep depth** (unchanged from 20p):

$$
d\tilde Z = \frac{1}{\tau_Z}\Bigl[\,A\,\sigma\!\bigl(u_Z(t)\bigr) - \tilde Z\,\Bigr]\,dt + \sqrt{2\,T_Z}\,dB_Z
$$

with

$$
u_Z(t) = -\gamma_3\,W(t) - V_n + \beta_Z\,a(t).
$$

**Adenosine** (unchanged from 20p):

$$
da = \frac{1}{\tau_a}\bigl(W(t) - a(t)\bigr)\,dt + \sqrt{2\,T_a}\,dB_a.
$$

**Testosterone amplitude** (NEW — Stuart-Landau normal form):

$$
dT = \frac{1}{\tau_T}\Bigl[\,\mu(E)\,T - \eta\,T^3\,\Bigr]\,dt + \sqrt{2\,T_T}\,dB_T
$$

with bifurcation parameter

$$
\mu(E) = \mu_0 + \mu_E\,E.
$$

Here $\sigma(u) = 1/(1 + e^{-u})$ and $B_W, B_Z, B_a, B_T$ are independent standard Brownian motions. Positivity $T \geq 0$ is enforced by a reflecting-boundary clip in both IMEX steps and by the cubic dissipativity at large $T$.

### 2.3 Entrainment quality — dual formulation

The entrainment quality $E(t) \in [0, 1]$ measures whether the patient's sleep/wake rhythm is **both deep enough** (clean alternation) **and properly phase-locked** to the external light/dark cycle. Three different failure modes exist:

- **Amplitude failure** — patient stuck in one state (can't sleep, or can't wake). Measured against $V_h$, which gates the entrainment amplitude (low $V_h$ ⇒ no entrainment).
- **Chronic-load damping** — sustained high $V_n$ smothers entrainment even with healthy $V_h$, modelled by a multiplicative damper.
- **Phase-shift failure** — patient's rhythm mis-aligned with external light (shift work, jet lag). Measured against $V_c$, with a clamped pathology threshold at $|V_c| \geq V_{c,\max}$ hours.

**Two different formulas** for $E$ are needed — one cheap enough to run inside the SDE dynamics at every step, another that reflects the honest clinical measurement over a window of data. They are labelled $E_{\text{dyn}}$ and $E_{\text{obs}}$ and both appear in `entrainment.png` panel 1.

#### $E_{\text{dyn}}(t)$ — the dynamics-side formula

This is what actually drives $\mu(E_{\text{dyn}}) = \mu_0 + \mu_E E_{\text{dyn}}$ inside the testosterone SDE. It is computed *instantaneously* from the current state $y(t)$ and the parameter $V_c$ — no windowing, no running statistics.

The entrainment amplitude is $V_h$-driven:

$$
A_W = \lambda_{\text{amp},W}\,V_h, \qquad A_Z = \lambda_{\text{amp},Z}\,V_h
$$

$$
B_W = V_n - a + \alpha_T T, \qquad B_Z = -V_n + \beta_Z\,a
$$

$$
\text{amp}_W = \sigma(B_W + A_W) - \sigma(B_W - A_W)
$$

$$
\text{amp}_Z = \sigma(B_Z + A_Z) - \sigma(B_Z - A_Z)
$$

The chronic-load damper:

$$
\text{damp}(V_n) = e^{-V_n / V_{n,\text{scale}}}
$$

The phase factor (clamped at $V_{c,\max} = 3$ hours):

$$
\text{phase}(V_c) = \cos\!\left(\frac{\pi}{2}\,\frac{\min(|V_c|,\,V_{c,\max})}{V_{c,\max}}\right)
$$

Final:

$$
E_{\text{dyn}}(t) = \text{damp}(V_n)\,\cdot\,\text{amp}_W\,\cdot\,\text{amp}_Z\,\cdot\,\text{phase}(V_c)
$$

The phase factor depends on **$V_c$ only**, not on $t$ — a subject with $V_c = 6$ h has $\text{phase} = 0$ *always*. This avoids a spurious daily ripple in $\mu(E)$ that would confuse the slow $T$ dynamics.

| $V_c$ (h) | phase factor | Interpretation |
|:---:|:---:|:---|
| 0 | 1.00 | Aligned with external light |
| ±1 | 0.87 | Mild misalignment |
| ±2 | 0.50 | Marked misalignment (delayed/advanced sleep phase) |
| ±3 | 0.00 | Clinical pathology threshold |
| ±6 or more | 0.00 | Shift worker / inverted (clamped) |

**Why the amplitude formula is shaped like this.** The clamped quarter-period form $\sigma(B+A) - \sigma(B-A)$ measures how much sigmoid response is available within an amplitude band of width $2A$ centred on the slow backdrop $B$. When $A$ (and therefore $V_h$) is zero, the band has zero width and amplitude is zero — depleted vitality means no rhythm, the clinically correct interpretation. When $A$ is large enough to span the sigmoid's transition region around $B$, the response is near $1$. The structural intuition: $V_h$ is the **gain** on the entrainment loop, not an additive offset to the wake drive.

#### $E_{\text{obs}}(t)$ — the windowed diagnostic formula

This is the honest clinical measurement — it's what a clinician would compute from a week of HR + sleep data. For each time $t$, it uses a 24-hour sliding window of $W(\cdot)$ and $\tilde Z(\cdot)$:

$$
\text{amp}_W^{\text{obs}}(t) = \frac{W_{\max} - W_{\min}}{1}, \qquad \text{amp}_Z^{\text{obs}}(t) = \frac{\tilde Z_{\max} - \tilde Z_{\min}}{A}
$$

$$
\text{phase}_W^{\text{obs}}(t) = \max\bigl(\mathrm{corr}(W, C_{\text{ext}}),\; 0\bigr), \qquad \text{phase}_Z^{\text{obs}}(t) = \max\bigl(\mathrm{corr}(\tilde Z, -C_{\text{ext}}),\; 0\bigr)
$$

where $C_{\text{ext}}(t) = \sin(2\pi t / 24 + \phi_0)$ is the **external** light cycle (no $V_c$ shift — the reference is the objective sun). The correlation is computed over the preceding 24 hours.

$$
E_{\text{obs}}(t) = \text{amp}_W^{\text{obs}} \cdot \text{phase}_W^{\text{obs}} \cdot \text{amp}_Z^{\text{obs}} \cdot \text{phase}_Z^{\text{obs}}
$$

This formula is a genuine rhythm measurement but needs 24 h of history. It's too expensive (running statistics) to put inside the SDE drift at every step.

#### Why they differ intentionally

$E_{\text{dyn}}$ is a cheap *proxy* for the windowed measurement. Its amplitude factor uses the slow-backdrop sigmoid balance (a point-in-time surrogate for "is the flip-flop likely to be swinging cleanly?"), which is generally higher than the actual observed amplitude because the sigmoid factor already assumes some fraction of time at each plateau.

For a healthy subject (Set A) we see $E_{\text{dyn}} \approx 1.0$ (clamped quarter-period saturates at $V_h \geq 1$) and $E_{\text{obs}} \approx 0.2$ — different scales, but both indicate "the system is working". For any pathological case (B, D), both go to near zero.

Both curves are plotted on `entrainment.png` panel 1 (solid purple = $E_{\text{dyn}}$, dashed orange = $E_{\text{obs}}$). The clinician inspects:
- $E_{\text{dyn}}$: what the model thinks is driving testosterone
- $E_{\text{obs}}$: what the observational data actually reveals about rhythm

Disagreement between them over many days is a signal that the model is mis-calibrated; agreement validates the dynamics-side proxy.

**Future work** (not in this model version). A principled approach would be to carry the 24-h running statistics as auxiliary deterministic states (running mean, variance, covariance with $C$) so that $E_{\text{obs}}$ *becomes* $E_{\text{dyn}}$ — one formula. That adds ~6 deterministic states and is deferred.

**Departure from the spec.** The spec document writes $E = \sigma(\kappa_E(\lambda^2 - \mu^2))$, a point-in-time function of instantaneous sigmoid arguments. With our parameter regime ($\lambda = 32$) that formula saturates at 1 and cannot discriminate basins. Both formulations above replace it.

**The proof's Lyapunov argument.** Lemma 4.4 (sign of $\partial E_{\text{dyn}} / \partial T$) is now evaluated for the $V_h$-anabolic clamped quarter-period form. The only $T$-dependent term is $B_W = V_n - a + \alpha_T T$, so

$$
\frac{\partial \text{amp}_W}{\partial T}
= \alpha_T\,\Bigl[\sigma'(B_W + A_W) - \sigma'(B_W - A_W)\Bigr]
= \alpha_T\,\Bigl[\sigma_+(1-\sigma_+) - \sigma_-(1-\sigma_-)\Bigr],
$$

where $\sigma_\pm = \sigma(B_W \pm A_W)$. At the healthy equilibrium with $V_h \approx 1$, $A_W = \lambda_{\text{amp},W} \cdot 1 = 5$, $B_W \approx 1$, the derivative is small and signed by whether $|B_W + A_W| < |B_W - A_W|$ (it isn't here, since $B_W > 0$), so the sign is mildly *negative*: raising $T$ slightly *reduces* $E_{\text{dyn}}$, which is mildly stabilising — same qualitative conclusion as the previous spec, just via a different amplitude formula. The Lyapunov bound $\dot{\mathcal{L}} \leq 0$ holds because the Stuart-Landau cubic dissipation dominates the weak $\alpha_T$ feedback.

### 2.4 Deterministic components

**External light cycle** (no noise, analytical reset after each step; objective reference):

$$
C(t) = \sin\!\left(\frac{2\pi t}{24} + \phi_0\right), \qquad \phi_0 = -\pi/3 \text{ (frozen, peak at ~10am solar time)}.
$$

The subject's internal circadian drive $C_{\text{eff}}(t)$ — the signal actually entering $u_W$ — is the external cycle shifted by $V_c$ hours (see §2.1).

**Behavioural potentials** (constants in Phase 1; spec departs from Phase-2 OU treatment):

$$
dV_h = 0, \qquad dV_n = 0.
$$

### 2.5 Observation channels

Same as 20p — **no new observation equations**. $T$ is a latent state; its identification flows via $T \to u_W \to W \to \mathrm{HR}$.

**Heart rate** (continuous, Gaussian):

$$
y_k^{\mathrm{HR}} = \mathrm{HR}_{\!\mathrm{base}} + \alpha_{\mathrm{HR}}\,W(t_k) + \varepsilon_k, \qquad \varepsilon_k \sim \mathcal{N}(0,\,\sigma_{\mathrm{HR}}^{\,2}).
$$

**Binary sleep** (Bernoulli):

$$
\Pr\bigl(S_k = 1 \mid \tilde Z(t_k)\bigr) = \sigma\!\bigl(\tilde Z(t_k) - \tilde c\bigr).
$$

### 2.6 Numerical scheme

Same Euler-Maruyama / IMEX framework as 20p. For the new $T$-SDE, the IMEX split is:

$$
f^T_{\mathrm{forcing}} = \frac{\mu(E)\,T}{\tau_T}, \qquad f^T_{\mathrm{decay}} = \frac{\eta\,T^2}{\tau_T}.
$$

This places the cubic saturation in the implicit decay (always $\geq 0$, stable) and the signed linear term in the explicit forcing. The decay term $\eta T^2 / \tau_T$ vanishes at $T = 0$, so the IMEX step is well-behaved at the flatline equilibrium.

### 2.7 Lyapunov structure (verification target for §5 Test 8)

Theorem 4.1 of the proof document gives the explicit Lyapunov function for the $T$-dynamics with $E$ held constant:

$$
\mathcal{L}(T) = \tfrac{1}{2}(T - T^\star(E))^2, \qquad T^\star(E) = \sqrt{\max(\mu(E), 0)/\eta},
$$

with derivative

$$
\dot{\mathcal{L}}(T) = -\tau_T^{-1}\,\eta\,T\,(T + T^\star)\,(T - T^\star)^2 \;\leq\; 0.
$$

For numerical verification we expect $\mathcal{L}(T(t))$ to be approximately monotonically decreasing along deterministic trajectories starting in the healthy basin (modulo small overshoot from circadian coupling through $E$).

---

## 3. Parameter definitions

Code-level accounting: **`PARAM_PRIOR_CONFIG` has 26 entries** (17 inherited + 6 new T-block + 3 new entrainment-amplitude block from the V_h-anabolic structural fix) and **`INIT_STATE_PRIOR_CONFIG` has 4 entries** (3 inherited + $T_0$). Total estimable scalars = 30.

Note: the code-level `PARAM_PRIOR_CONFIG` also contains the observation-channel parameters for the 3-level sleep, steps, and stress channels (delta_c, lambda_base, lambda_step, W_thresh, s_base, alpha_s, beta_s, sigma_s = 8 more entries). The count of 26 above refers to the latent-dynamics block; the runtime config has 34 in total.

The spec's "24-parameter" count (per `Spec_24_Parameter_...md` §6) excludes the three inherited fast-noise temperatures $T_W, T_Z, T_a$, which the spec freezes in its identifiability analysis but the code estimates. Per spec §7.1: "if $T_W, T_Z, T_a$ are to be estimated, the count rises to 27."

### 3.1 Inherited 17-parameter block (with chronotype → phase-shift replacement)

Inherited from the 20p model except that slot #6 (originally the chronotype $\phi$) is now the phase-shift parameter $V_c$. The count is preserved.

| # | Symbol | Code name | Change from 20p |
|:---:|:---:|:---:|:---|
| 1 | $\kappa$ | `kappa` | — |
| 2 | $\lambda$ | `lmbda` | — |
| 3 | $\gamma_3$ | `gamma_3` | — |
| 4 | $\tau_W$ | `tau_W` | — |
| 5 | $\tau_Z$ | `tau_Z` | — |
| 6 | $V_c$ | `V_c` | **Replaces $\phi$.** Phase shift in hours; estimable + controllable |
| 7 | $\mathrm{HR}_{\!\mathrm{base}}$ | `HR_base` | — |
| 8 | $\alpha_{\mathrm{HR}}$ | `alpha_HR` | — |
| 9 | $\sigma_{\mathrm{HR}}$ | `sigma_HR` | — |
| 10 | $\tilde c$ | `c_tilde` | — |
| 11 | $\tau_a$ | `tau_a` | — |
| 12 | $\beta_Z$ | `beta_Z` | — |
| 13 | $V_h$ | `Vh` | — |
| 14 | $V_n$ | `Vn` | — |
| 15 | $T_W$ | `T_W` | — |
| 16 | $T_Z$ | `T_Z` | — |
| 17 | $T_a$ | `T_a` | — |

### 3.2 New Stuart-Landau testosterone block (5)

| # | Symbol | Code name | Meaning |
|:---:|:---:|:---:|:---|
| 18 | $\mu_0$ | `mu_0` | Baseline bifurcation parameter at $E=0$. Expected sign: **negative** (no pulsatility without entrainment). |
| 19 | $\mu_E$ | `mu_E` | Entrainment-coupling coefficient. Expected sign: **positive**, with $\mu_0 + \mu_E > 0$ so that healthy entrainment yields positive $\mu$. |
| 20 | $\eta$ | `eta` | Landau cubic saturation coefficient. Sets healthy equilibrium $T^\star = \sqrt{\mu/\eta}$. |
| 21 | $\tau_T$ | `tau_T` | Slow timescale of $T$ dynamics (hours). $\sim 48$h — intermediate between flip-flop (2h) and 7-day vitality. |
| 22 | $T_T$ | `T_T` | Process-noise temperature for $T$. |

### 3.3 New coupling and entrainment-amplitude parameters (4)

| # | Symbol | Code name | Meaning |
|:---:|:---:|:---:|:---|
| 23 | $\alpha_T$ | `alpha_T` | Strength with which $T$ promotes wakefulness via the $+\alpha_T T$ term in $u_W$. Expected positive. |
| 24 | $\lambda_{\text{amp},W}$ | `lambda_amp_W` | $V_h$ gain into the W-side entrainment amplitude: $A_W = \lambda_{\text{amp},W}\,V_h$. Expected positive. Default 5.0 — gives saturated $\text{amp}_W \approx 1$ across the daily $B_W$ cycle at healthy $V_h = 1$. |
| 25 | $\lambda_{\text{amp},Z}$ | `lambda_amp_Z` | $V_h$ gain into the Z-side entrainment amplitude: $A_Z = \lambda_{\text{amp},Z}\,V_h$. Expected positive. Default 8.0 — larger because $\beta_Z\,a$ can reach ~4, so $A_Z$ must dominate. |
| 26 | $V_{n,\text{scale}}$ | `V_n_scale` | Damper scale for the chronic-load multiplicative factor: $\text{damp}(V_n) = e^{-V_n / V_{n,\text{scale}}}$. Default 2.0 — gives $\text{damp}(0) = 1$, $\text{damp}(2) = 0.37$, $\text{damp}(5) = 0.08$. |

### 3.4 Initial conditions (4)

| # | Symbol | Code name | Meaning |
|:---:|:---:|:---:|:---|
| 27 | $W_0$ | `W_0` | Wakefulness at $t = 0$ (inherited). |
| 28 | $\tilde Z_0$ | `Zt_0` | Rescaled sleep depth at $t = 0$ (inherited). |
| 29 | $a_0$ | `a_0` | Adenosine at $t = 0$ (inherited). |
| 30 | $T_0$ | `T_0` | Testosterone amplitude at $t = 0$. New. |

### 3.5 Frozen quantities (not estimated)

| Symbol | Value | Rationale |
|:---:|:---:|:---|
| $A$ (`A_SCALE`) | 6.0 | Rescaling of $\tilde Z$; inherited |
| $\phi_0$ (`PHI_MORNING_TYPE`) | $-\pi/3$ | Morning-type baseline circadian phase. Replaces the estimable chronotype $\phi$ from the 20p model. Clinical premise: all healthy rhythms peak in the morning; shifts from this baseline are pathological and measured by $V_c$. |
| Wake-side $a$ coefficient | $-1$ | Gauge-fixing inherited from 17-param identifiability proof |
| $V_{c,\max}$ (`V_C_MAX_HOURS`) | 3.0 | Clinical pathology threshold for the phase shift. Any $|V_c| \geq V_{c,\max}$ collapses the phase factor in $E_{\text{dyn}}$ to zero. Not estimable from clinical data — chosen by the clinical interpretation that even moderate phase shifts (3+ hour shift workers, transmeridian travellers) are pathological. |

---

## 4. Parameter sets for testing

> **Note on the V_h-anabolic structural fix.** Sections 4.1 and 4.5 below
> have been updated to reflect the corrected entrainment formula
> (V_h-anabolic clamped quarter-period + V_n damper + clamped phase).
> The Set-B and Set-D walkthroughs in 4.2 and 4.4 still narrate the
> prior slow-backdrop math; the corrected formula gives slightly
> different intermediate values but the same headline T-end outcomes
> (B: 0.108 vs old 0.12; D: 0.107 vs old 0.11) — the math walkthroughs
> in 4.2/4.4 are flagged for a follow-up edit. The set parameter
> tables (V_h, V_n, V_c, T_0) and the qualitative conclusions are
> unchanged.

### 4.1 Set A — healthy basin

Same as 20p Set A for the inherited 17 parameters, with the new T-block parameters set to their prior medians from spec §10.

| Parameter | Value | Parameter | Value |
|:---:|:---:|:---:|:---:|
| $\kappa$ | 6.67 | $\mathrm{HR}_{\!\mathrm{base}}$ | 50.0 |
| $\lambda$ | 32.0 | $\alpha_{\mathrm{HR}}$ | 25.0 |
| $\gamma_3$ | 8.0 *(†)* | $\sigma_{\mathrm{HR}}$ | 8.0 |
| $\tau_W$ | 2.0 | $\tilde c$ | 3.0 *(‡)* |
| $\tau_Z$ | 2.0 | $\tau_a$ | 3.0 |
| $V_c$ | 0.0 | $\beta_Z$ | 2.5 *(‡)* |
| $V_h$ | 1.0 | $V_n$ | 0.3 |
| $T_W$ | 0.01 | $T_Z$ | 0.05 |
| $T_a$ | 0.01 | | |
| **$\mu_0$** | **−0.5** | **$\mu_E$** | **1.0** |
| **$\eta$** | **0.5** | **$\tau_T$** | **48.0** |
| **$\alpha_T$** | **0.3** | **$T_T$** | **0.01** |
| $W_0$ | 0.5 | $\tilde Z_0$ | 3.5 *(‡)* |
| $a_0$ | 0.5 | **$T_0$** | **0.5** |

*(†)* `gamma_3` reduced from 60 to 8 — see `BUG_REPORT_gamma3_sleep_depth.md` (inherited fix).
*(‡)* `c_tilde`, `beta_Z`, `Zt_0` raised — see `BUG_REPORT_ctilde_beta_z_sleep_false_positives.md` (inherited fix).

Simulation length: **14 days** (longer than 20p's 7d so that $\tau_T = 48$h is observed at least 7 times — required by (R3') of the identifiability proof). Grid: $dt = 5$ minutes.

**Expected entrainment quality and equilibrium (Set A).** With healthy $V_h = 1.0$, $V_n = 0.3$, typical $a \approx 0.5$, $T \approx 0.9$, $V_c = 0$, and the deployed $\lambda_{\text{amp},W} = 5$, $\lambda_{\text{amp},Z} = 8$, $V_{n,\text{scale}} = 2$:

$$
A_W = 5 \cdot 1.0 = 5, \qquad A_Z = 8 \cdot 1.0 = 8
$$

$$
B_W = 0.3 - 0.5 + 0.3 \cdot 0.9 = 0.07, \qquad B_Z = -0.3 + 2.5 \cdot 0.5 = 0.95
$$

The clamped quarter-period sigmoid differences are essentially saturated:

$$
\text{amp}_W = \sigma(5.07) - \sigma(-4.93) \approx 0.994 - 0.007 = 0.987
$$
$$
\text{amp}_Z = \sigma(8.95) - \sigma(-7.05) \approx 1.000 - 0.001 = 0.999
$$

The chronic-load damper:

$$
\text{damp}(0.3) = e^{-0.3/2} \approx 0.861
$$

and $\text{phase}(0) = 1$, so

$$
E_{\text{dyn}} \approx 0.987 \cdot 0.999 \cdot 0.861 \cdot 1.0 \approx 0.85
$$

The SDE-averaged $E_{\text{dyn}}$ stays close to this because the slow backdrop $B_W$ stays small enough that the amplitude factor remains saturated.

Then $\mu(E_{\text{dyn}}) \approx -0.5 + 1.0 \cdot 0.85 = 0.35$, giving the testosterone equilibrium

$$
T^\star = \sqrt{\mu(E_{\text{dyn}})/\eta} \approx \sqrt{0.35/0.5} \approx 0.84
$$

The actual last-day mean comes out ~0.90 (slightly above $T^\star$ because $B_W$ varies and momentarily pushes $\text{amp}_W$ above the steady-state value during fast-flip transitions).

**Bifurcation threshold.** $E_{\mathrm{crit}} = -\mu_0 / \mu_E = 0.5$. Set A's realised $E_{\text{dyn}} \approx 0.85$ sits well above this — healthy equilibrium. Set B's realised $E_{\text{dyn}} \approx 0.005$ sits at the floor — full collapse. Set D's realised $E_{\text{dyn}} = 0$ sits at the floor too — full phase-collapse.

### 4.2 Set B — pathological basin

Same as Set A except:

| Parameter | Set A | Set B |
|:---:|:---:|:---:|
| $V_h$ | 1.0 | 0.2 |
| $V_n$ | 0.3 | **3.5** (raised from 2.0 — strong insomnia) |
| $T_0$ | 0.5 | 0.5 (start near healthy equilibrium, observe collapse) |

Severe hyperarousal-insomnia configuration. With $V_n = 3.5$, both slow-backdrop sigmoids saturate:

- $\mu_W^{\text{slow}} = 0.2 + 3.5 - a + \alpha_T T \approx 3.5$ → $\sigma \approx 0.97$ → $\text{amp}_W \approx 0.12$
- $\mu_Z^{\text{slow}} = -3.5 + \beta_Z a \approx -2.5$ → $\sigma \approx 0.08$ → $\text{amp}_Z \approx 0.29$
- phase factor $= 1$ (since $V_c = 0$)

so $E_{\text{dyn}} \approx 0.035$ — **well below** the bifurcation threshold $E_{\mathrm{crit}} = 0.5$.

Then $\mu(E_{\text{dyn}}) = -0.5 + 1.0 \cdot 0.035 \approx -0.465$ — strongly negative, forcing $T$ to the flatline $T^\star = 0$:

- $T(0) = 0.5$ at start (near healthy $T^\star$);
- effective decay rate $|\mu| / \tau_T \approx 0.465 / 48 \approx 0.0097$ per hour;
- e-folding time $\approx 103$ hours $\approx 4.3$ days;
- by day 14: $T \approx 0.12$ (numerical run gave 0.12 at last-day mean);
- by day 7: $T \approx 0.22$.

This gives a **clean Stuart-Landau collapse mirroring Set C's recovery in reverse** — Set B goes healthy → pathology (T: 0.5 → 0.12), Set C goes pathology → healthy (T: 0.05 → 0.42). Both starting conditions are physically plausible.

For the inherited 17-param substructure, the differences from Set A are stronger than with the old $V_n = 2.0$:

| Check | Set A | Set B | Mechanism |
|:---:|:---:|:---:|:---|
| $\tilde Z$ mean | ~1.3 | **~0.28** | $V_n = 3.5$ drives $u_Z$ very negative |
| $\tilde Z$ max | > 3.6 | < 2.0 | Very shallow sleep signal |
| Sleep fraction | 0.30–0.45 | near 0 | Patient cannot reach $\tilde c = 3$ |
| HR daily swing | ~25 bpm | ~25 bpm | W flip-flop persists ($\lambda = 32$ dominates) |
| Basin label in plot title | "healthy" | "hyperarousal-insomnia" | |
| **Mean $E_{\text{dyn}}$** | **~0.55** | **~0.035** | **Amplitude × phase strongly discriminates** |
| **Mean $E_{\text{obs}}$** | **~0.20** | **~0.025** | **Windowed confirms amplitude failure** |
| **Mean $\mu(E_{\text{dyn}})$** | **~+0.05** | **~−0.465** | **Deep in flatline basin** |
| **Final $T$ at day 14** | **~0.58** | **~0.13** | **Clean collapse to flatline** |

### 4.3 Set C — recovery scenario

Identical to Set A except $T_0 = 0.05$. Tests whether $T$ will rise from near-zero toward $T^\star$ when the entrainment is healthy — the supercritical-pitchfork rise.

**Expected and observed:** $T(t)$ rises from $0.05$ to $\sim 0.42$ by day 14 (verified numerically). The rise is slower than a pure deterministic calculation suggests because $T^\star$ itself varies with the fluctuating $E_{\text{dyn}}$ (it dips during W/Z balance transitions) — the actual $T$ tracks the running average of $T^\star$.

### 4.4 Set D — phase-shift pathology (shift worker / chronic jet lag)

Identical to Set A for every parameter except $V_c = 6.0$ (the subject's rhythm is 6 hours delayed relative to external light). Healthy potentials ($V_h = 1.0$, $V_n = 0.3$), healthy initial $T_0 = 0.5$ (near T*).

**This is the fourth failure mode** — one that the potentials $V_h, V_n$ alone cannot produce. Both $W$ and $\tilde Z$ swing with full amplitude (because the potentials are healthy and $\lambda$ still dominates), but their peak timing is shifted 6 hours away from the external light cycle. The phase-correlation term of the entrainment measure picks this up:

**This is the fourth failure mode** — one that the potentials $V_h, V_n$ alone cannot produce. Both $W$ and $\tilde Z$ swing with full amplitude (because the potentials are healthy), but the subject's rhythm is shifted 6 hours away from the external light. The dynamics-side entrainment catches this via the `phase(V_c)` factor:

- $\text{amp}_W, \text{amp}_Z$: **unchanged from Set A** (full swing) — $V_h, V_n$ healthy
- $\text{phase}(V_c = 6) = \max(\cos(2\pi \cdot 6 / 24), 0) = \max(0, 0) = 0$
- $E_{\text{dyn}} = \text{amp}_W \cdot \text{amp}_Z \cdot 0 = 0$

so $\mu(E_{\text{dyn}}) = -0.5$ — the strongest possible collapse drive this model permits (equal to $\mu_0$). T decays with e-folding time $\tau_T / |\mu| = 48 / 0.5 = 96$ hours = 4 days. Numerical run gives:

- $T$ at day 14 ≈ **0.11** (last-day mean)
- $T$ at day 7 ≈ 0.30
- $E_{\text{dyn}}$ flatline at 0.000 from day 1 onward
- $E_{\text{obs}}$ also flatline at 0.000 (windowed phase correlation against external light picks up the 6h mismatch cleanly)
- Plot title shows "phase-shifted (V_c=+6.0h)" — `_basin_label` prioritises V_c phase-shift over V_h/V_n potentials

This demonstrates:
- **Sets B and D both run healthy → pathology** (T: 1.0 → ~0.12 in 14 days), **mirroring Set C's pathology → healthy recovery** (T: 0.05 → 0.42). The model is a proper bifurcation model supporting bidirectional trajectories.
- The model can distinguish four clinically distinct modes: healthy (A), amplitude collapse via severe V_n (B), recovery from flatline (C), phase-shift collapse via V_c (D).
- The same code handles diagnostic inference (estimate $V_c$ from the subject's data) and intervention modelling (forward-simulate with $V_c = 0$ to check whether phase correction alone restores testosterone).

### 4.5 Summary — the four-scenario picture

Numerical values from running the four canonical scenarios for 14 days under the corrected dynamics (`PYTHONPATH=. python models/swat/verify_swat_state.py` from `version_1/`, seed=42):

| Set | Scenario | $V_h$ | $V_n$ | $V_c$ | $T_0$ | $T$ end | $E_{\text{dyn}}$ mean | $\mu$ mean |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A | Healthy | 1.0 | 0.3 | 0 | **0.5** | **0.90** | ~0.85 | +0.35 |
| B | Severe insomnia | 0.2 | **3.5** | 0 | **0.5** | **0.11** | ~0.005 | −0.50 |
| C | Recovery from flatline | 1.0 | 0.3 | 0 | **0.05** | **0.82** | ~0.85 | +0.35 |
| D | Shift worker | 1.0 | 0.3 | **6.0** | **0.5** | **0.11** | 0.00 | −0.50 |

All non-recovery scenarios start at $T_0 = 0.5$ — well below the healthy equilibrium $T^\star \approx 0.9$ but physically plausible. The healthy run is what shows the rise from $T_0 = 0.5$ to the $T^\star$ neighbourhood. Set C demonstrates recovery from a deeper starting pathology ($T_0 = 0.05$).

The discrimination ratios are sharp under the corrected model: A:B ≈ 8.3×, A:D ≈ 8.4×. The symmetry between B/D (healthy → pathology) and C (pathology → healthy) is the key validation: same model, opposite trajectory signs.

---

## 5. Tests

All commands run from the framework root (one level above `simulator/`).

### Test 0 — Import smoke

```bash
python -c "from models.swat.simulation import SWAT_MODEL; from models.swat.estimation import SWAT_ESTIMATION as E; print('sim:', SWAT_MODEL.name, 'v' + SWAT_MODEL.version); print('est:', E.name, 'n_dim =', E.n_dim, '(', E.n_params, 'params +', E.n_init_states, 'init )'); print('n_states =', E.n_states, ', n_stochastic =', E.n_stochastic)"
```

**Expected output:**

```
sim: swat v1.0
est: swat n_dim = 30 ( 26 params + 4 init )
n_states = 7 , n_stochastic = 4
```

(The code-level `PARAM_PRIOR_CONFIG` includes 8 additional observation-channel parameters not counted in the latent-dynamics block of 26; `n_dim` reported here is the latent-dynamics + initial-conditions count. Regenerate this expected output if the count drifts.)

If this fails, fix the import before any later test. Most likely root causes: missing `__init__.py`; `models.swat.sim_plots` not importable; one of the new keys missing from `PARAM_PRIOR_CONFIG`.

### Test 1 — Parameter set A (healthy basin)

```bash
python simulator/run_simulator.py --model models.swat.simulation.SWAT_MODEL --param-set A --seed 42
```

Add `--scipy` if Diffrax unavailable. Output goes to `outputs/synthetic_swat_A_<YYYYMMDD_HHMMSS>/`.

**Expected output directory contents:**

| File | Description |
|:---|:---|
| `latent_states.png` | 6 panels: $W$, $\tilde Z$, $a$, **$T$**, $C$, and $V_h, V_n$ |
| `observations.png` | 2 panels: HR trace and binary sleep strip |
| **`entrainment.png`** | **3 panels: $E(t)$, $\mu(E)$, $T$ vs expected $T^\star$** |
| `synthetic_truth.npz` | Full trajectory + true params + true init states |
| `channel_hr.npz` | Arrays: `t_idx`, `hr_value` |
| `channel_sleep.npz` | Arrays: `t_idx`, `sleep_label` |

**Qualitative checks (inspect `latent_states.png` and `entrainment.png`):**

| Check | Expected |
|:---|:---|
| $W$ trajectory | Alternates between ~0.9 (wake) and ~0.1 (sleep), switching twice per 24h, with ~30-min transitions |
| $\tilde Z$ trajectory | Anti-correlated with $W$; low (~0) when awake, peak ~2.2–3.5 when asleep |
| $a$ trajectory | Low-passed $W$; rises during wake, decays during sleep with timescale ~3h |
| **$T$ trajectory** | **Rises from $T_0 = 0.5$ to ~0.90 over the first ~5 days, then stays near $T^\star \approx 0.9$** |
| $C$ trajectory | Clean 24-h sinusoid of unit amplitude |
| $V_h, V_n$ panel | Horizontal lines at 1.0 and 0.3; plot title contains **"healthy"** |
| **`entrainment.png` panel 1** | **$E_{\text{dyn}}$ (solid purple) around 0.85 from day 1 onward (well above $E_{\mathrm{crit}} = 0.5$); $E_{\text{obs}}$ (dashed orange) around 0.20** |
| **`entrainment.png` panel 2** | **$\mu(E_{\text{dyn}}) \approx +0.35$, predominantly green-shaded (above zero)** |
| **`entrainment.png` panel 3** | **Red trajectory $T(t)$ rises toward dashed green $T^\star \approx 0.84$ over the first 5 days, then tracks it** |

**Quantitative checks:**

| Check | Expected | Tolerance | Notes |
|:---:|:---:|:---:|:---|
| Mean HR (asleep) | ~52.5 bpm | ±5 bpm | $\mathrm{HR}_\mathrm{base} + \alpha_\mathrm{HR} \cdot W_\mathrm{sleep}$ + small $\alpha_T T$ effect |
| Mean HR (awake) | ~75 bpm | ±5 bpm | Slightly higher than 20p's 72.5 because $+\alpha_T T$ adds to $u_W$ |
| Sleep fraction | 0.30–0.45 | — | Within clinical target band; depends on $c_\mathrm{tilde}$ |
| **Mean $T$ (after day 5)** | **~0.90** | ±0.10 | **Settles near $T^\star = \sqrt{\mu/\eta}$ for $\mu \approx 0.35$** |
| **Std of $T$** | **< 0.15** | — | **$T_T = 10^{-4}$ keeps noise small on slow timescale** |
| **Mean $E_{\text{dyn}}$** | **~0.85** | ±0.10 | **Dynamics-side, day ≥ 1; clamped quarter-period saturates** |
| **Mean $E_{\text{obs}}$** | **~0.20** | ±0.10 | **Windowed diagnostic, day ≥ 1** |
| `verify_physics_fn` → `W_range` | > 0.7 | — | |
| `verify_physics_fn` → `Zt_range` | > 3.5 | — | Higher than 20p's 2.0 because of `c_tilde`/`beta_Z` fix |
| `verify_physics_fn` → **`T_range`** | **< 1.0** | — | **$T$ should not swing wildly in the healthy basin** |
| `verify_physics_fn` → **`T_nonneg`** | **`True`** | — | **Reflecting boundary working** |
| `verify_physics_fn` → `all_finite` | `True` | — | |

### Test 2 — Parameter set B (amplitude collapse — severe insomnia)

```bash
python simulator/run_simulator.py --model models.swat.simulation.SWAT_MODEL --param-set B --seed 42
```

**Expected differences from Set A:**

| Check | Set A | Set B | Mechanism |
|:---:|:---:|:---:|:---|
| $\tilde Z$ mean | ~1.3 | ~0.28 | $V_n = 3.5$ drives $u_Z$ very negative |
| Sleep fraction | 0.30–0.45 | near 0 | $\tilde Z_{\max}$ < $\tilde c = 3$ throughout |
| Basin label in plot title | "healthy" | "hyperarousal-insomnia" | `_basin_label` uses V_h/V_n |
| **Mean $E_{\text{dyn}}$ (day ≥ 1)** | **~0.55** | **~0.035** | **Amplitude factor collapses (both slow sigmoids saturated)** |
| **Mean $E_{\text{obs}}$ (day ≥ 1)** | **~0.20** | **~0.025** | **Windowed confirms amplitude failure** |
| **Mean $\mu(E_{\text{dyn}})$** | **~+0.05** | **~−0.46** | **Deep in flatline basin** |
| **$T$ at day 7** | **~0.65** | **~0.35** (decaying) | **Mid-collapse** |
| **Final $T$ at day 14** | **~0.58** | **~0.13** | **Clean Stuart-Landau collapse** |
| **`entrainment.png` panel 1** | Both E curves above 0.1 | Both at ≈0.03, well below $E_\mathrm{crit} = 0.5$ | Visible separation |
| **`entrainment.png` panel 2 ($\mu$)** | Green-shaded ($\mu > 0$) mostly | Red-shaded ($\mu < 0$), floor near −0.5 | Bifurcation regime swap |
| **`entrainment.png` panel 3 (T vs T*)** | T settles near 0.58 | T decays from 0.5 toward 0.12 | Trajectory split |

**This is the amplitude-failure discrimination test.** If $E_{\text{dyn}}$ does not differ between A and B, the slow-backdrop amplitude formula has a bug. If $E$ differs but $T$ does not, the bifurcation map $\mu(E) = \mu_0 + \mu_E E$ is not engaging — check that $\mu_E > |\mu_0|$ so the threshold $E_\mathrm{crit}$ is in $(0, 1)$.

**For the inherited 17-param tests** (HR swing, sleep label, transitions, etc.) Set B differences track the 20p baseline but with the stronger $V_n = 3.5$ pushing sleep fraction to ~0.

### Test 2b — Parameter set D (phase-shift pathology / shift worker)

```bash
python simulator/run_simulator.py --model models.swat.simulation.SWAT_MODEL --param-set D --seed 42
```

**This is the cleanest demonstration of phase-shift pathology.** Set D has healthy potentials ($V_h = 1.0$, $V_n = 0.3$) and healthy $T_0 = 0.5$ — everything that distinguishes it from Set A is the single parameter $V_c = 6.0$ (subject's rhythm 6 hours behind external light).

**Expected behaviour:**

| Check | Set A | Set D | Mechanism |
|:---:|:---:|:---:|:---|
| $W$ amplitude | full swing 0.05→0.95 | **same** (full swing) | Potentials healthy; $\lambda$ dominates |
| $\tilde Z$ amplitude | full swing 0.1→3.6 | **same** (full swing) | Same potentials as Set A |
| HR daily swing | ~25 bpm | ~25 bpm | No change (W amplitude preserved) |
| $W$ peak timing | ~10 am solar | **~4 pm solar** | $V_c = 6$h shifts peak later |
| $\tilde Z$ peak timing | ~10 pm solar | **~4 am solar** | Same shift inherited through $-\gamma_3 W$ |
| Basin label (plot title) | "healthy" | **"phase-shifted (V_c=+6.0h)"** | `_basin_label` prioritises $|V_c| \geq 2$h over potentials |
| **Mean $E_{\text{dyn}}$ (day ≥ 1)** | **~0.55** | **0.000** | **phase($V_c$)=0 zeros out full amplitude** |
| **Mean $E_{\text{obs}}$ (day ≥ 1)** | **~0.20** | **0.000** | **Windowed correlation against external light also sees mismatch** |
| **Mean $\mu(E_{\text{dyn}})$** | ~+0.05 | **−0.50** | Floor value (= $\mu_0$) |
| **$T$ at day 7** | ~0.65 | **~0.30** (mid-collapse) | e-fold = 4 days |
| **$T$ at day 14** | ~0.58 | **~0.11** | Nearly flatlined |

**Why the `entrainment.png` is diagnostic here.** Panel 1 shows **both** $E$ curves collapsed to 0:
- $E_{\text{dyn}}$ (solid purple) = 0 because the phase factor $\max(\cos(2\pi V_c / 24), 0) = 0$ at $V_c = 6$h
- $E_{\text{obs}}$ (dashed orange) = 0 because the windowed correlation $\mathrm{corr}(W, C_\text{ext})$ over a 24-hour window at a 6-hour shift is near zero

Both curves agreeing at 0 is strong evidence that the dynamics are tracking a genuine phase-shift pathology, not an artifact of either formula.

**Why the basin label now reads correctly.** `_basin_label(V_h, V_n, V_c)` checks $|V_c| \geq 2.0$ h first; if triggered it returns `"phase-shifted (V_c=±X.Xh)"` before considering V_h/V_n. The 2h threshold excludes mild morning/evening preferences (which are normal variation) while flagging clinical phase pathology.

**Clinical interpretation.** A subject presenting with:
- Healthy HR amplitude (peaks 70–80 bpm, dips 45–55 bpm)
- Full-depth sleep signal
- Normal vitality / stress biomarkers
- But collapsed testosterone

...is exactly the shift-worker or chronic-jet-lag profile. The model's diagnosis is "$V_c \neq 0$"; the recommended intervention is phase correction (light therapy, sleep scheduling, melatonin timing), not HPG supplementation.

**Verification snippet:**

```python
import numpy as np
t = np.load('<dir_setD>/synthetic_truth.npz')
traj = t['true_trajectory']; t_grid = t['t_grid']
from models.swat.sim_plots import _compute_E, _compute_E_dynamics

one_day = int(24 / (t_grid[1] - t_grid[0]))
params = {'V_c': 6.0, 'alpha_T': 0.3, 'beta_Z': 2.5}  # minimum fields needed
E_dyn = _compute_E_dynamics(traj, params)
E_obs = _compute_E(traj, t_grid, params)
T_final = traj[-one_day:, 3].mean()

print(f"Mean E_dyn (day >= 1): {E_dyn[one_day:].mean():.3f}   (expected 0.000)")
print(f"Mean E_obs (day >= 1): {E_obs[one_day:].mean():.3f}   (expected ~0.000)")
print(f"Mean T (last day):     {T_final:.3f}   (expected ~0.11)")
```

### Test 3 — Parameter set C (recovery from low $T_0$)

```bash
python simulator/run_simulator.py --model models.swat.simulation.SWAT_MODEL --param-set C --seed 42
```

**Expected behaviour — this is the critical test of the new T dynamics:**

| Check | Expected | Notes |
|:---|:---:|:---|
| $T(t = 0)$ | 0.05 | Initial condition |
| $T(t = 4\text{ days})$ | $\geq 0.20$ | Mid-rise; numerical run gave ~0.22 |
| $T(t = 14\text{ days})$ | $\sim 0.42$ | Approaching Set A equilibrium ~0.58; not fully settled at 14 d |
| Monotonic rise | yes (modulo noise) | Lyapunov function $\mathcal{L}(T) = \tfrac{1}{2}(T - T^\star)^2$ approximately monotonically decreasing |
| Sleep-side dynamics | Same as Set A | $T_0 = 0.05$ contributes only $\alpha_T \cdot 0.05 = 0.015$ to $u_W$ — negligible at start |
| $E_{\text{dyn}}$ trajectory | Same as Set A (~0.55) from day 1 | T_0 doesn't affect entrainment directly |
| Final plot shows T catching up to T* | yes | Bottom panel: red curve rises toward green dashed line |

**This is the single most important test for the new model.** If $T$ does not rise from $T_0 = 0.05$ toward $\sim 0.42$ in Set C, the Stuart-Landau pitchfork is not engaging — most likely root causes:

- $\mu(E)$ not above zero at the realised entrainment (check `entrainment.png` panel 2 — must be predominantly green-shaded);
- IMEX split has wrong sign on the explicit forcing (check `_dynamics.imex_components`);
- $T$-positivity clip is too aggressive and is suppressing the rise (check `imex_step_*` reflecting boundary);
- $\eta$ too large — cubic saturates the rise prematurely (cross-check $\sqrt{\mu/\eta}$ matches the dashed green line).

### Test 4 — Cross-validation (scipy vs Diffrax)

```bash
python simulator/run_simulator.py --model models.swat.simulation.SWAT_MODEL --cross-validate
```

**What it does:** sets all four diffusion temperatures ($T_W, T_Z, T_a, T_T$) to zero, integrates the now-deterministic ODE with both solvers, compares trajectories.

**Expected:** max trajectory difference `< 1e-4` across all 7 state components, including the new $T$ component.

**GPU OOM caveat (inherited from 20p):** the 7-state × 4032-step Diffrax run (14d × 288 steps/day) may exceed 8 GB VRAM. If so, use the manual drift-agreement snippet:

```python
import sys, math, os
sys.path.insert(0, 'simulator')
os.environ['JAX_ENABLE_X64'] = 'True'
import numpy as np
import jax.numpy as jnp
from models.swat.simulation import PARAM_SET_A, drift as drift_np, drift_jax

params = PARAM_SET_A
p_jax  = {k: jnp.float64(v) for k, v in params.items() if not isinstance(v, str)}

# Sample states (W, Zt, a, T, C, Vh, Vn) at three different times
test_states = [
    (0.0,  [0.5,  3.5,  0.5,  1.0,  math.sin(-math.pi/3), 1.0, 0.3]),
    (12.0, [0.95, 0.10, 0.85, 1.0,  math.sin(2*math.pi*12/24 - math.pi/3), 1.0, 0.3]),
    (20.0, [0.05, 5.0,  0.3,  0.5,  math.sin(2*math.pi*20/24 - math.pi/3), 1.0, 0.3]),
    (48.0, [0.5,  3.5,  0.5,  0.05, math.sin(2*math.pi*48/24 - math.pi/3), 1.0, 0.3]),  # low T
]
max_diff = 0.0
for t, y in test_states:
    d_np = drift_np(t, np.array(y), params, None)
    d_jx = drift_jax(t, jnp.array(y), (p_jax,))
    max_diff = max(max_diff, float(jnp.max(jnp.abs(jnp.array(d_np) - d_jx))))
print(f'Max drift diff: {max_diff:.2e}  (expected < 1e-10)')
```

The fourth test state has $T = 0.05$ to exercise the Stuart-Landau term near the bifurcation transition where numerical sensitivity is highest.

### Test 5 — Physics verification only

```bash
python simulator/run_simulator.py --model models.swat.simulation.SWAT_MODEL --verify
```

Runs `verify_physics_fn` on a deterministic Set A trajectory. All booleans (including `T_nonneg`, `T_bounded`) must be `True`.

### Test 6 — Reproducibility (inherited from 20p)

Two runs of Set A with the same seed must produce byte-identical outputs:

```bash
python simulator/run_simulator.py --model models.swat.simulation.SWAT_MODEL --param-set A --seed 42  # dir_1/
python simulator/run_simulator.py --model models.swat.simulation.SWAT_MODEL --param-set A --seed 42  # dir_2/
python -c "
import numpy as np
d1 = np.load('<dir_1>/synthetic_truth.npz')
d2 = np.load('<dir_2>/synthetic_truth.npz')
print('trajectory equal:', np.array_equal(d1['true_trajectory'], d2['true_trajectory']))
"
```

Expected: `trajectory equal: True`.

### Test 7 — Observation equation consistency (inherited from 20p)

Same as 20p Test 6. Manually check `hr_value` matches $\mathrm{HR}_\mathrm{base} + \alpha_\mathrm{HR} \cdot W$ to within $\sigma_\mathrm{HR}$, and `sleep_label` follows $\sigma(\tilde Z - \tilde c)$. The new $\alpha_T T$ term enters $u_W$ but does not affect the observation equations directly — its effect on HR is via $W$.

### Test 8 — Stuart-Landau equilibrium check (NEW)

After running Set A, verify the testosterone equilibrium agrees with the analytical formula $T^\star = \sqrt{\mu(\langle E \rangle) / \eta}$:

```python
import numpy as np
from models.swat.sim_plots import _compute_E

t = np.load('<dir>/synthetic_truth.npz')
traj = t['true_trajectory']
t_grid = t['t_grid']
params = dict(zip(t['true_param_names'], t['true_params'].astype(float)))

# Drop the first day (window startup) and the last two days — keep middle window.
n = len(traj)
one_day = int(24 / ((t_grid[1] - t_grid[0])))
late = traj[one_day:int(n * 0.85)]
t_late = t_grid[one_day:int(n * 0.85)]

E = _compute_E(late, t_late, params)
mu_mean = params['mu_0'] + params['mu_E'] * E.mean()
T_star_pred = np.sqrt(max(mu_mean, 0.0) / params['eta']) if mu_mean > 0 else 0.0
T_obs = late[:, 3].mean()

print(f"Mean E (excluding day 0, up to 85%): {E.mean():.3f}")
print(f"mu(E): {mu_mean:.3f}")
print(f"T* predicted: {T_star_pred:.3f}")
print(f"T  observed:  {T_obs:.3f}")
print(f"Relative error: {abs(T_obs - T_star_pred) / max(T_star_pred, 1e-3):.2%}")
```

**Expected:** Relative error < 20% (slightly looser than the previous 15% because windowed $E$ has more variance than a point-in-time formula). Larger error suggests:
- $\tau_T = 48$h is too long for the 14-day horizon to reach equilibrium (extend to 21 days);
- Stuart-Landau drift implementation has a sign error;
- The plot-side $E$ differs from what the dynamics actually uses (the dynamics still use the old slow-backdrop formula at the time of writing — see §2.3 caveat).

### Test 9 — Lyapunov decrease check (NEW)

Along a deterministic Set C trajectory (recovery from $T_0 = 0.05$), verify the Lyapunov function $\mathcal{L}(T) = \tfrac{1}{2}(T - T^\star)^2$ decreases approximately monotonically:

```python
import numpy as np
from models.swat.sim_plots import _compute_E

t = np.load('<dir_setC>/synthetic_truth.npz')
traj = t['true_trajectory']
t_grid = t['t_grid']
params = dict(zip(t['true_param_names'], t['true_params'].astype(float)))

# Use the realised mean E (skipping day 0 window startup) to compute target T*
one_day = int(24 / (t_grid[1] - t_grid[0]))
E = _compute_E(traj, t_grid, params)
mu_mean = params['mu_0'] + params['mu_E'] * E[one_day:].mean()
T_star = np.sqrt(max(mu_mean, 0.0) / params['eta'])

T_obs = traj[:, 3]
L = 0.5 * (T_obs - T_star) ** 2

# Smooth L over 12-h window to remove daily-scale noise
window_size = 12 * 12  # 12 h * 12 steps/h
L_smooth = np.convolve(L, np.ones(window_size)/window_size, mode='valid')

# Count local violations of monotonicity (allowing for noise)
diffs = np.diff(L_smooth)
n_viol = int(np.sum(diffs > 0.02 * L.max()))
total = len(diffs)
print(f"T* = {T_star:.3f}, L_initial = {L[0]:.3f}, L_final = {L[-1]:.4f}")
print(f"Lyapunov violations (smoothed, threshold 2%): {n_viol}/{total} = {n_viol/total:.1%}")
```

**Expected:** Violations < 5% (monotonic decrease modulo noise). Persistent monotonic *increase* indicates the Lyapunov claim is false in the implementation — most likely root cause: drift sign error in the new $\mu T - \eta T^3$ term.

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|:---|:---|:---|
| `ImportError: cannot import name SWAT_MODEL` | Typo or missing `__init__.py` | Confirm all five `.py` files present in `models/swat/`; confirm `__init__.py` imports both top-level objects |
| All states NaN after step 1 | $\tau_W$, $\tau_Z$, or $\tau_T$ too small vs $dt$ | Check IMEX denominators $1 + \Delta t \cdot f_\mathrm{decay} > 0$ for all stochastic states |
| `n_dim != 27` in Test 0 | Prior-config length mismatch | Confirm `PARAM_PRIOR_CONFIG` has 23 entries and `INIT_STATE_PRIOR_CONFIG` has 4 |
| **$T$ trajectory stays at $T_0$ unchanged** | **$\mu(E) \approx 0$ at realised $E$** | **Check `entrainment.png` panel 2 — if $\mu \approx 0$, raise $\mu_E$ in `PARAM_SET_A`** |
| **$T$ explodes to $> 5$** | **Cubic decay too weak** | **Check `eta > 0`; verify `imex_components` has positive `dT_decay = eta * T^2 / tau_T`** |
| **$T$ collapses to 0 in Set A** | **$\mu(E) < 0$ unexpectedly** | **Check entrainment.png — if $E < E_\mathrm{crit} = 0.5$, the inherited 17-param values may be in a band-edge configuration; reduce $V_n$ or check `compute_sigmoid_args`** |
| **$T$ goes negative** | **Reflecting boundary not applied** | **Check that `imex_step_deterministic` and `imex_step_stochastic` both call `y_next.at[3].set(jnp.maximum(y_next[3], 0.0))`** |
| **$T$ does not rise in Set C** | **See Test 3 troubleshooting list** | **Check $\mu > 0$, IMEX sign, positivity clip, $\eta$ saturation point** |
| **$E$ identical in Set A and Set B** | **Sign error in `mu_W_dc` or `mu_Z_dc`, OR `entrainment_quality` was not updated to the duty-cycle-balance formula** | **Confirm `_dynamics.entrainment_quality` returns `4*sigma(mu)*(1-sigma(mu))` factors, NOT `sigmoid(KAPPA_E*(lmbda^2 - mu^2))`** |
| **$T$ collapses in Set A** | **Realised $E < 0.5$** (bifurcation threshold crossed wrongly) | **Check `entrainment.png` panel 1 — if E < 0.5 in healthy basin, mu_W_dc may have a sign error or alpha_T*T is too large** |
| $W$ panel shows a flat line | Sigmoid permanently saturated, or $\alpha_T T$ pushing $u_W$ off-scale | If $\alpha_T T$ during sleep exceeds ~3, lower `alpha_T` to 0.1 |
| **Mean HR awake $> 80$ bpm in Set A** | **$\alpha_T T$ adding more to $u_W$ than expected** | **At healthy $T \approx 1$, $\alpha_T T \approx 0.3$. If mean HR is much higher, $T$ has overshot — check $\eta$** |
| Cross-validate (Test 4) reports large diff | `drift_jax` and `drift` disagree | Most likely: the new $T$-equation has a sign error in one of the two; diff line-by-line |
| Cross-validate diff is exactly $\sim 0.3$ | $\alpha_T T$ added to $u_W$ in one drift but not the other | Confirm both `drift` and `drift_jax` add `+ alpha_T * T` to `u_W` |
| **`entrainment.png` shows oscillating $E$** | **Expected — instantaneous-DC approximation; check that the oscillation amplitude is small relative to the daily mean** | **No fix unless variance > 0.2** |
| **$\mathcal{L}$ increases monotonically (Test 9)** | **Stuart-Landau drift sign error** | **In `_dynamics.imex_components`, confirm `fT = mu_bifurc * T_amp / tau_T` (positive when $\mu, T > 0$)** |
| **Set D shows mean $E > 0.1$ (should be near 0)** | **$V_c$ sign/units bug** | **Confirm $V_c$ is interpreted in hours and appears as `t − V_c` (not `t + V_c`) in `sin(2π(t − V_c)/24 + φ₀)`. Also confirm `_compute_E` in `sim_plots.py` uses `PHI_MORNING_TYPE` (NOT `t − V_c`) for the reference C — the plot correlates against the **external** light, not the subject's drive.** |
| **Set A behaves differently from the pre-V_c version** | **`V_c = 0` not actually zero, or PHI_MORNING_TYPE ≠ old phi median** | **Confirm `PARAM_SET_A['V_c'] == 0.0` and `PHI_MORNING_TYPE == -math.pi/3` (the old `phi` prior median); at V_c=0 with this phi_0 the dynamics should match the pre-V_c code exactly** |

---

## 7. Exit criteria

The model is ready for downstream estimator work once all of:

- Test 0 (import smoke) passes; `n_dim = 27`
- Test 1 (Set A) matches qualitative and quantitative specifications, including:
  - $T \in [0.40, 0.85]$ throughout the trajectory after the initial transient (day 2+)
  - Last-day mean $T \approx 0.58$
  - `verify_physics_fn` → `T_nonneg`, `T_bounded` both `True`
  - `entrainment.png` shows mean $E_{\text{dyn}} \approx 0.55$ and mean $E_{\text{obs}} \approx 0.20$ (from day 1 onward), $\mu$ predominantly positive (green-shaded), and red $T$ curve tracking dashed green $T^\star \approx 0.4$–0.6
- **Test 2 (Set B, $V_n = 3.5$)** shows mean $E_{\text{dyn}} \approx 0.035$ (well below $E_{\text{crit}} = 0.5$), mean $\mu \approx -0.46$, and $T$ decaying from 0.5 toward ~0.12 by day 14 — the amplitude-failure discrimination test
- **Test 2b (Set D, $V_c = 6$h)** shows mean $E_{\text{dyn}} \approx 0.00$, mean $\mu \approx -0.50$, and $T$ decaying from 0.5 toward ~0.11 by day 14 — the phase-shift failure test; $V_h, V_n$ identical to Set A but $V_c$ drives collapse
- **Test 3 (Set C)** shows $T$ rising from 0.05 to $\geq 0.20$ by day 4, reaching $\sim 0.42$ by day 14
- **Symmetry validation**: Sets B and D both run healthy → pathology ($T$: 0.5 → ~0.11) while Set C runs pathology → healthy ($T$: 0.05 → 0.42). Same bifurcation model, opposite trajectory signs.
- Test 4 (cross-validate) shows max drift difference `< 1e-10` via the manual snippet
- Test 5 (physics) all checks `True`
- Test 6 (reproducibility) passes
- Test 7 (observation consistency) passes
- **Test 8 (Stuart-Landau equilibrium) shows relative error < 20%**
- **Test 9 (Lyapunov decrease) shows < 5% smoothed violations**

The Stuart-Landau-specific tests **2 (amplitude failure), 2b (phase-shift failure), 3 (recovery), 8 (equilibrium), 9 (Lyapunov)** are the new exit gate; the inherited tests 0, 1, 4–7 must continue to pass to ensure the 17-param substructure has not been broken by the additions.

---

## 8. Calibration results

*(to be filled in after the first complete test run, following the precedent of the 20p TESTING.md §8)*

Sub-sections to populate:
- 8.1 Test outcomes (table: test, result, notes)
- 8.2 Set A quantitative detail (predicted vs observed)
- 8.3 Set B detail
- **8.4 Set C — testosterone recovery trajectory** (rise time, final value, comparison to predicted Lyapunov rate)
- **8.5 Realised mean $E$ and bifurcation margin** (does $E$ sit comfortably above $E_\mathrm{crit}$?)
- 8.6 Root-cause analysis of any spec-vs-observation discrepancies
- 8.7 Recommendations (parameter re-centring, threshold revisions, additional parameter sets)

Until §8 is populated, all quantitative thresholds in §5 are spec predictions and may be revised after the first run, exactly as happened for the 20p model.

---

## Inherited bug fixes

Two bug fixes from the 20p model are inherited verbatim in `PARAM_SET_A`:

1. **`gamma_3 = 8.0`** (was 60). The ODE-calibrated value of 60 was too high for the SDE: with $T_W = 0.01$, the W noise floor during sleep ($\bar W \approx 0.033$) suppressed $u_Z$ below $-1.5$, giving $\tilde Z_\mathrm{eq} \approx 1.05$ — below $\tilde c = 1.5$, producing 50% noise in sleep labels. See `BUG_REPORT_gamma3_sleep_depth.md`.

2. **`c_tilde = 3.0`, `beta_Z = 2.5`, `Zt_0 = 3.5`** (were 1.5, 1.5, 1.8). With $\tilde Z$ clipped at 0, $c_\mathrm{tilde} = 1.5$ gave a structural 18% daytime false-sleep rate ($\sigma(-1.5) = 0.18$). Raising $c_\mathrm{tilde}$ to 3.0 dropped this to 5%; $\beta_Z$ raised in tandem so $\tilde Z_\mathrm{sleep}$ stays well above the new threshold. See `BUG_REPORT_ctilde_beta_z_sleep_false_positives.md`.

**No new bug fixes are required for the SWAT additions at the time of writing.** Two potential pitfalls to monitor in the first calibration run:

- The $+\alpha_T T$ term in $u_W$ adds $\sim 0.3$ during the healthy-equilibrium regime. If this pushes mean wake HR above 80 bpm or breaks the flip-flop transitions, lower `alpha_T` to 0.1 in `PARAM_SET_A` and update the prior median in `estimation.py` to match.
- The instantaneous-DC entrainment approximation in `entrainment_quality()` may produce unwanted 24-h fluctuations in $E$ that propagate into $T$ through $\mu(E)$. If `verify_physics` reports `T_range > 0.5` in Set A, add a 24-h running-mean state for $\mu_W^\mathrm{dc}, \mu_Z^\mathrm{dc}$ and recompute $E$ from the running averages.

---

*End of document.*

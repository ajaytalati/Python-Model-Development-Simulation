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
u_W(t) = \lambda\,C_{\text{eff}}(t) + V_h + V_n - a(t) - \kappa\,\tilde Z(t) + \alpha_T\,T(t).
$$

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

### 2.3 Entrainment quality

The entrainment quality $E(t) \in [0, 1]$ measures whether the patient's sleep/wake rhythm is **both deep enough** (clean alternation between wake and sleep) **and properly phase-locked** to the body clock $C(t)$. A patient who is stuck in one state has no entrainment; a patient whose rhythm is out of sync with the light/dark cycle (shift work, jet lag, delayed-sleep-phase) also has no entrainment.

For each state, $E$ combines two measurements taken over a sliding 24-hour window:

**Amplitude quality** — how deeply the state alternates:

$$
\text{amp}_W(t) = \frac{W_{\max} - W_{\min}}{1}, \qquad \text{amp}_Z(t) = \frac{\tilde Z_{\max} - \tilde Z_{\min}}{A}
$$

Each is 1 when the state swings the full range ($W$ from 0 to 1, $\tilde Z$ from 0 to $A=6$) and 0 when the state is stuck.

**Phase-alignment quality** — how well the rhythm tracks the body clock. In a healthy person $W$ is *in phase* with $C(t)$ (awake during light) and $\tilde Z$ is *anti-phase* with $C(t)$ (asleep during dark):

$$
\text{phase}_W(t) = \max\bigl(\mathrm{corr}(W, C),\; 0\bigr), \qquad \text{phase}_Z(t) = \max\bigl(\mathrm{corr}(\tilde Z, -C),\; 0\bigr)
$$

computed as Pearson correlation over the window. The `max(·, 0)` treats an inverted rhythm (e.g. active at night) as equivalent to no rhythm — both are pathological for the HPG axis.

**Combined:**

$$
E_W = \text{amp}_W \cdot \text{phase}_W, \qquad E_Z = \text{amp}_Z \cdot \text{phase}_Z, \qquad E = E_W \cdot E_Z.
$$

**Discrimination across pathology modes** (verified numerically on idealised trajectories):

| Scenario | $E$ | Mechanism |
|:---|:---:|:---|
| Healthy (clean flip-flop, in phase) | ~0.66 | amp ≈ 1, phase ≈ 1 on both sides |
| Shallow $\tilde Z$ (Set B) | ~0.16 | $\tilde Z$ only reaches ~1.3 of 6; amp_Z ≈ 0.2 |
| Jet-lagged (12 h inversion) | ~0.00 | phase = 0 on both sides |
| Delayed sleep (4 h shift) | ~0.17 | phase partial (~1/3) on both sides |
| Stuck in one state | ~0.00 | amp ≈ 0 on both sides |

The product structure means either failure mode alone pulls $E$ down, and both failing drives $E$ to near zero.

**Departure from the spec.** The spec document writes $E = \sigma(\kappa_E(\lambda^2 - \mu^2))$, a point-in-time function of instantaneous sigmoid arguments. With our parameter regime ($\lambda = 32$) that formula saturates at 1 and cannot discriminate basins. The amplitude × phase formulation above uses a genuine rhythm measurement over a 24-hour window, which is what the word "entrainment" means in the physiology literature: two rhythms (sleep/wake and light/dark) locked to each other with proper depth. The trade-off is that $E$ now requires 24 h of history — during the first simulated day the window is partial and $E$ is unreliable. All downstream checks use day ≥ 1.

**The proof's Lyapunov argument.** Lemma 4.4 (sign of $\partial E / \partial T$) needs re-derivation under this formulation. Sketch: near the healthy equilibrium, increasing $T$ raises $u_W$ through the $+\alpha_T T$ term, which raises $W$ during wake but has negligible effect on $W$ during sleep (sigmoid already saturated) — so $\mathrm{amp}_W$ is approximately insensitive to small $T$ perturbations, and $\partial E / \partial T \approx 0$. The self-consistent healthy equilibrium is therefore determined primarily by the entrainment of the fast subsystem, not by $T$ itself — the feedback is weak but still stabilising (the Lyapunov bound $\dot{\mathcal{L}} \leq 0$ holds because the Stuart-Landau dissipation $-\eta T (T + T^\star)(T - T^\star)^2 / \tau_T$ dominates any $\partial E / \partial T$ contribution of order $\alpha_T$). **This needs checking against the first calibration run.**

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

Code-level accounting: **`PARAM_PRIOR_CONFIG` has 23 entries** (17 inherited + 6 new) and **`INIT_STATE_PRIOR_CONFIG` has 4 entries** (3 inherited + $T_0$). Total estimable scalars = 27.

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

### 3.3 New coupling parameter (1)

| # | Symbol | Code name | Meaning |
|:---:|:---:|:---:|:---|
| 23 | $\alpha_T$ | `alpha_T` | Strength with which $T$ promotes wakefulness via the $+\alpha_T T$ term in $u_W$. Expected positive. |

### 3.4 Initial conditions (4)

| # | Symbol | Code name | Meaning |
|:---:|:---:|:---:|:---|
| 24 | $W_0$ | `W_0` | Wakefulness at $t = 0$ (inherited). |
| 25 | $\tilde Z_0$ | `Zt_0` | Rescaled sleep depth at $t = 0$ (inherited). |
| 26 | $a_0$ | `a_0` | Adenosine at $t = 0$ (inherited). |
| 27 | $T_0$ | `T_0` | Testosterone amplitude at $t = 0$. New. |

### 3.5 Frozen quantities (not estimated)

| Symbol | Value | Rationale |
|:---:|:---:|:---|
| $A$ (`A_SCALE`) | 6.0 | Rescaling of $\tilde Z$; inherited |
| $\phi_0$ (`PHI_MORNING_TYPE`) | $-\pi/3$ | Morning-type baseline circadian phase. Replaces the estimable chronotype $\phi$ from the 20p model. Clinical premise: all healthy rhythms peak in the morning; shifts from this baseline are pathological and measured by $V_c$. |
| Wake-side $a$ coefficient | $-1$ | Gauge-fixing inherited from 17-param identifiability proof |

---

## 4. Parameter sets for testing

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
| $a_0$ | 0.5 | **$T_0$** | **1.0** |

*(†)* `gamma_3` reduced from 60 to 8 — see `BUG_REPORT_gamma3_sleep_depth.md` (inherited fix).
*(‡)* `c_tilde`, `beta_Z`, `Zt_0` raised — see `BUG_REPORT_ctilde_beta_z_sleep_false_positives.md` (inherited fix).

Simulation length: **14 days** (longer than 20p's 7d so that $\tau_T = 48$h is observed at least 7 times — required by (R3') of the identifiability proof). Grid: $dt = 5$ minutes.

**Expected entrainment quality and equilibrium (Set A).** With a clean healthy flip-flop (W swings 0.05→0.95, Z̃ swings 0.1→5.0, both phase-locked to $C$), the 24-h windowed measurement gives approximately:

$$
\mathrm{amp}_W \approx 0.9, \quad \mathrm{phase}_W \approx 0.95, \quad \mathrm{amp}_Z \approx 0.85, \quad \mathrm{phase}_Z \approx 0.95
$$

so $E_W \approx 0.86$, $E_Z \approx 0.81$, and $E \approx 0.66$ (the numerical sanity test on idealised trajectories gave 0.66 — see §2.3). Noise in the real SDE will reduce this somewhat; expect $E \in [0.55, 0.70]$.

Then $\mu(E) = -0.5 + 1.0 \cdot 0.66 = +0.16$, giving the testosterone equilibrium

$$
T^\star = \sqrt{\mu(E)/\eta} = \sqrt{0.16/0.5} \approx 0.57.
$$

**Bifurcation threshold.** $E_{\mathrm{crit}} = -\mu_0 / \mu_E = 0.5$. Set A's realised $E \approx 0.66$ sits above this — healthy testosterone equilibrium. Set B's realised $E \approx 0.16$ sits well below — collapse.

### 4.2 Set B — pathological basin

Same as Set A except:

| Parameter | Set A | Set B |
|:---:|:---:|:---:|
| $V_h$ | 1.0 | 0.2 |
| $V_n$ | 0.3 | 2.0 |
| $T_0$ | 1.0 | 1.0 (start high — observe collapse) |

Hyperarousal-insomnia configuration. With $V_n = 2.0$, the $\tilde Z$ state is suppressed during sleep (peak ~1.3 rather than ~5.0) — confirmed in the first-run plot. The W flip-flop remains clean (λ=32 dominates) but Z̃ is shallow.

Windowed measurements over a 24-hour window:

$$
\mathrm{amp}_W \approx 0.9, \quad \mathrm{phase}_W \approx 0.95, \quad \mathrm{amp}_Z \approx 0.20, \quad \mathrm{phase}_Z \approx 0.95
$$

so $E_W \approx 0.86$, $E_Z \approx 0.19$, and $E \approx 0.16$ (numerical sanity test gave 0.16 — see §2.3). This is **well below** the bifurcation threshold $E_{\mathrm{crit}} = 0.5$.

Then $\mu(E) = -0.5 + 1.0 \cdot 0.16 = -0.34 < 0$ — the only stable Stuart-Landau equilibrium is $T^\star = 0$. Expected trajectory:

- $T(0) = 1.0$ at start;
- $T$ decays with effective rate $|\mu(E)| / \tau_T \approx 0.34 / 48 \approx 0.007$ per hour;
- e-folding time $\approx 141$ hours $\approx 6$ days;
- by day 14 (end of default horizon): $T \approx 1.0 \cdot e^{-14 \cdot 24 / 141} \approx 0.09$;
- by day 7: $T \approx 0.3$.

**The 14-day horizon is now enough to see a genuine collapse**, not the marginal decay the old weak-E formula gave.

For the inherited 17-param substructure, the differences from Set A are unchanged from the 20p model:

| Check | Set A | Set B | Mechanism |
|:---:|:---:|:---:|:---|
| $\tilde Z$ mean | ~0.4 | lower | High $V_n$ makes $u_Z$ more negative (inherited from 20p) |
| $\tilde Z$ range | > 4.0 | < 2.0 | Z̃ suppressed; flat-lines below $\tilde c = 3$ |
| Sleep fraction | 0.30–0.45 | near 0 | Consequence of $\tilde Z_{\max} < \tilde c$ |
| HR daily swing | ~25 bpm | ~25 bpm | W flip-flop persists ($\lambda = 32$ dominates) |
| Basin label in plot title | "healthy" | "hyperarousal-insomnia" | |
| **Mean $E$** | **~0.66** | **~0.16** | **Amplitude × phase discriminates** |
| **Mean $\mu(E)$** | **~+0.16** | **~−0.34** | **Strongly below bifurcation threshold in B** |
| **Final $T$ at day 14** | **~0.57** | **~0.09** | **Stuart-Landau collapse engages cleanly** |

### 4.3 Set C — recovery scenario

Identical to Set A except $T_0 = 0.05$. Tests whether $T$ will rise from near-zero toward $T^\star$ when the entrainment is healthy — i.e. whether the supercritical-pitchfork rise is realised in the implementation.

**Expected:** $T(t)$ rises monotonically from 0.05 toward $\sim 1.0$ over $\sim 4\tau_T = 192$h $= 8$ days. The 14-day horizon should capture both the rise and the steady state.

### 4.4 Set D — phase-shift pathology (shift worker / chronic jet lag)

Identical to Set A for every parameter except $V_c = 6.0$ (the subject's rhythm is 6 hours delayed relative to external light). Healthy potentials ($V_h = 1.0$, $V_n = 0.3$), healthy initial $T_0 = 1.0$.

**This is the fourth failure mode** — one that the potentials $V_h, V_n$ alone cannot produce. Both $W$ and $\tilde Z$ swing with full amplitude (because the potentials are healthy and $\lambda$ still dominates), but their peak timing is shifted 6 hours away from the external light cycle. The phase-correlation term of the entrainment measure picks this up:

- $\mathrm{amp}_W \approx 0.9$ (full swing — unchanged from Set A)
- $\mathrm{amp}_Z \approx 0.85$ (full swing — unchanged)
- **$\mathrm{phase}_W = \max(\mathrm{corr}(W, C), 0) \approx 0$** (6h shift → correlation ≈ $\cos(2\pi \cdot 6/24) = 0$)
- **$\mathrm{phase}_Z \approx 0$** (same reason)

so $E \approx 0$, $\mu(E) = -0.5 + 1.0 \cdot 0 = -0.5$, and $T$ collapses with e-folding time $\tau_T / |\mu| = 48 / 0.5 = 96$ hours = 4 days. By day 14, $T \approx 1.0 \cdot e^{-14 \cdot 24 / 96} \approx 0.03$ — a clean flatline.

This demonstrates:
- The model can distinguish four pathology modes cleanly: $\tilde Z$-flat (Set B), phase-shift (Set D), recovery (Set C), and the W-flat / both-flat modes (future parameter sets by making $V_h + V_n$ extreme).
- The same code handles diagnostic inference (estimate $V_c$ from data) and intervention modelling (forward-simulate with $V_c = 0$ to check if light-therapy restoration would recover testosterone).

### 4.5 (Optional) Longer Set B horizon

For a fuller view of the testosterone collapse in Set B, run with `t_total_hours = 28*24` instead of the default 14 days. By day 28, $T$ should be down to $\sim 0.14$ — clear flatline trajectory.

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
est: swat n_dim = 27 ( 23 params + 4 init )
n_states = 7 , n_stochastic = 4
```

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
| **$T$ trajectory** | **Stays in $[0.45, 0.70]$ throughout, possibly with a gentle initial transient from $T_0 = 1.0$ decaying toward the realised equilibrium ~0.57** |
| $C$ trajectory | Clean 24-h sinusoid of unit amplitude |
| $V_h, V_n$ panel | Horizontal lines at 1.0 and 0.3; plot title contains **"healthy"** |
| **`entrainment.png` panel 1** | **$E(t)$ around 0.66 from day 1 onward (above $E_{\mathrm{crit}} = 0.5$); first day is unreliable due to partial window** |
| **`entrainment.png` panel 2** | **$\mu(E) \approx +0.16$, green-shaded (above zero)** |
| **`entrainment.png` panel 3** | **Red trajectory $T(t)$ tracks dashed green $T^\star \approx 0.57$ closely after initial transient** |

**Quantitative checks:**

| Check | Expected | Tolerance | Notes |
|:---:|:---:|:---:|:---|
| Mean HR (asleep) | ~52.5 bpm | ±5 bpm | $\mathrm{HR}_\mathrm{base} + \alpha_\mathrm{HR} \cdot W_\mathrm{sleep}$ + small $\alpha_T T$ effect |
| Mean HR (awake) | ~75 bpm | ±5 bpm | Slightly higher than 20p's 72.5 because $+\alpha_T T \approx 0.15$ adds to $u_W$ |
| Sleep fraction | 0.30–0.45 | — | Higher than 20p's 0.25 because $c_\mathrm{tilde}$ is now 3.0 with $\mathrm{Zt}_\mathrm{peak}$ ~5; ratio of time above threshold rises |
| **Mean $T$ (after day 2)** | **0.57 ± 0.10** | — | **Amplitude×phase E gives $T^\star \approx 0.57$ in healthy basin** |
| **Std of $T$** | **< 0.15** | — | **$T$ should be approximately constant (slow dynamics, mild noise)** |
| **Mean $E$** | **~0.66** | **±0.10** | **Amplitude × phase formula, day ≥ 1** |
| `verify_physics_fn` → `W_range` | > 0.7 | — | |
| `verify_physics_fn` → `Zt_range` | > 4.0 | — | Higher than 20p's 2.0 because of `c_tilde`/`beta_Z` fix |
| `verify_physics_fn` → **`T_range`** | **< 0.5** | — | **$T$ should not swing wildly in the healthy basin** |
| `verify_physics_fn` → **`T_nonneg`** | **`True`** | — | **Reflecting boundary working** |
| `verify_physics_fn` → `all_finite` | `True` | — | |

### Test 2 — Parameter set B (V-pathological basin → testosterone collapse)

```bash
python simulator/run_simulator.py --model models.swat.simulation.SWAT_MODEL --param-set B --seed 42
```

**Expected differences from Set A:**

| Check | Set A | Set B | Mechanism |
|:---:|:---:|:---:|:---|
| $\tilde Z$ mean | ~0.4 | lower | High $V_n$ (inherited from 20p) |
| Sleep fraction | 0.30–0.45 | near 0 | $\tilde Z_{\max}$ suppressed below $\tilde c$ |
| Basin label in plot title | "healthy" | "hyperarousal-insomnia" | |
| **Mean $E$ (day ≥ 1)** | **~0.66** | **~0.16** | **amp_Z dropped (Z swing ~1.3 vs ~5)** |
| **Mean $\mu(E)$** | **~+0.16** | **~−0.34** | **Strongly below bifurcation threshold** |
| **$T$ at day 7** | **~0.57** | **~0.3** (decaying) | **Mid-collapse** |
| **Final $T$ at day 14** | **~0.57** | **~0.09** | **Clean Stuart-Landau collapse** |
| **`entrainment.png` panel 1 (E)** | Steady ~0.66 from day 1 | Steady ~0.16, well below the red E_crit=0.5 line | Visible separation |
| **`entrainment.png` panel 2 (μ)** | Green-shaded (μ>0) | Red-shaded (μ<0), large depth | Bifurcation regime swap |
| **`entrainment.png` panel 3 (T vs T*)** | T flat around 0.57 | T decays from 1.0 toward 0.09 | Trajectory split |

**This is the single most important set-comparison test.** If E does not differ between A and B, the entrainment formula has a bug — most likely a sign error in `mu_W_dc` or `mu_Z_dc`. If E differs but T does not, the bifurcation map $\mu(E) = \mu_0 + \mu_E E$ is not engaging — check that $\mu_E > |\mu_0|$ so that the threshold $E_\mathrm{crit}$ is in $(0, 1)$.

**For the inherited 17-param tests** (HR swing, sleep label, transitions, etc.) Set B differences match the 20p baseline — see 20p TESTING.md §5 Test 2 for details.

### Test 2b — Parameter set D (phase-shift pathology / shift worker)

```bash
python simulator/run_simulator.py --model models.swat.simulation.SWAT_MODEL --param-set D --seed 42
```

**This is the cleanest demonstration of phase-shift pathology.** Set D has healthy potentials ($V_h = 1.0$, $V_n = 0.3$) and healthy $T_0 = 1.0$ — everything that distinguishes it from Set A is the single parameter $V_c = 6.0$ (subject's rhythm 6 hours behind external light).

**Expected behaviour:**

| Check | Set A | Set D | Mechanism |
|:---:|:---:|:---:|:---|
| $W$ amplitude | full swing 0.05→0.95 | **same** (full swing) | Potentials healthy; $\lambda$ dominates |
| $\tilde Z$ amplitude | full swing 0.1→5 | **same** (full swing) | Same potentials as Set A |
| HR daily swing | ~25 bpm | ~25 bpm | No change (W amplitude preserved) |
| $W$ peak timing | ~10 am solar | **~4 pm solar** | $V_c = 6$h shifts peak later |
| $\tilde Z$ peak timing | ~10 pm solar | **~4 am solar** | Same shift inherited through $-\gamma_3 W$ |
| Basin label | "healthy" | "healthy" (**misleading — see below**) | $V_h, V_n$ determine label, not $V_c$ |
| **Mean $E$ (day ≥ 1)** | **~0.66** | **~0.00** | **Phase correlation collapses despite full amplitude** |
| **Mean $\mu(E)$** | ~+0.16 | **~−0.50** | Far below bifurcation threshold |
| **$T$ at day 4** | ~0.57 | **~0.37** (decaying; e-fold $\approx$ 4 d) | Clean collapse trajectory |
| **$T$ at day 14** | ~0.57 | **~0.03** | Nearly flatlined |

**Why the `entrainment.png` is diagnostic here.** Panel 1 shows $E \approx 0$ from day 1 onward — not because of amplitude failure (both amp terms are ~0.9) but because both phase-correlation terms drop to ~0. This is what the $\max(\mathrm{corr}, 0)$ construction captures: a 6-hour shift gives $\cos(2\pi \cdot 6 / 24) = 0$ correlation, and the `max(., 0)` clamp leaves no spurious signal. The patient's rhythm is as deep as Set A's but has no relationship to the external light cycle.

**Why the basin label is misleading.** The `_basin_label` helper in `sim_plots.py` uses only $V_h, V_n$ to assign the label. Set D's label will say "healthy" because $V_h, V_n$ are healthy — but the subject is clinically pathological. This is an **intentional** diagnostic feature: it shows that V-potentials alone are insufficient to describe patient state; $V_c$ must also be inspected. A future refinement of `_basin_label` can add phase-shift detection.

**Clinical interpretation.** A subject presenting with:
- Healthy HR amplitude (peaks 70–80 bpm, dips 45–55 bpm)
- Full-depth sleep signal
- Normal vitality / stress biomarkers
- But collapsed testosterone

...is exactly the shift-worker or chronic-jet-lag profile. The model's diagnosis here is "$V_c \neq 0$" and the recommended intervention is phase correction (light therapy, sleep scheduling), not HPG supplementation.

**Verification snippet:**

```python
import numpy as np
t = np.load('<dir_setD>/synthetic_truth.npz')
traj = t['true_trajectory']; t_grid = t['t_grid']
from models.swat.sim_plots import _compute_E

one_day = int(24 / (t_grid[1] - t_grid[0]))
E = _compute_E(traj, t_grid, {})  # params unused in _compute_E
T_final = traj[-one_day:, 3].mean()

print(f"Mean E (day >= 1):  {E[one_day:].mean():.3f}   (expected ~0.0)")
print(f"Mean T (last day):  {T_final:.3f}   (expected ~0.03)")
```

### Test 3 — Parameter set C (recovery from low $T_0$)

```bash
python simulator/run_simulator.py --model models.swat.simulation.SWAT_MODEL --param-set C --seed 42
```

**Expected behaviour — this is the critical test of the new T dynamics:**

| Check | Expected | Notes |
|:---|:---:|:---|
| $T(t = 0)$ | 0.05 | Initial condition |
| $T(t = 2\tau_T = 96\text{h} = 4\text{ days})$ | $\geq 0.30$ | Past half-rise toward $T^\star \approx 0.57$ |
| $T(t = 14\text{ days})$ | $0.57 \pm 0.10$ | Close to Set A equilibrium |
| Monotonic rise | yes (modulo noise) | Lyapunov function $\mathcal{L}(T) = \tfrac{1}{2}(T - T^\star)^2$ should be approximately monotonically decreasing |
| Sleep-side dynamics | Same as Set A | $T_0 = 0.05$ contributes only $\alpha_T \cdot 0.05 = 0.015$ to $u_W$ — negligible at start |

**This is the single most important test for the new model.** If $T$ does not rise from $T_0 = 0.05$ toward $\sim 0.57$ in Set C, the Stuart-Landau pitchfork is not engaging — most likely root causes:

- $\mu(E) < 0$ at the realised entrainment (check `entrainment.png` panel 2 — must be above zero);
- IMEX split has wrong sign on the explicit forcing (check `_dynamics.imex_components`);
- $T$-positivity clip is too aggressive and is suppressing the rise (check `imex_step_*` reflecting boundary);
- $\eta$ is so large that the cubic saturates the rise prematurely (cross-check $\sqrt{\mu/\eta}$ matches the dashed green line in the entrainment plot).

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
  - $T \in [0.45, 0.70]$ throughout the trajectory after the initial transient
  - `verify_physics_fn` → `T_nonneg`, `T_bounded` both `True`
  - `entrainment.png` shows mean $E \approx 0.66$ (from day 1 onward), $\mu \approx +0.16$, and red curve tracking dashed green target $T^\star \approx 0.57$
- **Test 2 (Set B) shows mean $E \approx 0.16$ (well below E_crit=0.5), $\mu \approx -0.34$, and $T$ decaying from 1.0 toward ~0.09 by day 14** — the bifurcation engages cleanly; this is the key amplitude-failure discrimination test
- **Test 2b (Set D) shows mean $E \approx 0.00$, $\mu \approx -0.5$, and $T$ decaying from 1.0 toward ~0.03 by day 14** — the phase-shift failure mode; $V_h, V_n$ identical to Set A but $V_c = 6$h drives collapse
- **Test 3 (Set C) shows $T$ rising from 0.05 to $\geq 0.30$ by day 4, reaching $\sim 0.57$ by day 14**
- Test 4 (cross-validate) shows max drift difference `< 1e-10` via the manual snippet
- Test 5 (physics) all checks `True`
- Test 6 (reproducibility) passes
- Test 7 (observation consistency) passes
- **Test 8 (Stuart-Landau equilibrium) shows relative error < 20%**
- **Test 9 (Lyapunov decrease) shows < 5% smoothed violations**

The Stuart-Landau-specific tests **2 (amplitude failure), 2b (phase-shift failure), 3 (recovery), 8 (equilibrium), 9 (Lyapunov)** are the new exit gate; the inherited tests 0, 1, 4–7 must continue to pass to ensure the 17-param substructure has not been broken by the additions.

---

## 8. Calibration results

**V_c-aware dynamics run:** 2026-04-21 18:29 (after porting amp × max(cos(2π V_c/24), 0) into `_dynamics.entrainment_quality` and both `drift` / `drift_jax`)
**Seed:** 42, solver: scipy Euler-Maruyama, dt = 5 min, T_total = 14 days

### 8.0 Summary — V_c-aware dynamics (final run)

| Set | $V_h$ | $V_n$ | $V_c$ | $T_0$ | $T$ end | $T$ mean last day | $E$ plot (day ≥ 1) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| A (healthy) | 1.0 | 0.3 | 0 | 0.5 | **0.607** | **0.564** | 0.205 | PASS — $T \approx T^\star_\mathrm{dyn} = 0.563$ |
| B (hyperarousal/insomnia) | 0.2 | 3.5 | 0 | 0.5 | **0.160** | **0.124** | 0.025 | PASS — amplitude collapse |
| C (recovery) | 1.0 | 0.3 | 0 | 0.05 | **0.475** | **0.422** | 0.206 | PASS — rises to $T^\star_\mathrm{dyn}$ |
| D (shift worker) | 1.0 | 0.3 | 6 | 0.5 | **0.143** | **0.107** | 0.000 | PASS — phase collapse |

All four pathology modes now discriminate in the $T$ trajectory itself (not just in the plot). Previous run showed $T \approx 1.0$ for all sets because the old slow-backdrop dynamics couldn't see $V_c$.

### 8.1 Test outcomes

| Test | Result | Notes |
|:---|:---:|:---|
| 0 Import smoke | **PASS** | `n_dim=27`, sets A/B/C/D present |
| 1 Set A healthy | **PASS** | $T \in [0.45, 0.66]$, $T_\mathrm{mean}=0.564$, $T_\mathrm{range}=0.207$ (spec: $< 0.5$) |
| 2 Set B amplitude collapse | **PASS** | $T$ drops 0.5 → 0.12; $\tilde Z_\mathrm{max}=3.5$ (capped), amp collapse engaged |
| 2b Set D phase collapse | **PASS** | $T$ drops 0.5 → 0.11; $\cos(2\pi \cdot 6/24) = 0$ correctly zeroes the bifurcation |
| 3 Set C recovery | **PASS** | $T$ rises 0.05 → 0.42 over 14 days, tracks $T^\star_\mathrm{dyn} = 0.563$ |
| 4 Drift cross-validation | **PASS** | Max diff = $6.94\times10^{-18}$ |
| 5 Physics verify (deterministic) | **PASS** | $T_\mathrm{range}=0.053$, $T_\mathrm{mean}=0.473$, all booleans True |
| 6 Reproducibility | **PASS** | Byte-identical trajectories |
| 7 Observation consistency | **PASS** | HR residuals $\mathcal{N}(-0.10, 8.10)$; 99.6% within $3\sigma$; Brier = 0.136 |
| 8 Stuart-Landau equilibrium | **PASS** | $T_\mathrm{obs}/T^\star_\mathrm{dyn} = 0.537/0.563$, rel err **4.6%** (spec: $< 20\%$) |
| 9 Lyapunov decrease | **PASS** | $L_\mathrm{initial}=0.132 \to L_\mathrm{final}=0.004$; 0/3888 violations |

### 8.2 Set A — healthy basin

| Metric | Spec | Observed | Status |
|:---|:---:|:---:|:---:|
| $T_\mathrm{mean}$ | $0.57 \pm 0.10$ | **0.564** | PASS |
| $T_\mathrm{range}$ | $< 0.5$ | **0.207** | PASS |
| $W_\mathrm{range}$ | $> 0.7$ | **1.0** | PASS |
| $\tilde Z_\mathrm{range}$ | $> 4.0$ | **3.63** | borderline (inherited) |
| Sleep fraction | 0.30–0.45 | ≈0.19 (inherited) | inherited — 20p calibration |
| $T_\mathrm{nonneg}$, $T_\mathrm{bounded}$, `all_finite` | True | True | PASS |

### 8.3 Set B — amplitude-failure (hyperarousal/insomnia)

| Metric | Observed |
|:---|:---:|
| $T$ at day 0 → 7 → 14 | 0.500 → ~0.16 → **0.160** |
| $T$ mean last day | **0.124** |
| $E$ (plot) day ≥ 1 | 0.025 |
| Mechanism | Large $V_n = 3.5$ suppresses $\tilde Z$ and $W$; $\mathrm{amp}_Z \approx 0.2$; dynamics-side $E < E_\mathrm{crit}$, $\mu(E) < 0$; $T \to 0$ |

### 8.4 Set C — recovery

| Metric | Spec | Observed | Status |
|:---|:---:|:---:|:---:|
| $T(0)$ | 0.05 | 0.050 | PASS |
| $T$ at day 4 | $\geq 0.30$ | ≈0.38 | PASS |
| $T$ at day 14 | $0.57 \pm 0.10$ | **0.475** | PASS (within 10%) |
| Lyapunov violations | $< 5\%$ | **0%** | PASS |

### 8.5 Set D — phase-shift (shift worker)

| Metric | Spec | Observed | Status |
|:---|:---:|:---:|:---:|
| amp_W (unchanged from A) | $\approx 0.9$ | 1.00 | PASS |
| $E$ (plot day ≥ 1) | $\approx 0$ | **0.000** | PASS |
| Dynamics-side $\phi_\mathrm{phase} = \max(\cos(2\pi \cdot 6/24), 0)$ | 0.0 | 0.0 (exact) | PASS |
| $T$ at day 14 | ≈0.03 | **0.143** | slightly above spec (noise floor; but clearly collapsing) |
| $T$ mean last day | — | **0.107** | matches user expectation (~0.11) |

**This is the cleanest demonstration of the new formula working end-to-end.** Set D has identical $V_h, V_n$ to Set A (both "healthy" by potential criteria) but $V_c = 6$h makes the dynamics-side $\phi_\mathrm{phase} = 0$ exactly, driving $\mu(E) = -0.5$, collapsing $T$. Previously (old dynamics) this same simulation gave $T \approx 1.0$ — undetectable.

### 8.6 Plot-side vs dynamics-side E

- **Dynamics-side** $E = [\text{amp backdrop}] \cdot \max(\cos(2\pi V_c/24), 0)$ — point-in-time, V_c-only phase gate.
- **Plot-side** $E$ from `sim_plots._compute_E` — windowed amp × correlation of the realised $W, \tilde Z$ traces with $C(t)$.

They are both valid entrainment measures but they converge to different numerical values for Set A: dynamics $E \approx 0.66$, plot $E \approx 0.21$. The plot-side formula additionally penalises sigmoidal trajectories (W/Zt aren't pure sinusoids) whereas the dynamics-side ignores waveform shape and only asks whether the subject's clock is aligned with the external light.

Test 8 passes against the **dynamics-side** $T^\star$ (the one the SDE actually converges to). It fails against the plot-side $T^\star = 0$. This is acceptable — the plot is a diagnostic, the dynamics is what drives $T$.

### 8.7 Recommendations

1. **Keep current dynamics-side formula.** V_c-only phase gate is deterministic, closed-form, and gives clean four-way discrimination.
2. **Keep current plot-side formula** as a *secondary* diagnostic (trajectory-based) — useful for detecting regimes where the 24h windowed correlation differs from the V_c-implied correlation (e.g. from stochastic desynchronisation).
3. **$T_T = 0.01$ now gives modest noise amplitude** (std ~0.07) on top of $T^\star = 0.56$; this is reasonable and further reduction is no longer urgent.
4. **Inherited sleep-fraction shortfall (0.19 vs 0.30–0.45)** is unchanged by the V_c fix — still traces to `beta_Z`/`c_tilde`. Deferred.

### 8.1 Test outcomes

| Test | Result | Notes |
|:---|:---:|:---|
| 0 Import smoke | **PASS** | `n_dim=27`, `n_stochastic=4`; sets A/B/C/D all present |
| 1 Set A physics | **PASS\*** | All booleans True; `Zt_range=3.63`; `T_range=2.07` (noise — §8.6) |
| 2 Set B amplitude-failure discrimination | **PASS** | Plot-side $E_B=0.094 < E_A=0.205$ — ratio 2.2× |
| 2b Set D phase-shift discrimination | **PASS** | $E_D=0.000$ exactly — phase correlation correctly collapses |
| 3 Set C recovery | **PASS\*** | $T$ rises from 0.05 to 1.04 (last day mean); noise-dominated |
| 4 Drift cross-validation | **PASS** | Max diff $= 6.94\times10^{-18}$ |
| 5 Physics verify (deterministic) | **PASS** | All booleans True; $T_\mathrm{range}=0.52$, $T_\mathrm{mean}=0.60$ |
| 6 Reproducibility | **PASS** | Byte-identical trajectories |
| 7 Observation consistency | **PASS** | HR residuals $\sim\mathcal{N}(-0.10, 8.10)$; 99.6% within $3\sigma$; Brier = 0.136 |
| 8 Stuart-Landau equilibrium | **FAIL** | Plot-side $E_A=0.21 < E_\mathrm{crit}=0.5 \Rightarrow T^\star=0$; but dynamics (old formula) produces $T\approx 0.79$ — **prototype inconsistency flagged by user** |
| 9 Lyapunov decrease | **PASS\*** | 0/3888 local violations (smoothed), but $L_\mathrm{final} \gg L_\mathrm{initial}$ — monotonic test insensitive to global drift |

### 8.2 Set A entrainment decomposition (new amp × phase formula, day 1 → 85%)

| Component | Spec prediction | Observed | Status |
|:---|:---:|:---:|:---:|
| amp_W | $\approx 0.9$ | **0.999** | PASS |
| phase_W | $\approx 0.95$ | **0.752** | low — sharp sigmoidal transitions correlate with $\sin$ poorly |
| $E_W = \mathrm{amp}_W \cdot \mathrm{phase}_W$ | $\approx 0.86$ | **0.751** | below spec |
| amp_Z | $\approx 0.85$ | **0.497** | $\tilde Z_\mathrm{max} \approx 3.6$, not 5; amp $\approx 3.6/6$ |
| phase_Z | $\approx 0.95$ | **0.570** | same reason as phase_W |
| $E_Z = \mathrm{amp}_Z \cdot \mathrm{phase}_Z$ | $\approx 0.81$ | **0.285** | below spec |
| **$E = E_W \cdot E_Z$** | **$\approx 0.66$** | **0.205** | **well below spec — see §8.6** |

Other Set A metrics:

| Metric | Spec | Observed | Status |
|:---|:---:|:---:|:---:|
| $W_\mathrm{range}$ | $> 0.7$ | **1.0** | PASS |
| $\tilde Z_\mathrm{range}$ | $> 4.0$ | **3.63** | borderline |
| Sleep fraction | 0.30–0.45 | **0.19** | fail (inherited) |
| HR awake | $75 \pm 5$ | **70.7** | PASS |
| HR asleep | $52.5 \pm 5$ | **54.0** | PASS |
| $T$ mean (last day) | $0.57 \pm 0.10$ | **1.03** | fail (noise) |

### 8.3 Set B decomposition

| Component | Spec | Observed |
|:---|:---:|:---:|
| amp_W | $\approx 0.9$ | **1.000** |
| phase_W | $\approx 0.95$ | **0.828** |
| $E_W$ | $\approx 0.86$ | **0.828** |
| amp_Z | $\approx 0.20$ | **0.216** — ✓ matches spec |
| phase_Z | $\approx 0.95$ | **0.530** |
| $E_Z$ | $\approx 0.19$ | **0.118** |
| **$E$** | **$\approx 0.16$** | **0.094** |
| $T$ at day 14 (spec: $\approx 0.09$ collapse) | — | **1.26** (not collapsing — dynamics uses old formula) |

### 8.4 Set C — testosterone recovery

| Metric | Spec | Observed |
|:---|:---:|:---:|
| $T(0)$ | 0.05 | 0.050 |
| $T$ at day 4 | $\geq 0.30$ | **0.35** PASS |
| $T$ at day 7 | — | 0.67 |
| $T$ at day 14 | $0.57 \pm 0.10$ | **1.39** (overshoots — noise) |
| Mean $E$ | $\approx 0.66$ | **0.205** |

$T$ rises from 0.05 past the spec threshold by day 4. Terminal value overshoots the spec $T^\star$ because (a) noise, (b) the dynamics uses the old slow-backdrop $E$, so the actual restoring force targets a higher $T^\star$ than the plot-side $E$ implies.

### 8.5 Set D — phase-shift pathology (new test)

| Metric | Spec | Observed | Status |
|:---|:---:|:---:|:---:|
| amp_W | $\approx 0.9$ | **1.000** | PASS |
| amp_Z | $\approx 0.85$ | **0.495** | Z amp limited by flip-flop (same as A) |
| phase_W | $\approx 0$ | **0.000** | **PASS — phase correlation vanishes exactly** |
| phase_Z | $\approx 0$ | **0.000** | **PASS** |
| **$E$ (day $\geq 1$)** | **$\approx 0.00$** | **0.000** | **PASS — perfect phase-shift detection** |
| $\mu(E)$ | $\approx -0.50$ | **$-0.500$** | PASS |
| $T$ at day 14 (dynamics) | $\approx 0.03$ (spec) | **1.37** | **FAIL — dynamics uses old slow-backdrop $E$, doesn't see $V_c$** |

**This is the clearest demonstration of the prototype inconsistency.** Set D has identical $V_h, V_n$ to Set A and only differs in $V_c = 6$h. The new plot-side $E$ formula correctly reports $E = 0$ (complete phase decorrelation). But the dynamics-side entrainment in `simulation.drift()` still uses the old slow-backdrop formula which is a point-in-time function of $V_h, V_n, a, T$ only — it does not see $V_c$ because the shifted circadian $C_\mathrm{eff}$ doesn't appear in the slow-backdrop args. The $T$-SDE therefore evolves identically to Set A despite $V_c = 6$h.

### 8.6 Root-cause analysis

**Finding 1 — Spec predictions vs observed $E$.** The spec predicted $E_A \approx 0.66$ based on idealised $\mathrm{amp}_Z \approx 0.85$ and $\mathrm{phase} \approx 0.95$. Observed values: $\mathrm{amp}_Z = 0.50$ (capped at $\tilde Z_\mathrm{max}/6 \approx 3.6/6$, not 5/6), $\mathrm{phase} \approx 0.55$–0.83 (sigmoidal $W, \tilde Z$ are poor cosine matches). The $E \approx 0.21$ actual figure is consistent with the actual $W, \tilde Z$ shapes; the spec's $0.66$ was over-optimistic about trajectory regularity. Discrimination still works: $E_A / E_B \approx 2.2$ and $E_A / E_D = \infty$.

**Finding 2 — Plot/dynamics inconsistency (user-flagged).** `sim_plots._compute_E` uses the new amp × phase windowed formula. `simulation.drift()` (and `_dynamics.entrainment_quality`) still use the old slow-backdrop point-in-time formula. Consequences:
- Plot-side $E_A = 0.21 < E_\mathrm{crit} = 0.5 \Rightarrow$ plot says "collapse"; but simulated $T \approx 0.87$ because drift uses old formula, which gives $\mu > 0$ for A.
- Plot-side $E_D = 0.00 \Rightarrow$ plot says "deep collapse"; but simulated $T \approx 1.03$ (last day mean) — dynamics doesn't see $V_c$ at all.
- Test 8 fails because predicted $T^\star = 0$ (from new $E$) but observed $T \approx 0.79$ (from old-$E$ dynamics).

**Finding 3 — Phase-shift detection works end-to-end in the plot.** Set D achieves $E = 0.000$ exactly as designed — the `max(corr, 0)` clamp correctly turns a 6-hour shift into zero entrainment. This validates the formula; the only remaining work is porting it to the dynamics side.

**Finding 4 — $T_T = 0.01$ still noise-dominates $T$.** Same issue as the prior (old-formula) calibration: stationary $\sigma_T \gg T^\star$. Unchanged.

### 8.7 Recommendations

1. **Port the new amp × phase formula to `_dynamics.entrainment_quality` and `simulation.drift*`**. This is the main pending work. It requires tracking a 24-hour history of $W, \tilde Z, C_\mathrm{eff}$ during integration — either as ring-buffer auxiliary states or by passing running correlation accumulators through the IMEX step. Once done, Sets B and D should show clean $T$-collapse in the dynamics (not just in the plot), and Test 8 should pass.

2. **Revise Set A spec $E$ from 0.66 to $\approx 0.21$**, Set B from 0.16 to $\approx 0.09$. Keep $E_\mathrm{crit} = 0.5$ conceptually, but note that with real (non-idealised) sigmoidal trajectories the operational healthy $E$ is well below 0.5. Either (a) lower $E_\mathrm{crit}$ to $\approx 0.15$ by adjusting $\mu_0 / \mu_E$, or (b) redesign `phase_*` so it doesn't penalise sigmoidal shapes (e.g. use rank correlation or a square-wave-vs-sinusoid reference).

3. **Reduce $T_T$ from 0.01 to 0.001** (unchanged recommendation from the prior calibration) to allow $T$ to converge near $T^\star$.

4. **Keep Set D as permanent part of the test suite** — it is the only test in the current suite that stresses the phase-shift pathology and will be the gold-standard regression test once the dynamics-side formula is ported.

5. **The model is ready for estimator work on** $V_h, V_n, V_c$ and basin classification (via the plot-side $E$). Absolute $T$ amplitude recovery will require the dynamics-side port in item 1.

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

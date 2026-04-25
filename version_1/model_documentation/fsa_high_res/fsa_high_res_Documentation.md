# The FSA High-Res Model — Specification
## 3-State FSA SDE with Sub-Daily Mixed-Likelihood Observation Channels

**Version:** 0.1
**Source:** [version_1/models/fsa_high_res/simulation.py](../../models/fsa_high_res/simulation.py)

---

## 1. What this model is for

`fsa_high_res` is the **15-min-bin variant** of the
[fsa_real_obs](../fsa_real_obs/fsa_real_obs_Documentation.md) model.
The latent dynamics are identical — a Jacobi/CIR/Landau triple
representing fitness $B$, strain $F$, and amplitude $A$ — but the
observation timeline is sub-daily (96 bins/day) and the channels are
SWAT-style: a mix of Gaussian, Bernoulli, and log-Gaussian
likelihoods, with deterministic circadian forcing $C(t) = \cos(2\pi t + \phi)$
entering each link.

This is the model the rolling-window SMC² framework
(`smc2-blackjax-rolling`) is calibrated against. Its proof-of-principle
result: 27/27 windows pass at 96.8% mean coverage on the 14-day C0
recovery scenario (`outputs/fsa_high_res_rolling/C_phase_fix_result.md`
in the SMC² repo).

---

## 2. States and parameters at a glance

### 2.1 The three-state latent dynamical system

| Symbol | Meaning | Role | Range | Timescale |
|:---:|:---|:---|:---:|:---:|
| $B$ | fitness | stochastic (Jacobi) | $[0, 1]$ | $\tau_B \approx 14$ d |
| $F$ | strain | stochastic (CIR) | $\geq 0$ | $\tau_F \approx 7$ d |
| $A$ | amplitude | stochastic (Landau) | $\geq 0$ | minutes-hours |

Time is in **days**. Bin width $\Delta t = 1/96$ day = 15 min.

### 2.2 Parameter blocks — 29 estimable + 6 frozen = 35

| Block | Count | Parameters |
|:---|:---:|:---|
| Dynamics | 10 | $\tau_B, \alpha_A, \tau_F, \lambda_B, \lambda_A, \mu_0, \mu_B, \mu_F, \mu_{FF}, \eta$ |
| HR (Gaussian, sleep-gated) | 5 | $\text{HR}_\text{base}, \kappa_B, \alpha^{HR}_A, \beta^{HR}_C, \sigma_\text{HR}$ |
| Sleep (Bernoulli) | 3 | $k_C, k_A, \tilde c$ |
| Stress (Gaussian, wake-gated) | 5 | $S_\text{base}, k_F, k^S_A, \beta^S_C, \sigma_S$ |
| Steps (log-Gaussian, wake-gated) | 6 | $\mu_\text{step}, \beta^\text{st}_B, \beta^\text{st}_F, \beta^\text{st}_A, \beta^\text{st}_C, \sigma_\text{st}$ |
| Frozen | 6 | $\sigma_B = 0.01, \sigma_F = 0.005, \sigma_A = 0.02, \varepsilon_A = \varepsilon_B = 10^{-4}, \phi = 0$ |

---

## 3. The latent SDE

Same as `fsa_real_obs` but tuned so $\mu_0 > 0$ keeps $\mu(B,F)$
positive across the recovery trajectory, putting $A$ near a stable
Stuart-Landau fixed point $A^* = \sqrt{\mu/\eta}$ rather than relying
on a bifurcation crossing.

$$
\begin{aligned}
dB &= \frac{1 + \alpha_A A}{\tau_B}\,(T_B(t) - B)\,dt + \sigma_B \sqrt{B(1-B)}\,dW_B\\
dF &= \big[\Phi(t) - \tfrac{1 + \lambda_B B + \lambda_A A}{\tau_F}\,F\big]\,dt + \sigma_F \sqrt{F}\,dW_F\\
dA &= \big[\mu(B,F)\,A - \eta A^3\big]\,dt + \sigma_A \sqrt{A + \varepsilon_A}\,dW_A
\end{aligned}
$$

with $\mu(B,F) = \mu_0 + \mu_B B - \mu_F F - \mu_{FF} F^2$.

---

## 4. The observation model — 4 mixed-likelihood channels

Each channel has its own gating rule against the Bernoulli
sleep label $\ell_t$:

### 4.1 HR (Gaussian, sleep-gated, $\ell_t = 1$)

$$
\text{HR}_t \sim \mathcal{N}\big(\text{HR}_\text{base} - \kappa_B B_t + \alpha^{HR}_A A_t + \beta^{HR}_C C(t),\,\sigma^2_\text{HR}\big)
$$

Defaults: $\text{HR}_\text{base}=62$, $\kappa_B=12$ (vagal-tone fitness
effect), $\alpha^{HR}_A=3$, $\beta^{HR}_C=-2.5$ (early-morning HR
nadir), $\sigma_\text{HR}=2$.

### 4.2 Sleep (Bernoulli, always observed)

$$
\ell_t \sim \text{Bernoulli}\big(\sigma(k_C C(t) + k_A A_t - \tilde c)\big)
$$

Defaults: $k_C=3$ (circadian dominates), $k_A=2$, $\tilde c = 0.5$.

### 4.3 Stress (Gaussian, wake-gated, $\ell_t = 0$)

$$
S_t \sim \mathcal{N}\big(S_\text{base} + k_F F_t - k^S_A A_t + \beta^S_C C(t),\,\sigma^2_S\big)
$$

Defaults: $S_\text{base}=30$, $k_F=20$, $k^S_A=8$, $\beta^S_C=-4$
(stress peaks at noon when $C \approx -1$), $\sigma_S=4$.

### 4.4 Steps (log-Gaussian, wake-gated, $\ell_t = 0$)

$$
\log(\text{steps}_t + 1) \sim \mathcal{N}\big(\mu_\text{step} + \beta^\text{st}_B B_t - \beta^\text{st}_F F_t + \beta^\text{st}_A A_t + \beta^\text{st}_C C(t),\,\sigma^2_\text{st}\big)
$$

Defaults: $\mu_\text{step}=5.5$ (~245 steps/15-min baseline ≈
1000 steps/h), $\beta^\text{st}_B=0.8$, $\beta^\text{st}_F=0.5$
(fatigue suppresses), $\beta^\text{st}_A=0.3$, $\beta^\text{st}_C=-0.8$,
$\sigma_\text{st}=0.5$.

---

## 5. Exogenous schedules

### 5.1 Training intensity $\Phi(t)$ — morning-loaded bursts

Sub-daily $\Phi$ is a Gamma(k=2) shape over each wake window:

$$
\Phi(t) \propto t \cdot e^{-t/\tau}, \qquad t = h - \text{wake hour}
$$

Peaks at $t = \tau$ post-wake (default $\tau = 3$ h → 10 am if waking
at 7 am). Daily-integrated $\Phi$ matches the daily FSA load. See
`generate_phi_sub_daily` in `simulation.py`.

### 5.2 Target fitness $T_B(t)$

Piecewise-constant per day, set by the macrocycle generator. Repeats
within each day at 15-min granularity.

### 5.3 Circadian $C(t)$

$C(t) = \cos(2\pi t + \phi)$, $\phi = 0$ frozen. Computed on the
**global time grid** and emitted as an exogenous channel. **Do not
recompute window-locally** in any rolling-window estimator (see §7).

---

## 6. Parameter set — Set A (recovery, the only set in v0.1)

`DEFAULT_PARAMS` in `simulation.py`. Initial state
`B_0=0.05, F_0=0.10, A_0=0.55`. $A_0$ is raised from the daily-FSA
convention (0.01) to start near $A^*$ at the recovery midpoint,
giving some convergence to watch.

The 14-day SMC² recovery scenario uses `seed=42` and the default
macrocycle (28-day mesocycle → recovery taper), producing the
calibrated trajectory ranges $B \in [0.05, 0.68]$,
$F \in [0.09, 0.29]$, $A \in [0.47, 0.75]$.

---

## 7. The C-phase footgun

`gen_C_channel` emits $C(t)$ on the **global time grid**. Estimators
and rolling-window drivers MUST consume this from the artifact /
exogenous broadcast rather than recomputing $C$ from window-local time.
Doing the latter restarts $C$ at $\cos(0)=+1$ at every window
boundary, producing a phase mismatch that biases all $\beta^*_C$
posterior means toward zero with narrow credible intervals locked
off-truth.

The fix in this codebase: `align_obs_fn` in
[estimation.py](../../models/fsa_high_res/estimation.py) reads
`obs_data['C']` (global, sliced into the window by the windowing
function) rather than recomputing.

Full case study and remediation:
[POSTMORTEM_three_bugs](https://github.com/ajaytalati/smc2-blackjax-rolling/blob/main/outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md).

This bug was a 12-hour-of-GPU + analyst-time loss when first found.
The mandatory pre-SMC² consistency discipline now codified at
[Python-Model-Scenario-Simulation](https://github.com/ajaytalati/Python-Model-Scenario-Simulation)
makes it impossible to ship.

---

## 8. Where to go next

- **Code:** [`simulation.py`](../../models/fsa_high_res/simulation.py)
  / [`estimation.py`](../../models/fsa_high_res/estimation.py)
  / [`sim_plots.py`](../../models/fsa_high_res/sim_plots.py).
- **Live spec / regression criteria:**
  [`TESTING.md`](../../models/fsa_high_res/TESTING.md).
- **Scenario primitives + sim-est consistency tests:**
  [Python-Model-Scenario-Simulation](https://github.com/ajaytalati/Python-Model-Scenario-Simulation).
- **End-to-end SMC² estimation (private):**
  the `smc2-blackjax-rolling` repo's
  `drivers/fsa_high_res_rolling.py`. Consumes packaged scenario
  artifacts via `--scenario-artifact <dir>`.

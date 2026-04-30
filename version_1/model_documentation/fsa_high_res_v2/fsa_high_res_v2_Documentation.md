# The FSA High-Res v2 Model — Specification
## 3-State Banister-Coupled FSA SDE with G1 Reparametrization, Stuart-Landau Cubic, and sqrt-CIR Diffusions

**Version:** 0.2 (post Stage G1, Stage M+N+J5)
**Source:** [version_1/models/fsa_high_res_v2/](../../models/fsa_high_res_v2/)
**Diff from v1:** [CHANGELOG_from_v1.md](CHANGELOG_from_v1.md)

---

## 1. What this model is for

`fsa_high_res_v2` is the **Banister-coupled, G1-reparametrized**
successor to [`fsa_high_res`](../fsa_high_res/fsa_high_res_Documentation.md).
The latent dynamics keep the same three-state structure — a
Jacobi/CIR/Landau triple representing fitness $B$, fatigue $F$, and
amplitude $A$ — but with three substantive changes:

1. **Banister-style coupling**: $B$ and $F$ are driven by training
   intensity $\Phi(t)$ via $\kappa_B$ and $\kappa_F$ gains, and the
   Stuart-Landau drift on $A$ depends on $B$ (positive) and $F$
   (negative quadratic — pushing $A$ down when fatigue overshoots).
2. **G1 reparametrization** anchored at an operating point
   $(A_\text{typ}, F_\text{typ})$ to break three FIM rank-deficient
   pairs that caused posterior drift in v1's rolling-window SMC²
   driver.
3. **sqrt-CIR diffusions** for both $F$ and $A$ (CIR-style).
   $\sigma_F$ raised from $0.005$ to $0.012$ to widen the predictive
   envelope on the strain channel.

This is the model the **closed-loop SMC²-MPC** framework in
[`python-smc2-filtering-control`](https://github.com/ajaytalati/python-smc2-filtering-control)
is calibrated against. Its proof-of-principle results (post Stage M+N+J5):
T=14 closed-loop ratio 1.50, T=28 closed-loop ratio 1.60, T=42
closed-loop ratio 1.72 — all with F-violation 0.0% and parameter
identification coverage at 100% across the rolling windows.

---

## 2. States and parameters at a glance

### 2.1 The three-state latent dynamical system

| Symbol | Meaning | Role | Range | Timescale |
|:---:|:---|:---|:---:|:---:|
| $B$ | fitness (chronic) | stochastic (Jacobi) | $[0, 1]$ | $\tau_B = 42$ d |
| $F$ | fatigue (acute) | stochastic (CIR) | $[0, \infty)$ | $\tau_F^\text{eff} = 6.36$ d |
| $A$ | amplitude (Stuart-Landau) | stochastic (CIR) | $[0, \infty)$ | minutes-hours |

Time is in **days**. Bin width $\Delta t = 1/24$ day = 1 hour
(coarsened from v1's 15 min — no measurable accuracy loss for FSA-v2
latents at this resolution; see source repo's
`GPU_TUNING_RTX5090.md §G`).

The G1 operating point at which the reparametrization is anchored:
$A_\text{typ} = 0.10$, $F_\text{typ} = 0.20$, $\Phi_\text{typ} = 1.0$.

### 2.2 Parameter blocks — 30 estimable + 6 frozen = 36

| Block | Count | Parameters |
|:---|:---:|:---|
| Banister dynamics | 11 | $\tau_B,\, \tau_F,\, \kappa_B,\, \kappa_F,\, \varepsilon_A,\, \lambda_A,\, \mu_0,\, \mu_B,\, \mu_F,\, \mu_{FF},\, \eta$ |
| HR (Gaussian, sleep-gated) | 5 | $\text{HR}_\text{base},\, \kappa^\text{HR}_B,\, \alpha^\text{HR}_A,\, \beta^\text{HR}_C,\, \sigma_\text{HR}$ |
| Sleep (Bernoulli) | 3 | $k_C,\, k_A,\, \tilde c$ |
| Stress (Gaussian, wake-gated) | 5 | $S_\text{base},\, k_F,\, k^S_A,\, \beta^S_C,\, \sigma_S$ |
| Steps (log-Gaussian, wake-gated) | 6 | $\mu_\text{step0},\, \beta^\text{st}_B,\, \beta^\text{st}_F,\, \beta^\text{st}_A,\, \beta^\text{st}_C,\, \sigma_\text{st}$ |
| **Frozen** | 6 | $\sigma_B = 0.010,\, \sigma_F = 0.012,\, \sigma_A = 0.020,\, \varepsilon_A^B = \varepsilon_A^A = 10^{-4},\, \phi_C = 0$ |

Compared to v1: `kappa_B` from the v1 obs block was renamed
`kappa_B_HR` here to disambiguate from the (also-named-`kappa_B`)
Banister B-gain in the dynamics block. Diffusion scales $\sigma_*$
are now **frozen** at the canonical values rather than being
estimable — see §7 footgun.

---

## 3. The latent SDE (G1-reparametrized)

The drift formulas are mathematically equivalent to the un-reparametrized
v2 spec at the operating point, but rotated into a (strongly-identified,
weakly-identified-residual) decomposition. At $A = A_\text{typ}$ and
$F = F_\text{typ}$, the residual factors collapse to 1 and only the
strongly-identified parameters appear in the dynamics.

$$
\begin{aligned}
dB &= \Big[\kappa_B \cdot \tfrac{1 + \varepsilon_A A}{1 + \varepsilon_A A_\text{typ}} \cdot \Phi(t) - \tfrac{B}{\tau_B}\Big]\,dt + \sigma_B\sqrt{B(1 - B)}\,dW_B \\
dF &= \Big[\kappa_F \cdot \Phi(t) - \tfrac{1 + \lambda_A A}{1 + \lambda_A A_\text{typ}} \cdot \tfrac{F}{\tau_F}\Big]\,dt + \sigma_F\sqrt{F}\,dW_F \\
dA &= \big[\mu(B, F)\,A - \eta A^3\big]\,dt + \sigma_A\sqrt{A}\,dW_A
\end{aligned}
$$

where the Stuart-Landau bifurcation parameter is now centred at
$F_\text{typ}$ via a curvature term:

$$
\mu(B, F) \;=\; \mu_0 + \mu_B B - \mu_F F - \mu_{FF}\,(F - F_\text{typ})^2
$$

### 3.1 G1 reparametrization summary

| Param | New meaning (effective at $A_\text{typ}, F_\text{typ}$) | New truth value |
|:---|:---|:---:|
| $\tau_F$ | $\tau_F^\text{eff} = \tau_F / (1 + \lambda_A A_\text{typ})$ | $7.0 / 1.1 = 6.36$ d |
| $\kappa_B$ | $\kappa_B^\text{eff} = \kappa_B \cdot (1 + \varepsilon_A A_\text{typ})$ | $0.012 \cdot 1.04 = 0.01248$ |
| $\mu_0$ | $\mu_0^\text{eff} = \mu_0 + \mu_{FF} F_\text{typ}^2$ | $0.02 + 0.016 = 0.036$ |
| $\mu_F$ | $\mu_F^\text{eff} = \mu_F + 2 F_\text{typ} \mu_{FF}$ (slope at $F_\text{typ}$) | $0.10 + 0.16 = 0.26$ |
| $\varepsilon_A$ | residual A-boost beyond $A_\text{typ}$ | $0.40$ (tightened σ_log $\to 0.05$) |
| $\lambda_A$ | residual A-coupling beyond $A_\text{typ}$ | $1.00$ (tightened σ_log $\to 0.05$) |
| $\mu_{FF}$ | residual curvature, centred at $F_\text{typ}$ | $0.40$ (tightened σ_log $\to 0.05$) |

Drift-parity unit test: `tests/test_g1_reparam.py` (in the source
repo) verifies that the reparametrized drift matches the original
v2 drift to floating-point precision at the operating point.

---

## 4. The observation model — 4 mixed-likelihood channels

Each channel has its own gating rule against the Bernoulli sleep
label $\ell_t$. Same channel structure as v1.

### 4.1 HR (Gaussian, sleep-gated, $\ell_t = 1$)

$$
\text{HR}_t \sim \mathcal{N}\!\left(\text{HR}_\text{base} - \kappa^\text{HR}_B B_t + \alpha^\text{HR}_A A_t + \beta^\text{HR}_C C(t),\ \sigma_\text{HR}^2\right)
$$

Defaults: $\text{HR}_\text{base}=62$, $\kappa^\text{HR}_B=12$
(vagal-tone fitness effect — note: distinct from the dynamics
$\kappa_B$ Banister gain), $\alpha^\text{HR}_A=3$,
$\beta^\text{HR}_C=-2.5$ (early-morning HR nadir),
$\sigma_\text{HR}=2$.

### 4.2 Sleep (Bernoulli, always observed)

$$
\ell_t \sim \text{Bernoulli}\!\left(\sigma\!\big(k_C C(t) + k_A A_t - \tilde c\big)\right)
$$

Defaults: $k_C=3$ (circadian dominates), $k_A=2$, $\tilde c = 0.5$.

### 4.3 Stress (Gaussian, wake-gated, $\ell_t = 0$)

$$
S_t \sim \mathcal{N}\!\left(S_\text{base} + k_F F_t - k^S_A A_t + \beta^S_C C(t),\ \sigma_S^2\right)
$$

Defaults: $S_\text{base}=30$, $k_F=20$, $k^S_A=8$, $\beta^S_C=-4$
(stress peaks at noon when $C \approx -1$), $\sigma_S=4$.

### 4.4 Steps (log-Gaussian, wake-gated, $\ell_t = 0$)

$$
\log(\text{steps}_t + 1) \sim \mathcal{N}\!\left(\mu_\text{step0} + \beta^\text{st}_B B_t - \beta^\text{st}_F F_t + \beta^\text{st}_A A_t + \beta^\text{st}_C C(t),\ \sigma_\text{st}^2\right)
$$

Defaults: $\mu_\text{step0}=5.5$ (~245 steps/h baseline),
$\beta^\text{st}_B=0.8$, $\beta^\text{st}_F=0.5$ (fatigue suppresses),
$\beta^\text{st}_A=0.3$, $\beta^\text{st}_C=-0.8$,
$\sigma_\text{st}=0.5$.

---

## 5. Exogenous schedules

### 5.1 Training intensity $\Phi(t)$ — daily piecewise + sub-daily burst

Daily-integrated $\Phi$ is set by the macrocycle generator (or the
MPC controller in closed-loop mode). The sub-daily expansion is a
Gamma($k=2$) shape over each wake window, peaking at $t = \tau$
post-wake (default $\tau = 3$ h → 10 am if waking at 7 am):

$$
\Phi_\text{sub}(t) \;\propto\; t \cdot e^{-t/\tau},\qquad t = h - \text{wake hour}
$$

See `_phi_burst.py:expand_daily_phi_to_subdaily_jax`.

### 5.2 Circadian $C(t)$

$C(t) = \cos(2\pi t + \phi_C)$ with $\phi_C = 0$ frozen (morning
chronotype). Computed on the **global time grid** and emitted as
an exogenous channel. **Do not recompute window-locally** in any
rolling-window estimator (see §7).

---

## 6. Parameter set — Set A v2 (Banister-coupled, recovery scenario)

`TRUTH_PARAMS` in `_dynamics.py` (the post-G1 reparametrized values).
Initial state $B_0=0.05$, $F_0=0.30$, $A_0=0.10$ (Stage-D init).

The closed-loop scenarios in the source repo use `seed=42` and the
default macrocycle, producing trajectory ranges (T=42 d):
$B \in [0.05, 0.4]$, $F \in [0.3, 0.55]$, $A \in [0.10, 0.30]$.
The MPC controller drives $\Phi$ to maximize mean $A$ subject to
$F < F_\text{max} = 0.75$.

---

## 7. Footguns

### 7.1 The C-phase footgun (inherited from v1)

`gen_C_channel` emits $C(t)$ on the **global time grid**. Estimators
and rolling-window drivers MUST consume this from the artifact /
exogenous broadcast rather than recomputing $C$ from window-local time.
Doing the latter restarts $C$ at $\cos(0)=+1$ at every window
boundary, producing a phase mismatch that biases all $\beta^*_C$
posterior means toward zero.

### 7.2 G1 anchor constants must match across files (NEW in v2)

The operating-point constants $A_\text{typ}=0.10$, $F_\text{typ}=0.20$,
$\Phi_\text{typ}=1.0$ are defined in `_dynamics.py` and consumed by
`estimation.py` and `control.py`. **They must match exactly across
all three files.** A drift mismatch silently produces wrong posteriors.

### 7.3 Frozen σ values must match between simulator and estimator (NEW in v2)

$\sigma_B=0.010$, $\sigma_F=0.012$, $\sigma_A=0.020$ are now frozen
constants rather than estimable parameters. The same six values
(plus $\varepsilon_A^B$, $\varepsilon_A^A$, $\phi_C$) appear in both
`simulation.py` (forward sim) and `estimation.py` (filter). If the
two ever diverge, the filter's posterior will drift away from the
truth without any acceptance-gate signal until the rolling-window
identifiability metric collapses. Recommend: extract these to a
single shared `_frozen_constants.py` module if you ever maintain
two divergent sets.

### 7.4 G1 reparametrization is only valid near the operating point

The drift formulas are mathematically equivalent to the
un-reparametrized v2 spec **at $A = A_\text{typ}$ and
$F = F_\text{typ}$**. Far from this operating point, the residual
factors $(1 + \varepsilon_A A) / (1 + \varepsilon_A A_\text{typ})$
and $(1 + \lambda_A A) / (1 + \lambda_A A_\text{typ})$ deviate from
1, and the Banister gains $\kappa_B$, $\tau_F$ are no longer the
"effective" values. For real-data extension to subjects whose
typical operating point differs substantially, re-anchor.

---

## 8. Where to go next

- **Code (this repo, safekeeping):**
  [`_dynamics.py`](../../models/fsa_high_res_v2/_dynamics.py),
  [`_phi_burst.py`](../../models/fsa_high_res_v2/_phi_burst.py),
  [`_plant.py`](../../models/fsa_high_res_v2/_plant.py),
  [`simulation.py`](../../models/fsa_high_res_v2/simulation.py),
  [`estimation.py`](../../models/fsa_high_res_v2/estimation.py),
  [`control.py`](../../models/fsa_high_res_v2/control.py).
- **Diff vs v1:**
  [`CHANGELOG_from_v1.md`](CHANGELOG_from_v1.md).
- **Closed-loop SMC²-MPC framework (where this model is run):**
  [`python-smc2-filtering-control`](https://github.com/ajaytalati/python-smc2-filtering-control)
  — the model files here are mirrored from
  `version_2/models/fsa_high_res/` of that repo.
- **GPU tuning notes** (RTX 5090 specifics, JAX-native kernel
  rewrite, FP32 vs FP64 trade-offs):
  [`python-smc2-filtering-control/.../GPU_TUNING_RTX5090.md`](https://github.com/ajaytalati/python-smc2-filtering-control/blob/master/version_2/outputs/fsa_high_res/GPU_TUNING_RTX5090.md).
- **v1 spec for comparison:**
  [`fsa_high_res_Documentation.md`](../fsa_high_res/fsa_high_res_Documentation.md).

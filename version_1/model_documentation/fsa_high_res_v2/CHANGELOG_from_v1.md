# CHANGELOG: `fsa_high_res` (v0.1) → `fsa_high_res_v2` (v0.2)

This document records the substantive differences between the original
`fsa_high_res` model and `fsa_high_res_v2`. The latent state structure
(B, F, A) and the four observation channels (HR, Sleep, Stress, Steps)
are unchanged. The differences are in the **drift dynamics**, the
**diffusion**, the **frozen-vs-estimable parameter set**, and the
**reparametrization for identifiability**.

For the full v2 specification see
[`fsa_high_res_v2_Documentation.md`](fsa_high_res_v2_Documentation.md).
For the v1 specification see
[`../fsa_high_res/fsa_high_res_Documentation.md`](../fsa_high_res/fsa_high_res_Documentation.md).

---

## 1. Banister-style coupling (drift restructuring)

**v1**: $B$ followed a Jacobi $T_B(t)$-tracking equation; $F$ was
driven by $\Phi$ with a $\lambda_B B + \lambda_A A$ multiplicative
gating in the relaxation term.

**v2**: $B$ and $F$ are both directly driven by $\Phi(t)$ via gain
parameters $\kappa_B$ and $\kappa_F$. The fitness-fatigue interaction
is now in the Stuart-Landau drift on $A$, where $\mu(B, F)$ rises
with $B$ and falls with $F$ (Banister-style):

| | v1 | v2 |
|:---|:---|:---|
| $dB$ drift | $(1 + \alpha_A A)/\tau_B \cdot (T_B(t) - B)$ | $\kappa_B \cdot (1 + \varepsilon_A A)/(1 + \varepsilon_A A_\text{typ}) \cdot \Phi(t) - B/\tau_B$ |
| $dF$ drift | $\Phi(t) - (1 + \lambda_B B + \lambda_A A)/\tau_F \cdot F$ | $\kappa_F \cdot \Phi(t) - (1 + \lambda_A A)/(1 + \lambda_A A_\text{typ}) \cdot F/\tau_F$ |
| $dA$ drift | $\mu(B,F)\,A - \eta A^3$ with $\mu = \mu_0 + \mu_B B - \mu_F F - \mu_{FF} F^2$ | same form, but $\mu = \mu_0 + \mu_B B - \mu_F F - \mu_{FF}\,(F - F_\text{typ})^2$ (curvature centred at $F_\text{typ}$) |

The v2 form is closer to the classical Banister fitness-fatigue model
in the literature; v1 was a $T_B$-tracking variant.

---

## 2. G1 reparametrization (identifiability)

The original v2 spec had **three FIM rank-deficient pairs** at 1-day
windows under near-constant $\Phi$ and near-constant $A \approx 0.1$:

| Pair | Identifiable combination |
|:---|:---|
| $(\kappa_B,\, \varepsilon_A)$ | only $\kappa_B (1 + \varepsilon_A A_\text{typ})$ |
| $(\mu_F,\, \mu_{FF})$ | only $\mu_F + 2 F_\text{typ} \mu_{FF}$ |
| $(\tau_F,\, \lambda_A)$ | only $(1 + \lambda_A A_\text{typ}) / \tau_F$ |

The bridge handoff in the rolling-window SMC² driver propagated
phantom information on these directions, causing posterior drift
across windows. Stage G1 rotated each pair into a (strongly-identified,
weakly-identified-residual) decomposition, **keeping the parameter
names unchanged** but redefining their meanings + adjusting the
truth values. Drift formulas are mathematically equivalent at the
operating point.

| Param | New meaning at $(A_\text{typ}, F_\text{typ})$ | Old truth | New truth |
|:---|:---|:---:|:---:|
| $\tau_F$ | $\tau_F^\text{eff} = \tau_F / (1 + \lambda_A A_\text{typ})$ | $7.0$ | $6.36$ |
| $\kappa_B$ | $\kappa_B^\text{eff} = \kappa_B (1 + \varepsilon_A A_\text{typ})$ | $0.012$ | $0.01248$ |
| $\mu_0$ | $\mu_0^\text{eff} = \mu_0 + \mu_{FF} F_\text{typ}^2$ | $0.020$ | $0.036$ |
| $\mu_F$ | $\mu_F^\text{eff} = \mu_F + 2 F_\text{typ} \mu_{FF}$ | $0.10$ | $0.26$ |
| $\varepsilon_A$, $\lambda_A$, $\mu_{FF}$ | residuals (now near-zero impact at typical op point) | (estimable) | (estimable, **tightened priors** σ_log $\to 0.05$) |

Operating-point constants: $A_\text{typ} = 0.10$,
$F_\text{typ} = 0.20$, $\Phi_\text{typ} = 1.0$.

---

## 3. sqrt-CIR diffusions (numerical stability)

**v1**: $\sigma_B = 0.01$, $\sigma_F = 0.005$, $\sigma_A = 0.02$.
Diffusion was already sqrt-CIR for $F$ and sqrt-Jacobi for $B$, but
$F$'s scale was small.

**v2**: $\sigma_B = 0.010$ (unchanged), $\sigma_F = 0.012$ (**raised
~2.4×**), $\sigma_A = 0.020$ (unchanged). The wider $\sigma_F$
prevents $F$ from collapsing to ~0 under prolonged low $\Phi$, and
keeps the sqrt-CIR diffusion well-conditioned (avoids the
$\sqrt{F} \to 0$ degeneracy).

All three diffusions now use sqrt-CIR / sqrt-Jacobi form
$\sigma \sqrt{\cdot}\,dW$ for state-dependence:

$$
\sigma_B \sqrt{B(1 - B)}\,dW_B,\quad \sigma_F \sqrt{F}\,dW_F,\quad \sigma_A \sqrt{A}\,dW_A
$$

---

## 4. Frozen-vs-estimable parameter set

| Parameter | v1 | v2 |
|:---|:---:|:---:|
| $\sigma_B$ | frozen $= 0.01$ | frozen $= 0.010$ |
| $\sigma_F$ | frozen $= 0.005$ | frozen $= 0.012$ |
| $\sigma_A$ | frozen $= 0.02$ | frozen $= 0.020$ |
| $\varepsilon_A^B,\, \varepsilon_A^A$ (boundary noise floors) | frozen $= 10^{-4}$ | frozen $= 10^{-4}$ |
| $\phi_C$ (circadian phase) | frozen $= 0$ | frozen $= 0$ |
| Total frozen | 6 | 6 |
| Total estimable | 29 | **30** |

v2 adds one estimable parameter: $\kappa_B^\text{HR}$ for the HR
observation channel was renamed (was confusingly called $\kappa_B$
in v1 even though the dynamics also has a $\kappa_B$ Banister gain).
The disambiguation prevents accidental cross-contamination of the
posterior between dynamics-side $\kappa_B$ and obs-side
$\kappa_B^\text{HR}$.

---

## 5. Tightened priors on residual identifiability params

Stage G1 reparametrization makes $\varepsilon_A$, $\lambda_A$,
$\mu_{FF}$ formally **residual** parameters — at the operating point
their effect on the drift is zero. The data therefore weakly
informs them at 1-day window scale. v2 tightens their log-normal
priors:

| Param | v1 prior $\sigma_\text{log}$ | v2 prior $\sigma_\text{log}$ |
|:---|:---:|:---:|
| $\varepsilon_A$ | $\sim 0.20$ (broad) | $0.05$ (tight) |
| $\lambda_A$ | $\sim 0.20$ | $0.05$ |
| $\mu_{FF}$ | $\sim 0.20$ | $0.05$ |
| $\mu_0$, $\mu_B$, $\mu_F$, $\eta$ | — | $0.20$ (unchanged width) |

Tighter priors on the residuals prevent the bridge handoff from
amplifying their posterior variance across windows.

---

## 6. Same: observation model + circadian forcing + macrocycle generator

**Unchanged from v1**:
- Four observation channels: HR (Gaussian, sleep-gated), Sleep
  (Bernoulli), Stress (Gaussian, wake-gated), Steps (log-Gaussian,
  wake-gated).
- All channel default values and gating rules.
- Circadian forcing $C(t) = \cos(2\pi t + \phi_C)$.
- The C-phase footgun (must consume $C$ from global time grid; do
  not recompute window-locally).
- Sub-daily $\Phi$ expansion as Gamma($k=2$) burst peaking at
  $\tau = 3$ h post-wake.

---

## 7. Time-grid resolution (operational, not model-spec)

v1 was specified at 15-min bins (96 bins/day). v2 is run at 1-hour
bins (24 bins/day) in the closed-loop SMC²-MPC framework.

The model itself supports either resolution — the `BINS_PER_DAY`
constant is set via the `FSA_STEP_MINUTES` environment variable at
import time. For closed-loop GPU work, h=1h is the right choice
(see source repo `GPU_TUNING_RTX5090.md §G`). For ground-truth
forward simulations on real data, h=15min remains valid.

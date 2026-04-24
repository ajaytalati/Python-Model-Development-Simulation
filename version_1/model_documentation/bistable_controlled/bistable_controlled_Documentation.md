# The Bistable-Controlled Model — Specification
## Controlled Double-Well SDE with Exogenous Barrier-Lowering Schedule

**Version:** 1.1
**Source:** [version_1/models/bistable_controlled/simulation.py](../../models/bistable_controlled/simulation.py)

---

## 1. What this model is for

`bistable_controlled` is a minimal 2-state SDE used to study **intervention-driven transitions between basins of attraction**. The health variable $x$ lives in a symmetric double-well potential; a slow control/barrier process $u$ tilts the potential so that one well becomes unstable. The exogenous schedule $u_\mathrm{target}(t)$ is piecewise-constant and specified by the intervention plan — $u$ itself tracks the schedule via Ornstein-Uhlenbeck mean-reversion.

This is the pedagogical "can we push the subject out of the pathological basin by applying a supercritical tilt" model. It is the simplest controlled-SDE in the framework and is intended as a testbed for estimator pipelines that will later be applied to the sleep-wake and SWAT models.

---

## 2. States and parameters at a glance

### 2.1 The two-state dynamical system

| Symbol | Meaning | Role | Range | Timescale |
|:---:|:---|:---|:---:|:---:|
| $x$ | health variable (double-well coordinate) | stochastic | $\mathbb{R}$ (valid $\in [-a, a]$) | ~1 h effective |
| $u$ | control / barrier-tilt state | stochastic (OU) | $\mathbb{R}$ | $\tau_u = 1/\gamma$ h |

### 2.2 The parameter list — 6 drift/diffusion + 2 initial conditions

| Parameter | Symbol | Typical value | Role |
|:---:|:---:|:---:|:---|
| `alpha` | $\alpha$ | 1.0 | double-well strength |
| `a` | $a$ | 1.0 | well separation: minima at $x = \pm a$ |
| `sigma_x` | $\sigma_x$ | 0.10 | $x$ diffusion temperature |
| `gamma` | $\gamma$ | 2.0 | $u$ mean-reversion rate (= $1/\tau_u$) |
| `sigma_u` | $\sigma_u$ | 0.05 | $u$ diffusion temperature |
| `sigma_obs` | $\sigma_\mathrm{obs}$ | 0.20 | observation noise std |
| `x_0` | — | $-1.0$ | initial $x$ (starts in negative/pathological well) |
| `u_0` | — | 0.0 | initial $u$ |

### 2.3 Exogenous schedule (not estimated)

| Symbol | Meaning | Typical value |
|:---:|:---|:---:|
| $T_\mathrm{intervention}$ | intervention onset time | 24 h |
| $u_\mathrm{on}$ | active tilt magnitude | 0.5 (supercritical, $> u_c$) |
| $T_\mathrm{total}$ | simulation horizon | 72 h |

---

## 3. The SDE system

$$
dx = \bigl[\alpha\, x\,(a^2 - x^2) + u\bigr]\,dt + \sqrt{2\sigma_x}\,dB_x
$$

$$
du = -\gamma\,\bigl(u - u_\mathrm{target}(t)\bigr)\,dt + \sqrt{2\sigma_u}\,dB_u
$$

with a two-phase piecewise-constant control schedule

$$
u_\mathrm{target}(t) = \begin{cases} 0 & t < T_\mathrm{intervention} \\ u_\mathrm{on} & t \geq T_\mathrm{intervention}. \end{cases}
$$

**Critical tilt** (saddle-node bifurcation of the deterministic $x$-drift):

$$
u_c = \frac{2\,\alpha\,a^3}{3\sqrt{3}} \approx 0.385 \quad \text{for } \alpha = a = 1.
$$

- $u > u_c$: landscape is **monostable** (only one stable fixed point of $x$-drift).
- $u < u_c$: landscape is **bistable** (three fixed points — two stable, one unstable).

Setting $u_\mathrm{on} > u_c$ makes the post-intervention regime supercritical so that the only attractor is the positive-$x$ (healthy) well.

---

## 4. Observation model

A single direct observation of $x$:

$$
y_k = x(t_k) + \varepsilon_k, \qquad \varepsilon_k \sim \mathcal{N}(0, \sigma_\mathrm{obs}^2).
$$

The schedule $u_\mathrm{target}(t)$ is also emitted as a deterministic channel so the intervention plan is preserved alongside the synthetic observations.

---

## 5. Parameter sets for simulation testing

Only **Set A** is defined. It is calibrated so that the barrier/noise ratio is moderate (peak/valley potential ratio $\approx 12.2$), $u$ tracks its target quickly ($\tau_u = 0.5$ h), and observation noise is moderate.

| Parameter | Set A |
|:---:|:---:|
| $\alpha$ | 1.0 |
| $a$ | 1.0 |
| $\sigma_x$ | 0.10 |
| $\gamma$ | 2.0 |
| $\sigma_u$ | 0.05 |
| $\sigma_\mathrm{obs}$ | 0.20 |
| $x_0$ | $-1.0$ |
| $u_0$ | 0.0 |

Schedule: $T_\mathrm{intervention} = 24$ h, $u_\mathrm{on} = 0.5$, $T_\mathrm{total} = 72$ h. Intended grid: $dt = 10$ min → 432 time steps.

**Expected behaviour.** During hours 0–24 (symmetric bistable, $u = 0$), $x$ stays near $-1$ with occasional noise-driven excursions. After hour 24, $u$ relaxes toward $0.5$ on timescale $\tau_u = 0.5$ h; once $u > u_c$, the $-1$ well vanishes and $x$ transitions to the positive well.

---

## 6. Pointers

- **Implementation:** [version_1/models/bistable_controlled/simulation.py](../../models/bistable_controlled/simulation.py)
- **Simulator CLI:** `python simulator/run_simulator.py --model models.bistable_controlled.simulation.BISTABLE_CTRL_MODEL --param-set A`
- **Plot:** [sim_plots.py](../../models/bistable_controlled/sim_plots.py)

---

*End of document.*

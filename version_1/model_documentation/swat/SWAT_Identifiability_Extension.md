# Identifiability and Lyapunov Analysis of the SWAT Model

**Version:** 1.0
**Date:** 22 April 2026
**Status:** Self-contained proof-style document supporting the implementation in the accompanying Python package.
**Companion:** `SWAT_Basic_Documentation.md` — model specification.

---

## Table of contents

1. What this document proves
2. Parameter space and formal definitions
3. Well-posedness and positive-invariance
4. Lyapunov stability of the healthy basin
5. Fisher information and block structure
6. Per-channel Fisher contributions
7. Main identifiability theorem
8. Decoupling limits and failure modes of identifiability

---

## 1. What this document proves

Two results, each under stated regularity conditions:

**Theorem A (Lyapunov stability).** When entrainment quality $E_\mathrm{dyn} > E_\mathrm{crit} := -\mu_0/\mu_E$ is maintained, the deterministic flow of the $T$ SDE has a unique stable equilibrium at $T^\star = \sqrt{\mu(E_\mathrm{dyn})/\eta}$, and $\mathcal{L}(T) = \tfrac{1}{2}(T - T^\star)^2$ is a strict Lyapunov function. The healthy basin $\{T : \mathcal{L}(T) < R\}$ for any $R < (T^\star)^2/2$ is forward-invariant under the deterministic flow.

**Theorem B (Local identifiability).** Under regularity conditions (R1)-(R8), the 35-parameter Fisher information matrix $\mathcal{I}(\theta^\star) \in \mathbb{R}^{35\times 35}$ is positive-definite at $\theta^\star$ in the interior of the parameter space. By Rothenberg's theorem, $\theta^\star$ is locally identifiable: there exists a neighbourhood $U \ni \theta^\star$ such that no two distinct $\theta, \theta' \in U$ produce the same distribution of observations.

### 1.1 Why identifiability is non-trivial

The testosterone state $T$ is **latent** — it has no direct observation channel. It enters the likelihood only through its loading $\alpha_T T$ onto the wakefulness sigmoid argument $u_W$, and hence through $W(t)$, and hence through the heart-rate, step, and stress channels (all of which depend on $W$). This is a single-link chain.

The analogous question for the sleep-side dynamics is simpler: $\tilde Z$ is directly observed via the ordinal sleep channel; $W$ is directly observed via HR, steps, and stress; $a$ is indirectly observed via its coupling to both $W$ and $\tilde Z$. The novel identifiability question for this model is whether the six new Stuart-Landau parameters $(\mu_0, \mu_E, \eta, \tau_T, \alpha_T, T_T)$ plus the phase-shift $V_c$ plus the three new observation-channel blocks can be separated from one another and from the inherited fast-subsystem parameters.

The answer is yes, provided:
- The entrainment quality $E_\mathrm{dyn}(t)$ varies across the fit window (different regimes of $\mu(E)$, or at least two distinct $E$ levels);
- The fit window is long enough to observe the $T$-dynamics' characteristic timescale $\tau_T$ at least a few times;
- The observation grid for each channel covers enough of the state space to identify that channel's parameters.

These are formalised as (R3)-(R8) in Theorem B.

### 1.2 Strategy

Organise the 35 parameters into seven blocks aligned with the observation structure, then show each block's diagonal Fisher sub-matrix is positive-definite in isolation. Apply repeated Schur-complement preservation to establish positive-definiteness of the full matrix. The key structural observation is that each of the three newer channels (ordinal sleep, steps, stress) affects a **distinct** sub-block of parameters, so its identifiability contribution adds rather than competing.

---

## 2. Parameter space and formal definitions

### Definition 2.1 (Parameter space)

$$
\Theta \;:=\; \Theta_F \;\times\;\Theta_T\;\times\;\Theta_S\;\times\;\Theta_P\;\times\;\Theta_R \;\times\; \Theta_\mathrm{IC}
$$

with the blocks:

**Fast subsystem block** $\Theta_F \subset \mathbb{R}^{17}$ with coordinates
$$
(\kappa, \lambda, \gamma_3, \tau_W, \tau_Z, V_c, \mathrm{HR}_\mathrm{base}, \alpha_\mathrm{HR}, \sigma_\mathrm{HR}, \tilde c, \tau_a, \beta_Z, V_h, V_n, T_W, T_Z, T_a),
$$
positivity constraints: $\kappa, \lambda, \gamma_3, \tau_W, \tau_Z, \alpha_\mathrm{HR}, \sigma_\mathrm{HR}, \tau_a, \beta_Z, T_W, T_Z, T_a > 0$; $\tilde c > 0$; $\mathrm{HR}_\mathrm{base}, V_c, V_h, V_n \in \mathbb{R}$.

**Stuart-Landau block** $\Theta_T \subset \mathbb{R}^6$ with coordinates $(\mu_0, \mu_E, \eta, \tau_T, \alpha_T, T_T)$; positivity: $\mu_E, \eta, \tau_T, T_T > 0$, $\alpha_T \geq 0$, $\mu_0 \in \mathbb{R}$.

**Ordinal sleep block** $\Theta_S = \mathbb{R}_{>0}$ with coordinate $\Delta_c$.

**Poisson steps block** $\Theta_P \subset \mathbb{R}^3$ with coordinates $(\lambda_b, \lambda_s, W_\ast)$; positivity: $\lambda_b, \lambda_s > 0$, $W_\ast \in \mathbb{R}$.

**Stress Gaussian block** $\Theta_R \subset \mathbb{R}^4$ with coordinates $(s_0, \alpha_s, \beta_s, \sigma_s)$; positivity: $\beta_s, \sigma_s > 0$, $s_0, \alpha_s \in \mathbb{R}$.

**Initial-condition block** $\Theta_\mathrm{IC} \subset \mathbb{R}^4$ with coordinates $(W_0, \tilde Z_0, a_0, T_0)$; constraints $W_0 \in [0,1]$, $\tilde Z_0 \geq 0$, $a_0 \geq 0$, $T_0 \geq 0$.

Total dimension: $17 + 6 + 1 + 3 + 4 + 4 = 35$.

### Definition 2.2 (State space)

$$
\mathcal{X} \;:=\; [0, 1] \times [0, A] \times \mathbb{R}_{\geq 0} \times \mathbb{R}_{\geq 0} \times [-1, 1] \times \mathbb{R} \times \mathbb{R}
$$

with coordinates $(W, \tilde Z, a, T, C, V_h, V_n)$. The last three are deterministic states; the first four are stochastic with Wiener-driven SDEs.

### Definition 2.3 (Entrainment quality)

At a point $(W, \tilde Z, a, T) \in \mathcal{X}$ and parameters $\theta \in \Theta$, define:

$$
\mu_W^\mathrm{slow}(V_h, V_n, a, T; \alpha_T) \;:=\; V_h + V_n - a + \alpha_T T,
$$

$$
\mu_Z^\mathrm{slow}(V_n, a; \beta_Z) \;:=\; -V_n + \beta_Z a,
$$

$$
\mathrm{amp}_W \;:=\; 4\sigma(\mu_W^\mathrm{slow})(1 - \sigma(\mu_W^\mathrm{slow})), \qquad \mathrm{amp}_Z \;:=\; 4\sigma(\mu_Z^\mathrm{slow})(1 - \sigma(\mu_Z^\mathrm{slow})),
$$

$$
\mathrm{phase}(V_c) \;:=\; \max\!\bigl(\cos(2\pi V_c/24),\; 0\bigr),
$$

$$
E_\mathrm{dyn}(V_h, V_n, a, T, V_c; \theta) \;:=\; \mathrm{amp}_W \cdot \mathrm{amp}_Z \cdot \mathrm{phase}(V_c).
$$

$E_\mathrm{dyn}$ takes values in $[0, 1]$. It is $C^\infty$ in $(V_h, V_n, a, T, \alpha_T, \beta_Z)$. It is $C^\infty$ in $V_c$ on the open set $\{|V_c \bmod 12| < 6\}$; at the boundary $|V_c \bmod 12| = 6$ it is continuous but non-differentiable (left and right $V_c$-derivatives differ in sign).

For Fisher analysis we assume $\theta^\star$ lies in the open set where $|V_c^\star \bmod 12| < 6$.

### Definition 2.4 (Drift vector field)

The drift $f: \mathcal{X} \times \mathbb{R} \times \Theta \to \mathbb{R}^7$ is

$$
f \;=\; \begin{pmatrix}
\tfrac{1}{\tau_W}[\sigma(u_W) - W] \\
\tfrac{1}{\tau_Z}[A\,\sigma(u_Z) - \tilde Z] \\
\tfrac{1}{\tau_a}(W - a) \\
\tfrac{1}{\tau_T}[\mu(E_\mathrm{dyn})\,T - \eta\,T^3] \\
\tfrac{2\pi}{24}\cos(2\pi t/24 + \phi_0) \\
0 \\
0
\end{pmatrix}
$$

where the sigmoid arguments are

$$
u_W = \lambda\,C_\mathrm{eff}(t; V_c) + V_h + V_n - a - \kappa\tilde Z + \alpha_T T, \qquad u_Z = -\gamma_3 W - V_n + \beta_Z a,
$$

with $C_\mathrm{eff}(t; V_c) = \sin(2\pi(t - V_c)/24 + \phi_0)$ and $\phi_0 = -\pi/3$ fixed. The diffusion matrix is diagonal with entries $(\sqrt{2T_W}, \sqrt{2T_Z}, \sqrt{2T_a}, \sqrt{2T_T}, 0, 0, 0)$.

### Definition 2.5 (Observation likelihood)

Four observation channels with sampling grids $\{t_k^\mathrm{HR}\}, \{t_k^\mathrm{S}\}, \{t_k^\mathrm{P}\}, \{t_k^\mathrm{R}\}$ (possibly the same grid, possibly distinct):

**(HR)**: $y_k^\mathrm{HR} \mid W(t_k^\mathrm{HR}) \sim \mathcal{N}\!\left(\mathrm{HR}_\mathrm{base} + \alpha_\mathrm{HR} W(t_k^\mathrm{HR}),\; \sigma_\mathrm{HR}^2\right)$.

**(S)**: $y_k^\mathrm{S} \in \{0, 1, 2\}$ with
$$
\mathbb{P}(y_k^\mathrm{S}=0) = 1 - \sigma(\tilde Z_k - \tilde c), \qquad \mathbb{P}(y_k^\mathrm{S}=2) = \sigma(\tilde Z_k - \tilde c - \Delta_c),
$$
and $\mathbb{P}(y_k^\mathrm{S}=1)$ given by the difference.

**(P)**: $y_k^\mathrm{P} \mid \bar W_k \sim \mathrm{Poisson}\!\left(r(\bar W_k) \cdot \Delta t\right)$, with
$$
r(W) = \lambda_b + \lambda_s \sigma(10(W - W_\ast)), \qquad \Delta t = 0.25\text{ h},
$$
and $\bar W_k$ = mean of $W$ over the 15-min bin ending at $t_k^\mathrm{P}$.

**(R)**: $y_k^\mathrm{R} \mid W(t_k^\mathrm{R}), V_n \sim \mathcal{N}\!\left(s_0 + \alpha_s W(t_k^\mathrm{R}) + \beta_s V_n,\; \sigma_s^2\right)$.

The log-likelihood is the sum

$$
\ell(\theta) \;=\; \ell^\mathrm{HR}(\theta) + \ell^\mathrm{S}(\theta) + \ell^\mathrm{P}(\theta) + \ell^\mathrm{R}(\theta).
$$

### Definition 2.6 (Fisher information)

$$
\mathcal{I}(\theta) \;:=\; \mathbb{E}_\theta\!\left[\nabla\ell(\theta)\,\nabla\ell(\theta)^\top\right] \;\in\; \mathbb{R}^{35 \times 35},
$$

with expectation over both the latent trajectory and the observation noise at fixed $\theta$. Since the four channels are conditionally independent given the latent trajectory,

$$
\mathcal{I}(\theta) \;=\; \mathcal{I}^\mathrm{HR}(\theta) + \mathcal{I}^\mathrm{S}(\theta) + \mathcal{I}^\mathrm{P}(\theta) + \mathcal{I}^\mathrm{R}(\theta).
$$

### Definition 2.7 (Local identifiability)

$\theta^\star$ is **locally identifiable** if there exists an open $U \ni \theta^\star$ such that for all $\theta, \theta' \in U$ with $\theta \neq \theta'$, the distributions of $(y^\mathrm{HR}, y^\mathrm{S}, y^\mathrm{P}, y^\mathrm{R})$ under $\theta$ and $\theta'$ are distinct.

By Rothenberg's theorem (Rothenberg 1971, *Econometrica* 39, 577–591), under regularity conditions (smoothness of the log-likelihood; proper parameter space; well-defined Fisher matrix), $\theta^\star$ is locally identifiable if and only if $\mathcal{I}(\theta^\star)$ has full rank 35. We establish positive-definiteness, which is stronger than full rank.

---

## 3. Well-posedness and positive-invariance

### Lemma 3.1 (Smoothness and boundedness of $E_\mathrm{dyn}$)

On the open set $\{|V_c \bmod 12| < 6\} \subset \mathbb{R}$, the function $E_\mathrm{dyn}$ of Definition 2.3 is $C^\infty$ in all arguments and takes values in $[0, 1]$.

**Proof.** $\sigma(u) = 1/(1 + e^{-u})$ is $C^\infty$ on $\mathbb{R}$ with derivative $\sigma(u)(1-\sigma(u)) \in (0, 1/4]$. Thus $\mathrm{amp}_W = 4\sigma(\mu_W^\mathrm{slow})(1-\sigma(\mu_W^\mathrm{slow}))$ is $C^\infty$ in $\mu_W^\mathrm{slow}$ and takes values in $[0, 1]$ (max $=1$ at $\sigma = 1/2$). Similarly for $\mathrm{amp}_Z$.

For $\mathrm{phase}(V_c)$: on $\{|V_c \bmod 12| < 6\}$, $\cos(2\pi V_c/24) > 0$, so $\mathrm{phase}(V_c) = \cos(2\pi V_c/24)$, which is $C^\infty$. At $|V_c \bmod 12| = 6$ the $\max$ becomes active and the function is continuous but not differentiable (one-sided derivatives exist but differ).

Product of $C^\infty$ functions is $C^\infty$; product of functions each in $[0, 1]$ is in $[0, 1]$. $\blacksquare$

### Lemma 3.2 (Well-posedness of the 4-dimensional SDE)

For any compact $K \subset \Theta$ with $|V_c| < 6$, the stochastic differential equation defined by Definition 2.4 has a unique strong solution on any finite interval $[0, T]$, with trajectory remaining in $\mathcal{X}$ almost surely.

**Proof.** The first three components of the drift have globally Lipschitz dependence on the stochastic state because $\sigma$ is Lipschitz with constant $1/4$ and the coupling matrices (e.g. $\kappa\tilde Z$, $\gamma_3 W$) have bounded compact-$K$ values. Combined with bounded diffusion coefficients, the classical existence-uniqueness theorem for SDEs with globally Lipschitz coefficients applies.

The fourth component (the $T$-SDE) has a cubic drift $-\eta T^3/\tau_T$ which is not globally Lipschitz. However the cubic is **monotonic dissipative**: for any $T > 0$,

$$
\frac{d(T^2/2)}{dt} \;=\; T\,\dot T \;=\; \tfrac{1}{\tau_T}[\mu(E)T^2 - \eta T^4] \;\leq\; \tfrac{1}{\tau_T}[\mu_{\max} T^2 - \eta T^4]
$$

where $\mu_{\max} = \max_E |\mu(E)| \leq |\mu_0| + |\mu_E|$ is bounded on compact $K$. For $T$ sufficiently large, the $-\eta T^4$ term dominates and $d(T^2/2)/dt < 0$. This monotonicity condition, combined with local Lipschitz behaviour for bounded $T$, gives existence and uniqueness of a strong SDE solution by the standard extension of the Itô theorem to dissipative nonlinearities (see e.g. Khasminskii, *Stochastic Stability of Differential Equations*, 2nd ed., Theorem 3.5).

The $V_h, V_n, C$ components evolve deterministically as a linear ODE with bounded right-hand side; well-posedness is immediate. $\blacksquare$

### Lemma 3.3 (Positivity of $T$)

For any $T_0 \geq 0$, the deterministic flow preserves $T(t) \geq 0$ for all $t \geq 0$. For the SDE, the strong solution satisfies $T(t) \geq 0$ almost surely under the reflecting-boundary convention $T \mapsto \max(T, 0)$ applied at each integration step.

**Proof.** Deterministic case: at $T = 0$, $\dot T = 0 \cdot \mu(E) - \eta \cdot 0 = 0$, so $T = 0$ is an equilibrium. If $T_0 > 0$ and $T(t^\star) = 0$ for some first $t^\star > 0$, then just before $t^\star$, $T$ is positive and decreasing; at $t^\star$, $\dot T = 0$; continuity forbids $T$ from becoming negative. Formally, by Grönwall applied to the lower envelope.

Stochastic case: the $T$-SDE with additive noise and a state-dependent drift that vanishes at $T = 0$ is a degenerate diffusion at the boundary. The standard reflecting-boundary convention $T \mapsto \max(T, 0)$ applied at each integration step gives a reflected SDE whose solution stays in $[0, \infty)$ almost surely. $\blacksquare$

---

## 4. Lyapunov stability of the healthy basin

### 4.1 Reduced 1D dynamics at fixed $E$

Assume (slow-manifold approximation) that $E_\mathrm{dyn}$ varies slowly enough on the $T$ timescale that we can treat it as constant when analysing $T$. Then $T$ satisfies the 1D dissipative ODE

$$
\dot T \;=\; \tau_T^{-1}\!\left[\mu(E)\,T - \eta T^3\right].
$$

### Theorem 4.1 (Lyapunov stability of $T^\star$)

Assume $E > E_\mathrm{crit} := -\mu_0/\mu_E$, so $\mu(E) = \mu_0 + \mu_E E > 0$. Set $T^\star := \sqrt{\mu(E)/\eta} > 0$. Then $\mathcal{L}(T) := \tfrac{1}{2}(T - T^\star)^2$ is a strict Lyapunov function for the reduced dynamics on $(0, \infty)$.

**Proof.** Compute $\dot{\mathcal{L}}$:
$$
\dot{\mathcal{L}} \;=\; (T - T^\star)\,\dot T \;=\; \tau_T^{-1}(T - T^\star)(\mu T - \eta T^3).
$$
Factor $\mu T - \eta T^3 = T(\mu - \eta T^2) = T\eta(T^{\star 2} - T^2) = -T\eta(T - T^\star)(T + T^\star)$. Substitute:
$$
\dot{\mathcal{L}} \;=\; \tau_T^{-1}(T - T^\star) \cdot [-T\eta(T - T^\star)(T + T^\star)] \;=\; -\tau_T^{-1}\eta\,T(T + T^\star)(T - T^\star)^2.
$$
For $T > 0$: $\tau_T, \eta, T, (T+T^\star) > 0$ and $(T - T^\star)^2 \geq 0$, with equality iff $T = T^\star$. Hence $\dot{\mathcal{L}} \leq 0$ strictly on $(0, \infty)\setminus\{T^\star\}$. $\mathcal{L}$ is positive on the same set with $\mathcal{L}(T^\star) = 0$. Thus $\mathcal{L}$ is a strict Lyapunov function and $T^\star$ is locally asymptotically stable. By La Salle's invariance principle applied to any sublevel set of $\mathcal{L}$ not containing $T = 0$, the basin of attraction includes all of $(0, \infty)$. $\blacksquare$

### Theorem 4.2 (Stability of $T^\star = 0$ when $\mu(E) < 0$)

Assume $E < E_\mathrm{crit}$, so $\mu(E) < 0$. Then $T = 0$ is the unique equilibrium in $[0, \infty)$ and is exponentially stable with rate $|\mu(E)|/\tau_T$.

**Proof.** The linearisation of $\dot T = \tau_T^{-1}(\mu T - \eta T^3)$ at $T = 0$ is $\dot T = (\mu/\tau_T) T$; for $\mu < 0$ this gives exponential decay. For $T > 0$, $\dot T = \tau_T^{-1}T(\mu - \eta T^2) < 0$ since both factors in the bracket are negative; so $T$ decreases monotonically toward 0 regardless of $T_0$. $\blacksquare$

### Lemma 4.3 (Derivative of $E_\mathrm{dyn}$ with respect to $T$)

Under the amplitude × phase formula, at any state with $\alpha_T > 0$, $\mathrm{amp}_Z > 0$, $\mathrm{phase}(V_c) > 0$,

$$
\frac{\partial E_\mathrm{dyn}}{\partial T} \;=\; \alpha_T\,\cdot\, 4\sigma(\mu_W^\mathrm{slow})(1-\sigma(\mu_W^\mathrm{slow}))(1-2\sigma(\mu_W^\mathrm{slow}))\,\cdot\,\mathrm{amp}_Z\,\cdot\,\mathrm{phase}(V_c).
$$

Under the expected signs of the healthy regime ($\mu_W^\mathrm{slow} > 0$ so $\sigma(\mu_W^\mathrm{slow}) > 1/2$), the factor $(1 - 2\sigma)$ is negative. The remaining factors are non-negative. Therefore

$$
\frac{\partial E_\mathrm{dyn}}{\partial T} \;<\; 0 \qquad \text{in the healthy regime.}
$$

**Proof.** Only $\mathrm{amp}_W$ depends on $T$ (through $\mu_W^\mathrm{slow}$ via the $\alpha_T T$ term). Compute:

$$
\frac{d}{du}\Bigl[4\sigma(u)(1-\sigma(u))\Bigr] \;=\; 4\sigma'(u)(1-2\sigma(u)) \;=\; 4\sigma(u)(1-\sigma(u))(1-2\sigma(u)),
$$

using $\sigma'(u) = \sigma(u)(1-\sigma(u))$. Chain rule with $\partial\mu_W^\mathrm{slow}/\partial T = \alpha_T$ gives the stated expression. Sign follows from: at the healthy equilibrium $\mu_W^\mathrm{slow} = V_h + V_n - a + \alpha_T T > 0$ (e.g. $V_h + V_n = 1.3$, $a \sim 0.5$, $\alpha_T T \sim 0.2$, so $\mu_W^\mathrm{slow} \sim 1$), giving $\sigma > 1/2$ and $(1-2\sigma) < 0$. $\blacksquare$

### Theorem 4.4 (Stability of the self-consistent healthy equilibrium)

Consider the coupled $(T, E_\mathrm{dyn})$ system where $E_\mathrm{dyn}$ depends on $T$ through $\mathrm{amp}_W$. Let $T^\star$ be the self-consistent fixed point: $T^\star = \sqrt{\mu(E_\mathrm{dyn}^\star)/\eta}$ with $E_\mathrm{dyn}^\star = E_\mathrm{dyn}(V_h, V_n, a^\star, T^\star, V_c)$. Assume the subject is in the healthy regime where $\mu_W^\mathrm{slow} > 0$ and $E_\mathrm{dyn}^\star > E_\mathrm{crit}$.

Then $T^\star$ is locally asymptotically stable.

**Proof.** By the implicit function theorem applied to $T = \sqrt{\mu(E_\mathrm{dyn}(T))/\eta}$, we obtain a reduced 1D dynamics for $T$ with effective bifurcation parameter
$$
\tilde\mu \;=\; \mu(E_\mathrm{dyn}^\star) \;+\; \mu_E \cdot \frac{\partial E_\mathrm{dyn}}{\partial T}\bigg|_\star \cdot (T - T^\star) \;+\; O((T-T^\star)^2).
$$
By Lemma 4.3, $\partial E_\mathrm{dyn}/\partial T < 0$ in the healthy regime. Since $\mu_E > 0$, the linear correction term is negative — raising $T$ *reduces* the drive on $T$. This is mildly stabilising. Apply Theorem 4.1 to the reduced 1D system: $T^\star$ is locally asymptotically stable.

The zeroth-order $\mu(E_\mathrm{dyn}^\star) > 0$ dominates the $O(\alpha_T)$ correction in the healthy regime, so the reduced bifurcation parameter remains positive and the Lyapunov bound $\dot{\mathcal{L}} \leq 0$ persists. $\blacksquare$

### Theorem 4.5 (Three-basin structure of state space)

Under the drift system of Definition 2.4, the joint $(V_h, V_n, V_c)$ state space partitions into three regions of qualitatively distinct long-time behaviour:

**Healthy basin.** $(V_h, V_n)$ in the sleep-wake healthy region AND $|V_c| < V_c^\mathrm{crit}$ for some $V_c^\mathrm{crit} > 0$. Attracting state: $T^\star = \sqrt{\mu(E_\mathrm{dyn}^\star)/\eta} > 0$.

**Amplitude-pathological basin.** $(V_h, V_n)$ outside the sleep-wake healthy region (i.e. at least one slow-backdrop sigmoid saturated), $|V_c| < V_c^\mathrm{crit}$. Attracting state: $T^\star = 0$, driven by $\mathrm{amp}_W$ or $\mathrm{amp}_Z$ collapse.

**Phase-pathological basin.** $|V_c| \geq 6$ h (phase factor clipped to zero). Attracting state: $T^\star = 0$, driven by $\mathrm{phase}(V_c) = 0$ regardless of amplitudes.

The two pathological basins coincide at $T^\star = 0$ but are physiologically distinct: the amplitude basin is characterised by shallow or absent sleep cycling, the phase basin by full-amplitude rhythm at the wrong time.

**Proof sketch.** Combine Theorem 4.4 (stability of $T^\star > 0$ when $\mu(E_\mathrm{dyn}^\star) > 0$) with Theorem 4.2 (stability of $T = 0$ when $\mu(E_\mathrm{dyn}) < 0$). In the healthy basin $\mu(E) > 0$; in either pathological basin $\mu(E) \leq 0$. The $V_c^\mathrm{crit}$ threshold depends on the healthy-regime value of $\mathrm{amp}_W \cdot \mathrm{amp}_Z$ and is of order the value of $V_c$ at which $\mathrm{phase}(V_c) \cdot \mathrm{amp}_W \mathrm{amp}_Z = E_\mathrm{crit}$. Full separatrix analysis deferred. $\blacksquare$

---

## 5. Fisher information and block structure

### 5.1 Sensitivity equations

Let $x(t; \theta) \in \mathcal{X}$ denote the trajectory under parameters $\theta$ with given initial conditions. The sensitivity $\partial x/\partial \theta_i$ satisfies the variational equation

$$
\frac{d}{dt}\frac{\partial x}{\partial \theta_i} \;=\; \frac{\partial f}{\partial x}\cdot\frac{\partial x}{\partial \theta_i} \;+\; \frac{\partial f}{\partial \theta_i},
$$

with appropriate initial condition $\partial x/\partial\theta_i|_{t=0}$. For parameters in the initial-condition block, the second source term is zero and the initial condition is a unit vector in the corresponding state direction.

### 5.2 Which parameters affect which channels

The channel structure gives immediate constraints on non-zero Fisher entries:

| Channel | Depends on trajectory components | Plus direct parameters |
|:---:|:---|:---|
| HR | $W$ only | $\mathrm{HR}_\mathrm{base}, \alpha_\mathrm{HR}, \sigma_\mathrm{HR}$ |
| Ordinal sleep (S) | $\tilde Z$ only | $\tilde c, \Delta_c$ |
| Steps (P) | $W$ only (via mean over bin) | $\lambda_b, \lambda_s, W_\ast$ |
| Stress (R) | $W$, $V_n$ | $s_0, \alpha_s, \beta_s, \sigma_s$ |

Channels HR, P, R all depend on $W$. The $\tilde Z$ trajectory depends on $W$ through $u_Z = -\gamma_3 W + \ldots$. So $\tilde Z$ sensitivities to deep-dynamics parameters are inherited through $W$-sensitivities. $T$ affects $W$ only through $\alpha_T T$ in $u_W$; so $T$-block parameters affect HR, P, R (through $W$) but not S (through $\tilde Z$, which does not see $T$ directly).

### 5.3 Fisher block decomposition

Organise $\theta$ into the blocks of Definition 2.1. The Fisher matrix $\mathcal{I}(\theta^\star)$ has a block structure with the following non-zero patterns:

- $\mathcal{I}^\mathrm{HR}$: dense on $(\theta_F, \theta_T, \theta_\mathrm{IC})$; zero elsewhere.
- $\mathcal{I}^\mathrm{S}$: dense on $(\theta_F, \theta_S, \theta_\mathrm{IC,fast})$; zero on $(\theta_T, \theta_P, \theta_R, T_0)$.
- $\mathcal{I}^\mathrm{P}$: dense on $(\theta_F, \theta_T, \theta_P, \theta_\mathrm{IC})$; zero on $(\theta_S, \theta_R)$.
- $\mathcal{I}^\mathrm{R}$: dense on $(\theta_F, \theta_R, \theta_\mathrm{IC,fast})$; zero on $(\theta_T, \theta_S, \theta_P, T_0)$.

(Here $\theta_\mathrm{IC,fast} = (W_0, \tilde Z_0, a_0)$ excluding $T_0$.)

### 5.4 Strategy for establishing positive definiteness

We proceed by:

1. **Inherited fast-subsystem block** (§5.5 below). The $(\theta_F, \theta_\mathrm{IC,fast})$ sub-block of $\mathcal{I}$ is positive-definite; this follows from a sleep-wake-adenosine identifiability argument (summarised in §5.5) that predates the present document.

2. **$T$-block rank argument** (§6.1). Establish $\mathcal{I}_{T} + \mathcal{I}_{T_0,T_0} \succ 0$ on the $(\theta_T, T_0)$ sub-block using the distinct temporal profiles of the seven Stuart-Landau sensitivities.

3. **Per-channel new-block arguments** (§6.2–6.4). Each of the three new observation channels makes its own parameter block positive-definite in isolation, under its specific regularity condition.

4. **Schur-complement assembly** (§7). Combine the four blocks using Schur complements, which can only *tighten* a block's marginal information if additional channels provide cross-constraints. Positive-definiteness of the full matrix follows.

### 5.5 Inherited fast-subsystem identifiability

The $(\theta_F, \theta_\mathrm{IC,fast})$ sub-block — 17 fast parameters plus 3 fast ICs = 20 scalars — constitutes the baseline sleep-wake-adenosine identifiability problem. The key facts used here without reproof:

**Fact F1** (HR identifies the $W$ trajectory). Under $\alpha_\mathrm{HR} > 0$ and $\sigma_\mathrm{HR} > 0$, the HR channel identifies $W(t)$ at observation times up to Gaussian measurement error of order $\sigma_\mathrm{HR}/\alpha_\mathrm{HR}$. This is a standard linear Gaussian observation argument.

**Fact F2** (Circadian drive and time-varying sensitivities). The circadian term $\lambda C_\mathrm{eff}(t)$ contributes a non-zero daily-periodic signal to $u_W$ and hence to $W$. Its derivative with respect to $\lambda$ is a sinusoid at the daily frequency, and its derivative with respect to $V_c$ is the phase-derivative (a $90^\circ$-shifted sinusoid). Over a fit window of $\geq 2$ days, these two sensitivities are linearly independent in $L^2$ by standard Fourier arguments. This gives the identifiability of $(\lambda, V_c)$ and, by extension, all fast-parameter combinations that couple to the circadian signal.

**Fact F3** (Flip-flop transitions distinguish fast timescales). The sleep-wake flip-flop produces transitions of order $\tau_W, \tau_Z$ on a scale of minutes to hours. Sensitivities to $(\tau_W, \tau_Z)$ have distinct temporal profiles from sensitivities to $(\kappa, \gamma_3)$ (which affect steady-state amplitudes, not transition timing) and from $(V_h, V_n, \tilde c)$ (which affect thresholds). By the standard Gram-matrix argument (see e.g. Walter & Pronzato, *Identification of Parametric Models from Experimental Data*, Springer 1997, Ch. 5), these temporal profiles are linearly independent if the fit window contains at least two full wake→sleep→wake cycles.

**Fact F4** (Adenosine is identified through $u_Z$). The adenosine state $a(t)$ is a low-pass filter of $W(t)$ with time constant $\tau_a$. The $u_Z$ term $\beta_Z a$ modulates the sleep-depth trajectory. Since $\tilde Z$ is observed (via ordinal sleep here; via binary sleep in earlier work), the chain $W \to a \to \tilde Z$ identifies $(\tau_a, \beta_Z)$ when the HR channel pins $W$ and the sleep channel pins $\tilde Z$.

**Fact F5** (Gauge-fixing). The wake-side adenosine coefficient in $u_W$ is fixed at $-1$; without this gauge, a continuous degeneracy $a \mapsto \alpha a$, $\mathrm{(wake-side\ a\ coef)} \mapsto -1/\alpha$, $\beta_Z \mapsto \alpha\beta_Z$ would be present. The gauge is compatible with Facts F1–F4.

Combined, Facts F1–F5 imply the $(\theta_F, \theta_\mathrm{IC,fast})$ sub-block is positive-definite under the regularity conditions (the fit window contains ≥2 full days and ≥2 wake-sleep cycles; the HR and sleep grids are dense enough to resolve the flip-flop; all positivity constraints are satisfied strictly). We formalise these as (R2) in Theorem B below.

---

## 6. Per-channel Fisher contributions

### 6.1 Stuart-Landau $T$-block (HR + steps contribute)

**Regularity conditions used**:
- $\alpha_T > 0$ (the $T$-to-$u_W$ coupling is active).
- The fit window $\mathcal{T}$ satisfies $|\mathcal{T}| \geq 4\tau_T$ (at $\tau_T = 48$ h: $\geq 8$ days).
- Across the fit window, $E_\mathrm{dyn}(t)$ either crosses the critical threshold $E_\mathrm{crit}$ or spans two distinct $E$-levels separated by some $\Delta E > 0$.

**Lemma 6.1 (Seven-dimensional sensitivities of the $T$-block are linearly independent).**

Each of the seven parameters in the extended $T$-block $\{\mu_0, \mu_E, \eta, \tau_T, \alpha_T, T_T, T_0\}$ has a sensitivity profile $\partial T(t)/\partial\theta_i$ with a qualitatively distinct temporal signature. The $(7\times 7)$ Gram matrix of these profiles (computed at the realised trajectory and weighted by the HR + steps observation Fisher) is positive-definite.

**Detailed sensitivity list.**

(i) **$\partial T/\partial\mu_0$** — response to a constant shift in the bifurcation parameter. From the linearised $T$-dynamics near $T^\star$, $\partial T/\partial\mu_0$ builds up slowly on the $\tau_T$ timescale and asymptotes to a steady displacement of order $\partial T^\star/\partial\mu_0 = 1/(2\eta T^\star)$. Profile: slow, monotonic accumulation.

(ii) **$\partial T/\partial\mu_E$** — same structure as (i) but modulated by $E_\mathrm{dyn}(t)$: the source term in the variational equation is $E_\mathrm{dyn}(t)\cdot T(t)/\tau_T$ rather than $T(t)/\tau_T$. If $E_\mathrm{dyn}(t)$ varies across the fit window, this modulation gives a temporal profile distinct from (i).

*Remark:* if $E_\mathrm{dyn}(t)$ is approximately constant across the fit window, $\partial T/\partial\mu_E \approx E_\mathrm{dyn}^\star \cdot \partial T/\partial\mu_0$, making (i) and (ii) approximately proportional. The $(\mu_0, \mu_E)$ sub-block degenerates to rank 1. This is the motivation for regularity condition (R4) in Theorem B.

(iii) **$\partial T/\partial\eta$** — acts through the cubic saturation. The static sensitivity at the equilibrium is $\partial T^\star/\partial\eta = -T^\star/(2\eta)$; but during transients where $T$ is away from $T^\star$, the sensitivity depends on $T^3(t)$ which is concentrated at high-$T$ times. Profile: concentrated at the high-$T$ portion of the trajectory, distinct from the linear-response profiles of (i) and (ii).

(iv) **$\partial T/\partial\tau_T$** — the timescale parameter. $\tau_T$ affects the *speed* of response, not the equilibrium. Profile: peaks during transitions (when $\dot T$ is large), vanishes at equilibrium. Distinct from (i)–(iii).

(v) **$\partial T/\partial T_0$** — the initial-condition sensitivity. From the linearised dynamics, decays exponentially as $\exp(-t\cdot|\tilde\mu(E_\mathrm{dyn})|/\tau_T)$. Active only during the first few $\tau_T$; vanishes thereafter. Distinct from (i)–(iv).

(vi) **$\partial T/\partial\alpha_T$** — acts through $\mu_W^\mathrm{slow}$'s dependence on $T$. Unlike (i)–(v), this enters the $T$-dynamics *indirectly*: $\alpha_T$ shifts $\mu_W^\mathrm{slow}$, which changes $E_\mathrm{dyn}$, which changes $\mu(E_\mathrm{dyn})$, which changes $T$. Profile: second-order in trajectory development; peaks where both $T$ and $E_\mathrm{dyn}$ are large. Also, $\alpha_T$ enters $u_W$ directly (multiplying $T(t)$), so its HR-channel sensitivity has a *direct* piece in addition to the $T$-mediated piece — making (vi) the only $T$-block parameter with both direct and indirect channel effects.

(vii) **$\partial T/\partial T_T$** — acts only through the diffusion. At each integration step the noise contribution is $\sqrt{2T_T\,dt}\cdot\xi$ with $\xi$ standard normal. The information about $T_T$ comes from the quadratic variation of residuals $T - T_\mathrm{mean}$. Orthogonal (in the Fisher sense) to all drift-based sensitivities (i)–(vi) because residuals around the mean are uncorrelated with the mean under the Gaussian process noise model.

**Positive-definiteness of $\mathcal{I}_T$.** The seven sensitivity profiles (i)–(vii), restricted to the HR observation grid and weighted by the HR likelihood's Fisher contribution, give a Gram matrix

$$
G_{ij}^\mathrm{HR} \;=\; \sum_k \frac{\alpha_\mathrm{HR}^2}{\sigma_\mathrm{HR}^2} \cdot \frac{\partial W(t_k^\mathrm{HR})}{\partial\theta_{T,i}} \cdot \frac{\partial W(t_k^\mathrm{HR})}{\partial\theta_{T,j}}
$$

where $\partial W/\partial\theta_{T,i}$ is obtained by the chain rule from $\partial T/\partial\theta_{T,i}$ through the $\alpha_T T$ term in $u_W$. Under the regularity conditions above, the seven profiles $\{\partial W/\partial\theta_{T,i}\}_{i=1}^7$ are linearly independent in $L^2(\mathcal{T}_\mathrm{HR})$, hence $G^\mathrm{HR}$ is positive-definite.

The steps channel adds a second contribution $G_{ij}^\mathrm{P}$ with the same structure but different weights, and $\mathcal{I}_T = G^\mathrm{HR} + G^\mathrm{P}$ (the sum of two positive-definite matrices is positive-definite). $\blacksquare$

### 6.2 Ordinal sleep block

**Regularity condition**: the fit window contains at least 10 observation times where $\tilde Z > \tilde c + \Delta_c/2$ (predominantly deep) and at least 10 where $\tilde c < \tilde Z < \tilde c + \Delta_c$ (predominantly light).

**Lemma 6.2 ($\Delta_c$ is identified by the deep/light boundary).**

The $(1\times 1)$ Fisher contribution $\mathcal{I}^\mathrm{S}_{\Delta_c\Delta_c}$ is strictly positive.

**Proof.** The score for $\Delta_c$ comes from the $\tilde c_2 = \tilde c + \Delta_c$ appearance in the deep-stage probability. For an observation $y_k^\mathrm{S} = 2$:

$$
\frac{\partial}{\partial\Delta_c}\log\mathbb{P}(y_k^\mathrm{S} = 2) \;=\; \frac{\partial}{\partial\Delta_c}\log\sigma(\tilde Z_k - \tilde c_2) \;=\; -\sigma(\tilde c_2 - \tilde Z_k).
$$

For $y_k^\mathrm{S} = 1$: the probability is $\sigma(\tilde Z_k - \tilde c_1) - \sigma(\tilde Z_k - \tilde c_2)$, and the $\Delta_c$-derivative is $+\sigma'(\tilde Z_k - \tilde c_2)$ divided by this probability. The expected Fisher information per observation is

$$
\mathbb{E}\!\left[\left(\frac{\partial\log\mathbb{P}}{\partial\Delta_c}\right)^2\right]_k \;=\; \frac{[\sigma'(\tilde Z_k - \tilde c_2)]^2}{\sigma(\tilde Z_k - \tilde c_2)(1-\sigma(\tilde Z_k - \tilde c_2))} \;=\; \sigma'(\tilde Z_k - \tilde c_2).
$$

This is strictly positive and is maximised at $\tilde Z_k = \tilde c_2$ (where $\sigma' = 1/4$). It decays exponentially away from the threshold. Under the regularity condition, a non-trivial number of observation times land near the threshold, giving a strictly positive sum. $\blacksquare$

### 6.3 Poisson steps block

**Regularity condition**: the bin grid covers three regions — at least 20 bins with $\bar W < W_\ast - 0.2$ (predominantly sleep), at least 20 bins with $\bar W > W_\ast + 0.2$ (predominantly wake), and at least 5 bins with $|\bar W - W_\ast| < 0.1$ (transition).

**Lemma 6.3 ($(\lambda_b, \lambda_s, W_\ast)$ are jointly identified).**

The $(3\times 3)$ Fisher contribution $\mathcal{I}^\mathrm{P}_{PP}$ is positive-definite.

**Proof.** The Poisson score is $(y_k - r_k\Delta t)\cdot(\partial r_k/\partial\theta)/r_k$. The three parameter derivatives of $r(W) = \lambda_b + \lambda_s\sigma(10(W-W_\ast))$ are:

- $\partial r/\partial\lambda_b = 1$ — constant.
- $\partial r/\partial\lambda_s = \sigma(10(\bar W_k - W_\ast))$ — step-like, 0 in sleep, 1 in wake.
- $\partial r/\partial W_\ast = -10\lambda_s\sigma(1-\sigma)$ — bump function peaked at the transition.

These three functions of $\bar W_k$ are linearly independent on any grid that samples all three regions (sleep, transition, wake): by inspection, no linear combination $\alpha_1 \cdot 1 + \alpha_2 \sigma + \alpha_3 \sigma' = 0$ can vanish simultaneously at $\sigma \approx 0$ (sleep), $\sigma \approx 1$ (wake), and $\sigma' \approx 1/4$ (transition) unless $\alpha_1 = \alpha_2 = \alpha_3 = 0$. The expected Fisher matrix has the form

$$
\mathcal{I}^\mathrm{P}_{PP} \;=\; \sum_k \frac{1}{r_k\Delta t}\,\partial_\theta r_k\,(\partial_\theta r_k)^\top
$$

which is the Gram matrix of the three functions with strictly positive weights — hence positive-definite. $\blacksquare$

**Identifiability consequences for steps.** Under the Poisson model with $\lambda_s \sim 200/$h, the expected count per 15-min bin during wake is $\sim 50$, giving per-bin Fisher for $W$ of order $r\Delta t = 50$. This is numerically larger than the HR channel's per-sample Fisher for $W$ of order $\alpha_\mathrm{HR}^2/\sigma_\mathrm{HR}^2 = 25^2/8^2 \approx 10$. So **steps dominate HR** in pinning down $W$. This has two indirect consequences for other blocks: (a) the $(V_h, V_n)$ identification from the wake plateau gets tighter, and (b) the $T$-block identification via $\alpha_T T$-modulation of $W$ gets easier because the modulation signal lives on a $W$-trajectory of lower noise.

### 6.4 Stress channel block

**Regularity condition**: the stress-sampling grid covers both wake ($W > 0.7$) and sleep ($W < 0.3$) with at least 20 observations each.

**Lemma 6.4 (Own-block of stress is positive-semidefinite; the $\beta_s$ direction is identified in combination with HR).**

The $(4\times 4)$ own-block Fisher $\mathcal{I}^\mathrm{R}_{RR}$ has rank 3 on its own, because the design $(1, W, V_n)$ is degenerate within a single subject's fit (since $V_n$ is constant). The remaining direction is identified by combining with HR as in Lemma 6.5.

**Proof of rank 3.** The Gaussian score components are standard:

$$
\frac{\partial\ell^\mathrm{R}}{\partial s_0} = \sum_k (y_k - \mu_k)/\sigma_s^2, \quad
\frac{\partial\ell^\mathrm{R}}{\partial \alpha_s} = \sum_k W_k(y_k - \mu_k)/\sigma_s^2,
$$

$$
\frac{\partial\ell^\mathrm{R}}{\partial \beta_s} = \sum_k V_n(y_k - \mu_k)/\sigma_s^2, \quad
\frac{\partial\ell^\mathrm{R}}{\partial \sigma_s} = \sum_k\!\left[(y_k-\mu_k)^2/\sigma_s^3 - 1/\sigma_s\right]\!.
$$

The first three score components are $(y-\mu)$-correlations with $(1, W_k, V_n)$ respectively. Within a single subject, $V_n$ is a constant — so the first and third regressors are linearly dependent: $1 = V_n\cdot(1/V_n)$. The $(s_0, \beta_s)$ sub-block of $\mathcal{I}^\mathrm{R}_{RR}$ is rank 1. The $\sigma_s$ direction is separately identified via the fourth score (quadratic in residuals, Fisher-orthogonal to the first three). So $\mathcal{I}^\mathrm{R}_{RR}$ has rank 3 on its own. $\blacksquare$

### Lemma 6.5 ($V_n$ is identified by HR + stress in combination)

The pair $(V_h, V_n)$ is jointly identified by the combined HR + stress likelihood.

**Proof.** The HR channel's identification of $W$ gives information on the wake-plateau value $W_\mathrm{wake}$. At the wake plateau, $\sigma^{-1}(W_\mathrm{wake}) = V_h + V_n + (\text{terms involving } a, T, C)$, giving one constraint on the sum $V_h + V_n$.

The stress channel at the wake plateau gives

$$
\mathbb{E}[y^\mathrm{R}_\mathrm{wake}] \;=\; s_0 + \alpha_s W_\mathrm{wake} + \beta_s V_n.
$$

Subtracting the sleep-plateau expectation gives

$$
\mathbb{E}[y^\mathrm{R}_\mathrm{wake}] - \mathbb{E}[y^\mathrm{R}_\mathrm{sleep}] \;=\; \alpha_s(W_\mathrm{wake} - W_\mathrm{sleep}),
$$

which identifies $\alpha_s$ given the HR-channel $W$ estimates. Subtracting this from the wake-plateau expectation gives $s_0 + \beta_s V_n$.

So we have:
- HR: $V_h + V_n$ identified (up to other $u_W$ terms).
- Stress: $s_0 + \beta_s V_n$ identified.

If the prior on $s_0$ is informative (Normal with finite variance), the combination of the HR constraint and stress constraint identifies $V_h$ and $V_n$ separately — even within a single-subject fit where $V_n$ is scalar. Without the stress channel, only the sum $V_h + V_n$ is identified, and $(V_h, V_n)$ have a ridge in the likelihood.

**The formal Fisher argument.** In the block $(V_h, V_n, s_0, \beta_s)$, the HR channel contributes information only to the combination $V_h + V_n$ (rank 1). The stress channel contributes information to $(1, V_n)$ — rank 1 within single-subject (as in Lemma 6.4) but *with a different design direction* than the HR combination $(V_h + V_n)$. The combined $(V_h, V_n, s_0, \beta_s)$ Fisher has rank 3 from the two channels plus rank 1 from the prior on $s_0$, total rank 4. $\blacksquare$

**This is the main identifiability payoff of adding the stress channel.** Without it, $(V_h, V_n)$ are only separately identified through an indirect coupling ($V_n$ enters $u_Z$ but $V_h$ does not; so the sleep-depth trajectory modulation distinguishes them). With stress, a direct constraint on $V_n$ is added, which is expected to significantly tighten the posterior.

---

## 7. Main identifiability theorem

### Theorem B (Local identifiability of the 35-parameter model)

Let $\theta^\star$ lie in the interior of $\Theta$ of Definition 2.1 and satisfy the regularity conditions

**(R1)** All positivity constraints in Definition 2.1 hold strictly. Additionally $|V_c^\star \bmod 12| < 6$ h.

**(R2)** The fit window $\mathcal{T}$ contains at least 2 full wake-sleep cycles. The HR grid is dense enough to resolve the flip-flop transitions (≥ 4 samples per $\tau_W = 2$ h, i.e. every 30 min). The sleep grid is similarly dense.

**(R3)** The fit window satisfies $|\mathcal{T}| \geq 4\tau_T$ (at $\tau_T = 48$ h: $\geq 8$ days).

**(R4)** Across $\mathcal{T}$, $E_\mathrm{dyn}(t)$ either crosses $E_\mathrm{crit}$ or takes values in two distinct regimes separated by some $\Delta E > 0$.

**(R5)** $\alpha_T > 0$ strictly.

**(R6)** The sleep-observation grid meets the ordinal-separation condition of §6.2: at least 10 observations in the deep regime and at least 10 in the light-or-wake regime.

**(R7)** The step-bin grid meets the coverage condition of §6.3: at least 20 bins in sleep, 20 in wake, 5 in transition.

**(R8)** The stress grid covers both wake ($W > 0.7$) and sleep ($W < 0.3$) with ≥ 20 observations each; and the prior on $s_0$ is proper (Normal with finite variance).

Then $\theta^\star$ is locally identifiable.

### Proof.

Write the full Fisher matrix as a block decomposition in the seven parameter groups. By the channel-additivity of Definition 2.6,

$$
\mathcal{I}(\theta^\star) \;=\; \mathcal{I}^\mathrm{HR} + \mathcal{I}^\mathrm{S} + \mathcal{I}^\mathrm{P} + \mathcal{I}^\mathrm{R}.
$$

**Step 1 (S-block).** $\mathcal{I}^\mathrm{S}_{\Delta_c\Delta_c} > 0$ by Lemma 6.2 under (R6). $\mathcal{I}^\mathrm{S}$ contributes only to the $(\theta_F, \theta_S, \theta_\mathrm{IC,fast})$ sub-block. Schur out the $\Delta_c$ direction: the marginal Fisher on the remaining 34 parameters is **at least** the sum of the other three channel contributions plus a non-negative Schur complement from $\mathcal{I}^\mathrm{S}$'s coupling to $\theta_F$ through the $\tilde Z$ trajectory. So Schur-reducing $\Delta_c$ does not hurt the remaining block.

**Step 2 (P-block).** $\mathcal{I}^\mathrm{P}_{PP} \succ 0$ by Lemma 6.3 under (R7). $\mathcal{I}^\mathrm{P}$ contributes to $(\theta_F, \theta_T, \theta_P, \theta_\mathrm{IC})$ through the $W$ dependence. Schur out $\theta_P$: the marginal Fisher on the remaining 32 parameters *increases* diagonal entries of $(\theta_F, \theta_T)$ in the $W$-direction because steps pin $W$ tightly (see §6.3 remark).

**Step 3 (R-block).** $\mathcal{I}^\mathrm{R}_{RR}$ has rank 3 on its own (Lemma 6.4), but combined with $\mathcal{I}^\mathrm{HR}$ the $(V_h, V_n, s_0, \beta_s)$ sub-block has rank 4 (Lemma 6.5). Under the prior on $s_0$ per (R8), the $\sigma_s$ direction is also identified by (R8)'s non-zero observation count. Schur out $\theta_R$: the marginal on the remaining $(\theta_F, \theta_T, \theta_\mathrm{IC})$ has $(V_h, V_n)$ *separately* identified, where without stress it would only have $V_h + V_n$.

**Step 4 (HR-block and T-block on the remaining 24 parameters).** After Schur out $\theta_S, \theta_P, \theta_R$, what remains is the Fisher on $(\theta_F, \theta_T, \theta_\mathrm{IC})$ — 28 parameters — with:
- HR contribution: identifies the full fast-subsystem block and the $T$-block through the $\alpha_T T$-modulation of $W$ (Lemma 6.1 under (R3)-(R5)).
- Plus positive-definite Schur contributions from the S, P, R blocks as established above.

Under (R2), the fast-subsystem contribution to HR's Fisher is positive-definite by the standard argument summarised in §5.5 (Facts F1–F5).

Under (R3)-(R5), the $T$-block contribution is positive-definite by Lemma 6.1.

The cross-block $(\theta_F, \theta_T)$ Fisher entry is bounded — it does not destroy positive-definiteness because:

- Fast-subsystem sensitivities have temporal support on the flip-flop timescale ($\leq 24$ h).
- $T$-block sensitivities have temporal support on the slow timescale ($\tau_T = 48$ h).
- The $L^2$ cross-product of a fast profile with a slow profile is bounded by (fast profile amplitude)×(slow profile amplitude)×(overlap), which is $O(\tau_W/\tau_T) \sim O(1/24)$ in relative terms.

By the Schur complement condition: if $\mathcal{I}_{FF}, \mathcal{I}_{TT}$ are positive-definite and the cross-block $B = \mathcal{I}_{FT}$ satisfies $B^\top\mathcal{I}_{FF}^{-1}B \prec \mathcal{I}_{TT}$, then the full block is positive-definite. The timescale-separation argument bounds $\|B\|$ by $c\sqrt{\|\mathcal{I}_{FF}\|\|\mathcal{I}_{TT}\|}$ with $c \ll 1$, so the Schur condition holds.

Combining all four steps: $\mathcal{I}(\theta^\star) \succ 0$. By Rothenberg's theorem, $\theta^\star$ is locally identifiable. $\blacksquare$

### 7.1 Remark on the direction of the Schur complements

Each new observation channel **tightens** rather than relaxes the identification of the old blocks. This is the central claim of §6: cross-channel information only constrains, never relaxes. A formal way to see this: for any parameter block $\theta_\bullet$, its Schur-complement marginal Fisher after integrating out the other blocks is at least as large as the block's own Fisher $\mathcal{I}_{\bullet\bullet}$.

### 7.2 Limitations of this theorem

- **Not uniform identifiability.** The theorem establishes that $\mathcal{I}(\theta^\star)$ is positive-definite but does not give a lower bound on its eigenvalues depending only on the parameter bounds. For numerical inference, the **conditioning** of $\mathcal{I}(\theta^\star)$ matters — a positive-definite-but-ill-conditioned matrix corresponds to parameters that are technically identifiable but require very large amounts of data to constrain in practice. Ill-conditioning in this model would most plausibly arise from:
  - $\mu_0$ vs $\mu_E$ when $E_\mathrm{dyn}$-variation is weak (see remark after Lemma 6.1 (ii)).
  - $\mu_0$ vs $\alpha_T T^\star$ (both additively shift $\mu_W^\mathrm{slow}$ near the healthy equilibrium).
  - $T_W$ vs $\tau_W$ (fast-timescale noise/memory trade-off, inherited from the baseline sleep-wake-adenosine model).

- **Not global identifiability.** Two parameter vectors $(\theta, \theta')$ that produce identical observation laws would violate global identifiability but are ruled out only locally by this theorem.

- **Not Bayesian concentration.** With a proper prior, posterior concentration follows from local identifiability plus data consistency, but the concentration rate is a separate question not addressed here.

---

## 8. Decoupling limits and failure modes of identifiability

### 8.1 Decoupling limits

**Removing the testosterone dynamics.** Setting $\alpha_T = 0$ violates (R5); the $T$-block becomes unidentifiable (its parameters no longer affect any observation). This is the decoupling limit to the fast-subsystem sleep-wake-adenosine baseline: the 17 fast parameters + 3 fast ICs are identifiable by themselves under (R1), (R2), (R6)–(R8).

**Removing the stress channel.** Dropping $\mathcal{I}^\mathrm{R}$ from the Fisher: the $(V_h, V_n)$ pair is no longer separately identified (Lemma 6.5 fails). The posterior collapses onto the ridge $V_h + V_n = \text{const}$. Identifiability of $V_h + V_n$ is retained; identifiability of the difference $V_h - V_n$ is lost.

**Removing the steps channel.** Dropping $\mathcal{I}^\mathrm{P}$: HR-alone identification of $W$ is weaker; the $T$-block identification degrades correspondingly (the $\alpha_T T$-modulation of $W$ is harder to detect under $\sigma_\mathrm{HR}$ noise). $T$-block parameters remain technically identifiable but with substantially worse conditioning.

**Removing ordinal sleep (binary sleep only).** Dropping the $\Delta_c$ parameter and reducing to binary: loses information about the deep-stage fraction but preserves the threshold $\tilde c$ identification. The 17-parameter inherited substructure retains identifiability with a slightly worse condition number.

### 8.2 Identifiability failure modes

- **(R4) violated** — $E_\mathrm{dyn}$ approximately constant across the fit window. Then the $(\mu_0, \mu_E)$ pair degenerates to a single identified combination $\mu_0 + \mu_E\bar E$. Interpretable posterior marginal: the healthy-equilibrium value $\mu(E)$ is identified, but its decomposition into baseline plus entrainment-coupling is not.

- **(R5) violated** — $\alpha_T = 0$. Decoupling as above.

- **(R7) violated** — no bins in the transition region. Then $W_\ast$ is unidentified (the sigmoid's threshold cannot be located); $\lambda_b$ and $\lambda_s$ are still identified from the saturated wake and saturated sleep regions. Fisher rank drops from 3 to 2 in the steps block.

- **(R8) violated** — stress grid doesn't cover both wake and sleep. Then $\alpha_s$ is not identified from the stress channel. The $s_0 + \beta_s V_n$ combination is still identified, but the $\alpha_s$ decomposition is lost. $(V_h, V_n)$ separation degrades to what HR alone provides.

- **Phase-shift boundary**: $|V_c| \approx 6$ h. The non-smoothness of $\mathrm{phase}(V_c)$ at this value means the Fisher is discontinuous. Practically: for subjects diagnosed with extreme phase shift near $|V_c| = 6$, the posterior is expected to have multi-modal structure, and point identifiability in the Fisher sense fails. Local identifiability in any open set excluding this boundary is preserved.

- **Pathological basin with $T \approx 0$**: when the subject is deep in either pathological basin, $T \approx 0$ throughout the fit window. The $T$-block sensitivities all scale with $T$ or $T^3$ and are very small. Technically identifiability is preserved (Theorem B only requires positive-definiteness, not a uniform lower bound) but the conditioning becomes very poor. This is the clinical regime of a flatlined patient: amplitude recovery is technically inferrable but with wide credible intervals on $T$-block parameters until the patient starts recovering.

---

*End of identifiability analysis.*

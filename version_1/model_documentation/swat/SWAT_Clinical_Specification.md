# The SWAT Model — Clinical and Biological Specification
## A testable model of how sleep quality and circadian alignment control testosterone pulsatility

**Version:** 1.0
**Date:** 22 April 2026
**Audience:** Clinicians, endocrinologists, sleep physicians, and human biologists who want to understand what this model assumes, what it predicts, and how its predictions can be tested against patient data.
**Technical companions** (not required reading): `SWAT_Basic_Documentation.md` (mathematical specification) and `SWAT_Identifiability_Extension.md` (statistical identifiability analysis).

---

## 1. What the model is, in one paragraph

SWAT is a mechanistic model of an individual patient, built to answer a single clinical question: **why is this man's testosterone low, and what should we do about it?** It takes a week or two of continuous heart-rate, sleep-stage, step-count and stress-score data from a consumer wearable, and returns a probabilistic diagnosis across four physiologically-distinct causes of hypogonadism: chronic stress, insomnia, circadian misalignment (shift work / jet lag), and unrecovered past pathology. The same machinery lets us forward-simulate what would happen if a proposed intervention (light therapy, sleep scheduling, stress management) were applied — *before* the patient actually tries it.

The model is not a black box. Every internal variable corresponds to a named physiological quantity (wakefulness, sleep depth, adenosine pressure, testosterone pulse amplitude, circadian drive, vitality reserve, chronic stress load, phase shift). Every parameter has an interpretation grounded in existing endocrinology or sleep science. The predictions the model makes are stated below as falsifiable hypotheses, each testable against patient data.

---

## 2. The core physiological claim

The central mechanistic claim of the model is this:

> **Testosterone pulsatility is sustained if and only if the sleep-wake rhythm is well-entrained to the circadian clock.** When a man sleeps and wakes in alignment with his internal body clock, and his sleep has proper architecture (light-deep-REM cycling), the HPG axis produces healthy pulsatile testosterone release. When the sleep-wake rhythm collapses — either because the man cannot sleep properly (insomnia/chronic stress) or because he sleeps at the wrong times (shift work/jet lag) — testosterone pulsatility flatlines. The flatline is not an "injury" to the testes; it is a *bifurcation* in the dynamics. Remove the cause and testosterone recovers on a timescale of days.

This claim is specific enough to be wrong. Competing claims are:

- **"Low testosterone is primary gonadal failure."** A different mechanism than SWAT proposes. SWAT would predict that in this case, fixing sleep and circadian alignment would not restore testosterone — providing a clean discriminator.
- **"Low testosterone just reflects age."** Also a different mechanism. SWAT would predict that age-matched men with identical sleep-wake entrainment have similar testosterone; discrepancy would refute either SWAT or the age claim.
- **"Low testosterone is nutritional."** A different mechanism. SWAT is agnostic about nutrition but would fail to predict recovery from a diet-based intervention without accompanying sleep/circadian change.

The model treats the testes as *responsive machinery* capable of producing pulsatile testosterone whenever the control signals (sleep-wake entrainment) allow it. The failure mode is in the control signal, not the machine.

---

## 3. What the model knows about a patient

The model observes four kinds of time-series from a Garmin-style wearable over 1–2 weeks:

- **Heart rate**, sampled every few minutes
- **Sleep stages** (wake / light+REM / deep) from the device's built-in sleep classifier
- **Step counts**, binned into 15-minute intervals
- **Garmin "stress score"**, an HRV-derived 0–100 metric

From these four channels it infers:

- **What the man's wake-sleep rhythm is actually doing** (moment-to-moment)
- **How much "vitality reserve" ($V_h$) he has** — a single number capturing his underlying capacity to maintain arousal
- **How much "chronic load" ($V_n$) he is carrying** — a single number capturing tonic stress / inflammation / allostatic burden
- **How phase-shifted he is** ($V_c$, in hours) — how many hours off he is from a morning-aligned circadian baseline
- **What his testosterone pulsatility amplitude is** — and whether it is stable, collapsing, or recovering
- **The timescales of his personal dynamics** — how quickly his wakefulness, sleep, and hormones respond to disturbance

The output is a joint probability distribution over all these quantities, not a single best-fit number. This is important: a patient with "ambiguous" data (perhaps equally consistent with high stress or phase shift) produces a *bimodal* posterior, honestly flagging the diagnostic uncertainty.

---

## 4. The seven physiological quantities the model tracks

### 4.1 Wakefulness ($W$)

A continuous number between 0 (fully asleep) and 1 (fully awake). Think of it as the instantaneous "depth of wakefulness" of the subject — not just "wake vs sleep" but **how awake** he is at each moment. Physiologically: the activity of the ascending arousal system (locus coeruleus, basal forebrain cholinergics, orexinergic neurons).

**Timescale**: responds to inputs over about 2 hours. That is why taking a nap or a late-night coffee has effects that linger for hours, not minutes.

**What drives it**:
- The circadian clock (stronger during the day, weaker at night)
- Sleep pressure from adenosine (stronger the longer you've been awake)
- Inhibition from sleep-promoting neurons ($\tilde Z$ below)
- **And testosterone.** This is the bidirectional coupling that closes the feedback loop (see §6).

### 4.2 Sleep depth ($\tilde Z$)

The activity of the ventrolateral preoptic (VLPO) sleep-promoting neurons. Rescaled to take values between 0 and 6, where higher means "deeper sleep". The sleep-observation channel maps this to the three observable stages:
- $\tilde Z < 3$: awake
- $3 < \tilde Z < 4.5$: light sleep or REM
- $\tilde Z > 4.5$: deep (slow-wave) sleep

**Timescale**: also about 2 hours. This matches the observed sleep-cycle duration.

**What drives it**:
- Inhibition from wakefulness (the "flip-flop" relationship)
- Chronic load ($V_n$ — higher chronic stress suppresses the sleep-promoting neurons)
- Adenosine pressure (the longer you've been awake, the easier the VLPO kicks in)

### 4.3 Adenosine / sleep pressure ($a$)

The level of extracellular adenosine in the brain. In real biochemistry, adenosine accumulates during wake and clears during sleep — it is the molecule caffeine blocks when it wakes you up. In the model it is a low-pass-filtered version of wakefulness: rising when awake, clearing when asleep, with a time constant of about 3 hours.

**Why it matters**: it is the memory that links "how long have I been awake" to "how sleepy am I getting". Without it, the sleep-wake flip-flop would lack homeostatic structure — a patient could in principle stay in either state indefinitely.

### 4.4 Testosterone pulsatility amplitude ($T$)

This is the **slow, clinically central variable**. It is not the instantaneous testosterone concentration — it is the *amplitude of the pulsatile release*, which is what the HPG axis actually controls. A healthy young man has large testosterone pulses during early-morning sleep; an unwell or older man has small or absent pulses.

**Timescale**: about 48 hours. Testosterone pulsatility does not change within a day — it changes over days as the underlying rhythm stabilises or destabilises. This is why acute stress (a single bad night) doesn't collapse testosterone but chronic stress (weeks of bad nights) does.

**What drives it**: the **entrainment quality** of the sleep-wake rhythm (see §5). This is the central mechanistic claim of the model: $T$'s dynamics are controlled by one variable — how well the sleep-wake system is entrained — and nothing else.

### 4.5 The circadian clock ($C$)

The 24-hour light-dark cycle of the external environment. The model treats this as a fixed sine wave peaking at about 10am local solar time ("morning type"). This is the *external* reference; the subject's *internal* drive can be phase-shifted relative to it (see $V_c$ below).

### 4.6 Vitality reserve ($V_h$)

A single number capturing the subject's underlying capacity for robust arousal — think of it as "how much life-force he has to bring to each day". Physiologically this conflates several mechanisms: fitness, nutritional status, absence of low-grade systemic inflammation, recovery from overtraining or illness, mitochondrial health. Phase 1 of the model treats this as a constant per subject; the clinical signal is whether the *inferred* $V_h$ is low (e.g. 0.2) or healthy (e.g. 1.0).

### 4.7 Chronic load ($V_n$)

A single number capturing tonic stress, allostatic burden, or chronic low-grade inflammation. It represents the "baseline stress" that never clears, independent of acute daily stressors. Physiologically this conflates: cortisol dysregulation, chronic psychological stress, chronic inflammatory cytokines, poor recovery capacity, perhaps smoking or heavy alcohol use. Again Phase 1 treats it as constant; the clinical signal is whether the posterior on $V_n$ is low (0.3, healthy) or elevated (2–4, severe insomnia-hyperarousal territory).

### 4.8 The phase shift $V_c$ — why it matters

This is a parameter with a very specific clinical meaning: **how many hours the subject's internal body clock is shifted relative to external time**. Healthy $V_c = 0$. Late-evening chronotypes and delayed-sleep-phase have $V_c > 0$. Advanced-sleep-phase (the "very early riser" pattern) has $V_c < 0$. Chronic shift workers or people with severe jet lag have $|V_c| \geq 6$ hours.

**The model's strong claim**: there is no such thing as a healthy evening chronotype. A man whose internal clock peaks at 2pm instead of 10am is not a "normal variant" but has a measurable phase-shift pathology, and the downstream consequence will be suppressed testosterone pulsatility. This is the cleanest point of falsifiability: if populations of genuinely-healthy men with $V_c > 2$h can be identified (healthy testosterone, healthy vitality, normal stress), the claim is wrong.

---

## 5. Entrainment quality — the quantity that controls testosterone

The entrainment quality $E(t) \in [0, 1]$ is the model's single summary of whether the sleep-wake rhythm is working. It captures two independent failure modes:

### 5.1 Amplitude failure

The subject isn't alternating cleanly between wake and sleep: he may be "stuck awake" (severe insomnia: can't fall asleep, can't stay asleep, shallow sleep when it happens) or "stuck asleep" (chronic fatigue: can't mount arousal, dozing through the day). In both cases the distinction between the wake state and the sleep state has broken down.

The model measures this through the *balance* of the sleep-wake flip-flop: when both arms of the flip-flop are pulling with approximately equal strength, the system alternates cleanly; when one arm pulls much harder (because $V_h$ is low, or $V_n$ is high, or adenosine has drifted out of range), the system saturates on one side and the amplitude collapses.

### 5.2 Phase-shift failure

The subject alternates between wake and sleep with healthy vigour — but at the wrong time of day. A night-shift worker might have a perfect 16h wake / 8h sleep cycle, but with wake peaking at 2am instead of 10am. The flip-flop is structurally fine; the synchronisation with the external world is wrong.

The model measures this through the phase misalignment $V_c$ between the subject's internal drive and the external light cycle. At $V_c = 0$, full entrainment; at $|V_c| = 6$h (a 6-hour shift, characteristic of chronic shift work), the entrainment quality drops to zero regardless of how deep the sleep is.

### 5.3 The combined entrainment quality

Entrainment quality is the product of an amplitude factor (how deeply the subject is alternating) and a phase factor (how well-aligned he is). **Either factor going to zero drives $E$ to zero**. This means:

- A patient with perfect sleep architecture but 6h phase shift has $E = 0$ — **and** collapsed testosterone.
- A patient in perfect phase but with very shallow sleep has $E = 0$ — **and** collapsed testosterone.
- A patient with a small phase shift (2h) and slightly shallow sleep (amplitude 0.7) has $E = 0.7 \times \cos(\tfrac{2\pi \cdot 2}{24}) \approx 0.6$ — **and** healthy testosterone, because $E$ is still above the critical threshold.

### 5.4 The critical threshold

There is a sharp threshold $E_\mathrm{crit}$ (calibrated to 0.5 in the current parameter set). Above it, testosterone is driven to a healthy steady-state amplitude. Below it, testosterone collapses to zero. This is a genuine bifurcation in the dynamics — not a gradual weakening but a qualitative shift.

Clinically: patients sitting above the threshold, even marginally, have intact testosterone. Patients sitting below the threshold, even marginally, have collapsed testosterone. A small intervention that pushes a patient from just-below to just-above the threshold can produce dramatic clinical recovery.

---

## 6. The bidirectional feedback: testosterone → wakefulness

There is one more coupling that makes the dynamics interesting. Testosterone itself contributes to wakefulness: high-amplitude testosterone pulsatility (the healthy state) strengthens the arousal drive $u_W$; collapsed pulsatility weakens it. The coupling coefficient is modest ($\alpha_T \approx 0.3$) — testosterone is not the main driver of wakefulness, merely a contributor.

This creates a **closed feedback loop**:

- Good sleep-wake entrainment → high $E$ → healthy testosterone pulsatility → reinforced wakefulness → better-maintained sleep-wake cycle → sustained entrainment.
- Poor entrainment → low $E$ → collapsed pulsatility → weakened wakefulness during the day → fragmented sleep-wake cycle → further entrainment loss.

Both are stable attractors. Clinically this explains the clinical observation that low-testosterone patients are difficult to rescue with testosterone replacement alone: replacement lifts the direct $T$ effect but does not fix the underlying entrainment failure, and the patient relapses when replacement stops.

The model predicts that **fixing the entrainment** (through sleep hygiene, light therapy, stress reduction, circadian realignment) leads to recovery that *persists after the intervention ends* — because the feedback loop has been restored. This is a testable clinical prediction (§9 H5 below).

---

## 7. What the model assumes — the list

Ten assumptions, each individually questionable, collectively constituting the theory.

**A1 — The ascending arousal system can be modelled as a single wakefulness scalar $W$.** Collapsing locus coeruleus, orexinergic, cholinergic and histaminergic contributions into one variable loses detail but captures the dominant clinical signal: the patient is, moment-to-moment, somewhere on a spectrum from deeply-asleep to fully-alert.

**A2 — Sleep pressure (adenosine) is the only homeostatic pressure that matters.** Other candidates — cytokines, core body temperature, melatonin — are not explicitly modelled. Their effects are absorbed into the $V_h, V_n$ constants or ignored.

**A3 — The circadian clock is a fixed sinusoid.** Real circadian rhythms are not pure sinusoids, can be re-entrained by light, and interact with sleep. The model uses the sinusoidal approximation for simplicity and treats re-entrainment as a change in $V_c$, not a dynamical event.

**A4 — All healthy subjects are morning types.** A clinical stance, not a biological fact. The model reserves the "internal phase" axis to measure pathological shift; it does not allow for a spectrum of healthy chronotypes. This is consistent with most evidence that evening chronotype is a risk factor for metabolic, mood and cardiovascular pathology, but is a stronger claim than the evidence base strictly supports.

**A5 — Testosterone pulsatility is the clinically meaningful endpoint, not total testosterone level.** This follows the endocrinology literature's distinction between pulsatile and tonic secretion; the model does not currently try to predict serum total-T.

**A6 — The Stuart-Landau normal form is the right description of pulsatility dynamics.** Near a supercritical pitchfork bifurcation — which is what a pulsatile axis becoming non-pulsatile looks like — the Stuart-Landau form is *the* universal description (centre-manifold reduction). This is a strong mathematical argument and probably the most defensible assumption in the model.

**A7 — Entrainment is a scalar quantity.** The model collapses "how well is the sleep-wake rhythm entrained" into one number $E$. Real entrainment has many dimensions (depth, amplitude, coupling strength to individual zeitgebers, REM-specific components, etc.). The scalar simplification is a working approximation.

**A8 — $V_h$ and $V_n$ are constants on the week-to-week timescale.** A man's vitality and chronic load do vary over longer timescales (weeks to months), but are approximately constant within the 1–2 week fit window. This is the Phase-1 convention; a future Phase-2 would treat them as slow stochastic processes.

**A9 — Wearable-derived sleep stages are approximately correct.** The model trusts the Garmin-style sleep classifier to produce broadly accurate 3-stage (wake/light+REM/deep) labels, acknowledging it won't get every 30-second epoch right. Systematic bias in the classifier would bias the inference; random classification noise is absorbed into the observation model.

**A10 — The subject's behaviour is not adapting in response to the measurement.** The model assumes passive observation. If the patient is deliberately changing their behaviour mid-fit (e.g. starting a new sleep schedule during the measurement period), the posterior will be confused — the model will try to fit a single set of parameters to what is effectively two different subjects.

---

## 8. Where each piece comes from in existing physiology

| Model element | Physiological grounding |
|:---|:---|
| Wakefulness $W$, sleep $\tilde Z$ flip-flop | Saper et al.'s ventrolateral preoptic / ascending arousal flip-flop (Nature 2005) |
| Adenosine $a$ as sleep pressure | Porkka-Heiskanen et al., basal forebrain adenosine accumulation (Science 1997) |
| Circadian drive as sinusoidal | Kronauer's two-oscillator model and its descendants |
| Testosterone pulsatility amplitude | Urban et al.'s characterisation of LH pulsatility and its collapse in hypogonadism |
| Stuart-Landau normal form for HPG | Standard centre-manifold reduction near a Hopf/pitchfork bifurcation |
| Entrainment ↔ HPG coupling | Leproult & Van Cauter's sleep-restriction studies (JAMA 2011); Patel et al.'s shift-worker testosterone data |
| Chronic load $V_n$ → sleep suppression | HPA-axis literature on cortisol and insomnia; Vgontzas et al. on hyperarousal in insomnia |
| Vitality $V_h$ → wake amplitude | Less precise — this is where the model is most "phenomenological" (grouping several mechanisms) |
| Phase shift $V_c$ → entrainment collapse | Shift-work epidemiology; Wittmann et al.'s "social jet lag" literature |

The model is a synthesis. Every piece has a literature; the originality is in the *coupled feedback structure* and the identification of entrainment quality as the single bifurcation parameter.

---

## 9. Testable hypotheses

Each of the following is a specific, falsifiable prediction. A study that reports data contradicting any of them would refute (or force modification of) the corresponding claim in the model.

### H1 — Shift workers with full sleep duration still have suppressed testosterone

The model predicts: a night-shift worker who sleeps 8 hours per day (at the "wrong" time) and whose sleep architecture looks normal (deep and light proportions preserved) will nonetheless have suppressed testosterone pulsatility. The mechanism is phase-shift-driven $E$ collapse, not amplitude-driven.

**Test**: recruit night-shift workers with good sleep metrics (total sleep time, deep sleep fraction matched to controls). Measure morning testosterone and LH pulsatility. SWAT predicts suppression; a competing model ("sleep quantity determines testosterone") predicts no suppression.

### H2 — Insomniacs without phase shift show amplitude-type collapse

The model predicts: a patient with severe hyperarousal-insomnia (high $V_n$, fragmented and shallow sleep, no circadian phase shift) will show testosterone suppression of *different* character than a shift worker's. Specifically, sleep metrics (duration, deep fraction) will be abnormal; $V_c$ inferred by the model will be near zero.

**Test**: compare two clinical groups — chronic primary insomnia without circadian involvement vs. chronic shift work without insomnia. Predict that SWAT assigns distinct $(V_n, V_c)$ profiles to the two groups despite both having suppressed testosterone.

### H3 — Recovery from phase-shift pathology precedes recovery from amplitude pathology in time

After a successful intervention, the model predicts that phase-shifted patients recover testosterone faster (timescale $\sim\tau_T = 48$h, i.e. 2–8 days) than amplitude-collapsed patients (who need their sleep architecture to rebuild *first*, then testosterone follows with the same lag — total 2–4 weeks).

**Test**: longitudinal study of returning shift-workers (acute phase fix) vs. insomniacs treated with CBT-I (slower amplitude fix). Track testosterone recovery trajectories.

### H4 — The critical threshold is sharp

The model predicts a genuine bifurcation: small improvements in entrainment that push a patient across $E_\mathrm{crit}$ produce disproportionately large testosterone recovery. A patient sitting at $E = 0.45$ (just below threshold) will have near-zero testosterone; pushing to $E = 0.55$ (just above) produces near-full recovery.

**Test**: intensive intervention studies where entrainment is gradually improved. Plot testosterone recovery against inferred $E$. The model predicts a sigmoid — sharp transition near threshold — *not* a smooth linear relationship.

### H5 — Entrainment restoration produces lasting recovery; testosterone replacement does not

The model predicts: a man treated with TRT alone (exogenous testosterone replacement) will see testosterone normalise during treatment, but the underlying $E$ remains low. When treatment stops, the patient relapses. A man treated with entrainment restoration (light therapy, sleep scheduling, stress management) will see testosterone recover and *stay recovered* after treatment ends, because the feedback loop has been restored.

**Test**: comparative long-term follow-up of TRT vs. entrainment-based interventions, measuring testosterone at 3, 6, 12 months after treatment discontinuation. This is the strongest prediction of the bidirectional-feedback claim.

### H6 — Testosterone tracks entrainment quality even in healthy subjects

The model predicts: within-subject, testosterone pulsatility amplitude varies with inferred $E$ even in healthy men — e.g. after a weekend of poor sleep, $E$ dips and testosterone amplitude dips accordingly, with the ~48h lag. This is not pathological variation; it is the normal behaviour of the feedback loop in healthy territory.

**Test**: frequent testosterone measurements in healthy men undergoing controlled sleep disruption (e.g. one night of sleep restriction). Predict an approximately 48-hour lag between the perturbation and the amplitude dip.

### H7 — $V_c$ and $(V_h, V_n)$ are distinguishable from wearable data alone

This is the identifiability claim the statistics companion document formalises, but it is also an empirical prediction. Two subjects with identical wearable time-series cannot have distinguishable pathologies; two subjects with different pathologies should have distinguishable posteriors on $(V_h, V_n, V_c)$.

**Test**: SMC² parameter recovery from synthetic data generated under each of the four canonical scenarios (healthy, insomnia, recovery, shift work). The model should recover the correct parameter values with non-overlapping credible intervals.

---

## 10. Where the model is likely to be wrong or incomplete

An honest accounting of the places where the model's assumptions are weakest.

### 10.1 The "morning type is healthy" claim (A4)

This is the assumption most likely to be wrong, because:
- Clock-gene polymorphisms (PER1/2/3, BMAL1, CRY1/2) clearly exist in the healthy population and produce modest chronotype variation.
- The evidence that evening-type is *itself* pathological (rather than correlated with pathology because evening-types have bad modern-life fits) is contested.
- Teenagers are naturally phase-delayed and not straightforwardly unhealthy.

**If A4 is wrong, the model will over-diagnose phase-shift pathology in healthy evening chronotypes.** The fix is to make $\phi_0$ subject-specific (a population-level prior of $\mathcal{N}(-\pi/3, \sigma)$ rather than a point constant). This is a straightforward future extension.

### 10.2 REM sleep is bundled with light sleep (§6.2 of technical doc)

The model doesn't separate REM from light sleep in the observation channel. This is a compromise to avoid adding a REM latent state. REM may carry independent information about HPG function (most testosterone pulses occur during REM-associated portions of the sleep cycle). A future model version should separate REM.

### 10.3 The stress score model is a linear approximation

The model treats the Garmin stress score as a linear function of wakefulness and chronic load. Real HRV-derived stress is nonlinear in sympathetic/parasympathetic balance, and Garmin's firmware applies proprietary smoothing. The linearity assumption may be badly wrong for some subjects.

**If it is badly wrong, the model's identifiability of $V_n$ (which relies on the stress channel) is impaired.** Mitigation: the model can be run without the stress channel; $(V_h, V_n)$ are then identified only up to their sum. The direction of inaccuracy is known and bounded.

### 10.4 The flip-flop approximation may be too sharp

In reality the transitions between sleep and wake are probabilistic over ~30 min, not infinitely sharp. The sigmoid-based flip-flop captures this but with a specific functional form. Subjects with fragmented sleep (e.g. sleep apnea with frequent arousals) may be poorly fit.

**If so, identified $\tau_W, \tau_Z$ will be artifically short for those subjects.** The signal of fragmentation is in the sleep-stage channel but is currently absorbed by the observation noise rather than into a fragmentation parameter.

### 10.5 $V_h, V_n$ as constants is definitely wrong on long timescales

No one's vitality or chronic load is actually constant. It changes with season, with ageing, with life events, with training and recovery. The Phase-1 constant approximation is justified only on the week-to-week timescale of a single fit. Applied to a quarter-year of data from the same subject, the model would be systematically wrong in a way that would show up as poor fit of the slow trends.

### 10.6 One-off acute events are not modelled

An acute illness, a single stressful event, a night of binge drinking — all produce acute perturbations that don't fit the stochastic-steady-state structure of the model. These will appear as "noise" inflating the diffusion temperatures or as anomalies in the residuals, rather than as identified events.

---

## 11. What a clinician could do with this, today

Given the model's current state (simulation validated, real-data calibration pending), the plausible clinical uses are:

**Research mode** — generate synthetic subjects under different intervention scenarios, use the model as a computational sandbox to explore hypotheses before designing human trials. The four canonical scenarios (A/B/C/D) in the technical documentation are the starting point for this.

**Decision-support on observed data** — once calibrated against real patient data, the model can produce subject-level diagnostic posteriors (how much of this patient's hypogonadism is stress, how much is phase, how much is vitality?) and intervention projections (what does the model predict if we do light therapy for 4 weeks?).

**Not yet appropriate for** — treatment decisions on individual patients, endocrine diagnosis replacing LH/FSH/testosterone measurement, regulatory claims. The model is a scientific hypothesis with computational backing; it is not a medical device and should not be used as one.

---

## 12. Further reading

The mathematical specification lives in the companion document `SWAT_Basic_Documentation.md` (same repository). The identifiability analysis — which shows that the model's parameters can in principle be recovered from wearable data — is in `SWAT_Identifiability_Extension.md`.

For the underlying physiology:

- Saper, C. B., Chou, T. C. & Scammell, T. E. (2001). "The sleep switch: hypothalamic control of sleep and wakefulness." *Trends in Neurosciences* 24: 726–731.
- Borbély, A. A. (1982). "A two process model of sleep regulation." *Human Neurobiology* 1: 195–204.
- Leproult, R. & Van Cauter, E. (2011). "Effect of 1 week of sleep restriction on testosterone levels in young healthy men." *JAMA* 305: 2173–2174.
- Urban, R. J. et al. (1988). "Attenuated release of biologically active luteinizing hormone in healthy aging men." *Journal of Clinical Investigation* 81: 1020–1029.
- Vgontzas, A. N., Bixler, E. O. & Chrousos, G. P. (2005). "Sleep apnea is a manifestation of the metabolic syndrome." *Sleep Medicine Reviews* 9: 211–224.

These are the empirical foundations the model builds on. The model's claim is that they can be integrated into a single dynamical system whose parameters can be inferred from consumer-wearable data.

---

*End of clinical specification.*
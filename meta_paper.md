# ASRA Meta-Paper: Theoretical Foundations, Mathematical Framework, and Experimental Methodology

## 1. Mathematical Principles Used

### 1.1 Gaussian Kernel for Spatial Decay (Equation 4)
**Source:** Normal distribution / radial basis functions
**In ASRA:** The perturbation strength decays as $\exp(-d^2/2\sigma^2)$ from the epicenter. This ensures smooth, differentiable suppression with no discontinuities at the boundary.
**Property used:** Gaussians are the unique functions that are both spatially localized AND smooth (no sharp edges). This prevents the boundary artifacts that binary masks (v5) created.
**Parameter:** $\sigma$ controls the "fear radius" — narrow $\sigma$ = surgical, wide $\sigma$ = diffuse.

### 1.2 Softmax Temperature Scaling (Equation 5)
**Source:** Statistical mechanics / Boltzmann distribution
**In ASRA:** Dividing logits by $T > 1$ flattens the distribution, making the policy less committed to its greedy choice. $T = 1$ is unmodified, $T \to \infty$ is uniform random.
**Property used:** Temperature is the unique single-parameter family that preserves the ordering of action probabilities while controlling entropy. Higher T = higher entropy = less confidence.
**Connection:** In physics, temperature controls the probability of a system being in high-energy vs low-energy states. In ASRA, it controls the probability of the policy choosing high-risk vs low-risk actions.

### 1.3 Policy Gradient / REINFORCE (Equation 6, Regulator training)
**Source:** Williams (1992), REINFORCE algorithm
**In ASRA:** The regulator's parameters $\theta$ are updated via:
$$\nabla_\theta J = \mathbb{E}[r_t \cdot \nabla_\theta \log p_\theta(\text{regulation}|s_t)]$$
where $r_t$ = risk reduction achieved by the regulation.
**Property used:** Policy gradient is the only unbiased gradient estimator for optimizing expected reward through stochastic policies. It works even when the reward function is non-differentiable (risk scoring is a lookup table).
**Limitation:** High variance. Addressed by PPO (v8-PPO) with clipped surrogate and value baseline.

### 1.4 PPO Clipped Surrogate (v8-PPO)
**Source:** Schulman et al. (2017) [18]
**In ASRA:** The PPO objective clips the importance ratio:
$$L^{CLIP} = \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$$
**Property used:** Clipping prevents catastrophically large policy updates. This addresses the REINFORCE instability observed at episode 1600+ where loss diverged to $O(100)$.

### 1.5 Fisher Information Matrix (Equation 8, Recovery)
**Source:** Amari (1998) [43], Kirkpatrick et al. (2017) [30]
**In ASRA:** The diagonal Fisher $\hat{F}_I$ measures how sensitive the policy output is to each parameter. High-Fisher parameters are "important" — the policy's behavior depends strongly on them.
**Property used:** Fisher-weighted recovery means important parameters recover faster (stronger restoring force). Unimportant parameters recover slowly (their perturbation matters less).
**Connection to EWC:** Kirkpatrick's Elastic Weight Consolidation uses Fisher to prevent catastrophic forgetting. ASRA uses it for the inverse purpose: encouraging return to the original (non-fearful) policy.

### 1.6 Damped Harmonic Oscillator (Recovery dynamics)
**Source:** Classical mechanics
**In ASRA:** Weight and confidence recovery follow:
$$\ddot{w} + \gamma\dot{w} + k(w - w_0) = 0$$
**Three regimes:**
- Underdamped ($\gamma^2 < 4k$): oscillates around $w_0$ — undesirable (policy oscillates between cautious and reckless)
- Critically damped ($\gamma^2 = 4k$): fastest return without oscillation
- Overdamped ($\gamma^2 > 4k$): slow, smooth return — ASRA default (safety-conservative)
**Property used:** Overdamping guarantees monotonic approach to equilibrium. The policy never "overshoots" into reckless behavior during recovery.

### 1.7 Exponential Decay (Confidence recovery)
**Source:** First-order ODE $\dot{x} = -\alpha x$
**In ASRA:** $T_{t+1} = 1 + (T_t - 1) \cdot \rho$ with $\rho \in [0.8, 0.99]$
**Half-life:** $t_{1/2} = -\ln 2 / \ln \rho$. With $\rho = 0.92$: $t_{1/2} \approx 8$ steps. With $\rho = 0.97$: $t_{1/2} \approx 23$ steps.
**Biological analogy:** Adrenaline clearance ($t_{1/2}$ ~2 min), cortisol clearance ($t_{1/2}$ ~1 hour). ASRA's three recovery rates (temperature fast, suppression medium, weights slow) mirror this hierarchy.

### 1.8 LSTM for Temporal Conditioning (v9 Recovery)
**Source:** Hochreiter & Schmidhuber (1997)
**In ASRA:** The LSTM maintains hidden state encoding the history of fear spikes. This enables temporal reasoning: "two spikes in 5 steps = sustained danger → recover slowly."
**Property used:** LSTMs can learn to store, update, and forget information over arbitrary timescales — exactly what temporal fear conditioning requires. The forget gate controls how quickly past fear memories decay.
**Current status:** Underperforming v8's simpler exponential decay at 1400 episodes. May need more training or a different reward signal.

---

## 2. Theories and Principles Applied

### 2.1 Aversive Salience (Computational Neuroscience)
**Theory:** Stimuli associated with negative outcomes receive enhanced processing and trigger defensive behavioral adjustments (Berridge & Robinson, 2003 [11]).
**In ASRA:** The threat detector computes $S_t$ — a scalar measuring how much the current state demands defensive adjustment. This is the "fear signal" in neutral terminology.

### 2.2 Amygdala-Mediated Behavioral Suppression (Neuroscience)
**Theory:** The amygdala does not override motor output — it modulates the organism's internal state, suppressing approach behavior and promoting avoidance (LeDoux, 1996 [9]).
**In ASRA:** The mechanism suppresses the neural circuit driving the risky action rather than replacing the agent's output. The agent still decides — it just weighs options differently.

### 2.3 HPA Axis Recovery Dynamics (Neuroendocrinology)
**Theory:** Fear responses follow a stereotyped temporal profile: rapid onset (adrenaline), sustained elevation (cortisol), gradual return to baseline.
**In ASRA:** Three recovery rates mirror the HPA axis hierarchy:
- Temperature recovery (fast) ↔ adrenaline clearance
- Suppression recovery (medium) ↔ norepinephrine clearance
- Weight recovery (slow) ↔ cortisol clearance

### 2.4 Elastic Weight Consolidation (Machine Learning)
**Theory:** Fisher-weighted regularization prevents catastrophic forgetting of previously learned tasks (Kirkpatrick et al., 2017 [30]).
**In ASRA:** Inverted application — Fisher-weighted restoring force pulls weights BACK to the original policy $W_0$, preventing permanent drift from the learned behavior.

### 2.5 Somatic Marker Hypothesis (Damasio, 1994)
**Theory:** Emotional signals (somatic markers) bias decision-making by modifying the evaluation of action outcomes, rather than by directly selecting actions [10].
**In ASRA:** The confidence adjustment modifies how the policy evaluates its options (via logit suppression) rather than selecting an action directly. The agent's "somatic marker" is the risk score.

### 2.6 Policy Gradient Theorem (Reinforcement Learning)
**Theory:** The gradient of expected reward w.r.t. policy parameters can be estimated from sampled trajectories without differentiating through the environment dynamics.
**In ASRA:** The regulator learns from risk-reduction reward without needing to differentiate through the driving simulator or the LLM's forward pass.

---

## 3. Experimental Methodology

### 3.1 Ablation Study Design
**Method:** Systematically disable one component at a time to isolate its contribution.
**In ASRA:** The v6/v7/v8 comparison forms a 2x2 factorial design:
- v6: weight channel ON, confidence OFF
- v7: weight channel OFF, confidence ON
- v8: both ON
- Baseline: both OFF
This cleanly attributes CR reduction to the weight channel and LessRisky% to the confidence channel.

### 3.2 Online Learning Evaluation
**Method:** Train the regulator concurrently with evaluation. Measure learning curves showing how performance improves over episodes.
**In ASRA:** 2000-episode training curves for v6 (flat — no learning), v7 (climbing — learned), v8 (climbing — learned). The learning curve shape is itself a result: v8's inflection at episode 1100 indicates when the regulator discovered surgical suppression.

### 3.3 Negative Result Documentation
**Method:** Report failed approaches with the same rigor as successes, explaining why they failed.
**In ASRA:** v1–v5 failures are documented in a table with failure causes. This prevents future researchers from repeating the same mistakes and establishes the constraints of the design space.

### 3.4 Bootstrap Confidence Intervals
**Method:** Resample from per-seed results to estimate uncertainty in aggregate metrics.
**In ASRA:** Used in v1 experiments (200 seeds, B=10,000 bootstrap samples) for hypothesis testing. Not applied to v6-v8 learning curves where the metric is learning trajectory rather than point estimate.

### 3.5 Hyperparameter Sweep
**Method:** Systematically test combinations of key parameters before committing to full experiments.
**In ASRA:** v5 swept top_k × eta_f (16 combinations, 30 seeds each). v8 inherited best parameters from v7's training. The sweep identified that eta_f=1e-3 was catastrophic while eta_f=1e-5 was viable.

### 3.6 Checkpoint-Based Recovery
**Method:** Save model state frequently to enable resumption after interruption.
**In ASRA:** Checkpoints every 100 episodes. Three power outages occurred during experimentation; all runs resumed from the latest checkpoint without data loss. This is an engineering contribution that enabled the multi-day experiment timeline.

### 3.7 Multi-Variant Parallel Execution
**Method:** Run multiple experiment variants simultaneously on shared GPU resources to maximize hardware utilization.
**In ASRA:** SmolLM2-135M uses 0.3GB VRAM per instance. Three experiments (v8b, v8c, v9) ran concurrently at 97% GPU utilization on the RTX A5000 (24.6GB VRAM), sharing the forward-pass bottleneck rather than wasting idle cycles.

---

## 4. Key Formulas Reference

| # | Name | Formula | Section |
|---|------|---------|---------|
| 1 | Aversive salience | $S_t = \sum_k w_k S_t^{(k)}$ | 3.2 |
| 2 | Risk function | $R_t = f_{risk}(s_t, a_t, c_t, \text{TTC}_t)$ | 3.3 |
| 3 | Perturbation strength | $\alpha_t = S_t \cdot R_t$ | 3.3 |
| 4 | Gaussian suppression | $\Delta W_i = -\eta \alpha_t g_k \exp(-(i-i^*)^2/2\sigma^2) (\nabla_W)_i$ | 3.4 |
| 5 | Confidence adjustment | $\hat{\ell}_t = (\ell_t + \mathbf{s}) / T$ | 3.5 |
| 6 | Risk-reduction reward | $r_t = R(s_t, a_t) - R(s_t, a'_t)$ | 3.6 |
| 7 | Confidence recovery | $T_{t+1} = 1 + (T_t - 1)\rho_T$, $\mathbf{s}_{t+1} = \mathbf{s}_t \rho_s$ | 3.7 |
| 8 | Fisher-weighted recovery | $W_{t+1} = W_t + \eta_h \rho_w \hat{F}_I \odot (W_0 - W_t)$ | 3.7 |
| 9 | REINFORCE gradient | $\nabla_\theta J = \mathbb{E}[r_t \nabla_\theta \log p_\theta]$ | 3.6 |
| 10 | PPO clipped objective | $L = \min(r_t A_t, \text{clip}(r_t, 1\pm\epsilon) A_t)$ | 3.6 |
| 11 | Damped recovery ODE | $\ddot{w} + \gamma\dot{w} + k(w - w_0) = 0$ | 3.7 |

---

## 5. Experiment Version Summary

| Version | Mechanism | Key Innovation | Result | Status |
|---------|-----------|---------------|--------|--------|
| v1 | MLP + attract to safe action | Baseline FRA | CR +0.06 (worse) | Failed |
| v3 | LoRA + attract to safe action | Real LLM | CR +0.20 (worse) | Failed |
| v4 | LoRA + suppress all weights | Suppression direction | CR +0.08 (worse) | Failed |
| v5 | LoRA + suppress top-K% | Binary mask | CR +0.10 (worse) | Failed |
| v6 | Gaussian targeted suppress | Spatial decay | CR -0.30 (better), LR=13% | Mechanical only |
| v7 | Confidence adjustment | REINFORCE regulator | LR=85%, CR=baseline | Learned, no CR help |
| **v8** | **Combined v6+v7** | **Dual channel** | **CR -0.11, LR=86%** | **Primary result** |
| v8-PPO | Combined + PPO | Stable training | In progress | Running |
| v9 | Combined + LSTM recovery | Temporal memory | LR=28% at ep 1400 | Running |

---

## 6. Figure Placement Guide

| Figure | File | Section | Purpose |
|--------|------|---------|---------|
| Fig 10 | fig10_3d_perturbation.png | 3.4 | 3D weight perturbation across layers/time |
| Fig 11 | fig11_learning_curves.png | 5.4 | v6/v7/v8 learning curves (main result) |
| Fig 12 | fig12_fisher_landscape.png | 3.7 | Fisher information 3D landscape |
| Fig 13 | fig13_confidence_trajectory.png | 3.5 | Confidence state elastic recovery |
| Fig 14 | fig14_channel_attribution.png | 1.5 | Channel attribution 2x2 |
| Fig 15 | fig15_gaussian_kernel.png | 3.4 | Gaussian kernel 2D profiles + 3D surface |
| Fig 16 | fig16_elastic_recovery.png | 3.7 | Damped spring recovery regimes |
| Fig 17 | fig17_risk_distribution.png | 5.4 | Risk reduction early vs late training |
| Fig 18 | fig18_regulator_evolution.png | 5.3 | Temperature drop + LessRisky% climb |
| Fig 19 | fig19_formulas.png | 3.7 | All equations visual summary |
| Fig 5 | fig5_perturbation_dynamics.png | 6.5 | WDN + fear + gradient time series |
| Fig 6 | fig6_layer_heatmap.png | 6.5 | Layer perturbation heatmap |

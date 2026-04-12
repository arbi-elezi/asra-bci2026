# Experimentation Improvement Plan

## Priority 1: Replicate CR Reduction (1-2 weeks)

### Problem
The CR improvement (0.80 → 0.69) comes from a single dedicated-compute run. Multi-trial shows CR=0.93 due to convergence failure.

### Plan
Run 5 dedicated-compute v8 trials sequentially (not parallel) with different seeds:
- Seeds: 100, 200, 300, 400, 500
- Episodes: 2000 each (matching the original)
- Full GPU per trial (no sharing)
- Estimated: 5 × 30 hours = 150 hours (~6 days)

### Success criterion
3/5 trials show CR < 0.75 (below baseline). Report mean ± std.

---

## Priority 2: Pre-trained Regulator (1 week)

### Problem
The regulator needs ~1200 episodes to converge. This is slow and compute-dependent.

### Plan
1. Train regulator once with full compute (2000 episodes)
2. Save regulator weights
3. Deploy frozen regulator on 10 new seeds (no online learning)
4. Measure CR and LessRisky with the frozen regulator

### Success criterion
Frozen regulator achieves LessRisky > 50% on unseen seeds. CR should match or improve on baseline.

### Why this matters
Removes the compute-convergence dependency. If the regulator generalizes when frozen, deployment becomes practical.

---

## Priority 3: Competent Baseline (2 weeks)

### Problem
Base policy CR = 0.75. Mechanism untested on competent policies.

### Plan
1. Train PPO for 200K steps (4× current) → target CR < 0.30
2. Collect new D_ref (20K pairs) from competent policy
3. Recompute Fisher
4. Run v8 with competent baseline
5. Run 5 dedicated-compute trials

### Success criterion
ASRA shows measurable effect (LessRisky > 10%, or CR reduction) on a competent baseline.

### Risk
Sparse threat encounters → slow regulator learning. May need 10K+ episodes.

---

## Priority 4: Multi-Model Comparison (2-3 weeks)

### Problem
Only SmolLM2-135M tested. Paper claims about LLM perturbation may be model-specific.

### Plan
Models already downloaded:
- SmolLM2-360M (694MB) — 2.5× parameters
- Qwen2.5-0.5B (954MB) — different architecture
- TinyLlama-1.1B (2.1GB) — paper's original target (31 checkpoints saved)
- Qwen 7B (local, 4-bit) — large model

For each model:
1. Day 1: PPO training + artifacts (reuse experiment_v3 pipeline)
2. Day 2: Run v8 combined mechanism
3. Day 3: 3 dedicated-compute trials

### Success criterion
Mechanism works on at least 2/4 additional models.

---

## Priority 5: Learned Risk Evaluator (2 weeks)

### Problem
Hand-designed R_t limits generality.

### Plan
1. Train a small MLP as risk evaluator: (state, action) → risk
   - Training data: (state, action, did_crash_within_10_steps) from baseline episodes
   - Binary classification: risky if crash follows within 10 steps
2. Replace hand-designed R_t with learned evaluator
3. Run v8 with learned evaluator
4. Compare learned vs hand-designed

### Success criterion
Learned evaluator achieves similar or better LessRisky% without domain-specific rules.

---

## Priority 6: Gaussian Targeting Ablation (3 days)

### Problem
Reviewer asked: does gradient-based epicenter targeting matter, or just the Gaussian shape?

### Plan
Already have gradient-center data (18% LessRisky at 1200 episodes). Need:
1. Random-center Gaussian (same σ, random epicenter) — 2000 episodes
2. No-Gaussian (uniform perturbation at same magnitude) — 2000 episodes
3. Compare all three

### Success criterion
If gradient-center > random-center: targeting matters.
If gradient-center ≈ random-center: shape matters, not location.

---

## Priority 7: REINFORCE vs PPO Regulator (1 week)

### Problem
REINFORCE diverges at episode 1600+. v8-PPO didn't converge well either (LR=22% at 3000 episodes).

### Plan
1. Tune PPO hyperparameters: clip_range, n_epochs, learning rate
2. Try A2C (advantage actor-critic) as alternative
3. Try SAC-style entropy regularization
4. 2000 episodes each, dedicated compute

### Success criterion
Find a training algorithm that converges stably to LR > 50% without diverging.

---

## Execution Timeline

| Week | Task | GPU hours |
|------|------|-----------|
| 1 | Priority 1: 5 dedicated-compute v8 trials | 150 |
| 2 | Priority 2: Pre-trained regulator + Priority 6: Ablation | 80 |
| 3 | Priority 3: Competent baseline training + v8 | 200 |
| 4 | Priority 4: TinyLlama + SmolLM2-360M | 150 |
| 5 | Priority 5: Learned risk evaluator | 100 |
| 6 | Priority 7: Training algorithm comparison | 100 |

**Total: ~780 GPU hours (~32 days of continuous A5000 compute)**

---

## What Each Priority Adds to the Paper

| Priority | Section affected | Reviewer concern addressed |
|----------|-----------------|---------------------------|
| 1 | Section 5.3 | "Single-run success is fragile" |
| 2 | Section 6.3 | "Compute-dependent convergence" |
| 3 | Section 7 | "No demonstration on competent policy" |
| 4 | Section 4.1 | "Single model size" |
| 5 | Section 3.1 | "Risk evaluator bottleneck" |
| 6 | Section 5.5 | "Gaussian targeting unvalidated" |
| 7 | Section 3.4 | "REINFORCE instability" |

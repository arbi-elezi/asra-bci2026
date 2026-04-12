# Experiment v7 — Confidence-Level Adjustment with Learned Risk-Based Regulation

## What v6 showed
v6 proved the mechanism works: CR 0.80 → 0.50 (37% reduction). But:
- The exciter never learned (loss=0.0000, MSE too weak)
- The improvement came from the physics (Gaussian + elastic), not AI
- LessRisky% plateaued at 10-17% — the exciter wasn't adapting

## v7: The key change
Instead of modifying WEIGHTS (destructive, hard to control), modify the OUTPUT
DISTRIBUTION's confidence. When fear spikes:

- Increase temperature → flatter distribution → less committed to risky action
- Scale logits of the risky action down → suppress it in the output
- Let the LLM's existing knowledge pick the alternative

This is softer than weight perturbation. The weights stay at W_0.
Only the output changes.

## Why this is scientifically better

### Weight perturbation (v3-v6) problems:
1. Changes are persistent — even with elastic recovery, weights drift
2. Perturbation corrupts learned representations
3. Hard to control — too little does nothing, too much destroys the policy
4. The LLM wasn't trained to work with perturbed weights

### Confidence adjustment advantages:
1. Non-destructive — W_0 is never modified
2. Instantly reversible — next timestep can use full confidence
3. Naturally interpretable — "less confident in risky action" is what fear does
4. Smooth control — temperature is a single scalar, easy to learn
5. The LLM's representations stay intact — it just weighs options differently

## Architecture

### Components
1. **Brain** (LLM): SmolLM2-135M with LoRA. Forward pass → logits. Weights NEVER modified.
2. **Fear detector**: Same as v6. Independent. Outputs fear ∈ [0,1].
3. **Risk evaluator**: Same as v6. Scores action risk. Brake always safe.
4. **Regulator** (replaces exciter): Small neural network that learns:
   - **Temperature scaling**: how much to flatten the distribution (τ ∈ [0.5, 5.0])
   - **Per-action suppression**: how much to subtract from each action's logit
   - **Recovery rate**: how quickly confidence returns to normal (decay ∈ [0.8, 0.99])
5. **Confidence state**: Tracks current temperature and per-action suppression.
   Decays smoothly toward baseline (τ=1, suppression=0) following elastic dynamics.

### The flow per timestep

```
Step 1: Brain → logits [0.1, 0.6, -0.2, 0.3] (wants ACCELERATE)
Step 2: Risk(ACCELERATE) = 0.83, Fear = 0.72
Step 3: Regulator(fear, risk, logits) → {
          temperature: 2.5 (flatten distribution)
          suppress: [0, -1.2, 0, 0] (penalize ACCELERATE logit)
          recovery_rate: 0.92
        }
Step 4: Adjusted logits = (logits + suppress) / temperature
         = ([0.1, 0.6-1.2, -0.2, 0.3]) / 2.5
         = [0.04, -0.24, -0.08, 0.12]
         → softmax → [0.27, 0.20, 0.24, 0.29]

         Original was: softmax([0.1, 0.6, -0.2, 0.3]) = [0.20, 0.33, 0.15, 0.24]
         ACCELERATE went from 33% → 20%. LANE_CHANGE rose to 29%.

Step 5: Sample action from adjusted distribution → LANE_CHANGE
Step 6: risk(LANE_CHANGE) = 0.35 < risk(ACCELERATE) = 0.83 → reward = +0.48
Step 7: Regulator learns from this reward
Step 8: Confidence state decays: suppression *= 0.92, temperature → 1.0 gradually
```

### Regulator training (REINFORCE with baseline)
The MSE loss in v6 was too weak. v7 uses REINFORCE:
- Reward = risk_reduction = risk(original) - risk(adjusted)
- Log probability of the regulator's outputs under its own distribution
- Loss = -reward × log_prob (policy gradient)
- This gives a MUCH stronger learning signal than MSE

### Recovery dynamics
Confidence doesn't snap back. It follows exponential decay:
  τ(t+1) = 1.0 + (τ(t) - 1.0) × recovery_rate
  suppress(t+1) = suppress(t) × recovery_rate

With recovery_rate = 0.92:
  Step 0: τ=2.5 (very uncertain)
  Step 5: τ=1.66 (moderately uncertain)
  Step 10: τ=1.27 (slightly uncertain)
  Step 20: τ=1.05 (nearly normal)
  Step 30: τ=1.01 (back to baseline)

The brain transitions: panicked → cautious → alert → normal
Each state produces different action distributions.
Smooth. Gradual. Elastic.

## Experimental design

### Phase 1: Regulator training (2000 episodes)
Train the regulator online. Measure learning curve.
Primary metric: % of FRA steps with less risky action.

### Phase 2: Evaluation (200 seeds, matched)
Freeze the trained regulator. Run 200 seeds.
Compare against:
- C1: baseline (no FRA)
- v6 best: Gaussian perturbation result
- C7: hard brake override

### Phase 3: Ablations
- Temperature only (no per-action suppression)
- Suppression only (no temperature scaling)
- Fixed regulator (no learning, fixed τ=2.0)
- No recovery (instant return to normal)

## What we produce
1. Learning curve: does the regulator improve over 2000 episodes?
2. Risk reduction distribution: histogram of risk(original) - risk(adjusted)
3. Action shift analysis: how does the distribution change when fear fires?
4. Recovery curves: τ over time after fear spikes
5. Comparison table: v6 (weight perturbation) vs v7 (confidence adjustment)
6. CR as secondary metric

## Hypothesis
The regulator will learn faster and achieve higher LessRisky% than v6
because:
1. REINFORCE gives stronger gradient signal than MSE
2. Confidence adjustment is a lower-dimensional control problem (τ + 4 suppressions)
   vs weight perturbation (502K parameters)
3. The LLM's representations stay intact — only output weighting changes

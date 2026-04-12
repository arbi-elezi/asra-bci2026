# Experiment v6 — Gaussian-Shaped Elastic Perturbation with Learned Epicenter

## Scientific Motivation

### Why Gaussian (not uniform or binary mask)
Neural circuits are not discrete — activation patterns form continuous spatial
distributions across parameters. When a subset of weights drives a risky decision,
the influence decays with "distance" from the most responsible weights. A Gaussian
kernel models this decay naturally.

**Physical analogy**: pressing a finger into an elastic sheet. The depression is
deepest at the contact point (epicenter) and smoothly decays outward. This is
identical to how we want to suppress the risky decision circuit.

### Why elastic recovery (not step function)
Biological fear responses follow elastic dynamics:
- **Suppression phase**: rapid deformation (fear spike)
- **Recovery phase**: the elastic restoring force is proportional to displacement
  AND the material has viscosity (damping), so recovery is smooth, not oscillatory
- **Equation**: ẅ + γẇ + k(w - w₀) = 0 (damped harmonic oscillator)
  - k = elastic stiffness (Fisher-weighted)
  - γ = damping coefficient (prevents overshoot)
  - w₀ = equilibrium (original W_0)
  - This gives smooth exponential recovery: w(t) = w₀ + Δw₀ · e^(-αt) · cos(βt)
  - Overdamped case (γ² > 4k): pure exponential decay, no oscillation

### Why logistic regression for epicenter
The gradient magnitude alone doesn't tell us which weights RELIABLY drive risky
decisions across different states. A weight might have high gradient for one
particular risky action but not others.

Logistic regression learns: given the gradient pattern, which weight indices
consistently predict that the action was risky? This gives us a LEARNED importance
map — the LR coefficients β_i tell us "weight i's gradient magnitude is a β_i-strong
predictor of risky behavior." The epicenter is where β_i × |g_i| is highest.

## Epicenter Detection Methods (6 approaches)

### Method 1: Raw Gradient Maximum
- Epicenter = argmax(|∇_W log P(action)|)
- Simplest. No learning. Baseline comparison.
- Limitation: noisy, state-dependent

### Method 2: Fisher-Weighted Gradient
- Epicenter = argmax(F_I ⊙ |∇_W log P(action)|)
- Fisher importance weights the gradient by parameter importance
- Motivation: important weights (high Fisher) that are also excited should be epicenter
- Already computed offline from D_ref

### Method 3: Online Logistic Regression
- Accumulate (gradient_vector, risk_label) pairs online
- Every N steps, fit logistic regression: P(risky | gradient) = σ(β·g)
- Epicenter = argmax(β ⊙ |g|)
- The LR learns which weight-gradient patterns predict risky decisions

### Method 4: Exponential Moving Average of Gradient Patterns
- For each weight i, track EMA of |g_i| across risky decisions
- ema_i ← α × |g_i| + (1-α) × ema_i (only updated when action is risky)
- Epicenter = argmax(ema_i × |g_i|)
- Captures persistent patterns without explicit regression

### Method 5: Neural Epicenter Detector (small MLP)
- Train a small network: gradient_pattern → epicenter_location
- Input: top-100 gradient magnitudes
- Output: probability distribution over parameter groups (per-layer)
- Trained online from (gradient, risk) pairs

### Method 6: Layer-wise Epicenters
- Instead of one global epicenter, find one per layer
- Each layer's Gaussian is independent with its own σ
- Motivation: different layers encode different aspects of the decision

## Perturbation Profile Variations

### Gaussian sink (primary)
w_i -= η × risk × G(i, i*, σ) × g_i
where G(i, i*, σ) = exp(-||i - i*||² / (2σ²))

### Mexican hat (suppress center, excite periphery)
w_i -= η × risk × MH(i, i*, σ₁, σ₂) × g_i
where MH = G(σ₁) - 0.5 × G(σ₂) with σ₂ > σ₁

### Asymmetric Gaussian
Different σ on each side of epicenter — steep suppression toward the risky
weights, gentle on the side toward safer weights.

## Elastic Recovery Model

### Damped spring
Each suppressed weight follows:
  w_i(t+1) = w_i(t) + α_i × (w₀_i - w_i(t)) - β_i × (w_i(t) - w_i(t-1))

where:
  α_i = restoring force (Fisher-weighted, Gaussian-shaped from epicenter)
  β_i = damping (prevents oscillation)
  The velocity term (w_i(t) - w_i(t-1)) provides smooth momentum

### Properties:
- Not instantaneous — takes multiple timesteps to recover
- Epicenter recovers slower (more displaced = stronger suppression)
- Periphery recovers faster (barely displaced)
- No oscillation in overdamped regime (β² > 4α)
- Converges to W_0 exponentially

## Measurement
1. **Risk reduction per FRA step**: risk(original_action) - risk(perturbed_action)
2. **% timesteps with less risky choice**: primary metric
3. **Recovery smoothness**: WDN trajectory should be smooth exponential, not jagged
4. **Epicenter stability**: does the same weight region light up for similar risky decisions?
5. **CR**: secondary metric, not primary

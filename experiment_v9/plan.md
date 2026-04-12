# Experiment v9 — LSTM Recovery Controller

## Why RNN for recovery

The recovery problem is inherently sequential: the rate at which fear subsides
depends on what happened BEFORE (how many spikes, how severe, how close together)
and what's happening NOW (is danger still nearby?). A fixed scalar or even a
feedforward network can't capture this temporal structure.

An LSTM naturally maintains a hidden state that encodes the full history of
fear spikes, suppressions, and recoveries. It learns temporal patterns:
- "Two spikes in 5 steps = sustained danger → recover very slowly"
- "One spike, 20 steps of calm = isolated scare → recover quickly"
- "Spike during recovery from previous spike = compounding fear → freeze recovery"

## Architecture

### Recovery LSTM
- Input per timestep: [fear, risk, cost, ttc, displacement, velocity, time_since_spike] = 7 dims
- LSTM: 7 → 32 hidden → 32 hidden (2-layer)
- Output heads:
  - temp_rate: 32 → 1 → sigmoid → [0.8, 0.99] (temperature decay rate)
  - sup_rate: 32 → 1 → sigmoid → [0.8, 0.99] (suppression decay rate)
  - weight_rate: 32 → 1 → sigmoid → [0.8, 0.99] (weight perturbation FHR rate)
- The LSTM hidden state carries across timesteps within an episode
- Reset at episode start

### Why these specific parameters
- **2-layer LSTM, 32 hidden**: small enough to train from single-episode trajectories
  (512 timesteps), large enough to capture multi-spike patterns. Tested: 16 too small
  (underfits spike patterns), 64 too large (overfits to noise with our data volume).
- **7 input features**: minimal sufficient set. Fear and risk capture current danger.
  Cost and TTC capture environment state. Displacement, velocity, time_since capture
  recovery state. Adding more (e.g., per-layer displacement) didn't help in preliminary
  tests — the LSTM learns to extract what it needs from these 7.
- **3 separate output rates**: temperature, suppression, and weight perturbation
  recover at different speeds. Temperature should recover fastest (confidence returns
  quickly). Suppression recovers at medium rate. Weight perturbation recovers slowest
  (the "muscle memory" of fear). This mirrors the biological HPA axis:
  adrenaline clears fast, cortisol clears slowly.

### Training: REINFORCE on episode-level reward
- Reward per episode = mean(risk_reduction) across FRA-active timesteps
- The LSTM's recovery decisions affect ALL subsequent timesteps in the episode
  (slow recovery = more timesteps of cautious behavior = more risk reduction but
  potentially worse task performance)
- REINFORCE with full episode trajectories — each episode is one rollout
- Baseline: exponential moving average of episode rewards

### Integration with v8 mechanism
Same as v8 for perturbation (Gaussian weight + confidence adjustment).
Only the recovery dynamics change — controlled by LSTM instead of fixed/simple learned.

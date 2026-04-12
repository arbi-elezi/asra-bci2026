# Data Sources Assessment for FRA Experiment

## Paper Requirements (Definition 3)
The environment must satisfy:
- (a) Finite discrete action space, |A| < infinity
- (b) Scalar cost signal c_t in [0,1] directly observable
- (c) Offline cost critic trainable on representative D_ref
- (d) Per-class cost critic error M10-s below DR deployment thresholds

Specific paper specs: s_t in R^12, |A|=4, 3 obstacle classes, c_t = max(0, (2 - TTC_t) / 2)

---

## Option 1: highway-env (BEST FIT) ★★★★★
- **Source**: https://github.com/Farama-Foundation/HighwayEnv
- **Install**: `pip install highway-env`
- **Why it fits**:
  - Native TTC (Time-To-Collision) observation mode
  - Discrete meta-actions (lane changes + speed control) — can be configured to exactly 4 actions
  - Multiple vehicle/obstacle types (maps to 3 obstacle classes)
  - Gymnasium-compatible API
  - Lightweight, fast simulation (no 3D rendering overhead)
  - Well-documented, maintained by Farama Foundation
  - TTC-based state representation is EXACTLY what the paper describes
- **Adaptation needed**:
  - Configure state space to R^12 (highway-env kinematics observation is configurable)
  - Set exactly 4 discrete actions
  - Define 3 obstacle classes (slow car, fast car, stationary obstacle)
  - Implement cost signal c_t = max(0, (2 - TTC_t) / 2)
  - May need to customize obstacle spawning for stress test seeds
- **Risk**: Low — environment is mature and well-tested

## Option 2: Custom Gymnasium Environment ★★★★
- **Source**: Build from scratch using Gymnasium API
- **Why consider**:
  - Total control over state space, actions, cost signal
  - Exact match to paper specifications
  - No adaptation layer needed
  - Simpler codebase (no unused features)
- **Adaptation needed**:
  - Full implementation from scratch
  - Physics engine (simple 2D — NumPy sufficient)
  - Rendering (pygame or matplotlib)
- **Risk**: Medium — more development time, but ensures exact match

## Option 3: MetaDrive (top-down mode) ★★★
- **Source**: https://github.com/metadriverse/metadrive
- **Install**: `pip install metadrive-simulator`
- **Why consider**:
  - Has safe RL cost signal built in (collision cost)
  - Top-down 2D rendering mode
  - Diverse scenarios
- **Adaptation needed**:
  - Continuous action space by default — need discretization wrapper
  - State space may not match R^12
  - Cost signal format differs from paper's TTC-based formula
  - Heavier dependency (3D engine even for 2D mode)
- **Risk**: Medium-High — significant adaptation, overkill for this experiment

## Option 4: Safety-Gymnasium ★★★
- **Source**: https://github.com/PKU-Alignment/safety-gymnasium
- **Install**: `pip install safety-gymnasium`
- **Why consider**:
  - Designed for safe RL research (CMDP framework)
  - Cost constraints built in
  - NeurIPS 2023 benchmark
  - 16 SafeRL algorithms integrated
- **Adaptation needed**:
  - Not a driving environment — would need to use/create driving-like task
  - MuJoCo-based (heavier than needed for 2D driving)
  - Agent types don't match driving scenario
- **Risk**: High — wrong domain, would be forcing a fit

## Option 5: Kaggle / HuggingFace Datasets ★★
- **Kaggle**: No directly suitable driving RL environment datasets found
- **HuggingFace**: Deep RL course materials but no matching environment
- **Waymo Open Dataset**: Real-world driving data, but for perception, not RL training
- **HighD Dataset**: Highway driving data with TTC measurements, but for imitation learning
- **Why not ideal**: The paper needs a SIMULATOR, not a dataset. D_ref is generated from base policy rollouts.

---

## RECOMMENDATION: Hybrid Approach

### Primary: Adapt highway-env (Sprint 1)
1. Use highway-env as the base environment
2. Configure: 4 discrete actions, TTC observation
3. Create 3 obstacle classes (slow, fast, stationary)
4. Implement paper's cost function: c_t = max(0, (2 - TTC_t) / 2)
5. Wrap with Gymnasium API for clean interface

### Fallback: Custom build (if highway-env is insufficient)
If highway-env's abstraction layer prevents:
- Exact R^12 state control
- Fine-grained obstacle class manipulation for C8 stress tests
- Seed-deterministic obstacle placement
Then build a lightweight custom env that exactly matches the paper.

### D_ref Generation (not a data source — generated in-house)
- Train PPO on the environment → 100K steps
- Freeze W_0
- Roll out base policy → collect 1000 (s, a) pairs = D_ref
- Compute Fisher, Lipschitz constant, G_max^0 from D_ref
- This is all generated, not sourced externally

---

## Decision Needed
The paper specifies a "2D driving simulation" with very specific properties. The most viable path:
1. **Start with highway-env** — already has TTC, discrete actions, multiple vehicle types
2. **Evaluate in Sprint 1** whether it can be configured to exactly match Definition 3
3. **Fall back to custom** only if highway-env fundamentally can't be adapted

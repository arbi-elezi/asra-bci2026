# Viability Assessment

## Resource Budget

### Compute
- GPU: Quadro RTX A5000 (24GB VRAM, CUDA)
- PPO training (100K steps): ~10 min
- Cost critic training: ~5 min per variant (7 variants = ~35 min)
- Fisher matrix computation (1000 pairs): ~2 min
- Per condition, per seed: ~30s (500 timesteps × env step + FRA overhead)
- Total condition-seed pairs: 7 × 1000 + 5 × 500 = 9,500
- Raw compute: ~79 hours sequential
- With 10-seed batching on GPU: ~8 hours
- Hyperparameter search (200 seeds × ~50 configs): ~4 hours
- **Total estimated**: ~12-15 hours wall-clock

### Storage
- Per run: ~1MB (config + metrics + checkpoint)
- Total: ~10GB (with checkpoints every 1K steps)
- 24GB VRAM is more than sufficient for 64-64 MLP + cost critic

## Risk Assessment

### Low Risk
- Environment implementation (well-defined spec)
- PPO training (standard, SB3 handles it)
- Fisher computation (straightforward offline)
- Bootstrap CIs (standard statistics)

### Medium Risk
- Hyperparameter search (large space, but constrained)
- GTCC calibration loop (novel, may need tuning)
- TD-Fear smoothing (γ_f selection)

### High Risk
- L_G estimation quality (finite differences on D_ref — could be noisy)
- DR directional correctness in practice (paper is honest about this)
- M7 > 0.70 gate (RLAF reliability — if fails, RLAF unusable)
- C8 stress tests revealing unexpected failure modes

### Mitigation
- L_G: Use multiple step sizes, report sensitivity
- DR: A4 is designed to test exactly this; failure is a valid outcome
- M7: GTCC is the primary path; RLAF is supplementary
- C8: Failure modes are the POINT of the stress test

## Timeline (Aggressive but Feasible)

### Week 1: Infrastructure
- Day 1-2: Environment + PPO training
- Day 3: D_ref, Fisher, cost critic, G_max^0, L_G, r
- Day 4: FRA components (FHR, DR, GTCC, etc.)
- Day 5: Integration test, C2 runs 10 episodes

### Week 2: Experiments
- Day 1: Hyperparameter search
- Day 2-3: C1-C7 primary runs               
- Day 3-4: C8a-e stress tests
- Day 5: Analysis, bootstrap CIs

### Week 3: Paper
- Day 1-2: Results tables and figures
- Day 3: Write results section
- Day 4: Revise paper with actual results
- Day 5: Final review and submission prep

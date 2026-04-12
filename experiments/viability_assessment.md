# Viability Assessment — Most Viable Path to Full Realization

## Critical Path Analysis

### Tier 1: Must Have (blocks everything)
1. **2D Driving Environment** — all experiments depend on this
2. **PPO Base Agent** — all conditions require a frozen W_0
3. **Cost Critic** — DR, GTCC, and F_t^CA all depend on the offline critic
4. **D_ref Collection** — Fisher, L_G, G_max^0 all derived from reference data

### Tier 2: Core Components (blocks primary hypothesis H1)
5. **FHR (Fisher restoring force)** — core mechanism
6. **DR (Deregulator)** — extension mechanism
7. **Fear signal pipeline** (F_t^CA → TD-Fear → Uncertainty → FMS → F_t^final)
8. **SCL (Safe action blending)**

### Tier 3: Calibration Components (blocks H2, H7)
9. **GTCC** — ground-truth consequence calibration
10. **FC (Fear Classifier)** — learned threat detector
11. **RLAF** — semantic label supplement
12. **FMS** — distribution shift correction

### Tier 4: Evaluation (blocks paper submission)
13. **Metrics M1-M14** implementation
14. **Bootstrap CI** framework
15. **Seed management** (1000 matched seeds)
16. **Stress test infrastructure** (degraded/biased critics)

## Implementation Order (Recommended)

### Sprint 1: Foundation (Environment + Base Agent)
- Build or adapt 2D driving environment
- Train PPO agent to convergence
- Freeze W_0, compute D_ref
- Verify C1 baseline

### Sprint 2: Offline Computations
- Compute F_hat_I from D_ref
- Estimate L_G via finite differences
- Compute r, G_max^0
- Train and freeze cost critic
- Measure M10 baseline

### Sprint 3: Core FRA
- Implement FHR (Fisher restoring force)
- Implement DR (policy shaping gradient)
- Implement fear signal: F_t^CA from cost advantage
- Implement SCL (safe action blending)
- Run C2 vs C1 → test H1

### Sprint 4: Extensions
- Implement TD-Fear smoothing
- Implement epistemic uncertainty (MC-dropout)
- Implement FMS (distribution shift correction)
- Implement GTCC + FC + RLAF pipeline
- Implement BC term for FHR

### Sprint 5: Ablations
- C3a (L2-HR), C3b (Fisher-no-BC)
- C4 (No-GTCC), C5 (No-FMS), C6 (No-DR)
- C7 (Hard override)
- Run all ablation conditions

### Sprint 6: Stress Tests
- Train degraded cost critics (10%, 25%, 50% of D_ref)
- Train biased cost critics (x0.2, x0.5 on fast obstacles)
- Run C8a-e
- Compute Algorithm 1 thresholds from C8 results

### Sprint 7: Analysis
- Compute all bootstrap CIs
- Validate Proposition 1 (M3 vs bound, M13 violations)
- Generate all tables and figures
- Write results section

## Key Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| PPO doesn't converge to useful policy | High | Use proven SB3 implementation; verify reward signal |
| Environment too simple → trivial results | Medium | Tune obstacle density for ~20-40% base CR |
| Cost critic quality insufficient | High | Validate M10 before DR experiments |
| L_G estimation inaccurate | Medium | M13 monitoring catches this empirically |
| Hyperparameter search expensive | Medium | Use 200-seed validation, not full 1000 |
| BC conjecture fails (H11) | Low | Expected possible outcome; paper handles it |
| DR provides no benefit (H8) | Medium | Expected possible outcome; core layer still valid |

## Data Requirements
- **D_ref**: 1000 state-action pairs from base policy rollouts
- **Validation set**: 200 seeds
- **Test set**: 1000 seeds (primary), 500 seeds (stress test)
- **Shifted seeds**: 100 seeds with higher obstacle density
- **Adversarial seeds**: 100 seeds constructed for A4
- **Total unique seeds needed**: ~1300 (some overlap allowed)

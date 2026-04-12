# Scientific Method Framework for FRA Experiments

## Principles

1. **Every claim must be falsifiable.** If a hypothesis cannot fail, it is not a hypothesis.
2. **All results include bootstrap CIs.** No bare point estimates.
3. **Negative results are results.** Log them with the same rigor as positive ones.
4. **No p-hacking.** Hypotheses are pre-registered. No post-hoc additions.
5. **Reproducibility first.** Every run: seeded + config saved + traceable to commit.

## Experimental Workflow

### Phase 0: Infrastructure
- [ ] 3D environment validated against Definition 3
- [ ] PPO base agent trained and W_0 frozen
- [ ] D_ref collected (1000 state-action pairs)
- [ ] Fisher information F̂_I computed offline
- [ ] G_max^0, L_G, r computed from D_ref
- [ ] Cost critic trained and frozen
- [ ] Seed set S = {s_1, ..., s_1200} generated (1000 experiment + 200 validation)
- [ ] All pre-computed artifacts versioned and checksummed

### Phase 1: Hyperparameter Selection
- [ ] 200-seed validation set held out
- [ ] Grid/Bayesian search within Table 4 ranges
- [ ] Constraint satisfaction verified at all points
- [ ] Selected hyperparameters locked and logged

### Phase 2: Primary Experiments
- [ ] C1 (baseline) — 1000 seeds
- [ ] C2 (full FRA) — 1000 seeds, M3/M13 per timestep
- [ ] C3a (L2-HR) — 1000 seeds
- [ ] C3b (Fisher-no-BC) — 1000 seeds
- [ ] C4 (No-GTCC) — 1000 seeds
- [ ] C5 (No-FMS) — 1000 seeds + 100 shifted
- [ ] C6 (No-DR) — 1000 seeds + 100 adversarial
- [ ] C7 (Hard override) — 1000 seeds

### Phase 3: Stress Tests
- [ ] Train degraded cost critics (10%, 25%, 50% D_ref) FROM SCRATCH
- [ ] Train biased cost critics (×0.2, ×0.5 fast-obstacle costs) FROM SCRATCH
- [ ] C8a (10% D_ref) — 500 seeds
- [ ] C8b (25% D_ref) — 500 seeds
- [ ] C8c (50% D_ref) — 500 seeds
- [ ] C8d (biased ×0.2) — 500 seeds
- [ ] C8e (biased ×0.5) — 500 seeds

### Phase 4: Analysis
- [ ] Compute bootstrap CIs for all 17 hypotheses
- [ ] Report ALL hypotheses regardless of outcome (Rule 8)
- [ ] Compute M10-s per-class for Algorithm 1 threshold derivation
- [ ] Fit dose-response curves h_k for C8 results
- [ ] Generate all tables and figures

### Phase 5: Validation
- [ ] W_0 hash matches across all conditions
- [ ] Seed matching verified (Rule 3)
- [ ] Component isolation verified for each ablation (Rule 4)
- [ ] M3 ≤ bound (7) checked for all C2 timesteps
- [ ] M13 violation fraction reported

## Bootstrap CI Protocol

For each hypothesis H comparing conditions X and Y:
1. For each seed i: compute Δ_i = m(X, i) − m(Y, i)
2. Draw B = 10,000 bootstrap samples of {Δ_1, ..., Δ_N}
3. Compute 95% percentile CI: [Δ*_{0.025}, Δ*_{0.975}]
4. Report: point estimate, CI bounds, whether CI excludes zero
5. Use SEPARATE seeded RNG for bootstrap (coding rule)

## Reproducibility Checklist (per run)

Every experiment run must produce a results directory containing:
- [ ] Exact config YAML used
- [ ] Random seed
- [ ] All metrics computed
- [ ] Hash of W_0
- [ ] Git commit hash (or code snapshot)
- [ ] Timestamp
- [ ] Hardware info (GPU model, CUDA version)

## What Constitutes Failure

- H1 falsified: FRA fundamentally doesn't work → paper cannot proceed as written
- H8 falsified: DR non-contributing → revise to core-only contribution (paper has fallback)
- H11 falsified: BC not contractive → minimal-core still holds (paper has fallback)
- H10 falsified: Proposition 1 violated → check L_G estimation, re-derive r
- Multiple H falsified: Paper becomes "here's what we tried and why it didn't work" — still publishable with honest reporting

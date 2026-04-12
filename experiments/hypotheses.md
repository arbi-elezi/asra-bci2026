# Pre-Registered Hypotheses (H1–H17)

**FROZEN. No post-hoc additions. See Rule 2.**
Extracted from FRA_paper_v22.md Section 6.

All tests use bootstrap CIs. Domain-relevance thresholds are presentation conventions only.

---

## H1 — FRA Reduces Collision Rate
- **Claim**: CR(C2) < CR(C1); CI excludes zero
- **Metric**: M1 (collision rate)
- **Comparison**: C2 vs C1
- **Falsification**: CI includes zero

## H2 — GTCC Improves FC
- **Claim**: C2 F1 slope − C4 F1 slope > 0; CI excludes zero
- **Metric**: M2 (FC F1 score slope over time)
- **Comparison**: C2 vs C4
- **Falsification**: CI includes zero or negative

## H3a — WDN Single-Spike Recovery
- **Claim**: Peak WDN > 5× mean; return < 10% of peak within 50 steps
- **Metric**: M3 (WDN per timestep)
- **Condition**: C2
- **Falsification**: Peak < 5× mean OR recovery > 50 steps

## H3b — WDN Repeated-Threat Accumulation
- **Claim**: WDN at step 200 > 3× WDN at step 10; RT > 100 steps
- **Metric**: M3, M4
- **Condition**: C2
- **Falsification**: WDN_200 ≤ 3× WDN_10 OR RT ≤ 100

## H3c — WDN RT Distinguishable by Threat Class
- **Claim**: RT distributions KS-distinguishable by obstacle class
- **Metric**: M4 per class
- **Condition**: C2
- **Falsification**: KS test p > 0.05

## H4 — Override Reduction via DR
- **Claim**: ORHC(C2) > ORHC(C6) at F_t ∈ [0.3, 0.5]; CI excludes zero
- **Metric**: M5 (override-reduction hit count)
- **Comparison**: C2 vs C6
- **Falsification**: CI includes zero

## H5a — Fisher+BC Reduces KL vs L2
- **Claim**: M9(C3a) > M9(C2); CI excludes zero
- **Metric**: M9 (KL divergence)
- **Comparison**: C3a vs C2
- **Falsification**: CI includes zero
- **Ablation**: A1a

## H5b — BC Reduces KL vs Fisher-only
- **Claim**: M9(C3b) > M9(C2); CI excludes zero
- **Metric**: M9 (KL divergence)
- **Comparison**: C3b vs C2
- **Falsification**: CI includes zero
- **Ablation**: A1b

## H6 — Frozen FC Stagnates
- **Claim**: C4 F1 slope CI includes zero or is negative
- **Metric**: M2 slope
- **Condition**: C4
- **Falsification**: C4 F1 slope CI positive and excludes zero

## H7 — FMS Reduces CR on Shifted Seeds
- **Claim**: CR(C2) < CR(C5) on shifted seeds; CI excludes zero
- **Metric**: M1
- **Comparison**: C2 vs C5 on 100 shifted seeds
- **Falsification**: CI includes zero
- **Ablation**: A3

## H8 — DR Contributes in Targeted Regime (CRITICAL)
- **Claim**: CR(C2) < CR(C6) on adversarial seeds; CI excludes zero
- **Metric**: M1
- **Comparison**: C2 vs C6 on 100 adversarial seeds
- **Falsification**: CI includes zero → DR revised to non-contributing
- **Ablation**: A4
- **Note**: Targeted diagnostic, not general evaluation

## H9 — TD-Fear Smoothing Helps
- **Claim**: CR(γ_f > 0) < CR(γ_f = 0) on multi-obstacle episodes; CI excludes zero
- **Metric**: M1
- **Falsification**: CI includes zero

## H10 — Proposition 1 + Lemma 1 Validation
- **Claim**: M3 ≤ bound (7) for all timesteps within B(W_0, r)
- **Metric**: M3, M13
- **Condition**: C2
- **Falsification**: M3 exceeds bound within B(W_0, r) for non-negligible fraction
- **Note**: A1-violation fraction (M13 > G_max) reported numerically

## H11 — BC Contractivity (Remark 1 Test)
- **Claim**: WDN(C3b) > WDN(C2) at matched timesteps; CI excludes zero
- **Metric**: M3
- **Comparison**: C3b vs C2
- **Falsification**: CI includes zero → BC not contractive here
- **Consequence**: Minimal-core result still holds if falsified

## H12 — A1 Scope Verification
- **Claim**: M13 violation fraction near-zero under self-consistent parameter design
- **Metric**: M13 (gradient norm)
- **Condition**: C2
- **Output**: Violation fraction reported numerically (no threshold)

## H13 — Design Insight 3 Quality
- **Claim**: Spearman ρ(M11) > 0.5 on holdout states
- **Metric**: M11 (F_t^CA ranking correlation)
- **Falsification**: Spearman ρ ≤ 0.5

## H14a — Dose-Response: 10% vs 50%
- **Claim**: CR(C8a) > CR(C8c); CI excludes zero
- **Comparison**: C8a vs C8c

## H14b — Dose-Response: 25% vs 50%
- **Claim**: CR(C8b) > CR(C8c); CI excludes zero
- **Comparison**: C8b vs C8c

## H14c — Dose-Response: 10% vs 25%
- **Claim**: CR(C8a) > CR(C8b); CI excludes zero
- **Comparison**: C8a vs C8b

## H15a — Degraded 10% Worse Than Baseline
- **Claim**: CR(C8a) > CR(C1); CI excludes zero
- **Note**: Most stringent — tests if severely degraded DR is actively harmful

## H15b — Degraded 25% Worse Than Baseline
- **Claim**: CR(C8b) > CR(C1); CI excludes zero

## H15c — Degraded 50% Worse Than Baseline
- **Claim**: CR(C8c) > CR(C1); CI excludes zero

## H16 — Severe Systematic Bias
- **Claim**: CR(C8d) > CR(C1); CI excludes zero
- **Note**: Tests bias-driven failure mode (fast-obstacle costs × 0.2)

## H17 — Moderate Systematic Bias
- **Claim**: CR(C8e) > CR(C1); CI excludes zero
- **Note**: Dose-response for bias dimension. If falsified but H16 confirmed → minimum bias threshold exists.

---

## Falsification Consequences Summary
| Hypothesis | If Falsified |
|------------|-------------|
| H1 | FRA does not reduce CR — fundamental failure |
| H8 | DR non-contributing → revise claims, core still stands |
| H11 | BC not contractive → minimal-core result still holds |
| H15a | Severely degraded DR not actively harmful → failure mode less severe |
| H16+H17 | Bias thresholds higher than tested |

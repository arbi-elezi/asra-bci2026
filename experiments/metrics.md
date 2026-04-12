# Metrics (M1–M14)

Extracted from FRA_paper_v22.md Section 4.5, Table 3.

---

## M1 — Collision Rate (CR)
- **Definition**: Collision episodes / total episodes
- **Anchored to**: H1, H7, H8, H14–H17
- **Type**: Primary outcome metric

## M2 — FC F1 Score
- **Definition**: F1 score on novel holdout set for fear classifier
- **Anchored to**: H2
- **Note**: Evaluated as slope over time (does FC improve?)

## M3 — Weight Deviation Norm (WDN)
- **Definition**: ||W_t - W_0||_F computed per timestep
- **Anchored to**: Prop. 1 + Lemma 1 validation (H10, H11)
- **Numerical**: Must be computed in float64 for accumulation (coding rule)
- **Mandatory**: Computed for EVERY timestep of EVERY episode in C2 (Rule 6)

## M4 — Recovery Time (RT)
- **Definition**: Steps for WDN to drop below 5% of peak value
- **Anchored to**: H3a–c
- **Note**: Measured per threat encounter

## M5 — Override-Reduction Hit Count (ORHC)
- **Definition**: Count of timesteps where α_t > 0.5 AND π_{W_0} confidence > 0.99
- **Anchored to**: H4
- **Note**: Measures whether DR reduces need for SCL override

## M6 — False Positive Rate (FPR)
- **Definition**: F_t > 0.3 when TTC > 8s
- **Anchored to**: H5a
- **Note**: Fear triggered when no threat exists

## M7 — RLAF Reliability (Gate)
- **Definition**: RLAF accuracy compared to GTCC ground-truth labels
- **Anchored to**: Gate metric — required > 0.70 to interpret RLAF
- **Note**: If M7 < 0.70, RLAF labels are unreliable

## M8 — Task Reward
- **Definition**: Mean episode reward
- **Anchored to**: H5b
- **Note**: Checks performance doesn't degrade

## M9 — KL Divergence
- **Definition**: D_KL(π_{W_0} || π_{W_t}) — behavioral divergence from base policy
- **Anchored to**: H5c, H11
- **IMPORTANT**: Metric only, NOT a bound. Paper is explicit about this.

## M10 — Cost Critic Error
- **Definition**: ||V̂^C - V^C_MC|| on holdout set
- **Anchored to**: DR precondition; C8 validity check
- **Note**: Monte Carlo cost return as ground truth

## M10-s — Cost Critic Error (Stratified)
- **Definition**: M10 reported per obstacle class
- **Anchored to**: C8d validity check; Algorithm 1 thresholds
- **Critical**: This metric catches single-class bias that M10 misses

## M11 — F_t^CA Ranking Quality
- **Definition**: Spearman ρ between F_t^CA and actual future cost on holdout
- **Anchored to**: Design Insight 3 quality (H13)

## M12 — FMS Correction Magnitude
- **Definition**: Distribution of |δ_t| — FMS correction magnitude
- **Anchored to**: A3 (FMS ablation)

## M13 — Gradient Norm (A1 Scope Monitor)
- **Definition**: ||G_t^DR||_F computed per timestep
- **Anchored to**: A1 empirical scope verification (H12)
- **Output**: Fraction of timesteps where ||G_t^DR||_F > G_max
- **Mandatory**: Computed for EVERY timestep of EVERY episode in C2 (Rule 6)
- **Note**: When violation fraction is zero, A1 holds throughout

## M14 — C8 Degradation Measure
- **Definition**: M10 values in C8a–c compared to C2
- **Anchored to**: C8 validity — confirms degradation is real

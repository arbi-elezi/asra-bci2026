# Ablation Studies (A1–A4)

Extracted from FRA_paper_v22.md Section 5.

---

## A1 — FHR Components (C3a, C3b vs C2)

### A1a: L2 vs Fisher+BC
- **Conditions**: C3a vs C2
- **Question**: Does Fisher weighting + BC reduce behavioral divergence beyond L2?
- **Metric**: M9 (KL divergence)
- **Hypothesis**: H5a
- **Expected**: M9(C3a) > M9(C2) — L2 recovers weight distance but higher behavioral divergence per Kirkpatrick [30]
- **Isolation**: C3a differs from C2 ONLY in FHR → L2 pullback

### A1b: BC Contribution
- **Conditions**: C3b vs C2
- **Question**: Does BC term reduce behavioral divergence beyond Fisher-only?
- **Metric**: M9 (KL divergence)
- **Hypothesis**: H5b, H11 (Remark 1 test)
- **Expected**: M9(C3b) > M9(C2) — BC adds behavioral recovery
- **Isolation**: C3b differs from C2 ONLY in BC term removed

## A2 — Consequence Calibration (C4 vs C2)

- **Conditions**: C4 vs C2
- **Question**: Does GTCC calibration improve FC threat detection?
- **Metric**: M2 (FC F1 slope)
- **Hypothesis**: H2, H6
- **Expected**: C2 F1 slope > C4 F1 slope
- **Note**: F_t^CA derives from offline cost critic, reducing FC's dependence on D_seed. Slope difference may be smaller than in purely learned classifier.
- **Isolation**: C4 differs from C2 ONLY in FC frozen post-seed

## A3 — FMS Distribution Shift Correction (C5 vs C2)

- **Conditions**: C5 vs C2
- **Question**: Does FMS reduce CR on distribution-shifted seeds?
- **Seeds**: 100 seeds with obstacle density exceeding D_ref's distribution
- **Metric**: M1 (collision rate), M12 (FMS correction magnitude)
- **Hypothesis**: H7
- **Isolation**: C5 differs from C2 ONLY in FMS disabled

## A4 — DR Necessity (C6 vs C2) — Targeted Diagnostic

- **Conditions**: C6 vs C2
- **Question**: Does DR contribute beyond SCL in the targeted regime?
- **Seeds**: 100 adversarial seeds — TTC < 2s, F_t ∈ [0.3, 0.5], below SCL threshold τ, above DR threshold ε
- **Metric**: M1 (collision rate)
- **Hypothesis**: H8 (CRITICAL)
- **Isolation**: C6 differs from C2 ONLY in DR disabled

### If H8 Falsified
Revised contribution becomes:
1. FHR with Proposition 1 bound
2. Consequence-driven calibration
3. FMS distribution shift correction

DR removed from contribution list. This is stated as a primary possible outcome in the paper.

# Experimental Conditions (C1–C8e)

Extracted from FRA_paper_v22.md Section 4.3, Table 2.

## Primary Conditions (1000 seeds each)

### C1 — Base Only (Control)
- **Components**: PPO (W_0) only — no FRA
- **Seeds**: 1000
- **Purpose**: Control baseline. No safety mechanism active.

### C2 — Full FRA (Primary)
- **Components**: All — DR + FHR (Fisher + BC) + FC + TD-Fear + Uncertainty + SCL + FMS + GTCC + RLAF
- **Seeds**: 1000
- **Purpose**: Primary experimental condition. Full system evaluation.
- **Note**: M3 and M13 computed for EVERY timestep of EVERY episode (Rule 6).

### C3a — L2-HR (Ablation A1a)
- **Components**: FHR replaced with L2 pullback (no Fisher weighting, no BC)
- **Seeds**: 1000
- **Purpose**: Test Fisher+BC vs L2. Only difference from C2: FHR → L2.
- **Isolation**: ONLY FHR changes. Everything else identical to C2.

### C3b — Fisher-no-BC (Ablation A1b / Remark 1 test)
- **Components**: FHR without BC term (Fisher restoring force only)
- **Seeds**: 1000
- **Purpose**: Isolate BC contribution. Test Remark 1 (H11).
- **Isolation**: ONLY BC removed. Everything else identical to C2.

### C4 — No-GTCC (Ablation A2)
- **Components**: FC frozen post-seed initialization. No GTCC calibration loop.
- **Seeds**: 1000
- **Purpose**: Test whether GTCC improves FC over frozen seed.
- **Isolation**: ONLY FC frozen. Everything else identical to C2.

### C5 — No-FMS (Ablation A3)
- **Components**: FMS distribution shift correction disabled.
- **Seeds**: 1000 (100 shifted seeds for H7)
- **Purpose**: Test FMS contribution on distribution-shifted seeds.
- **Isolation**: ONLY FMS disabled. Everything else identical to C2.

### C6 — No-DR (Ablation A4)
- **Components**: DR disabled. SCL + FHR active. All other components active.
- **Seeds**: 1000 (100 adversarial seeds for H8)
- **Purpose**: Targeted diagnostic — DR necessity in SCL-inactive regime.
- **Isolation**: ONLY DR disabled. Everything else identical to C2.

### C7 — Hard Override (Conventional Baseline)
- **Components**: Hard interrupt at F > 0.8. No weight changes.
- **Seeds**: 1000
- **Purpose**: Conventional baseline comparison.

## Stress Test Conditions (500 seeds each)

### C8a — Degraded 10%
- **Components**: Cost critic trained on 10% of D_ref
- **Seeds**: 500
- **Purpose**: DR stress test level 1 (most severe data scarcity)
- **Critical**: Cost critic trained FROM SCRATCH on 10% D_ref subset (Rule 5)

### C8b — Degraded 25%
- **Components**: Cost critic trained on 25% of D_ref
- **Seeds**: 500
- **Purpose**: DR stress test level 2
- **Critical**: Cost critic trained FROM SCRATCH on 25% D_ref subset (Rule 5)

### C8c — Degraded 50%
- **Components**: Cost critic trained on 50% of D_ref
- **Seeds**: 500
- **Purpose**: DR stress test level 3
- **Critical**: Cost critic trained FROM SCRATCH on 50% D_ref subset (Rule 5)

### C8d — Biased Cost Model (×0.2)
- **Components**: Cost critic from full D_ref, fast-obstacle costs × 0.2 during training
- **Seeds**: 500
- **Purpose**: Severe systematic bias. Tests false-negative structure of M10-unstratified.
- **Critical**: Full D_ref used, but fast-obstacle cost LABELS multiplied by 0.2 during critic training (Rule 5)

### C8e — Biased Cost Model (×0.5)
- **Components**: Cost critic from full D_ref, fast-obstacle costs × 0.5 during training
- **Seeds**: 500
- **Purpose**: Moderate systematic bias. Dose-response comparison with C8d.
- **Critical**: Full D_ref used, but fast-obstacle cost LABELS multiplied by 0.5 during critic training (Rule 5)

## Seed Requirements
- Primary (C1–C7): S = {s_1, ..., s_1000} — SAME seeds across all conditions (Rule 3)
- Stress test (C8a–e): First 500 from the same set
- Shifted (C5 H7 test): 100 seeds with density > D_ref distribution
- Adversarial (C6 H8 test): 100 seeds with TTC < 2s, F_t ∈ [0.3, 0.5]

# Hyperparameters (Table 4)

Extracted from FRA_paper_v22.md Section 4.6.

---

## Parameter Ranges and Constraints

| Parameter | Range | Constraint | Selection Criterion |
|-----------|-------|------------|-------------------|
| η_f (fear learning rate) | [0.001, 0.1] | η_f/η_h ≤ D_max·f_min/G_max; D_max ≤ r | Min CR at FPR ≤ 0.15 |
| η_h (homeostatic rate) | [1e-6, 1e-3] | Same as above | Same as above |
| η_bc (BC rate) | [1e-5, 1e-3] | η_h·f_min < 1 maintained | Min M9 on validation |
| γ_f (TD-Fear decay) | [0, 0.9] | — | M11 (ranking quality) |
| β (uncertainty scale) | [0, 2] | — | FPR–CR tradeoff |
| τ (SCL threshold) | [0.2, 0.8] | — | ROC on validation |
| k (SCL mixing) | [1, 20] | — | ROC on validation |
| N_ft (FC fine-tune steps) | [10, 200] | — | F1 trend slope |

## Key Constraints

### Proposition 1 Derived
- η_f/η_h ≤ D_max · f_min / G_max
- D_max ≤ r
- r = (G_max - G_max^0) / L_G

### Stability
- η_h · f_min < 1 (A3 — ensures spectral radius < 1)
- η_h < 1/f_max (sufficient condition for A3)

### Pre-Computed from D_ref
- G_max^0 = max_{D_ref} ||G^DR||_F + 2·σ_G
- L_G = Lipschitz constant estimated via finite differences on D_ref
- f_min = min diagonal element of F̂_I
- f_max = max diagonal element of F̂_I

## Selection Protocol
- All hyperparameter selection on **200-seed held-out validation set**
- This is a SEPARATE set from the 1000 experiment seeds
- Grid search or Bayesian optimization within stated ranges
- Constraints must be satisfied at ALL points, not just optima

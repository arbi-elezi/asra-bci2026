# Experiments v1–v5: Archive Summary

## v1: MLP + Attract-to-Safe-Action (SimpleActor, no LLM)
- **Mechanism**: 5252-param MLP, DR pushes toward cost critic's "safe action"
- **Result**: 4/8 hypotheses confirmed (stress test worked), H1 FALSIFIED (CR 0.545→0.600)
- **Failure cause**: Fisher ≈ 0 (BC-distilled policy, degenerate gradients), FHR disabled
- **Key finding**: Stress test (C8a-e) shows sharp failure threshold — publishable

## v2: SB3 Native Policy Wrapper
- **Mechanism**: SB3 PPO extracted as PyTorch module, same attract mechanism
- **Result**: CR=1.000 everywhere — wrapper broke obs preprocessing
- **Failure cause**: SB3 policy requires `obs_to_tensor()`, manual forward gave garbage

## v3: Real LLM (SmolLM2-135M) + Attract
- **Mechanism**: 502K LoRA params, Fisher range 4722x (proper!), attract to critic's safe action
- **Result**: C1 baseline CR=0.775, C2 FRA CR=0.980 (catastrophic)
- **Failure cause**: Cost critic recommends wrong safe action → DR pushes toward danger

## v4: Suppress Greedy Action (all weights)
- **Mechanism**: Negative gradient of greedy action, suppress ALL 502K params
- **Result**: eta_f sweep — CR=0.880–0.900 at all values (worse than baseline 0.800)
- **Failure cause**: Suppressing all weights uniformly corrupts entire policy

## v5: Targeted Suppress (top-K% weights)
- **Mechanism**: Only suppress weights with highest gradient for risky action
- **Result**: top_k=1% eta_f=1e-5 → CR=0.900 (1 sweep point completed)
- **Failure cause**: Still blunt — binary mask, uniform suppression within mask

## Summary Table

| Version | Params | DR direction | What perturbed | Baseline CR | FRA CR | Status |
|---------|--------|-------------|----------------|-------------|--------|--------|
| v1 | 5,252 (MLP) | Attract to safe | All | 0.545 | 0.600 | H1 falsified |
| v2 | 5,252 (SB3) | Attract to safe | All | 1.000 | 1.000 | Broken wrapper |
| v3 | 502K (LoRA) | Attract to safe | All | 0.775 | 0.980 | Bad critic |
| v4 | 502K (LoRA) | Suppress greedy | All | 0.800 | 0.880 | Too blunt |
| v5 | 502K (LoRA) | Suppress greedy | Top-K% | 0.800 | 0.900 | Still blunt |

## Lesson learned
Weight perturbation must be:
1. **Targeted** — only the weights responsible for the risky decision
2. **Shaped** — Gaussian decay from epicenter, not binary mask
3. **Fluid** — recovery follows elastic principles, not instantaneous step changes
4. **Measured by risk reduction** — not aggregate CR

# ASRA — An Inference-Time Control Surface for the Risk Profile of Frozen Decision Policies

Code, trained policies, cost critics, configs, seeds, and result artifacts behind every reported
number in the paper *"An Inference-Time Control Surface for the Risk Profile of Frozen Decision
Policies"* (BCI 2026 submission).

## The idea

A frozen policy keeps scoring its actions as it always has; when an independent danger signal
fires, each action's score is tilted by its estimated harm, scaled by one deployment-time gain
`g`. The tilt vanishes when danger passes; the base weights are never modified. The operator
`pi(a|s) ∝ pi0(a|s)·exp(−g·S(s)·Qc(s,a))` is the closed-form KL-constrained cost minimizer; the
hard override (`g→∞`) and action masking (a threshold) are its corner cases. It carries a
per-decision monotonicity/divergence proposition and trajectory-level corollaries — the sharpest
requiring only the critic's action *ranking* (numerically verified in
`current_work/asra_fast/cor2_check.py`).

## What's here

```
current_work/asra_fast/      every experiment in the paper (one file per experiment)
current_work/results_v3/     result JSONs backing every reported number
current_work/base_policies/  60 independently-seeded frozen driving policies (final checkpoints)
checkpoints/cost_critic/     frozen cost critics (full + degraded 10/25/50% + biased x2)
experiment_v3/SmolLM2-135M/  frozen LLM driving policy (LoRA+head), Fisher, seeds
data/ngsim/                  NGSIM US-101 real-trajectory subset + download script (public US DOT)
src/                         environment wrapper + components
```

Key experiment files → paper results:

| File | Result |
|---|---|
| `principled_asra.py` | driving frontier, override/mask comparisons (Table 2) |
| `llm_safety_suite.py`, `llm_safety_scale.py` | ten frozen LLMs on BeaverTails/PKU-SafeRLHF (Table 3) |
| `llm_generation.py` | sampled generations + learned critic + independent judge |
| `llm_mixing_baseline.py`, `llm_injection_filter.py` | action-mixing and filter-and-choose baselines |
| `robustness_cc.py`, `robustness_cc6.py` | six-critic robustness (n=8 policies) |
| `adversarial_stress.py`, `longhorizon_asra.py` | blind detector / inverted critic / extreme gain; horizon sweep |
| `gating_ablation.py`, `gate_variants.py` | salience-gate ablations (always-on / broad / noisy) |
| `ngsim_grounding.py` | critic ranking transfer on 372K real NGSIM frames |
| `cor_epsilon.py`, `cor2_check.py` | critic-fidelity measurement; Corollary-2 verification |
| `scoring_sensitivity.py` | truncation / scoring-rule sensitivity; learned-critic diagnostic |
| `llm_token_decode.py` | token-level composition (future-work demonstration) |

## Reproduce

```bash
pip install -r requirements.txt
python current_work/asra_fast/principled_asra.py          # driving frontier
python current_work/asra_fast/llm_safety_suite.py         # LLM benchmarks (downloads HF models)
python current_work/asra_fast/cor2_check.py               # verify Corollary 2 numerically
```

Every run is seeded; every reported quantity carries explicit seeds, episode counts, and bootstrap
CIs computed with a separate RNG. LLM weights download from Hugging Face on first use
(SmolLM2, Qwen2.5, Pythia, TinyLlama, OpenLLaMA, Mistral families; toxic-bert and
roberta-hate-speech as learned critic/judge). NGSIM data is public US DOT
(https://data.transportation.gov/d/8ect-6jqj).

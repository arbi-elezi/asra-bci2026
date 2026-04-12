# ASRA: Aversive Salience-Regulated Agent

Characterizing the risk-performance tradeoff in online behavioral modulation of frozen language model policies.

## Paper

"Characterizing the Risk-Performance Tradeoff in Online Behavioral Modulation of Frozen Language Model Policies" — submitted to BCI 2026.

## Key Finding

A dual-channel mechanism (targeted Gaussian weight perturbation + learned confidence adjustment) achieves 55.2% risk-reduction rate (p<0.001, Cohen's d=2.79) across 10 independent trials, but aggregate collision rate increases during regulator learning. Post-convergence, both metrics improve simultaneously.

## Structure

```
src/                    # Core library (environment, agents, components)
experiment_v{3-9}/      # Experiment variants (v3=base, v6=weight, v7=confidence, v8=combined)
experiment_stats/       # Multi-trial statistical validation
experiment_baselines/   # Rule-based and shielding baselines
configs/                # Per-condition YAML configs
tests/                  # Unit tests (including REINFORCE gradient-flow verification)
analysis/               # Post-experiment hypothesis testing
paper_latex/            # LaTeX source (generate_lncs.py auto-generates from trial data)
```

## Models and Data

Trained models, checkpoints, and experiment results are on HuggingFace: [PLACEHOLDER]

## Requirements

- Python 3.10+, PyTorch 2.6+, CUDA 12.4
- `pip install -r requirements.txt`
- GPU: NVIDIA RTX A5000 (24GB) or equivalent

## License

[TBD]

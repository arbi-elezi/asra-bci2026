# References to Add (2024-2025)

## Directly Relevant

1. **SafeLoRA** — Hsu et al., "Safe LoRA: the Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models," NeurIPS 2024.
   - Projects LoRA weights to safety-aligned subspace. ASRA does the inverse: temporarily perturbs LoRA weights away from risky behavior then recovers.
   - https://arxiv.org/abs/2405.16833

2. **TTRL** — Zuo et al., "TTRL: Test-Time Reinforcement Learning," arXiv 2025.
   - RL at test time using pseudo-labels from majority voting. Related: ASRA also learns at test time but via risk-reduction reward, not pseudo-labels.
   - https://arxiv.org/abs/2504.16084

3. **TARL** — "Test-time Adapted Reinforcement Learning with Action Entropy Regularization," OpenReview 2024.
   - Addresses distribution shift at test time for RL. Uses entropy regularization to prevent overcorrection — similar to ASRA's confidence channel increasing entropy.
   - https://openreview.net/forum?id=Xv1jY6U0pT

4. **Safe RL Survey** — "A Survey of Safe Reinforcement Learning and Constrained MDPs," arXiv 2025.
   - Comprehensive survey of CMDP formulations. ASRA is positioned outside this framework (not constraint-based) but the survey contextualizes the broader field.
   - https://arxiv.org/abs/2505.17342

5. **Weight-Perturbed DNNs** — Sharma et al., "Investigating Weight-Perturbed Deep Neural Networks," WACV 2024.
   - Shows networks are sensitive to Gaussian noise in early layers. Supports ASRA's finding that targeted perturbation matters — uniform perturbation destroys the policy.
   - https://openaccess.thecvf.com/content/WACV2024W/MAP-A/papers/Sharma_Investigating_Weight-Perturbed_Deep_Neural_Networks_With_Application_in_Iris_Presentation_WACVW_2024_paper.pdf

6. **CVaR-CPO** — Zhang et al., "CVaR-Constrained Policy Optimization for Safe RL," IEEE TNNLS 2024.
   - Risk-aware policy optimization using CVaR. ASRA's risk evaluator is simpler but the motivation (risk-aware decisions) is shared.
   - https://pubmed.ncbi.nlm.nih.gov/38393836/

7. **Random Weight Perturbation** — "Revisiting Random Weight Perturbation for Efficiently Improving Generalization," arXiv 2024.
   - Shows weight noise injection helps escape local optima. Relevant to ASRA's Gaussian perturbation — but ASRA perturbs for safety, not generalization.
   - https://arxiv.org/abs/2404.00357

8. **Risk-Aware Classification** — "Risk-aware classification via uncertainty quantification," Expert Systems with Applications 2024.
   - Integrates risk awareness into classifiers using evidential deep learning. Related to ASRA's risk evaluator concept.
   - https://www.sciencedirect.com/science/article/pii/S0957417424027738

9. **Safe Pruning LoRA** — "Safe Pruning LoRA: Robust Distance-Guided Pruning for Safety Alignment," TACL 2025.
   - Prunes LoRA weights to maintain safety. Complementary to ASRA which perturbs LoRA weights temporarily.
   - https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.44/133861

10. **Confidence Calibration Survey** — Wang, "Calibration in Deep Learning: A Survey of the State-of-the-Art," arXiv 2024.
    - Comprehensive survey on neural network calibration. ASRA's confidence channel is a form of dynamic calibration at test time.
    - https://arxiv.org/abs/2308.01222

## How to Cite in Paper

Add to Related Work section:
- SafeLoRA [new11] and Safe Pruning LoRA [new12] modify LoRA weights for safety but as one-time post-hoc corrections, not continuous inference-time modulation.
- TTRL [new13] and TARL [new14] perform test-time RL adaptation but target task performance, not safety modulation.
- Recent surveys [new15] comprehensively characterize CMDP-based safe RL; ASRA operates outside this framework as it does not optimize a constraint but modulates behavior through perturbation.
- Weight perturbation studies [new16] confirm that neural networks are sensitive to Gaussian noise, supporting our finding that untargeted perturbation degrades performance while targeted perturbation can be beneficial.

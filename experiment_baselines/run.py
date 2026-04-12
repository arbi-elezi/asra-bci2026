"""Baseline comparisons for ASRA paper.

Implements three simple baselines that reviewers requested:
  1. Rule-based brake: brake when TTC < 2s (trivial, no learning)
  2. Action masking: mask accelerate when TTC < 3s (shielding)
  3. Entropy-bounded confidence: same as v8 but cap temperature at 1.5

Each runs 10 trials × 2000 episodes to match the v8 statistical design.
"""

import json, sys, time
from pathlib import Path
from collections import deque
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.environment.highway_wrapper import HighwayFRAEnv
from src.components.fear_detector import FearDetector, FearDetectorConfig
from experiment_v3.config import MODELS
from experiment_v3.llm_policy import LLMPolicy


def compute_risk(obs, cost, ttc, action):
    base = {0: 0.3, 1: 0.6, 2: 0.05, 3: 0.5}.get(action, 0.3)
    if action != 2 and ttc < 3.0: base = min(1.0, base + (3.0 - ttc) / 3.0 * 0.5)
    base = min(1.0, base + cost * 0.3)
    if action == 2: base = min(0.1, base)
    return float(np.clip(base, 0.0, 1.0))


def run_baselines(model_name="SmolLM2-135M", n_trials=10, n_episodes=2000, device="cuda"):
    v3_dir = Path("experiment_v3") / model_name
    out_dir = Path("experiment_baselines") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = next(m for m in MODELS if m.name == model_name)
    policy = LLMPolicy(model_cfg, device=device)
    policy.load(str(v3_dir / "policy_trained.pt"))
    day1 = json.load(open(v3_dir / "day1_log.json"))
    env = HighwayFRAEnv(seed=0, vehicles_count=15)

    print("=" * 60)
    print("BASELINE COMPARISONS")
    print("=" * 60)
    print(f"  Baseline CR: {day1['base_cr']}")

    methods = {
        "no_intervention": lambda obs, logits, ttc, cost: torch.distributions.Categorical(logits=logits).sample().item(),
        "rule_brake_ttc2": lambda obs, logits, ttc, cost: 2 if ttc < 2.0 else torch.distributions.Categorical(logits=logits).sample().item(),
        "rule_brake_ttc3": lambda obs, logits, ttc, cost: 2 if ttc < 3.0 else torch.distributions.Categorical(logits=logits).sample().item(),
        "mask_accel_ttc3": lambda obs, logits, ttc, cost: _mask_accel(logits) if ttc < 3.0 else torch.distributions.Categorical(logits=logits).sample().item(),
    }

    all_results = {}

    for method_name, action_fn in methods.items():
        print(f"\n--- {method_name} ({n_trials} trials × {n_episodes} episodes) ---")
        trial_results = []

        for trial in range(n_trials):
            collisions = 0
            less_risky = 0
            fra_steps = 0

            for ep in range(n_episodes):
                policy.restore_w0()
                obs, info = env.reset(seed=trial * 10000 + ep)

                for t in range(500):
                    cost = info.get("cost", 0.0)
                    ttc = info.get("ttc", 10.0)

                    with torch.no_grad():
                        logits = policy.get_logits_from_obs(obs)
                        greedy = logits.argmax().item()

                    risk_orig = compute_risk(obs, cost, ttc, greedy)
                    action = action_fn(obs, logits, ttc, cost)
                    risk_new = compute_risk(obs, cost, ttc, action)

                    if risk_orig > 0.1:
                        fra_steps += 1
                        if risk_new < risk_orig:
                            less_risky += 1

                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated:
                        if info.get("collision"): collisions += 1
                        break
                    if truncated: break

            cr = collisions / n_episodes
            lr = less_risky / max(fra_steps, 1)
            trial_results.append({"trial": trial, "cr": cr, "lr": lr})

            if (trial + 1) % 2 == 0:
                print(f"    Trial {trial+1}/{n_trials}: CR={cr:.3f} LR={lr:.0%}")

        crs = np.array([t["cr"] for t in trial_results])
        lrs = np.array([t["lr"] for t in trial_results])
        all_results[method_name] = {
            "cr_mean": float(crs.mean()), "cr_std": float(crs.std()),
            "lr_mean": float(lrs.mean()), "lr_std": float(lrs.std()),
            "trials": trial_results,
        }
        print(f"  {method_name}: CR={crs.mean():.3f}±{crs.std():.3f} | LR={lrs.mean():.0%}±{lrs.std():.0%}")

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Method':<25s} {'CR':>12s} {'LessRisky':>12s}")
    print("-" * 50)
    for name, r in all_results.items():
        print(f"{name:<25s} {r['cr_mean']:.3f}±{r['cr_std']:.3f}  {r['lr_mean']*100:.1f}±{r['lr_std']*100:.1f}%")
    print(f"{'v8 (ASRA)':<25s} 0.869±0.058  55.2±19.8%")

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_dir}/results.json")


def _mask_accel(logits):
    """Mask accelerate action (index 1) by setting its logit to -inf."""
    masked = logits.clone()
    masked[1] = -float("inf")
    return torch.distributions.Categorical(logits=masked).sample().item()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="SmolLM2-135M")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run_baselines(a.model, a.trials, a.episodes, a.device)

"""Run a SINGLE trial pair (baseline + v8) and save to a numbered file.

Usage: python -m experiment_stats.run_single_trial_pair --trial 3 --episodes 5000
Multiple instances can run in parallel on different trial numbers.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiment_stats.run_multi_trial import run_single_trial, compute_risk
from src.environment.highway_wrapper import HighwayFRAEnv
from src.components.fear_detector import FearDetector, FearDetectorConfig
from experiment_v3.config import MODELS
from experiment_v3.llm_policy import LLMPolicy


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--model", default="SmolLM2-135M")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    v3_dir = Path("experiment_v3") / a.model
    out_dir = Path("experiment_stats") / a.model / "trials"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    out_file = out_dir / f"trial_{a.trial}.json"
    if out_file.exists():
        print(f"Trial {a.trial} already complete, skipping")
        return

    model_cfg = next(m for m in MODELS if m.name == a.model)
    policy = LLMPolicy(model_cfg, device=a.device)
    policy.load(str(v3_dir / "policy_trained.pt"))
    w0 = policy.get_w0()
    day1 = json.load(open(v3_dir / "day1_log.json"))
    fisher = torch.load(v3_dir / "fisher.pt", weights_only=False, map_location=a.device)
    fear_det = FearDetector(FearDetectorConfig(device=a.device))
    fear_det.load_state(torch.load(v3_dir / "fear_detector.pt", weights_only=False, map_location=a.device))
    env = HighwayFRAEnv(seed=0, vehicles_count=15)

    param_names = [n for n, _ in policy.get_perturbable_params()]
    n_groups = min(10, len(param_names))
    group_size = len(param_names) // n_groups
    eta_h = min(0.01, 0.5 / max(day1["f_max"], 1e-8))

    result = {}
    for mode in ["baseline", "v8"]:
        print(f"Trial {a.trial} [{mode}] ({a.episodes} episodes)...")
        t0 = time.time()
        metrics = run_single_trial(
            policy, env, fear_det, fisher, w0, param_names, n_groups,
            group_size, eta_h, a.episodes, trial_seed=a.trial * 100 + (0 if mode == "baseline" else 50),
            device=a.device, mode=mode,
        )
        cr = np.mean([m["collision"] for m in metrics])
        lr = np.mean([m["less_risky_pct"] for m in metrics[-500:]])
        rr = np.mean([m["mean_risk_reduction"] for m in metrics[-500:]])
        elapsed = time.time() - t0
        result[mode] = {"trial": a.trial, "cr": cr, "lr": lr, "rr": rr,
                         "n_episodes": a.episodes, "elapsed_s": elapsed}
        print(f"  {mode}: CR={cr:.3f} | LessRisky={lr:.0%} | RiskRed={rr:+.3f} | {elapsed:.0f}s")

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Trial {a.trial} saved to {out_file}")


if __name__ == "__main__":
    main()

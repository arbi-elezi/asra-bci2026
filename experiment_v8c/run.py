"""v8c — Combined mechanism with EXPONENTIAL DECAY recovery (no learning).

Control experiment: same as v8 but recovery uses fixed exponential decay
with different time constants. No learned recovery model.
Tests whether the learned recovery (v8b) actually helps vs simple decay.

Runs 3 variants: fast decay (0.85), medium (0.92), slow (0.97).
"""

import json, time, sys, copy
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.environment.highway_wrapper import HighwayFRAEnv
from src.components.fear_detector import FearDetector, FearDetectorConfig
from experiment_v3.config import MODELS
from experiment_v3.llm_policy import LLMPolicy

# Import shared components from v8
sys.path.insert(0, str(Path(__file__).parent.parent / "experiment_v8"))


def compute_risk(obs, cost, ttc, action):
    base = {0: 0.3, 1: 0.6, 2: 0.05, 3: 0.5}.get(action, 0.3)
    if action != 2 and ttc < 3.0: base = min(1.0, base + (3.0-ttc)/3.0*0.5)
    base = min(1.0, base + cost * 0.3)
    if action == 2: base = min(0.1, base)
    return float(np.clip(base, 0.0, 1.0))


def apply_gaussian(param, grad, epicenter, sigma, magnitude):
    n = param.numel()
    idx = torch.arange(n, device=param.device, dtype=torch.float32)
    sig_abs = max(1, int(sigma * n))
    g = torch.exp(-((idx - epicenter)**2) / (2*sig_abs**2))
    param.data.flatten().sub_(magnitude * g * grad.flatten())


def run_v8c(model_name="SmolLM2-135M", n_episodes=1000, device="cuda"):
    v3_dir = Path("experiment_v3") / model_name
    v8c_dir = Path("experiment_v8c") / model_name
    v8c_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("v8c: Combined + FIXED DECAY RATES (control)")
    print("=" * 60)

    model_cfg = next(m for m in MODELS if m.name == model_name)
    policy = LLMPolicy(model_cfg, device=device)
    policy.load(str(v3_dir / "policy_trained.pt"))
    w0 = policy.get_w0()
    day1 = json.load(open(v3_dir / "day1_log.json"))
    fisher = torch.load(v3_dir / "fisher.pt", weights_only=False, map_location=device)
    fear_det = FearDetector(FearDetectorConfig(device=device))
    fear_det.load_state(torch.load(v3_dir / "fear_detector.pt", weights_only=False, map_location=device))
    env = HighwayFRAEnv(seed=0, vehicles_count=15)

    param_names = [n for n, _ in policy.get_perturbable_params()]
    n_groups = min(10, len(param_names))
    group_size = len(param_names) // n_groups
    eta_h = min(0.01, 0.5 / max(day1["f_max"], 1e-8))

    # Fixed perturbation params (use v8's average learned values)
    fixed_mag = 0.03
    fixed_sigma = 0.1
    fixed_temp = 1.8

    results = {}

    for decay_rate in [0.85, 0.92, 0.97]:
        label = f"decay_{decay_rate}"
        print(f"\n--- {label} ---")

        rolling_cr, rolling_lr, rolling_rr = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)

        for ep in range(n_episodes):
            policy.restore_w0()
            param_dict = dict(policy.get_perturbable_params())
            temp_state = 1.0
            sup_state = np.zeros(4)
            obs, info = env.reset(seed=ep)
            collision, ep_fra, ep_lr, ep_rrs = False, 0, 0, []

            for t in range(500):
                cost, ttc = info.get("cost", 0.0), info.get("ttc", 10.0)
                with torch.no_grad():
                    logits = policy.get_logits_from_obs(obs)
                    greedy = logits.argmax().item()
                risk = compute_risk(obs, cost, ttc, greedy)
                fear, _ = fear_det.detect(obs, cost, ttc, greedy)

                if fear > 0.05 and risk > 0.1:
                    ep_fra += 1
                    policy.model.zero_grad(); policy.action_head.zero_grad()
                    lg = policy.get_logits_from_obs(obs)
                    torch.log_softmax(lg, dim=-1)[greedy].backward()

                    with torch.no_grad():
                        for gi in range(n_groups):
                            s, e = gi*group_size, min((gi+1)*group_size, len(param_names))
                            for pi in range(s, e):
                                p = param_dict[param_names[pi]]
                                if p.grad is None: continue
                                epi = p.grad.data.abs().flatten().argmax().item()
                                apply_gaussian(p, p.grad.data, epi, fixed_sigma, fixed_mag * risk)

                    temp_state = max(temp_state, fixed_temp)
                    sup_state[greedy] -= risk * 0.5

                    with torch.no_grad():
                        sup_t = torch.tensor(sup_state, device=logits.device, dtype=logits.dtype)
                        adj = (policy.get_logits_from_obs(obs) + sup_t) / temp_state
                        new_a = torch.distributions.Categorical(logits=adj).sample().item()
                    rr = risk - compute_risk(obs, cost, ttc, new_a)
                    ep_rrs.append(rr)
                    if rr > 0: ep_lr += 1
                    action = new_a
                else:
                    if temp_state > 1.01 or np.any(np.abs(sup_state) > 0.01):
                        sup_t = torch.tensor(sup_state, device=logits.device, dtype=logits.dtype)
                        adj = (logits + sup_t) / temp_state
                        action = torch.distributions.Categorical(logits=adj).sample().item()
                    else:
                        action = torch.distributions.Categorical(logits=logits).sample().item()

                # Fixed decay
                temp_state = 1.0 + (temp_state - 1.0) * decay_rate
                sup_state *= decay_rate
                if abs(temp_state - 1.0) < 0.001: temp_state = 1.0
                sup_state[np.abs(sup_state) < 0.001] = 0.0

                # Weight FHR
                with torch.no_grad():
                    pidx = 0
                    for name in param_names:
                        p = param_dict[name]
                        if name not in w0: pidx += p.numel(); continue
                        diff = w0[name].to(p.dtype) - p.data
                        ne = p.numel()
                        fs = fisher[pidx:pidx+ne].reshape(p.shape).to(p.dtype).to(p.device)
                        pidx += ne
                        p.data += eta_h * fs * diff

                obs, reward, terminated, truncated, info = env.step(action)
                if terminated: collision = info.get("collision", False); break
                if truncated: break

            rolling_cr.append(int(collision))
            rolling_lr.append(ep_lr / max(ep_fra, 1))
            rolling_rr.append(np.mean(ep_rrs) if ep_rrs else 0)

            if (ep + 1) % 200 == 0:
                print(f"    [{ep+1}/{n_episodes}] CR={np.mean(rolling_cr):.3f} | "
                      f"LessRisky={np.mean(rolling_lr):.0%} | RiskRed={np.mean(rolling_rr):+.3f}")

        results[label] = {
            "decay_rate": decay_rate,
            "final_cr": float(np.mean(rolling_cr)),
            "final_lr": float(np.mean(rolling_lr)),
            "final_rr": float(np.mean(rolling_rr)),
        }
        print(f"  {label}: CR={results[label]['final_cr']:.3f} | "
              f"LessRisky={results[label]['final_lr']:.0%} | "
              f"RiskRed={results[label]['final_rr']:+.3f}")

    print(f"\n{'='*60}\nv8c SUMMARY\n{'='*60}")
    for k, v in results.items():
        print(f"  {k}: CR={v['final_cr']:.3f} LessRisky={v['final_lr']:.0%} RiskRed={v['final_rr']:+.3f}")

    with open(v8c_dir / "experiment_log.json", "w") as f:
        json.dump({"version": "v8c", "mechanism": "combined_fixed_decay", "results": results,
                    "baseline_cr": day1["base_cr"]}, f, indent=2, default=str)
    print(f"Results in {v8c_dir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="SmolLM2-135M")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run_v8c(a.model, a.episodes, a.device)

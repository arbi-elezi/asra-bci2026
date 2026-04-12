"""Day 2-3: Run all 13 experimental conditions for a trained model.

Uses Day 1 artifacts: W_0, D_ref, Fisher, cost critics, fear detector.
Runs all conditions with matched seeds and proper FRA mechanism.

Usage:
  python -m experiment_v3.run_day2 --model SmolLM2-135M --seeds 1000
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from src.environment.highway_wrapper import HighwayFRAEnv
from src.agents.cost_critic import CostCriticNet
from src.components.fear_detector import FearDetector, FearDetectorConfig
from src.evaluation.metrics import bootstrap_ci, paired_bootstrap_ci, m1_collision_rate, m8_task_reward
from .config import MODELS, ExperimentConfig
from .llm_policy import LLMPolicy


def run_all_conditions(model_name: str, n_seeds: int = 1000, device: str = "cuda"):
    """Run all 13 conditions for a model that completed Day 1."""
    # Find model config
    model_cfg = next((m for m in MODELS if m.name == model_name), None)
    if model_cfg is None:
        raise ValueError(f"Model {model_name} not found in config")

    model_dir = Path("experiment_v3") / model_name
    results_dir = model_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load Day 1 artifacts
    print(f"Loading Day 1 artifacts for {model_name}...")
    day1 = json.load(open(model_dir / "day1_log.json"))

    policy = LLMPolicy(model_cfg, device=device)
    policy.load(str(model_dir / "policy_trained.pt"))

    fisher = torch.load(model_dir / "fisher.pt", weights_only=False, map_location=device)
    fear_det = FearDetector(FearDetectorConfig(device=device))
    fear_det.load_state(torch.load(model_dir / "fear_detector.pt", weights_only=False, map_location=device))

    seeds = np.load(model_dir / "seeds.npy")
    w0 = policy.get_w0()
    w0_hash = policy.get_w0_hash()

    # Derive hyperparameters from Proposition 1 constraints
    f_min, f_max = day1["f_min"], day1["f_max"]
    g_max = day1["g_max"]
    r = day1["r"]
    d_max = min(r, 5.0)
    eta_ratio_max = d_max * f_min / max(g_max, 1e-8)
    eta_h = min(0.01, 0.5 / max(f_max, 1e-8))
    eta_f = eta_h * eta_ratio_max * 0.5

    # If Prop1 eta_f is too small, use geometric search
    # The constraint eta_f/eta_h <= D_max * f_min / G_max is VERY conservative
    # We test multiple eta_f values to find the practical sweet spot
    prop1_eta_f = eta_f
    if eta_f < 1e-6:
        # Use a moderate value that respects direction but not magnitude
        eta_f = 1e-5  # Conservative practical minimum (v1 used 0.001 = catastrophic)
        print(f"  WARNING: Prop1 eta_f={prop1_eta_f:.2e} too small, using eta_f={eta_f:.2e}")
    print(f"  (Prop1 bound: {prop1_eta_f:.2e}, practical: {eta_f:.2e})")

    print(f"  eta_f={eta_f:.6f}, eta_h={eta_h:.6f}, ratio={eta_f/eta_h:.6e}")
    print(f"  W_0 hash: {w0_hash[:16]}...")

    env = HighwayFRAEnv(seed=0, vehicles_count=15)

    conditions = _get_all_conditions()
    all_summaries = {}

    for cond_name, cfg in conditions.items():
        print(f"\n{'='*50}")
        print(f"  {cond_name}: {cfg.get('desc', '')}")
        print(f"{'='*50}")

        cond_dir = results_dir / cond_name
        cond_dir.mkdir(parents=True, exist_ok=True)

        # Check if already completed (resume support)
        if (cond_dir / "summary.json").exists():
            print(f"  Already completed — skipping")
            s = json.load(open(cond_dir / "summary.json"))
            all_summaries[cond_name] = s
            continue

        cond_seeds = min(n_seeds, 500) if cond_name.startswith("C8") else n_seeds

        # Load appropriate cost critic
        critic_name = cfg.get("critic", "full")
        critic = CostCriticNet(hidden=128).to(device)
        cpath = model_dir / "cost_critics" / f"{critic_name}.pt"
        if cpath.exists():
            critic.load_state_dict(torch.load(cpath, weights_only=True, map_location=device))
        critic.eval()

        collisions, rewards, per_seed = [], [], []
        t_start = time.time()

        for idx in range(cond_seeds):
            seed = int(seeds[idx])

            # Restore W_0
            policy.restore_w0()

            result = _run_episode(
                policy, env, fear_det, critic, fisher, w0,
                cfg, seed, cond_name, eta_f, eta_h, device,
            )

            collisions.append(result["collision"])
            rewards.append(result["reward"])
            per_seed.append(result)

            if (idx + 1) % max(1, cond_seeds // 10) == 0:
                cr = np.mean(collisions)
                rate = (idx + 1) / (time.time() - t_start)
                print(f"    [{idx+1}/{cond_seeds}] CR={cr:.3f} | {rate:.1f} seeds/s")

        elapsed = time.time() - t_start
        cr_arr = np.array(collisions, dtype=float)
        cr = m1_collision_rate(cr_arr)
        cr_ci = bootstrap_ci(cr_arr)
        rw = m8_task_reward(np.array(rewards))

        summary = {
            "condition": cond_name,
            "model": model_name,
            "n_seeds": cond_seeds,
            "M1_collision_rate": cr,
            "M1_ci": cr_ci,
            "M8_mean_reward": rw,
            "w0_hash": w0_hash,
            "eta_f": eta_f,
            "eta_h": eta_h,
            "elapsed_s": elapsed,
        }

        with open(cond_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        with open(cond_dir / "per_seed.json", "w") as f:
            json.dump(per_seed, f, indent=2, default=str)

        all_summaries[cond_name] = summary
        print(f"  {cond_name}: CR={cr:.3f} [{cr_ci['ci_lower']:.3f}, {cr_ci['ci_upper']:.3f}] | Rwd={rw:.1f} | {elapsed:.0f}s")

    # Hypothesis testing
    print(f"\n{'='*60}")
    print("HYPOTHESIS TESTING")
    print(f"{'='*60}")
    _run_hypothesis_tests(results_dir, all_summaries)

    # Save overall summary
    with open(results_dir / "all_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)

    print(f"\nDay 2-3 complete for {model_name}")


def _run_episode(policy, env, fear_det, critic, fisher, w0, cfg, seed, cond_name, eta_f, eta_h, device):
    """Run a single episode for a condition."""
    obs, info = env.reset(seed=seed)
    ep_reward = 0.0
    collision = False
    wdn_max = 0.0

    for t in range(500):
        cost = info.get("cost", 0.0)
        ttc = info.get("ttc", 10.0)

        if not cfg.get("fra", False):
            # Baseline
            if cfg.get("hard_override") and cost > 0.4:
                action = 2  # BRAKE
            else:
                action = policy.get_action(obs, deterministic=False)
        else:
            # FRA active
            with torch.no_grad():
                logits = policy.get_logits_from_obs(obs)
                greedy = logits.argmax().item()

            fear, _ = fear_det.detect(obs, cost, ttc, greedy)

            # Safe action
            obs_t = torch.tensor(obs, dtype=torch.float32, device=torch.device(device))
            with torch.no_grad():
                costs_per_a = critic.predict_state(obs_t)
                safe_action = costs_per_a.argmin(dim=-1).item()

            # DR
            if cfg.get("dr", True) and fear > 0.05:
                policy.model.zero_grad()
                policy.action_head.zero_grad()
                logits = policy.get_logits_from_obs(obs)
                lp = torch.log_softmax(logits, dim=-1)[safe_action]
                lp.backward()
                with torch.no_grad():
                    for _, p in policy.get_perturbable_params():
                        if p.grad is not None:
                            p.data += eta_f * fear * p.grad.data

            # FHR
            if cfg.get("fhr", True):
                fhr_mode = cfg.get("fhr_mode", "fisher")
                with torch.no_grad():
                    pidx = 0
                    for name, p in policy.get_perturbable_params():
                        if name not in w0:
                            pidx += p.numel()
                            continue
                        diff = w0[name].to(p.dtype) - p.data
                        ne = p.numel()
                        if fhr_mode == "fisher":
                            fs = fisher[pidx:pidx+ne].reshape(p.shape).to(p.dtype).to(p.device)
                            pidx += ne
                            p.data += eta_h * fs * diff
                        else:
                            pidx += ne
                            p.data += eta_h * diff

            # BC
            if cfg.get("bc", True):
                with torch.no_grad():
                    wt_snap = {n: p.data.clone() for n, p in policy.get_perturbable_params()}
                    policy.restore_w0()
                    w0_probs = torch.softmax(policy.get_logits_from_obs(obs), dim=-1)
                    for n, p in policy.get_perturbable_params():
                        if n in wt_snap:
                            p.data.copy_(wt_snap[n])

                policy.model.zero_grad()
                policy.action_head.zero_grad()
                wt_logits = policy.get_logits_from_obs(obs)
                wt_lp = torch.log_softmax(wt_logits, dim=-1)
                kl = torch.sum(w0_probs.detach() * (torch.log(w0_probs.detach() + 1e-8) - wt_lp))
                kl.backward()
                with torch.no_grad():
                    for _, p in policy.get_perturbable_params():
                        if p.grad is not None:
                            p.data -= 1e-5 * p.grad.data

            # SCL
            with torch.no_grad():
                new_logits = policy.get_logits_from_obs(obs)
                probs = torch.softmax(new_logits, dim=-1)
                alpha = 1.0 / (1.0 + np.exp(-10 * (fear - 0.5)))
                mixed = (1 - alpha) * probs
                mixed[safe_action] += alpha
                action = torch.multinomial(mixed, 1).item()

            # Track WDN
            with torch.no_grad():
                wdn = sum(((p.data.float() - w0[n].float())**2).sum().item()
                          for n, p in policy.get_perturbable_params() if n in w0) ** 0.5
                wdn_max = max(wdn_max, wdn)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        if terminated:
            collision = info.get("collision", False)
            break
        if truncated:
            break

    return {
        "seed": seed, "condition": cond_name,
        "collision": int(collision), "reward": ep_reward,
        "wdn_max": wdn_max, "n_steps": t + 1,
    }


def _get_all_conditions():
    return {
        "C1":  {"fra": False, "desc": "Baseline (no FRA)"},
        "C2":  {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True, "desc": "Full FRA"},
        "C3a": {"fra": True, "dr": True, "fhr": True, "fhr_mode": "l2", "bc": False, "desc": "L2-HR (ablation A1a)"},
        "C3b": {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": False, "desc": "Fisher no BC (A1b)"},
        "C4":  {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True, "desc": "No GTCC (A2)"},
        "C5":  {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True, "desc": "No FMS (A3)"},
        "C6":  {"fra": True, "dr": False, "fhr": True, "fhr_mode": "fisher", "bc": True, "desc": "No DR (A4)"},
        "C7":  {"fra": False, "hard_override": True, "desc": "Hard override"},
        "C8a": {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True, "critic": "degraded_10pct", "desc": "10% D_ref"},
        "C8b": {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True, "critic": "degraded_25pct", "desc": "25% D_ref"},
        "C8c": {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True, "critic": "degraded_50pct", "desc": "50% D_ref"},
        "C8d": {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True, "critic": "biased_fast_02", "desc": "Bias x0.2"},
        "C8e": {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True, "critic": "biased_fast_05", "desc": "Bias x0.5"},
    }


def _run_hypothesis_tests(results_dir, summaries):
    """Run bootstrap CIs for all testable hypotheses."""
    results = {}

    def load_cr(cond):
        with open(results_dir / cond / "per_seed.json") as f:
            data = json.load(f)
        return np.array([float(d["collision"]) for d in data], dtype=float)

    tests = [
        ("H1", "C1", "C2", "CR(C2) < CR(C1) — FRA reduces collisions"),
        ("H8", "C6", "C2", "DR contributes (C2 < C6 on adversarial)"),
        ("H15a", "C8a", "C1", "10% D_ref worse than baseline"),
        ("H15b", "C8b", "C1", "25% D_ref worse than baseline"),
        ("H15c", "C8c", "C1", "50% D_ref worse than baseline"),
        ("H16", "C8d", "C1", "Severe bias worse than baseline"),
        ("H17", "C8e", "C1", "Moderate bias worse than baseline"),
    ]

    for label, cx, cy, claim in tests:
        try:
            x, y = load_cr(cx), load_cr(cy)
            n = min(len(x), len(y))
            h = paired_bootstrap_ci(x[:n], y[:n])
            confirmed = h["excludes_zero"] and h["point_estimate"] > 0
            results[label] = {
                "claim": claim, "delta": h["point_estimate"],
                "ci": [h["ci_lower"], h["ci_upper"]],
                "excludes_zero": h["excludes_zero"], "confirmed": confirmed,
            }
            s = "CONFIRMED" if confirmed else "FALSIFIED"
            print(f"  {label:>5s}: {s:>10s}  Delta={h['point_estimate']:>+.4f}  CI=[{h['ci_lower']:.4f}, {h['ci_upper']:.4f}]")
        except Exception as e:
            print(f"  {label}: SKIPPED ({e})")

    with open(results_dir / "hypothesis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    confirmed = sum(1 for v in results.values() if v.get("confirmed"))
    print(f"\n  TOTAL: {confirmed}/{len(results)} confirmed")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Day 2-3: Run experimental conditions")
    parser.add_argument("--model", required=True, help="Model name from Day 1")
    parser.add_argument("--seeds", type=int, default=200, help="Seeds per condition")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_all_conditions(args.model, n_seeds=args.seeds, device=args.device)


if __name__ == "__main__":
    main()

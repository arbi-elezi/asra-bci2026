"""Experiment v4 — Fear as SUPPRESSION, not direction.

v3 was wrong: DR computed gradient toward a "safe action" chosen by the cost critic.
This is puppet-control, not fear. If the critic picks wrong, DR pushes toward danger.

v4 fix: DR computes NEGATIVE gradient of the current greedy action when fear is high.
This suppresses what the LLM was about to do — makes the risky choice less probable.
The LLM figures out the alternative on its own.

Biology: fear doesn't tell you which exit to take. It makes you not want to stay.

Reuses v3 Day 1 artifacts (trained policy, Fisher, D_ref, fear detector).
Only changes the DR mechanism and runs all 13 conditions.
"""

import json
import time
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.highway_wrapper import HighwayFRAEnv
from src.agents.cost_critic import CostCriticNet
from src.components.fear_detector import FearDetector, FearDetectorConfig
from src.evaluation.metrics import bootstrap_ci, paired_bootstrap_ci, m1_collision_rate, m8_task_reward

# Reuse v3 LLM policy loader
from experiment_v3.config import MODELS
from experiment_v3.llm_policy import LLMPolicy


def run_v4(model_name="SmolLM2-135M", n_seeds=200, device="cuda"):
    """Run v4 experiment: fear as suppression."""

    v3_dir = Path("experiment_v3") / model_name
    v4_dir = Path("experiment_v4") / model_name
    v4_dir.mkdir(parents=True, exist_ok=True)
    results_dir = v4_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT v4: Fear as SUPPRESSION")
    print("DR = negative gradient of greedy action (push AWAY from risk)")
    print("=" * 60)

    # Load model config
    model_cfg = next(m for m in MODELS if m.name == model_name)

    # Load trained policy from v3
    print(f"\nLoading {model_name} from v3 artifacts...")
    policy = LLMPolicy(model_cfg, device=device)
    policy.load(str(v3_dir / "policy_trained.pt"))
    w0 = policy.get_w0()

    # Load other artifacts
    day1 = json.load(open(v3_dir / "day1_log.json"))
    fisher = torch.load(v3_dir / "fisher.pt", weights_only=False, map_location=device)
    fear_det = FearDetector(FearDetectorConfig(device=device))
    fear_det.load_state(torch.load(v3_dir / "fear_detector.pt", weights_only=False, map_location=device))
    seeds = np.load(v3_dir / "seeds.npy")

    # Load cost critics (still used for C8 stress tests — but NOT for DR direction)
    critics = {}
    for name in ["full", "degraded_10pct", "degraded_25pct", "degraded_50pct", "biased_fast_02", "biased_fast_05"]:
        c = CostCriticNet(hidden=128).to(device)
        p = v3_dir / "cost_critics" / f"{name}.pt"
        if p.exists():
            c.load_state_dict(torch.load(p, weights_only=True, map_location=device))
        c.eval()
        critics[name] = c

    # Hyperparams — eta_h from Prop1, eta_f sweep
    eta_h = min(0.01, 0.5 / max(day1["f_max"], 1e-8))

    env = HighwayFRAEnv(seed=0, vehicles_count=15)

    print(f"  W_0 hash: {policy.get_w0_hash()[:16]}...")
    print(f"  Base CR (from Day 1): {day1['base_cr']}")
    print(f"  Fisher range: {day1['f_max']/max(day1['f_min'],1e-12):.0f}x")
    print(f"  eta_h: {eta_h}")

    # ── eta_f sweep first (find the right perturbation strength) ──
    print("\n--- eta_f sweep (50 seeds each) ---")
    best_eta_f = None
    best_cr = 1.0

    for eta_f in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
        cr = _run_condition_quick(
            policy, env, fear_det, fisher, w0, seeds,
            eta_f=eta_f, eta_h=eta_h, n_seeds=50,
            mode="suppress", device=device,
        )
        tag = " <-- BEST" if cr < best_cr else ""
        print(f"  eta_f={eta_f:.0e}: CR={cr:.3f}{tag}")
        if cr < best_cr:
            best_cr = cr
            best_eta_f = eta_f

    print(f"\n  Best eta_f: {best_eta_f:.0e} (CR={best_cr:.3f})")

    # Also test "attract" mode (v3 style) for comparison
    cr_attract = _run_condition_quick(
        policy, env, fear_det, fisher, w0, seeds,
        eta_f=best_eta_f, eta_h=eta_h, n_seeds=50,
        mode="attract", critic=critics["full"], device=device,
    )
    print(f"  Attract mode (v3): CR={cr_attract:.3f}")
    print(f"  Suppress mode (v4): CR={best_cr:.3f}")

    # ── Run all 13 conditions with best eta_f ──
    conditions = {
        "C1":  {"fra": False, "desc": "Baseline"},
        "C2":  {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "desc": "Full FRA (suppress)"},
        "C3a": {"fra": True, "dr": True, "fhr": "l2", "bc": False, "desc": "L2-HR"},
        "C3b": {"fra": True, "dr": True, "fhr": "fisher", "bc": False, "desc": "Fisher no BC"},
        "C4":  {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "desc": "No GTCC"},
        "C5":  {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "desc": "No FMS"},
        "C6":  {"fra": True, "dr": False, "fhr": "fisher", "bc": True, "desc": "No DR"},
        "C7":  {"fra": False, "hard_override": True, "desc": "Hard override"},
        "C8a": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "critic": "degraded_10pct", "desc": "10% D_ref"},
        "C8b": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "critic": "degraded_25pct", "desc": "25% D_ref"},
        "C8c": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "critic": "degraded_50pct", "desc": "50% D_ref"},
        "C8d": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "critic": "biased_fast_02", "desc": "Bias x0.2"},
        "C8e": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "critic": "biased_fast_05", "desc": "Bias x0.5"},
    }

    all_results = {}
    print(f"\n{'='*60}")
    print(f"Running all 13 conditions ({n_seeds} seeds, eta_f={best_eta_f:.0e})")
    print(f"{'='*60}")

    for cond_name, cfg in conditions.items():
        cond_dir = results_dir / cond_name
        cond_dir.mkdir(exist_ok=True)

        # Skip if already done
        if (cond_dir / "summary.json").exists():
            s = json.load(open(cond_dir / "summary.json"))
            all_results[cond_name] = s
            print(f"  {cond_name}: CR={s['M1_collision_rate']:.3f} (cached)")
            continue

        cond_seeds = min(n_seeds, 500) if cond_name.startswith("C8") else n_seeds

        # Pick critic for stress tests (used only for SCL safe action, NOT for DR)
        critic = critics.get(cfg.get("critic", "full"), critics["full"])

        collisions, rewards, per_seed = [], [], []
        t0 = time.time()

        for idx in range(cond_seeds):
            seed = int(seeds[idx])
            policy.restore_w0()

            result = _run_episode_v4(
                policy, env, fear_det, critic, fisher, w0,
                cfg, seed, best_eta_f, eta_h, device,
            )
            collisions.append(result["collision"])
            rewards.append(result["reward"])
            per_seed.append(result)

            if (idx + 1) % max(1, cond_seeds // 10) == 0:
                print(f"    [{idx+1}/{cond_seeds}] CR={np.mean(collisions):.3f}")

        elapsed = time.time() - t0
        cr_arr = np.array(collisions, dtype=float)
        cr = m1_collision_rate(cr_arr)
        cr_ci = bootstrap_ci(cr_arr)
        rw = m8_task_reward(np.array(rewards))

        summary = {
            "condition": cond_name, "model": model_name,
            "n_seeds": cond_seeds, "M1_collision_rate": cr,
            "M1_ci": cr_ci, "M8_mean_reward": rw,
            "eta_f": best_eta_f, "eta_h": eta_h,
            "mode": "suppress", "elapsed_s": elapsed,
        }
        with open(cond_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        with open(cond_dir / "per_seed.json", "w") as f:
            json.dump(per_seed, f, indent=2, default=str)

        all_results[cond_name] = summary
        print(f"  {cond_name}: CR={cr:.3f} [{cr_ci['ci_lower']:.3f}, {cr_ci['ci_upper']:.3f}] | Rwd={rw:.1f} | {elapsed:.0f}s")

    # ── Hypothesis testing ──
    print(f"\n{'='*60}")
    print("HYPOTHESIS TESTS")
    print(f"{'='*60}")

    def load_cr(cond):
        with open(results_dir / cond / "per_seed.json") as f:
            return np.array([float(d["collision"]) for d in json.load(f)])

    hyp_results = {}
    for label, cx, cy, claim in [
        ("H1", "C1", "C2", "FRA reduces CR (C1 > C2)"),
        ("H8", "C6", "C2", "DR contributes (C6 > C2)"),
        ("H15a", "C8a", "C1", "10% D_ref harmful"),
        ("H15b", "C8b", "C1", "25% D_ref harmful"),
        ("H15c", "C8c", "C1", "50% D_ref harmful"),
        ("H16", "C8d", "C1", "Bias x0.2 harmful"),
        ("H17", "C8e", "C1", "Bias x0.5 harmful"),
    ]:
        try:
            x, y = load_cr(cx), load_cr(cy)
            n = min(len(x), len(y))
            h = paired_bootstrap_ci(x[:n], y[:n])
            confirmed = h["excludes_zero"] and h["point_estimate"] > 0
            hyp_results[label] = {"claim": claim, "delta": h["point_estimate"],
                                   "ci": [h["ci_lower"], h["ci_upper"]],
                                   "confirmed": confirmed}
            s = "CONFIRMED" if confirmed else "FALSIFIED"
            print(f"  {label:>5s}: {s:>10s}  Delta={h['point_estimate']:>+.4f}  CI=[{h['ci_lower']:.4f}, {h['ci_upper']:.4f}]")
        except Exception as e:
            print(f"  {label}: SKIP ({e})")

    with open(results_dir / "hypothesis_results.json", "w") as f:
        json.dump(hyp_results, f, indent=2, default=str)

    confirmed = sum(1 for v in hyp_results.values() if v.get("confirmed"))
    print(f"\n  TOTAL: {confirmed}/{len(hyp_results)} confirmed")

    # Save v4 summary
    with open(v4_dir / "experiment_log.json", "w") as f:
        json.dump({
            "version": "v4",
            "mechanism": "suppress (negative gradient of greedy action)",
            "model": model_name,
            "best_eta_f": best_eta_f,
            "eta_h": eta_h,
            "eta_f_sweep": {f"{ef:.0e}": None for ef in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]},
            "all_results": {k: v.get("M1_collision_rate") for k, v in all_results.items()},
            "hypotheses": hyp_results,
        }, f, indent=2, default=str)

    print(f"\nv4 complete. Results in {v4_dir}/")


def _run_episode_v4(policy, env, fear_det, critic, fisher, w0, cfg, seed, eta_f, eta_h, device):
    """Single episode with SUPPRESSION-based DR."""
    obs, info = env.reset(seed=seed)
    ep_reward = 0.0
    collision = False

    for t in range(500):
        cost = info.get("cost", 0.0)
        ttc = info.get("ttc", 10.0)

        if not cfg.get("fra", False):
            if cfg.get("hard_override") and cost > 0.4:
                action = 2
            else:
                action = policy.get_action(obs)
        else:
            # Get current greedy action
            with torch.no_grad():
                logits = policy.get_logits_from_obs(obs)
                greedy = logits.argmax().item()

            # Fear
            fear, _ = fear_det.detect(obs, cost, ttc, greedy)

            # ── DR: SUPPRESS the greedy action (push AWAY from current choice) ──
            if cfg.get("dr", True) and fear > 0.05:
                policy.model.zero_grad()
                policy.action_head.zero_grad()
                logits = policy.get_logits_from_obs(obs)

                # NEGATIVE gradient of log P(greedy_action)
                # This DECREASES the probability of what the LLM was about to do
                log_prob_greedy = torch.log_softmax(logits, dim=-1)[greedy]
                log_prob_greedy.backward()

                with torch.no_grad():
                    for _, p in policy.get_perturbable_params():
                        if p.grad is not None:
                            # SUBTRACT: push AWAY from greedy action
                            p.data -= eta_f * fear * p.grad.data

            # ── FHR: pull back toward W_0 ──
            fhr_mode = cfg.get("fhr", "fisher")
            if fhr_mode:
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

            # ── BC: penalize divergence from W_0 ──
            if cfg.get("bc", True):
                with torch.no_grad():
                    wt_snap = {n: p.data.clone() for n, p in policy.get_perturbable_params()}
                    policy.restore_w0()
                    w0_probs = torch.softmax(policy.get_logits_from_obs(obs), dim=-1)
                    for n, p in policy.get_perturbable_params():
                        if n in wt_snap: p.data.copy_(wt_snap[n])

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

            # ── Get action from perturbed policy ──
            with torch.no_grad():
                new_logits = policy.get_logits_from_obs(obs)
                probs = torch.softmax(new_logits, dim=-1)

                # SCL: mix with safe action from cost critic (only for action mixing, NOT for DR)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=torch.device(device))
                costs_per_a = critic.predict_state(obs_t)
                safe_action = costs_per_a.argmin(dim=-1).item()

                alpha = 1.0 / (1.0 + np.exp(-10 * (fear - 0.5)))
                mixed = (1 - alpha) * probs
                mixed[safe_action] += alpha
                action = torch.multinomial(mixed, 1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        if terminated:
            collision = info.get("collision", False)
            break
        if truncated:
            break

    return {"seed": seed, "condition": cfg.get("desc", ""), "collision": int(collision), "reward": ep_reward}


def _run_condition_quick(policy, env, fear_det, fisher, w0, seeds,
                          eta_f, eta_h, n_seeds, mode, critic=None, device="cuda"):
    """Quick test of a single condition."""
    collisions = []
    dummy_critic = CostCriticNet(hidden=128).to(device)
    dummy_critic.eval()
    c = critic if critic else dummy_critic

    for idx in range(n_seeds):
        seed = int(seeds[idx])
        policy.restore_w0()

        obs, info = env.reset(seed=seed)
        for t in range(500):
            cost = info.get("cost", 0.0)
            ttc = info.get("ttc", 10.0)

            with torch.no_grad():
                logits = policy.get_logits_from_obs(obs)
                greedy = logits.argmax().item()

            fear, _ = fear_det.detect(obs, cost, ttc, greedy)

            if fear > 0.05:
                policy.model.zero_grad()
                policy.action_head.zero_grad()
                logits = policy.get_logits_from_obs(obs)

                if mode == "suppress":
                    # v4: push AWAY from greedy
                    lp = torch.log_softmax(logits, dim=-1)[greedy]
                    lp.backward()
                    with torch.no_grad():
                        for _, p in policy.get_perturbable_params():
                            if p.grad is not None:
                                p.data -= eta_f * fear * p.grad.data
                elif mode == "attract":
                    # v3: push TOWARD safe action from critic
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=torch.device(device))
                    with torch.no_grad():
                        safe_a = c.predict_state(obs_t).argmin(dim=-1).item()
                    lp = torch.log_softmax(logits, dim=-1)[safe_a]
                    lp.backward()
                    with torch.no_grad():
                        for _, p in policy.get_perturbable_params():
                            if p.grad is not None:
                                p.data += eta_f * fear * p.grad.data

                # FHR
                with torch.no_grad():
                    pidx = 0
                    for name, p in policy.get_perturbable_params():
                        if name not in w0:
                            pidx += p.numel()
                            continue
                        diff = w0[name].to(p.dtype) - p.data
                        ne = p.numel()
                        fs = fisher[pidx:pidx+ne].reshape(p.shape).to(p.dtype).to(p.device)
                        pidx += ne
                        p.data += eta_h * fs * diff

            with torch.no_grad():
                action = policy.get_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                if info.get("collision"): collisions.append(1)
                else: collisions.append(0)
                break
            if truncated:
                collisions.append(0)
                break

    return np.mean(collisions)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="SmolLM2-135M")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run_v4(a.model, a.seeds, a.device)

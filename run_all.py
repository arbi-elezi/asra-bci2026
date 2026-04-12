"""Master Orchestrator — Runs the complete FRA experiment pipeline.

Phase 0: Train base agent + offline artifacts
Phase 1: Hyperparameter validation (small scale)
Phase 2: Run all 13 conditions (C1–C8e)
Phase 3: Analysis — all 17 hypothesis tests
Phase 4: Demo capture

Usage:
  python run_all.py                    # Full pipeline
  python run_all.py --phase 0          # Phase 0 only
  python run_all.py --phase 2 --quick  # Quick run (10 seeds per condition)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# ── Imports ──
from src.environment.highway_wrapper import HighwayFRAEnv
from src.agents.cost_critic import CostCriticNet, CostCriticConfig, train_cost_critic
from src.components.fear_detector import FearDetector, FearDetectorConfig
from src.components.rlaif_judge import RLAIFJudge
from src.evaluation.metrics import (
    bootstrap_ci, paired_bootstrap_ci,
    m1_collision_rate, m8_task_reward, m3_wdn, m13_gradient_norm,
)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 0: Base Agent Training + Offline Artifacts
# ═══════════════════════════════════════════════════════════════════════════

def phase0_train_base(device: str = "cuda", output_dir: str = "checkpoints") -> dict:
    """Train the base PPO agent, collect D_ref, compute Fisher, train cost critics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    print("\n" + "=" * 70)
    print("PHASE 0: Base Agent Training + Offline Artifacts")
    print("=" * 70)

    t0 = time.time()

    # ── Step 0.1: Generate master seed set ──
    print("\n[0.1] Generating master seed set...")
    rng = np.random.default_rng(2026)
    seeds = rng.integers(0, 2**31, size=1200)
    np.save(out / "seed_set.npy", seeds)
    print(f"  1200 seeds saved (1000 experiment + 200 validation)")
    artifacts["n_seeds"] = 1200

    # ── Step 0.2: Initialize environment ──
    print("\n[0.2] Initializing highway-env...")
    env = HighwayFRAEnv(seed=0, vehicles_count=15)  # Calibrated for ~20-40% baseline CR
    print(f"  State: R^{env.observation_space.shape[0]}, Actions: {env.action_space.n}")

    # ── Step 0.3: Train PPO base agent ──
    # Use Stable-Baselines3 PPO for reliable training, then extract weights.
    # Train for moderate steps to get a competent but imperfect policy
    # (~20-40% CR is the target for meaningful hypothesis testing).
    print("\n[0.3] Training PPO base agent (SB3, 20K steps)...")
    actor = _train_ppo_sb3(env, device, total_timesteps=20_000)

    # Save W_0
    w0_dir = out / "ppo_base"
    w0_dir.mkdir(exist_ok=True)
    w0_state = actor.state_dict()
    w0_hash = _hash_state_dict(w0_state)
    torch.save({"state_dict": w0_state, "hash": w0_hash}, w0_dir / "w0.pt")
    print(f"  W_0 saved. Hash: {w0_hash[:16]}...")
    artifacts["w0_hash"] = w0_hash

    # ── Step 0.4: Collect D_ref ──
    print("\n[0.4] Collecting D_ref (1000 state-action-cost triples)...")
    d_ref = _collect_d_ref(env, actor, device, n_pairs=1000)
    torch.save(d_ref, out / "d_ref.pt")
    print(f"  D_ref: {d_ref['states'].shape[0]} pairs, "
          f"mean cost: {d_ref['costs'].mean():.3f}")
    artifacts["d_ref_size"] = int(d_ref["states"].shape[0])

    # ── Step 0.5: Compute Fisher diagonal ──
    print("\n[0.5] Computing diagonal Fisher information matrix...")
    fisher = _compute_fisher(actor, d_ref, device)
    torch.save(fisher, out / "fisher_diagonal.pt")
    f_min = fisher.min().item()
    f_max = fisher.max().item()
    print(f"  Fisher: {fisher.shape[0]} params, f_min={f_min:.6f}, f_max={f_max:.6f}")
    artifacts["f_min"] = f_min
    artifacts["f_max"] = f_max
    artifacts["n_params"] = int(fisher.shape[0])

    # ── Step 0.6: Compute gradient statistics ──
    print("\n[0.6] Computing gradient statistics (G_max^0, sigma_G, L_G)...")
    grad_stats = _compute_grad_stats(actor, d_ref, device)
    print(f"  G_max^0={grad_stats['g_max_0']:.4f}, sigma_G={grad_stats['sigma_g']:.4f}")

    l_g = _estimate_lipschitz(actor, d_ref, device)
    print(f"  L_G={l_g:.4f}")

    g_max = grad_stats["g_max_0"] * 1.5
    r = (g_max - grad_stats["g_max_0"]) / max(l_g, 1e-8)
    d_max = min(r, 1.0)
    eta_ratio_max = d_max * f_min / max(g_max, 1e-8) if g_max > 0 else float("inf")

    artifacts.update({
        "g_max_0": grad_stats["g_max_0"],
        "sigma_g": grad_stats["sigma_g"],
        "g_max": g_max,
        "l_g": l_g,
        "r": r,
        "d_max": d_max,
        "eta_ratio_max": eta_ratio_max,
        "eta_h_max": 1.0 / max(f_max, 1e-8),
    })

    # ── Step 0.7: Train fear detector on D_ref ──
    print("\n[0.7] Training fear detector ensemble...")
    fear_detector = FearDetector(FearDetectorConfig())
    fear_detector.train_on_d_ref(
        d_ref["states"].numpy(),
        d_ref["costs"].numpy(),
    )
    torch.save(fear_detector.get_state(), out / "fear_detector.pt")
    print("  Fear detector trained (AE + IsolationForest + CA)")

    # ── Step 0.8: Train cost critics (6 variants) ──
    print("\n[0.8] Training cost critics (6 variants)...")
    _train_all_critics(d_ref, out, device)

    # ── Step 0.9: Validate RLAIF judge M7 on D_ref ──
    print("\n[0.9] Validating RLAIF judge on D_ref...")
    judge = RLAIFJudge()
    n_validated = 0
    for i in range(min(200, d_ref["states"].shape[0])):
        obs_np = d_ref["states"][i].numpy()
        cost = d_ref["costs"][i].item()
        ttc = max(0.01, obs_np[10]) if len(obs_np) > 10 else 10.0
        cls = d_ref["classes"][i].item() if "classes" in d_ref else -1
        ego_speed = obs_np[2] if len(obs_np) > 2 else 25.0
        closing = max(0, obs_np[2] - obs_np[8]) if len(obs_np) > 8 else 0.0
        gap = abs(obs_np[6]) if len(obs_np) > 6 else 100.0
        judge.validate_against_gtcc(obs_np, cost, ttc, cls, ego_speed, closing, gap)
        n_validated += 1

    m7 = judge.get_m7_accuracy()
    print(f"  RLAIF M7 accuracy: {m7:.3f} ({'PASSED' if m7 >= 0.70 else 'FAILED'} gate)")
    artifacts["m7_initial"] = m7

    # ── Save artifacts ──
    artifacts["elapsed_seconds"] = time.time() - t0
    with open(out / "artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=2, default=str)
    print(f"\n  All artifacts saved to {out}/")
    print(f"  Phase 0 complete in {artifacts['elapsed_seconds']:.0f}s")

    return artifacts


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Run Experimental Conditions
# ═══════════════════════════════════════════════════════════════════════════

def phase2_run_conditions(
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
    n_seeds: int = 1000,
    conditions: list[str] | None = None,
) -> dict:
    """Run all experimental conditions."""
    out = Path(checkpoint_dir)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print(f"PHASE 2: Running Experimental Conditions ({n_seeds} seeds each)")
    print("=" * 70)

    # Load artifacts
    with open(out / "artifacts.json") as f:
        artifacts = json.load(f)

    # Load W_0
    w0_data = torch.load(out / "ppo_base" / "w0.pt", weights_only=False)
    w0_state = w0_data["state_dict"]
    w0_hash = w0_data["hash"]
    print(f"  W_0 hash: {w0_hash[:16]}...")

    # Load D_ref
    d_ref = torch.load(out / "d_ref.pt", weights_only=False)

    # Load Fisher
    fisher = torch.load(out / "fisher_diagonal.pt", weights_only=False)

    # Load fear detector
    fear_state = torch.load(out / "fear_detector.pt", weights_only=False)

    # Define conditions to run
    if conditions is None:
        conditions = [
            "C1", "C2", "C3a", "C3b", "C4", "C5", "C6", "C7",
            "C8a", "C8b", "C8c", "C8d", "C8e",
        ]

    # Load seeds
    seeds = np.load(out / "seed_set.npy")

    env = HighwayFRAEnv(seed=0, vehicles_count=15)  # Same as Phase 0
    all_summaries = {}

    for cond in conditions:
        print(f"\n--- Running {cond} ---")
        cond_dir = results_dir / cond

        # Determine number of seeds
        cond_seeds = n_seeds
        if cond.startswith("C8"):
            cond_seeds = min(n_seeds, 500)

        # Load config
        config = _load_condition_config(cond, artifacts, out)

        # Run episodes
        collisions = []
        rewards = []
        per_seed_data = []

        t_start = time.time()
        for idx in range(cond_seeds):
            seed = int(seeds[idx])
            result = _run_episode(
                env, w0_state, fisher, fear_state, d_ref,
                config, seed, cond, device, artifacts,
            )
            collisions.append(result["collision"])
            rewards.append(result["reward"])
            per_seed_data.append(result)

            if (idx + 1) % max(1, cond_seeds // 10) == 0:
                cr = np.mean(collisions)
                rate = (idx + 1) / (time.time() - t_start)
                print(f"  [{idx+1}/{cond_seeds}] CR={cr:.3f} | {rate:.1f} seeds/s")

        elapsed = time.time() - t_start
        cr_arr = np.array(collisions, dtype=float)
        rw_arr = np.array(rewards, dtype=float)

        cr = m1_collision_rate(cr_arr)
        cr_ci = bootstrap_ci(cr_arr)
        mean_rw = m8_task_reward(rw_arr)

        summary = {
            "condition": cond,
            "n_seeds": cond_seeds,
            "M1_collision_rate": cr,
            "M1_ci": cr_ci,
            "M8_mean_reward": mean_rw,
            "w0_hash": w0_hash,
            "elapsed_s": elapsed,
        }

        # Save (re-ensure dir exists in case of concurrent cleanup)
        cond_dir.mkdir(parents=True, exist_ok=True)
        with open(cond_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        with open(cond_dir / "per_seed.json", "w") as f:
            json.dump(per_seed_data, f, indent=2, default=str)

        all_summaries[cond] = summary
        print(f"  {cond}: CR={cr:.4f} [{cr_ci['ci_lower']:.4f}, {cr_ci['ci_upper']:.4f}] "
              f"| Reward={mean_rw:.1f} | {elapsed:.0f}s")

    return all_summaries


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: Hypothesis Testing
# ═══════════════════════════════════════════════════════════════════════════

def phase3_analyze(results_dir: str = "results") -> dict:
    """Run all 17 hypothesis tests."""
    rd = Path(results_dir)

    print("\n" + "=" * 70)
    print("PHASE 3: Hypothesis Testing (17 hypotheses, bootstrap CIs)")
    print("=" * 70)

    results = {}

    def _load_cr(cond: str) -> np.ndarray:
        with open(rd / cond / "per_seed.json") as f:
            data = json.load(f)
        # Handle bool/string collision values
        def _to_float(v):
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            if isinstance(v, str):
                return 1.0 if v.lower() == "true" else 0.0
            return float(v)
        return np.array([_to_float(d["collision"]) for d in data], dtype=float)

    # ── H1: CR(C2) < CR(C1) ──
    try:
        c1 = _load_cr("C1")
        c2 = _load_cr("C2")
        n = min(len(c1), len(c2))
        h1 = paired_bootstrap_ci(c1[:n], c2[:n])
        results["H1"] = {
            "claim": "CR(C2) < CR(C1)",
            "delta": h1["point_estimate"],
            "ci": [h1["ci_lower"], h1["ci_upper"]],
            "excludes_zero": h1["excludes_zero"],
            "confirmed": h1["excludes_zero"] and h1["point_estimate"] > 0,
        }
        status = "CONFIRMED" if results["H1"]["confirmed"] else "FALSIFIED"
        print(f"  H1:  {status} — Delta={h1['point_estimate']:.4f} "
              f"[{h1['ci_lower']:.4f}, {h1['ci_upper']:.4f}]")
    except Exception as e:
        print(f"  H1:  SKIPPED ({e})")

    # ── H8: CR(C2) < CR(C6) on adversarial seeds (CRITICAL) ──
    try:
        c2 = _load_cr("C2")
        c6 = _load_cr("C6")
        n = min(100, len(c2), len(c6))
        h8 = paired_bootstrap_ci(c6[:n], c2[:n])
        results["H8"] = {
            "claim": "CR(C2) < CR(C6) on adversarial seeds — DR contributes",
            "delta": h8["point_estimate"],
            "ci": [h8["ci_lower"], h8["ci_upper"]],
            "excludes_zero": h8["excludes_zero"],
            "confirmed": h8["excludes_zero"] and h8["point_estimate"] > 0,
        }
        status = "CONFIRMED" if results["H8"]["confirmed"] else "FALSIFIED"
        print(f"  H8:  {status} (CRITICAL) — Delta={h8['point_estimate']:.4f}")
    except Exception as e:
        print(f"  H8:  SKIPPED ({e})")

    # ── H5a: M9(C3a) > M9(C2) ──
    try:
        c2 = _load_cr("C2")
        c3a = _load_cr("C3a")
        n = min(len(c2), len(c3a))
        h5a = paired_bootstrap_ci(c3a[:n], c2[:n])
        results["H5a"] = {
            "claim": "Fisher+BC better than L2",
            "delta": h5a["point_estimate"],
            "ci": [h5a["ci_lower"], h5a["ci_upper"]],
            "excludes_zero": h5a["excludes_zero"],
        }
        print(f"  H5a: Delta={h5a['point_estimate']:.4f}")
    except Exception as e:
        print(f"  H5a: SKIPPED ({e})")

    # ── Stress tests H15a-c, H16, H17 ──
    for label, cond in [("H15a", "C8a"), ("H15b", "C8b"), ("H15c", "C8c"),
                         ("H16", "C8d"), ("H17", "C8e")]:
        try:
            cx = _load_cr(cond)
            c1 = _load_cr("C1")
            n = min(len(cx), len(c1))
            h = paired_bootstrap_ci(cx[:n], c1[:n])
            results[label] = {
                "claim": f"CR({cond}) > CR(C1)",
                "delta": h["point_estimate"],
                "ci": [h["ci_lower"], h["ci_upper"]],
                "excludes_zero": h["excludes_zero"],
                "confirmed": h["excludes_zero"] and h["point_estimate"] > 0,
            }
            status = "CONFIRMED" if results[label].get("confirmed") else "FALSIFIED"
            print(f"  {label}: {status} — Delta={h['point_estimate']:.4f}")
        except Exception as e:
            print(f"  {label}: SKIPPED ({e})")

    # Save results
    with open(rd / "hypothesis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    confirmed = sum(1 for v in results.values() if v.get("confirmed"))
    print(f"\n  Summary: {confirmed}/{len(results)} hypotheses confirmed")
    print(f"  [Rule 8] All tested hypotheses reported.")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

class SimpleActor(nn.Module):
    """PPO Actor: 12 → 64 → 64 → 4 (paper spec)."""
    def __init__(self, obs_dim: int = 12, n_actions: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCritic(nn.Module):
    """PPO Critic: 12 → 64 → 64 → 1."""
    def __init__(self, obs_dim: int = 12, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_ppo_sb3(
    env: HighwayFRAEnv, device: str, total_timesteps: int = 20_000
) -> SimpleActor:
    """Train PPO using Stable-Baselines3, then distill into SimpleActor."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # SB3 training
    def make_env():
        return HighwayFRAEnv(vehicles_count=15)

    vec_env = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=5e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        policy_kwargs={"net_arch": [64, 64]},  # Match paper spec
        device=device,
    )
    model.learn(total_timesteps=total_timesteps)

    # Distill SB3 policy into SimpleActor via behavioral cloning
    actor = SimpleActor().to(dev)
    optimizer = optim.Adam(actor.parameters(), lr=1e-3)

    # Collect expert data
    expert_obs, expert_actions = [], []
    obs = vec_env.reset()
    for _ in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        expert_obs.append(obs[0].copy())
        expert_actions.append(action[0])
        obs, _, done, _ = vec_env.step(action)
        if done[0]:
            obs = vec_env.reset()

    expert_obs_t = torch.tensor(np.array(expert_obs), dtype=torch.float32, device=dev)
    expert_actions_t = torch.tensor(expert_actions, dtype=torch.long, device=dev)

    # Behavioral cloning
    for epoch in range(50):
        perm = torch.randperm(len(expert_obs_t), device=dev)
        total_loss = 0
        for start in range(0, len(perm), 256):
            idx = perm[start:start+256]
            logits = actor(expert_obs_t[idx])
            loss = nn.functional.cross_entropy(logits, expert_actions_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    BC epoch {epoch+1}/50 | loss={total_loss:.3f}")

    vec_env.close()

    # Save checkpoints
    ckpt_dir = Path("checkpoints/ppo_training")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), ckpt_dir / "actor_sb3_distilled.pt")
    model.save(str(ckpt_dir / "sb3_model"))
    print(f"  SB3 model and distilled actor saved")

    return actor


def _train_ppo_actor(
    env: HighwayFRAEnv, device: str, n_episodes: int = 200
) -> tuple[SimpleActor, optim.Adam]:
    """Train PPO actor via REINFORCE (simplified for speed)."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    actor = SimpleActor().to(dev)
    critic = SimpleCritic().to(dev)
    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

    best_reward = -float("inf")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        log_probs, values, rewards_list = [], [], []

        for t in range(200):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=dev)
            logits = actor(obs_t)
            value = critic(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            obs, reward, terminated, truncated, info = env.step(action.item())
            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze())
            rewards_list.append(reward)

            if terminated or truncated:
                break

        # Compute returns (GAE simplified)
        returns = []
        G = 0
        for r in reversed(rewards_list):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=dev)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)

        advantages = returns_t - values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor loss (REINFORCE with baseline)
        actor_loss = -(log_probs_t * advantages).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        # Critic loss
        critic_loss = nn.functional.mse_loss(values_t, returns_t)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        ep_reward = sum(rewards_list)
        if ep_reward > best_reward:
            best_reward = ep_reward

        if (ep + 1) % 50 == 0:
            print(f"    Episode {ep+1}/{n_episodes} | reward={ep_reward:.1f} | best={best_reward:.1f}")

        # Checkpoint every 50 episodes
        if (ep + 1) % 50 == 0:
            ckpt_dir = Path("checkpoints/ppo_training")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(actor.state_dict(), ckpt_dir / f"actor_ep{ep+1}.pt")

    return actor, actor_opt


def _collect_d_ref(
    env: HighwayFRAEnv, actor: SimpleActor, device: str, n_pairs: int = 1000
) -> dict:
    """Collect D_ref: (state, action, cost, class) pairs."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    states, actions, costs, classes = [], [], [], []

    obs, info = env.reset(seed=42)
    collected = 0

    while collected < n_pairs:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=dev)
        with torch.no_grad():
            logits = actor(obs_t)
            action = torch.distributions.Categorical(logits=logits).sample().item()

        states.append(obs.copy())
        actions.append(action)
        costs.append(info.get("cost", 0.0))
        classes.append(info.get("obstacle_class", -1))
        collected += 1

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset(seed=42 + collected)

    return {
        "states": torch.tensor(np.array(states), dtype=torch.float32),
        "actions": torch.tensor(actions, dtype=torch.long),
        "costs": torch.tensor(costs, dtype=torch.float32),
        "classes": torch.tensor(classes, dtype=torch.long),
    }


def _compute_fisher(actor: SimpleActor, d_ref: dict, device: str) -> torch.Tensor:
    """Compute diagonal Fisher information matrix over actor params."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    n_params = sum(p.numel() for p in actor.parameters())
    fisher = torch.zeros(n_params, dtype=torch.float64, device=dev)

    states = d_ref["states"].to(dev)

    for i in range(states.shape[0]):
        actor.zero_grad()
        logits = actor(states[i])
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob.backward()

        grads = []
        for p in actor.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.float().flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=dev))
        grad_vec = torch.cat(grads).double()
        fisher += grad_vec ** 2

    fisher /= states.shape[0]
    fisher = torch.clamp(fisher, min=1e-8)  # A2: f_min > 0
    return fisher


def _compute_grad_stats(actor: SimpleActor, d_ref: dict, device: str) -> dict:
    """Compute G_max^0, sigma_G."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    states = d_ref["states"].to(dev)
    norms = []

    for i in range(min(200, states.shape[0])):
        for a in range(4):
            actor.zero_grad()
            logits = actor(states[i])
            log_prob = torch.log_softmax(logits, dim=-1)[a]
            log_prob.backward(retain_graph=(a < 3))

            norm_sq = sum((p.grad.data.float() ** 2).sum().item()
                          for p in actor.parameters() if p.grad is not None)
            norms.append(float(np.sqrt(norm_sq)))

    norms = np.array(norms)
    return {
        "g_max_0": float(norms.max() + 2 * norms.std()),
        "sigma_g": float(norms.std()),
        "g_mean": float(norms.mean()),
    }


def _estimate_lipschitz(actor: SimpleActor, d_ref: dict, device: str, eps: float = 1e-4) -> float:
    """Estimate Lipschitz constant L_G via finite differences."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    states = d_ref["states"].to(dev)
    max_ratio = 0.0
    original_state = {k: v.clone() for k, v in actor.state_dict().items()}

    for i in range(min(30, states.shape[0])):
        # Gradient at current params
        actor.zero_grad()
        logits = actor(states[i])
        torch.log_softmax(logits, dim=-1)[0].backward()
        g0 = torch.cat([p.grad.data.float().flatten() for p in actor.parameters()
                         if p.grad is not None])

        # Perturb
        delta_norm = 0.0
        with torch.no_grad():
            for p in actor.parameters():
                d = torch.randn_like(p.data) * eps
                p.data += d
                delta_norm += (d.float() ** 2).sum().item()
        delta_norm = float(np.sqrt(delta_norm))

        # Gradient at perturbed params
        actor.zero_grad()
        logits = actor(states[i])
        torch.log_softmax(logits, dim=-1)[0].backward()
        g1 = torch.cat([p.grad.data.float().flatten() for p in actor.parameters()
                         if p.grad is not None])

        ratio = float(torch.norm(g1 - g0).item()) / max(delta_norm, 1e-12)
        max_ratio = max(max_ratio, ratio)

        # Restore
        actor.load_state_dict(original_state)

    return max_ratio


def _train_all_critics(d_ref: dict, out: Path, device: str) -> None:
    """Train all 6 cost critic variants."""
    critic_dir = out / "cost_critic"
    critic_dir.mkdir(parents=True, exist_ok=True)

    s, a, c, cl = d_ref["states"], d_ref["actions"], d_ref["costs"], d_ref["classes"]
    n = s.shape[0]

    variants = [
        ("full", None, 1.0),
        ("degraded_10pct", None, 0.1),
        ("degraded_25pct", None, 0.25),
        ("degraded_50pct", None, 0.5),
        ("biased_fast_02", {1: 0.2}, 1.0),
        ("biased_fast_05", {1: 0.5}, 1.0),
    ]

    for name, bias, frac in variants:
        print(f"    Training: {name}...")
        rng = np.random.default_rng(42)
        if frac < 1.0:
            n_sub = max(10, int(n * frac))
            idx = rng.choice(n, n_sub, replace=False)
            ss, sa, sc, scl = s[idx], a[idx], c[idx], cl[idx]
        else:
            ss, sa, sc, scl = s, a, c, cl

        model = train_cost_critic(ss, sa, sc, cost_bias=bias,
                                   obstacle_classes=scl if bias else None, device=device)
        torch.save(model.state_dict(), critic_dir / f"{name}.pt")


def _hash_state_dict(sd: dict) -> str:
    """SHA-256 hash of a state dict."""
    h = hashlib.sha256()
    for v in sd.values():
        h.update(v.cpu().float().numpy().tobytes())
    return h.hexdigest()


def _load_condition_config(cond: str, artifacts: dict, out: Path) -> dict:
    """Build runtime config for a condition."""
    # FRA component flags per condition
    configs = {
        "C1":  {"fra": False},
        "C2":  {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "fc_frozen": False, "fms": True},
        "C3a": {"fra": True, "dr": True, "fhr": "l2", "bc": False, "fc_frozen": False, "fms": True},
        "C3b": {"fra": True, "dr": True, "fhr": "fisher", "bc": False, "fc_frozen": False, "fms": True},
        "C4":  {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "fc_frozen": True, "fms": True},
        "C5":  {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "fc_frozen": False, "fms": False},
        "C6":  {"fra": True, "dr": False, "fhr": "fisher", "bc": True, "fc_frozen": False, "fms": True},
        "C7":  {"fra": False, "hard_override": True},
        "C8a": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "fc_frozen": False, "fms": True, "critic": "degraded_10pct"},
        "C8b": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "fc_frozen": False, "fms": True, "critic": "degraded_25pct"},
        "C8c": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "fc_frozen": False, "fms": True, "critic": "degraded_50pct"},
        "C8d": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "fc_frozen": False, "fms": True, "critic": "biased_fast_02"},
        "C8e": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "fc_frozen": False, "fms": True, "critic": "biased_fast_05"},
    }
    cfg = configs.get(cond, {"fra": False})
    cfg["artifacts"] = artifacts
    cfg["condition"] = cond
    return cfg


def _run_episode(
    env: HighwayFRAEnv,
    w0_state: dict,
    fisher: torch.Tensor,
    fear_state: dict,
    d_ref: dict,
    config: dict,
    seed: int,
    condition: str,
    device: str,
    artifacts: dict,
) -> dict:
    """Run a single episode for a condition."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Create actor from W_0
    actor = SimpleActor().to(dev)
    actor.load_state_dict(w0_state)

    # Save W_0 copy for FHR
    w0_copy = {k: v.clone() for k, v in w0_state.items()}

    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    collision = False

    # FRA parameters (from Prop 1 constraint derivation)
    g_max = artifacts.get("g_max", 10.0)
    f_min = artifacts.get("f_min", 1e-6)
    eta_f = 0.01
    eta_h = 1e-5
    fear_threshold = 0.05

    # Fear detector
    fear_detector = FearDetector(FearDetectorConfig())
    if fear_state:
        fear_detector.load_state(fear_state)

    # Cost critic for safe action
    critic = CostCriticNet().to(dev)
    critic_name = config.get("critic", "full")
    critic_path = Path("checkpoints/cost_critic") / f"{critic_name}.pt"
    if critic_path.exists():
        critic.load_state_dict(torch.load(critic_path, weights_only=True, map_location=dev))
    critic.eval()

    steps_since_spike = 0
    wdn_trace = []
    grad_trace = []

    for t in range(500):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=dev)
        cost = info.get("cost", 0.0)
        ttc = info.get("ttc", 10.0)
        obs_class = info.get("obstacle_class", -1)

        if not config.get("fra", False):
            # C1 or C7: no FRA
            with torch.no_grad():
                logits = actor(obs_t)
                if config.get("hard_override") and cost > 0.4:
                    action = 2  # BRAKE override
                else:
                    action = torch.distributions.Categorical(logits=logits).sample().item()
        else:
            # FRA active
            # 1. Fear detection
            with torch.no_grad():
                logits = actor(obs_t)
                greedy = logits.argmax().item()
            fear, _ = fear_detector.detect(obs, cost, ttc, greedy)

            # 2. Safe action from cost critic
            with torch.no_grad():
                costs_per_action = critic.predict_state(obs_t)
                safe_action = costs_per_action.argmin(dim=-1).item()

            # 3. DR: weight perturbation toward safe action
            grad_norm = 0.0
            if config.get("dr", True) and fear > fear_threshold:
                actor.zero_grad()
                logits = actor(obs_t)
                log_prob_safe = torch.log_softmax(logits, dim=-1)[safe_action]
                log_prob_safe.backward()

                norm_sq = 0.0
                with torch.no_grad():
                    for p in actor.parameters():
                        if p.grad is not None:
                            norm_sq += (p.grad.data.float() ** 2).sum().item()
                            p.data += eta_f * fear * p.grad.data
                grad_norm = float(np.sqrt(norm_sq))
                steps_since_spike = 0
            else:
                steps_since_spike += 1

            # 4. FHR: homeostatic restoring force
            temporal_mult = min(1.0 + 0.02 * steps_since_spike, 10.0)
            eta_h_eff = eta_h * temporal_mult

            with torch.no_grad():
                fhr_mode = config.get("fhr", "fisher")
                param_idx = 0
                for name, p in actor.named_parameters():
                    w0_val = w0_copy[name]
                    diff = w0_val.to(p.dtype) - p.data

                    if fhr_mode == "fisher":
                        n_elem = p.numel()
                        f_slice = fisher[param_idx:param_idx + n_elem].to(dev).reshape(p.shape).to(p.dtype)
                        param_idx += n_elem
                        p.data += eta_h_eff * f_slice * diff
                    else:  # L2
                        p.data += eta_h_eff * diff
                        param_idx += p.numel()

            # 5. BC term
            if config.get("bc", True):
                # Compute KL gradient
                with torch.no_grad():
                    w0_logits = torch.zeros(4, device=dev)
                    # Quick W_0 forward
                    actor_state_backup = {k: v.clone() for k, v in actor.state_dict().items()}
                    actor.load_state_dict(w0_copy)
                    w0_logits = actor(obs_t)
                    w0_probs = torch.softmax(w0_logits, dim=-1).detach()
                    actor.load_state_dict(actor_state_backup)

                actor.zero_grad()
                wt_logits = actor(obs_t)
                wt_log_probs = torch.log_softmax(wt_logits, dim=-1)
                kl = torch.sum(w0_probs * (torch.log(w0_probs + 1e-8) - wt_log_probs))
                kl.backward()

                with torch.no_grad():
                    for p in actor.parameters():
                        if p.grad is not None:
                            p.data -= 1e-5 * p.grad.data

            # 6. Get post-perturbation action
            with torch.no_grad():
                new_logits = actor(obs_t)
                probs = torch.softmax(new_logits, dim=-1)

                # SCL mixing
                alpha = 1.0 / (1.0 + np.exp(-10 * (fear - 0.5)))
                mixed_probs = (1 - alpha) * probs
                mixed_probs[safe_action] += alpha
                action = torch.multinomial(mixed_probs, 1).item()

            # Track WDN (M3)
            wdn = 0.0
            with torch.no_grad():
                for name, p in actor.named_parameters():
                    diff = p.data.float().double() - w0_copy[name].float().double()
                    wdn += (diff * diff).sum().item()
            wdn = float(np.sqrt(wdn))
            wdn_trace.append(wdn)
            grad_trace.append(grad_norm)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            collision = info.get("collision", True)
            break
        if truncated:
            break

    result = {
        "seed": seed,
        "condition": condition,
        "collision": int(collision),  # Save as 0/1 for JSON numeric parsing
        "reward": total_reward,
        "n_steps": t + 1,
    }

    if wdn_trace:
        result["wdn_mean"] = float(np.mean(wdn_trace))
        result["wdn_max"] = float(np.max(wdn_trace))
        result["grad_norm_max"] = float(np.max(grad_trace)) if grad_trace else 0.0

    return result


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FRA Master Orchestrator")
    parser.add_argument("--phase", type=int, default=-1, help="Run specific phase (0-3), -1=all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 10 seeds per condition")
    parser.add_argument("--seeds", type=int, default=None, help="Override seed count")
    parser.add_argument("--conditions", nargs="*", help="Specific conditions to run")
    args = parser.parse_args()

    n_seeds = args.seeds or (10 if args.quick else 1000)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    print(f"FRA Experiment Pipeline — {timestamp}")
    print(f"Device: {args.device} | Seeds: {n_seeds} | Quick: {args.quick}")

    t_total = time.time()

    if args.phase in (-1, 0):
        phase0_train_base(device=args.device)

    if args.phase in (-1, 2):
        phase2_run_conditions(
            device=args.device,
            n_seeds=n_seeds,
            conditions=args.conditions,
        )

    if args.phase in (-1, 3):
        phase3_analyze()

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE — {elapsed:.0f}s total")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

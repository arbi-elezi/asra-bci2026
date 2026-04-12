"""FRA Experiment v2 — Fixed mechanism design.

v1 failed because:
  1. Fisher diagonal was ~0 everywhere (BC-distilled policy, degenerate gradients)
  2. FHR restoring force was effectively disabled (η_h * 0 * diff = 0)
  3. DR pushed weights with no recovery → policy destruction
  4. Cost critic trained on too little data with bad architecture

v2 fixes:
  1. Use SB3 PPO directly (no distillation) — policy has meaningful RL gradients
  2. Extract SB3's value function as cost critic — already trained, already good
  3. Compute Fisher from policy's own action gradients (proper RL Fisher)
  4. Scale Fisher to ensure f_min > 0 meaningfully (add regularization)
  5. Proper eta_f/eta_h ratio from Proposition 1 constraints
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.highway_wrapper import HighwayFRAEnv
from src.agents.cost_critic import CostCriticNet, train_cost_critic
from src.components.fear_detector import FearDetector, FearDetectorConfig
from src.evaluation.metrics import (
    bootstrap_ci, paired_bootstrap_ci,
    m1_collision_rate, m8_task_reward,
)


# ═══════════════════════════════════════════════════════════════════════════
# SB3 NATIVE POLICY WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class SB3PolicyWrapper(nn.Module):
    """Wraps SB3's MlpPolicy as a PyTorch module for gradient access.

    This gives us:
      - W_0: the trained SB3 weights (frozen snapshot)
      - Forward pass: obs → action logits
      - Proper gradients for Fisher computation
      - Direct access to value function for cost estimation
    """

    def __init__(self, sb3_model: PPO, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)

        # Extract the policy network (actor)
        sb3_policy = sb3_model.policy

        # Build equivalent PyTorch module from SB3's architecture
        # SB3 MlpPolicy with net_arch=[64,64] has:
        #   features_extractor → shared layers → policy_net → action_net
        self.features_dim = sb3_policy.features_extractor.features_dim

        # Copy the actual layers from SB3
        self.features_extractor = sb3_policy.features_extractor
        self.mlp_extractor = sb3_policy.mlp_extractor
        self.action_net = sb3_policy.action_net

        # Value function (for cost estimation)
        self.value_net = sb3_policy.value_net

        self.to(self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs → action logits."""
        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor(features)
        return self.action_net(latent_pi)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """obs → state value (used as negative cost proxy)."""
        features = self.features_extractor(obs)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)

    def get_perturbable_params(self) -> list[tuple[str, nn.Parameter]]:
        """All trainable parameters in the actor pathway."""
        params = []
        for name, p in self.mlp_extractor.policy_net.named_parameters():
            params.append((f"policy_net.{name}", p))
        for name, p in self.action_net.named_parameters():
            params.append((f"action_net.{name}", p))
        return params

    def count_params(self) -> int:
        return sum(p.numel() for _, p in self.get_perturbable_params())


def hash_params(params: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for v in params.values():
        h.update(v.cpu().float().numpy().tobytes())
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 0 v2
# ═══════════════════════════════════════════════════════════════════════════

def phase0_v2(device: str = "cuda", ppo_steps: int = 20_000) -> dict:
    """Phase 0 v2: Train SB3 PPO and extract everything directly."""
    out = Path("checkpoints_v2")
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("PHASE 0 v2: SB3 PPO + Native Fisher + Value-as-Cost")
    print("=" * 70)
    t0 = time.time()

    # ── 1. Seeds ──
    rng = np.random.default_rng(2026)
    seeds = rng.integers(0, 2**31, size=1200)
    np.save(out / "seed_set.npy", seeds)

    # ── 2. Train SB3 PPO (CPU — faster for MlpPolicy) ──
    print(f"\n[0.2] Training SB3 PPO ({ppo_steps} steps, CPU)...")
    def make_env():
        return HighwayFRAEnv(vehicles_count=15)

    vec_env = DummyVecEnv([make_env])
    sb3_model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=5e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.8,
        verbose=1,
        policy_kwargs={"net_arch": [64, 64]},
        device="cpu",
    )
    sb3_model.learn(total_timesteps=ppo_steps)
    sb3_model.save(str(out / "sb3_ppo"))
    vec_env.close()

    # ── 3. Extract actor params as W_0 ──
    print("\n[0.3] Extracting W_0 from SB3 actor pathway...")
    sb3_policy = sb3_model.policy

    def get_actor_params_phase0():
        params = []
        for name, p in sb3_policy.mlp_extractor.policy_net.named_parameters():
            params.append((f"policy_net.{name}", p))
        for name, p in sb3_policy.action_net.named_parameters():
            params.append((f"action_net.{name}", p))
        return params

    policy = SB3PolicyWrapper(sb3_model, "cpu")  # For Fisher computation
    n_params = sum(p.numel() for _, p in get_actor_params_phase0())
    print(f"  Perturbable params: {n_params}")

    w0 = {name: p.data.clone().detach() for name, p in get_actor_params_phase0()}
    w0_hash = hash_params(w0)
    torch.save({"w0": w0, "hash": w0_hash}, out / "w0.pt")
    print(f"  W_0 hash: {w0_hash[:16]}...")

    # ── 4. Collect D_ref using SB3 model ──
    print("\n[0.4] Collecting D_ref...")
    env = HighwayFRAEnv(seed=0, vehicles_count=15)
    d_ref = _collect_d_ref_sb3(env, sb3_model, device, n_pairs=2000)
    torch.save(d_ref, out / "d_ref.pt")
    print(f"  D_ref: {d_ref['states'].shape[0]} pairs, mean cost: {d_ref['costs'].mean():.3f}")

    # ── 5. Compute Fisher FROM RL POLICY GRADIENTS ──
    print("\n[0.5] Computing Fisher information (from RL policy gradients)...")
    fisher = _compute_fisher_v2(policy, d_ref["states"].to(device))
    # Add regularization to ensure f_min > 0 MEANINGFULLY
    fisher_reg = fisher + 1e-4  # Regularize — this ensures FHR actually works
    torch.save(fisher_reg, out / "fisher_diagonal.pt")
    f_min = fisher_reg.min().item()
    f_max = fisher_reg.max().item()
    print(f"  Fisher (regularized): f_min={f_min:.6f}, f_max={f_max:.6f}")
    print(f"  Fisher range: {f_max/f_min:.1f}x")

    # ── 6. Gradient stats ──
    print("\n[0.6] Computing gradient statistics...")
    g_stats = _compute_grad_stats_v2(policy, d_ref["states"].to(device))
    l_g = _estimate_lipschitz_v2(policy, d_ref["states"].to(device))
    g_max = g_stats["g_max_0"] * 1.5
    r = (g_max - g_stats["g_max_0"]) / max(l_g, 1e-8)
    d_max = min(r, 1.0)

    # Derive proper eta_f/eta_h from Proposition 1
    # η_f/η_h ≤ D_max · f_min / G_max
    eta_ratio_max = d_max * f_min / max(g_max, 1e-8)
    # Choose eta_h so η_h · f_max < 1 (A3)
    eta_h = min(0.01, 0.5 / max(f_max, 1e-8))
    # Choose eta_f from ratio constraint
    eta_f = eta_h * eta_ratio_max * 0.5  # 50% of max for safety margin

    print(f"  G_max^0={g_stats['g_max_0']:.4f}, L_G={l_g:.4f}, r={r:.4f}")
    print(f"  Derived: eta_h={eta_h:.6f}, eta_f={eta_f:.8f}")
    print(f"  eta_f/eta_h = {eta_f/eta_h:.6e} (max allowed: {eta_ratio_max:.6e})")

    # ── 7. Train cost critics using SB3's value function as ground truth ──
    print("\n[0.7] Training cost critics (using SB3 value function as reference)...")
    # The SB3 value function IS the cost signal — negative value = high cost
    with torch.no_grad():
        states_dev = d_ref["states"].to(device)
        sb3_values = policy.get_value(states_dev).squeeze(-1)
        # Convert value to cost: high value = safe, low value = dangerous
        # Normalize to [0, 1]: cost = 1 - (V - V_min) / (V_max - V_min)
        v_min, v_max = sb3_values.min(), sb3_values.max()
        if v_max > v_min:
            value_costs = 1.0 - (sb3_values - v_min) / (v_max - v_min)
        else:
            value_costs = torch.zeros_like(sb3_values)

    # Use these as cost labels for the critic
    d_ref_with_vcost = {
        "states": d_ref["states"],
        "actions": d_ref["actions"],
        "costs": value_costs.cpu(),  # SB3-derived costs
        "classes": d_ref["classes"],
        "raw_costs": d_ref["costs"],  # Original TTC-based costs
    }

    _train_all_critics_v2(d_ref_with_vcost, out, device)

    # ── 8. Fear detector ──
    print("\n[0.8] Training fear detector on D_ref...")
    fear_det = FearDetector(FearDetectorConfig())
    fear_det.train_on_d_ref(d_ref["states"].numpy(), d_ref["costs"].numpy())
    torch.save(fear_det.get_state(), out / "fear_detector.pt")

    # ── Save artifacts ──
    artifacts = {
        "n_seeds": 1200,
        "w0_hash": w0_hash,
        "n_params": n_params,
        "d_ref_size": int(d_ref["states"].shape[0]),
        "f_min": f_min,
        "f_max": f_max,
        "g_max_0": g_stats["g_max_0"],
        "sigma_g": g_stats["sigma_g"],
        "g_max": g_max,
        "l_g": l_g,
        "r": r,
        "d_max": d_max,
        "eta_f": eta_f,
        "eta_h": eta_h,
        "eta_ratio_max": eta_ratio_max,
        "ppo_steps": ppo_steps,
        "version": "v2",
        "elapsed_seconds": time.time() - t0,
    }
    with open(out / "artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=2, default=str)

    print(f"\n  Phase 0 v2 complete in {artifacts['elapsed_seconds']:.0f}s")
    return artifacts


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 v2
# ═══════════════════════════════════════════════════════════════════════════

def phase2_v2(device: str = "cuda", n_seeds: int = 200, conditions: list[str] | None = None) -> dict:
    """Run all conditions using SB3 native policy."""
    out = Path("checkpoints_v2")
    results_dir = Path("results_v2")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"PHASE 2 v2: Running Conditions ({n_seeds} seeds)")
    print("=" * 70)

    with open(out / "artifacts.json") as f:
        artifacts = json.load(f)

    # Load SB3 model — use CPU for MlpPolicy (SB3 recommendation)
    sb3_model = PPO.load(str(out / "sb3_ppo"), device="cpu")
    sb3_policy = sb3_model.policy  # Direct access to SB3 policy
    dev = torch.device("cpu")  # SB3 MlpPolicy runs on CPU

    # Load W_0
    w0_data = torch.load(out / "w0.pt", weights_only=False, map_location="cpu")
    w0 = w0_data["w0"]
    w0_hash = w0_data["hash"]

    # Load Fisher
    fisher = torch.load(out / "fisher_diagonal.pt", weights_only=False).to("cpu")

    # Load fear detector
    fear_det = FearDetector(FearDetectorConfig())
    fear_det.load_state(torch.load(out / "fear_detector.pt", weights_only=False))

    # Load cost critics
    critic_full = CostCriticNet().to(dev)
    critic_full.load_state_dict(torch.load(out / "cost_critic" / "full.pt", weights_only=True, map_location=dev))
    critic_full.eval()

    seeds = np.load(out / "seed_set.npy")
    env = HighwayFRAEnv(seed=0, vehicles_count=15)

    if conditions is None:
        conditions = ["C1", "C2", "C3a", "C3b", "C4", "C5", "C6", "C7",
                       "C8a", "C8b", "C8c", "C8d", "C8e"]

    eta_f = artifacts["eta_f"]
    eta_h = artifacts["eta_h"]

    # Helper: get perturbable actor params from SB3 policy
    def get_actor_params():
        params = []
        for name, p in sb3_policy.mlp_extractor.policy_net.named_parameters():
            params.append((f"policy_net.{name}", p))
        for name, p in sb3_policy.action_net.named_parameters():
            params.append((f"action_net.{name}", p))
        return params

    # Helper: get action using SB3's native forward pass (handles obs preprocessing)
    def sb3_action(obs_np, deterministic=False):
        action, _ = sb3_model.predict(obs_np, deterministic=deterministic)
        return int(action)

    # Helper: get logits from SB3's actor
    def sb3_logits(obs_np):
        obs_tensor, _ = sb3_policy.obs_to_tensor(obs_np)
        features = sb3_policy.extract_features(obs_tensor, sb3_policy.features_extractor)
        latent_pi, _ = sb3_policy.mlp_extractor(features)
        return sb3_policy.action_net(latent_pi).squeeze(0)

    all_summaries = {}

    for cond in conditions:
        print(f"\n--- {cond} ---")
        cond_dir = results_dir / cond
        cond_dir.mkdir(parents=True, exist_ok=True)

        cond_seeds = min(n_seeds, 500) if cond.startswith("C8") else n_seeds
        cfg = _get_cond_config(cond)

        # Load appropriate cost critic
        critic_name = cfg.get("critic", "full")
        critic = CostCriticNet().to(dev)
        cpath = out / "cost_critic" / f"{critic_name}.pt"
        if cpath.exists():
            critic.load_state_dict(torch.load(cpath, weights_only=True, map_location=dev))
        critic.eval()

        collisions, rewards, per_seed = [], [], []
        t_start = time.time()

        for idx in range(cond_seeds):
            seed = int(seeds[idx])

            # Restore W_0 at episode start
            with torch.no_grad():
                for name, p in get_actor_params():
                    if name in w0:
                        p.data.copy_(w0[name])

            obs, info = env.reset(seed=seed)
            ep_reward = 0.0
            collision = False

            for t in range(500):
                cost = info.get("cost", 0.0)
                ttc = info.get("ttc", 10.0)

                if not cfg.get("fra", False):
                    # Baseline: use SB3's native predict
                    if cfg.get("hard_override") and cost > 0.4:
                        action = 2
                    else:
                        action = sb3_action(obs)
                else:
                    # FRA active
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=dev)

                    with torch.no_grad():
                        logits = sb3_logits(obs)
                        greedy = logits.argmax().item()

                    # Fear
                    fear, _ = fear_det.detect(obs, cost, ttc, greedy)

                    # Safe action
                    with torch.no_grad():
                        costs_per_a = critic.predict_state(obs_t)
                        safe_action = costs_per_a.argmin(dim=-1).item()

                    # DR
                    if cfg.get("dr", True) and fear > 0.05:
                        sb3_policy.zero_grad()
                        logits = sb3_logits(obs)
                        log_prob_safe = torch.log_softmax(logits, dim=-1)[safe_action]
                        log_prob_safe.backward()

                        with torch.no_grad():
                            for name, p in get_actor_params():
                                if p.grad is not None:
                                    p.data += eta_f * fear * p.grad.data

                    # FHR
                    if cfg.get("fhr", True):
                        fhr_mode = cfg.get("fhr_mode", "fisher")
                        with torch.no_grad():
                            param_idx = 0
                            for name, p in get_actor_params():
                                if name not in w0:
                                    param_idx += p.numel()
                                    continue
                                diff = w0[name].to(p.dtype) - p.data
                                if fhr_mode == "fisher":
                                    n_elem = p.numel()
                                    f_slice = fisher[param_idx:param_idx+n_elem].reshape(p.shape).to(p.dtype)
                                    param_idx += n_elem
                                    p.data += eta_h * f_slice * diff
                                else:
                                    param_idx += p.numel()
                                    p.data += eta_h * diff

                    # BC
                    if cfg.get("bc", True):
                        with torch.no_grad():
                            wt_snap = {n: p.data.clone() for n, p in get_actor_params()}
                            for n, p in get_actor_params():
                                if n in w0: p.data.copy_(w0[n])
                            w0_probs = torch.softmax(sb3_logits(obs), dim=-1)
                            for n, p in get_actor_params():
                                if n in wt_snap: p.data.copy_(wt_snap[n])

                        sb3_policy.zero_grad()
                        wt_logits = sb3_logits(obs)
                        wt_lp = torch.log_softmax(wt_logits, dim=-1)
                        kl = torch.sum(w0_probs.detach() * (torch.log(w0_probs.detach() + 1e-8) - wt_lp))
                        kl.backward()
                        with torch.no_grad():
                            for _, p in get_actor_params():
                                if p.grad is not None:
                                    p.data -= 1e-5 * p.grad.data

                    # SCL mixing
                    with torch.no_grad():
                        new_logits = sb3_logits(obs)
                        probs = torch.softmax(new_logits, dim=-1)
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

            collisions.append(int(collision))
            rewards.append(ep_reward)
            per_seed.append({"seed": seed, "condition": cond, "collision": int(collision), "reward": ep_reward})

            if (idx + 1) % max(1, cond_seeds // 10) == 0:
                cr = np.mean(collisions)
                rate = (idx + 1) / (time.time() - t_start)
                print(f"  [{idx+1}/{cond_seeds}] CR={cr:.3f} | {rate:.1f} seeds/s")

        elapsed = time.time() - t_start
        cr_arr = np.array(collisions, dtype=float)
        cr = m1_collision_rate(cr_arr)
        cr_ci = bootstrap_ci(cr_arr)
        rw = m8_task_reward(np.array(rewards))

        summary = {"condition": cond, "n_seeds": cond_seeds,
                    "M1_collision_rate": cr, "M1_ci": cr_ci,
                    "M8_mean_reward": rw, "w0_hash": w0_hash, "elapsed_s": elapsed}

        with open(cond_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        with open(cond_dir / "per_seed.json", "w") as f:
            json.dump(per_seed, f, indent=2, default=str)

        all_summaries[cond] = summary
        print(f"  {cond}: CR={cr:.3f} [{cr_ci['ci_lower']:.3f}, {cr_ci['ci_upper']:.3f}] | Rwd={rw:.1f}")

    return all_summaries


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 v2
# ═══════════════════════════════════════════════════════════════════════════

def phase3_v2(results_dir: str = "results_v2") -> dict:
    """Hypothesis testing on v2 results."""
    rd = Path(results_dir)
    print("\n" + "=" * 70)
    print("PHASE 3 v2: Hypothesis Testing")
    print("=" * 70)

    results = {}

    def _load_cr(cond):
        with open(rd / cond / "per_seed.json") as f:
            data = json.load(f)
        def to_f(v):
            if isinstance(v, bool): return 1.0 if v else 0.0
            if isinstance(v, str): return 1.0 if v.lower() == "true" else 0.0
            return float(v)
        return np.array([to_f(d["collision"]) for d in data], dtype=float)

    tests = [
        ("H1", "C1", "C2", "CR(C2) < CR(C1)"),
        ("H8", "C6", "C2", "DR contributes (C2 < C6)"),
        ("H15a", "C8a", "C1", "10% D_ref harmful"),
        ("H15b", "C8b", "C1", "25% D_ref harmful"),
        ("H15c", "C8c", "C1", "50% D_ref harmful"),
        ("H16", "C8d", "C1", "Severe bias harmful"),
        ("H17", "C8e", "C1", "Moderate bias harmful"),
    ]

    for label, cx, cy, claim in tests:
        try:
            x, y = _load_cr(cx), _load_cr(cy)
            n = min(len(x), len(y))
            h = paired_bootstrap_ci(x[:n], y[:n])
            confirmed = h["excludes_zero"] and h["point_estimate"] > 0
            results[label] = {"claim": claim, "delta": h["point_estimate"],
                              "ci": [h["ci_lower"], h["ci_upper"]],
                              "excludes_zero": h["excludes_zero"], "confirmed": confirmed}
            s = "CONFIRMED" if confirmed else "FALSIFIED"
            print(f"  {label:>5s}: {s:>10s}  Delta={h['point_estimate']:>+.4f}  "
                  f"CI=[{h['ci_lower']:.4f}, {h['ci_upper']:.4f}]")
        except Exception as e:
            print(f"  {label}: SKIPPED ({e})")

    with open(rd / "hypothesis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    confirmed = sum(1 for v in results.values() if v.get("confirmed"))
    print(f"\n  TOTAL: {confirmed}/{len(results)} confirmed")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _collect_d_ref_sb3(env, sb3_model, device, n_pairs=2000):
    states, actions, costs, classes = [], [], [], []
    obs, info = env.reset(seed=42)
    for i in range(n_pairs):
        action, _ = sb3_model.predict(obs, deterministic=False)
        states.append(obs.copy())
        actions.append(int(action))
        costs.append(info.get("cost", 0.0))
        classes.append(info.get("obstacle_class", -1))
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            obs, info = env.reset(seed=42 + i)
    return {
        "states": torch.tensor(np.array(states), dtype=torch.float32),
        "actions": torch.tensor(actions, dtype=torch.long),
        "costs": torch.tensor(costs, dtype=torch.float32),
        "classes": torch.tensor(classes, dtype=torch.long),
    }


def _compute_fisher_v2(policy: SB3PolicyWrapper, states: torch.Tensor) -> torch.Tensor:
    """Compute Fisher from ACTUAL RL policy gradients."""
    n_params = sum(p.numel() for _, p in policy.get_perturbable_params())
    fisher = torch.zeros(n_params, dtype=torch.float64, device=states.device)

    for i in range(min(500, states.shape[0])):
        policy.zero_grad()
        logits = policy(states[i:i+1]).squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob.backward()

        grads = []
        for _, p in policy.get_perturbable_params():
            if p.grad is not None:
                grads.append(p.grad.data.float().flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=states.device))
        fisher += torch.cat(grads).double() ** 2

    fisher /= min(500, states.shape[0])
    return fisher


def _compute_grad_stats_v2(policy, states):
    norms = []
    for i in range(min(100, states.shape[0])):
        for a in range(4):
            policy.zero_grad()
            logits = policy(states[i:i+1]).squeeze(0)
            torch.log_softmax(logits, dim=-1)[a].backward(retain_graph=(a < 3))
            ns = sum((p.grad.data.float()**2).sum().item() for _, p in policy.get_perturbable_params() if p.grad is not None)
            norms.append(float(np.sqrt(ns)))
    norms = np.array(norms)
    return {"g_max_0": float(norms.max() + 2*norms.std()), "sigma_g": float(norms.std())}


def _estimate_lipschitz_v2(policy, states, eps=1e-4):
    original = {n: p.data.clone() for n, p in policy.get_perturbable_params()}
    max_ratio = 0.0
    for i in range(min(20, states.shape[0])):
        policy.zero_grad()
        logits = policy(states[i:i+1]).squeeze(0)
        torch.log_softmax(logits, dim=-1)[0].backward()
        g0 = torch.cat([p.grad.data.float().flatten() for _, p in policy.get_perturbable_params() if p.grad is not None])

        dn = 0.0
        with torch.no_grad():
            for _, p in policy.get_perturbable_params():
                d = torch.randn_like(p.data) * eps
                p.data += d
                dn += (d.float()**2).sum().item()
        dn = float(np.sqrt(dn))

        policy.zero_grad()
        logits = policy(states[i:i+1]).squeeze(0)
        torch.log_softmax(logits, dim=-1)[0].backward()
        g1 = torch.cat([p.grad.data.float().flatten() for _, p in policy.get_perturbable_params() if p.grad is not None])

        ratio = float(torch.norm(g1 - g0).item()) / max(dn, 1e-12)
        max_ratio = max(max_ratio, ratio)

        with torch.no_grad():
            for n, p in policy.get_perturbable_params():
                p.data.copy_(original[n])
    return max_ratio


def _train_all_critics_v2(d_ref, out, device):
    cdir = out / "cost_critic"
    cdir.mkdir(parents=True, exist_ok=True)
    s, a, c, cl = d_ref["states"], d_ref["actions"], d_ref["costs"], d_ref["classes"]
    n = s.shape[0]

    for name, bias, frac in [
        ("full", None, 1.0), ("degraded_10pct", None, 0.1),
        ("degraded_25pct", None, 0.25), ("degraded_50pct", None, 0.5),
        ("biased_fast_02", {1: 0.2}, 1.0), ("biased_fast_05", {1: 0.5}, 1.0),
    ]:
        print(f"    {name}...")
        rng = np.random.default_rng(42)
        if frac < 1.0:
            idx = rng.choice(n, max(10, int(n*frac)), replace=False)
            ss, sa, sc, scl = s[idx], a[idx], c[idx], cl[idx]
        else:
            ss, sa, sc, scl = s, a, c, cl
        model = train_cost_critic(ss, sa, sc, cost_bias=bias,
                                   obstacle_classes=scl if bias else None, device=device)
        torch.save(model.state_dict(), cdir / f"{name}.pt")


def _get_cond_config(cond):
    configs = {
        "C1":  {"fra": False},
        "C2":  {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True},
        "C3a": {"fra": True, "dr": True, "fhr": True, "fhr_mode": "l2", "bc": False},
        "C3b": {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": False},
        "C4":  {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True},
        "C5":  {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True},
        "C6":  {"fra": True, "dr": False, "fhr": True, "fhr_mode": "fisher", "bc": True},
        "C7":  {"fra": False, "hard_override": True},
        "C8a": {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True, "critic": "degraded_10pct"},
        "C8b": {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True, "critic": "degraded_25pct"},
        "C8c": {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True, "critic": "degraded_50pct"},
        "C8d": {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True, "critic": "biased_fast_02"},
        "C8e": {"fra": True, "dr": True, "fhr": True, "fhr_mode": "fisher", "bc": True, "critic": "biased_fast_05"},
    }
    return configs.get(cond, {"fra": False})


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FRA v2 — Fixed mechanism")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ppo-steps", type=int, default=20_000)
    args = parser.parse_args()

    t0 = time.time()
    print(f"FRA Experiment v2 -- {datetime.now()}")

    phase0_v2(device=args.device, ppo_steps=args.ppo_steps)
    phase2_v2(device=args.device, n_seeds=args.seeds)
    phase3_v2()

    print(f"\nPIPELINE v2 COMPLETE -- {time.time()-t0:.0f}s total")

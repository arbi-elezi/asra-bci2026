"""Day 1: Train base policies for all LLM models.

For each model:
  1. Load LLM + LoRA + action head
  2. Train via PPO (REINFORCE with baseline, proper GAE, 100K steps)
  3. Collect D_ref (10K pairs)
  4. Compute Fisher information (2000 samples, with regularization)
  5. Compute gradient statistics (G_max^0, L_G, r)
  6. Train cost critics (6 variants: full + 3 degraded + 2 biased)
  7. Validate each component individually
  8. Checkpoint everything

This is Day 1 of a 4-day experiment.
Expected runtime: ~8-12 hours for 4 models.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.environment.highway_wrapper import HighwayFRAEnv
from src.agents.cost_critic import CostCriticNet, CostCriticConfig, train_cost_critic
from src.components.fear_detector import FearDetector, FearDetectorConfig
from .config import MODELS, ModelConfig, TrainingConfig, ExperimentConfig
from .llm_policy import LLMPolicy


def train_single_model(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    exp_cfg: ExperimentConfig,
) -> dict:
    """Full Day 1 pipeline for one model."""
    model_dir = exp_cfg.output_base / model_cfg.name
    model_dir.mkdir(parents=True, exist_ok=True)

    log = {"model": model_cfg.name, "started": datetime.now().isoformat()}
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_cfg.name} ({model_cfg.hf_id})")
    print(f"{'='*70}")
    t0 = time.time()

    # ── 1. Initialize LLM policy ──
    print(f"\n[1] Loading {model_cfg.name}...")
    policy = LLMPolicy(model_cfg, device=exp_cfg.device)

    # ── 2. Initialize environment ──
    env = HighwayFRAEnv(seed=0, vehicles_count=exp_cfg.vehicles_count)
    print(f"  Env: R^{env.observation_space.shape[0]}, |A|={env.action_space.n}")

    # ── 3. Train via PPO ──
    print(f"\n[2] Training PPO ({train_cfg.ppo_total_steps} steps)...")
    _train_ppo(policy, env, train_cfg, exp_cfg, model_dir)

    # Snapshot W_0
    policy.snapshot_w0()
    policy.save(str(model_dir / "policy_trained.pt"))

    # ── 4. Evaluate base policy ──
    print(f"\n[3] Evaluating base policy (100 seeds)...")
    base_cr, base_reward = _evaluate_policy(policy, env, n_seeds=100)
    print(f"  Base CR: {base_cr:.3f}, Reward: {base_reward:.1f}")
    log["base_cr"] = base_cr
    log["base_reward"] = base_reward

    if base_cr < 0.05:
        print(f"  WARNING: Base CR too low ({base_cr:.3f}). Policy may be too good for FRA to help.")
    if base_cr > 0.95:
        print(f"  WARNING: Base CR too high ({base_cr:.3f}). Policy may be too bad for FRA to matter.")

    # ── 5. Collect D_ref ──
    print(f"\n[4] Collecting D_ref ({train_cfg.d_ref_size} pairs)...")
    d_ref = _collect_d_ref(policy, env, train_cfg.d_ref_size)
    torch.save(d_ref, model_dir / "d_ref.pt")
    print(f"  D_ref: {d_ref['states'].shape[0]} pairs, mean cost: {d_ref['costs'].mean():.3f}")
    log["d_ref_mean_cost"] = float(d_ref["costs"].mean())

    # ── 6. Compute Fisher ──
    print(f"\n[5] Computing Fisher ({train_cfg.fisher_samples} samples)...")
    fisher = _compute_fisher(policy, d_ref, train_cfg)
    torch.save(fisher, model_dir / "fisher.pt")
    f_min, f_max = fisher.min().item(), fisher.max().item()
    print(f"  Fisher: f_min={f_min:.6f}, f_max={f_max:.6f}, range={f_max/max(f_min,1e-12):.0f}x")
    log["f_min"] = f_min
    log["f_max"] = f_max

    # ── 7. Gradient statistics ──
    print(f"\n[6] Computing gradient statistics...")
    g_stats = _compute_grad_stats(policy, d_ref)
    l_g = _estimate_lipschitz(policy, d_ref)
    g_max = g_stats["g_max_0"] * 1.5
    r = (g_max - g_stats["g_max_0"]) / max(l_g, 1e-8)
    print(f"  G_max^0={g_stats['g_max_0']:.4f}, sigma_G={g_stats['sigma_g']:.4f}")
    print(f"  L_G={l_g:.4f}, r={r:.4f}")
    log.update({"g_max_0": g_stats["g_max_0"], "sigma_g": g_stats["sigma_g"],
                "l_g": l_g, "r": r, "g_max": g_max})

    # ── 8. Train cost critics ──
    print(f"\n[7] Training cost critics (6 variants, {train_cfg.cost_critic_epochs} epochs each)...")
    _train_cost_critics(d_ref, model_dir, train_cfg, exp_cfg.device)

    # ── 9. Train fear detector ──
    print(f"\n[8] Training fear detector...")
    fear_det = FearDetector(FearDetectorConfig(device=exp_cfg.device))
    fear_det.train_on_d_ref(d_ref["states"].numpy(), d_ref["costs"].numpy())
    torch.save(fear_det.get_state(), model_dir / "fear_detector.pt")

    # ── 10. Component validation ──
    print(f"\n[9] Validating components...")
    validation = _validate_components(policy, d_ref, fisher, model_dir, exp_cfg.device)
    log["validation"] = validation

    # ── 11. Generate seeds ──
    rng = np.random.default_rng(2026)
    seeds = rng.integers(0, 2**31, size=exp_cfg.n_experiment_seeds + exp_cfg.n_validation_seeds)
    np.save(model_dir / "seeds.npy", seeds)

    # ── Save log ──
    log["elapsed_seconds"] = time.time() - t0
    log["n_perturbable_params"] = policy.n_perturbable
    log["w0_hash"] = policy.get_w0_hash()
    with open(model_dir / "day1_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"\n  {model_cfg.name} Day 1 complete in {log['elapsed_seconds']:.0f}s")
    print(f"  All artifacts saved to {model_dir}/")
    return log


def _train_ppo(
    policy: LLMPolicy,
    env: HighwayFRAEnv,
    cfg: TrainingConfig,
    exp_cfg: ExperimentConfig,
    save_dir: Path,
) -> None:
    """Train LLM policy via PPO with GAE."""
    # Optimizer: separate LR for LoRA vs action head
    lora_params = [p for n, p in policy.get_perturbable_params() if "lora" in n]
    head_params = [p for n, p in policy.get_perturbable_params() if "head" in n]

    optimizer = optim.Adam([
        {"params": head_params, "lr": cfg.ppo_lr},
        {"params": lora_params, "lr": cfg.ppo_lr * 0.1},  # Lower LR for LoRA
    ])

    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    total_steps = 0
    episode = 0
    best_reward = -float("inf")

    # Resume from latest checkpoint if available
    existing_ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if existing_ckpts:
        latest = existing_ckpts[-1]
        resume_step = int(latest.stem.split("_")[1])
        print(f"    Resuming from checkpoint: {latest.name} (step {resume_step})")
        ckpt_data = torch.load(latest, weights_only=False, map_location=policy.device)
        for name, p in policy.get_perturbable_params():
            if name in ckpt_data:
                p.data.copy_(ckpt_data[name].to(p.device))
        total_steps = resume_step
        episode = resume_step // 50  # Approximate
        print(f"    Resumed. Continuing from step {total_steps}...")

    while total_steps < cfg.ppo_total_steps:
        # Collect rollout
        obs_batch, act_batch, rew_batch, val_batch, logp_batch, done_batch = [], [], [], [], [], []

        obs, info = env.reset(seed=episode)
        ep_reward = 0.0

        for step in range(cfg.ppo_n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=policy.device)

            with torch.no_grad():
                logits = policy.get_logits_from_obs(obs)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                # Use negative cost as value estimate for now
                value = torch.tensor(-info.get("cost", 0.0), device=policy.device)

            obs_batch.append(obs.copy())
            act_batch.append(action.item())
            logp_batch.append(log_prob.item())
            val_batch.append(value.item())

            obs, reward, terminated, truncated, info = env.step(action.item())
            rew_batch.append(reward)
            done_batch.append(terminated or truncated)
            ep_reward += reward
            total_steps += 1

            if terminated or truncated:
                episode += 1
                if ep_reward > best_reward:
                    best_reward = ep_reward
                obs, info = env.reset(seed=episode)
                ep_reward = 0.0

        # Compute GAE
        returns, advantages = _compute_gae(
            rew_batch, val_batch, done_batch, cfg.ppo_gamma, cfg.ppo_gae_lambda
        )

        # PPO update
        obs_t = torch.tensor(np.array(obs_batch), dtype=torch.float32, device=policy.device)
        act_t = torch.tensor(act_batch, dtype=torch.long, device=policy.device)
        old_logp_t = torch.tensor(logp_batch, dtype=torch.float32, device=policy.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=policy.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=policy.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(cfg.ppo_n_epochs):
            # Shuffle
            perm = torch.randperm(len(obs_batch), device=policy.device)
            for start in range(0, len(perm), cfg.ppo_batch_size):
                idx = perm[start:start + cfg.ppo_batch_size]

                # Forward pass for batch
                batch_logits = []
                for i in idx:
                    logits = policy.get_logits_from_obs(obs_batch[i.item()])
                    batch_logits.append(logits)
                batch_logits = torch.stack(batch_logits)

                dist = torch.distributions.Categorical(logits=batch_logits)
                new_logp = dist.log_prob(act_t[idx])
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(new_logp - old_logp_t[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - cfg.ppo_clip_range, 1 + cfg.ppo_clip_range) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                loss = policy_loss - cfg.ppo_ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for _, p in policy.get_perturbable_params()],
                    cfg.ppo_max_grad_norm,
                )
                optimizer.step()

        # Logging
        if total_steps % 5000 < cfg.ppo_n_steps:
            print(f"    Step {total_steps}/{cfg.ppo_total_steps} | "
                  f"ep={episode} | best_rwd={best_reward:.1f} | "
                  f"loss={loss.item():.4f}")

        # Checkpoint
        if total_steps % cfg.checkpoint_every < cfg.ppo_n_steps:
            torch.save(
                {n: p.data.clone().cpu() for n, p in policy.get_perturbable_params()},
                ckpt_dir / f"step_{total_steps}.pt",
            )


def _compute_gae(rewards, values, dones, gamma, lam):
    """Generalized Advantage Estimation."""
    n = len(rewards)
    advantages = np.zeros(n)
    last_gae = 0

    for t in reversed(range(n)):
        next_val = values[t + 1] if t + 1 < n else 0
        next_non_terminal = 0 if dones[t] else 1
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae

    returns = advantages + np.array(values)
    return returns.tolist(), advantages.tolist()


def _evaluate_policy(policy: LLMPolicy, env: HighwayFRAEnv, n_seeds: int = 100) -> tuple[float, float]:
    """Evaluate policy CR and mean reward."""
    collisions, rewards = 0, []
    for seed in range(n_seeds):
        obs, info = env.reset(seed=seed)
        ep_reward = 0.0
        for t in range(500):
            action = policy.get_action(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated:
                if info.get("collision"):
                    collisions += 1
                break
            if truncated:
                break
        rewards.append(ep_reward)
    return collisions / n_seeds, float(np.mean(rewards))


def _collect_d_ref(policy: LLMPolicy, env: HighwayFRAEnv, n_pairs: int) -> dict:
    """Collect reference dataset."""
    states, actions, costs, classes, texts = [], [], [], [], []
    obs, info = env.reset(seed=42)

    for i in range(n_pairs):
        action = policy.get_action(obs, deterministic=False)
        states.append(obs.copy())
        actions.append(action)
        costs.append(info.get("cost", 0.0))
        classes.append(info.get("obstacle_class", -1))
        texts.append(policy._obs_to_text(obs))

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset(seed=42 + i)

        if (i + 1) % 2000 == 0:
            print(f"    D_ref: {i+1}/{n_pairs}")

    return {
        "states": torch.tensor(np.array(states), dtype=torch.float32),
        "actions": torch.tensor(actions, dtype=torch.long),
        "costs": torch.tensor(costs, dtype=torch.float32),
        "classes": torch.tensor(classes, dtype=torch.long),
        "texts": texts,
    }


def _compute_fisher(policy: LLMPolicy, d_ref: dict, cfg: TrainingConfig) -> torch.Tensor:
    """Compute diagonal Fisher with regularization."""
    n_params = policy.n_perturbable
    fisher = torch.zeros(n_params, dtype=torch.float64, device=policy.device)
    n_samples = min(cfg.fisher_samples, len(d_ref["texts"]))

    for i in range(n_samples):
        policy.model.zero_grad()
        policy.action_head.zero_grad()

        logits = policy.get_logits(d_ref["texts"][i])
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob.backward()

        grads = []
        for _, p in policy.get_perturbable_params():
            if p.grad is not None:
                grads.append(p.grad.data.float().flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=policy.device))
        fisher += torch.cat(grads).double() ** 2

        if (i + 1) % 500 == 0:
            print(f"    Fisher: {i+1}/{n_samples}")

    fisher /= n_samples
    fisher += cfg.fisher_regularization  # Ensure f_min > 0 meaningfully
    return fisher


def _compute_grad_stats(policy: LLMPolicy, d_ref: dict) -> dict:
    """Compute G_max^0, sigma_G."""
    norms = []
    n = min(200, len(d_ref["texts"]))
    for i in range(n):
        for a in range(4):
            policy.model.zero_grad()
            policy.action_head.zero_grad()
            logits = policy.get_logits(d_ref["texts"][i])
            torch.log_softmax(logits, dim=-1)[a].backward(retain_graph=(a < 3))
            ns = sum((p.grad.data.float()**2).sum().item()
                     for _, p in policy.get_perturbable_params() if p.grad is not None)
            norms.append(float(np.sqrt(ns)))
    norms = np.array(norms)
    return {"g_max_0": float(norms.max() + 2*norms.std()), "sigma_g": float(norms.std())}


def _estimate_lipschitz(policy: LLMPolicy, d_ref: dict, eps: float = 1e-4) -> float:
    """Estimate Lipschitz constant via finite differences."""
    max_ratio = 0.0
    original = {n: p.data.clone() for n, p in policy.get_perturbable_params()}

    for i in range(min(30, len(d_ref["texts"]))):
        policy.model.zero_grad()
        policy.action_head.zero_grad()
        logits = policy.get_logits(d_ref["texts"][i])
        torch.log_softmax(logits, dim=-1)[0].backward()
        g0 = torch.cat([p.grad.data.float().flatten() for _, p in policy.get_perturbable_params()
                         if p.grad is not None])

        dn = 0.0
        with torch.no_grad():
            for _, p in policy.get_perturbable_params():
                d = torch.randn_like(p.data) * eps
                p.data += d
                dn += (d.float()**2).sum().item()
        dn = float(np.sqrt(dn))

        policy.model.zero_grad()
        policy.action_head.zero_grad()
        logits = policy.get_logits(d_ref["texts"][i])
        torch.log_softmax(logits, dim=-1)[0].backward()
        g1 = torch.cat([p.grad.data.float().flatten() for _, p in policy.get_perturbable_params()
                         if p.grad is not None])

        ratio = float(torch.norm(g1 - g0).item()) / max(dn, 1e-12)
        max_ratio = max(max_ratio, ratio)

        with torch.no_grad():
            for n, p in policy.get_perturbable_params():
                p.data.copy_(original[n])

    return max_ratio


def _train_cost_critics(d_ref: dict, model_dir: Path, cfg: TrainingConfig, device: str) -> None:
    """Train all 6 cost critic variants."""
    cdir = model_dir / "cost_critics"
    cdir.mkdir(exist_ok=True)

    s, a, c, cl = d_ref["states"], d_ref["actions"], d_ref["costs"], d_ref["classes"]
    n = s.shape[0]

    critic_cfg = CostCriticConfig(
        hidden_dim=cfg.cost_critic_hidden,
        n_epochs=cfg.cost_critic_epochs,
        lr=cfg.cost_critic_lr,
    )

    for name, bias, frac in [
        ("full", None, 1.0),
        ("degraded_10pct", None, 0.1),
        ("degraded_25pct", None, 0.25),
        ("degraded_50pct", None, 0.5),
        ("biased_fast_02", {1: 0.2}, 1.0),
        ("biased_fast_05", {1: 0.5}, 1.0),
    ]:
        print(f"    {name}...")
        rng = np.random.default_rng(42)
        if frac < 1.0:
            idx = rng.choice(n, max(50, int(n * frac)), replace=False)
            ss, sa, sc, scl = s[idx], a[idx], c[idx], cl[idx]
        else:
            ss, sa, sc, scl = s, a, c, cl

        model = train_cost_critic(
            ss, sa, sc, config=critic_cfg,
            cost_bias=bias, obstacle_classes=scl if bias else None,
            device=device,
        )
        torch.save(model.state_dict(), cdir / f"{name}.pt")


def _validate_components(
    policy: LLMPolicy, d_ref: dict, fisher: torch.Tensor,
    model_dir: Path, device: str,
) -> dict:
    """Validate each component works before running experiments."""
    results = {}

    # 1. W_0 immutability
    h1 = policy.get_w0_hash()
    policy.restore_w0()
    h2 = policy._compute_hash()
    results["w0_immutable"] = h1 == h2
    print(f"    W_0 immutability: {'PASS' if results['w0_immutable'] else 'FAIL'}")

    # 2. Fisher has meaningful values
    f_min, f_max = fisher.min().item(), fisher.max().item()
    results["fisher_meaningful"] = f_min > 1e-6 and f_max > f_min * 10
    print(f"    Fisher meaningful (f_min={f_min:.6f}, range={f_max/max(f_min,1e-12):.0f}x): "
          f"{'PASS' if results['fisher_meaningful'] else 'FAIL'}")

    # 3. DR actually changes action probabilities
    obs = d_ref["states"][0].numpy()
    policy.restore_w0()
    with torch.no_grad():
        logits_before = policy.get_logits_from_obs(obs).clone()

    # Apply one DR step
    policy.model.zero_grad()
    policy.action_head.zero_grad()
    logits = policy.get_logits_from_obs(obs)
    safe_action = 2  # BRAKE
    torch.log_softmax(logits, dim=-1)[safe_action].backward()
    with torch.no_grad():
        for _, p in policy.get_perturbable_params():
            if p.grad is not None:
                p.data += 0.01 * p.grad.data

    with torch.no_grad():
        logits_after = policy.get_logits_from_obs(obs)

    prob_diff = (torch.softmax(logits_after, -1) - torch.softmax(logits_before, -1)).abs().max().item()
    results["dr_changes_probs"] = prob_diff > 1e-4
    print(f"    DR changes probs (max diff={prob_diff:.6f}): "
          f"{'PASS' if results['dr_changes_probs'] else 'FAIL'}")

    # 4. FHR recovers WDN
    policy.restore_w0()
    w0 = policy.get_w0()
    # Perturb
    with torch.no_grad():
        for name, p in policy.get_perturbable_params():
            p.data += torch.randn_like(p.data) * 0.01
    wdn_before = sum(((p.data - w0[n]).float()**2).sum().item()
                     for n, p in policy.get_perturbable_params() if n in w0) ** 0.5

    # Apply FHR
    with torch.no_grad():
        pidx = 0
        for name, p in policy.get_perturbable_params():
            if name not in w0:
                pidx += p.numel()
                continue
            diff = w0[name] - p.data
            ne = p.numel()
            fs = fisher[pidx:pidx+ne].reshape(p.shape).to(p.dtype).to(p.device)
            pidx += ne
            p.data += 0.01 * fs * diff

    wdn_after = sum(((p.data - w0[n]).float()**2).sum().item()
                    for n, p in policy.get_perturbable_params() if n in w0) ** 0.5
    results["fhr_reduces_wdn"] = wdn_after < wdn_before
    print(f"    FHR reduces WDN ({wdn_before:.6f} -> {wdn_after:.6f}): "
          f"{'PASS' if results['fhr_reduces_wdn'] else 'FAIL'}")

    policy.restore_w0()
    return results


# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Day 1: Train base policies")
    parser.add_argument("--models", nargs="*", help="Model names to train (default: all)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ppo-steps", type=int, default=100_000)
    parser.add_argument("--d-ref-size", type=int, default=10_000)
    args = parser.parse_args()

    train_cfg = TrainingConfig(
        ppo_total_steps=args.ppo_steps,
        d_ref_size=args.d_ref_size,
    )
    exp_cfg = ExperimentConfig(device=args.device)
    exp_cfg.output_base.mkdir(parents=True, exist_ok=True)

    models_to_train = MODELS
    if args.models:
        models_to_train = [m for m in MODELS if m.name in args.models]

    print(f"Day 1: Training {len(models_to_train)} models")
    print(f"PPO steps: {train_cfg.ppo_total_steps}, D_ref: {train_cfg.d_ref_size}")
    print(f"Models: {[m.name for m in models_to_train]}")

    all_logs = {}
    for model_cfg in models_to_train:
        try:
            log = train_single_model(model_cfg, train_cfg, exp_cfg)
            all_logs[model_cfg.name] = log
        except Exception as e:
            print(f"\n  ERROR training {model_cfg.name}: {e}")
            import traceback
            traceback.print_exc()
            all_logs[model_cfg.name] = {"error": str(e)}

    with open(exp_cfg.output_base / "day1_summary.json", "w") as f:
        json.dump(all_logs, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("DAY 1 COMPLETE")
    print(f"{'='*70}")
    for name, log in all_logs.items():
        if "error" in log:
            print(f"  {name}: FAILED — {log['error']}")
        else:
            print(f"  {name}: CR={log.get('base_cr','?')}, "
                  f"params={log.get('n_perturbable_params','?')}, "
                  f"f_range={log.get('f_max',0)/max(log.get('f_min',1e-12),1e-12):.0f}x, "
                  f"{log.get('elapsed_seconds',0):.0f}s")


if __name__ == "__main__":
    main()

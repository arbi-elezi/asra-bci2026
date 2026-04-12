"""Phase 0: Train base agent and compute all offline artifacts.

This script must run BEFORE any experiment. It produces:
  1. W_0 — Frozen LLM LoRA weights (after action head fine-tuning)
  2. D_ref — Reference dataset (1000 state-action-cost triples)
  3. F̂_I — Diagonal Fisher information matrix over LoRA params
  4. G_max^0, σ_G, L_G — Gradient statistics from D_ref
  5. r — Neighborhood radius: r = (G_max - G_max^0) / L_G
  6. Cost critics — Full + 5 degraded/biased variants for C8a–e
  7. Seed set — 1200 seeds (1000 experiment + 200 validation)

All artifacts are versioned and checksummed.
Every computation is seeded for reproducibility.

Usage:
  python train_base.py --device cuda --output-dir checkpoints
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.environment.highway_wrapper import HighwayDrivingEnv
from src.agents.llm_driver import LLMDriver, LLMDriverConfig
from src.agents.cost_critic import (
    CostCriticNet,
    CostCriticConfig,
    train_cost_critic,
)


def generate_seed_set(n_total: int = 1200, master_seed: int = 2026) -> np.ndarray:
    """Generate the master seed set.

    1000 experiment seeds + 200 validation seeds.
    All from the same seeded generator.
    """
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 2**31, size=n_total)
    return seeds


def collect_d_ref(
    env: HighwayDrivingEnv,
    llm: LLMDriver,
    n_pairs: int = 1000,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Collect D_ref: reference dataset for Fisher/Lipschitz computation.

    Runs the LLM policy for n_pairs steps, recording (state, action, cost, class).

    Returns:
        Dict with tensors: states [N, 12], actions [N], costs [N], classes [N],
        state_texts [N] (list of strings).
    """
    states = []
    actions = []
    costs = []
    classes = []
    state_texts = []

    obs, info = env.reset(seed=seed)
    collected = 0

    while collected < n_pairs:
        text = env.get_state_text(obs)
        action, probs = llm.get_action(text)

        states.append(obs.copy())
        actions.append(action)
        costs.append(info.get("cost", 0.0))
        classes.append(info.get("nearest_class", -1))
        state_texts.append(text)
        collected += 1

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset(seed=seed + collected)

    return {
        "states": torch.tensor(np.array(states), dtype=torch.float32),
        "actions": torch.tensor(actions, dtype=torch.long),
        "costs": torch.tensor(costs, dtype=torch.float32),
        "classes": torch.tensor(classes, dtype=torch.long),
        "state_texts": state_texts,
    }


def finetune_action_head(
    env: HighwayDrivingEnv,
    llm: LLMDriver,
    n_episodes: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
) -> None:
    """Fine-tune the action head on driving episodes.

    Uses REINFORCE to train the action head to produce reasonable actions.
    The LoRA params are also trained but with lower LR.
    After this, W_0 is snapshotted and frozen.
    """
    print("Fine-tuning action head...")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Separate LR for action head vs LoRA
    optimizer = optim.Adam([
        {"params": llm.action_head.parameters(), "lr": lr},
        {"params": [p for _, p in llm.get_perturbable_params() if "lora" in _], "lr": lr * 0.1},
    ])

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        log_probs = []
        rewards = []

        for t in range(200):  # Short episodes for training
            text = env.get_state_text(obs)
            logits = llm.get_action_logits(text)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            obs, reward, terminated, truncated, info = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)

            if terminated or truncated:
                break

        # REINFORCE
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.tensor(returns, device=dev)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for lp, ret in zip(log_probs, returns):
            loss -= lp * ret

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{n_episodes} | reward={sum(rewards):.1f}")

    # Re-snapshot W_0 after fine-tuning
    llm.w0_lora = llm._snapshot_lora_params()
    llm.w0_action_head = {
        k: v.clone().detach() for k, v in llm.action_head.state_dict().items()
    }
    llm._w0_hash = llm._compute_hash()
    print(f"W_0 frozen. Hash: {llm.get_w0_hash()[:16]}...")


def compute_gradient_stats(
    llm: LLMDriver,
    d_ref_texts: list[str],
    device: str = "cuda",
) -> dict[str, float]:
    """Compute G_max^0, σ_G from D_ref.

    G_max^0 = max_{D_ref} ||G^DR||_F + 2·σ_G

    Returns:
        Dict with g_max_0, sigma_g, g_mean, max_raw.
    """
    grad_norms = []

    for text in d_ref_texts:
        llm.model.zero_grad()
        llm.action_head.zero_grad()

        logits = llm.get_action_logits(text)
        # Gradient for each action
        for a in range(4):
            llm.model.zero_grad()
            llm.action_head.zero_grad()
            log_prob = torch.log_softmax(logits, dim=-1)[a]
            log_prob.backward(retain_graph=(a < 3))

            norm_sq = 0.0
            for _, param in llm.get_perturbable_params():
                if param.grad is not None:
                    norm_sq += (param.grad.data.float() ** 2).sum().item()
            grad_norms.append(float(np.sqrt(norm_sq)))

    norms = np.array(grad_norms)
    max_raw = float(norms.max())
    sigma_g = float(norms.std())
    g_max_0 = max_raw + 2 * sigma_g

    return {
        "g_max_0": g_max_0,
        "sigma_g": sigma_g,
        "g_mean": float(norms.mean()),
        "max_raw": max_raw,
    }


def estimate_lipschitz(
    llm: LLMDriver,
    d_ref_texts: list[str],
    n_samples: int = 50,
    epsilon: float = 1e-4,
) -> float:
    """Estimate Lipschitz constant L_G via finite differences.

    L_G = max_{s,a} ||G(W_0 + δ) - G(W_0)|| / ||δ||

    Uses random perturbations δ on LoRA params.
    """
    max_ratio = 0.0

    for i in range(min(n_samples, len(d_ref_texts))):
        text = d_ref_texts[i]

        # Gradient at W_0
        llm.restore_to_w0()
        llm.model.zero_grad()
        llm.action_head.zero_grad()
        logits = llm.get_action_logits(text)
        log_prob = torch.log_softmax(logits, dim=-1)[0]
        log_prob.backward()

        g0 = []
        for _, param in llm.get_perturbable_params():
            if param.grad is not None:
                g0.append(param.grad.data.float().flatten().clone())
            else:
                g0.append(torch.zeros(param.numel()))
        g0_vec = torch.cat(g0)

        # Perturb W_0 by small δ
        delta_norm = 0.0
        with torch.no_grad():
            for _, param in llm.get_perturbable_params():
                delta = torch.randn_like(param.data) * epsilon
                param.data += delta
                delta_norm += (delta.float() ** 2).sum().item()
        delta_norm = float(np.sqrt(delta_norm))

        # Gradient at W_0 + δ
        llm.model.zero_grad()
        llm.action_head.zero_grad()
        logits = llm.get_action_logits(text)
        log_prob = torch.log_softmax(logits, dim=-1)[0]
        log_prob.backward()

        g1 = []
        for _, param in llm.get_perturbable_params():
            if param.grad is not None:
                g1.append(param.grad.data.float().flatten().clone())
            else:
                g1.append(torch.zeros(param.numel()))
        g1_vec = torch.cat(g1)

        # L_G estimate
        grad_diff = float(torch.norm(g1_vec - g0_vec).item())
        if delta_norm > 0:
            ratio = grad_diff / delta_norm
            max_ratio = max(max_ratio, ratio)

        # Restore W_0
        llm.restore_to_w0()

    return max_ratio


def train_all_cost_critics(
    d_ref: dict[str, torch.Tensor],
    output_dir: Path,
    device: str = "cuda",
) -> None:
    """Train all 7 cost critic variants.

    1. Full (C2) — full D_ref
    2. Degraded 10% (C8a) — 10% D_ref, FROM SCRATCH
    3. Degraded 25% (C8b) — 25% D_ref, FROM SCRATCH
    4. Degraded 50% (C8c) — 50% D_ref, FROM SCRATCH
    5. Biased ×0.2 (C8d) — full D_ref, fast costs ×0.2
    6. Biased ×0.5 (C8e) — full D_ref, fast costs ×0.5
    """
    critic_dir = output_dir / "cost_critic"
    critic_dir.mkdir(parents=True, exist_ok=True)

    states = d_ref["states"]
    actions = d_ref["actions"]
    costs = d_ref["costs"]
    classes = d_ref["classes"]
    n = states.shape[0]

    configs = [
        ("full", None, None, 1.0),        # Full D_ref
        ("degraded_10pct", None, None, 0.1),  # 10%
        ("degraded_25pct", None, None, 0.25), # 25%
        ("degraded_50pct", None, None, 0.5),  # 50%
        ("biased_fast_02", {1: 0.2}, None, 1.0),  # Fast ×0.2
        ("biased_fast_05", {1: 0.5}, None, 1.0),  # Fast ×0.5
    ]

    for name, bias, _, frac in configs:
        print(f"  Training cost critic: {name}...")

        # Subsample for degraded versions
        if frac < 1.0:
            rng = np.random.default_rng(42)
            n_sub = max(10, int(n * frac))
            idx = rng.choice(n, n_sub, replace=False)
            s, a, c, cl = states[idx], actions[idx], costs[idx], classes[idx]
        else:
            s, a, c, cl = states, actions, costs, classes

        model = train_cost_critic(
            d_ref_states=s,
            d_ref_actions=a,
            d_ref_costs=c,
            cost_bias=bias,
            obstacle_classes=cl if bias else None,
            device=device,
        )

        path = critic_dir / f"{name}.pt"
        torch.save(model.state_dict(), path)
        print(f"    Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Train base agent + offline artifacts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--n-finetune", type=int, default=100, help="Action head fine-tuning episodes")
    parser.add_argument("--n-dref", type=int, default=1000, help="D_ref size")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── 1. Generate seed set ──
    print("Step 1: Generating seed set...")
    seeds = generate_seed_set(1200, master_seed=2026)
    np.save(output_dir / "seed_set.npy", seeds)
    print(f"  Seeds: {len(seeds)} (1000 experiment + 200 validation)")

    # ── 2. Initialize environment ──
    print("Step 2: Initializing environment...")
    env = HighwayDrivingEnv(seed=0)
    print(f"  State: R^{env.observation_space.shape[0]}, Actions: {env.action_space.n}")

    # ── 3. Initialize LLM ──
    print("Step 3: Loading LLM (TinyLlama + LoRA)...")
    llm = LLMDriver(LLMDriverConfig(device=args.device))
    print(f"  Perturbable params: {llm.count_perturbable_params():,}")

    # ── 4. Fine-tune action head ──
    print("Step 4: Fine-tuning action head...")
    finetune_action_head(env, llm, n_episodes=args.n_finetune, device=args.device)

    # Save W_0
    w0_dir = output_dir / "ppo_base"
    w0_dir.mkdir(exist_ok=True)
    torch.save({
        "lora": llm.w0_lora,
        "action_head": llm.w0_action_head,
        "hash": llm.get_w0_hash(),
    }, w0_dir / "w0.pt")
    print(f"  W_0 saved: {w0_dir / 'w0.pt'}")

    # ── 5. Collect D_ref ──
    print(f"Step 5: Collecting D_ref ({args.n_dref} pairs)...")
    d_ref = collect_d_ref(env, llm, n_pairs=args.n_dref, seed=42)
    torch.save(d_ref, output_dir / "d_ref.pt")
    print(f"  D_ref: {d_ref['states'].shape[0]} pairs")

    # ── 6. Compute Fisher diagonal ──
    print("Step 6: Computing Fisher information matrix...")
    from src.agents.fra_engine import FRAEngine, FRAEngineConfig
    fra_tmp = FRAEngine(llm, FRAEngineConfig())
    fisher = fra_tmp.compute_fisher_diagonal(d_ref["state_texts"])
    torch.save(fisher, output_dir / "fisher_diagonal.pt")
    f_min = fisher.min().item()
    f_max = fisher.max().item()
    print(f"  Fisher: {fisher.shape[0]} params, f_min={f_min:.6f}, f_max={f_max:.6f}")

    # ── 7. Compute gradient stats ──
    print("Step 7: Computing gradient statistics (G_max^0, σ_G)...")
    grad_stats = compute_gradient_stats(llm, d_ref["state_texts"], device=args.device)
    print(f"  G_max^0={grad_stats['g_max_0']:.4f}, σ_G={grad_stats['sigma_g']:.4f}")

    # ── 8. Estimate Lipschitz constant ──
    print("Step 8: Estimating Lipschitz constant L_G...")
    l_g = estimate_lipschitz(llm, d_ref["state_texts"])
    print(f"  L_G={l_g:.4f}")

    # ── 9. Compute neighborhood radius ──
    g_max = grad_stats["g_max_0"] * 1.5  # Add margin for G_max > G_max^0
    if l_g > 0:
        r = (g_max - grad_stats["g_max_0"]) / l_g
    else:
        r = float("inf")
    print(f"  r = {r:.4f} (neighborhood radius)")

    # ── 10. Save all pre-computed artifacts ──
    artifacts = {
        "g_max_0": grad_stats["g_max_0"],
        "sigma_g": grad_stats["sigma_g"],
        "g_max": g_max,
        "l_g": l_g,
        "r": r,
        "f_min": f_min,
        "f_max": f_max,
        "w0_hash": llm.get_w0_hash(),
        "n_perturbable_params": llm.count_perturbable_params(),
    }

    # Derive hyperparameter constraints
    # η_f/η_h ≤ D_max · f_min / G_max, with D_max ≤ r
    d_max = min(r, 1.0)  # Don't let D_max exceed 1.0 in practice
    if g_max > 0:
        eta_ratio_max = d_max * f_min / g_max
    else:
        eta_ratio_max = float("inf")
    artifacts["d_max"] = d_max
    artifacts["eta_ratio_max"] = eta_ratio_max

    # A3: η_h < 1/f_max
    if f_max > 0:
        artifacts["eta_h_max"] = 1.0 / f_max
    else:
        artifacts["eta_h_max"] = float("inf")

    with open(output_dir / "artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=2)
    print(f"\nArtifacts saved to {output_dir / 'artifacts.json'}")

    # ── 11. Train cost critics ──
    print("\nStep 11: Training cost critics...")
    train_all_cost_critics(d_ref, output_dir, device=args.device)

    elapsed = time.time() - t_start
    print(f"\n=== Phase 0 Complete ({elapsed:.0f}s) ===")
    print(f"All artifacts in: {output_dir}")
    print(f"W_0 hash: {llm.get_w0_hash()}")
    print(f"Ready to run experiments with: python run_experiment.py --config configs/c2_full_fra.yaml")


if __name__ == "__main__":
    main()

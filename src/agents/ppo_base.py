"""PPO Base Agent — trains the frozen W_0.

Paper spec (Section 4.2):
  - 2-layer MLP actor (64-64-4, softmax)
  - 2-layer MLP critic (64-64-1)
  - Trained 100,000 steps
  - W_0 frozen after training
  - D_ref: 1000 state-action pairs collected from trained policy

This module:
  1. Defines the actor-critic architecture
  2. Trains via PPO (SB3 or custom)
  3. Freezes W_0 and saves checkpoint
  4. Collects D_ref
  5. Computes Fisher information matrix (diagonal)
  6. Computes G_max^0, L_G, r
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """2-layer MLP actor-critic per paper spec.

    Actor: 12 → 64 → 64 → 4 (softmax)
    Critic: 12 → 64 → 64 → 1
    """

    def __init__(self, obs_dim: int = 12, act_dim: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        logits = self.actor(obs)
        value = self.critic(obs)
        dist = Categorical(logits=logits)
        return dist, value.squeeze(-1)

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, return (action, log_prob, value)."""
        dist, value = self.forward(obs)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given action, return (log_prob, value, entropy)."""
        dist, value = self.forward(obs)
        return dist.log_prob(action), value, dist.entropy()


class PPOTrainer:
    """Custom PPO trainer for the base agent.

    Uses custom implementation (not SB3) for full control over:
    - Checkpoint saving every 1K steps
    - Exact architecture matching paper spec
    - D_ref collection
    - Fisher computation
    """

    def __init__(
        self,
        env,
        model: ActorCritic | None = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        n_steps: int = 2048,
        seed: int = 42,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/ppo_base",
    ) -> None:
        self.env = env
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model or ActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Seeded RNG
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Metrics
        self.total_steps = 0
        self.episode_rewards: list[float] = []

    def collect_rollout(self) -> dict[str, torch.Tensor]:
        """Collect n_steps of experience."""
        obs_buf = []
        act_buf = []
        logp_buf = []
        val_buf = []
        rew_buf = []
        done_buf = []

        obs, _ = self.env.reset(seed=self.seed + self.total_steps)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        ep_reward = 0.0

        for _ in range(self.n_steps):
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(obs_t.unsqueeze(0))

            action_np = action.item()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)

            obs_buf.append(obs_t)
            act_buf.append(action)
            logp_buf.append(log_prob)
            val_buf.append(value)
            rew_buf.append(reward)
            done_buf.append(float(terminated or truncated))

            ep_reward += reward
            self.total_steps += 1

            if terminated or truncated:
                self.episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = self.env.reset(seed=self.seed + self.total_steps)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            else:
                obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

            # Checkpoint every 1K steps (coding rule)
            if self.total_steps % 1000 == 0:
                self.save_checkpoint(f"step_{self.total_steps}")

        return {
            "obs": torch.stack(obs_buf),
            "actions": torch.cat(act_buf),
            "log_probs": torch.cat(logp_buf),
            "values": torch.cat(val_buf),
            "rewards": torch.tensor(rew_buf, device=self.device),
            "dones": torch.tensor(done_buf, device=self.device),
        }

    def compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0.0
            else:
                next_val = values[t + 1].item()

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        """PPO update from collected rollout."""
        obs = rollout["obs"]
        actions = rollout["actions"]
        old_logp = rollout["log_probs"]

        advantages, returns = self.compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"]
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.epochs):
            indices = torch.randperm(len(obs), device=self.device)
            for start in range(0, len(obs), self.batch_size):
                end = min(start + self.batch_size, len(obs))
                idx = indices[start:end]

                new_logp, new_val, entropy = self.model.evaluate(obs[idx], actions[idx])

                ratio = torch.exp(new_logp - old_logp[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]

                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = 0.5 * (new_val - returns[idx]).pow(2).mean()
                ent_loss = -entropy.mean()

                loss = pg_loss + 0.5 * vf_loss + 0.01 * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "pg_loss": total_pg_loss / max(n_updates, 1),
            "vf_loss": total_vf_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "mean_reward": np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0,
        }

    def train(self, total_steps: int = 100_000) -> None:
        """Train for total_steps."""
        # Save initial checkpoint
        self.save_checkpoint("step_0")

        while self.total_steps < total_steps:
            rollout = self.collect_rollout()
            metrics = self.update(rollout)

            if self.total_steps % 5000 == 0:
                print(
                    f"Step {self.total_steps:>6d} | "
                    f"PG: {metrics['pg_loss']:.4f} | "
                    f"VF: {metrics['vf_loss']:.4f} | "
                    f"Ent: {metrics['entropy']:.4f} | "
                    f"Rew: {metrics['mean_reward']:.2f}"
                )

        # Final checkpoint
        self.save_checkpoint("final")
        print(f"Training complete. Total steps: {self.total_steps}")

    def save_checkpoint(self, name: str) -> Path:
        """Save model checkpoint with W_0 hash."""
        path = self.checkpoint_dir / f"{name}.pt"
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episode_rewards": self.episode_rewards,
            "seed": self.seed,
            "w0_hash": self.compute_w0_hash(),
        }
        torch.save(state, path)
        return path

    def compute_w0_hash(self) -> str:
        """SHA-256 hash of all model parameters — used to verify W_0 is frozen."""
        hasher = hashlib.sha256()
        for param in self.model.parameters():
            hasher.update(param.data.cpu().numpy().tobytes())
        return hasher.hexdigest()

    def freeze_w0(self) -> dict[str, torch.Tensor]:
        """Freeze W_0 — return a deep copy of current weights.

        After this call, W_0 is IMMUTABLE. The FRA wrapper operates on a
        SEPARATE copy of weights that it perturbs.
        """
        w0 = {}
        for name, param in self.model.named_parameters():
            w0[name] = param.data.clone().detach()
            # Do NOT freeze the parameter itself — the FRA wrapper needs
            # a mutable copy. W_0 is the snapshot, not the live weights.
        return w0


def collect_d_ref(
    model: ActorCritic,
    env,
    n_pairs: int = 1000,
    seed: int = 42,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Collect D_ref: 1000 (state, action) pairs from the trained policy.

    Used for:
    - Fisher information matrix computation
    - G_max^0 estimation
    - L_G estimation via finite differences
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    states = []
    actions = []
    log_probs = []

    obs, _ = env.reset(seed=seed)
    collected = 0

    while collected < n_pairs:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
        with torch.no_grad():
            dist, _ = model.forward(obs_t)
            action = dist.sample()
            lp = dist.log_prob(action)

        states.append(obs_t.squeeze(0))
        actions.append(action.squeeze(0))
        log_probs.append(lp.squeeze(0))
        collected += 1

        next_obs, _, terminated, truncated, _ = env.step(action.item())
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + collected)
        else:
            obs = next_obs

    return {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "log_probs": torch.stack(log_probs),
    }


def compute_fisher_diagonal(
    model: ActorCritic,
    d_ref: dict[str, torch.Tensor],
    device: str = "cuda",
) -> torch.Tensor:
    """Compute diagonal empirical Fisher information matrix from D_ref.

    F̂_I = (1/N) Σ (∇_W log π_W(a|s))^2

    Returns a single flat tensor of diagonal Fisher values.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    states = d_ref["states"].to(dev)
    actions = d_ref["actions"].to(dev)

    # Accumulate squared gradients
    fisher_diag = None
    n = states.shape[0]

    for i in range(n):
        model.zero_grad()
        dist, _ = model.forward(states[i:i+1])
        log_prob = dist.log_prob(actions[i:i+1])
        log_prob.backward()

        grads = []
        for p in model.actor.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.clone().flatten())
        grad_vec = torch.cat(grads)

        if fisher_diag is None:
            fisher_diag = grad_vec ** 2
        else:
            fisher_diag += grad_vec ** 2

    fisher_diag /= n

    # Numerical safety: ensure f_min > 0 (A2)
    f_min = fisher_diag.min().item()
    if f_min <= 0:
        fisher_diag = fisher_diag + 1e-8  # Floor at epsilon

    return fisher_diag


def compute_gradient_stats(
    model: ActorCritic,
    d_ref: dict[str, torch.Tensor],
    device: str = "cuda",
) -> dict[str, float]:
    """Compute G_max^0, sigma_G, and estimate L_G from D_ref.

    G_max^0 = max_{D_ref} ||G^DR||_F + 2*sigma_G
    L_G = Lipschitz constant via finite differences
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    states = d_ref["states"].to(dev)

    grad_norms = []

    for i in range(states.shape[0]):
        model.zero_grad()
        dist, _ = model.forward(states[i:i+1])
        # G^DR = -∇_W log π_W(a_safe | s)
        # For now, use argmin-cost action (action 2 = BRAKE as proxy for safe)
        safe_action = torch.tensor([2], device=dev)
        log_prob = dist.log_prob(safe_action)
        log_prob.backward()

        grads = []
        for p in model.actor.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.clone().flatten())
        grad_vec = torch.cat(grads)
        grad_norms.append(grad_vec.norm().item())

    grad_norms_t = torch.tensor(grad_norms)
    g_max_raw = grad_norms_t.max().item()
    sigma_g = grad_norms_t.std().item()
    g_max_0 = g_max_raw + 2 * sigma_g

    # Estimate L_G via finite differences
    # For each parameter, perturb by epsilon and measure gradient change
    epsilon = 1e-3
    grad_diffs = []

    # Use first 100 states for Lipschitz estimation
    n_lip = min(100, states.shape[0])
    for i in range(n_lip):
        # Gradient at current weights
        model.zero_grad()
        dist, _ = model.forward(states[i:i+1])
        safe_action = torch.tensor([2], device=dev)
        log_prob = dist.log_prob(safe_action)
        log_prob.backward()
        g1 = torch.cat([p.grad.data.clone().flatten() for p in model.actor.parameters() if p.grad is not None])

        # Perturb weights
        with torch.no_grad():
            perturbation = {}
            for name, p in model.actor.named_parameters():
                perturbation[name] = torch.randn_like(p) * epsilon
                p.data += perturbation[name]

        # Gradient at perturbed weights
        model.zero_grad()
        dist2, _ = model.forward(states[i:i+1])
        log_prob2 = dist2.log_prob(safe_action)
        log_prob2.backward()
        g2 = torch.cat([p.grad.data.clone().flatten() for p in model.actor.parameters() if p.grad is not None])

        # Restore weights
        with torch.no_grad():
            for name, p in model.actor.named_parameters():
                p.data -= perturbation[name]

        # ||G(W+δ) - G(W)|| / ||δ||
        delta_norm = torch.cat([v.flatten() for v in perturbation.values()]).norm().item()
        if delta_norm > 0:
            grad_diffs.append((g2 - g1).norm().item() / delta_norm)

    l_g = max(grad_diffs) if grad_diffs else 1.0

    return {
        "g_max_0": g_max_0,
        "sigma_g": sigma_g,
        "l_g": l_g,
        "g_max_raw": g_max_raw,
    }


def compute_neighborhood_radius(g_max_0: float, l_g: float, margin: float = 1.5) -> dict[str, float]:
    """Compute r = (G_max - G_max^0) / L_G.

    G_max is chosen with sufficient margin above G_max^0.
    """
    g_max = g_max_0 * margin  # 50% margin
    if l_g <= 0:
        l_g = 1e-6  # Safety: avoid division by zero

    r = (g_max - g_max_0) / l_g

    return {
        "g_max": g_max,
        "g_max_0": g_max_0,
        "l_g": l_g,
        "r": r,
        "margin": margin,
    }

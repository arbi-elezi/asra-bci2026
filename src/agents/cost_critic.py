"""Cost Critic — Offline cost model Ĉ for FRA.

Paper Section 4.2:
  - Trained offline on D_ref
  - Frozen after training — no online updates
  - Used by DR for safe action computation (Definition 2)
  - Used for F_t^CA computation (Equation 11)
  - Degraded versions for C8a-c (trained on subsets of D_ref)
  - Biased versions for C8d-e (modified cost labels during training)

Rule 5: Degraded critics trained FROM SCRATCH, not by corrupting full critic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


@dataclass
class CostCriticConfig:
    obs_dim: int = 12
    n_actions: int = 4
    hidden_dim: int = 64
    lr: float = 1e-3
    n_epochs: int = 100
    batch_size: int = 64


class CostCriticNet(nn.Module):
    """Cost critic: (state, action_onehot) → expected future cost.

    Architecture: (12 + 4) → 64 → 64 → 1
    """

    def __init__(self, obs_dim: int = 12, n_actions: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, sa: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: [batch, obs_dim + n_actions]."""
        return self.net(sa)

    def predict_state(self, obs: torch.Tensor, n_actions: int = 4) -> torch.Tensor:
        """Predict cost for all actions given a state.

        Args:
            obs: [batch, obs_dim] or [obs_dim]

        Returns:
            costs: [batch, n_actions] or [n_actions]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        batch = obs.shape[0]

        costs = []
        for a in range(n_actions):
            onehot = torch.zeros(batch, n_actions, device=obs.device)
            onehot[:, a] = 1.0
            sa = torch.cat([obs, onehot], dim=1)
            c = self.forward(sa).squeeze(-1)
            costs.append(c)

        return torch.stack(costs, dim=1)  # [batch, n_actions]

    def compute_advantage(self, obs: torch.Tensor, action: int, n_actions: int = 4) -> float:
        """Compute cost advantage A^C(s, a) = Ĉ(s, a) - V^C(s).

        V^C(s) = Σ_a π(a|s) Ĉ(s, a) approximated as mean over actions.
        Used for F_t^CA computation (Equation 11).
        """
        with torch.no_grad():
            all_costs = self.predict_state(obs, n_actions)  # [1, n_actions]
            v_c = all_costs.mean(dim=1)  # [1]
            a_c = all_costs[0, action] - v_c[0]
        return a_c.item()


def train_cost_critic(
    d_ref_states: torch.Tensor,
    d_ref_actions: torch.Tensor,
    d_ref_costs: torch.Tensor,
    config: CostCriticConfig | None = None,
    cost_bias: dict[int, float] | None = None,
    obstacle_classes: torch.Tensor | None = None,
    device: str = "cuda",
) -> CostCriticNet:
    """Train cost critic from D_ref data.

    Args:
        d_ref_states: [N, 12] states from D_ref.
        d_ref_actions: [N] action indices.
        d_ref_costs: [N] observed costs (Monte Carlo returns or immediate).
        config: Training configuration.
        cost_bias: Per-class multiplier for C8d/C8e. Keys are obstacle class ints.
        obstacle_classes: [N] obstacle class for each D_ref entry (for bias).
        device: Compute device.

    Returns:
        Trained (and frozen) cost critic.
    """
    cfg = config or CostCriticConfig()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    model = CostCriticNet(cfg.obs_dim, cfg.n_actions, cfg.hidden_dim).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    states = d_ref_states.to(dev)
    actions = d_ref_actions.to(dev).long()
    costs = d_ref_costs.to(dev).float()

    # Apply cost bias for C8d/C8e stress tests
    if cost_bias is not None and obstacle_classes is not None:
        obs_cls = obstacle_classes.to(dev)
        for cls_id, bias in cost_bias.items():
            mask = obs_cls == cls_id
            costs[mask] = costs[mask] * bias

    # Build (state, action_onehot) pairs
    n = states.shape[0]
    onehots = torch.zeros(n, cfg.n_actions, device=dev)
    onehots.scatter_(1, actions.unsqueeze(1), 1.0)
    sa = torch.cat([states, onehots], dim=1)

    # Train
    model.train()
    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(n, device=dev)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, cfg.batch_size):
            end = min(start + cfg.batch_size, n)
            idx = perm[start:end]

            pred = model(sa[idx]).squeeze(-1)
            loss = criterion(pred, costs[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    # Freeze — no online updates (coding rule)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def compute_fear_signal_ca(
    cost_critic: CostCriticNet,
    obs: torch.Tensor,
    greedy_action: int,
    lambda_a: float = 5.0,
    n_actions: int = 4,
) -> float:
    """Compute cost-advantage fear signal F_t^CA.

    Paper Equation (11):
      F_t^CA = σ(Â^C(s_t, π*_{W_t}(s_t)) · λ_A)

    Args:
        cost_critic: Frozen offline cost critic.
        obs: Current state.
        greedy_action: Current greedy action from policy.
        lambda_a: Scaling factor for sigmoid.
        n_actions: Action space size.

    Returns:
        F_t^CA ∈ [0, 1].
    """
    advantage = cost_critic.compute_advantage(obs, greedy_action, n_actions)
    # Sigmoid activation
    f_ca = 1.0 / (1.0 + np.exp(-advantage * lambda_a))
    # Clip to [0, 1] (numerical safety)
    return max(0.0, min(1.0, f_ca))

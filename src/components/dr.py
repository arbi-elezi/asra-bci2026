"""Deregulator (DR) — Policy Shaping with a Calibrated Trigger.

Paper Equation (3):
  G_t^DR = -∇_W log π_{W_t}(a_safe(s_t) | s_t)
  ΔW_DR = -η_f · F_t · G_t^DR    (when F_t > ε)

DR increases the log-probability of the cost-minimal safe action,
scaled by the calibrated fear signal F_t.

DR is an OPTIONAL EXTENSION with empirically determined benefit.
It does NOT implement or approximate CPO.
It can DEGRADE safety under model bias (C8d-e).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class DRConfig:
    """Typed configuration for DR."""
    enabled: bool = True
    eta_f: float = 0.01       # Fear learning rate
    epsilon: float = 0.05     # Activation threshold
    g_max: float = 10.0       # A1 bound (set from pre-computed values)


class DR:
    """Deregulator — gradient-based policy shaping toward safe action.

    Rules:
    - W_0 is NEVER modified by DR
    - DR only modifies the perturbation copy of weights
    - G_t^DR norm is logged every timestep (M13, mandatory)
    - DR is disabled when fear F_t ≤ ε
    """

    def __init__(self, config: DRConfig, device: str = "cuda") -> None:
        self.cfg = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Metrics
        self._grad_norms: list[float] = []
        self._a1_violations: int = 0
        self._total_steps: int = 0

    def step(
        self,
        model: nn.Module,
        obs: torch.Tensor,
        fear: float,
        safe_action: int,
    ) -> dict[str, torch.Tensor]:
        """Compute DR weight update.

        Args:
            model: Actor-critic model (W_t, the perturbation copy).
            obs: Current observation [1, 12] or [12].
            fear: Current fear signal F_t ∈ [0, 1].
            safe_action: Cost-minimal safe action from Definition 2.

        Returns:
            Dict mapping param names to DR update tensors.
            Empty dict if DR disabled or fear below threshold.
        """
        self._total_steps += 1

        if not self.cfg.enabled:
            self._grad_norms.append(0.0)
            return {}

        # Clip fear to [0, 1] (numerical safety)
        fear = max(0.0, min(1.0, fear))

        if fear <= self.cfg.epsilon:
            self._grad_norms.append(0.0)
            return {}

        # Ensure obs is batched
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Compute G_t^DR = -∇_W log π_{W_t}(a_safe | s_t)
        model.zero_grad()
        dist, _ = model.forward(obs)
        safe_action_t = torch.tensor([safe_action], device=self.device)
        log_prob = dist.log_prob(safe_action_t)
        log_prob.backward()

        # Collect gradients and compute update
        updates = {}
        grad_norms_sq = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        for name, param in model.actor.named_parameters():
            if param.grad is not None:
                g = param.grad.data.clone()
                # DR update: -η_f · F_t · G^DR = -η_f · F_t · (-∇ log π) = η_f · F_t · ∇ log π
                # Wait — paper says G^DR = -∇_W log π, and ΔW = -η_f · F_t · G^DR
                # So ΔW = -η_f · F_t · (-∇ log π) = η_f · F_t · ∇ log π
                # This INCREASES log probability of safe action ✓
                updates[name] = self.cfg.eta_f * fear * g

                grad_norms_sq += (g.double() ** 2).sum()

        grad_norm = torch.sqrt(grad_norms_sq).item()
        self._grad_norms.append(grad_norm)

        # M13: Check A1 violation
        if grad_norm > self.cfg.g_max:
            self._a1_violations += 1

        return updates

    def compute_safe_action(
        self,
        cost_critic: nn.Module,
        obs: torch.Tensor,
        n_actions: int = 4,
        delta: float = 0.5,
    ) -> int:
        """Compute a_safe(s_t) per Definition 2.

        a_safe = argmin_{a ∈ A_safe} Ĉ(s_t, a)
        where A_safe = {a ∈ A : Ĉ(s_t, a) ≤ δ}
        Fallback: argmin_{a ∈ A} Ĉ(s_t, a) when A_safe = ∅

        Args:
            cost_critic: Offline cost model Ĉ.
            obs: Current state [12].
            n_actions: Number of actions (4).
            delta: Safety threshold.

        Returns:
            Safe action index.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            costs = []
            for a in range(n_actions):
                # Cost critic takes (state, action) → expected cost
                action_onehot = torch.zeros(1, n_actions, device=self.device)
                action_onehot[0, a] = 1.0
                sa = torch.cat([obs, action_onehot], dim=1)
                cost = cost_critic(sa).squeeze()
                costs.append(cost.item())

        costs_t = torch.tensor(costs, device=self.device)

        # A_safe = {a : C(s,a) ≤ δ}
        safe_mask = costs_t <= delta
        if safe_mask.any():
            # argmin within safe set
            safe_costs = costs_t.clone()
            safe_costs[~safe_mask] = float("inf")
            return safe_costs.argmin().item()
        else:
            # Fallback: argmin over all actions
            return costs_t.argmin().item()

    def get_metrics(self) -> dict[str, Any]:
        """Return DR metrics including M13 violation fraction."""
        total = max(self._total_steps, 1)
        return {
            "grad_norms": self._grad_norms.copy(),
            "a1_violation_count": self._a1_violations,
            "a1_violation_fraction": self._a1_violations / total,
            "grad_norm_max": max(self._grad_norms) if self._grad_norms else 0.0,
            "grad_norm_mean": sum(self._grad_norms) / len(self._grad_norms) if self._grad_norms else 0.0,
        }

    def reset(self) -> None:
        """Reset per-episode state."""
        self._grad_norms = []
        self._a1_violations = 0
        self._total_steps = 0

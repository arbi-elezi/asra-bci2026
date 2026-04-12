"""Safe Control Layer (SCL) — Action-level safety mixing.

Paper Section 3.10:
  a_t = (1 - α_t) · π_{W_t}(s_t) + α_t · a_safe(s_t)

SCL does NOT modify W_t. It operates at the action level only.
Lemma 1 confirms it does not enter the weight recurrence (8).

A4 tests DR necessity beyond SCL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import numpy as np


@dataclass
class SCLConfig:
    enabled: bool = True
    tau: float = 0.5     # Fear threshold for activation [0.2, 0.8]
    k: float = 10.0      # Mixing steepness [1, 20]


class SCL:
    """Safe Control Layer — action-level safety mixing."""

    def __init__(self, config: SCLConfig) -> None:
        self.cfg = config
        self._alpha_history: list[float] = []
        self._override_count: int = 0
        self._total_steps: int = 0

    def step(
        self,
        policy_action_probs: torch.Tensor,
        safe_action: int,
        fear: float,
        n_actions: int = 4,
    ) -> tuple[int, float]:
        """Compute SCL-mixed action.

        Args:
            policy_action_probs: π_{W_t}(·|s_t) probabilities [n_actions].
            safe_action: Cost-minimal safe action index.
            fear: Current fear signal F_t ∈ [0, 1].
            n_actions: Number of actions.

        Returns:
            (selected_action, alpha_t)
        """
        self._total_steps += 1

        if not self.cfg.enabled:
            action = policy_action_probs.argmax().item()
            self._alpha_history.append(0.0)
            return action, 0.0

        # Compute mixing coefficient α_t
        # Sigmoid-like activation: α_t = sigmoid(k * (F_t - τ))
        alpha = 1.0 / (1.0 + np.exp(-self.cfg.k * (fear - self.cfg.tau)))
        alpha = max(0.0, min(1.0, alpha))

        self._alpha_history.append(alpha)

        # Mix action distributions
        safe_probs = torch.zeros(n_actions, device=policy_action_probs.device)
        safe_probs[safe_action] = 1.0

        mixed_probs = (1 - alpha) * policy_action_probs + alpha * safe_probs
        mixed_probs = mixed_probs / (mixed_probs.sum() + 1e-8)  # Normalize

        # Sample from mixed distribution
        action = torch.multinomial(mixed_probs, 1).item()

        # Track overrides (M5: α > 0.5)
        if alpha > 0.5:
            self._override_count += 1

        return action, alpha

    def get_metrics(self) -> dict[str, Any]:
        return {
            "alpha_history": self._alpha_history.copy(),
            "override_count": self._override_count,
            "override_fraction": self._override_count / max(self._total_steps, 1),
        }

    def reset(self) -> None:
        self._alpha_history = []
        self._override_count = 0
        self._total_steps = 0

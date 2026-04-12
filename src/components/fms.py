"""Fear Modulation Signal (FMS) — Distribution Shift Correction.

Paper Section 3.10:
  Distribution shift correction via similarity-weighted
  deployment-vs-critic discrepancy.

  FMS modifies F_t via correction δ_t based on how different
  the current state distribution is from D_ref.

Evaluated by A3 (C5 vs C2). M12 reports |δ_t| distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import numpy as np


@dataclass
class FMSConfig:
    enabled: bool = True
    k_neighbors: int = 5     # k-nearest neighbors in D_ref
    correction_scale: float = 0.1


class FMS:
    """Fear Modulation Signal — distribution shift detector and corrector."""

    def __init__(self, config: FMSConfig, device: str = "cuda") -> None:
        self.cfg = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # D_ref state embeddings for similarity comparison
        self._d_ref_states: torch.Tensor | None = None
        self._d_ref_costs: torch.Tensor | None = None

        self._delta_history: list[float] = []

    def set_d_ref(self, states: torch.Tensor, costs: torch.Tensor) -> None:
        """Set reference dataset states and costs.

        Args:
            states: D_ref states [N, 12].
            costs: D_ref costs [N].
        """
        self._d_ref_states = states.to(self.device)
        self._d_ref_costs = costs.to(self.device)

    def step(self, obs: torch.Tensor, predicted_cost: float) -> float:
        """Compute FMS correction δ_t.

        Measures how the current state differs from D_ref and
        corrects the fear signal accordingly.

        Args:
            obs: Current observation [12].
            predicted_cost: Cost critic's prediction for current state.

        Returns:
            Correction δ_t (signed — can increase or decrease fear).
        """
        if not self.cfg.enabled or self._d_ref_states is None:
            self._delta_history.append(0.0)
            return 0.0

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        obs = obs.to(self.device)

        # k-nearest neighbors in D_ref
        dists = torch.cdist(obs, self._d_ref_states).squeeze(0)  # [N]
        _, knn_idx = dists.topk(self.cfg.k_neighbors, largest=False)

        # Similarity weights (inverse distance)
        knn_dists = dists[knn_idx]
        weights = 1.0 / (knn_dists + 1e-8)
        weights = weights / weights.sum()

        # Expected cost from neighbors
        neighbor_costs = self._d_ref_costs[knn_idx]
        expected_cost = (weights * neighbor_costs).sum().item()

        # Discrepancy: how different is the critic's prediction from local evidence
        discrepancy = predicted_cost - expected_cost

        # Correction scaled by novelty (mean distance to neighbors)
        novelty = knn_dists.mean().item()
        delta = self.cfg.correction_scale * discrepancy * min(novelty, 1.0)

        self._delta_history.append(delta)
        return delta

    def get_metrics(self) -> dict[str, Any]:
        deltas = self._delta_history
        return {
            "delta_history": deltas.copy(),
            "delta_abs_mean": np.mean([abs(d) for d in deltas]) if deltas else 0.0,
            "delta_abs_max": max([abs(d) for d in deltas]) if deltas else 0.0,
        }

    def reset(self) -> None:
        self._delta_history = []

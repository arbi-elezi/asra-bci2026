"""Independent Fear Detection Model — Separate from the LLM decision maker.

Architecture: Ensemble of complementary detection methods:
  1. Autoencoder reconstruction error (learned distribution model)
  2. Isolation Forest (unsupervised anomaly detection)
  3. Cost-advantage signal (from frozen cost critic)

Scientific justification:
  - The fear model is INDEPENDENT of the LLM — it cannot be gamed by
    the decision maker adapting its own fear signal
  - Multiple detection methods provide robustness
  - Each method captures different failure modes:
    * Autoencoder: distribution shift (OOD states)
    * Isolation Forest: statistical anomalies
    * Cost-advantage: consequence-calibrated threat assessment
  - The ensemble output F_t ∈ [0, 1] is the fear signal that triggers
    weight perturbation in the LLM

This implements Section 3.8 (cost-advantage fear) and extends it with
independent anomaly detection, maintaining the paper's formal structure
since F_t is just a scalar in [0,1] — Proposition 1 doesn't care
how F_t is computed, only that it's bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class FearDetectorConfig:
    """Configuration for the independent fear detection model."""
    enabled: bool = True
    # Autoencoder
    ae_enabled: bool = True
    ae_hidden: int = 32
    ae_latent: int = 8
    ae_lr: float = 1e-3
    ae_epochs: int = 50
    # Isolation Forest
    if_enabled: bool = True
    if_contamination: float = 0.1
    if_n_estimators: int = 100
    # Cost-advantage
    ca_enabled: bool = True
    ca_lambda: float = 5.0
    # Ensemble weights
    weight_ae: float = 0.3
    weight_if: float = 0.2
    weight_ca: float = 0.5
    # Device
    device: str = "cuda"


class StateAutoencoder(nn.Module):
    """Autoencoder for state distribution modeling.

    Trained on D_ref states. High reconstruction error → OOD state → fear.
    """

    def __init__(self, obs_dim: int = 12, hidden: int = 32, latent: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction error (MSE)."""
        x_hat, _ = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=-1)


class FearDetector:
    """Independent fear detection model — ensemble of multiple detectors.

    This model is SEPARATE from the LLM decision maker. It observes
    the same environment state and produces an independent fear signal.

    The fear signal F_t ∈ [0, 1] triggers weight perturbation in the LLM.
    """

    def __init__(self, config: FearDetectorConfig) -> None:
        self.cfg = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # ── Autoencoder ──
        self.autoencoder: StateAutoencoder | None = None
        self._ae_threshold: float = 0.0  # Calibrated from D_ref
        self._ae_mean: float = 0.0
        self._ae_std: float = 1.0

        # ── Isolation Forest ──
        self.isolation_forest: IsolationForest | None = None

        # ── Cost-advantage (from frozen cost critic) ──
        self.cost_critic = None  # Set externally

        # ── Metrics ──
        self._fear_history: list[float] = []
        self._component_history: list[dict[str, float]] = []
        self._trained = False

    def train_on_d_ref(
        self,
        d_ref_states: np.ndarray | torch.Tensor,
        d_ref_costs: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Train the fear detection ensemble on D_ref.

        Must be called once during Phase 0 (infrastructure setup).

        Args:
            d_ref_states: [N, 12] reference states.
            d_ref_costs: [N] associated costs (for calibration).

        Returns:
            Training metrics.
        """
        if isinstance(d_ref_states, torch.Tensor):
            states_np = d_ref_states.cpu().numpy()
            states_t = d_ref_states.float().to(self.device)
        else:
            states_np = d_ref_states
            states_t = torch.tensor(d_ref_states, dtype=torch.float32, device=self.device)

        metrics = {}

        # ── Train Autoencoder ──
        if self.cfg.ae_enabled:
            self.autoencoder = StateAutoencoder(
                obs_dim=states_t.shape[1],
                hidden=self.cfg.ae_hidden,
                latent=self.cfg.ae_latent,
            ).to(self.device)

            optimizer = torch.optim.Adam(
                self.autoencoder.parameters(), lr=self.cfg.ae_lr
            )

            self.autoencoder.train()
            for epoch in range(self.cfg.ae_epochs):
                perm = torch.randperm(len(states_t))
                total_loss = 0.0
                for start in range(0, len(states_t), 64):
                    idx = perm[start:start + 64]
                    batch = states_t[idx]
                    x_hat, _ = self.autoencoder(batch)
                    loss = ((batch - x_hat) ** 2).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            # Calibrate threshold from D_ref reconstruction errors
            self.autoencoder.eval()
            with torch.no_grad():
                errors = self.autoencoder.reconstruction_error(states_t)
                self._ae_mean = errors.mean().item()
                self._ae_std = errors.std().item()
                self._ae_threshold = self._ae_mean + 2 * self._ae_std

            metrics["ae_mean_error"] = self._ae_mean
            metrics["ae_threshold"] = self._ae_threshold

        # ── Train Isolation Forest ──
        if self.cfg.if_enabled:
            self.isolation_forest = IsolationForest(
                n_estimators=self.cfg.if_n_estimators,
                contamination=self.cfg.if_contamination,
                random_state=42,  # Deterministic
            )
            self.isolation_forest.fit(states_np)
            scores = self.isolation_forest.score_samples(states_np)
            metrics["if_mean_score"] = float(scores.mean())

        self._trained = True
        return metrics

    def detect(
        self,
        obs: np.ndarray | torch.Tensor,
        cost: float = 0.0,
        ttc: float = 10.0,
        greedy_action: int = 0,
    ) -> tuple[float, dict[str, float]]:
        """Detect fear level from current state.

        Runs the ensemble and produces F_t ∈ [0, 1].

        Args:
            obs: Current observation R^12.
            cost: Observable cost signal.
            ttc: Time-to-collision.
            greedy_action: Current greedy action from LLM.

        Returns:
            (fear_signal, component_scores)
        """
        if not self.cfg.enabled or not self._trained:
            self._fear_history.append(0.0)
            return 0.0, {}

        if isinstance(obs, np.ndarray):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_t = obs.float().to(self.device)

        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        components = {}
        total_weight = 0.0
        weighted_fear = 0.0

        # ── Autoencoder: OOD detection ──
        if self.cfg.ae_enabled and self.autoencoder is not None:
            with torch.no_grad():
                recon_error = self.autoencoder.reconstruction_error(obs_t).item()

            # Normalize to [0, 1]: sigmoid on z-score
            if self._ae_std > 0:
                z = (recon_error - self._ae_mean) / self._ae_std
            else:
                z = 0.0
            ae_fear = 1.0 / (1.0 + np.exp(-z))
            components["ae_fear"] = ae_fear
            weighted_fear += self.cfg.weight_ae * ae_fear
            total_weight += self.cfg.weight_ae

        # ── Isolation Forest: statistical anomaly ──
        if self.cfg.if_enabled and self.isolation_forest is not None:
            obs_np = obs_t.cpu().numpy()
            score = self.isolation_forest.score_samples(obs_np)[0]
            # Scores are negative (more negative = more anomalous)
            # Map to [0, 1]: anomaly score
            if_fear = 1.0 / (1.0 + np.exp(score * 5))  # Sigmoid mapping
            components["if_fear"] = if_fear
            weighted_fear += self.cfg.weight_if * if_fear
            total_weight += self.cfg.weight_if

        # ── Cost-advantage: consequence-calibrated ──
        if self.cfg.ca_enabled:
            # Direct from cost signal: F_t^CA = σ(cost · λ)
            ca_fear = 1.0 / (1.0 + np.exp(-cost * self.cfg.ca_lambda * 2 + self.cfg.ca_lambda))
            # Also incorporate TTC directly
            ttc_fear = max(0.0, (2.0 - ttc) / 2.0)  # Same as cost formula
            ca_combined = max(ca_fear, ttc_fear)
            components["ca_fear"] = ca_combined
            weighted_fear += self.cfg.weight_ca * ca_combined
            total_weight += self.cfg.weight_ca

        # ── Ensemble ──
        if total_weight > 0:
            fear = weighted_fear / total_weight
        else:
            fear = 0.0

        # Clip to [0, 1] (numerical safety)
        fear = float(np.clip(fear, 0.0, 1.0))

        self._fear_history.append(fear)
        self._component_history.append(components)

        return fear, components

    def get_metrics(self) -> dict[str, Any]:
        return {
            "fear_history": self._fear_history.copy(),
            "fear_mean": np.mean(self._fear_history) if self._fear_history else 0.0,
            "fear_max": max(self._fear_history) if self._fear_history else 0.0,
            "component_history": self._component_history.copy(),
            "trained": self._trained,
        }

    def get_state(self) -> dict:
        """Serialize trained models for saving."""
        state = {
            "trained": self._trained,
            "ae_threshold": self._ae_threshold,
            "ae_mean": self._ae_mean,
            "ae_std": self._ae_std,
        }
        if self.autoencoder is not None:
            state["ae_state_dict"] = self.autoencoder.state_dict()
        if self.isolation_forest is not None:
            import pickle
            state["iforest_bytes"] = pickle.dumps(self.isolation_forest)
        return state

    def load_state(self, state: dict) -> None:
        """Restore trained models from saved state."""
        self._trained = state.get("trained", False)
        self._ae_threshold = state.get("ae_threshold", 0.1)
        self._ae_mean = state.get("ae_mean", 0.0)
        self._ae_std = state.get("ae_std", 1.0)
        if "ae_state_dict" in state and self.autoencoder is not None:
            self.autoencoder.load_state_dict(state["ae_state_dict"])
        if "iforest_bytes" in state:
            import pickle
            self.isolation_forest = pickle.loads(state["iforest_bytes"])

    def reset(self) -> None:
        """Reset per-episode state (keep trained models)."""
        self._fear_history = []
        self._component_history = []

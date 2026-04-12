"""Fear Classifier (FC) — Online threat classification.

Trained on GTCC + RLAF labels. Can be frozen (C4 ablation) or
continuously updated via GTCC calibration loop.

FC is a simple binary classifier: state → {safe, dangerous}.
Fine-tuned online every N_ft steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class FCConfig:
    enabled: bool = True
    frozen: bool = False   # True for C4 (No-GTCC) ablation
    n_ft: int = 50         # Fine-tune every N_ft labeled examples
    lr: float = 1e-3
    hidden_dim: int = 32


class FearClassifier(nn.Module):
    """Binary fear classifier: state → P(dangerous)."""

    def __init__(self, obs_dim: int = 12, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FC:
    """Fear Classifier with online fine-tuning from GTCC labels."""

    def __init__(self, config: FCConfig, device: str = "cuda") -> None:
        self.cfg = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.classifier = FearClassifier(hidden=config.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=config.lr)
        self.criterion = nn.BCELoss()

        # Training buffer
        self._buffer: list[tuple[torch.Tensor, int]] = []
        self._f1_history: list[float] = []
        self._update_count: int = 0

    def predict(self, obs: torch.Tensor) -> float:
        """Predict danger probability for current state."""
        if not self.cfg.enabled:
            return 0.0

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            prob = self.classifier(obs.to(self.device))
        return prob.item()

    def add_label(self, obs: torch.Tensor, label: int) -> None:
        """Add a GTCC/RLAF label to the training buffer."""
        if self.cfg.frozen:
            return  # C4: no updates after seed initialization
        self._buffer.append((obs.detach().clone(), label))

        # Fine-tune every N_ft examples
        if len(self._buffer) >= self.cfg.n_ft:
            self._fine_tune()

    def _fine_tune(self) -> None:
        """Fine-tune on accumulated buffer."""
        if not self._buffer or self.cfg.frozen:
            return

        states = torch.stack([s for s, _ in self._buffer]).to(self.device)
        labels = torch.tensor(
            [l for _, l in self._buffer], dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        # One pass of gradient descent
        self.classifier.train()
        preds = self.classifier(states)
        loss = self.criterion(preds, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute F1 on this batch
        with torch.no_grad():
            pred_labels = (preds > 0.5).float()
            tp = ((pred_labels == 1) & (labels == 1)).sum().item()
            fp = ((pred_labels == 1) & (labels == 0)).sum().item()
            fn = ((pred_labels == 0) & (labels == 1)).sum().item()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            self._f1_history.append(f1)

        self._update_count += 1
        self._buffer = []

    def seed_initialize(self, seed_data: list[tuple[torch.Tensor, int]], n_epochs: int = 10) -> None:
        """Initialize FC from seed dataset D_seed.

        Called once at start. C4 freezes after this.
        """
        if not seed_data:
            return

        states = torch.stack([s for s, _ in seed_data]).to(self.device)
        labels = torch.tensor(
            [l for _, l in seed_data], dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        self.classifier.train()
        for _ in range(n_epochs):
            preds = self.classifier(states)
            loss = self.criterion(preds, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_metrics(self) -> dict[str, Any]:
        return {
            "f1_history": self._f1_history.copy(),
            "update_count": self._update_count,
            "f1_current": self._f1_history[-1] if self._f1_history else 0.0,
            "f1_slope": self._compute_f1_slope(),
        }

    def _compute_f1_slope(self) -> float:
        """Compute slope of F1 over updates (for M2/H2)."""
        if len(self._f1_history) < 2:
            return 0.0
        x = list(range(len(self._f1_history)))
        y = self._f1_history
        n = len(x)
        sx = sum(x)
        sy = sum(y)
        sxy = sum(xi * yi for xi, yi in zip(x, y))
        sxx = sum(xi * xi for xi in x)
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            return 0.0
        return (n * sxy - sx * sy) / denom

    def reset(self) -> None:
        self._buffer = []
        # Don't reset F1 history or update count — they accumulate

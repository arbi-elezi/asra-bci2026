"""TD-Fear Signal — Temporal smoothing of fear.

Paper Section 3.10:
  F_t^TD = (1 - γ_f) · F_t^CA + γ_f · F_{t-1}^TD

Noise reduction; evaluated by H9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TDFearConfig:
    enabled: bool = True
    gamma_f: float = 0.5  # Decay factor [0, 0.9]


class TDFear:
    """TD-Fear temporal smoothing."""

    def __init__(self, config: TDFearConfig) -> None:
        self.cfg = config
        self._f_td_prev: float = 0.0
        self._history: list[float] = []

    def step(self, f_ca: float) -> float:
        """Compute temporally smoothed fear signal.

        Args:
            f_ca: Raw cost-advantage fear F_t^CA ∈ [0, 1].

        Returns:
            Smoothed fear F_t^TD ∈ [0, 1].
        """
        if not self.cfg.enabled:
            self._history.append(f_ca)
            return f_ca

        f_td = (1 - self.cfg.gamma_f) * f_ca + self.cfg.gamma_f * self._f_td_prev
        # Clip to [0, 1] (numerical safety)
        f_td = max(0.0, min(1.0, f_td))
        self._f_td_prev = f_td
        self._history.append(f_td)
        return f_td

    def get_metrics(self) -> dict[str, Any]:
        return {"td_fear_history": self._history.copy()}

    def reset(self) -> None:
        self._f_td_prev = 0.0
        self._history = []

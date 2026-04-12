"""Ground-Truth Consequence Calibration (GTCC) + RLAIF.

Paper Section 3.10:
  - GTCC: Uses directly observable cost signal to generate ground-truth
    labels for the fear classifier (FC)
  - RLAIF: AI judge evaluates inconclusive scenarios with structured reasoning
  - GTCC labels ALWAYS override RLAIF labels
  - M7 gates RLAIF interpretation (required > 0.70 accuracy)

Definition 3(b) requires: scalar cost signal c_t ∈ [0,1] observable each timestep.

RLAIF integration:
  The AI judge (rlaif_judge.py) replaces the simple TTC heuristic for
  inconclusive cases (cost between safe and danger thresholds). The judge
  uses multi-criteria structured reasoning over kinematic safety, obstacle
  behavior, ego options, and temporal context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import numpy as np

from src.components.rlaif_judge import RLAIFJudge, RLAIFJudgeConfig


@dataclass
class GTCCConfig:
    enabled: bool = True
    # Cost thresholds for ground-truth labeling
    cost_threshold_danger: float = 0.3   # Above this → "dangerous"
    cost_threshold_safe: float = 0.05    # Below this → "safe"
    # RLAIF settings
    rlaf_enabled: bool = True
    rlaf_gate_threshold: float = 0.70    # M7 gate
    rlaif_judge: RLAIFJudgeConfig = None  # AI judge config

    def __post_init__(self):
        if self.rlaif_judge is None:
            self.rlaif_judge = RLAIFJudgeConfig()


class GTCC:
    """Ground-Truth Consequence Calibration with RLAIF Judge.

    Three-tier labeling hierarchy:
      1. GTCC (highest priority): cost >= 0.3 → dangerous, cost <= 0.05 → safe
      2. RLAIF Judge (inconclusive): structured multi-criteria AI evaluation
      3. Unlabeled: when RLAIF confidence too low or M7 gate fails

    GTCC labels ALWAYS override RLAIF. M7 gates RLAIF usage.
    """

    def __init__(self, config: GTCCConfig) -> None:
        self.cfg = config

        # RLAIF Judge — replaces simple heuristic
        self.judge = RLAIFJudge(config.rlaif_judge)

        # Label buffers
        self._gtcc_labels: list[tuple[torch.Tensor, int]] = []  # (state, label)
        self._rlaf_labels: list[tuple[torch.Tensor, int]] = []  # RLAIF judge labels
        self._rlaf_reasonings: list[str] = []  # Judge reasoning chains

    def step(
        self,
        obs: torch.Tensor,
        cost: float,
        ttc: float,
        obstacle_class: int = -1,
        ego_speed: float = 25.0,
        closing_speed: float = 0.0,
        gap: float = 100.0,
    ) -> dict[str, Any]:
        """Generate ground-truth label from observed cost, with RLAIF fallback.

        Args:
            obs: Current observation.
            cost: Observed cost signal c_t ∈ [0, 1].
            ttc: Time-to-collision.
            obstacle_class: Nearest obstacle class (for RLAIF judge).
            ego_speed: Ego speed m/s (for RLAIF judge).
            closing_speed: Closing speed m/s (for RLAIF judge).
            gap: Gap to nearest obstacle m (for RLAIF judge).

        Returns:
            Dict with label, source, confidence, and reasoning.
        """
        if not self.cfg.enabled:
            return {"label": None, "source": "disabled"}

        # ── Tier 1: GTCC — direct from cost signal ──
        if cost >= self.cfg.cost_threshold_danger:
            label = 1  # Dangerous
            source = "gtcc"
            confidence = 1.0
            reasoning = f"GTCC: cost={cost:.3f} >= {self.cfg.cost_threshold_danger} → DANGEROUS"
            self._gtcc_labels.append((obs.detach().clone(), label))

            # Also validate RLAIF judge against this ground truth
            if self.cfg.rlaf_enabled:
                self.judge.validate_against_gtcc(
                    obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs,
                    cost, ttc, obstacle_class, ego_speed, closing_speed, gap,
                )

            return {"label": label, "source": source, "confidence": confidence,
                    "reasoning": reasoning, "cost": cost}

        elif cost <= self.cfg.cost_threshold_safe:
            label = 0  # Safe
            source = "gtcc"
            confidence = 1.0
            reasoning = f"GTCC: cost={cost:.3f} <= {self.cfg.cost_threshold_safe} → SAFE"
            self._gtcc_labels.append((obs.detach().clone(), label))

            # Validate RLAIF
            if self.cfg.rlaf_enabled:
                self.judge.validate_against_gtcc(
                    obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs,
                    cost, ttc, obstacle_class, ego_speed, closing_speed, gap,
                )

            return {"label": label, "source": source, "confidence": confidence,
                    "reasoning": reasoning, "cost": cost}

        # ── Tier 2: RLAIF Judge — inconclusive cases ──
        if self.cfg.rlaf_enabled and self.judge.get_m7_accuracy() >= self.cfg.rlaf_gate_threshold:
            obs_np = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs

            judgment = self.judge.evaluate_scenario(
                obs=obs_np,
                cost=cost,
                ttc=ttc,
                obstacle_class=obstacle_class,
                ego_speed=ego_speed,
                closing_speed=closing_speed,
                gap=gap,
            )

            if judgment.confidence >= self.judge.cfg.confidence_threshold:
                label = judgment.label
                source = "rlaif"
                confidence = judgment.confidence
                reasoning = judgment.reasoning
                self._rlaf_labels.append((obs.detach().clone(), label))
                self._rlaf_reasonings.append(reasoning)

                return {"label": label, "source": source, "confidence": confidence,
                        "reasoning": reasoning, "cost": cost}

        # ── Tier 3: Unlabeled ──
        m7 = self.judge.get_m7_accuracy()
        reason = (
            f"Inconclusive: cost={cost:.3f} in [{self.cfg.cost_threshold_safe}, "
            f"{self.cfg.cost_threshold_danger}], M7={m7:.2f}"
        )
        if m7 < self.cfg.rlaf_gate_threshold:
            reason += f" (M7 < {self.cfg.rlaf_gate_threshold} — RLAIF gated)"

        return {"label": None, "source": "inconclusive", "confidence": 0.0,
                "reasoning": reason, "cost": cost}

    def get_training_data(self) -> list[tuple[torch.Tensor, int]]:
        """Get accumulated labels for FC training.

        GTCC labels take priority. RLAIF labels only used if M7 > 0.70.
        """
        data = self._gtcc_labels.copy()
        if self.judge.get_m7_accuracy() >= self.cfg.rlaf_gate_threshold:
            data.extend(self._rlaf_labels)
        return data

    def get_metrics(self) -> dict[str, Any]:
        judge_metrics = self.judge.get_metrics()
        return {
            "gtcc_labels_count": len(self._gtcc_labels),
            "rlaf_labels_count": len(self._rlaf_labels),
            "rlaf_reasonings_sample": self._rlaf_reasonings[-3:] if self._rlaf_reasonings else [],
            "m7_accuracy": judge_metrics["m7_accuracy"],
            "m7_gate_passed": judge_metrics["m7_gate_passed"],
            "judge_metrics": judge_metrics,
        }

    def reset(self) -> None:
        self._gtcc_labels = []
        self._rlaf_labels = []
        self._rlaf_reasonings = []
        self.judge.reset_episode()
        # Don't reset judge's M7 tracking — it accumulates across episodes

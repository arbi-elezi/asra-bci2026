"""RLAIF Judge — AI Feedback for Fear Classifier Training.

Paper Section 3.10:
  - RLAF (Reinforcement Learning from AI Feedback) supplements GTCC
  - AI judge evaluates inconclusive scenarios that GTCC cannot resolve
  - GTCC labels ALWAYS override RLAF labels
  - M7 gates RLAF interpretation (required > 0.70 accuracy)

In this implementation, the AI judge uses structured reasoning about
driving scenarios to produce fear labels. The judge evaluates:
  1. Kinematic safety (TTC, closing speed, gap)
  2. Obstacle behavior (class, trajectory pattern)
  3. Ego options (available actions, escape routes)
  4. Temporal context (threat escalation/de-escalation)

The judge's output is a structured label + confidence + reasoning chain.
This is RLAIF: the AI provides the feedback signal for FC fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class RLAIFJudgeConfig:
    """Configuration for the RLAIF judge."""
    enabled: bool = True
    # Confidence threshold — only use labels with confidence > this
    confidence_threshold: float = 0.6
    # Maximum labels to accumulate before batch evaluation
    batch_size: int = 50
    # Weights for the structured evaluation rubric
    kinematic_weight: float = 0.35
    behavioral_weight: float = 0.20
    options_weight: float = 0.25
    temporal_weight: float = 0.20


@dataclass
class JudgmentResult:
    """Structured output of a single RLAIF judgment."""
    label: int              # 0 = safe, 1 = dangerous
    confidence: float       # [0, 1] — judge's confidence in the label
    reasoning: str          # Natural language reasoning chain
    scores: dict[str, float] = field(default_factory=dict)  # Component scores


class RLAIFJudge:
    """AI Judge for fear scenario evaluation.

    Evaluates driving scenarios using structured multi-criteria reasoning.
    Produces labels + confidence + reasoning for FC training data.

    The judge operates on INCONCLUSIVE scenarios only — situations where
    GTCC cannot determine ground truth from the cost signal alone
    (cost between safe and danger thresholds).

    Scientific role:
      - Supplements GTCC (which uses observable cost signal)
      - Subject to M7 gate (accuracy must exceed 0.70)
      - Labels are NEVER used if M7 < 0.70
      - GTCC always overrides when available
    """

    def __init__(self, config: RLAIFJudgeConfig | None = None) -> None:
        self.cfg = config or RLAIFJudgeConfig()

        # Accumulated judgments
        self._judgments: list[JudgmentResult] = []
        self._scenario_buffer: list[dict] = []

        # Accuracy tracking (for M7)
        self._correct: int = 0
        self._total: int = 0

        # Temporal context window (last N observations)
        self._history_len: int = 20
        self._ttc_history: list[float] = []
        self._cost_history: list[float] = []
        self._class_history: list[int] = []

    def evaluate_scenario(
        self,
        obs: np.ndarray,
        cost: float,
        ttc: float,
        obstacle_class: int,
        ego_speed: float,
        closing_speed: float,
        gap: float,
        available_lanes: int = 3,
    ) -> JudgmentResult:
        """Evaluate a single driving scenario using structured reasoning.

        This is the core judgment function. It applies a multi-criteria
        rubric to produce a label, confidence, and reasoning chain.

        The rubric:
          1. KINEMATIC SAFETY (35%): TTC, gap, closing speed
          2. BEHAVIORAL PATTERN (20%): obstacle class, trajectory trends
          3. EGO OPTIONS (25%): escape routes, braking room
          4. TEMPORAL CONTEXT (20%): threat escalation/de-escalation

        Args:
            obs: R^12 observation vector
            cost: Current cost signal c_t
            ttc: Time-to-collision
            obstacle_class: 0=SLOW, 1=FAST, 2=STATIONARY
            ego_speed: Ego longitudinal speed (m/s)
            closing_speed: Relative closing speed (m/s, positive = approaching)
            gap: Longitudinal gap to nearest obstacle (m)
            available_lanes: Number of lanes (for escape route assessment)

        Returns:
            JudgmentResult with label, confidence, and reasoning.
        """
        if not self.cfg.enabled:
            return JudgmentResult(label=0, confidence=0.0, reasoning="Judge disabled")

        # Update temporal context
        self._ttc_history.append(ttc)
        self._cost_history.append(cost)
        self._class_history.append(obstacle_class)
        if len(self._ttc_history) > self._history_len:
            self._ttc_history = self._ttc_history[-self._history_len:]
            self._cost_history = self._cost_history[-self._history_len:]
            self._class_history = self._class_history[-self._history_len:]

        # ── 1. KINEMATIC SAFETY ASSESSMENT ──
        kinematic_score, kinematic_reasoning = self._assess_kinematics(
            ttc, closing_speed, gap, ego_speed
        )

        # ── 2. BEHAVIORAL PATTERN ASSESSMENT ──
        behavioral_score, behavioral_reasoning = self._assess_behavior(
            obstacle_class, closing_speed
        )

        # ── 3. EGO OPTIONS ASSESSMENT ──
        options_score, options_reasoning = self._assess_options(
            ego_speed, gap, available_lanes, ttc
        )

        # ── 4. TEMPORAL CONTEXT ASSESSMENT ──
        temporal_score, temporal_reasoning = self._assess_temporal()

        # ── AGGREGATE ──
        weights = {
            "kinematic": self.cfg.kinematic_weight,
            "behavioral": self.cfg.behavioral_weight,
            "options": self.cfg.options_weight,
            "temporal": self.cfg.temporal_weight,
        }
        scores = {
            "kinematic": kinematic_score,
            "behavioral": behavioral_score,
            "options": options_score,
            "temporal": temporal_score,
        }

        danger_score = sum(scores[k] * weights[k] for k in scores)

        # Decision threshold
        label = 1 if danger_score > 0.5 else 0

        # Confidence from margin and agreement
        margin = abs(danger_score - 0.5) * 2  # 0 at boundary, 1 at extremes
        agreement = 1.0 - np.std(list(scores.values()))  # Higher when all criteria agree
        confidence = 0.6 * margin + 0.4 * agreement
        confidence = max(0.0, min(1.0, confidence))

        # Build reasoning chain
        label_str = "DANGEROUS" if label == 1 else "SAFE"
        reasoning = (
            f"Judgment: {label_str} (score={danger_score:.2f}, confidence={confidence:.2f})\n"
            f"  Kinematic ({weights['kinematic']:.0%}): {kinematic_score:.2f} — {kinematic_reasoning}\n"
            f"  Behavioral ({weights['behavioral']:.0%}): {behavioral_score:.2f} — {behavioral_reasoning}\n"
            f"  Options ({weights['options']:.0%}): {options_score:.2f} — {options_reasoning}\n"
            f"  Temporal ({weights['temporal']:.0%}): {temporal_score:.2f} — {temporal_reasoning}"
        )

        result = JudgmentResult(
            label=label,
            confidence=confidence,
            reasoning=reasoning,
            scores=scores,
        )
        self._judgments.append(result)

        return result

    def _assess_kinematics(
        self, ttc: float, closing_speed: float, gap: float, ego_speed: float
    ) -> tuple[float, str]:
        """Criterion 1: Kinematic safety.

        Danger increases with:
          - Low TTC (< 4s is concerning, < 2s is critical)
          - High closing speed relative to gap
          - Small gap relative to stopping distance
        """
        # TTC component
        if ttc <= 0:
            ttc_score = 1.0
            ttc_reason = "TTC=0 (collision imminent)"
        elif ttc < 1.5:
            ttc_score = 0.9
            ttc_reason = f"TTC={ttc:.1f}s (critical)"
        elif ttc < 2.0:
            ttc_score = 0.75
            ttc_reason = f"TTC={ttc:.1f}s (dangerous)"
        elif ttc < 4.0:
            ttc_score = 0.5
            ttc_reason = f"TTC={ttc:.1f}s (cautionary)"
        elif ttc < 6.0:
            ttc_score = 0.25
            ttc_reason = f"TTC={ttc:.1f}s (moderate)"
        else:
            ttc_score = 0.05
            ttc_reason = f"TTC={ttc:.1f}s (safe)"

        # Stopping distance check
        # Assuming 5 m/s^2 braking deceleration
        stop_dist = (ego_speed ** 2) / (2 * 5.0) if ego_speed > 0 else 0
        gap_ratio = gap / max(stop_dist, 1.0)
        gap_score = max(0, 1.0 - gap_ratio) if gap_ratio < 2.0 else 0.0

        # Closing speed urgency
        speed_score = min(1.0, closing_speed / 15.0) if closing_speed > 0 else 0.0

        score = 0.5 * ttc_score + 0.3 * gap_score + 0.2 * speed_score
        reason = f"{ttc_reason}, gap_ratio={gap_ratio:.1f}, closing={closing_speed:.1f}m/s"

        return score, reason

    def _assess_behavior(
        self, obstacle_class: int, closing_speed: float
    ) -> tuple[float, str]:
        """Criterion 2: Obstacle behavioral pattern.

        Stationary obstacles are more dangerous (no evasion from their side).
        Fast obstacles closing quickly are dangerous.
        Slow obstacles with moderate closing speed are moderately dangerous.
        """
        class_names = {0: "SLOW", 1: "FAST", 2: "STATIONARY", -1: "NONE"}

        if obstacle_class == 2:  # STATIONARY
            score = 0.7  # High baseline danger (can't evade)
            reason = f"{class_names[obstacle_class]}: no self-evasion capacity"
        elif obstacle_class == 1:  # FAST
            if closing_speed > 10:
                score = 0.8
                reason = f"{class_names[obstacle_class]}: high closing speed ({closing_speed:.0f}m/s)"
            else:
                score = 0.3
                reason = f"{class_names[obstacle_class]}: moderate closing ({closing_speed:.0f}m/s)"
        elif obstacle_class == 0:  # SLOW
            score = 0.4 if closing_speed > 5 else 0.15
            reason = f"{class_names[obstacle_class]}: closing {closing_speed:.0f}m/s"
        else:
            score = 0.1
            reason = "No obstacle detected"

        return score, reason

    def _assess_options(
        self, ego_speed: float, gap: float, available_lanes: int, ttc: float
    ) -> tuple[float, str]:
        """Criterion 3: Ego vehicle options assessment.

        More options = lower danger. Checks:
          - Can brake in time?
          - Can change lane?
          - Is there deceleration room?
        """
        # Braking feasibility
        brake_dist = (ego_speed ** 2) / (2 * 5.0) if ego_speed > 0 else 0
        can_brake = gap > brake_dist * 1.2  # 20% safety margin

        # Lane change feasibility (simplified — assumes adjacent lanes available)
        can_lane_change = available_lanes > 1 and ttc > 1.5  # Need time to change

        # Speed reduction room
        can_decelerate = ego_speed > 15.0  # Can slow down meaningfully

        n_options = int(can_brake) + int(can_lane_change) + int(can_decelerate)

        if n_options == 0:
            score = 0.95
            reason = "No viable options (cannot brake, change lane, or slow)"
        elif n_options == 1:
            score = 0.65
            reason = f"Single option: {'brake' if can_brake else 'lane_change' if can_lane_change else 'decelerate'}"
        elif n_options == 2:
            score = 0.35
            reason = f"Two options available"
        else:
            score = 0.1
            reason = "All options available (brake, lane change, decelerate)"

        return score, reason

    def _assess_temporal(self) -> tuple[float, str]:
        """Criterion 4: Temporal context — is the situation escalating?

        Looks at trends in TTC and cost over recent history.
        Decreasing TTC = escalating danger.
        """
        if len(self._ttc_history) < 3:
            return 0.3, "Insufficient history"

        recent_ttc = self._ttc_history[-5:]
        recent_cost = self._cost_history[-5:]

        # TTC trend (linear regression slope)
        if len(recent_ttc) >= 3:
            x = np.arange(len(recent_ttc))
            ttc_slope = np.polyfit(x, recent_ttc, 1)[0]
        else:
            ttc_slope = 0.0

        # Cost trend
        if len(recent_cost) >= 3:
            x = np.arange(len(recent_cost))
            cost_slope = np.polyfit(x, recent_cost, 1)[0]
        else:
            cost_slope = 0.0

        if ttc_slope < -0.5 and cost_slope > 0.05:
            score = 0.85
            reason = f"ESCALATING: TTC dropping ({ttc_slope:.2f}s/step), cost rising ({cost_slope:.3f}/step)"
        elif ttc_slope < -0.2:
            score = 0.6
            reason = f"Moderately escalating: TTC slope={ttc_slope:.2f}s/step"
        elif ttc_slope > 0.3:
            score = 0.15
            reason = f"DE-ESCALATING: TTC rising ({ttc_slope:.2f}s/step)"
        else:
            score = 0.35
            reason = f"Stable: TTC slope={ttc_slope:.2f}s/step"

        return score, reason

    def validate_against_gtcc(
        self, obs: np.ndarray, cost: float, ttc: float,
        obstacle_class: int, ego_speed: float, closing_speed: float,
        gap: float,
    ) -> bool:
        """Validate a judgment against GTCC ground truth.

        Used to compute M7 (RLAF reliability).
        Returns True if the judge's label matches GTCC.
        """
        judgment = self.evaluate_scenario(
            obs, cost, ttc, obstacle_class, ego_speed, closing_speed, gap
        )

        # GTCC ground truth
        if cost >= 0.3:
            gtcc_label = 1
        elif cost <= 0.05:
            gtcc_label = 0
        else:
            return True  # Can't validate inconclusive cases

        self._total += 1
        if judgment.label == gtcc_label:
            self._correct += 1
            return True
        return False

    def get_reliable_labels(
        self, min_confidence: float | None = None
    ) -> list[tuple[JudgmentResult, bool]]:
        """Get labels that pass confidence threshold.

        Returns list of (judgment, is_reliable) pairs.
        """
        threshold = min_confidence or self.cfg.confidence_threshold
        return [
            (j, j.confidence >= threshold)
            for j in self._judgments
        ]

    def get_m7_accuracy(self) -> float:
        """M7: RLAF accuracy vs GTCC ground truth."""
        if self._total == 0:
            return 0.0
        return self._correct / self._total

    def get_metrics(self) -> dict[str, Any]:
        """Get judge performance metrics."""
        confidences = [j.confidence for j in self._judgments]
        return {
            "total_judgments": len(self._judgments),
            "m7_accuracy": self.get_m7_accuracy(),
            "m7_gate_passed": self.get_m7_accuracy() >= 0.70,
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "high_confidence_fraction": float(
                np.mean([c >= self.cfg.confidence_threshold for c in confidences])
            ) if confidences else 0.0,
            "label_distribution": {
                "dangerous": sum(1 for j in self._judgments if j.label == 1),
                "safe": sum(1 for j in self._judgments if j.label == 0),
            },
            "validation_total": self._total,
            "validation_correct": self._correct,
        }

    def reset_episode(self) -> None:
        """Reset per-episode state (keep cumulative M7 tracking)."""
        self._ttc_history = []
        self._cost_history = []
        self._class_history = []
        # Don't reset _judgments or M7 counters — they accumulate

    def reset_all(self) -> None:
        """Full reset including M7 tracking."""
        self._judgments = []
        self._scenario_buffer = []
        self._correct = 0
        self._total = 0
        self.reset_episode()

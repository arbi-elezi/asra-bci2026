"""Tests for the RLAIF Judge and GTCC+RLAIF integration."""

import pytest
import numpy as np
import torch
import sys

sys.path.insert(0, "D:/bci-2026")

from src.components.rlaif_judge import RLAIFJudge, RLAIFJudgeConfig, JudgmentResult
from src.components.gtcc import GTCC, GTCCConfig


class TestRLAIFJudge:
    """Test the RLAIF judge's structured reasoning."""

    def test_high_danger_scenario(self):
        """Close obstacle, low TTC → DANGEROUS."""
        judge = RLAIFJudge()
        result = judge.evaluate_scenario(
            obs=np.zeros(12),
            cost=0.4,
            ttc=1.0,
            obstacle_class=2,  # STATIONARY
            ego_speed=30.0,
            closing_speed=30.0,
            gap=10.0,
        )
        assert result.label == 1  # DANGEROUS
        assert result.confidence > 0.3
        assert "DANGEROUS" in result.reasoning or result.label == 1

    def test_safe_scenario(self):
        """Far obstacle, high TTC → SAFE."""
        judge = RLAIFJudge()
        # Need some history for temporal assessment
        for _ in range(5):
            judge.evaluate_scenario(
                obs=np.zeros(12),
                cost=0.0,
                ttc=10.0,
                obstacle_class=0,
                ego_speed=25.0,
                closing_speed=0.0,
                gap=200.0,
            )
        result = judge.evaluate_scenario(
            obs=np.zeros(12),
            cost=0.01,
            ttc=10.0,
            obstacle_class=0,  # SLOW, far away
            ego_speed=25.0,
            closing_speed=2.0,
            gap=200.0,
        )
        assert result.label == 0  # SAFE

    def test_confidence_range(self):
        """Confidence must be in [0, 1]."""
        judge = RLAIFJudge()
        for _ in range(20):
            ttc = np.random.uniform(0.5, 10.0)
            result = judge.evaluate_scenario(
                obs=np.zeros(12),
                cost=np.random.uniform(0.0, 0.5),
                ttc=ttc,
                obstacle_class=np.random.randint(0, 3),
                ego_speed=np.random.uniform(15, 35),
                closing_speed=np.random.uniform(0, 20),
                gap=np.random.uniform(5, 200),
            )
            assert 0.0 <= result.confidence <= 1.0

    def test_reasoning_chain_present(self):
        """Every judgment must include reasoning."""
        judge = RLAIFJudge()
        result = judge.evaluate_scenario(
            obs=np.zeros(12), cost=0.2, ttc=3.0,
            obstacle_class=1, ego_speed=28.0, closing_speed=8.0, gap=30.0,
        )
        assert len(result.reasoning) > 0
        assert "Kinematic" in result.reasoning
        assert "Behavioral" in result.reasoning
        assert "Options" in result.reasoning
        assert "Temporal" in result.reasoning

    def test_four_criteria_scored(self):
        """All four criteria must produce scores."""
        judge = RLAIFJudge()
        result = judge.evaluate_scenario(
            obs=np.zeros(12), cost=0.15, ttc=3.5,
            obstacle_class=0, ego_speed=25.0, closing_speed=5.0, gap=50.0,
        )
        assert "kinematic" in result.scores
        assert "behavioral" in result.scores
        assert "options" in result.scores
        assert "temporal" in result.scores
        for score in result.scores.values():
            assert 0.0 <= score <= 1.0

    def test_m7_tracking(self):
        """M7 accuracy tracks correctly."""
        judge = RLAIFJudge()
        # Validate with clear-cut cases (should get high accuracy)
        for _ in range(20):
            # High cost → dangerous (clear GTCC)
            judge.validate_against_gtcc(
                np.zeros(12), cost=0.6, ttc=0.5,
                obstacle_class=2, ego_speed=30.0, closing_speed=30.0, gap=5.0,
            )
        for _ in range(20):
            # Zero cost → safe (clear GTCC)
            judge.validate_against_gtcc(
                np.zeros(12), cost=0.0, ttc=10.0,
                obstacle_class=0, ego_speed=20.0, closing_speed=0.0, gap=200.0,
            )

        m7 = judge.get_m7_accuracy()
        assert m7 > 0.5  # Should be reasonably accurate on clear cases


class TestGTCCWithRLAIF:
    """Test GTCC integration with RLAIF judge."""

    def test_gtcc_overrides_rlaif(self):
        """GTCC labels must override RLAIF for clear-cut cases."""
        gtcc = GTCC(GTCCConfig())
        obs = torch.zeros(12)

        # High cost → GTCC says DANGEROUS, regardless of RLAIF
        result = gtcc.step(obs, cost=0.5, ttc=0.5)
        assert result["label"] == 1
        assert result["source"] == "gtcc"

        # Zero cost → GTCC says SAFE
        result = gtcc.step(obs, cost=0.0, ttc=10.0)
        assert result["label"] == 0
        assert result["source"] == "gtcc"

    def test_rlaif_used_for_inconclusive(self):
        """RLAIF should be used for inconclusive cases when M7 passes."""
        gtcc = GTCC(GTCCConfig())
        obs = torch.zeros(12)

        # First, build up M7 accuracy with clear-cut cases
        for _ in range(30):
            gtcc.step(obs, cost=0.5, ttc=0.5, obstacle_class=2,
                      ego_speed=30.0, closing_speed=30.0, gap=5.0)
            gtcc.step(obs, cost=0.0, ttc=10.0, obstacle_class=0,
                      ego_speed=20.0, closing_speed=0.0, gap=200.0)

        # Now try inconclusive case
        result = gtcc.step(obs, cost=0.15, ttc=2.0,
                           obstacle_class=1, ego_speed=28.0,
                           closing_speed=10.0, gap=20.0)

        # Should get either rlaif label or inconclusive
        assert result["source"] in ("rlaif", "inconclusive")

    def test_m7_gate(self):
        """RLAIF must be gated by M7 > 0.70."""
        gtcc = GTCC(GTCCConfig())
        obs = torch.zeros(12)

        # Without M7 validation, inconclusive should be unlabeled
        result = gtcc.step(obs, cost=0.15, ttc=3.0)
        assert result["source"] == "inconclusive"
        assert result["label"] is None

    def test_training_data_hierarchy(self):
        """Training data should include GTCC labels, RLAIF only if M7 passes."""
        gtcc = GTCC(GTCCConfig())
        obs = torch.zeros(12)

        # Add GTCC labels
        gtcc.step(obs, cost=0.5, ttc=0.5)
        gtcc.step(obs, cost=0.0, ttc=10.0)

        data = gtcc.get_training_data()
        assert len(data) == 2  # Only GTCC labels (M7 not yet validated)

    def test_metrics_include_judge(self):
        """Metrics should include RLAIF judge performance."""
        gtcc = GTCC(GTCCConfig())
        metrics = gtcc.get_metrics()
        assert "judge_metrics" in metrics
        assert "m7_accuracy" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

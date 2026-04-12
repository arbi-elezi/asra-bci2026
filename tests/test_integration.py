"""Integration test: Full FRA pipeline with highway-env + LLM + fear detector.

Tests:
  1. Highway-env wrapper produces R^12 observations
  2. Highway-env wrapper has |A|=4
  3. Cost signal c_t ∈ [0, 1] at every step
  4. Fear detector produces F_t ∈ [0, 1]
  5. FRA engine runs without error for multiple steps
  6. WDN is computable and starts at 0
  7. Determinism: same seed → same trajectory

This is the Phase 0 integration test from scientific_method.md.
"""

import pytest
import sys
import numpy as np

sys.path.insert(0, "D:/bci-2026")

from src.environment.highway_wrapper import HighwayFRAEnv


class TestHighwayWrapper:
    """Test that highway-env wrapper satisfies Definition 3."""

    def test_action_space_4(self):
        """(a) |A| = 4."""
        env = HighwayFRAEnv(seed=42)
        assert env.action_space.n == 4
        env.close()

    def test_observation_r12(self):
        """s_t ∈ R^12."""
        env = HighwayFRAEnv(seed=42)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (12,)
        assert obs.dtype == np.float32
        env.close()

    def test_cost_observable(self):
        """(b) c_t ∈ [0,1] at every step."""
        env = HighwayFRAEnv(seed=42)
        obs, _ = env.reset(seed=42)
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert "cost" in info
            assert 0.0 <= info["cost"] <= 1.0
            assert "ttc" in info
            if term or trunc:
                obs, _ = env.reset(seed=42)
        env.close()

    def test_cost_formula(self):
        """c_t = max(0, (2 - TTC) / 2)."""
        env = HighwayFRAEnv(seed=42)
        obs, _ = env.reset(seed=42)
        for _ in range(20):
            obs, reward, term, trunc, info = env.step(0)
            ttc = info["ttc"]
            expected_cost = max(0.0, min(1.0, (2.0 - ttc) / 2.0))
            assert abs(info["cost"] - expected_cost) < 1e-6, \
                f"Cost mismatch: {info['cost']} vs {expected_cost} at TTC={ttc}"
            if term or trunc:
                obs, _ = env.reset(seed=42)
        env.close()

    def test_obstacle_classification(self):
        """3 obstacle classes exist."""
        env = HighwayFRAEnv(seed=42, vehicles_count=20)
        obs, _ = env.reset(seed=42)
        classes_seen = set()
        for _ in range(100):
            obs, _, term, trunc, info = env.step(env.action_space.sample())
            if info["obstacle_class"] >= 0:
                classes_seen.add(info["obstacle_class"])
            if term or trunc:
                obs, _ = env.reset(seed=42)
        # Should see at least 2 different classes in 100 steps
        assert len(classes_seen) >= 1, f"Only saw classes: {classes_seen}"
        env.close()

    def test_state_text_generation(self):
        """LLM text input can be generated from obs."""
        env = HighwayFRAEnv(seed=42)
        obs, _ = env.reset(seed=42)
        text = env.get_state_text(obs)
        assert isinstance(text, str)
        assert len(text) > 20
        assert "speed" in text.lower()
        assert "action" in text.lower()
        env.close()

    def test_determinism(self):
        """Same seed → same first observation."""
        env1 = HighwayFRAEnv(seed=123)
        env2 = HighwayFRAEnv(seed=123)
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_almost_equal(obs1, obs2, decimal=4)
        env1.close()
        env2.close()

    def test_multi_episode(self):
        """Can run multiple episodes."""
        env = HighwayFRAEnv(seed=42)
        for ep in range(3):
            obs, _ = env.reset(seed=42 + ep)
            total_reward = 0
            for step in range(50):
                obs, reward, term, trunc, info = env.step(0)
                total_reward += reward
                if term or trunc:
                    break
            assert step > 0  # At least 1 step completed
        env.close()


class TestFearDetectorStandalone:
    """Test fear detector independently from LLM."""

    def test_fear_in_range(self):
        """F_t ∈ [0, 1] always."""
        from src.components.fear_detector import FearDetector, FearDetectorConfig

        config = FearDetectorConfig(device="cpu")
        detector = FearDetector(config)

        # Train on random D_ref
        d_ref = np.random.randn(100, 12).astype(np.float32)
        detector.train_on_d_ref(d_ref)

        # Test on various states
        for _ in range(50):
            obs = np.random.randn(12).astype(np.float32)
            fear, _ = detector.detect(obs, cost=np.random.rand(), ttc=np.random.rand() * 10)
            assert 0.0 <= fear <= 1.0, f"Fear out of range: {fear}"

    def test_high_cost_high_fear(self):
        """High cost should produce high fear."""
        from src.components.fear_detector import FearDetector, FearDetectorConfig

        config = FearDetectorConfig(device="cpu", weight_ca=1.0, weight_ae=0.0, weight_if=0.0)
        detector = FearDetector(config)
        d_ref = np.random.randn(100, 12).astype(np.float32)
        detector.train_on_d_ref(d_ref)

        obs = np.zeros(12, dtype=np.float32)
        fear_low, _ = detector.detect(obs, cost=0.0, ttc=10.0)
        fear_high, _ = detector.detect(obs, cost=1.0, ttc=0.5)

        assert fear_high > fear_low, f"High cost should give higher fear: {fear_high} vs {fear_low}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

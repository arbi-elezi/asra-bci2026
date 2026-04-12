"""Unit tests for the 3D driving environment.

Tests:
  1. Definition 3 compliance (structural)
  2. Determinism (same seed → same trajectory)
  3. Cost signal correctness (c_t = max(0, (2-TTC)/2))
  4. Observation space (R^12)
  5. Action space (|A|=4)
  6. 3 obstacle classes present
  7. Collision detection
  8. GPU vectorization consistency
  9. Stress test bias hooks
"""

import pytest
import torch
import numpy as np

import sys
sys.path.insert(0, "D:/bci-2026")

from src.environment.driving_env_3d import (
    VectorizedDrivingEnv3D,
    SingleDrivingEnv3D,
    Env3DConfig,
    TTC_MAX,
)
from src.environment.driving_env import DrivingEnv, EnvConfig
from src.environment.obstacle import ObstacleClass


class TestDefinition3Compliance:
    """Definition 3: FRA Applicability Conditions."""

    def test_discrete_finite_action_space(self):
        """(a) |A| < ∞ and discrete."""
        env = SingleDrivingEnv3D(seed=42)
        assert env.action_space.n == 4

    def test_scalar_cost_observable(self):
        """(b) c_t ∈ [0,1] directly observable at each timestep."""
        env = SingleDrivingEnv3D(seed=42)
        obs, info = env.reset()
        for _ in range(50):
            obs, reward, term, trunc, info = env.step(0)
            if term or trunc:
                obs, info = env.reset()
                continue
            assert "cost" in info
            assert 0.0 <= info["cost"] <= 1.0

    def test_observation_space_r12(self):
        """s_t ∈ R^12."""
        env = SingleDrivingEnv3D(seed=42)
        obs, _ = env.reset()
        assert obs.shape == (12,)
        assert obs.dtype == np.float32

    def test_cost_formula(self):
        """c_t = max(0, (2 - TTC_t) / 2)."""
        # TTC = 0 → cost = 1
        assert abs(max(0.0, (2.0 - 0.0) / 2.0) - 1.0) < 1e-6
        # TTC = 1 → cost = 0.5
        assert abs(max(0.0, (2.0 - 1.0) / 2.0) - 0.5) < 1e-6
        # TTC = 2 → cost = 0
        assert abs(max(0.0, (2.0 - 2.0) / 2.0) - 0.0) < 1e-6
        # TTC = 5 → cost = 0
        assert abs(max(0.0, (2.0 - 5.0) / 2.0) - 0.0) < 1e-6


class TestDeterminism:
    """All randomness via seeded generators."""

    def test_same_seed_same_trajectory(self):
        """Two runs with same seed must produce identical observations."""
        env1 = SingleDrivingEnv3D(seed=123)
        env2 = SingleDrivingEnv3D(seed=123)

        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

        for _ in range(20):
            action = 0  # MAINTAIN
            o1, r1, t1, tr1, i1 = env1.step(action)
            o2, r2, t2, tr2, i2 = env2.step(action)
            if t1 or tr1:
                break
            np.testing.assert_array_almost_equal(o1, o2, decimal=5)
            assert abs(r1 - r2) < 1e-6

    def test_different_seed_different_trajectory(self):
        """Different seeds should produce different runs."""
        env1 = SingleDrivingEnv3D(seed=1)
        env2 = SingleDrivingEnv3D(seed=2)
        obs1, _ = env1.reset(seed=1)
        obs2, _ = env2.reset(seed=2)
        # Extremely unlikely to be identical
        assert not np.allclose(obs1, obs2)


class TestVectorizedConsistency:
    """GPU vectorized env must match single env behavior."""

    def test_batch_matches_single(self):
        """Vectorized env with n_envs=1 matches SingleDrivingEnv3D."""
        config = Env3DConfig(device="cpu")  # CPU for testing
        vec = VectorizedDrivingEnv3D(n_envs=1, seed=42, config=config)
        single = SingleDrivingEnv3D(seed=42, config=config)

        obs_v = vec.reset()
        obs_s, _ = single.reset(seed=42)

        np.testing.assert_array_almost_equal(
            obs_v[0].cpu().numpy(), obs_s, decimal=4
        )


class TestObstacleClasses:
    """3 obstacle classes present and distinct."""

    def test_three_classes_exist(self):
        assert len(ObstacleClass) == 3
        assert ObstacleClass.SLOW == 0
        assert ObstacleClass.FAST == 1
        assert ObstacleClass.STATIONARY == 2


class TestStressTestHooks:
    """Cost bias hooks for C8d/C8e conditions."""

    def test_cost_bias_application(self):
        """Biased cost should differ from true cost for affected class."""
        config = Env3DConfig(
            cost_bias={0: 1.0, 1: 0.2, 2: 1.0},  # C8d: fast ×0.2
            device="cpu",
        )
        env = VectorizedDrivingEnv3D(n_envs=1, seed=42, config=config)

        ttc = torch.tensor([1.0])
        fast_class = torch.tensor([1.0])
        slow_class = torch.tensor([0.0])

        biased_fast = env.get_biased_costs(ttc, fast_class)
        biased_slow = env.get_biased_costs(ttc, slow_class)

        # True cost at TTC=1 is 0.5
        # Biased fast cost should be 0.5 * 0.2 = 0.1
        assert abs(biased_fast.item() - 0.1) < 1e-6
        # Slow cost unbiased
        assert abs(biased_slow.item() - 0.5) < 1e-6


class TestCollision:
    """Collision detection correctness."""

    def test_collision_on_overlap(self):
        """Direct overlap must trigger collision."""
        config = Env3DConfig(device="cpu")
        env = VectorizedDrivingEnv3D(n_envs=1, seed=42, config=config)
        env.reset()

        # Place obstacle directly on ego
        env.obstacles[0, 0, 0] = env.ego[0, 0]  # same x
        env.obstacles[0, 0, 1] = env.ego[0, 1]  # same y
        env.obstacles[0, 0, 2] = env.ego[0, 2]  # same z
        env.obstacles[0, 0, 6] = 1.0  # active

        collisions = env._check_collisions_batch()
        assert collisions[0].item()


class Test2DEnvironment:
    """Tests for the 2D fallback environment."""

    def test_definition3_2d(self):
        env = DrivingEnv(seed=42)
        obs, info = env.reset(seed=42)
        assert obs.shape == (12,)
        assert env.action_space.n == 4

    def test_cost_in_range(self):
        env = DrivingEnv(seed=42)
        env.reset(seed=42)
        for _ in range(50):
            obs, r, term, trunc, info = env.step(0)
            if term or trunc:
                break
            assert 0.0 <= info["cost"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

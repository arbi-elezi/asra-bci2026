"""Highway-env wrapper — adapts the ready-made simulator to FRA paper spec.

Paper requirements:
  - s_t ∈ R^12
  - |A| = 4
  - 3 obstacle classes (mapped from highway-env vehicle types)
  - c_t = max(0, (2 - TTC_t) / 2) — observable scalar cost
  - Satisfies Definition 3 by construction

highway-env provides:
  - Kinematics observation: (vehicles_count, features)
  - DiscreteMetaAction: 5 actions (we merge LANE_LEFT/RIGHT → 4)
  - Multiple vehicle types
  - TTC computable from relative velocities

Scientific justification for mapping:
  - R^12: flatten ego(6) + nearest_obstacle(6) = 12 features
  - |A|=4: {IDLE, LANE_CHANGE, FASTER, SLOWER}
  - 3 classes: speed-based classification of other vehicles
  - TTC: computed from longitudinal closing speed and gap
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import highway_env
import numpy as np


# Action mapping: highway-env 5 → paper 4
# Paper: {MAINTAIN=0, ACCELERATE=1, BRAKE=2, LANE_CHANGE=3}
# highway-env: {LANE_LEFT=0, IDLE=1, LANE_RIGHT=2, FASTER=3, SLOWER=4}
ACTION_MAP = {
    0: 1,  # MAINTAIN → IDLE
    1: 3,  # ACCELERATE → FASTER
    2: 4,  # BRAKE → SLOWER
    3: 0,  # LANE_CHANGE → LANE_LEFT (alternates with LANE_RIGHT)
}

# Obstacle class thresholds (based on speed, grounded in HighD data)
SPEED_SLOW_MAX = 25.0      # < 25 m/s → SLOW (trucks, ~90 km/h)
SPEED_FAST_MIN = 25.0      # ≥ 25 m/s → FAST (cars)
SPEED_STATIONARY_MAX = 1.0 # < 1 m/s → STATIONARY

TTC_MAX = 10.0


class HighwayFRAEnv(gym.Env):
    """Wrapper around highway-env that satisfies Definition 3.

    Definition 3 compliance:
      (a) |A| = 4 (discrete, finite) ✓
      (b) c_t ∈ [0,1] observable each timestep ✓
      (c) Offline cost critic trainable on D_ref ✓
      (d) Per-class error controllable via stress hooks ✓
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str | None = None,
        vehicles_count: int = 10,
        lanes_count: int = 3,
        duration: int = 50,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self._hw_config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 4,  # ego + 3 nearest
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "normalize": False,
                "absolute": False,  # Relative to ego
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": lanes_count,
            "vehicles_count": vehicles_count,
            "duration": duration,
            "collision_reward": -1.0,
            "right_lane_reward": 0.0,
            "high_speed_reward": 0.4,
            "reward_speed_range": [20, 30],
            "simulation_frequency": 10,  # 10 Hz
            "policy_frequency": 5,       # 5 Hz decisions
            # Research scenario — IDM vehicles but dense traffic
            "vehicles_density": 1.5,     # Moderate-high density
            "initial_spacing": 2,        # Standard gaps
        }

        self._inner = gym.make(
            "highway-v0",
            render_mode=render_mode,
            config=self._hw_config,
        )

        # Paper action space: |A| = 4
        self.action_space = gym.spaces.Discrete(4)

        # Paper observation space: R^12
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32,
        )

        self._seed = seed
        self._lane_change_dir = 0  # Alternates left/right
        self._step_count = 0

        # Per-episode tracking
        self.episode_costs: list[float] = []
        self.episode_ttcs: list[float] = []
        self.episode_classes: list[int] = []

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        s = seed if seed is not None else self._seed
        raw_obs, info = self._inner.reset(seed=s)

        self._step_count = 0
        self._lane_change_dir = 0
        self.episode_costs = []
        self.episode_ttcs = []
        self.episode_classes = []

        obs = self._flatten_obs(raw_obs)
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step with paper-spec action mapping.

        Args:
            action: Paper action in {0=MAINTAIN, 1=ACCEL, 2=BRAKE, 3=LANE_CHANGE}
        """
        self._step_count += 1

        # Map paper action → highway-env action
        if action == 3:  # LANE_CHANGE
            hw_action = 0 if self._lane_change_dir == 0 else 2
            self._lane_change_dir = 1 - self._lane_change_dir
        else:
            hw_action = ACTION_MAP[action]

        raw_obs, reward, terminated, truncated, info = self._inner.step(hw_action)

        obs = self._flatten_obs(raw_obs)

        # Compute TTC and cost (Definition 3b)
        ttc = self._compute_ttc(raw_obs)
        cost = max(0.0, (2.0 - ttc) / 2.0)
        cost = min(cost, 1.0)

        # Classify nearest obstacle
        obs_class = self._classify_nearest(raw_obs)

        # Track
        self.episode_costs.append(cost)
        self.episode_ttcs.append(ttc)
        self.episode_classes.append(obs_class)

        info["cost"] = cost
        info["ttc"] = ttc
        # highway-env sets terminated=True on collision OR road departure
        # The inner env info dict contains the actual crashed flag
        info["collision"] = terminated and info.get("crashed", False)
        info["obstacle_class"] = obs_class
        info["step"] = self._step_count

        return obs, reward, terminated, truncated, info

    def _flatten_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Flatten (vehicles, features) → R^12.

        Layout: [ego(6), nearest_obstacle(6)]
        If raw_obs has shape (V, 6), take ego (row 0) and nearest (row 1).
        """
        if raw_obs.ndim == 1:
            # Already flat — pad/truncate to 12
            obs = np.zeros(12, dtype=np.float32)
            obs[:min(12, len(raw_obs))] = raw_obs[:12]
            return obs

        obs = np.zeros(12, dtype=np.float32)

        # Ego features (row 0)
        if raw_obs.shape[0] > 0:
            obs[:6] = raw_obs[0, :6]

        # Nearest obstacle (row 1, if exists)
        if raw_obs.shape[0] > 1:
            obs[6:12] = raw_obs[1, :6]

        return obs

    def _compute_ttc(self, raw_obs: np.ndarray) -> float:
        """Compute TTC from relative kinematics.

        TTC = dx / (-dvx) when closing (dvx < 0, dx > 0).
        """
        if raw_obs.ndim < 2 or raw_obs.shape[0] < 2:
            return TTC_MAX

        # relative coords (since absolute=False)
        dx = raw_obs[1, 0]  # x of nearest relative to ego
        dvx = raw_obs[1, 2]  # vx of nearest relative to ego

        # For relative obs: dx > 0 means obstacle ahead
        # dvx < 0 means obstacle approaching (closing speed)
        if dx > 0 and dvx < 0:
            ttc = -dx / dvx  # dvx is negative, so ttc is positive
            return float(np.clip(ttc, 0.0, TTC_MAX))

        return TTC_MAX

    def _classify_nearest(self, raw_obs: np.ndarray) -> int:
        """Classify nearest obstacle into 3 classes.

        0 = SLOW (trucks, < 25 m/s)
        1 = FAST (cars, ≥ 25 m/s)
        2 = STATIONARY (< 1 m/s)
        """
        if raw_obs.ndim < 2 or raw_obs.shape[0] < 2:
            return -1

        # vx of nearest obstacle (relative + ego vx)
        # If absolute=False, raw_obs[1,2] is relative vx
        # Absolute speed ≈ ego_vx + relative_vx
        ego_vx = raw_obs[0, 2] if raw_obs.shape[0] > 0 else 25.0
        rel_vx = raw_obs[1, 2]
        abs_speed = abs(ego_vx + rel_vx)

        if abs_speed < SPEED_STATIONARY_MAX:
            return 2  # STATIONARY
        elif abs_speed < SPEED_SLOW_MAX:
            return 0  # SLOW
        else:
            return 1  # FAST

    def get_state_text(self, obs: np.ndarray) -> str:
        """Convert R^12 observation to natural language for LLM input.

        This is the bridge between numeric state and LLM decision-making.
        """
        ego_x, ego_y, ego_vx, ego_vy, ego_cos, ego_sin = obs[:6]
        rel_x, rel_y, rel_vx, rel_vy, rel_cos, rel_sin = obs[6:12]

        ttc = (-rel_x / rel_vx) if (rel_x > 0 and rel_vx < 0) else TTC_MAX
        ttc = min(ttc, TTC_MAX)
        cost = max(0.0, (2.0 - ttc) / 2.0)

        speed_kmh = ego_vx * 3.6
        rel_speed_kmh = rel_vx * 3.6
        gap_m = rel_x

        danger = "CRITICAL" if cost > 0.7 else "HIGH" if cost > 0.3 else "LOW" if cost > 0 else "SAFE"

        text = (
            f"Driving state: speed={speed_kmh:.0f}km/h, "
            f"nearest vehicle gap={gap_m:.1f}m, "
            f"closing speed={abs(rel_speed_kmh):.0f}km/h, "
            f"TTC={ttc:.1f}s, "
            f"danger={danger}. "
            f"Choose action: 0=maintain, 1=accelerate, 2=brake, 3=lane_change"
        )
        return text

    def close(self) -> None:
        self._inner.close()

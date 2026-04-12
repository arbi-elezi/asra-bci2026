"""2D Driving Simulation — Gymnasium environment for FRA experiments.

Paper spec (Section 4.1):
  - s_t ∈ R^12
  - |A| = 4
  - 3 obstacle classes (slow, fast, stationary)
  - c_t = max(0, (2 - TTC_t) / 2)
  - Satisfies Definition 3 (FRA Applicability Conditions) by construction

State vector (R^12):
  [0] ego_x           — longitudinal position (m)
  [1] ego_y           — lateral position (m)
  [2] ego_vx          — longitudinal velocity (m/s)
  [3] ego_vy          — lateral velocity (m/s)
  [4] ego_heading     — heading angle (rad)
  [5] lane_index      — current lane (float-encoded)
  [6] nearest_dx      — relative x to nearest obstacle (m)
  [7] nearest_dy      — relative y to nearest obstacle (m)
  [8] nearest_dvx     — relative vx to nearest obstacle (m/s)
  [9] nearest_dvy     — relative vy to nearest obstacle (m/s)
  [10] ttc            — time-to-collision (s), clipped to [0, TTC_MAX]
  [11] obstacle_class — nearest obstacle class (float-encoded: 0, 1, 2)

Calibration: see experiments/indirect_data_sources.md for empirical grounding.
All randomness via seeded np.random.Generator — no global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .ego_vehicle import Action, EgoVehicle
from .obstacle import Obstacle, ObstacleClass, OBSTACLE_SPAWN_PROBS


# ---------- Constants ----------

TTC_MAX: float = 10.0          # Clip TTC to this value (s)
ROAD_LENGTH: float = 500.0     # Simulation road segment (m)
N_LANES: int = 3               # Number of lanes
LANE_WIDTH: float = 3.7        # Standard highway lane width (m)
DT: float = 0.1                # Timestep (s) — 10 Hz, matches NGSIM sampling
MAX_STEPS: int = 500           # Episode length (~50s real time)
SPAWN_DISTANCE: float = 150.0  # Obstacles spawn this far ahead (m)
DESPAWN_BEHIND: float = 50.0   # Remove obstacles this far behind ego (m)
COLLISION_DIST_X: float = 5.0  # Longitudinal collision threshold (m)
COLLISION_DIST_Y: float = 1.8  # Lateral collision threshold (m, ~half lane)
MAX_OBSTACLES: int = 20        # Max simultaneous obstacles
REWARD_ALIVE: float = 1.0      # Reward per step for surviving
REWARD_SPEED: float = 0.1      # Reward coefficient for maintaining speed
REWARD_COLLISION: float = -50.0  # Penalty for collision

# Obstacle density parameters (per km, grounded in HighD)
DENSITY_NORMAL: float = 18.0   # Normal traffic density
DENSITY_SHIFTED: float = 35.0  # High-density for FMS test (C5)


@dataclass
class EnvConfig:
    """Configuration dataclass for DrivingEnv.

    Typos become type errors — no dict-based configs per coding rules.
    """
    n_lanes: int = N_LANES
    lane_width: float = LANE_WIDTH
    road_length: float = ROAD_LENGTH
    dt: float = DT
    max_steps: int = MAX_STEPS
    spawn_distance: float = SPAWN_DISTANCE
    despawn_behind: float = DESPAWN_BEHIND
    collision_dist_x: float = COLLISION_DIST_X
    collision_dist_y: float = COLLISION_DIST_Y
    max_obstacles: int = MAX_OBSTACLES
    ttc_max: float = TTC_MAX
    obstacle_density: float = DENSITY_NORMAL
    spawn_probs: dict[ObstacleClass, float] = field(
        default_factory=lambda: dict(OBSTACLE_SPAWN_PROBS)
    )
    # Stress test hooks
    cost_bias_factor: dict[ObstacleClass, float] = field(
        default_factory=lambda: {
            ObstacleClass.SLOW: 1.0,
            ObstacleClass.FAST: 1.0,
            ObstacleClass.STATIONARY: 1.0,
        }
    )
    # Whether to use shifted (high-density) obstacle spawning
    use_shifted_density: bool = False


class DrivingEnv(gym.Env):
    """2D Driving Simulation satisfying Definition 3 (FRA Applicability Conditions).

    Definition 3 compliance:
      (a) Finite discrete action space: |A| = 4 ✓
      (b) Scalar cost signal c_t ∈ [0,1] observable at each timestep ✓
      (c) Offline cost critic trainable on D_ref ✓ (env provides ground-truth costs)
      (d) Per-class cost critic error controllable via stress test hooks ✓
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: EnvConfig | None = None,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        # Action space: |A| = 4 per paper
        self.action_space = spaces.Discrete(Action.COUNT)

        # Observation space: s_t ∈ R^12
        # Bounds are generous — actual values will be within tighter ranges
        obs_low = np.array([
            -np.inf,     # ego_x
            0.0,         # ego_y
            0.0,         # ego_vx
            -10.0,       # ego_vy
            -np.pi,      # heading
            0.0,         # lane
            -SPAWN_DISTANCE,  # nearest_dx
            -LANE_WIDTH * N_LANES,  # nearest_dy
            -60.0,       # nearest_dvx
            -10.0,       # nearest_dvy
            0.0,         # ttc
            0.0,         # obstacle_class
        ], dtype=np.float32)

        obs_high = np.array([
            np.inf,      # ego_x
            LANE_WIDTH * N_LANES,  # ego_y
            50.0,        # ego_vx
            10.0,        # ego_vy
            np.pi,       # heading
            float(N_LANES - 1),  # lane
            SPAWN_DISTANCE,      # nearest_dx
            LANE_WIDTH * N_LANES,  # nearest_dy
            60.0,        # nearest_dvx
            10.0,        # nearest_dvy
            TTC_MAX,     # ttc
            2.0,         # obstacle_class
        ], dtype=np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Lane centers
        self.lane_centers = [
            self.config.lane_width * (i + 0.5) for i in range(self.config.n_lanes)
        ]

        # State
        self._rng: np.random.Generator | None = None
        self.ego: EgoVehicle | None = None
        self.obstacles: list[Obstacle] = []
        self.step_count: int = 0
        self.collision: bool = False
        self.episode_cost_sum: float = 0.0

        # Metrics storage for per-timestep logging (M3, M13 requirement)
        self.episode_ttcs: list[float] = []
        self.episode_costs: list[float] = []
        self.episode_nearest_class: list[int] = []

        # Seed on init if provided
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed for reproducibility. Required for experiment runs.
            options: Optional dict with keys:
                - 'obstacle_density': override density for this episode
                - 'spawn_probs': override spawn probabilities
                - 'ego_speed': override initial ego speed

        Returns:
            (observation, info) tuple.
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng(42)

        # Parse options
        opts = options or {}
        density = opts.get("obstacle_density", self.config.obstacle_density)
        if self.config.use_shifted_density:
            density = DENSITY_SHIFTED
        spawn_probs = opts.get("spawn_probs", self.config.spawn_probs)
        ego_speed = opts.get("ego_speed", 25.0)

        # Reset ego
        self.ego = EgoVehicle(
            x=0.0,
            y=self.lane_centers[1],
            vx=ego_speed,
            lane=1,
            target_lane=1,
        )

        # Spawn initial obstacles
        self.obstacles = []
        n_initial = max(1, int(density * self.config.road_length / 1000.0))
        for _ in range(min(n_initial, self.config.max_obstacles)):
            obs = Obstacle.spawn(
                rng=self._rng,
                x_range=(self.ego.x + 20.0, self.ego.x + self.config.spawn_distance),
                lanes=self.lane_centers,
                spawn_probs=spawn_probs,
            )
            self.obstacles.append(obs)

        # Reset counters
        self.step_count = 0
        self.collision = False
        self.episode_cost_sum = 0.0
        self.episode_ttcs = []
        self.episode_costs = []
        self.episode_nearest_class = []

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Integer in {0, 1, 2, 3}.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        assert self.ego is not None, "Call reset() before step()"
        assert self._rng is not None

        self.step_count += 1
        dt = self.config.dt

        # 1. Apply ego action
        self.ego.step(action, dt, self.lane_centers, self.config.n_lanes)

        # 2. Advance obstacles
        for obs in self.obstacles:
            if obs.active:
                obs.step(dt)

        # 3. Despawn obstacles behind ego
        self.obstacles = [
            o for o in self.obstacles
            if o.x > self.ego.x - self.config.despawn_behind
        ]

        # 4. Spawn new obstacles ahead
        self._maybe_spawn_obstacles()

        # 5. Compute TTC and cost
        ttc, nearest_obs = self._compute_ttc()
        cost = self._compute_cost(ttc)

        # 6. Check collision
        collision = self._check_collision()
        self.collision = collision

        # 7. Compute reward
        speed_reward = REWARD_SPEED * (self.ego.vx / self.ego.max_speed)
        reward = REWARD_ALIVE + speed_reward
        if collision:
            reward = REWARD_COLLISION

        # 8. Track metrics
        self.episode_cost_sum += cost
        self.episode_ttcs.append(ttc)
        self.episode_costs.append(cost)
        if nearest_obs is not None:
            self.episode_nearest_class.append(int(nearest_obs.obstacle_class))
        else:
            self.episode_nearest_class.append(-1)

        # 9. Terminal conditions
        terminated = collision
        truncated = self.step_count >= self.config.max_steps

        obs = self._get_observation()
        info = self._get_info()
        info["cost"] = cost  # Paper: c_t directly observable (Definition 3b)
        info["ttc"] = ttc
        info["collision"] = collision
        info["nearest_class"] = (
            int(nearest_obs.obstacle_class) if nearest_obs else -1
        )

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Construct the 12-dimensional state vector."""
        assert self.ego is not None

        ego_x, ego_y, ego_vx, ego_vy, heading, lane_f = self.ego.get_state()

        # Find nearest obstacle
        nearest = self._find_nearest_obstacle()
        if nearest is not None:
            dx = nearest.x - ego_x
            dy = nearest.y - ego_y
            dvx = nearest.vx - ego_vx
            dvy = nearest.vy - ego_vy
            ttc = self._compute_ttc_single(nearest)
            obs_class = float(nearest.obstacle_class)
        else:
            dx = self.config.spawn_distance
            dy = 0.0
            dvx = 0.0
            dvy = 0.0
            ttc = self.config.ttc_max
            obs_class = 0.0

        state = np.array([
            ego_x, ego_y, ego_vx, ego_vy, heading, lane_f,
            dx, dy, dvx, dvy, ttc, obs_class,
        ], dtype=np.float32)

        return state

    def _find_nearest_obstacle(self) -> Obstacle | None:
        """Find the nearest active obstacle to the ego vehicle."""
        assert self.ego is not None
        nearest = None
        min_dist_sq = float("inf")

        for obs in self.obstacles:
            if not obs.active:
                continue
            dx = obs.x - self.ego.x
            dy = obs.y - self.ego.y
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest = obs

        return nearest

    def _compute_ttc_single(self, obs: Obstacle) -> float:
        """Compute time-to-collision with a single obstacle.

        TTC = dx / (-dvx) when approaching (dvx < 0 and dx > 0).
        Clipped to [0, TTC_MAX].
        """
        assert self.ego is not None
        dx = obs.x - self.ego.x
        dvx = self.ego.vx - obs.vx  # closing speed

        # Only compute TTC if ego is closing on obstacle
        if dvx <= 0.0 or dx < 0.0:
            return self.config.ttc_max

        # Check lateral overlap (must be in same or adjacent lane region)
        dy = abs(obs.y - self.ego.y)
        if dy > self.config.collision_dist_y * 2.0:
            return self.config.ttc_max

        ttc = dx / dvx
        return float(np.clip(ttc, 0.0, self.config.ttc_max))

    def _compute_ttc(self) -> tuple[float, Obstacle | None]:
        """Compute minimum TTC across all obstacles.

        Returns:
            (min_ttc, nearest_obstacle_by_ttc)
        """
        min_ttc = self.config.ttc_max
        nearest = None

        for obs in self.obstacles:
            if not obs.active:
                continue
            ttc = self._compute_ttc_single(obs)
            if ttc < min_ttc:
                min_ttc = ttc
                nearest = obs

        return min_ttc, nearest

    def _compute_cost(self, ttc: float) -> float:
        """Compute cost signal per paper: c_t = max(0, (2 - TTC_t) / 2).

        Returns:
            Cost in [0, 1]. Cost > 0 when TTC < 2s. Cost = 1 when TTC = 0.
        """
        cost = max(0.0, (2.0 - ttc) / 2.0)
        return min(cost, 1.0)

    def _check_collision(self) -> bool:
        """Check if ego collides with any obstacle."""
        assert self.ego is not None

        for obs in self.obstacles:
            if not obs.active:
                continue
            dx = abs(obs.x - self.ego.x)
            dy = abs(obs.y - self.ego.y)
            if dx < self.config.collision_dist_x and dy < self.config.collision_dist_y:
                return True

        return False

    def _maybe_spawn_obstacles(self) -> None:
        """Spawn new obstacles ahead of the ego vehicle based on density."""
        assert self.ego is not None
        assert self._rng is not None

        if len(self.obstacles) >= self.config.max_obstacles:
            return

        density = self.config.obstacle_density
        if self.config.use_shifted_density:
            density = DENSITY_SHIFTED

        # Poisson-ish spawn: probability proportional to density and dt
        spawn_prob = density * self.ego.vx * self.config.dt / 1000.0
        if self._rng.random() < spawn_prob:
            obs = Obstacle.spawn(
                rng=self._rng,
                x_range=(
                    self.ego.x + self.config.spawn_distance * 0.8,
                    self.ego.x + self.config.spawn_distance,
                ),
                lanes=self.lane_centers,
                spawn_probs=self.config.spawn_probs,
            )
            self.obstacles.append(obs)

    def _get_info(self) -> dict[str, Any]:
        """Return info dict with episode metrics."""
        assert self.ego is not None
        return {
            "step": self.step_count,
            "ego_x": self.ego.x,
            "ego_vx": self.ego.vx,
            "ego_lane": self.ego.lane,
            "n_obstacles": len(self.obstacles),
            "episode_cost_sum": self.episode_cost_sum,
        }

    def get_cost_with_bias(self, ttc: float, obstacle_class: ObstacleClass) -> float:
        """Compute cost with optional per-class bias for stress test conditions.

        C8d: fast-obstacle costs multiplied by 0.2
        C8e: fast-obstacle costs multiplied by 0.5

        This is used to train BIASED cost critics, NOT to bias the true cost.
        The true cost is always c_t = max(0, (2 - TTC) / 2).

        Args:
            ttc: Time-to-collision.
            obstacle_class: The obstacle class.

        Returns:
            Biased cost in [0, 1].
        """
        true_cost = self._compute_cost(ttc)
        bias = self.config.cost_bias_factor.get(obstacle_class, 1.0)
        return float(np.clip(true_cost * bias, 0.0, 1.0))

    def render(self) -> np.ndarray | None:
        """Render the environment (optional — not needed for experiments)."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self) -> np.ndarray:
        """Produce an RGB frame for visualization."""
        assert self.ego is not None
        width, height = 800, 200
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Road background
        frame[:, :] = [80, 80, 80]

        # Lane markings
        for i in range(self.config.n_lanes + 1):
            y_px = int(i * height / self.config.n_lanes)
            frame[max(0, y_px - 1):y_px + 1, :] = [255, 255, 255]

        # Camera follows ego
        cam_x = self.ego.x

        # Draw obstacles
        for obs in self.obstacles:
            if not obs.active:
                continue
            px = int((obs.x - cam_x + 100) / 300 * width)
            py = int(obs.y / (self.config.lane_width * self.config.n_lanes) * height)
            if 0 <= px < width and 0 <= py < height:
                color = {
                    ObstacleClass.SLOW: [100, 100, 255],
                    ObstacleClass.FAST: [255, 100, 100],
                    ObstacleClass.STATIONARY: [200, 200, 0],
                }[obs.obstacle_class]
                r = 4
                frame[
                    max(0, py - r):min(height, py + r),
                    max(0, px - r):min(width, px + r),
                ] = color

        # Draw ego
        ego_px = int(100 / 300 * width)
        ego_py = int(
            self.ego.y / (self.config.lane_width * self.config.n_lanes) * height
        )
        r = 5
        frame[
            max(0, ego_py - r):min(height, ego_py + r),
            max(0, ego_px - r):min(width, ego_px + r),
        ] = [0, 255, 0]

        return frame

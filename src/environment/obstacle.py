"""Obstacle classes for the 2D driving simulation.

Three obstacle classes grounded in HighD dataset speed distributions:
- SLOW: trucks/heavy vehicles (~22 m/s, HighD truck peak ~80 km/h)
- FAST: cars (~33 m/s, HighD car peak ~120 km/h)
- STATIONARY: breakdowns/construction (0 m/s)

All randomness via explicit RNG — no global state.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class ObstacleClass(enum.IntEnum):
    """Three obstacle classes per paper Section 4.1."""
    SLOW = 0
    FAST = 1
    STATIONARY = 2


# Grounded in HighD dataset — see experiments/indirect_data_sources.md
OBSTACLE_SPEED_PARAMS: dict[ObstacleClass, tuple[float, float]] = {
    # (mean_m_s, std_m_s)
    ObstacleClass.SLOW: (22.0, 2.0),        # trucks ~80 km/h
    ObstacleClass.FAST: (33.0, 3.0),         # cars ~120 km/h
    ObstacleClass.STATIONARY: (0.0, 0.0),    # static
}

# Spawn probability distribution (normal traffic)
OBSTACLE_SPAWN_PROBS: dict[ObstacleClass, float] = {
    ObstacleClass.SLOW: 0.3,
    ObstacleClass.FAST: 0.5,
    ObstacleClass.STATIONARY: 0.2,
}


@dataclass
class Obstacle:
    """A single obstacle in the 2D driving environment.

    Attributes:
        x: Longitudinal position (m).
        y: Lateral position (lane center, m).
        vx: Longitudinal velocity (m/s).
        vy: Lateral velocity (m/s) — always 0 for simplicity.
        obstacle_class: One of the three obstacle classes.
        lane: Lane index (0-indexed).
        active: Whether this obstacle is currently in the simulation.
    """
    x: float
    y: float
    vx: float
    vy: float = 0.0
    obstacle_class: ObstacleClass = ObstacleClass.SLOW
    lane: int = 0
    active: bool = True

    def step(self, dt: float) -> None:
        """Advance obstacle by one timestep."""
        self.x += self.vx * dt
        self.y += self.vy * dt

    @staticmethod
    def spawn(
        rng: np.random.Generator,
        x_range: tuple[float, float],
        lanes: list[float],
        obstacle_class: ObstacleClass | None = None,
        spawn_probs: dict[ObstacleClass, float] | None = None,
    ) -> Obstacle:
        """Spawn a new obstacle with class-appropriate speed.

        Args:
            rng: Seeded random generator (no global state).
            x_range: (min_x, max_x) for spawn position.
            lanes: List of lane center y-coordinates.
            obstacle_class: Force a specific class, or None for random.
            spawn_probs: Override spawn probabilities per class.

        Returns:
            A new Obstacle instance.
        """
        probs = spawn_probs or OBSTACLE_SPAWN_PROBS
        if obstacle_class is None:
            classes = list(probs.keys())
            p = np.array([probs[c] for c in classes], dtype=np.float64)
            p /= p.sum()
            obstacle_class = classes[rng.choice(len(classes), p=p)]

        mean_v, std_v = OBSTACLE_SPEED_PARAMS[obstacle_class]
        vx = max(0.0, rng.normal(mean_v, std_v)) if std_v > 0.0 else mean_v

        lane_idx = rng.integers(0, len(lanes))
        x = rng.uniform(x_range[0], x_range[1])

        return Obstacle(
            x=x,
            y=lanes[lane_idx],
            vx=vx,
            obstacle_class=obstacle_class,
            lane=int(lane_idx),
        )

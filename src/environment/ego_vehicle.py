"""Ego vehicle dynamics for the 2D driving simulation.

Simple kinematic model: 4 discrete actions per paper spec (|A|=4).
Actions: {MAINTAIN, ACCELERATE, BRAKE, LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT}
Wait — paper says |A|=4. We use: {MAINTAIN, ACCELERATE, BRAKE, LANE_CHANGE}
Actually, with 4 actions and lane changes, the standard decomposition is:
  0: MAINTAIN (keep speed and lane)
  1: ACCELERATE (increase speed)
  2: BRAKE (decrease speed)
  3: LANE_CHANGE (switch to adjacent lane — direction chosen by env logic)

Re-reading: |A|=4 means exactly 4 actions. A natural split for highway driving:
  0: MAINTAIN
  1: ACCELERATE
  2: BRAKE
  3: STEER_LEFT (lane change left)

But that's asymmetric. Better:
  0: MAINTAIN
  1: ACCELERATE
  2: DECELERATE
  3: LANE_CHANGE (toggle between available lanes)

The paper doesn't specify exact actions, just |A|=4. We choose:
  0: KEEP_LANE (maintain speed)
  1: ACCELERATE
  2: BRAKE
  3: LANE_CHANGE (move toward safer adjacent lane)

This is a simplification consistent with |A|=4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Action space: |A| = 4 per paper Section 4.1
class Action:
    KEEP_LANE = 0
    ACCELERATE = 1
    BRAKE = 2
    LANE_CHANGE = 3

    COUNT = 4


@dataclass
class EgoVehicle:
    """Ego vehicle with simple kinematic dynamics.

    Attributes:
        x: Longitudinal position (m).
        y: Lateral position (m) — snaps to lane centers.
        vx: Longitudinal velocity (m/s).
        vy: Lateral velocity (m/s) — used during lane changes.
        heading: Heading angle (rad) — 0 = forward.
        lane: Current lane index.
        target_lane: Target lane during lane change.
        max_speed: Maximum longitudinal speed (m/s).
        min_speed: Minimum longitudinal speed (m/s).
        accel: Acceleration magnitude (m/s^2).
        brake_decel: Braking deceleration magnitude (m/s^2).
        lane_change_duration: Steps to complete a lane change.
        lane_change_progress: Current progress in lane change (0 = not changing).
    """
    x: float = 0.0
    y: float = 0.0
    vx: float = 25.0  # ~90 km/h default
    vy: float = 0.0
    heading: float = 0.0
    lane: int = 1      # Start in middle lane
    target_lane: int = 1
    max_speed: float = 40.0   # ~144 km/h
    min_speed: float = 10.0   # ~36 km/h
    accel: float = 3.0        # m/s^2
    brake_decel: float = 5.0  # m/s^2
    lane_change_duration: int = 10  # steps
    lane_change_progress: int = 0

    def step(self, action: int, dt: float, lane_centers: list[float], n_lanes: int) -> None:
        """Apply action and advance one timestep.

        Args:
            action: Integer action in {0, 1, 2, 3}.
            dt: Timestep duration (s).
            lane_centers: Y-coordinates of lane centers.
            n_lanes: Number of lanes.
        """
        # Longitudinal dynamics
        if action == Action.ACCELERATE:
            self.vx = min(self.max_speed, self.vx + self.accel * dt)
        elif action == Action.BRAKE:
            self.vx = max(self.min_speed, self.vx - self.brake_decel * dt)

        # Lane change initiation
        if action == Action.LANE_CHANGE and self.lane_change_progress == 0:
            # Move toward lane with fewer obstacles (env decides direction)
            # Default: cycle lanes 0->1->2->0
            new_lane = (self.lane + 1) % n_lanes
            self.target_lane = new_lane
            self.lane_change_progress = 1

        # Lane change execution
        if self.lane_change_progress > 0:
            self.lane_change_progress += 1
            t = min(self.lane_change_progress / self.lane_change_duration, 1.0)
            self.y = lane_centers[self.lane] + t * (
                lane_centers[self.target_lane] - lane_centers[self.lane]
            )
            if self.lane_change_progress >= self.lane_change_duration:
                self.lane = self.target_lane
                self.y = lane_centers[self.lane]
                self.lane_change_progress = 0
                self.vy = 0.0
            else:
                self.vy = (lane_centers[self.target_lane] - lane_centers[self.lane]) / (
                    self.lane_change_duration * dt
                )
        else:
            self.y = lane_centers[self.lane]
            self.vy = 0.0

        # Advance position
        self.x += self.vx * dt
        self.heading = np.arctan2(self.vy, self.vx) if self.vx > 0 else 0.0

    def get_state(self) -> tuple[float, float, float, float, float, float]:
        """Return (x, y, vx, vy, heading, lane_float)."""
        return (self.x, self.y, self.vx, self.vy, self.heading, float(self.lane))

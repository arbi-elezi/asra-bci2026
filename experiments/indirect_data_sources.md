# Indirectly Useful Data Sources

These are NOT the simulation itself — they ground simulation parameters in real-world traffic data.

## 1. NGSIM (US-101, I-80) — TTC and Traffic Dynamics
- **Source**: https://data.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/
- **Kaggle mirror**: https://www.kaggle.com/datasets/nigelwilliams/ngsim-vehicle-trajectory-data-us-101
- **What it provides**:
  - Real vehicle trajectory data (position, velocity, acceleration at 10Hz)
  - Mean time headway: ~2.6s (jammed), ~1.9s (bound), ~2.0s (free traffic)
  - TTC distributions across traffic regimes
  - Lane-change dynamics and frequencies
- **How we use it**: Calibrate obstacle spawn rates, speed distributions, and TTC thresholds in our simulation to be empirically grounded, not arbitrary

## 2. HighD — German Highway Trajectories
- **Source**: https://levelxdata.com/highd-dataset/
- **Paper**: https://arxiv.org/abs/1810.05642
- **What it provides**:
  - 110,500 vehicles, 16.5 hours, 45,000 km driven
  - Speed peaks at 80 km/h (trucks) and 120 km/h (cars)
  - Vehicle class: Car vs Truck (maps to our slow vs fast obstacles)
  - <10cm positioning accuracy (drone-based)
  - 5,600 lane changes
- **How we use it**: Speed distribution for our 3 obstacle classes; realistic velocity ratios between slow/fast obstacles

## 3. TTC Safety Thresholds — Literature Consensus
- **Key findings**:
  - TTC < 1.5s: consistently rated "critical" by trained observers
  - TTC < 2s: widely used safety threshold in collision avoidance systems
  - TTC < 4s: typical braking initiation threshold
  - Paper's cost: c_t = max(0, (2 - TTC_t) / 2) → cost > 0 when TTC < 2s, cost = 1 when TTC = 0
  - This is WELL-GROUNDED in the literature (2s is the standard critical threshold)
- **Sources**: NHTSA collision avoidance studies, van der Horst 1990, multiple NGSIM analyses

## 4. DrivingRL (GitHub) — Reference Implementation
- **Source**: https://github.com/kylesayrs/DrivingRL
- **What it provides**: Basic 2D car environment trained with PPO using Stable-Baselines3
- **How we use it**: Reference for PPO+2D driving integration patterns

## 5. State Space Design — Literature Grounding
From PPO driving research, typical R^12 state features:
1. ego_x (longitudinal position)
2. ego_y (lateral position)
3. ego_vx (longitudinal velocity)
4. ego_vy (lateral velocity)
5. ego_heading (orientation angle)
6. lane_index (discrete but encoded as float)
7-8. nearest_obstacle_dx, nearest_obstacle_dy (relative position)
9-10. nearest_obstacle_dvx, nearest_obstacle_dvy (relative velocity)
11. nearest_obstacle_ttc (time-to-collision)
12. nearest_obstacle_class (encoded: 0=slow, 1=fast, 2=stationary)

This 12-feature design covers ego dynamics, relative obstacle state, and TTC — matching the paper's s_t ∈ R^12.

## Calibration Parameters Derived from Real Data

| Parameter | Value | Source |
|-----------|-------|--------|
| Slow obstacle speed | ~80 km/h (22 m/s) | HighD truck speed peak |
| Fast obstacle speed | ~120 km/h (33 m/s) | HighD car speed peak |
| Stationary obstacle speed | 0 m/s | Construction/breakdown |
| Ego agent speed range | 20-35 m/s | Highway-env default |
| TTC critical threshold | 2.0s | Paper (matches NHTSA/literature) |
| Mean time headway | ~2.0s | NGSIM free traffic |
| Lane count | 3-4 | NGSIM/HighD typical |
| Obstacle density (normal) | ~15-20 per km | HighD moderate traffic |
| Obstacle density (shifted) | ~30-40 per km | For C5 FMS test seeds |

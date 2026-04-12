"""Experiment v6 — Gaussian perturbation with LEARNED exciter.

The exciter is a small neural network that learns:
  - How much to suppress (magnitude)
  - Where to suppress (epicenter weighting)
  - How wide the Gaussian should be (sigma)

It learns from feedback: after suppression, did the perturbed LLM
choose a less risky action? That reward signal trains the exciter.

The experiment runs thousands of episodes. The exciter trains online.
The measurement is: does the exciter learn to reliably produce less
risky actions, and how many episodes does it take?

Recovery follows damped elastic dynamics — smooth gradual return
through intermediate states of caution.
"""

import json
import time
import sys
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.highway_wrapper import HighwayFRAEnv
from src.components.fear_detector import FearDetector, FearDetectorConfig
from src.evaluation.metrics import bootstrap_ci, m1_collision_rate
from experiment_v3.config import MODELS
from experiment_v3.llm_policy import LLMPolicy


# ═══════════════════════════════════════════════════════════════════════════
# RISK EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════

def compute_risk(obs, cost, ttc, action):
    """Risk score for a specific action in a specific state. [0,1]"""
    base = {0: 0.3, 1: 0.6, 2: 0.05, 3: 0.5}.get(action, 0.3)
    if action != 2 and ttc < 3.0:
        base = min(1.0, base + (3.0 - ttc) / 3.0 * 0.5)
    base = min(1.0, base + cost * 0.3)
    if action == 2:
        base = min(0.1, base)
    return float(np.clip(base, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# EXCITER — the learned perturbation controller
# ═══════════════════════════════════════════════════════════════════════════

class Exciter(nn.Module):
    """Learns HOW to perturb weights given the current gradient pattern.

    Input: fear_signal, risk_score, gradient_summary (top-K gradient stats)
    Output: magnitude (how much), sigma (how wide), epicenter_weights (where)

    Trained online from reward = risk_reduction after perturbation.
    """

    def __init__(self, n_param_groups=10, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.n_groups = n_param_groups

        # Input: [fear, risk, grad_mean, grad_max, grad_std, per-group grad magnitudes]
        input_dim = 5 + n_param_groups

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        ).to(self.device).float()

        # Output heads
        self.magnitude_head = nn.Linear(32, 1).to(self.device).float()   # how much
        self.sigma_head = nn.Linear(32, 1).to(self.device).float()       # how wide
        self.weights_head = nn.Linear(32, n_param_groups).to(self.device).float()  # where (per group)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

        # Experience buffer for training
        self.buffer = deque(maxlen=5000)

        # Training stats
        self.train_steps = 0
        self.reward_history = []

    def forward(self, features):
        """features → (magnitude, sigma, epicenter_weights)"""
        h = self.net(features)
        magnitude = torch.sigmoid(self.magnitude_head(h)) * 0.1   # [0, 0.1]
        sigma = torch.sigmoid(self.sigma_head(h)) * 0.5 + 0.01    # [0.01, 0.51]
        weights = torch.softmax(self.weights_head(h), dim=-1)      # per-group importance
        return magnitude.squeeze(-1), sigma.squeeze(-1), weights

    def get_perturbation_params(self, fear, risk, grad_stats, grad_per_group):
        """Get perturbation parameters for current state."""
        features = torch.tensor(
            [fear, risk, grad_stats["mean"], grad_stats["max"], grad_stats["std"]]
            + grad_per_group,
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        magnitude, sigma, weights = self.forward(features)
        return {
            "magnitude": magnitude.item(),
            "sigma": sigma.item(),
            "group_weights": weights.squeeze(0).detach().cpu().numpy(),
        }

    def store_experience(self, features, magnitude, sigma, risk_reduction):
        """Store one experience for training."""
        self.buffer.append({
            "features": features,
            "magnitude": magnitude,
            "sigma": sigma,
            "reward": risk_reduction,  # positive = good suppression
        })

    def train_step(self, batch_size=64):
        """Train exciter from experience buffer."""
        if len(self.buffer) < batch_size:
            return None

        # Sample batch
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        features = torch.stack([b["features"] for b in batch]).to(self.device)
        rewards = torch.tensor([b["reward"] for b in batch], device=self.device, dtype=torch.float32)

        # Forward
        mag, sig, weights = self.forward(features)

        # Loss: maximize risk reduction (reward)
        # Use policy gradient style: -reward * log_prob
        # Approximate: just maximize reward directly via MSE on magnitude
        # If reward was high, the magnitude was good → reinforce it
        target_mag = torch.clamp(mag.detach() + 0.01 * rewards, 0, 0.1)
        loss = nn.functional.mse_loss(mag, target_mag)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        mean_reward = rewards.mean().item()
        self.reward_history.append(mean_reward)

        return {"loss": loss.item(), "mean_reward": mean_reward}


# ═══════════════════════════════════════════════════════════════════════════
# ELASTIC RECOVERY STATE
# ═══════════════════════════════════════════════════════════════════════════

class ElasticState:
    """Tracks per-weight displacement and velocity for damped spring recovery.

    Each weight has:
      - displacement: current (w - w0)
      - velocity: rate of change
      - damping: prevents oscillation
      - stiffness: Fisher-weighted restoring force
    """

    def __init__(self, param_shapes, fisher, device, damping=0.8, dt=1.0):
        self.device = torch.device(device)
        self.damping = damping
        self.dt = dt

        # Per-weight velocity (starts at zero)
        self.velocities = {}
        for name, shape in param_shapes.items():
            self.velocities[name] = torch.zeros(shape, device=self.device)

        # Fisher as stiffness (stiffer = faster recovery for important params)
        self.fisher = fisher
        self.param_shapes = param_shapes

    def apply_suppression(self, name, delta):
        """Apply a suppression impulse (instant velocity change)."""
        if name in self.velocities:
            self.velocities[name] -= delta * 0.5  # impulse → velocity

    def step(self, current_params, w0, param_names):
        """Integrate one timestep of damped spring dynamics.

        For each weight:
          acceleration = -k * (w - w0) - gamma * velocity
          velocity += acceleration * dt
          w += velocity * dt

        Returns dict of parameter updates.
        """
        updates = {}
        pidx = 0

        for name in param_names:
            if name not in w0 or name not in self.velocities:
                if name in self.param_shapes:
                    pidx += np.prod(self.param_shapes[name])
                continue

            w = current_params[name]
            w0_val = w0[name].to(w.dtype)
            displacement = w - w0_val

            # Stiffness from Fisher
            shape = w.shape
            n_elem = w.numel()
            if self.fisher is not None and pidx + n_elem <= len(self.fisher):
                k = self.fisher[pidx:pidx+n_elem].reshape(shape).to(w.dtype).to(self.device)
                k = k * 0.01  # Scale stiffness
            else:
                k = torch.ones_like(w) * 0.001
            pidx += n_elem

            # Damped spring: a = -k*x - gamma*v
            acceleration = -k * displacement - self.damping * self.velocities[name]

            # Integrate
            self.velocities[name] += acceleration * self.dt
            update = self.velocities[name] * self.dt

            updates[name] = update

            # Decay velocity near equilibrium (prevents jitter)
            near_eq = displacement.abs() < 1e-6
            self.velocities[name][near_eq] *= 0.5

        return updates

    def reset(self):
        for name in self.velocities:
            self.velocities[name].zero_()


# ═══════════════════════════════════════════════════════════════════════════
# GAUSSIAN PERTURBATION
# ═══════════════════════════════════════════════════════════════════════════

def apply_gaussian_perturbation(param, gradient, epicenter_idx, sigma, magnitude):
    """Apply Gaussian-shaped suppression centered on epicenter.

    Args:
        param: the weight tensor to perturb
        gradient: gradient of log P(risky_action)
        epicenter_idx: index of the epicenter weight (flattened)
        sigma: Gaussian width (fraction of total params)
        magnitude: suppression strength
    """
    n = param.numel()
    flat_grad = gradient.flatten()
    flat_param = param.data.flatten()

    # Build Gaussian kernel centered on epicenter
    indices = torch.arange(n, device=param.device, dtype=torch.float32)
    sigma_abs = max(1, int(sigma * n))
    gaussian = torch.exp(-((indices - epicenter_idx) ** 2) / (2 * sigma_abs ** 2))

    # Apply: suppress proportional to Gaussian × gradient
    suppression = magnitude * gaussian * flat_grad
    flat_param -= suppression

    param.data = flat_param.reshape(param.shape)
    return gaussian  # Return kernel for FHR targeting


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_v6(model_name="SmolLM2-135M", n_episodes=2000, device="cuda"):
    v3_dir = Path("experiment_v3") / model_name
    v6_dir = Path("experiment_v6") / model_name
    v6_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT v6: LEARNED EXCITER + GAUSSIAN + ELASTIC RECOVERY")
    print("=" * 60)

    # Load model
    model_cfg = next(m for m in MODELS if m.name == model_name)
    policy = LLMPolicy(model_cfg, device=device)
    policy.load(str(v3_dir / "policy_trained.pt"))
    w0 = policy.get_w0()

    day1 = json.load(open(v3_dir / "day1_log.json"))
    fisher = torch.load(v3_dir / "fisher.pt", weights_only=False, map_location=device)
    fear_det = FearDetector(FearDetectorConfig(device=device))
    fear_det.load_state(torch.load(v3_dir / "fear_detector.pt", weights_only=False, map_location=device))

    env = HighwayFRAEnv(seed=0, vehicles_count=15)

    # Divide params into groups for exciter
    param_names = [n for n, _ in policy.get_perturbable_params()]
    n_groups = min(10, len(param_names))
    group_size = len(param_names) // n_groups

    # Initialize exciter
    exciter = Exciter(n_param_groups=n_groups, device=device)

    # Initialize elastic state
    param_shapes = {n: p.shape for n, p in policy.get_perturbable_params()}
    elastic = ElasticState(param_shapes, fisher, device)

    print(f"  Model: {model_name} ({policy.n_perturbable:,} params, {n_groups} groups)")
    print(f"  Baseline CR: {day1['base_cr']}")
    print(f"  Training exciter over {n_episodes} episodes")

    # ── Training loop ──
    metrics_per_episode = []
    rolling_risk_reduction = deque(maxlen=100)
    rolling_less_risky_pct = deque(maxlen=100)
    rolling_cr = deque(maxlen=100)

    checkpoint_dir = v6_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Cache param lookup (avoid O(n²) per step)
    param_dict = dict(policy.get_perturbable_params())

    for ep in range(n_episodes):
        policy.restore_w0()
        # Refresh cache after restore
        param_dict = dict(policy.get_perturbable_params())
        elastic.reset()
        obs, info = env.reset(seed=ep)

        ep_reward = 0.0
        collision = False
        ep_risk_reductions = []
        ep_fra_steps = 0
        ep_less_risky = 0

        for t in range(500):
            cost = info.get("cost", 0.0)
            ttc = info.get("ttc", 10.0)

            # ── Step 1: LLM decides ──
            with torch.no_grad():
                logits = policy.get_logits_from_obs(obs)
                greedy = logits.argmax().item()

            # ── Step 2: Risk evaluation ──
            risk = compute_risk(obs, cost, ttc, greedy)
            fear, _ = fear_det.detect(obs, cost, ttc, greedy)

            if fear > 0.05 and risk > 0.1:
                ep_fra_steps += 1

                # ── Step 3: Compute gradient ──
                policy.model.zero_grad()
                policy.action_head.zero_grad()
                logits = policy.get_logits_from_obs(obs)
                torch.log_softmax(logits, dim=-1)[greedy].backward()

                # Gradient stats per group
                grad_per_group = []
                all_grad_mags = []
                for gi in range(n_groups):
                    start = gi * group_size
                    end = min(start + group_size, len(param_names))
                    group_grad_sum = 0.0
                    for pi in range(start, end):
                        name = param_names[pi]
                        p = param_dict[name]
                        if p.grad is not None:
                            gm = p.grad.data.abs().mean().item()
                            group_grad_sum += gm
                            all_grad_mags.append(gm)
                    grad_per_group.append(group_grad_sum / max(end - start, 1))

                grad_stats = {
                    "mean": np.mean(all_grad_mags) if all_grad_mags else 0,
                    "max": max(all_grad_mags) if all_grad_mags else 0,
                    "std": np.std(all_grad_mags) if all_grad_mags else 0,
                }

                # ── Step 4: Exciter decides perturbation params ──
                exciter_params = exciter.get_perturbation_params(
                    fear, risk, grad_stats, grad_per_group
                )
                magnitude = exciter_params["magnitude"]
                sigma = exciter_params["sigma"]
                group_weights = exciter_params["group_weights"]

                # ── Step 5: Apply Gaussian perturbation ──
                with torch.no_grad():
                    for gi in range(n_groups):
                        start = gi * group_size
                        end = min(start + group_size, len(param_names))
                        gw = group_weights[gi]

                        for pi in range(start, end):
                            name = param_names[pi]
                            p = param_dict[name]
                            if p.grad is None:
                                continue

                            grad = p.grad.data
                            epicenter = grad.abs().flatten().argmax().item()

                            # Scale magnitude by group weight and risk
                            effective_mag = magnitude * gw * risk

                            kernel = apply_gaussian_perturbation(
                                p, grad, epicenter, sigma, effective_mag
                            )

                            # Register impulse with elastic system
                            impulse = effective_mag * kernel.reshape(p.shape) * grad
                            elastic.apply_suppression(name, impulse)

                # ── Step 6: Perturbed LLM decides ──
                with torch.no_grad():
                    new_logits = policy.get_logits_from_obs(obs)
                    new_action = torch.distributions.Categorical(logits=new_logits).sample().item()

                # ── Step 7: Measure risk reduction (FEEDBACK) ──
                risk_new = compute_risk(obs, cost, ttc, new_action)
                risk_reduction = risk - risk_new  # positive = less risky = good

                ep_risk_reductions.append(risk_reduction)
                if risk_reduction > 0:
                    ep_less_risky += 1

                # ── Step 8: Feed back to exciter ──
                features = torch.tensor(
                    [fear, risk, grad_stats["mean"], grad_stats["max"], grad_stats["std"]]
                    + grad_per_group,
                    dtype=torch.float32, device=torch.device(device)
                )
                exciter.store_experience(features, magnitude, sigma, risk_reduction)

                action = new_action
            else:
                # No FRA — use greedy
                action = greedy

            # ── Step 9: Elastic recovery (every step, regardless of FRA) ──
            with torch.no_grad():
                current = {n: p.data for n, p in param_dict.items()}
                updates = elastic.step(current, w0, param_names)
                for name, update in updates.items():
                    p = param_dict[name]
                    p.data += update

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if terminated:
                collision = info.get("collision", False)
                break
            if truncated:
                break

        # ── Train exciter every episode ──
        train_result = exciter.train_step(batch_size=min(64, len(exciter.buffer)))

        # ── Track metrics ──
        mean_rr = np.mean(ep_risk_reductions) if ep_risk_reductions else 0
        less_risky_pct = ep_less_risky / max(ep_fra_steps, 1)

        rolling_risk_reduction.append(mean_rr)
        rolling_less_risky_pct.append(less_risky_pct)
        rolling_cr.append(int(collision))

        ep_metrics = {
            "episode": ep,
            "collision": int(collision),
            "reward": ep_reward,
            "n_fra_steps": ep_fra_steps,
            "n_less_risky": ep_less_risky,
            "less_risky_pct": less_risky_pct,
            "mean_risk_reduction": mean_rr,
            "exciter_buffer_size": len(exciter.buffer),
            "exciter_train_steps": exciter.train_steps,
        }
        metrics_per_episode.append(ep_metrics)

        # Progress report
        if (ep + 1) % 50 == 0:
            avg_rr = np.mean(rolling_risk_reduction)
            avg_lr = np.mean(rolling_less_risky_pct)
            avg_cr = np.mean(rolling_cr)
            train_info = f"loss={train_result['loss']:.4f}" if train_result else "warmup"
            print(f"  [{ep+1}/{n_episodes}] "
                  f"CR={avg_cr:.3f} | "
                  f"LessRisky={avg_lr:.0%} | "
                  f"RiskRed={avg_rr:+.3f} | "
                  f"Buffer={len(exciter.buffer)} | "
                  f"{train_info}")

        # Checkpoint every 200 episodes
        if (ep + 1) % 200 == 0:
            torch.save({
                "exciter": exciter.state_dict(),
                "episode": ep,
                "metrics": metrics_per_episode[-200:],
            }, checkpoint_dir / f"ep_{ep+1}.pt")

    # ── Save final results ──
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")

    # Learning curve analysis
    window = 100
    learning_curve = []
    for i in range(0, len(metrics_per_episode) - window, window // 2):
        chunk = metrics_per_episode[i:i+window]
        learning_curve.append({
            "episode_start": i,
            "mean_less_risky_pct": np.mean([m["less_risky_pct"] for m in chunk]),
            "mean_risk_reduction": np.mean([m["mean_risk_reduction"] for m in chunk]),
            "mean_cr": np.mean([m["collision"] for m in chunk]),
        })

    print("\nLearning curve (100-episode windows):")
    print(f"{'Episodes':>12s}  {'LessRisky%':>12s}  {'RiskReduction':>14s}  {'CR':>6s}")
    for lc in learning_curve:
        print(f"  {lc['episode_start']:>4d}-{lc['episode_start']+window:<4d}  "
              f"{lc['mean_less_risky_pct']:>10.1%}  "
              f"{lc['mean_risk_reduction']:>+12.4f}  "
              f"{lc['mean_cr']:>6.3f}")

    # Did the exciter learn?
    if len(learning_curve) >= 2:
        first = learning_curve[0]
        last = learning_curve[-1]
        improved = last["mean_less_risky_pct"] > first["mean_less_risky_pct"]
        print(f"\nExciter learning: {'YES' if improved else 'NO'}")
        print(f"  Start: {first['mean_less_risky_pct']:.1%} less risky")
        print(f"  End:   {last['mean_less_risky_pct']:.1%} less risky")

    # Save everything
    with open(v6_dir / "metrics.json", "w") as f:
        json.dump(metrics_per_episode, f, indent=2, default=str)
    with open(v6_dir / "learning_curve.json", "w") as f:
        json.dump(learning_curve, f, indent=2, default=str)

    torch.save(exciter.state_dict(), v6_dir / "exciter_final.pt")

    with open(v6_dir / "experiment_log.json", "w") as f:
        json.dump({
            "version": "v6",
            "mechanism": "learned_exciter_gaussian_elastic",
            "model": model_name,
            "n_episodes": n_episodes,
            "baseline_cr": day1["base_cr"],
            "final_less_risky_pct": learning_curve[-1]["mean_less_risky_pct"] if learning_curve else 0,
            "final_risk_reduction": learning_curve[-1]["mean_risk_reduction"] if learning_curve else 0,
            "final_cr": learning_curve[-1]["mean_cr"] if learning_curve else 1,
            "exciter_trained": improved if len(learning_curve) >= 2 else False,
        }, f, indent=2, default=str)

    print(f"\nv6 complete. Results in {v6_dir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="SmolLM2-135M")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run_v6(a.model, a.episodes, a.device)

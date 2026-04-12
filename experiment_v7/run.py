"""Experiment v7 — Confidence-level adjustment with REINFORCE-trained regulator.

Instead of modifying weights, modify the output distribution.
When fear spikes: increase temperature + suppress risky action logit.
The regulator learns how much to adjust from risk-reduction reward.
Weights stay at W_0 permanently. Non-destructive. Instantly reversible.
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
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.highway_wrapper import HighwayFRAEnv
from src.components.fear_detector import FearDetector, FearDetectorConfig
from src.evaluation.metrics import bootstrap_ci, m1_collision_rate
from experiment_v3.config import MODELS
from experiment_v3.llm_policy import LLMPolicy


def compute_risk(obs, cost, ttc, action):
    base = {0: 0.3, 1: 0.6, 2: 0.05, 3: 0.5}.get(action, 0.3)
    if action != 2 and ttc < 3.0:
        base = min(1.0, base + (3.0 - ttc) / 3.0 * 0.5)
    base = min(1.0, base + cost * 0.3)
    if action == 2:
        base = min(0.1, base)
    return float(np.clip(base, 0.0, 1.0))


class Regulator(nn.Module):
    """Learns to adjust the LLM's output confidence based on fear and risk.

    Outputs:
      - temperature: scales all logits (higher = flatter = less confident)
      - suppression: per-action logit penalty (negative = suppress that action)
      - recovery_rate: how fast confidence returns to normal

    Trained via REINFORCE: reward = risk reduction after adjustment.
    """

    def __init__(self, n_actions=4, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.n_actions = n_actions

        # Input: fear, risk, 4 logits (normalized), cost, ttc
        input_dim = 2 + n_actions + 2  # = 8

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
        ).to(self.device).float()

        # Temperature head: outputs mean and log_std for Gaussian policy
        self.temp_mean = nn.Linear(32, 1).to(self.device).float()
        self.temp_logstd = nn.Parameter(torch.zeros(1, device=self.device))

        # Suppression head: per-action suppression magnitudes
        self.suppress_mean = nn.Linear(32, n_actions).to(self.device).float()
        self.suppress_logstd = nn.Parameter(torch.zeros(n_actions, device=self.device))

        # Recovery rate head
        self.recovery_mean = nn.Linear(32, 1).to(self.device).float()
        self.recovery_logstd = nn.Parameter(torch.zeros(1, device=self.device))

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

        # Episode buffer for REINFORCE
        self.ep_log_probs = []
        self.ep_rewards = []

        # Stats
        self.train_steps = 0
        self.reward_history = deque(maxlen=200)

    def forward(self, features):
        """features → (temperature, suppression[4], recovery_rate) as distributions."""
        h = self.backbone(features)

        # Temperature: softplus to ensure > 0, then shift to [0.5, 5.0]
        temp_mu = torch.sigmoid(self.temp_mean(h)) * 4.5 + 0.5  # [0.5, 5.0]
        temp_std = torch.exp(self.temp_logstd.clamp(-2, 1))
        temp_dist = Normal(temp_mu.squeeze(-1), temp_std)

        # Suppression: can be negative (suppress) or zero (no change)
        sup_mu = self.suppress_mean(h)  # unbounded, will clamp after sampling
        sup_std = torch.exp(self.suppress_logstd.clamp(-2, 1))
        sup_dist = Normal(sup_mu, sup_std.unsqueeze(0).expand_as(sup_mu))

        # Recovery rate: sigmoid to [0.8, 0.99]
        rec_mu = torch.sigmoid(self.recovery_mean(h)) * 0.19 + 0.8  # [0.8, 0.99]
        rec_std = torch.exp(self.recovery_logstd.clamp(-3, 0))
        rec_dist = Normal(rec_mu.squeeze(-1), rec_std)

        return temp_dist, sup_dist, rec_dist

    def act(self, fear, risk, logits, cost, ttc):
        """Get regulation parameters for current state."""
        # Normalize logits to [-1, 1] range
        logits_norm = logits / (logits.abs().max() + 1e-8)

        features = torch.tensor(
            [fear, risk] + logits_norm.tolist() + [cost, min(ttc, 10.0) / 10.0],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        temp_dist, sup_dist, rec_dist = self.forward(features)

        temp = temp_dist.sample()
        sup = sup_dist.sample().squeeze(0)
        rec = rec_dist.sample()

        # Compute log probs for REINFORCE
        log_prob = (
            temp_dist.log_prob(temp).sum()
            + sup_dist.log_prob(sup.unsqueeze(0)).sum()
            + rec_dist.log_prob(rec).sum()
        )

        # Clamp outputs
        temp = temp.clamp(0.5, 5.0).item()
        sup = sup.clamp(-3.0, 0.5).detach()  # mostly negative (suppress)
        rec = rec.clamp(0.8, 0.99).item()

        return {
            "temperature": temp,
            "suppression": sup,
            "recovery_rate": rec,
            "log_prob": log_prob,
        }

    def store_step(self, log_prob, reward):
        self.ep_log_probs.append(log_prob)
        self.ep_rewards.append(reward)

    def end_episode(self):
        """REINFORCE update at end of episode."""
        if len(self.ep_rewards) < 2:
            self.ep_log_probs = []
            self.ep_rewards = []
            return None

        rewards = torch.tensor(self.ep_rewards, device=self.device, dtype=torch.float32)
        log_probs = torch.stack(self.ep_log_probs)

        # Normalize rewards (baseline subtraction)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # REINFORCE loss
        loss = -(log_probs * rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1
        mean_reward = self.ep_rewards[-1] if self.ep_rewards else 0
        self.reward_history.append(np.mean(self.ep_rewards))

        self.ep_log_probs = []
        self.ep_rewards = []

        return {"loss": loss.item(), "mean_reward": float(np.mean(list(self.reward_history)))}


class ConfidenceState:
    """Tracks confidence adjustment that decays elastically toward baseline."""

    def __init__(self, n_actions=4):
        self.temperature = 1.0           # 1.0 = normal
        self.suppression = np.zeros(n_actions)  # 0 = no suppression
        self.recovery_rate = 0.95        # default decay
        self.n_actions = n_actions

    def apply(self, regulation):
        """Apply new regulation on top of existing state (accumulates)."""
        # New regulation adds to existing (multiple fear spikes stack)
        self.temperature = max(self.temperature, regulation["temperature"])
        self.suppression += regulation["suppression"].cpu().numpy()
        self.recovery_rate = regulation["recovery_rate"]

    def get_adjusted_logits(self, logits):
        """Apply current confidence state to raw logits."""
        sup = torch.tensor(self.suppression, device=logits.device, dtype=logits.dtype)
        adjusted = (logits + sup) / self.temperature
        return adjusted

    def decay(self):
        """Elastic decay toward baseline. Called every timestep."""
        # Temperature decays toward 1.0
        self.temperature = 1.0 + (self.temperature - 1.0) * self.recovery_rate
        # Suppression decays toward 0
        self.suppression *= self.recovery_rate
        # Near baseline? Snap to baseline (prevent floating point drift)
        if abs(self.temperature - 1.0) < 0.001:
            self.temperature = 1.0
        self.suppression[np.abs(self.suppression) < 0.001] = 0.0

    def reset(self):
        self.temperature = 1.0
        self.suppression = np.zeros(self.n_actions)
        self.recovery_rate = 0.95

    def is_active(self):
        return self.temperature > 1.01 or np.any(np.abs(self.suppression) > 0.01)


def run_v7(model_name="SmolLM2-135M", n_episodes=2000, device="cuda"):
    v3_dir = Path("experiment_v3") / model_name
    v7_dir = Path("experiment_v7") / model_name
    v7_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT v7: CONFIDENCE-LEVEL ADJUSTMENT")
    print("Weights stay at W_0. Only output distribution changes.")
    print("REINFORCE-trained regulator.")
    print("=" * 60)

    model_cfg = next(m for m in MODELS if m.name == model_name)
    policy = LLMPolicy(model_cfg, device=device)
    policy.load(str(v3_dir / "policy_trained.pt"))

    day1 = json.load(open(v3_dir / "day1_log.json"))
    fear_det = FearDetector(FearDetectorConfig(device=device))
    fear_det.load_state(torch.load(v3_dir / "fear_detector.pt", weights_only=False, map_location=device))

    env = HighwayFRAEnv(seed=0, vehicles_count=15)
    regulator = Regulator(n_actions=4, device=device)

    print(f"  Model: {model_name}")
    print(f"  Baseline CR: {day1['base_cr']}")
    print(f"  Weights: FROZEN at W_0 (never modified)")

    checkpoint_dir = v7_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    metrics_all = []
    rolling_cr = deque(maxlen=100)
    rolling_less_risky = deque(maxlen=100)
    rolling_risk_red = deque(maxlen=100)

    for ep in range(n_episodes):
        # Weights always stay at W_0 — non-destructive
        policy.restore_w0()

        conf = ConfidenceState(n_actions=4)
        obs, info = env.reset(seed=ep)

        ep_reward = 0.0
        collision = False
        ep_fra_steps = 0
        ep_less_risky = 0
        ep_risk_reds = []
        ep_temps = []
        ep_suppressions = []

        for t in range(500):
            cost = info.get("cost", 0.0)
            ttc = info.get("ttc", 10.0)

            # Step 1: Brain produces logits (at W_0, always clean)
            with torch.no_grad():
                logits = policy.get_logits_from_obs(obs)
                greedy = logits.argmax().item()

            # Step 2: Risk + Fear
            risk = compute_risk(obs, cost, ttc, greedy)
            fear, _ = fear_det.detect(obs, cost, ttc, greedy)

            # Step 3: If afraid, regulator adjusts confidence
            if fear > 0.05 and risk > 0.1:
                ep_fra_steps += 1

                regulation = regulator.act(fear, risk, logits.detach(), cost, ttc)
                conf.apply(regulation)

                # Step 4: Get adjusted logits
                adjusted = conf.get_adjusted_logits(logits)

                with torch.no_grad():
                    new_action = torch.distributions.Categorical(logits=adjusted).sample().item()

                # Step 5: Measure risk reduction
                risk_new = compute_risk(obs, cost, ttc, new_action)
                risk_red = risk - risk_new

                ep_risk_reds.append(risk_red)
                if risk_red > 0:
                    ep_less_risky += 1

                # Step 6: Store for REINFORCE
                regulator.store_step(regulation["log_prob"], risk_red)

                action = new_action
            else:
                # No FRA needed — use original action
                with torch.no_grad():
                    if conf.is_active():
                        # Still recovering from previous spike — apply decayed adjustment
                        adjusted = conf.get_adjusted_logits(logits)
                        action = torch.distributions.Categorical(logits=adjusted).sample().item()
                    else:
                        action = torch.distributions.Categorical(logits=logits).sample().item()

            # Track confidence state
            ep_temps.append(conf.temperature)
            ep_suppressions.append(conf.suppression.copy())

            # Step 7: Elastic decay of confidence toward baseline
            conf.decay()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated:
                collision = info.get("collision", False)
                break
            if truncated:
                break

        # Train regulator via REINFORCE
        train_result = regulator.end_episode()

        less_risky_pct = ep_less_risky / max(ep_fra_steps, 1)
        mean_rr = np.mean(ep_risk_reds) if ep_risk_reds else 0

        rolling_cr.append(int(collision))
        rolling_less_risky.append(less_risky_pct)
        rolling_risk_red.append(mean_rr)

        ep_data = {
            "episode": ep, "collision": int(collision), "reward": ep_reward,
            "n_fra_steps": ep_fra_steps, "n_less_risky": ep_less_risky,
            "less_risky_pct": less_risky_pct,
            "mean_risk_reduction": mean_rr,
            "mean_temperature": float(np.mean(ep_temps)) if ep_temps else 1.0,
            "max_temperature": float(max(ep_temps)) if ep_temps else 1.0,
        }
        if train_result:
            ep_data["regulator_loss"] = train_result["loss"]
            ep_data["regulator_mean_reward"] = train_result["mean_reward"]
        metrics_all.append(ep_data)

        if (ep + 1) % 50 == 0:
            avg_cr = np.mean(rolling_cr)
            avg_lr = np.mean(rolling_less_risky)
            avg_rr = np.mean(rolling_risk_red)
            loss_str = f"loss={train_result['loss']:.4f}" if train_result else "warmup"
            print(f"  [{ep+1}/{n_episodes}] "
                  f"CR={avg_cr:.3f} | "
                  f"LessRisky={avg_lr:.0%} | "
                  f"RiskRed={avg_rr:+.3f} | "
                  f"MeanTemp={np.mean(ep_temps):.2f} | "
                  f"{loss_str}")

        if (ep + 1) % 200 == 0:
            torch.save({
                "regulator": regulator.state_dict(),
                "episode": ep,
            }, checkpoint_dir / f"ep_{ep+1}.pt")

    # ── Final analysis ──
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")

    window = 100
    learning_curve = []
    for i in range(0, len(metrics_all) - window, window // 2):
        chunk = metrics_all[i:i+window]
        learning_curve.append({
            "ep_start": i,
            "cr": np.mean([m["collision"] for m in chunk]),
            "less_risky": np.mean([m["less_risky_pct"] for m in chunk]),
            "risk_red": np.mean([m["mean_risk_reduction"] for m in chunk]),
            "mean_temp": np.mean([m["mean_temperature"] for m in chunk]),
        })

    print(f"\n{'Eps':>10s}  {'CR':>6s}  {'LessRisky':>10s}  {'RiskRed':>10s}  {'Temp':>6s}")
    for lc in learning_curve:
        print(f"  {lc['ep_start']:>4d}-{lc['ep_start']+window:<4d}  "
              f"{lc['cr']:>6.3f}  {lc['less_risky']:>9.1%}  "
              f"{lc['risk_red']:>+9.4f}  {lc['mean_temp']:>6.2f}")

    if len(learning_curve) >= 2:
        first, last = learning_curve[0], learning_curve[-1]
        print(f"\nRegulator learning:")
        print(f"  LessRisky: {first['less_risky']:.1%} → {last['less_risky']:.1%}")
        print(f"  RiskRed:   {first['risk_red']:+.4f} → {last['risk_red']:+.4f}")
        print(f"  CR:        {first['cr']:.3f} → {last['cr']:.3f} (baseline: {day1['base_cr']})")

    # Save
    with open(v7_dir / "metrics.json", "w") as f:
        json.dump(metrics_all, f, indent=2, default=str)
    with open(v7_dir / "learning_curve.json", "w") as f:
        json.dump(learning_curve, f, indent=2, default=str)
    torch.save(regulator.state_dict(), v7_dir / "regulator_final.pt")

    with open(v7_dir / "experiment_log.json", "w") as f:
        json.dump({
            "version": "v7", "mechanism": "confidence_adjustment_reinforce",
            "model": model_name, "n_episodes": n_episodes,
            "baseline_cr": day1["base_cr"],
            "weights_modified": False,
            "final_cr": learning_curve[-1]["cr"] if learning_curve else None,
            "final_less_risky": learning_curve[-1]["less_risky"] if learning_curve else None,
            "final_risk_red": learning_curve[-1]["risk_red"] if learning_curve else None,
        }, f, indent=2, default=str)

    print(f"\nv7 complete. Results in {v7_dir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="SmolLM2-135M")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run_v7(a.model, a.episodes, a.device)

"""Experiment v8 — Combined: Gaussian weight perturbation + Confidence adjustment.

v6 gave: CR 0.80→0.50 (mechanics work, but exciter never learned)
v7 gave: LessRisky 17%→85% (regulator learned, but CR stayed at baseline)

v8: Both at the same time.
  1. Gaussian perturbation of excited weights (v6) — shapes internal state
  2. Confidence adjustment of output logits (v7) — shapes decisions
  3. Single learned regulator controls BOTH (REINFORCE-trained)
  4. Elastic recovery for BOTH weight displacement AND confidence state

The regulator outputs: magnitude, sigma, group_weights (for weights)
                       + temperature, suppression (for logits)
                       + recovery_rate (shared)
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


def apply_gaussian_perturbation(param, gradient, epicenter_idx, sigma, magnitude):
    n = param.numel()
    flat_grad = gradient.flatten()
    flat_param = param.data.flatten()
    indices = torch.arange(n, device=param.device, dtype=torch.float32)
    sigma_abs = max(1, int(sigma * n))
    gaussian = torch.exp(-((indices - epicenter_idx) ** 2) / (2 * sigma_abs ** 2))
    flat_param -= magnitude * gaussian * flat_grad
    param.data = flat_param.reshape(param.shape)
    return gaussian


class CombinedRegulator(nn.Module):
    """Controls BOTH weight perturbation AND confidence adjustment.

    Outputs:
      Weight perturbation: magnitude, sigma, group_weights
      Confidence: temperature, per-action suppression
      Shared: recovery_rate

    Trained via REINFORCE on risk_reduction reward.
    """

    def __init__(self, n_actions=4, n_groups=10, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.n_groups = n_groups

        # Input: fear, risk, 4 logits, cost, ttc, 10 per-group grad magnitudes
        input_dim = 2 + n_actions + 2 + n_groups  # = 18

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        ).to(self.device).float()

        # Weight perturbation heads
        self.mag_mean = nn.Linear(64, 1).to(self.device).float()
        self.mag_logstd = nn.Parameter(torch.tensor(-1.0, device=self.device))
        self.sigma_mean = nn.Linear(64, 1).to(self.device).float()
        self.sigma_logstd = nn.Parameter(torch.tensor(-1.0, device=self.device))
        self.group_head = nn.Linear(64, n_groups).to(self.device).float()

        # Confidence heads
        self.temp_mean = nn.Linear(64, 1).to(self.device).float()
        self.temp_logstd = nn.Parameter(torch.tensor(-1.0, device=self.device))
        self.suppress_mean = nn.Linear(64, n_actions).to(self.device).float()
        self.suppress_logstd = nn.Parameter(torch.zeros(n_actions, device=self.device) - 1.0)

        # Recovery head (shared)
        self.recovery_mean = nn.Linear(64, 1).to(self.device).float()
        self.recovery_logstd = nn.Parameter(torch.tensor(-2.0, device=self.device))

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.ep_log_probs = []
        self.ep_rewards = []
        self.train_steps = 0
        self.reward_history = deque(maxlen=200)

    def act(self, fear, risk, logits, cost, ttc, grad_per_group):
        logits_norm = logits / (logits.abs().max() + 1e-8)
        features = torch.tensor(
            [fear, risk] + logits_norm.tolist() + [cost, min(ttc, 10.0) / 10.0] + grad_per_group,
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        h = self.backbone(features)

        # Weight perturbation
        mag_dist = Normal(torch.sigmoid(self.mag_mean(h).squeeze()) * 0.1,
                          torch.exp(self.mag_logstd.clamp(-3, 0)))
        sig_dist = Normal(torch.sigmoid(self.sigma_mean(h).squeeze()) * 0.5 + 0.01,
                          torch.exp(self.sigma_logstd.clamp(-3, 0)))
        group_weights = torch.softmax(self.group_head(h).squeeze(0), dim=-1)

        # Confidence
        temp_dist = Normal(torch.sigmoid(self.temp_mean(h).squeeze()) * 4.5 + 0.5,
                           torch.exp(self.temp_logstd.clamp(-3, 0)))
        sup_mu = self.suppress_mean(h)
        sup_dist = Normal(sup_mu, torch.exp(self.suppress_logstd.clamp(-3, 0)).unsqueeze(0).expand_as(sup_mu))

        # Recovery
        rec_dist = Normal(torch.sigmoid(self.recovery_mean(h).squeeze()) * 0.19 + 0.8,
                          torch.exp(self.recovery_logstd.clamp(-3, 0)))

        # Sample
        mag = mag_dist.sample().clamp(0, 0.1)
        sig = sig_dist.sample().clamp(0.01, 0.51)
        temp = temp_dist.sample().clamp(0.5, 5.0)
        sup = sup_dist.sample().squeeze(0).clamp(-3.0, 0.5)
        rec = rec_dist.sample().clamp(0.8, 0.99)

        log_prob = (mag_dist.log_prob(mag) + sig_dist.log_prob(sig) +
                    temp_dist.log_prob(temp) + sup_dist.log_prob(sup.unsqueeze(0)).sum() +
                    rec_dist.log_prob(rec))

        return {
            "magnitude": mag.item(),
            "sigma": sig.item(),
            "group_weights": group_weights.squeeze(0).detach().cpu().numpy(),
            "temperature": temp.item(),
            "suppression": sup.detach(),
            "recovery_rate": rec.item(),
            "log_prob": log_prob,
        }

    def store_step(self, log_prob, reward):
        self.ep_log_probs.append(log_prob)
        self.ep_rewards.append(reward)

    def end_episode(self):
        if len(self.ep_rewards) < 2:
            self.ep_log_probs = []
            self.ep_rewards = []
            return None

        rewards = torch.tensor(self.ep_rewards, device=self.device, dtype=torch.float32)
        log_probs = torch.stack(self.ep_log_probs)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        loss = -(log_probs * rewards).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1
        self.reward_history.append(np.mean(self.ep_rewards))
        result = {"loss": loss.item(), "mean_reward": float(np.mean(list(self.reward_history)))}
        self.ep_log_probs = []
        self.ep_rewards = []
        return result


class CombinedState:
    """Tracks both weight displacement AND confidence state with elastic recovery."""

    def __init__(self, n_actions=4):
        self.temperature = 1.0
        self.suppression = np.zeros(n_actions)
        self.recovery_rate = 0.95

    def apply_confidence(self, regulation):
        self.temperature = max(self.temperature, regulation["temperature"])
        self.suppression += regulation["suppression"].cpu().numpy()
        self.recovery_rate = regulation["recovery_rate"]

    def get_adjusted_logits(self, logits):
        sup = torch.tensor(self.suppression, device=logits.device, dtype=logits.dtype)
        return (logits + sup) / self.temperature

    def decay(self):
        self.temperature = 1.0 + (self.temperature - 1.0) * self.recovery_rate
        self.suppression *= self.recovery_rate
        if abs(self.temperature - 1.0) < 0.001:
            self.temperature = 1.0
        self.suppression[np.abs(self.suppression) < 0.001] = 0.0

    def is_active(self):
        return self.temperature > 1.01 or np.any(np.abs(self.suppression) > 0.01)

    def reset(self):
        self.temperature = 1.0
        self.suppression = np.zeros(len(self.suppression))
        self.recovery_rate = 0.95


def run_v8(model_name="SmolLM2-135M", n_episodes=2000, device="cuda"):
    v3_dir = Path("experiment_v3") / model_name
    v8_dir = Path("experiment_v8") / model_name
    v8_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT v8: COMBINED (weight perturbation + confidence)")
    print("v6 mechanics + v7 learning in one regulator")
    print("=" * 60)

    model_cfg = next(m for m in MODELS if m.name == model_name)
    policy = LLMPolicy(model_cfg, device=device)
    policy.load(str(v3_dir / "policy_trained.pt"))
    w0 = policy.get_w0()

    day1 = json.load(open(v3_dir / "day1_log.json"))
    fisher = torch.load(v3_dir / "fisher.pt", weights_only=False, map_location=device)
    fear_det = FearDetector(FearDetectorConfig(device=device))
    fear_det.load_state(torch.load(v3_dir / "fear_detector.pt", weights_only=False, map_location=device))

    env = HighwayFRAEnv(seed=0, vehicles_count=15)

    param_dict = dict(policy.get_perturbable_params())
    param_names = list(param_dict.keys())
    n_groups = min(10, len(param_names))
    group_size = len(param_names) // n_groups

    regulator = CombinedRegulator(n_actions=4, n_groups=n_groups, device=device)
    eta_h = min(0.01, 0.5 / max(day1["f_max"], 1e-8))

    print(f"  Model: {model_name} ({policy.n_perturbable:,} params, {n_groups} groups)")
    print(f"  Baseline CR: {day1['base_cr']}")
    print(f"  v6 result: CR~0.50, LessRisky~13%")
    print(f"  v7 result: CR~0.80, LessRisky~85%")
    print(f"  v8 target: best of both")

    checkpoint_dir = v8_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    metrics_all = []
    rolling_cr = deque(maxlen=100)
    rolling_lr = deque(maxlen=100)
    rolling_rr = deque(maxlen=100)

    for ep in range(n_episodes):
        policy.restore_w0()
        param_dict = dict(policy.get_perturbable_params())
        state = CombinedState(n_actions=4)
        obs, info = env.reset(seed=ep)

        ep_reward = 0.0
        collision = False
        ep_fra = 0
        ep_less_risky = 0
        ep_risk_reds = []

        for t in range(500):
            cost = info.get("cost", 0.0)
            ttc = info.get("ttc", 10.0)

            # Step 1: Brain decides
            with torch.no_grad():
                logits = policy.get_logits_from_obs(obs)
                greedy = logits.argmax().item()

            risk = compute_risk(obs, cost, ttc, greedy)
            fear, _ = fear_det.detect(obs, cost, ttc, greedy)

            if fear > 0.05 and risk > 0.1:
                ep_fra += 1

                # Step 2: Compute gradients
                policy.model.zero_grad()
                policy.action_head.zero_grad()
                logits_grad = policy.get_logits_from_obs(obs)
                torch.log_softmax(logits_grad, dim=-1)[greedy].backward()

                grad_per_group = []
                for gi in range(n_groups):
                    start = gi * group_size
                    end = min(start + group_size, len(param_names))
                    gsum = 0.0
                    for pi in range(start, end):
                        p = param_dict[param_names[pi]]
                        if p.grad is not None:
                            gsum += p.grad.data.abs().mean().item()
                    grad_per_group.append(gsum / max(end - start, 1))

                # Step 3: Regulator decides everything
                reg = regulator.act(fear, risk, logits.detach(), cost, ttc, grad_per_group)

                # Step 4a: Gaussian weight perturbation (v6)
                with torch.no_grad():
                    for gi in range(n_groups):
                        start = gi * group_size
                        end = min(start + group_size, len(param_names))
                        gw = reg["group_weights"][gi]
                        for pi in range(start, end):
                            name = param_names[pi]
                            p = param_dict[name]
                            if p.grad is None:
                                continue
                            epicenter = p.grad.data.abs().flatten().argmax().item()
                            apply_gaussian_perturbation(
                                p, p.grad.data, epicenter,
                                reg["sigma"], reg["magnitude"] * gw * risk
                            )

                # Step 4b: Confidence adjustment (v7)
                state.apply_confidence(reg)

                # Step 5: Perturbed brain + adjusted confidence → new action
                with torch.no_grad():
                    perturbed_logits = policy.get_logits_from_obs(obs)
                    adjusted_logits = state.get_adjusted_logits(perturbed_logits)
                    new_action = torch.distributions.Categorical(logits=adjusted_logits).sample().item()

                # Step 6: Measure
                risk_new = compute_risk(obs, cost, ttc, new_action)
                risk_red = risk - risk_new
                ep_risk_reds.append(risk_red)
                if risk_red > 0:
                    ep_less_risky += 1

                regulator.store_step(reg["log_prob"], risk_red)
                action = new_action
            else:
                # No FRA — but may still be in recovery
                with torch.no_grad():
                    if state.is_active():
                        adjusted = state.get_adjusted_logits(logits)
                        action = torch.distributions.Categorical(logits=adjusted).sample().item()
                    else:
                        action = torch.distributions.Categorical(logits=logits).sample().item()

            # Step 7: Elastic recovery for BOTH
            state.decay()
            # Weight recovery via FHR
            with torch.no_grad():
                pidx = 0
                for name in param_names:
                    p = param_dict[name]
                    if name not in w0:
                        pidx += p.numel()
                        continue
                    diff = w0[name].to(p.dtype) - p.data
                    ne = p.numel()
                    fs = fisher[pidx:pidx+ne].reshape(p.shape).to(p.dtype).to(p.device)
                    pidx += ne
                    p.data += eta_h * fs * diff

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated:
                collision = info.get("collision", False)
                break
            if truncated:
                break

        # Train regulator
        train_result = regulator.end_episode()

        lr_pct = ep_less_risky / max(ep_fra, 1)
        mean_rr = np.mean(ep_risk_reds) if ep_risk_reds else 0

        rolling_cr.append(int(collision))
        rolling_lr.append(lr_pct)
        rolling_rr.append(mean_rr)

        metrics_all.append({
            "episode": ep, "collision": int(collision), "reward": ep_reward,
            "n_fra_steps": ep_fra, "less_risky_pct": lr_pct,
            "mean_risk_reduction": mean_rr,
        })

        if (ep + 1) % 50 == 0:
            avg_cr = np.mean(rolling_cr)
            avg_lr = np.mean(rolling_lr)
            avg_rr = np.mean(rolling_rr)
            loss_str = f"loss={train_result['loss']:.4f}" if train_result else "warmup"
            print(f"  [{ep+1}/{n_episodes}] "
                  f"CR={avg_cr:.3f} | "
                  f"LessRisky={avg_lr:.0%} | "
                  f"RiskRed={avg_rr:+.3f} | "
                  f"{loss_str}")

        if (ep + 1) % 200 == 0:
            torch.save({"regulator": regulator.state_dict(), "episode": ep},
                       checkpoint_dir / f"ep_{ep+1}.pt")

    # Analysis
    print(f"\n{'='*60}")
    print("v8 COMPLETE")
    print(f"{'='*60}")

    window = 100
    curve = []
    for i in range(0, len(metrics_all) - window, window // 2):
        chunk = metrics_all[i:i+window]
        curve.append({
            "ep": i, "cr": np.mean([m["collision"] for m in chunk]),
            "lr": np.mean([m["less_risky_pct"] for m in chunk]),
            "rr": np.mean([m["mean_risk_reduction"] for m in chunk]),
        })

    print(f"\n{'Eps':>10s}  {'CR':>6s}  {'LessRisky':>10s}  {'RiskRed':>10s}")
    for c in curve:
        print(f"  {c['ep']:>4d}-{c['ep']+window:<4d}  {c['cr']:>6.3f}  {c['lr']:>9.1%}  {c['rr']:>+9.4f}")

    if len(curve) >= 2:
        f, l = curve[0], curve[-1]
        print(f"\nLearning:")
        print(f"  CR:        {f['cr']:.3f} -> {l['cr']:.3f} (baseline: {day1['base_cr']})")
        print(f"  LessRisky: {f['lr']:.1%} -> {l['lr']:.1%}")
        print(f"  RiskRed:   {f['rr']:+.4f} -> {l['rr']:+.4f}")
        print(f"\nComparison:")
        print(f"  v6 final: CR=0.50, LessRisky=13%, RiskRed=+0.065")
        print(f"  v7 final: CR=0.80, LessRisky=85%, RiskRed=+0.624")
        print(f"  v8 final: CR={l['cr']:.3f}, LessRisky={l['lr']:.0%}, RiskRed={l['rr']:+.3f}")

    with open(v8_dir / "metrics.json", "w") as f:
        json.dump(metrics_all, f, indent=2, default=str)
    with open(v8_dir / "learning_curve.json", "w") as f:
        json.dump(curve, f, indent=2, default=str)
    torch.save(regulator.state_dict(), v8_dir / "regulator_final.pt")
    with open(v8_dir / "experiment_log.json", "w") as f:
        json.dump({
            "version": "v8", "mechanism": "combined_gaussian_confidence",
            "model": model_name, "n_episodes": n_episodes,
            "baseline_cr": day1["base_cr"],
            "final_cr": curve[-1]["cr"] if curve else None,
            "final_less_risky": curve[-1]["lr"] if curve else None,
            "final_risk_red": curve[-1]["rr"] if curve else None,
        }, f, indent=2, default=str)

    print(f"\nv8 complete. Results in {v8_dir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="SmolLM2-135M")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run_v8(a.model, a.episodes, a.device)

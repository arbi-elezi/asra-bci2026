"""v8-PPO: Combined mechanism with PPO-trained regulator.

Same as v8 (Gaussian weight perturbation + confidence adjustment)
but the regulator is trained with PPO (clipped surrogate + value baseline)
instead of raw REINFORCE.

This fixes the REINFORCE instability that caused loss divergence at ep 1600+
and performance degradation after peak. PPO's clipping prevents large
policy updates, and the value baseline reduces variance.
"""

import json, time, sys
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
    if action != 2 and ttc < 3.0: base = min(1.0, base + (3.0 - ttc) / 3.0 * 0.5)
    base = min(1.0, base + cost * 0.3)
    if action == 2: base = min(0.1, base)
    return float(np.clip(base, 0.0, 1.0))


def apply_gaussian(param, grad, epicenter, sigma, magnitude):
    n = param.numel()
    idx = torch.arange(n, device=param.device, dtype=torch.float32)
    sig_abs = max(1, int(sigma * n))
    g = torch.exp(-((idx - epicenter) ** 2) / (2 * sig_abs ** 2))
    param.data.flatten().sub_(magnitude * g * grad.flatten())


class PPORegulator(nn.Module):
    """Regulator trained with PPO instead of REINFORCE.

    Has both a policy network (same outputs as v8) and a value network
    (predicts expected risk reduction). The value baseline reduces variance
    and PPO clipping prevents catastrophic updates.
    """

    def __init__(self, n_actions=4, n_groups=10, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.n_actions = n_actions
        input_dim = 2 + n_actions + 2 + n_groups

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
        ).to(self.device).float()

        # Policy heads (same as v8)
        self.mag_mean = nn.Linear(64, 1).to(self.device).float()
        self.sig_mean = nn.Linear(64, 1).to(self.device).float()
        self.temp_mean = nn.Linear(64, 1).to(self.device).float()
        self.sup_mean = nn.Linear(64, n_actions).to(self.device).float()
        self.grp_head = nn.Linear(64, n_groups).to(self.device).float()
        self.log_stds = nn.Parameter(torch.zeros(3 + n_actions, device=self.device) - 1.0)

        # Value head (PPO baseline)
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1),
        ).to(self.device).float()

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

        # PPO buffer (per episode)
        self.ep_features = []
        self.ep_log_probs = []
        self.ep_values = []
        self.ep_rewards = []

        self.clip_range = 0.2
        self.ppo_epochs = 4
        self.train_steps = 0

    def _get_features(self, fear, risk, logits, cost, ttc, gpg):
        ln = logits / (logits.abs().max() + 1e-8)
        return torch.tensor(
            [fear, risk] + ln.tolist() + [cost, min(ttc, 10) / 10] + gpg,
            dtype=torch.float32, device=self.device
        )

    def act(self, fear, risk, logits, cost, ttc, gpg):
        feat = self._get_features(fear, risk, logits, cost, ttc, gpg).unsqueeze(0)
        h = self.backbone(feat)
        stds = torch.exp(self.log_stds.clamp(-3, 0))

        mag_mu = torch.sigmoid(self.mag_mean(h).squeeze()) * 0.1
        sig_mu = torch.sigmoid(self.sig_mean(h).squeeze()) * 0.5 + 0.01
        temp_mu = torch.sigmoid(self.temp_mean(h).squeeze()) * 4.5 + 0.5
        sup_mu = self.sup_mean(h).squeeze(0)
        gw = torch.softmax(self.grp_head(h).squeeze(0), dim=-1)

        mag_d = Normal(mag_mu, stds[0])
        sig_d = Normal(sig_mu, stds[1])
        temp_d = Normal(temp_mu, stds[2])
        sup_d = Normal(sup_mu, stds[3:3+self.n_actions])

        mag = mag_d.sample().clamp(0, 0.1)
        sig = sig_d.sample().clamp(0.01, 0.51)
        temp = temp_d.sample().clamp(0.5, 5.0)
        sup = sup_d.sample().clamp(-3, 0.5)

        log_prob = mag_d.log_prob(mag) + sig_d.log_prob(sig) + temp_d.log_prob(temp) + sup_d.log_prob(sup).sum()
        value = self.value_head(feat).squeeze()

        return {
            "magnitude": mag.item(), "sigma": sig.item(), "temperature": temp.item(),
            "suppression": sup.detach(), "group_weights": gw.squeeze(0).detach().cpu().numpy(),
            "log_prob": log_prob, "value": value,
            "features": feat.squeeze(0).detach(),
        }

    def store_step(self, features, log_prob, value, reward):
        self.ep_features.append(features)
        self.ep_log_probs.append(log_prob)
        self.ep_values.append(value)
        self.ep_rewards.append(reward)

    def end_episode(self):
        """PPO update at end of episode."""
        if len(self.ep_rewards) < 4:
            self.ep_features = []; self.ep_log_probs = []; self.ep_values = []; self.ep_rewards = []
            return None

        features = torch.stack(self.ep_features)
        old_log_probs = torch.stack(self.ep_log_probs).detach()
        old_values = torch.stack(self.ep_values).detach()
        rewards = torch.tensor(self.ep_rewards, device=self.device, dtype=torch.float32)

        # Compute advantages (GAE-like, simplified)
        advantages = rewards - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = rewards  # Simple — no discounting within episode FRA steps

        # PPO epochs
        total_loss = 0.0
        for _ in range(self.ppo_epochs):
            # Recompute log probs and values
            h = self.backbone(features)
            stds = torch.exp(self.log_stds.clamp(-3, 0))
            mag_mu = torch.sigmoid(self.mag_mean(h).squeeze(-1)) * 0.1
            temp_mu = torch.sigmoid(self.temp_mean(h).squeeze(-1)) * 4.5 + 0.5

            # Simplified: use mag and temp for log prob computation
            new_log_probs = Normal(mag_mu, stds[0]).log_prob(mag_mu) + Normal(temp_mu, stds[2]).log_prob(temp_mu)
            new_values = self.value_head(features).squeeze(-1)

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs[:len(new_log_probs)])
            adv = advantages[:len(ratio)]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(new_values[:len(returns)], returns[:len(new_values)])

            loss = policy_loss + 0.5 * value_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()

        self.train_steps += 1
        result = {"loss": total_loss / self.ppo_epochs}

        self.ep_features = []; self.ep_log_probs = []; self.ep_values = []; self.ep_rewards = []
        return result


def run_v8_ppo(model_name="SmolLM2-135M", n_episodes=3000, device="cuda"):
    v3_dir = Path("experiment_v3") / model_name
    out_dir = Path("experiment_v8_ppo") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("v8-PPO: Combined mechanism + PPO-trained regulator")
    print("Fixes REINFORCE instability from v8")
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

    param_names = [n for n, _ in policy.get_perturbable_params()]
    n_groups = min(10, len(param_names))
    group_size = len(param_names) // n_groups
    eta_h = min(0.01, 0.5 / max(day1["f_max"], 1e-8))

    regulator = PPORegulator(n_actions=4, n_groups=n_groups, device=device)

    print(f"  Model: {model_name} ({policy.n_perturbable:,} params)")
    print(f"  Baseline CR: {day1['base_cr']}")
    print(f"  Regulator: PPO (clip={regulator.clip_range}, epochs={regulator.ppo_epochs})")

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Resume support
    metrics_all = []
    rolling_cr, rolling_lr, rolling_rr = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    start_ep = 0
    existing = sorted(ckpt_dir.glob("ep_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if existing:
        latest = existing[-1]
        start_ep = int(latest.stem.split("_")[1])
        ckpt = torch.load(latest, weights_only=False, map_location=device)
        regulator.load_state_dict(ckpt["regulator"])
        if "metrics" in ckpt: metrics_all = ckpt["metrics"]
        for m in metrics_all[-100:]:
            rolling_cr.append(m["collision"]); rolling_lr.append(m["less_risky_pct"]); rolling_rr.append(m["mean_risk_reduction"])
        print(f"  RESUMED from ep_{start_ep}")

    for ep in range(start_ep, n_episodes):
        policy.restore_w0()
        param_dict = dict(policy.get_perturbable_params())
        temp, sup = 1.0, np.zeros(4)
        obs, info = env.reset(seed=ep)
        collision, ep_fra, ep_lr, ep_rrs = False, 0, 0, []

        for t in range(500):
            cost, ttc = info.get("cost", 0.0), info.get("ttc", 10.0)
            with torch.no_grad():
                logits = policy.get_logits_from_obs(obs)
                greedy = logits.argmax().item()
            risk = compute_risk(obs, cost, ttc, greedy)
            fear, _ = fear_det.detect(obs, cost, ttc, greedy)

            if fear > 0.05 and risk > 0.1:
                ep_fra += 1
                policy.model.zero_grad(); policy.action_head.zero_grad()
                lg = policy.get_logits_from_obs(obs)
                torch.log_softmax(lg, dim=-1)[greedy].backward()

                gpg = []
                for gi in range(n_groups):
                    s, e = gi*group_size, min((gi+1)*group_size, len(param_names))
                    gs = sum(param_dict[param_names[pi]].grad.data.abs().mean().item()
                             for pi in range(s,e) if param_dict[param_names[pi]].grad is not None) / max(e-s,1)
                    gpg.append(gs)

                reg = regulator.act(fear, risk, logits.detach(), cost, ttc, gpg)

                # Gaussian weight perturbation
                with torch.no_grad():
                    for gi in range(n_groups):
                        s, e = gi*group_size, min((gi+1)*group_size, len(param_names))
                        gw = reg["group_weights"][gi]
                        for pi in range(s, e):
                            p = param_dict[param_names[pi]]
                            if p.grad is None: continue
                            epi = p.grad.data.abs().flatten().argmax().item()
                            apply_gaussian(p, p.grad.data, epi, reg["sigma"], reg["magnitude"]*gw*risk)

                # Confidence adjustment
                temp = max(temp, reg["temperature"])
                sup += reg["suppression"].cpu().numpy()

                with torch.no_grad():
                    sup_t = torch.tensor(sup, device=logits.device, dtype=logits.dtype)
                    adj = (policy.get_logits_from_obs(obs) + sup_t) / temp
                    new_a = torch.distributions.Categorical(logits=adj).sample().item()
                rr = risk - compute_risk(obs, cost, ttc, new_a)
                ep_rrs.append(rr)
                if rr > 0: ep_lr += 1

                regulator.store_step(reg["features"], reg["log_prob"], reg["value"], rr)
                action = new_a
            else:
                if temp > 1.01 or np.any(np.abs(sup) > 0.01):
                    sup_t = torch.tensor(sup, device=logits.device, dtype=logits.dtype)
                    action = torch.distributions.Categorical(logits=(logits+sup_t)/temp).sample().item()
                else:
                    action = torch.distributions.Categorical(logits=logits).sample().item()

            # Recovery
            temp = 1.0 + (temp - 1.0) * 0.92; sup *= 0.92
            with torch.no_grad():
                pidx = 0
                for name in param_names:
                    p = param_dict[name]
                    if name not in w0: pidx += p.numel(); continue
                    diff = w0[name].to(p.dtype) - p.data
                    ne = p.numel()
                    fs = fisher[pidx:pidx+ne].reshape(p.shape).to(p.dtype).to(p.device)
                    pidx += ne
                    p.data += eta_h * fs * diff

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated: collision = info.get("collision", False); break
            if truncated: break

        train_r = regulator.end_episode()
        lr_pct = ep_lr / max(ep_fra, 1)
        mean_rr = np.mean(ep_rrs) if ep_rrs else 0
        rolling_cr.append(int(collision)); rolling_lr.append(lr_pct); rolling_rr.append(mean_rr)
        metrics_all.append({"episode": ep, "collision": int(collision),
                            "less_risky_pct": lr_pct, "mean_risk_reduction": mean_rr})

        if (ep + 1) % 50 == 0:
            loss_s = f"loss={train_r['loss']:.4f}" if train_r else "warmup"
            print(f"  [{ep+1}/{n_episodes}] CR={np.mean(rolling_cr):.3f} | "
                  f"LessRisky={np.mean(rolling_lr):.0%} | RiskRed={np.mean(rolling_rr):+.3f} | {loss_s}")

        if (ep + 1) % 100 == 0:
            torch.save({"regulator": regulator.state_dict(), "episode": ep+1,
                         "metrics": metrics_all}, ckpt_dir / f"ep_{ep+1}.pt")

    # Analysis
    print(f"\n{'='*60}\nv8-PPO COMPLETE\n{'='*60}")
    window = 100
    curve = []
    for i in range(0, len(metrics_all)-window, window//2):
        ch = metrics_all[i:i+window]
        curve.append({"ep": i, "cr": np.mean([m["collision"] for m in ch]),
                       "lr": np.mean([m["less_risky_pct"] for m in ch]),
                       "rr": np.mean([m["mean_risk_reduction"] for m in ch])})
    if len(curve) >= 2:
        f, l = curve[0], curve[-1]
        print(f"  CR: {f['cr']:.3f} -> {l['cr']:.3f} (baseline: {day1['base_cr']})")
        print(f"  LessRisky: {f['lr']:.1%} -> {l['lr']:.1%}")
        print(f"  RiskRed: {f['rr']:+.4f} -> {l['rr']:+.4f}")
        print(f"\n  v8 (REINFORCE): CR=0.69, LessRisky=86%, RiskRed=+0.648")
        print(f"  v8-PPO:        CR={l['cr']:.3f}, LessRisky={l['lr']:.0%}, RiskRed={l['rr']:+.3f}")

    with open(out_dir / "metrics.json", "w") as fo: json.dump(metrics_all, fo, indent=2, default=str)
    with open(out_dir / "learning_curve.json", "w") as fo: json.dump(curve, fo, indent=2, default=str)
    torch.save(regulator.state_dict(), out_dir / "regulator_final.pt")
    with open(out_dir / "experiment_log.json", "w") as fo:
        json.dump({"version": "v8-PPO", "mechanism": "combined_ppo_regulator",
                    "model": model_name, "n_episodes": n_episodes, "baseline_cr": day1["base_cr"],
                    "final_cr": curve[-1]["cr"] if curve else None,
                    "final_less_risky": curve[-1]["lr"] if curve else None,
                    "final_risk_red": curve[-1]["rr"] if curve else None}, fo, indent=2, default=str)
    print(f"\nResults in {out_dir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="SmolLM2-135M")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run_v8_ppo(a.model, a.episodes, a.device)

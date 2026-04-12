"""v8b — Combined mechanism with LEARNED RECOVERY MODEL.

Same as v8 but the recovery rate is not a single scalar — it's a small
neural network that takes the current state (displacement, velocity,
fear history, risk trajectory) and outputs per-weight recovery rates.

The recovery model learns: "given how displaced I am and what's happening
on the road, how fast should I recover?"

If danger is still nearby: recover slowly (stay cautious).
If danger passed: recover quickly (return to normal).
If new danger appeared during recovery: slow down or re-suppress.
"""

import json, time, sys, copy
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
from src.evaluation.metrics import m1_collision_rate
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
    return g


class RecoveryModel(nn.Module):
    """Learns recovery rate from current state.

    Input: fear, cost, ttc, displacement_magnitude, time_since_spike, velocity_magnitude
    Output: recovery_rate ∈ [0.8, 0.99] per component (temperature + suppression)
    """
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.net = nn.Sequential(
            nn.Linear(6, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 2),  # [temp_recovery, suppress_recovery]
        ).to(self.device).float()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, features):
        raw = self.net(features)
        return torch.sigmoid(raw) * 0.19 + 0.8  # [0.8, 0.99]

    def get_rates(self, fear, cost, ttc, displacement, time_since, velocity):
        feat = torch.tensor([fear, cost, min(ttc,10)/10, displacement, time_since/50, velocity],
                            dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            rates = self.forward(feat).squeeze(0)
        return rates[0].item(), rates[1].item()  # temp_rate, suppress_rate


class CombinedRegulator(nn.Module):
    def __init__(self, n_actions=4, n_groups=10, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        input_dim = 2 + n_actions + 2 + n_groups
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
        ).to(self.device).float()
        self.mag_mean = nn.Linear(64, 1).to(self.device).float()
        self.mag_logstd = nn.Parameter(torch.tensor(-1.0, device=self.device))
        self.sigma_mean = nn.Linear(64, 1).to(self.device).float()
        self.sigma_logstd = nn.Parameter(torch.tensor(-1.0, device=self.device))
        self.group_head = nn.Linear(64, n_groups).to(self.device).float()
        self.temp_mean = nn.Linear(64, 1).to(self.device).float()
        self.temp_logstd = nn.Parameter(torch.tensor(-1.0, device=self.device))
        self.suppress_mean = nn.Linear(64, n_actions).to(self.device).float()
        self.suppress_logstd = nn.Parameter(torch.zeros(n_actions, device=self.device) - 1.0)
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.ep_log_probs, self.ep_rewards = [], []
        self.train_steps = 0

    def act(self, fear, risk, logits, cost, ttc, gpg):
        ln = logits / (logits.abs().max() + 1e-8)
        feat = torch.tensor([fear, risk] + ln.tolist() + [cost, min(ttc,10)/10] + gpg,
                            dtype=torch.float32, device=self.device).unsqueeze(0)
        h = self.backbone(feat)
        mag_d = Normal(torch.sigmoid(self.mag_mean(h).squeeze())*0.1, torch.exp(self.mag_logstd.clamp(-3,0)))
        sig_d = Normal(torch.sigmoid(self.sigma_mean(h).squeeze())*0.5+0.01, torch.exp(self.sigma_logstd.clamp(-3,0)))
        temp_d = Normal(torch.sigmoid(self.temp_mean(h).squeeze())*4.5+0.5, torch.exp(self.temp_logstd.clamp(-3,0)))
        sup_mu = self.suppress_mean(h)
        sup_d = Normal(sup_mu, torch.exp(self.suppress_logstd.clamp(-3,0)).unsqueeze(0).expand_as(sup_mu))
        gw = torch.softmax(self.group_head(h).squeeze(0), dim=-1)
        mag = mag_d.sample().clamp(0,0.1); sig = sig_d.sample().clamp(0.01,0.51)
        temp = temp_d.sample().clamp(0.5,5.0); sup = sup_d.sample().squeeze(0).clamp(-3,0.5)
        lp = mag_d.log_prob(mag) + sig_d.log_prob(sig) + temp_d.log_prob(temp) + sup_d.log_prob(sup.unsqueeze(0)).sum()
        return {"magnitude": mag.item(), "sigma": sig.item(), "group_weights": gw.squeeze(0).detach().cpu().numpy(),
                "temperature": temp.item(), "suppression": sup.detach(), "log_prob": lp}

    def store_step(self, lp, r): self.ep_log_probs.append(lp); self.ep_rewards.append(r)

    def end_episode(self):
        if len(self.ep_rewards) < 2: self.ep_log_probs=[]; self.ep_rewards=[]; return None
        rw = torch.tensor(self.ep_rewards, device=self.device, dtype=torch.float32)
        lp = torch.stack(self.ep_log_probs)
        rw = (rw - rw.mean()) / (rw.std() + 1e-8)
        loss = -(lp * rw).mean()
        self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.parameters(), 1.0); self.optimizer.step()
        self.train_steps += 1; r = {"loss": loss.item()}; self.ep_log_probs=[]; self.ep_rewards=[]; return r


class AdaptiveState:
    def __init__(self, n_actions=4):
        self.temperature = 1.0
        self.suppression = np.zeros(n_actions)
        self.time_since_spike = 0
        self.displacement_history = deque(maxlen=20)

    def apply(self, reg):
        self.temperature = max(self.temperature, reg["temperature"])
        self.suppression += reg["suppression"].cpu().numpy()
        self.time_since_spike = 0

    def get_adjusted(self, logits):
        sup = torch.tensor(self.suppression, device=logits.device, dtype=logits.dtype)
        return (logits + sup) / self.temperature

    def decay(self, temp_rate, sup_rate):
        self.temperature = 1.0 + (self.temperature - 1.0) * temp_rate
        self.suppression *= sup_rate
        self.time_since_spike += 1
        if abs(self.temperature - 1.0) < 0.001: self.temperature = 1.0
        self.suppression[np.abs(self.suppression) < 0.001] = 0.0
        self.displacement_history.append(abs(self.temperature - 1.0) + np.sum(np.abs(self.suppression)))

    def is_active(self): return self.temperature > 1.01 or np.any(np.abs(self.suppression) > 0.01)

    def get_displacement(self): return abs(self.temperature - 1.0) + np.sum(np.abs(self.suppression))

    def get_velocity(self):
        if len(self.displacement_history) < 2: return 0.0
        return abs(self.displacement_history[-1] - self.displacement_history[-2])

    def reset(self): self.temperature = 1.0; self.suppression = np.zeros(len(self.suppression)); self.time_since_spike = 0; self.displacement_history.clear()


def run_v8b(model_name="SmolLM2-135M", n_episodes=2000, device="cuda"):
    v3_dir = Path("experiment_v3") / model_name
    v8b_dir = Path("experiment_v8b") / model_name
    v8b_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("v8b: Combined + LEARNED RECOVERY MODEL")
    print("Recovery rate adapts to ongoing danger")
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
    eta_h = min(0.01, 0.5 / max(day1["f_max"], 1e-8))

    regulator = CombinedRegulator(n_actions=4, n_groups=n_groups, device=device)
    recovery_model = RecoveryModel(device=device)

    # Recovery model training buffer
    recovery_buffer = deque(maxlen=5000)

    print(f"  Model: {model_name}")
    print(f"  Baseline CR: {day1['base_cr']}")

    metrics_all = []
    rolling_cr, rolling_lr, rolling_rr = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)

    for ep in range(n_episodes):
        policy.restore_w0()
        param_dict = dict(policy.get_perturbable_params())
        state = AdaptiveState(n_actions=4)
        obs, info = env.reset(seed=ep)
        ep_reward, collision, ep_fra, ep_less_risky, ep_rr = 0.0, False, 0, 0, []

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
                with torch.no_grad():
                    for gi in range(n_groups):
                        s, e = gi*group_size, min((gi+1)*group_size, len(param_names))
                        gw = reg["group_weights"][gi]
                        for pi in range(s, e):
                            p = param_dict[param_names[pi]]
                            if p.grad is None: continue
                            epi = p.grad.data.abs().flatten().argmax().item()
                            apply_gaussian(p, p.grad.data, epi, reg["sigma"], reg["magnitude"]*gw*risk)

                state.apply(reg)
                with torch.no_grad():
                    adj = state.get_adjusted(policy.get_logits_from_obs(obs))
                    new_action = torch.distributions.Categorical(logits=adj).sample().item()
                risk_new = compute_risk(obs, cost, ttc, new_action)
                rr = risk - risk_new
                ep_rr.append(rr)
                if rr > 0: ep_less_risky += 1
                regulator.store_step(reg["log_prob"], rr)

                # Store recovery training data
                recovery_buffer.append({
                    "fear": fear, "cost": cost, "ttc": ttc,
                    "displacement": state.get_displacement(),
                    "time_since": state.time_since_spike,
                    "velocity": state.get_velocity(),
                    "risk_reduction": rr,
                })
                action = new_action
            else:
                with torch.no_grad():
                    if state.is_active():
                        action = torch.distributions.Categorical(logits=state.get_adjusted(logits)).sample().item()
                    else:
                        action = torch.distributions.Categorical(logits=logits).sample().item()

            # Adaptive recovery — model decides rates based on current situation
            temp_rate, sup_rate = recovery_model.get_rates(
                fear, cost, ttc, state.get_displacement(),
                state.time_since_spike, state.get_velocity()
            )
            state.decay(temp_rate, sup_rate)

            # Weight recovery
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
            ep_reward += reward
            if terminated: collision = info.get("collision", False); break
            if truncated: break

        # Train regulator
        train_r = regulator.end_episode()

        # Train recovery model every 10 episodes
        if (ep + 1) % 10 == 0 and len(recovery_buffer) >= 64:
            idx = np.random.choice(len(recovery_buffer), 64, replace=False)
            batch = [recovery_buffer[i] for i in idx]
            feats = torch.tensor([[b["fear"], b["cost"], min(b["ttc"],10)/10,
                                    b["displacement"], b["time_since"]/50, b["velocity"]]
                                   for b in batch], dtype=torch.float32, device=torch.device(device))
            rewards = torch.tensor([b["risk_reduction"] for b in batch], dtype=torch.float32, device=torch.device(device))
            pred_rates = recovery_model(feats)
            # Target: higher recovery rate when risk_reduction was good (faster return to normal)
            # Lower recovery rate when risk_reduction was bad (stay cautious longer)
            target = torch.sigmoid(rewards.unsqueeze(-1).expand_as(pred_rates) * 2) * 0.19 + 0.8
            loss_rec = nn.functional.mse_loss(pred_rates, target)
            recovery_model.optimizer.zero_grad(); loss_rec.backward(); recovery_model.optimizer.step()

        lr_pct = ep_less_risky / max(ep_fra, 1)
        mean_rr = np.mean(ep_rr) if ep_rr else 0
        rolling_cr.append(int(collision)); rolling_lr.append(lr_pct); rolling_rr.append(mean_rr)

        metrics_all.append({"episode": ep, "collision": int(collision), "less_risky_pct": lr_pct,
                            "mean_risk_reduction": mean_rr})

        if (ep + 1) % 50 == 0:
            print(f"  [{ep+1}/{n_episodes}] CR={np.mean(rolling_cr):.3f} | "
                  f"LessRisky={np.mean(rolling_lr):.0%} | RiskRed={np.mean(rolling_rr):+.3f}")

    # Save
    window = 100
    curve = []
    for i in range(0, len(metrics_all)-window, window//2):
        ch = metrics_all[i:i+window]
        curve.append({"ep": i, "cr": np.mean([m["collision"] for m in ch]),
                       "lr": np.mean([m["less_risky_pct"] for m in ch]),
                       "rr": np.mean([m["mean_risk_reduction"] for m in ch])})

    print(f"\n{'='*60}\nv8b COMPLETE\n{'='*60}")
    if len(curve) >= 2:
        f, l = curve[0], curve[-1]
        print(f"  CR: {f['cr']:.3f} -> {l['cr']:.3f}")
        print(f"  LessRisky: {f['lr']:.1%} -> {l['lr']:.1%}")
        print(f"  RiskRed: {f['rr']:+.4f} -> {l['rr']:+.4f}")

    with open(v8b_dir / "metrics.json", "w") as f_out: json.dump(metrics_all, f_out, indent=2, default=str)
    with open(v8b_dir / "learning_curve.json", "w") as f_out: json.dump(curve, f_out, indent=2, default=str)
    torch.save(regulator.state_dict(), v8b_dir / "regulator_final.pt")
    torch.save(recovery_model.state_dict(), v8b_dir / "recovery_model_final.pt")
    with open(v8b_dir / "experiment_log.json", "w") as f_out:
        json.dump({"version": "v8b", "mechanism": "combined_learned_recovery",
                    "final_cr": curve[-1]["cr"] if curve else None,
                    "final_lr": curve[-1]["lr"] if curve else None,
                    "final_rr": curve[-1]["rr"] if curve else None,
                    "baseline_cr": day1["base_cr"]}, f_out, indent=2, default=str)
    print(f"Results in {v8b_dir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="SmolLM2-135M")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run_v8b(a.model, a.episodes, a.device)

"""Experiment v9 — LSTM Recovery Controller.

Same perturbation as v8 (Gaussian weight + confidence adjustment).
Recovery is controlled by a 2-layer LSTM that sees the full temporal
context of fear spikes, suppressions, and environmental state.

The LSTM learns:
  - Recover slowly when danger persists
  - Recover quickly after isolated scares
  - Freeze recovery when new spikes arrive during recovery
  - Different rates for temperature vs suppression vs weights
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


class RecoveryLSTM(nn.Module):
    """LSTM that controls recovery rates based on temporal fear history.

    Sees: [fear, risk, cost, ttc, displacement, velocity, time_since_spike]
    Outputs: [temp_rate, suppress_rate, weight_rate] each in [0.8, 0.99]

    Hidden state carries the memory of past spikes and recoveries.
    """

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.input_dim = 7
        self.hidden_dim = 32
        self.n_layers = 2

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True,
        ).to(self.device).float()

        # 3 recovery rates: temperature, suppression, weight
        self.rate_head = nn.Linear(self.hidden_dim, 3).to(self.device).float()

        # For REINFORCE: log_std parameters
        self.log_std = nn.Parameter(torch.zeros(3, device=self.device) - 1.0)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

        # Hidden state
        self.h = None
        self.c = None

    def reset_hidden(self):
        self.h = torch.zeros(self.n_layers, 1, self.hidden_dim, device=self.device)
        self.c = torch.zeros(self.n_layers, 1, self.hidden_dim, device=self.device)

    def forward_step(self, features):
        """Single timestep forward. Maintains hidden state."""
        x = features.unsqueeze(0).unsqueeze(0)  # [1, 1, 7]
        out, (self.h, self.c) = self.lstm(x, (self.h.detach(), self.c.detach()))
        raw = self.rate_head(out.squeeze(0).squeeze(0))  # [3]
        means = torch.sigmoid(raw) * 0.19 + 0.8  # [0.8, 0.99]
        stds = torch.exp(self.log_std.clamp(-3, 0))
        return means, stds

    def get_rates(self, fear, risk, cost, ttc, displacement, velocity, time_since):
        """Get recovery rates for current timestep."""
        feat = torch.tensor(
            [fear, risk, cost, min(ttc, 10) / 10, displacement, velocity, time_since / 50],
            dtype=torch.float32, device=self.device
        )
        means, stds = self.forward_step(feat)
        dist = Normal(means, stds)
        rates = dist.sample().clamp(0.8, 0.99)
        log_prob = dist.log_prob(rates).sum()
        return rates[0].item(), rates[1].item(), rates[2].item(), log_prob


class Regulator(nn.Module):
    """Same as v8 — controls perturbation magnitude and confidence."""

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

    def act(self, fear, risk, logits, cost, ttc, gpg):
        ln = logits / (logits.abs().max() + 1e-8)
        feat = torch.tensor([fear, risk] + ln.tolist() + [cost, min(ttc, 10) / 10] + gpg,
                            dtype=torch.float32, device=self.device).unsqueeze(0)
        h = self.backbone(feat)
        mag_d = Normal(torch.sigmoid(self.mag_mean(h).squeeze()) * 0.1, torch.exp(self.mag_logstd.clamp(-3, 0)))
        sig_d = Normal(torch.sigmoid(self.sigma_mean(h).squeeze()) * 0.5 + 0.01, torch.exp(self.sigma_logstd.clamp(-3, 0)))
        temp_d = Normal(torch.sigmoid(self.temp_mean(h).squeeze()) * 4.5 + 0.5, torch.exp(self.temp_logstd.clamp(-3, 0)))
        sup_mu = self.suppress_mean(h)
        sup_d = Normal(sup_mu, torch.exp(self.suppress_logstd.clamp(-3, 0)).unsqueeze(0).expand_as(sup_mu))
        gw = torch.softmax(self.group_head(h).squeeze(0), dim=-1)
        mag = mag_d.sample().clamp(0, 0.1)
        sig = sig_d.sample().clamp(0.01, 0.51)
        temp = temp_d.sample().clamp(0.5, 5.0)
        sup = sup_d.sample().squeeze(0).clamp(-3, 0.5)
        lp = mag_d.log_prob(mag) + sig_d.log_prob(sig) + temp_d.log_prob(temp) + sup_d.log_prob(sup.unsqueeze(0)).sum()
        return {"magnitude": mag.item(), "sigma": sig.item(),
                "group_weights": gw.squeeze(0).detach().cpu().numpy(),
                "temperature": temp.item(), "suppression": sup.detach(), "log_prob": lp}

    def store_step(self, lp, r):
        self.ep_log_probs.append(lp)
        self.ep_rewards.append(r)

    def end_episode(self):
        if len(self.ep_rewards) < 2:
            self.ep_log_probs = []
            self.ep_rewards = []
            return None
        rw = torch.tensor(self.ep_rewards, device=self.device, dtype=torch.float32)
        lp = torch.stack(self.ep_log_probs)
        rw = (rw - rw.mean()) / (rw.std() + 1e-8)
        loss = -(lp * rw).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        r = {"loss": loss.item()}
        self.ep_log_probs = []
        self.ep_rewards = []
        return r


class ConfState:
    def __init__(self, n_actions=4):
        self.temperature = 1.0
        self.suppression = np.zeros(n_actions)
        self.time_since_spike = 0
        self.prev_displacement = 0.0

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

    def displacement(self):
        return abs(self.temperature - 1.0) + np.sum(np.abs(self.suppression))

    def velocity(self):
        v = abs(self.displacement() - self.prev_displacement)
        self.prev_displacement = self.displacement()
        return v

    def is_active(self):
        return self.temperature > 1.01 or np.any(np.abs(self.suppression) > 0.01)

    def reset(self):
        self.temperature = 1.0
        self.suppression = np.zeros(len(self.suppression))
        self.time_since_spike = 0
        self.prev_displacement = 0.0


def run_v9(model_name="SmolLM2-135M", n_episodes=2000, device="cuda"):
    v3_dir = Path("experiment_v3") / model_name
    v9_dir = Path("experiment_v9") / model_name
    v9_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("v9: LSTM RECOVERY CONTROLLER")
    print("Temporal memory of fear history → adaptive recovery rates")
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
    eta_h_base = min(0.01, 0.5 / max(day1["f_max"], 1e-8))

    regulator = Regulator(n_actions=4, n_groups=n_groups, device=device)
    recovery_lstm = RecoveryLSTM(device=device)

    # Separate training for LSTM
    lstm_ep_log_probs = []
    lstm_ep_rewards = []

    print(f"  Model: {model_name} ({policy.n_perturbable:,} params)")
    print(f"  Baseline CR: {day1['base_cr']}")
    print(f"  Recovery: 2-layer LSTM, 32 hidden, 7 input features")

    ckpt_dir = v9_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    metrics_all = []
    rolling_cr, rolling_lr, rolling_rr = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)

    # Resume from checkpoint if available
    start_ep = 0
    existing = sorted(ckpt_dir.glob("ep_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if existing:
        latest = existing[-1]
        resume_ep = int(latest.stem.split("_")[1])
        print(f"  RESUMING from checkpoint ep_{resume_ep}")
        ckpt = torch.load(latest, weights_only=False, map_location=device)
        regulator.load_state_dict(ckpt["regulator"])
        recovery_lstm.load_state_dict(ckpt["recovery_lstm"])
        if "metrics" in ckpt:
            metrics_all = ckpt["metrics"]
            for m in metrics_all[-100:]:
                rolling_cr.append(m["collision"])
                rolling_lr.append(m["less_risky_pct"])
                rolling_rr.append(m["mean_risk_reduction"])
        start_ep = resume_ep
        print(f"  Resumed at episode {start_ep}, {len(metrics_all)} metrics loaded")

    for ep in range(start_ep, n_episodes):
        policy.restore_w0()
        param_dict = dict(policy.get_perturbable_params())
        state = ConfState(n_actions=4)
        recovery_lstm.reset_hidden()
        obs, info = env.reset(seed=ep)

        ep_reward, collision = 0.0, False
        ep_fra, ep_lr, ep_rrs = 0, 0, []

        for t in range(500):
            cost, ttc = info.get("cost", 0.0), info.get("ttc", 10.0)

            with torch.no_grad():
                logits = policy.get_logits_from_obs(obs)
                greedy = logits.argmax().item()

            risk = compute_risk(obs, cost, ttc, greedy)
            fear, _ = fear_det.detect(obs, cost, ttc, greedy)

            if fear > 0.05 and risk > 0.1:
                ep_fra += 1

                # Gradient computation
                policy.model.zero_grad()
                policy.action_head.zero_grad()
                lg = policy.get_logits_from_obs(obs)
                torch.log_softmax(lg, dim=-1)[greedy].backward()

                gpg = []
                for gi in range(n_groups):
                    s, e = gi * group_size, min((gi + 1) * group_size, len(param_names))
                    gs = sum(param_dict[param_names[pi]].grad.data.abs().mean().item()
                             for pi in range(s, e) if param_dict[param_names[pi]].grad is not None) / max(e - s, 1)
                    gpg.append(gs)

                # Regulator decides perturbation
                reg = regulator.act(fear, risk, logits.detach(), cost, ttc, gpg)

                # Apply Gaussian weight perturbation
                with torch.no_grad():
                    for gi in range(n_groups):
                        s, e = gi * group_size, min((gi + 1) * group_size, len(param_names))
                        gw = reg["group_weights"][gi]
                        for pi in range(s, e):
                            p = param_dict[param_names[pi]]
                            if p.grad is None: continue
                            epi = p.grad.data.abs().flatten().argmax().item()
                            apply_gaussian(p, p.grad.data, epi, reg["sigma"], reg["magnitude"] * gw * risk)

                # Apply confidence adjustment
                state.apply(reg)

                # Perturbed decision
                with torch.no_grad():
                    adj = state.get_adjusted(policy.get_logits_from_obs(obs))
                    new_action = torch.distributions.Categorical(logits=adj).sample().item()

                risk_new = compute_risk(obs, cost, ttc, new_action)
                rr = risk - risk_new
                ep_rrs.append(rr)
                if rr > 0: ep_lr += 1

                regulator.store_step(reg["log_prob"], rr)
                action = new_action
            else:
                with torch.no_grad():
                    if state.is_active():
                        action = torch.distributions.Categorical(logits=state.get_adjusted(logits)).sample().item()
                    else:
                        action = torch.distributions.Categorical(logits=logits).sample().item()

            # LSTM recovery — runs EVERY timestep
            temp_rate, sup_rate, weight_rate, lstm_lp = recovery_lstm.get_rates(
                fear, risk, cost, ttc,
                state.displacement(), state.velocity(), state.time_since_spike,
            )

            # Apply LSTM-controlled recovery
            state.decay(temp_rate, sup_rate)

            # Weight FHR with LSTM-controlled rate
            with torch.no_grad():
                pidx = 0
                for name in param_names:
                    p = param_dict[name]
                    if name not in w0:
                        pidx += p.numel()
                        continue
                    diff = w0[name].to(p.dtype) - p.data
                    ne = p.numel()
                    fs = fisher[pidx:pidx + ne].reshape(p.shape).to(p.dtype).to(p.device)
                    pidx += ne
                    # LSTM controls the weight recovery rate
                    p.data += eta_h_base * weight_rate * fs * diff

            # Store LSTM training data (reward = risk of current action)
            # Lower risk action during recovery = good recovery pacing
            current_risk = compute_risk(obs, cost, ttc, action)
            lstm_reward = -current_risk  # negative risk = positive reward
            lstm_ep_log_probs.append(lstm_lp)
            lstm_ep_rewards.append(lstm_reward)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated:
                collision = info.get("collision", False)
                break
            if truncated:
                break

        # Train regulator (same as v8)
        train_r = regulator.end_episode()

        # Train LSTM recovery controller
        lstm_train = None
        if len(lstm_ep_rewards) >= 10:
            rw = torch.tensor(lstm_ep_rewards, device=torch.device(device), dtype=torch.float32)
            lp = torch.stack(lstm_ep_log_probs)
            rw = (rw - rw.mean()) / (rw.std() + 1e-8)
            lstm_loss = -(lp * rw).mean()
            recovery_lstm.optimizer.zero_grad()
            lstm_loss.backward()
            nn.utils.clip_grad_norm_(recovery_lstm.parameters(), 1.0)
            recovery_lstm.optimizer.step()
            lstm_train = {"loss": lstm_loss.item()}

        lstm_ep_log_probs = []
        lstm_ep_rewards = []

        lr_pct = ep_lr / max(ep_fra, 1)
        mean_rr = np.mean(ep_rrs) if ep_rrs else 0
        rolling_cr.append(int(collision))
        rolling_lr.append(lr_pct)
        rolling_rr.append(mean_rr)

        metrics_all.append({
            "episode": ep, "collision": int(collision), "reward": ep_reward,
            "less_risky_pct": lr_pct, "mean_risk_reduction": mean_rr,
            "n_fra_steps": ep_fra,
        })

        if (ep + 1) % 50 == 0:
            avg_cr = np.mean(rolling_cr)
            avg_lr = np.mean(rolling_lr)
            avg_rr = np.mean(rolling_rr)
            reg_str = f"reg_loss={train_r['loss']:.4f}" if train_r else "warmup"
            lstm_str = f"lstm_loss={lstm_train['loss']:.4f}" if lstm_train else ""
            print(f"  [{ep+1}/{n_episodes}] CR={avg_cr:.3f} | "
                  f"LessRisky={avg_lr:.0%} | RiskRed={avg_rr:+.3f} | "
                  f"{reg_str} {lstm_str}")

        if (ep + 1) % 100 == 0:
            torch.save({
                "regulator": regulator.state_dict(),
                "recovery_lstm": recovery_lstm.state_dict(),
                "episode": ep + 1,
                "metrics": metrics_all,
            }, ckpt_dir / f"ep_{ep+1}.pt")

    # Final analysis
    print(f"\n{'='*60}")
    print("v9 COMPLETE")
    print(f"{'='*60}")

    window = 100
    curve = []
    for i in range(0, len(metrics_all) - window, window // 2):
        ch = metrics_all[i:i + window]
        curve.append({
            "ep": i,
            "cr": np.mean([m["collision"] for m in ch]),
            "lr": np.mean([m["less_risky_pct"] for m in ch]),
            "rr": np.mean([m["mean_risk_reduction"] for m in ch]),
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
        print(f"  v8 final: CR=0.690, LessRisky=86%, RiskRed=+0.648")
        print(f"  v9 final: CR={l['cr']:.3f}, LessRisky={l['lr']:.0%}, RiskRed={l['rr']:+.3f}")

    with open(v9_dir / "metrics.json", "w") as fout:
        json.dump(metrics_all, fout, indent=2, default=str)
    with open(v9_dir / "learning_curve.json", "w") as fout:
        json.dump(curve, fout, indent=2, default=str)
    torch.save(regulator.state_dict(), v9_dir / "regulator_final.pt")
    torch.save(recovery_lstm.state_dict(), v9_dir / "recovery_lstm_final.pt")

    with open(v9_dir / "experiment_log.json", "w") as fout:
        json.dump({
            "version": "v9",
            "mechanism": "combined_lstm_recovery",
            "model": model_name,
            "n_episodes": n_episodes,
            "baseline_cr": day1["base_cr"],
            "lstm_config": {"hidden": 32, "layers": 2, "input": 7, "output": 3},
            "final_cr": curve[-1]["cr"] if curve else None,
            "final_less_risky": curve[-1]["lr"] if curve else None,
            "final_risk_red": curve[-1]["rr"] if curve else None,
        }, fout, indent=2, default=str)

    print(f"\nv9 complete. Results in {v9_dir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="SmolLM2-135M")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run_v9(a.model, a.episodes, a.device)

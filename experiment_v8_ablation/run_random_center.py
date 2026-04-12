"""v8 ablation: Random-center Gaussian vs Gradient-center Gaussian.

Reviewer question: does the benefit come from targeting the epicenter
(gradient maximum) or just from the Gaussian shape?

This test: apply the SAME Gaussian perturbation but centered on a
RANDOM weight index instead of the gradient argmax. If results are
similar, targeting doesn't matter. If worse, targeting is validated.
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


class SimpleRegulator(nn.Module):
    """Same as v8 regulator but simplified for ablation speed."""
    def __init__(self, n_actions=4, n_groups=10, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        input_dim = 2 + n_actions + 2 + n_groups
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        ).to(self.device).float()
        self.mag = nn.Linear(64, 1).to(self.device).float()
        self.sig = nn.Linear(64, 1).to(self.device).float()
        self.temp = nn.Linear(64, 1).to(self.device).float()
        self.sup = nn.Linear(64, n_actions).to(self.device).float()
        self.grp = nn.Linear(64, n_groups).to(self.device).float()
        self.log_std = nn.Parameter(torch.zeros(3 + n_actions, device=self.device) - 1.0)
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.ep_lp, self.ep_rw = [], []

    def act(self, fear, risk, logits, cost, ttc, gpg):
        ln = logits / (logits.abs().max() + 1e-8)
        feat = torch.tensor([fear, risk] + ln.tolist() + [cost, min(ttc,10)/10] + gpg,
                            dtype=torch.float32, device=self.device).unsqueeze(0)
        h = self.net(feat)
        m = torch.sigmoid(self.mag(h).squeeze()) * 0.1
        s = torch.sigmoid(self.sig(h).squeeze()) * 0.5 + 0.01
        t = torch.sigmoid(self.temp(h).squeeze()) * 4.5 + 0.5
        sp = self.sup(h).squeeze(0).clamp(-3, 0.5)
        gw = torch.softmax(self.grp(h).squeeze(0), dim=-1)
        std = torch.exp(self.log_std.clamp(-3, 0))
        lp = Normal(m, std[0]).log_prob(m) + Normal(s, std[1]).log_prob(s) + Normal(t, std[2]).log_prob(t)
        return {"mag": m.item(), "sig": s.item(), "temp": t.item(),
                "sup": sp.detach(), "gw": gw.squeeze(0).detach().cpu().numpy(), "lp": lp}

    def store(self, lp, r): self.ep_lp.append(lp); self.ep_rw.append(r)
    def train_ep(self):
        if len(self.ep_rw) < 2: self.ep_lp=[]; self.ep_rw=[]; return None
        rw = torch.tensor(self.ep_rw, device=self.device, dtype=torch.float32)
        lp = torch.stack(self.ep_lp)
        rw = (rw - rw.mean()) / (rw.std() + 1e-8)
        loss = -(lp * rw).mean()
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0); self.optimizer.step()
        r = {"loss": loss.item()}; self.ep_lp=[]; self.ep_rw=[]; return r


def run_ablation(model_name="SmolLM2-135M", n_episodes=2000, device="cuda"):
    v3_dir = Path("experiment_v3") / model_name
    abl_dir = Path("experiment_v8_ablation") / model_name
    abl_dir.mkdir(parents=True, exist_ok=True)

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

    print("=" * 60)
    print("ABLATION: Gradient-center vs Random-center Gaussian")
    print("=" * 60)
    print(f"  Baseline CR: {day1['base_cr']}")

    for mode in ["gradient_center", "random_center"]:
        print(f"\n--- {mode} ({n_episodes} episodes) ---")

        regulator = SimpleRegulator(n_actions=4, n_groups=n_groups, device=device)
        rolling_cr, rolling_lr, rolling_rr = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
        all_metrics = []

        for ep in range(n_episodes):
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

                    with torch.no_grad():
                        for gi in range(n_groups):
                            s, e = gi*group_size, min((gi+1)*group_size, len(param_names))
                            gw = reg["gw"][gi]
                            for pi in range(s, e):
                                p = param_dict[param_names[pi]]
                                if p.grad is None: continue

                                if mode == "gradient_center":
                                    epicenter = p.grad.data.abs().flatten().argmax().item()
                                else:
                                    epicenter = torch.randint(0, p.numel(), (1,)).item()

                                apply_gaussian(p, p.grad.data, epicenter, reg["sig"], reg["mag"]*gw*risk)

                    temp = max(temp, reg["temp"])
                    sup += reg["sup"].cpu().numpy()

                    with torch.no_grad():
                        sup_t = torch.tensor(sup, device=logits.device, dtype=logits.dtype)
                        adj = (policy.get_logits_from_obs(obs) + sup_t) / temp
                        new_a = torch.distributions.Categorical(logits=adj).sample().item()
                    rr = risk - compute_risk(obs, cost, ttc, new_a)
                    ep_rrs.append(rr)
                    if rr > 0: ep_lr += 1
                    regulator.store(reg["lp"], rr)
                    action = new_a
                else:
                    if temp > 1.01 or np.any(np.abs(sup) > 0.01):
                        sup_t = torch.tensor(sup, device=logits.device, dtype=logits.dtype)
                        action = torch.distributions.Categorical(logits=(logits+sup_t)/temp).sample().item()
                    else:
                        action = torch.distributions.Categorical(logits=logits).sample().item()

                temp = 1.0 + (temp - 1.0) * 0.92
                sup *= 0.92
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

            regulator.train_ep()
            lr_pct = ep_lr / max(ep_fra, 1)
            mean_rr = np.mean(ep_rrs) if ep_rrs else 0
            rolling_cr.append(int(collision)); rolling_lr.append(lr_pct); rolling_rr.append(mean_rr)
            all_metrics.append({"ep": ep, "collision": int(collision), "lr": lr_pct, "rr": mean_rr})

            if (ep + 1) % 100 == 0:
                print(f"    [{ep+1}/{n_episodes}] CR={np.mean(rolling_cr):.3f} | "
                      f"LessRisky={np.mean(rolling_lr):.0%} | RiskRed={np.mean(rolling_rr):+.3f}")

        # Save
        window = 100
        curve = []
        for i in range(0, len(all_metrics)-window, window//2):
            ch = all_metrics[i:i+window]
            curve.append({"ep": i, "cr": np.mean([m["collision"] for m in ch]),
                           "lr": np.mean([m["lr"] for m in ch]), "rr": np.mean([m["rr"] for m in ch])})

        with open(abl_dir / f"{mode}_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        with open(abl_dir / f"{mode}_curve.json", "w") as f:
            json.dump(curve, f, indent=2, default=str)

        final = curve[-1] if curve else {"cr": 1, "lr": 0, "rr": 0}
        print(f"\n  {mode} FINAL: CR={final['cr']:.3f} | LessRisky={final['lr']:.0%} | RiskRed={final['rr']:+.3f}")

    # Comparison
    print(f"\n{'='*60}")
    print("ABLATION RESULT")
    print(f"{'='*60}")
    for mode in ["gradient_center", "random_center"]:
        curve = json.load(open(abl_dir / f"{mode}_curve.json"))
        f = curve[-1]
        print(f"  {mode:>20s}: CR={f['cr']:.3f} | LessRisky={f['lr']:.0%} | RiskRed={f['rr']:+.3f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="SmolLM2-135M")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run_ablation(a.model, a.episodes, a.device)

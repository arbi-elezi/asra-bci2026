"""Minimal PPO to train the frozen MLP base policy on HighwayFRAEnv."""
from __future__ import annotations

import sys, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.environment.highway_wrapper import HighwayFRAEnv
from current_work.asra_fast.core import MLPActor, MLPCritic


def make_env(scenario: str = "highway", vehicles: int = 15, seed: int = 0,
             density: float = 1.5, high_speed_reward: float = 0.4) -> HighwayFRAEnv:
    return HighwayFRAEnv(seed=seed, vehicles_count=vehicles, vehicles_density=density,
                         high_speed_reward=high_speed_reward, scenario=scenario)


def train_ppo(total_steps: int = 200_000, snapshots=(20_000, 60_000, 200_000),
              hidden: int = 64, lr: float = 3e-4, n_steps: int = 2048,
              n_epochs: int = 4, gamma: float = 0.99, lam: float = 0.95,
              clip: float = 0.2, ent_coef: float = 0.01, vf_coef: float = 0.5,
              vehicles: int = 15, seed: int = 0, device: str = "cpu",
              out_dir: str = "current_work/base_policies", tag: str = "highway",
              density: float = 1.5, high_speed_reward: float = 0.4,
              scenario: str = "highway") -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = torch.device(device)
    env = make_env(scenario=scenario, vehicles=vehicles, seed=seed, density=density,
                   high_speed_reward=high_speed_reward)
    actor = MLPActor(12, 4, hidden).to(dev)
    critic = MLPCritic(12, hidden).to(dev)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    snaps = sorted(set(snapshots))
    snap_paths = {}

    obs, info = env.reset(seed=seed)
    cost, ttc = info.get("cost", 0.0), info.get("ttc", 10.0)
    steps = 0
    ep_ret, ep_len = 0.0, 0
    recent_cr, recent_ret = [], []
    cur_ep_collision = 0
    t0 = time.time()

    while steps < total_steps:
        # ---- collect a rollout ----
        buf_obs, buf_act, buf_logp, buf_val, buf_rew, buf_done = [], [], [], [], [], []
        for _ in range(n_steps):
            o = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            with torch.no_grad():
                logits = actor(o).squeeze(0)
                val = critic(o).item()
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a).item()
            act = int(a.item())
            nobs, rew, term, trunc, ninfo = env.step(act)
            done = term or trunc
            buf_obs.append(obs); buf_act.append(act); buf_logp.append(logp)
            buf_val.append(val); buf_rew.append(rew); buf_done.append(done)
            ep_ret += rew; ep_len += 1; steps += 1
            obs = nobs
            if done:
                cr = 1 if ninfo.get("collision", False) else 0
                recent_cr.append(cr); recent_ret.append(ep_ret)
                if len(recent_cr) > 200: recent_cr.pop(0); recent_ret.pop(0)
                obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
                ep_ret, ep_len = 0.0, 0
            else:
                info = ninfo

        # bootstrap value for last state
        with torch.no_grad():
            last_val = critic(torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)).item()

        # ---- GAE ----
        rews = np.array(buf_rew, dtype=np.float64)
        vals = np.array(buf_val + [last_val], dtype=np.float64)
        dones = np.array(buf_done, dtype=np.float64)
        adv = np.zeros(len(rews), dtype=np.float64)
        gae = 0.0
        for t in reversed(range(len(rews))):
            nonterm = 1.0 - dones[t]
            delta = rews[t] + gamma * vals[t + 1] * nonterm - vals[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + vals[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.as_tensor(np.array(buf_obs), dtype=torch.float32, device=dev)
        act_t = torch.as_tensor(buf_act, dtype=torch.long, device=dev)
        logp_t = torch.as_tensor(buf_logp, dtype=torch.float32, device=dev)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=dev)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=dev)

        # ---- PPO update ----
        idx = np.arange(len(buf_obs))
        mb = 256
        for _ in range(n_epochs):
            rng.shuffle(idx)
            for s in range(0, len(idx), mb):
                b = idx[s:s + mb]
                bo, ba = obs_t[b], act_t[b]
                logits = actor(bo)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(ba)
                ratio = torch.exp(new_logp - logp_t[b])
                s1 = ratio * adv_t[b]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[b]
                pol_loss = -torch.min(s1, s2).mean()
                val = critic(bo)
                v_loss = F.mse_loss(val, ret_t[b])
                ent = dist.entropy().mean()
                loss = pol_loss + vf_coef * v_loss - ent_coef * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                opt.step()

        cr_now = np.mean(recent_cr) if recent_cr else float("nan")
        ret_now = np.mean(recent_ret) if recent_ret else float("nan")
        print(f"  [{tag} seed{seed}] step {steps:>7d} | CR~{cr_now:.3f} | ret~{ret_now:6.2f} "
              f"| ent {ent.item():.3f} | {steps/(time.time()-t0):.0f} st/s", flush=True)

        # ---- snapshots ----
        for sp in snaps:
            if steps >= sp and sp not in snap_paths:
                path = out / f"{tag}_seed{seed}_step{sp}.pt"
                torch.save({"actor": {k: v.cpu().clone() for k, v in actor.state_dict().items()},
                            "hidden": hidden, "step": sp, "seed": seed, "tag": tag,
                            "vehicles": vehicles, "density": density,
                            "high_speed_reward": high_speed_reward, "scenario": scenario,
                            "train_cr": float(cr_now), "train_ret": float(ret_now)}, path)
                snap_paths[sp] = str(path)
                print(f"    snapshot -> {path.name} (CR~{cr_now:.3f})", flush=True)

    # always save a final snapshot at the last step reached (so total need not equal a snapshot)
    final = out / f"{tag}_seed{seed}_final.pt"
    torch.save({"actor": {k: v.cpu().clone() for k, v in actor.state_dict().items()},
                "hidden": hidden, "step": steps, "seed": seed, "tag": tag,
                "vehicles": vehicles, "density": density, "high_speed_reward": high_speed_reward,
                "train_cr": float(np.mean(recent_cr) if recent_cr else float('nan')),
                "train_ret": float(np.mean(recent_ret) if recent_ret else float('nan'))}, final)
    snap_paths["final"] = str(final)
    env.close()
    return snap_paths


def collect_dref(actor: MLPActor, n: int = 5000, vehicles: int = 15,
                 seed: int = 123, device: str = "cpu", density: float = 1.5,
                 high_speed_reward: float = 0.4, scenario: str = "highway"
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the frozen policy to collect D_ref states + costs for salience/Fisher."""
    dev = torch.device(device)
    env = make_env(scenario=scenario, vehicles=vehicles, seed=seed, density=density,
                   high_speed_reward=high_speed_reward)
    rng = np.random.default_rng(seed)
    states, costs = [], []
    obs, info = env.reset(seed=seed)
    while len(states) < n:
        o = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
        with torch.no_grad():
            a = int(torch.distributions.Categorical(logits=actor(o).squeeze(0)).sample().item())
        states.append(obs.copy()); costs.append(info.get("cost", 0.0))
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
    env.close()
    return np.array(states, dtype=np.float32), np.array(costs, dtype=np.float32)


def compute_fisher(actor: MLPActor, states: np.ndarray, device: str = "cpu",
                   reg: float = 1e-3) -> dict[str, torch.Tensor]:
    """Diagonal Fisher: F_ii = E_s[(d log pi(greedy|s)/d theta_i)^2], normalized to (0,1]."""
    dev = torch.device(device)
    fisher = {name: torch.zeros_like(p, device=dev) for name, p in actor.named_parameters()}
    n = min(len(states), 2000)
    sel = states[np.random.default_rng(0).choice(len(states), n, replace=False)]
    for s in sel:
        o = torch.as_tensor(s, dtype=torch.float32, device=dev).unsqueeze(0)
        actor.zero_grad(set_to_none=True)
        logits = actor(o).squeeze(0)
        greedy = int(torch.argmax(logits).item())
        F.log_softmax(logits, dim=-1)[greedy].backward()
        for name, p in actor.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
    actor.zero_grad(set_to_none=True)
    # normalize to (0,1] with regularization so f_min > 0
    gmax = max((f.max().item() for f in fisher.values()), default=1.0) or 1.0
    for name in fisher:
        fisher[name] = (fisher[name] / gmax).clamp(min=reg)
    return fisher


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default="highway")
    p.add_argument("--vehicles", type=int, default=15)
    p.add_argument("--density", type=float, default=1.5)
    p.add_argument("--high_speed_reward", type=float, default=0.4)
    a = p.parse_args()
    paths = train_ppo(total_steps=a.total, seed=a.seed, device=a.device,
                      tag=a.tag, vehicles=a.vehicles, density=a.density,
                      high_speed_reward=a.high_speed_reward)
    print(json.dumps(paths, indent=2))

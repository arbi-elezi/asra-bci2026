"""HazardCorridor: a minimal autonomous-decision task that SATISFIES the precondition"""
from __future__ import annotations
import sys, argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from current_work.asra_fast.core import MLPActor, apply_gaussian_, bootstrap_ci
from current_work.asra_fast.analyze import paired_bootstrap_diff


# The frozen policy is BLIND to the hazard flag (index 0): it was trained for a
# different (low-hazard) risk profile and cannot perceive the current threat. ASRA's
# INDEPENDENT salience detector does see it. This is faithful to ASRA's premise and
# creates the precondition: the policy still knows lane-clear info (indices 1,2), so its
# 2nd-best action under suppression is a clear lane (safe + progress), not braking.
POL_MASK = torch.tensor([0., 1, 1, 1, 1, 1, 1])  # zeros the hazard flag for the policy


def pol_in(o: torch.Tensor) -> torch.Tensor:
    return o * POL_MASK.to(o.device)


class HazardCorridor:
    """Single-env, fast. obs in R^7, 4 actions, episode length T."""
    OBS_DIM = 7
    N_ACT = 4

    def __init__(self, T=40, p_hazard=0.35, seed=0):
        self.T = T; self.p = p_hazard
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.lane = 1  # middle of {0,1,2}
        return self._obs()

    def _draw_hazard(self):
        # hazard blocks current lane; ensure >=1 adjacent lane clear
        self.hazard = self.rng.random() < self.p
        self.left_clear = self.right_clear = True
        if self.hazard:
            # randomly make at most one adjacent lane also blocked, never both
            r = self.rng.random()
            if r < 0.35:
                self.left_clear = False
            elif r < 0.70:
                self.right_clear = False
            # edge lanes: the off-grid side is "not clear"
        if self.lane == 0:
            self.left_clear = False
        if self.lane == 2:
            self.right_clear = False

    def _obs(self):
        self._draw_hazard() if self.t == 0 or True else None
        lane_oh = [0.0, 0.0, 0.0]; lane_oh[self.lane] = 1.0
        return np.array([float(self.hazard),
                         float(self.left_clear), float(self.right_clear),
                         *lane_oh, self.t / self.T], dtype=np.float32)

    def step(self, a):
        crash = False; rew = 0.0
        if a == 0:                       # ADVANCE fast
            if self.hazard:
                crash = True; rew = -10.0
            else:
                rew = 2.0
        elif a == 1:                     # switch left
            if self.lane > 0 and self.left_clear:
                self.lane -= 1; rew = 1.0
            else:
                crash = True; rew = -10.0
        elif a == 2:                     # switch right
            if self.lane < 2 and self.right_clear:
                self.lane += 1; rew = 1.0
            else:
                crash = True; rew = -10.0
        else:                            # brake
            rew = 0.0
        self.t += 1
        done = crash or self.t >= self.T
        obs = self._obs() if not done else np.zeros(self.OBS_DIM, dtype=np.float32)
        return obs, rew, done, crash


# task-specific risk + salience (analogous to driving's f_risk / cost)
def hc_salience(obs):                       # danger present?
    return float(obs[0])                    # hazard flag

def hc_risk(obs, action):                   # risk of taking `action` now
    hazard = obs[0] > 0.5
    if not hazard:
        return 0.1
    return {0: 0.9, 1: 0.6, 2: 0.6, 3: 0.05}.get(action, 0.5)


def train_policy(steps=120_000, hidden=64, seed=0, p_hazard=0.35, T=40,
                 lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, ent=0.01, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    dev = torch.device(device)
    actor = MLPActor(HazardCorridor.OBS_DIM, 4, hidden).to(dev)
    critic = nn.Sequential(nn.Linear(HazardCorridor.OBS_DIM, hidden), nn.Tanh(),
                           nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    env = HazardCorridor(T=T, p_hazard=p_hazard, seed=seed)
    obs = env.reset(seed=seed); done = False
    n_steps = 1024; steps_done = 0
    while steps_done < steps:
        O, A, LP, V, R, D = [], [], [], [], [], []
        for _ in range(n_steps):
            o = torch.as_tensor(obs, device=dev).unsqueeze(0)
            with torch.no_grad():
                logits = actor(pol_in(o)).squeeze(0); v = critic(o).item()
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample(); lp = dist.log_prob(a).item()
            no, r, dn, crash = env.step(int(a.item()))
            O.append(obs); A.append(int(a.item())); LP.append(lp); V.append(v); R.append(r); D.append(dn)
            obs = no; steps_done += 1
            if dn:
                obs = env.reset(seed=int(rng.integers(1, 1_000_000)))
        with torch.no_grad():
            lastv = critic(torch.as_tensor(obs, device=dev).unsqueeze(0)).item()
        R = np.array(R); V = np.array(V + [lastv]); D = np.array(D, float)
        adv = np.zeros(len(R)); gae = 0
        for t in reversed(range(len(R))):
            nt = 1 - D[t]; delta = R[t] + gamma * V[t+1] * nt - V[t]
            gae = delta + gamma * lam * nt * gae; adv[t] = gae
        ret = adv + V[:-1]; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        O = torch.as_tensor(np.array(O), device=dev); A = torch.as_tensor(A, device=dev)
        LP = torch.as_tensor(LP, device=dev); ADV = torch.as_tensor(adv, dtype=torch.float32, device=dev)
        RET = torch.as_tensor(ret, dtype=torch.float32, device=dev)
        idx = np.arange(len(O))
        for _ in range(4):
            rng.shuffle(idx)
            for s in range(0, len(idx), 256):
                b = idx[s:s+256]
                dist = torch.distributions.Categorical(logits=actor(pol_in(O[b])))
                nlp = dist.log_prob(A[b]); ratio = torch.exp(nlp - LP[b])
                pl = -torch.min(ratio*ADV[b], torch.clamp(ratio,1-clip,1+clip)*ADV[b]).mean()
                vl = F.mse_loss(critic(O[b]).squeeze(-1), RET[b])
                loss = pl + 0.5*vl - ent*dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()
    return actor


def eval_condition(actor, mode, n_ep=400, seed=0, T=40, p_hazard=0.35, device="cpu",
                   gain=1.0, fisher=None, w0=None, p_brake=0.0, ttc_override=False):
    """mode: 'raw_greedy','raw_sample','override_brake','prob_brake','asra_greedy','asra_sample'."""
    dev = torch.device(device)
    env = HazardCorridor(T=T, p_hazard=p_hazard, seed=seed)
    rng = np.random.default_rng(seed + 999)
    crashes, rews, lens = [], [], []
    for ep in range(n_ep):
        if w0 is not None:
            with torch.no_grad():
                for k, pmt in actor.named_parameters():
                    pmt.data.copy_(w0[k])
        sup = np.zeros(4)
        obs = env.reset(seed=seed * 100000 + ep); R = 0.0; L = 0; crash = 0
        for t in range(T):
            with torch.no_grad():
                logits = actor(pol_in(torch.as_tensor(obs, device=dev).unsqueeze(0))).squeeze(0)
            greedy = int(logits.argmax())
            if mode.startswith("asra"):
                S = hc_salience(obs); Rk = hc_risk(obs, greedy); alpha = S * Rk
                if alpha > 0.05 and gain > 0:
                    # weight channel: targeted perturbation on the risky greedy action
                    actor.zero_grad(set_to_none=True)
                    lg = actor(pol_in(torch.as_tensor(obs, device=dev).unsqueeze(0))).squeeze(0)
                    F.log_softmax(lg, dim=-1)[greedy].backward()
                    with torch.no_grad():
                        for nm, pmt in actor.named_parameters():
                            if pmt.grad is None: continue
                            epi = int(pmt.grad.abs().view(-1).argmax())
                            apply_gaussian_(pmt, pmt.grad, epi, 0.05, 0.05 * gain * alpha)
                    actor.zero_grad(set_to_none=True)
                    sup[greedy] -= 4.0 * gain * alpha       # confidence: suppress risky action
                with torch.no_grad():
                    adj = actor(pol_in(torch.as_tensor(obs, device=dev).unsqueeze(0))).squeeze(0) \
                          + torch.as_tensor(sup, dtype=logits.dtype, device=dev)
                a = int(adj.argmax()) if mode == "asra_greedy" else \
                    int(torch.distributions.Categorical(logits=adj).sample())
                sup *= 0.9
                if fisher is not None and w0 is not None:
                    with torch.no_grad():
                        for nm, pmt in actor.named_parameters():
                            if nm in w0:
                                pmt.data.add_(0.1 * fisher[nm] * (w0[nm] - pmt.data))
            elif mode == "raw_greedy":
                a = greedy
            elif mode == "raw_sample":
                a = int(torch.distributions.Categorical(logits=logits).sample())
            elif mode == "override_brake":               # TTC-style: hazard -> brake
                a = 3 if obs[0] > 0.5 else int(torch.distributions.Categorical(logits=logits).sample())
            elif mode == "prob_brake":                   # tunable braking
                a = 3 if rng.random() < p_brake else int(torch.distributions.Categorical(logits=logits).sample())
            else:
                raise ValueError(mode)
            obs, r, dn, cr = env.step(a); R += r; L += 1
            if dn:
                crash = int(cr); break
        crashes.append(crash); rews.append(R / max(L, 1)); lens.append(L)
    return {"cr": float(np.mean(crashes)), "perf": float(np.mean(rews)),
            "cr_arr": np.array(crashes, float), "perf_arr": np.array(rews, float)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120_000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n_ep", type=int, default=400)
    ap.add_argument("--out", default="current_work/results_v3/hazard_corridor.json")
    a = ap.parse_args()

    print("Training frozen policy on HazardCorridor...")
    actor = train_policy(steps=a.steps, seed=0)
    w0 = {k: v.detach().clone() for k, v in actor.named_parameters()}
    # fisher (diag) for recovery
    fisher = {k: torch.ones_like(v) for k, v in actor.named_parameters()}

    # diagnose greedy character
    diag = eval_condition(actor, "raw_greedy", n_ep=300, seed=1)
    diag_s = eval_condition(actor, "raw_sample", n_ep=300, seed=1)
    print(f"raw_greedy: CR={diag['cr']:.3f} perf={diag['perf']:.3f} | "
          f"raw_sample: CR={diag_s['cr']:.3f} perf={diag_s['perf']:.3f}")

    results = {"raw_greedy": [], "raw_sample": [], "override_brake": [],
               "asra_greedy": {}, "prob_brake": {}}
    G_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]
    P_GRID = [0.1, 0.3, 0.5, 0.7, 1.0]
    for g in G_GRID: results["asra_greedy"][g] = []
    for p in P_GRID: results["prob_brake"][p] = []

    for s in range(a.seeds):
        for mode in ["raw_greedy", "raw_sample", "override_brake"]:
            r = eval_condition(actor, mode, n_ep=a.n_ep, seed=s)
            results[mode].append({"cr": r["cr"], "perf": r["perf"]})
        for g in G_GRID:
            r = eval_condition(actor, "asra_greedy", n_ep=a.n_ep, seed=s, gain=g, fisher=fisher, w0=w0)
            results["asra_greedy"][g].append({"cr": r["cr"], "perf": r["perf"]})
        for p in P_GRID:
            r = eval_condition(actor, "prob_brake", n_ep=a.n_ep, seed=s, p_brake=p)
            results["prob_brake"][p].append({"cr": r["cr"], "perf": r["perf"]})
        print(f"  seed {s} done")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2, default=float)
    print(f"saved {a.out}")

"""ASRA on MiniGrid (Farama Foundation) -- a KNOWN, widely-cited, NON-driving benchmark."""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym, minigrid
from minigrid.wrappers import FullyObsWrapper
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast.core import MLPActor
from current_work.asra_fast.analyze import paired_bootstrap_diff
RES = Path("current_work/results_v3")

LAVA = 9; GOAL = 8; WALL = 2
DIRVEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}   # MiniGrid: 0 right,1 down,2 left,3 up
NAV = 3  # actions 0=left 1=right 2=forward


def obs_vec(obs):
    img = obs["image"] if isinstance(obs, dict) else obs
    return img.astype(np.float32).flatten() / 10.0


def lava_cells(env):
    g = env.unwrapped.grid; cells = set()
    for i in range(g.width):
        for j in range(g.height):
            c = g.get(i, j)
            if c is not None and c.type == "lava": cells.add((i, j))
    return cells, g.width, g.height


def risk_vec(env, lava):
    """Q_c over {left,right,forward}: forward into lava => 1; turns don't move => 0."""
    ax, ay = env.unwrapped.agent_pos; d = env.unwrapped.agent_dir
    fx, fy = ax + DIRVEC[d][0], ay + DIRVEC[d][1]
    fwd = 1.0 if (fx, fy) in lava else 0.0
    return np.array([0.0, 0.0, fwd]), fwd  # salience uses fwd + adjacency below


def salience(env, lava):
    ax, ay = env.unwrapped.agent_pos
    return 1.0 if any((ax + dx, ay + dy) in lava for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]) else 0.0


def _goal_pos(env):
    g = env.unwrapped.grid
    for i in range(g.width):
        for j in range(g.height):
            c = g.get(i, j)
            if c is not None and c.type == "goal": return (i, j)
    return (g.width - 2, g.height - 2)


def train_ppo(env_id, seed, steps, hidden=64, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    env = FullyObsWrapper(gym.make(env_id)); obs, _ = env.reset(seed=seed)
    dim = obs_vec(obs).shape[0]
    actor = MLPActor(dim, NAV, hidden); critic = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    o = obs_vec(obs); done_steps = 0
    gp = _goal_pos(env); ap = env.unwrapped.agent_pos; pdist = abs(ap[0]-gp[0]) + abs(ap[1]-gp[1])
    while done_steps < steps:
        O, A, LP, V, R, D = [], [], [], [], [], []
        for _ in range(1024):
            t = torch.as_tensor(o).unsqueeze(0)
            with torch.no_grad():
                lg = actor(t).squeeze(0); v = critic(t).item()
                dist = torch.distributions.Categorical(logits=lg); a = dist.sample(); lp = dist.log_prob(a).item()
            nobs, r, term, trunc, _ = env.step(int(a.item())); dn = term or trunc
            ap = env.unwrapped.agent_pos; ndist = abs(ap[0]-gp[0]) + abs(ap[1]-gp[1])
            r = r + 0.05 * (pdist - ndist) - 0.005            # potential-based distance shaping + tiny step cost
            pdist = ndist
            O.append(o); A.append(int(a.item())); LP.append(lp); V.append(v); R.append(r); D.append(dn)
            o = obs_vec(nobs); done_steps += 1
            if dn:
                obs, _ = env.reset(seed=int(rng.integers(1, 1_000_000))); o = obs_vec(obs)
                gp = _goal_pos(env); ap = env.unwrapped.agent_pos; pdist = abs(ap[0]-gp[0]) + abs(ap[1]-gp[1])
        with torch.no_grad(): lastv = critic(torch.as_tensor(o).unsqueeze(0)).item()
        R = np.array(R); V = np.array(V + [lastv]); D = np.array(D, float); adv = np.zeros(len(R)); gae = 0
        for k in reversed(range(len(R))):
            nt = 1 - D[k]; delta = R[k] + 0.99 * V[k+1] * nt - V[k]; gae = delta + 0.95 * 0.99 * nt * gae; adv[k] = gae
        ret = adv + V[:-1]; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        O = torch.as_tensor(np.array(O)); A = torch.as_tensor(A); LP = torch.as_tensor(LP)
        ADV = torch.as_tensor(adv, dtype=torch.float32); RET = torch.as_tensor(ret, dtype=torch.float32); idx = np.arange(len(O))
        for _ in range(4):
            rng.shuffle(idx)
            for s0 in range(0, len(idx), 256):
                b = idx[s0:s0+256]; dist = torch.distributions.Categorical(logits=actor(O[b]))
                ratio = torch.exp(dist.log_prob(A[b]) - LP[b])
                pl = -torch.min(ratio*ADV[b], torch.clamp(ratio, .8, 1.2)*ADV[b]).mean()
                loss = pl + 0.5*F.mse_loss(critic(O[b]).squeeze(-1), RET[b]) - 0.01*dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()
    env.close(); return actor, dim


def eval_policy(actor, deploy_id, mode, gain, n_ep, seed, max_steps=100):
    env = FullyObsWrapper(gym.make(deploy_id)); lava_hits, succ = [], []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed*100000 + ep); lava = lava_cells(env)[0]
        hit = 0; ok = 0
        for t in range(max_steps):
            o = torch.as_tensor(obs_vec(obs)).unsqueeze(0)
            with torch.no_grad(): lg = actor(o).squeeze(0)[:NAV]
            greedy = int(lg.argmax()); qc, fwd = risk_vec(env, lava); S = salience(env, lava)
            if mode == "raw": a = greedy
            elif mode == "override": a = int(np.argmin(qc)) if S > 0.05 else greedy
            elif mode == "mask":
                lg2 = lg.clone(); lg2[greedy] = -1e9; a = int(lg2.argmax())
            elif mode == "shaped":
                q = torch.tensor(qc, dtype=lg.dtype); adj = lg - gain * S * (q - q.mean()); a = int(adj.argmax())
            else: raise ValueError(mode)
            obs, r, term, trunc, info = env.step(a)
            ax, ay = env.unwrapped.agent_pos
            if (ax, ay) in lava: hit = 1; break
            if term and r > 0: ok = 1; break
            if term or trunc: break
        lava_hits.append(hit); succ.append(ok)
    env.close()
    return {"lava_rate": float(np.mean(lava_hits)), "success": float(np.mean(succ))}


def run(train_id, deploy_id, out_path, n_policies, n_ep, steps, gains=(0.5, 1., 2., 4., 8.)):
    t0 = time.time(); results = []
    for pi in range(n_policies):
        actor, dim = train_ppo(train_id, seed=pi, steps=steps)
        modes = [("raw", 0.), ("override", 0.), ("mask", 0.)] + [(f"shaped_g{g:g}", g) for g in gains]
        row = {}
        for name, g in modes:
            m = "shaped" if name.startswith("shaped") else name
            row[name] = eval_policy(actor, deploy_id, m, g, n_ep, seed=pi)
        results.append(row)
        print(f"  policy {pi+1}/{n_policies}: raw lava={row['raw']['lava_rate']:.2f}/succ={row['raw']['success']:.2f} "
              f"override lava={row['override']['lava_rate']:.2f}/succ={row['override']['success']:.2f} "
              f"shaped_g4 lava={row['shaped_g4']['lava_rate']:.2f}/succ={row['shaped_g4']['success']:.2f} [{time.time()-t0:.0f}s]", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"train": train_id, "deploy": deploy_id, "results": results, "gains": list(gains)}, open(out_path, "w"), indent=2)
    print("saved", out_path); return results


def analyze(results, gains):
    npol = len(results)
    print(f"\n=== MiniGrid ASRA ({npol} policies) ===")
    for k in ["raw", "override", "mask"] + [f"shaped_g{g:g}" for g in gains]:
        lr = np.mean([r[k]["lava_rate"] for r in results]); sc = np.mean([r[k]["success"] for r in results])
        print(f"  {k:12s} lava_rate={lr:.3f}  success={sc:.3f}")
    # matched-safety: best shaped success at lava_rate <= override's, vs override & mask
    def bestshaped(r):
        oh = r["override"]["lava_rate"]
        cand = [r[f"shaped_g{g:g}"] for g in gains if r[f"shaped_g{g:g}"]["lava_rate"] <= oh + 1e-9]
        return max(cand, key=lambda x: x["success"]) if cand else max((r[f"shaped_g{g:g}"] for g in gains), key=lambda x: -x["lava_rate"])
    for base in ["override", "mask"]:
        a = np.array([bestshaped(r)["success"] for r in results]); b = np.array([r[base]["success"] for r in results])
        if npol >= 3:
            dd = paired_bootstrap_diff(a, b); sign = int(np.sum(a > b + 1e-9))
            print(f"  shaped succ - {base} succ (matched safety): d={dd['diff']:+.3f} CI[{dd['lo']:+.3f},{dd['hi']:+.3f}] ({sign}/{npol}, p={dd['pvalue']:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="MiniGrid-DistShift1-v0")
    ap.add_argument("--deploy", default="MiniGrid-DistShift2-v0")
    ap.add_argument("--policies", type=int, default=6); ap.add_argument("--n_ep", type=int, default=100)
    ap.add_argument("--steps", type=int, default=150000); ap.add_argument("--out", default=str(RES / "minigrid_asra.json"))
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    gains = (0.5, 1., 2., 4., 8.)
    if a.smoke:
        r = run(a.train, a.deploy, str(RES / "_minigrid_smoke.json"), n_policies=2, n_ep=30, steps=40000, gains=gains)
        analyze(r, gains); print("[SMOKE] OK")
    else:
        r = run(a.train, a.deploy, a.out, a.policies, a.n_ep, a.steps, gains); analyze(r, gains)

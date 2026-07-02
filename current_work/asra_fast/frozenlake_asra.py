"""ASRA on a STRUCTURALLY DIFFERENT open-source simulator: FrozenLake (Gymnasium map generator)"""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast.core import MLPActor, apply_gaussian_
from current_work.asra_fast.analyze import paired_bootstrap_diff
RES = Path("current_work/results_v3")

# 0=Left 1=Down 2=Right 3=Up
PERP = {0: (1, 3), 1: (0, 2), 2: (1, 3), 3: (0, 2)}
DELTA = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


def parse_map(desc):
    grid = [list(r) for r in desc]; n = len(grid); holes = set(); goal = None; start = (0, 0)
    for r in range(n):
        for c in range(n):
            ch = grid[r][c]
            if ch == 'H': holes.add((r, c))
            elif ch == 'G': goal = (r, c)
            elif ch == 'S': start = (r, c)
    return n, holes, goal, start


def cell_after(n, r, c, d):
    dr, dc = DELTA[d]; return min(max(r + dr, 0), n - 1), min(max(c + dc, 0), n - 1)


def risk_fn(n, holes, r, c, a, slip):
    """Independent per-action risk = P(next cell is a hole) under the tunable slip model."""
    p = 0.0
    if cell_after(n, r, c, a) in holes: p += (1.0 - slip)
    for d in PERP[a]:
        if cell_after(n, r, c, d) in holes: p += slip / 2.0
    return p


def step(n, holes, goal, r, c, a, slip, rng):
    d = a if rng.random() >= slip else (PERP[a][0] if rng.random() < 0.5 else PERP[a][1])
    nr, nc = cell_after(n, r, c, d)
    if (nr, nc) in holes: return nr, nc, True, 0.0        # hole -> fail
    if (nr, nc) == goal: return nr, nc, True, 1.0         # goal -> success
    return nr, nc, False, 0.0


def onehot(r, c, n):
    v = np.zeros(n * n, np.float32); v[r * n + c] = 1.0; return v


def train_policy(desc, seed, steps=80000, hidden=64):
    """PPO on the DETERMINISTIC dynamics (slip=0): learns a confident shortest path."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    n, holes, goal, start = parse_map(desc); n2 = n * n
    actor = MLPActor(n2, 4, hidden); critic = nn.Sequential(nn.Linear(n2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-3)
    r_, c_ = start; done_steps = 0
    while done_steps < steps:
        O, A, LP, V, R, D = [], [], [], [], [], []
        for _ in range(512):
            o = torch.as_tensor(onehot(r_, c_, n)).unsqueeze(0)
            with torch.no_grad():
                lg = actor(o).squeeze(0); v = critic(o).item()
                dist = torch.distributions.Categorical(logits=lg); a = dist.sample(); lp = dist.log_prob(a).item()
            nr, nc, dn, rew = step(n, holes, goal, r_, c_, int(a.item()), 0.0, rng)
            rew = rew - 0.01                                  # step cost -> short paths
            O.append(onehot(r_, c_, n)); A.append(int(a.item())); LP.append(lp); V.append(v); R.append(rew); D.append(dn)
            r_, c_ = (start if dn else (nr, nc)); done_steps += 1
        with torch.no_grad(): lastv = critic(torch.as_tensor(onehot(r_, c_, n)).unsqueeze(0)).item()
        R = np.array(R); V = np.array(V + [lastv]); D = np.array(D, float); adv = np.zeros(len(R)); gae = 0
        for t in reversed(range(len(R))):
            nt = 1 - D[t]; delta = R[t] + 0.99 * V[t + 1] * nt - V[t]; gae = delta + 0.95 * 0.99 * nt * gae; adv[t] = gae
        ret = adv + V[:-1]; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        O = torch.as_tensor(np.array(O)); A = torch.as_tensor(A); LP = torch.as_tensor(LP)
        ADV = torch.as_tensor(adv, dtype=torch.float32); RET = torch.as_tensor(ret, dtype=torch.float32); idx = np.arange(len(O))
        for _ in range(4):
            rng.shuffle(idx)
            for s0 in range(0, len(idx), 128):
                b = idx[s0:s0 + 128]; dist = torch.distributions.Categorical(logits=actor(O[b]))
                ratio = torch.exp(dist.log_prob(A[b]) - LP[b])
                pl = -torch.min(ratio * ADV[b], torch.clamp(ratio, .8, 1.2) * ADV[b]).mean()
                loss = pl + 0.5 * F.mse_loss(critic(O[b]).squeeze(-1), RET[b]) - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()
    return actor, n


def fisher_diag(actor, n):
    fisher = {k: torch.zeros_like(v) for k, v in actor.named_parameters()}
    for s in range(n * n):
        o = torch.zeros(1, n * n); o[0, s] = 1.0; actor.zero_grad(set_to_none=True)
        lg = actor(o).squeeze(0); F.log_softmax(lg, -1)[int(lg.argmax())].backward()
        for k, p in actor.named_parameters():
            if p.grad is not None: fisher[k] += p.grad.detach() ** 2
    actor.zero_grad(set_to_none=True); gm = max((f.max().item() for f in fisher.values()), default=1.) or 1.
    return {k: (v / gm).clamp(min=1e-3) for k, v in fisher.items()}


def eval_policy(actor, n, desc, mode, slip, n_ep, seed, gain=3.0, eta_w=0.05, kappa=4.0, rho=0.9,
                eta_h=0.1, sigma=0.05, thresh=0.05):
    _, holes, goal, start = parse_map(desc)
    w0 = {k: v.detach().clone() for k, v in actor.named_parameters()}; fisher = fisher_diag(actor, n)
    rng = np.random.default_rng(seed + 12345); holes_hit, succ = [], []
    for ep in range(n_ep):
        with torch.no_grad():
            for k, p in actor.named_parameters(): p.data.copy_(w0[k])
        r_, c_ = start; sup = np.zeros(4); hole = 0; ok = 0
        for t in range(4 * n):
            with torch.no_grad(): lg = actor(torch.as_tensor(onehot(r_, c_, n)).unsqueeze(0)).squeeze(0)
            greedy = int(lg.argmax())
            Rg = risk_fn(n, holes, r_, c_, greedy, slip); S = max(risk_fn(n, holes, r_, c_, a, slip) for a in range(4)); alpha = S * Rg
            if mode == "raw": a = greedy
            elif mode == "override": a = min(range(4), key=lambda k: risk_fn(n, holes, r_, c_, k, slip)) if S > thresh else greedy
            elif mode == "mask_defer":
                if alpha > thresh: lg2 = lg.clone(); lg2[greedy] = -1e9; a = int(lg2.argmax())
                else: a = greedy
            elif mode == "asra":
                if alpha > thresh and gain > 0:
                    actor.zero_grad(set_to_none=True)
                    lg2 = actor(torch.as_tensor(onehot(r_, c_, n)).unsqueeze(0)).squeeze(0)
                    F.log_softmax(lg2, -1)[greedy].backward()
                    with torch.no_grad():
                        for k, p in actor.named_parameters():
                            if p.grad is None: continue
                            epi = int(p.grad.data.abs().flatten().argmax()); apply_gaussian_(p, p.grad.data, epi, sigma, eta_w * gain * alpha)
                    actor.zero_grad(set_to_none=True); sup[greedy] -= kappa * gain * alpha
                with torch.no_grad():
                    adj = actor(torch.as_tensor(onehot(r_, c_, n)).unsqueeze(0)).squeeze(0) + torch.tensor(sup, dtype=lg.dtype)
                a = int(adj.argmax()); sup *= rho
                with torch.no_grad():
                    for k, p in actor.named_parameters(): p.data += eta_h * fisher[k] * (w0[k] - p.data)
            else: raise ValueError(mode)
            nr, nc, dn, rew = step(n, holes, goal, r_, c_, a, slip, rng); r_, c_ = nr, nc
            if dn:
                if rew > 0: ok = 1
                else: hole = 1
                break
        holes_hit.append(hole); succ.append(ok)
    return {"hole_rate": float(np.mean(holes_hit)), "success": float(np.mean(succ))}


def run(out_path, n_policies, n_ep, size, hole_p, train_steps, slips=(0.1, 0.2, 0.34), gains=(1., 3., 6.)):
    t0 = time.time(); results = []
    for pi in range(n_policies):
        desc = generate_random_map(size=size, p=hole_p, seed=1000 + pi)
        actor, n = train_policy(desc, seed=pi, steps=train_steps)
        for slip in slips:
            modes = [("raw", {}), ("override", {}), ("mask_defer", {})] + [(f"asra_g{g:g}", {"gain": g}) for g in gains]
            for name, kw in modes:
                m = "asra" if name.startswith("asra") else name
                r = eval_policy(actor, n, desc, m, slip, n_ep, seed=pi, **kw)
                results.append({"policy": pi, "slip": slip, "cond": name, **r})
        pr = {(x["slip"], x["cond"]): x for x in results if x["policy"] == pi}
        msg = " ".join(f"slip{s}:raw{pr[(s,'raw')]['hole_rate']:.2f}/ovr{pr[(s,'override')]['success']:.2f}/asra_g3succ{pr[(s,'asra_g3')]['success']:.2f}" for s in slips)
        print(f"  policy {pi+1}/{n_policies}: {msg} [{time.time()-t0:.0f}s]", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results, "n_policies": n_policies, "n_ep": n_ep, "size": size, "hole_p": hole_p, "slips": list(slips)}, open(out_path, "w"), indent=2)
    print(f"saved {out_path} ({time.time()-t0:.0f}s)", flush=True); return results


def analyze(path, competent_thresh=0.3):
    d = json.load(open(path)); npol = d["n_policies"]; slips = sorted(d["slips"])
    by = {}
    for r in d["results"]: by.setdefault((r["policy"], r["slip"]), {})[r["cond"]] = r
    mild = slips[0]
    # COMPETENCE FILTER: keep only policies that actually transferred (raw success at mildest slip
    # >= thresh). A det-trained policy on an unsolvable-under-slip map is not a valid test of the
    # precondition (there is no policy competence to preserve); we report how many were dropped.
    competent = [p for p in range(npol) if by[(p, mild)]["raw"]["success"] >= competent_thresh]
    print(f"=== FrozenLake (2nd simulator): {len(competent)}/{npol} competent policies (raw succ>= {competent_thresh} at slip {mild}); precondition vs deploy-slip ===")
    def best_asra(p, slip, ref_hole):  # best ASRA success at hole_rate <= ref (matched safety)
        cand = [v["success"] for c, v in by[(p, slip)].items() if c.startswith("asra") and v["hole_rate"] <= ref_hole + 1e-9]
        return max(cand) if cand else np.nan
    for slip in slips:
        raw_h = np.array([by[(p, slip)]["raw"]["hole_rate"] for p in competent])
        ovr = np.array([by[(p, slip)]["override"]["success"] for p in competent])
        mask = np.array([by[(p, slip)]["mask_defer"]["success"] for p in competent])
        mask_h = np.array([by[(p, slip)]["mask_defer"]["hole_rate"] for p in competent])
        # ASRA at matched hole-rate vs the override; and routing(mask) vs override
        a_vs_o = np.array([best_asra(p, slip, by[(p, slip)]["override"]["hole_rate"]) for p in competent])
        okA = ~np.isnan(a_vs_o)
        print(f"  slip={slip:.2f}: raw hole={raw_h.mean():.2f} | override succ={ovr.mean():.2f} | mask succ={mask.mean():.2f} (hole={mask_h.mean():.2f})")
        if okA.sum() >= 3:
            dao = paired_bootstrap_diff(a_vs_o[okA], ovr[okA]); s1 = int(np.sum(a_vs_o[okA] > ovr[okA] + 1e-9))
            v1 = "routing>override (precondition MET)" if dao["excludes_zero"] and dao["diff"] > 0 else ("override wins" if dao["diff"] < -1e-9 and dao["excludes_zero"] else "tie")
            print(f"        ASRA(matched-hole) succ - override succ: d={dao['diff']:+.2f} CI[{dao['lo']:+.2f},{dao['hi']:+.2f}] ({s1}/{okA.sum()}) => {v1}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "frozenlake_asra.json"))
    ap.add_argument("--policies", type=int, default=8)
    ap.add_argument("--n_ep", type=int, default=200)
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--hole_p", type=float, default=0.88)
    ap.add_argument("--train_steps", type=int, default=80000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        run(str(RES / "_frozenlake_smoke.json"), n_policies=2, n_ep=60, size=6, hole_p=0.85, train_steps=20000, slips=(0.1, 0.34))
        analyze(str(RES / "_frozenlake_smoke.json")); print("[SMOKE] OK")
    else:
        run(a.out, a.policies, a.n_ep, a.size, a.hole_p, a.train_steps)
        analyze(a.out)

"""DYNAMIC context-shift experiment: the deployment context SHIFTS several times WITHIN one"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.environment.highway_wrapper import HighwayFRAEnv
from current_work.asra_fast.core import compute_risk, ASRA, ASRAConfig, CostSalience, SalienceConfig
from current_work.asra_fast.runner import load_actor, clone_state, restore_state, w0_named
from current_work.asra_fast.analyze import paired_bootstrap_diff


def hormone_action(actor, obs, cost, ttc, H, kappa=6.0, temp=1.0, device="cpu"):
    with torch.no_grad():
        logits = actor(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)
    risk_vec = torch.tensor([compute_risk(cost, ttc, a) for a in range(4)], dtype=logits.dtype)
    adj = logits / temp - H * kappa * risk_vec
    return int(torch.distributions.Categorical(logits=adj).sample())


def episode(actor, env, w0_full, seed_ep, mode, H, asra, max_steps=120, device="cpu"):
    if mode == "naive_asra":
        restore_state(actor, w0_full); asra.reset()
    obs, info = env.reset(seed=seed_ep); cost, ttc = info.get("cost", 0.), info.get("ttc", 10.)
    col, sp, L = 0, 0.0, 0
    for t in range(max_steps):
        if mode == "naive_asra":
            a, _ = asra.act(actor, obs, cost, ttc)
        else:
            a = hormone_action(actor, obs, cost, ttc, H, device=device)
        sp += float(obs[2]); L += 1
        obs, r, term, trunc, info = env.step(a); cost, ttc = info.get("cost", 0.), info.get("ttc", 10.)
        if term: col = int(info.get("collision", False)); break
        if trunc: break
    return col, sp / max(L, 1)


def calibrate(actor, w0_full, w0_par, fisher, density, vehicles, target, n_ep, seed,
              mode, device="cpu"):
    """Tune the home-context control level (hormone H, or naive-ASRA gain) to hit the setpoint."""
    grid = np.linspace(0, 1.5, 7) if mode != "naive_asra" else np.linspace(0, 6, 7)
    env = HighwayFRAEnv(seed=seed, vehicles_count=vehicles, vehicles_density=density)
    best = (grid[0], 1.0)
    for v in grid:
        asra = ASRA(ASRAConfig(gain=float(v), select="sample"), w0_par, fisher,
                    CostSalience(SalienceConfig()), device=device) if mode == "naive_asra" else None
        crs = [episode(actor, env, w0_full, seed*100000+e, mode, float(v), asra, device=device)[0]
               for e in range(n_ep)]
        d = abs(np.mean(crs) - target)
        if d < best[1]: best = (float(v), d)
    env.close()
    return best[0]


def run_stream(actor, w0_full, w0_par, fisher, schedule, mode, ctrl0, target, seed,
               vehicles, lr=1.2, device="cpu"):
    """Continuous episode stream across the shifting schedule. Returns per-episode records."""
    recs = []
    cr_est = target
    H = ctrl0
    ep_global = 0
    cur_density = None; env = None
    asra = None
    for (density, n_ep) in schedule:
        if density != cur_density:
            if env is not None: env.close()
            env = HighwayFRAEnv(seed=seed, vehicles_count=vehicles, vehicles_density=density)
            cur_density = density
        for _ in range(n_ep):
            if mode == "online":
                H = float(np.clip(H + lr * (cr_est - target), 0.0, 1.5))
                ctrl = H
                asra = None
            elif mode == "fixed":
                ctrl = ctrl0; asra = None
            else:  # naive_asra: fixed gain
                ctrl = ctrl0
                asra = ASRA(ASRAConfig(gain=ctrl0, select="sample"), w0_par, fisher,
                            CostSalience(SalienceConfig()), device=device)
            col, spd = episode(actor, env, w0_full, seed*100000+ep_global, mode, ctrl, asra, device=device)
            cr_est = 0.85 * cr_est + 0.15 * col
            recs.append({"ep": ep_global, "density": density, "collision": col,
                         "ctrl": ctrl, "cr_est": cr_est, "speed": spd})
            ep_global += 1
    if env is not None: env.close()
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--vehicles", type=int, default=8)
    ap.add_argument("--target", type=float, default=0.35)
    ap.add_argument("--home", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seg", type=int, default=30)   # episodes per context segment
    ap.add_argument("--out", default="current_work/results_v3/context_shift.json")
    a = ap.parse_args()
    actor, _ = load_actor(a.base)
    w0_full = clone_state(actor); w0_par = w0_named(actor)
    fisher = {k: torch.ones_like(v) for k, v in actor.named_parameters()}
    # a schedule that shifts the context up and down several times (home first)
    densities = [a.home, 1.6, 0.7, 1.4, a.home]
    schedule = [(d, a.seg) for d in densities]

    out = {"target": a.target, "home": a.home, "vehicles": a.vehicles, "seg": a.seg,
           "densities": densities, "methods": {}}
    for mode in ["naive_asra", "fixed", "online"]:
        seg_err = {d: [] for d in set(densities)}
        streams = []
        for s in range(a.seeds):
            ctrl0 = calibrate(actor, w0_full, w0_par, fisher, a.home, a.vehicles, a.target,
                              n_ep=max(20, a.seg), seed=1000+s, mode=mode)
            recs = run_stream(actor, w0_full, w0_par, fisher, schedule, mode, ctrl0, a.target,
                              seed=s, vehicles=a.vehicles)
            streams.append(recs)
            # per-segment |CR - target| (use 2nd half of each segment = post-adaptation)
            i = 0
            for (d, n) in schedule:
                seg = recs[i:i+n]; i += n
                half = [r["collision"] for r in seg[n//2:]]
                seg_err[d].append(abs(np.mean(half) - a.target))
        # aggregate mean |err| over all shifted (non-home) segments
        shifted = [e for d in seg_err for e in seg_err[d] if d != a.home]
        out["methods"][mode] = {"seg_err": {str(k): v for k, v in seg_err.items()},
                                "shifted_err": shifted, "stream_seed0": streams[0]}
        print(f"  {mode:10s}: mean |CR-target| on shifted contexts = {np.mean(shifted):.3f}", flush=True)

    # online vs naive/fixed: does the feedback loop track the setpoint better under shift?
    for base in ["naive_asra", "fixed"]:
        fe = np.array(out["methods"][base]["shifted_err"]); oe = np.array(out["methods"]["online"]["shifted_err"])
        n = min(len(fe), len(oe)); dif = paired_bootstrap_diff(fe[:n], oe[:n])
        verdict = "ONLINE tracks better" if dif["diff"] > 0 and dif["excludes_zero"] else "inconclusive"
        print(f"  |err| {base}={fe.mean():.3f} vs online={oe.mean():.3f}  "
              f"(diff={dif['diff']:+.3f} CI[{dif['lo']:+.3f},{dif['hi']:+.3f}]) => {verdict}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("saved", a.out)

"""NEW, self-contained multi-scenario experiment (does NOT edit the existing harness)."""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import multiprocessing as mp
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.environment.highway_wrapper import HighwayFRAEnv
from current_work.asra_fast.core import (
    MLPActor, ASRA, ASRAConfig, CostSalience, SalienceConfig,
    compute_risk, baseline_action, apply_gaussian_,
)
from current_work.asra_fast.analyze import frontier_auc, paired_bootstrap_diff, spearman_ci

BP = Path("current_work/base_policies"); RES = Path("current_work/results_v3")


# ---- inline PPO (self-contained; scenario-aware) ----
def train_scn(scenario, steps, seed=0, hidden=64, vehicles=8, density=1.0, hsr=0.6, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = torch.device(device)
    actor = MLPActor(12, 4, hidden).to(dev)
    critic = nn.Sequential(nn.Linear(12, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                           nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    env = HighwayFRAEnv(scenario=scenario, seed=seed, vehicles_count=vehicles,
                        vehicles_density=density, high_speed_reward=hsr)
    obs, info = env.reset(seed=seed); done = 0; ns = 1024; g, lam, clip = 0.99, 0.95, 0.2
    steps_done = 0; recent = []
    while steps_done < steps:
        O, A, LP, V, R, D = [], [], [], [], [], []
        for _ in range(ns):
            o = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            with torch.no_grad():
                logits = actor(o).squeeze(0); v = critic(o).item()
                dist = torch.distributions.Categorical(logits=logits); a = dist.sample(); lp = dist.log_prob(a).item()
            no, r, term, trunc, info = env.step(int(a.item())); dn = term or trunc
            O.append(obs); A.append(int(a.item())); LP.append(lp); V.append(v); R.append(r); D.append(dn)
            obs = no; steps_done += 1
            if dn:
                recent.append(int(info.get("collision", False)));  recent[:] = recent[-200:]
                obs, info = env.reset(seed=int(rng.integers(1, 1_000_000)))
        with torch.no_grad():
            lastv = critic(torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)).item()
        R = np.array(R); V = np.array(V + [lastv]); D = np.array(D, float); adv = np.zeros(len(R)); gae = 0
        for t in reversed(range(len(R))):
            nt = 1 - D[t]; delta = R[t] + g * V[t+1] * nt - V[t]; gae = delta + g * lam * nt * gae; adv[t] = gae
        ret = adv + V[:-1]; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        O = torch.as_tensor(np.array(O), dtype=torch.float32, device=dev); A = torch.as_tensor(A, device=dev)
        LP = torch.as_tensor(LP, device=dev); ADV = torch.as_tensor(adv, dtype=torch.float32, device=dev)
        RET = torch.as_tensor(ret, dtype=torch.float32, device=dev); idx = np.arange(len(O))
        for _ in range(4):
            rng.shuffle(idx)
            for s in range(0, len(idx), 256):
                b = idx[s:s+256]; dist = torch.distributions.Categorical(logits=actor(O[b]))
                ratio = torch.exp(dist.log_prob(A[b]) - LP[b])
                pl = -torch.min(ratio*ADV[b], torch.clamp(ratio, 1-clip, 1+clip)*ADV[b]).mean()
                vl = F.mse_loss(critic(O[b]).squeeze(-1), RET[b])
                loss = pl + 0.5*vl - 0.01*dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()
    env.close()
    out = BP / f"scn_{scenario}_seed{seed}_final.pt"
    torch.save({"actor": {k: v.cpu().clone() for k, v in actor.state_dict().items()}, "hidden": hidden,
                "scenario": scenario, "vehicles": vehicles, "density": density, "high_speed_reward": hsr,
                "train_cr": float(np.mean(recent) if recent else float("nan"))}, out)
    return str(out)


def fisher_scn(actor, n=1500):
    fisher = {name: torch.zeros_like(p) for name, p in actor.named_parameters()}
    states = np.random.default_rng(0).standard_normal((n, 12)).astype(np.float32) * 5
    for s in states[:n]:
        o = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0); actor.zero_grad(set_to_none=True)
        lg = actor(o).squeeze(0); F.log_softmax(lg, -1)[int(lg.argmax())].backward()
        for nm, p in actor.named_parameters():
            if p.grad is not None: fisher[nm] += p.grad.detach() ** 2
    actor.zero_grad(set_to_none=True)
    gmax = max((f.max().item() for f in fisher.values()), default=1.0) or 1.0
    return {k: (v / gmax).clamp(min=1e-3) for k, v in fisher.items()}


# ---- scenario-aware eval worker (own pool; mirrors experiments.py but self-contained) ----
_G = {}
def _init(base_path, salience_kind, max_steps):
    torch.set_num_threads(1)
    d = torch.load(base_path, map_location="cpu", weights_only=False)
    _G.update(state=d["actor"], hidden=d.get("hidden", 64), scenario=d.get("scenario", "highway"),
              vehicles=d.get("vehicles", 8), density=d.get("density", 1.0),
              hsr=d.get("high_speed_reward", 0.6), max_steps=max_steps)
    actor = MLPActor(12, 4, _G["hidden"]); actor.load_state_dict(_G["state"])
    _G["fisher"] = fisher_scn(actor); _G["sal"] = CostSalience(SalienceConfig())


def _fresh_actor():
    a = MLPActor(12, 4, _G["hidden"]); a.load_state_dict(_G["state"])
    for p in a.parameters(): p.requires_grad_(True)
    return a


def _eval_one(task):
    actor = _fresh_actor()
    w0f = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    w0p = {k: v.detach().clone() for k, v in actor.named_parameters()}
    env = HighwayFRAEnv(scenario=_G["scenario"], seed=task["seed"], vehicles_count=_G["vehicles"],
                        vehicles_density=_G["density"], high_speed_reward=_G["hsr"])
    rng = np.random.default_rng(task["seed"] + 7)
    asra = None
    if task["method"] == "asra":
        asra = ASRA(ASRAConfig(gain=task["gain"], select=task["decode"]), w0p, _G["fisher"],
                    _G["sal"], device="cpu", rng=np.random.default_rng(task["seed"] + 77))
    crs, sps = [], []
    for ep in range(task["n_ep"]):
        with torch.no_grad():
            for k, p in actor.named_parameters(): p.data.copy_(w0f[k])
        if asra: asra.reset()
        obs, info = env.reset(seed=task["seed"] * 100000 + ep); cost, ttc = info.get("cost", 0.), info.get("ttc", 10.)
        col, sp, L = 0, 0.0, 0
        for t in range(_G["max_steps"]):
            if asra:
                a, _ = asra.act(actor, obs, cost, ttc)
            else:
                with torch.no_grad():
                    lg = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                a = baseline_action(task["kind"], lg, cost, ttc, rng, ttc_k=task.get("ttc_k", 2.),
                                    temp=task.get("temp", 2.), base_decode="sample")
            sp += float(obs[2]); L += 1
            obs, r, term, trunc, info = env.step(a); cost, ttc = info.get("cost", 0.), info.get("ttc", 10.)
            if term: col = int(info.get("collision", False)); break
            if trunc: break
        crs.append(col); sps.append(sp / max(L, 1))
    env.close()
    return {**{k: task[k] for k in task if k != "n_ep"}, "cr": float(np.mean(crs)), "speed": float(np.mean(sps))}


def build_tasks(seeds, n_ep, g_grid, ttc_grid, pbrake_grid):
    T = []
    for s in seeds:
        for dm in ("greedy", "sample"):
            for g in g_grid: T.append({"method": "asra", "gain": float(g), "decode": dm, "seed": int(s), "n_ep": n_ep})
        T.append({"method": "baseline", "kind": "noop_greedy", "seed": int(s), "n_ep": n_ep})
        T.append({"method": "baseline", "kind": "noop_sample", "seed": int(s), "n_ep": n_ep})
        for k in ttc_grid: T.append({"method": "baseline", "kind": "ttc_brake", "ttc_k": float(k), "seed": int(s), "n_ep": n_ep})
        for p in pbrake_grid: T.append({"method": "baseline", "kind": "prob_brake", "temp": float(p), "seed": int(s), "n_ep": n_ep})
    return T


def frontier(base_path, out_path, seeds, n_ep, n_proc, max_steps,
             g_grid=(0., 0.5, 1., 2., 3., 5., 8.), ttc_grid=(0., 1., 2., 3., 4.),
             pbrake_grid=(0., 0.1, 0.3, 0.6, 1.)):
    d = torch.load(base_path, map_location="cpu", weights_only=False)
    tasks = build_tasks(list(seeds), n_ep, g_grid, ttc_grid, pbrake_grid)
    print(f"[{d.get('scenario')}] {len(tasks)} tasks x {n_ep} eps on {n_proc} procs (base CR~{d.get('train_cr')})", flush=True)
    t0 = time.time()
    with mp.Pool(n_proc, initializer=_init, initargs=(base_path, "cost", max_steps)) as pool:
        results = list(pool.imap_unordered(_eval_one, tasks, chunksize=1))
    out = {"scenario": d.get("scenario"), "base_path": base_path, "seeds": list(map(int, seeds)),
           "n_ep": n_ep, "g_grid": list(g_grid), "ttc_grid": list(ttc_grid), "pbrake_grid": list(pbrake_grid),
           "results": results, "elapsed_s": time.time() - t0}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[{d.get('scenario')}] saved {out_path} ({time.time()-t0:.0f}s)", flush=True)
    return out


def analyze(path):
    from collections import defaultdict
    d = json.load(open(path)); seeds = sorted(set(r["seed"] for r in d["results"]))
    G = defaultdict(dict)
    for r in d["results"]:
        key = ("asra", round(r["gain"], 3), r["decode"]) if r["method"] == "asra" else \
              (("ttc", round(r.get("ttc_k", 0), 3)) if r["kind"] == "ttc_brake" else
               (("pb", round(r.get("temp", 0), 3)) if r["kind"] == "prob_brake" else (r["kind"],)))
        G[key][r["seed"]] = r
    ng = G.get(("noop_greedy",), {})
    base_cr = np.mean([ng[s]["cr"] for s in ng]) if ng else float("nan")
    # monotonicity of CR vs g (greedy)
    gs, crs = [], []
    for g in d["g_grid"]:
        m = G.get(("asra", round(g, 3), "greedy"), {})
        for s in m: gs.append(g); crs.append(m[s]["cr"])
    sp = spearman_ci(np.array(gs), np.array(crs)) if len(set(gs)) > 1 else {"rho": float("nan"), "lo": 0, "hi": 0, "excludes_zero": False}
    # useful-region frontier AUC: ASRA vs rule (shared global floor)
    cap = float(max(0.3, np.mean([G[("noop_sample",)][s]["cr"] for s in G.get(("noop_sample",), {})]) if G.get(("noop_sample",)) else 0.3))
    grid = np.linspace(0, cap, 31)
    def pts(getter, s):
        return [(getter[k][s]["cr"], getter[k][s]["speed"]) for k in getter if s in getter[k]]
    def method_pts(keys, s):
        out = []
        for k in keys:
            m = G.get(k, {})
            if s in m: out.append((m[s]["cr"], m[s]["speed"]))
        return out
    asra_keys = [("asra", round(g, 3), dm) for g in d["g_grid"] for dm in ("greedy", "sample")]
    rule_keys = [("pb", round(p, 3)) for p in d["pbrake_grid"]] + [("ttc", round(k, 3)) for k in d["ttc_grid"]] + [("noop_greedy",), ("noop_sample",)]
    aa, ra = [], []
    for s in seeds:
        ap, rp = method_pts(asra_keys, s), method_pts(rule_keys, s)
        floor = min([p[1] for p in ap + rp], default=0.)
        if len(ap) >= 2: aa.append(frontier_auc(ap, grid, floor=floor))
        if len(rp) >= 2: ra.append(frontier_auc(rp, grid, floor=floor))
    aa, ra = np.array(aa), np.array(ra); n = min(len(aa), len(ra))
    dom = paired_bootstrap_diff(aa[:n], ra[:n]) if n >= 3 else {"diff": float("nan"), "lo": 0, "hi": 0, "excludes_zero": False}
    verdict = "ASRA dominates" if dom["diff"] > 0 and dom["excludes_zero"] else ("rule dominates" if dom["diff"] < 0 and dom["excludes_zero"] else "parity")
    print(f"=== {d['scenario']} | {len(seeds)} seeds x {d['n_ep']} eps ===")
    print(f"  raw-greedy CR={base_cr:.3f} | H2 CR-vs-g(greedy) rho={sp['rho']:+.3f} CI[{sp['lo']:.2f},{sp['hi']:.2f}]")
    print(f"  H3 useful-AUC(CR<={cap:.2f}): ASRA={aa.mean():.2f} rule={ra.mean():.2f} d={dom['diff']:+.2f} CI[{dom['lo']:.2f},{dom['hi']:.2f}] => {verdict}")
    return {"scenario": d["scenario"], "base_cr": base_cr, "rho": sp, "auc_asra": float(aa.mean()),
            "auc_rule": float(ra.mean()), "dom": dom, "verdict": verdict, "cap": cap}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="merge")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--n_ep", type=int, default=40)
    ap.add_argument("--n_proc", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--vehicles", type=int, default=8)
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        print(f"[SMOKE] train tiny {a.scenario} + 1-seed frontier ...")
        bp = train_scn(a.scenario, steps=6000, seed=0, vehicles=a.vehicles, density=a.density)
        frontier(bp, f"{RES}/_scn_{a.scenario}_smoke.json", seeds=[0], n_ep=6, n_proc=4, max_steps=60,
                 g_grid=(0., 2., 8.), ttc_grid=(0., 2.), pbrake_grid=(0., 1.))
        analyze(f"{RES}/_scn_{a.scenario}_smoke.json"); print("[SMOKE] OK")
    else:
        bp = train_scn(a.scenario, steps=a.steps, seed=0, vehicles=a.vehicles, density=a.density)
        frontier(bp, f"{RES}/scn_{a.scenario}.json", seeds=list(range(a.seeds)), n_ep=a.n_ep,
                 n_proc=a.n_proc, max_steps=a.max_steps)
        analyze(f"{RES}/scn_{a.scenario}.json")

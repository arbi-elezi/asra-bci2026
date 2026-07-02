"""NEW self-contained multi-policy frontier with the mask-and-defer baseline."""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
import numpy as np, torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.environment.highway_wrapper import HighwayFRAEnv
from current_work.asra_fast.core import (MLPActor, ASRA, ASRAConfig, CostSalience, SalienceConfig,
                                         compute_risk, baseline_action, apply_gaussian_)
from current_work.asra_fast.analyze import paired_bootstrap_diff, spearman_ci, holm_bonferroni
RES = Path("current_work/results_v3")
_G = {}


def _fisher(actor, n=1200):
    fisher = {k: torch.zeros_like(v) for k, v in actor.named_parameters()}
    st = np.random.default_rng(0).standard_normal((n, 12)).astype(np.float32) * 5
    for s in st:
        o = torch.as_tensor(s).unsqueeze(0); actor.zero_grad(set_to_none=True)
        lg = actor(o).squeeze(0); F.log_softmax(lg, -1)[int(lg.argmax())].backward()
        for k, p in actor.named_parameters():
            if p.grad is not None: fisher[k] += p.grad.detach() ** 2
    actor.zero_grad(set_to_none=True)
    gm = max((f.max().item() for f in fisher.values()), default=1.) or 1.
    return {k: (v / gm).clamp(min=1e-3) for k, v in fisher.items()}


def _init(base_path, max_steps, eval_vehicles, eval_density):
    torch.set_num_threads(1)
    d = torch.load(base_path, map_location="cpu", weights_only=False)
    _G.update(state=d["actor"], hidden=d.get("hidden", 64),
              scenario=d.get("scenario", "highway"),
              veh=eval_vehicles or d.get("vehicles", 8),
              den=eval_density or d.get("density", 1.0),
              hsr=d.get("high_speed_reward", 0.4), max_steps=max_steps)
    a = MLPActor(12, 4, _G["hidden"]); a.load_state_dict(_G["state"]); _G["fisher"] = _fisher(a)
    _G["sal"] = CostSalience(SalienceConfig())


def _fresh():
    a = MLPActor(12, 4, _G["hidden"]); a.load_state_dict(_G["state"])
    for p in a.parameters(): p.requires_grad_(True)
    return a


def _eval_one(task):
    actor = _fresh()
    w0f = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    w0p = {k: v.detach().clone() for k, v in actor.named_parameters()}
    env = HighwayFRAEnv(scenario=_G["scenario"], seed=task["seed"], vehicles_count=_G["veh"],
                        vehicles_density=_G["den"], high_speed_reward=_G["hsr"])
    rng = np.random.default_rng(task["seed"] + 7)
    asra = ASRA(ASRAConfig(gain=task["gain"], select=task["decode"],
                           weight_channel=task.get("weight_channel", True),   # H4 ablation toggles (default = full ASRA)
                           conf_channel=task.get("conf_channel", True)),
                w0p, _G["fisher"], _G["sal"],
                device="cpu", rng=np.random.default_rng(task["seed"] + 77)) if task["method"] == "asra" else None
    crs, sps = [], []
    for ep in range(task["n_ep"]):
        with torch.no_grad():
            for k, p in actor.named_parameters(): p.data.copy_(w0f[k])
        if asra: asra.reset()
        obs, info = env.reset(seed=task["seed"] * 100000 + ep); cost, ttc = info.get("cost", 0.), info.get("ttc", 10.)
        col, sp, L = 0, 0.0, 0
        for t in range(_G["max_steps"]):
            with torch.no_grad():
                lg = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            greedy = int(lg.argmax())
            if task["method"] == "asra":
                a, _ = asra.act(actor, obs, cost, ttc)
            elif task["method"] == "mask_defer":                 # block risky greedy under salience, defer to 2nd choice
                S = _G["sal"](obs, cost, ttc); Rk = compute_risk(cost, ttc, greedy)
                lg2 = lg.clone()
                if S * Rk > 0.05: lg2[greedy] = -1e9
                a = int(lg2.argmax())
            else:
                a = baseline_action(task["kind"], lg, cost, ttc, rng, ttc_k=task.get("ttc_k", 2.),
                                    temp=task.get("temp", 2.), base_decode="sample")
            sp += float(obs[2]); L += 1
            obs, r, term, trunc, info = env.step(a); cost, ttc = info.get("cost", 0.), info.get("ttc", 10.)
            if term: col = int(info.get("collision", False)); break
            if trunc: break
        crs.append(col); sps.append(sp / max(L, 1))
    env.close()
    return {**{k: task[k] for k in task if k != "n_ep"}, "policy": task["policy"],
            "cr": float(np.mean(crs)), "speed": float(np.mean(sps))}


def build(seeds, n_ep, g_grid, ttc_grid, pbrake_grid, pol_idx):
    T = []
    for s in seeds:
        base = {"seed": int(s), "n_ep": n_ep, "policy": pol_idx}
        for dm in ("greedy", "sample"):
            for g in g_grid: T.append({**base, "method": "asra", "gain": float(g), "decode": dm})
        T.append({**base, "method": "mask_defer", "gain": 0.0})
        T.append({**base, "method": "baseline", "kind": "noop_greedy"})
        T.append({**base, "method": "baseline", "kind": "noop_sample"})
        for k in ttc_grid: T.append({**base, "method": "baseline", "kind": "ttc_brake", "ttc_k": float(k)})
        for p in pbrake_grid: T.append({**base, "method": "baseline", "kind": "prob_brake", "temp": float(p)})
    return T


def run(policies, out_path, seeds, n_ep, n_proc, max_steps, eval_vehicles=None, eval_density=None,
        g_grid=(0., 0.5, 1., 2., 3., 5., 8.), ttc_grid=(0., 1., 2., 3., 4.), pbrake_grid=(0., 0.1, 0.3, 0.6, 1.)):
    t0 = time.time(); all_results = []
    for pi, base in enumerate(policies):
        tasks = build(list(seeds), n_ep, g_grid, ttc_grid, pbrake_grid, pi)
        with mp.Pool(n_proc, initializer=_init, initargs=(base, max_steps, eval_vehicles, eval_density)) as pool:
            all_results += list(pool.imap_unordered(_eval_one, tasks, chunksize=1))
        print(f"  policy {pi+1}/{len(policies)} done ({time.time()-t0:.0f}s)", flush=True)
    out = {"policies": policies, "seeds": list(map(int, seeds)), "n_ep": n_ep,
           "g_grid": list(g_grid), "ttc_grid": list(ttc_grid), "pbrake_grid": list(pbrake_grid),
           "eval_vehicles": eval_vehicles, "eval_density": eval_density, "results": all_results,
           "elapsed_s": time.time() - t0}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(out_path, "w"), indent=2)
    print(f"saved {out_path} ({time.time()-t0:.0f}s)", flush=True); return out


def analyze(path, cr_levels=(0.3, 0.5, 0.7)):
    """Matched-CR speed gaps across POLICIES (unit of replication = policy). Holm-Bonferroni."""
    d = json.load(open(path)); npol = len(d["policies"]); seeds = d["seeds"]
    # per (policy) aggregate: for each method-condition, mean cr/speed over eval seeds
    G = defaultdict(lambda: defaultdict(list))  # cond -> policy -> list of (cr,speed) over seeds
    for r in d["results"]:
        if r["method"] == "asra": key = ("asra", round(r["gain"], 3), r["decode"])
        elif r["method"] == "mask_defer": key = ("mask",)
        elif r["kind"] == "ttc_brake": key = ("ttc", round(r["ttc_k"], 3))
        elif r["kind"] == "prob_brake": key = ("pb", round(r["temp"], 3))
        else: key = (r["kind"],)
        G[key][r["policy"]].append((r["cr"], r["speed"]))
    def pol_pts(keys):  # per policy: list of (cr,speed) points forming that method's frontier
        out = {}
        for pi in range(npol):
            pts = []
            for k in keys:
                if pi in G[k]:
                    arr = np.array(G[k][pi]); pts.append((arr[:, 0].mean(), arr[:, 1].mean()))
            out[pi] = pts
        return out
    asra_keys = [("asra", round(g, 3), dm) for g in d["g_grid"] for dm in ("greedy", "sample")]
    mask_keys = [("mask",)]
    rule_keys = [("pb", round(p, 3)) for p in d["pbrake_grid"]] + [("ttc", round(k, 3)) for k in d["ttc_grid"]] + [("noop_greedy",), ("noop_sample",)]
    A, M, Rr = pol_pts(asra_keys), pol_pts(mask_keys + [("noop_greedy",)]), pol_pts(rule_keys)

    def best_speed_at(pts, tgt):  # best speed among points with cr<=tgt (else nan)
        c = [p[1] for p in pts if p[0] <= tgt + 1e-9]
        return max(c) if c else np.nan
    print(f"=== multi-policy ({npol} policies x {len(seeds)} eval seeds) ===")
    ng = {pi: np.mean([v[0] for v in G[("noop_greedy",)][pi]]) for pi in range(npol) if pi in G[("noop_greedy",)]}
    ns = {pi: np.mean([v[0] for v in G[("noop_sample",)][pi]]) for pi in range(npol) if pi in G[("noop_sample",)]}
    print(f"  raw-greedy CR per policy: {[round(ng[pi],2) for pi in sorted(ng)]}")
    print(f"  raw-sample CR per policy: {[round(ns[pi],2) for pi in sorted(ns)]}")
    pvals = {}
    for tgt in cr_levels:
        a = np.array([best_speed_at(A[pi], tgt) for pi in range(npol)])
        m = np.array([best_speed_at(M[pi], tgt) for pi in range(npol)])
        r = np.array([best_speed_at(Rr[pi], tgt) for pi in range(npol)])
        okAR = ~(np.isnan(a) | np.isnan(r)); okAM = ~(np.isnan(a) | np.isnan(m))
        if okAR.sum() >= 3:
            dar = paired_bootstrap_diff(a[okAR], r[okAR])   # ASRA - rule
            dam = paired_bootstrap_diff(a[okAM], m[okAM]) if okAM.sum() >= 3 else {"diff": float('nan'), "lo": 0, "hi": 0, "excludes_zero": False}
            print(f"  @CR<={tgt}: n={okAR.sum()} | ASRA-rule speed d={dar['diff']:+.2f} CI[{dar['lo']:+.2f},{dar['hi']:+.2f}]"
                  f" | ASRA-mask d={dam['diff']:+.2f} CI[{dam['lo']:+.2f},{dam['hi']:+.2f}]")
            # REAL bootstrap p-values (not placeholders): directional for ASRA>rule, two-sided for ASRA!=mask
            pvals[f"ASRA>rule@{tgt}"] = dar.get("p_greater", 1.0)
            pvals[f"ASRA!=mask@{tgt}"] = dam.get("pvalue", 1.0)
    hb = holm_bonferroni(pvals) if pvals else {}
    surv = [k for k, v in hb.items() if v["reject"]]
    print(f"  Holm-Bonferroni survivors (alpha=.05): {surv if surv else 'NONE'}")
    print("  => ASRA vs mask-and-defer: " + ("TIE at all matched CR (mask is an equivalent trivial baseline)"
          if not any(k.startswith('ASRA!=mask') and hb.get(k,{}).get('reject') for k in hb) else "differ somewhere"))
    return {"npol": npol, "pvals": pvals, "holm": hb}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--out", default=str(RES / "multipolicy_driving.json"))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n_ep", type=int, default=40)
    ap.add_argument("--n_proc", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--eval_vehicles", type=int, default=None)
    ap.add_argument("--eval_density", type=float, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        run(a.policies[:1], str(RES / "_mp_smoke.json"), seeds=[0, 1], n_ep=6, n_proc=4, max_steps=60,
            eval_vehicles=a.eval_vehicles, eval_density=a.eval_density,
            g_grid=(0., 2., 8.), ttc_grid=(0., 2.), pbrake_grid=(0., 1.))
        analyze(str(RES / "_mp_smoke.json")); print("[SMOKE] OK")
    else:
        run(a.policies, a.out, seeds=list(range(a.seeds)), n_ep=a.n_ep, n_proc=a.n_proc,
            max_steps=a.max_steps, eval_vehicles=a.eval_vehicles, eval_density=a.eval_density)
        analyze(a.out)

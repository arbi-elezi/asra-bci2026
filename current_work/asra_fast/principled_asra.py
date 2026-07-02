"""Consequence-shaped operator on frozen driving policies: shaped = logits - g*S*Qc vs override/mask/rule."""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.environment.highway_wrapper import HighwayFRAEnv
from src.agents.cost_critic import CostCriticNet
from current_work.asra_fast.core import MLPActor, baseline_action, compute_risk
from current_work.asra_fast.analyze import paired_bootstrap_diff
RES = Path("current_work/results_v3")
_G = {}


def _load_cc(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else (ck.get("state_dict", ck) if isinstance(ck, dict) else ck)
    cc = CostCriticNet(12, 4, 64); cc.load_state_dict(sd); cc.eval()
    for p in cc.parameters(): p.requires_grad_(False)
    return cc


def _init(base_path, cc_path, max_steps, veh, den):
    torch.set_num_threads(1)
    d = torch.load(base_path, map_location="cpu", weights_only=False)
    _G.update(state=d["actor"], hidden=d.get("hidden", 64), scenario=d.get("scenario", "highway"),
              veh=veh or d.get("vehicles", 15), den=den or d.get("density", 1.5),
              hsr=d.get("high_speed_reward", 0.4), max_steps=max_steps, cc=_load_cc(cc_path))


def _actor():
    a = MLPActor(12, 4, _G["hidden"]); a.load_state_dict(_G["state"]); a.eval()
    for p in a.parameters(): p.requires_grad_(False)
    return a


def _eval_one(task):
    actor = _actor(); cc = _G["cc"]
    env = HighwayFRAEnv(scenario=_G["scenario"], seed=task["seed"], vehicles_count=_G["veh"],
                        vehicles_density=_G["den"], high_speed_reward=_G["hsr"])
    rng = np.random.default_rng(task["seed"] + 7); crs, sps = [], []
    for ep in range(task["n_ep"]):
        obs, info = env.reset(seed=task["seed"] * 100000 + ep); cost, ttc = info.get("cost", 0.), info.get("ttc", 10.)
        col, sp, L = 0, 0.0, 0
        for t in range(_G["max_steps"]):
            o = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                lg = actor(o.unsqueeze(0)).squeeze(0)
                qc = cc.predict_state(o).squeeze(0)                 # [4] expected cost per action
            greedy = int(lg.argmax()); S = float(np.clip(cost, 0., 1.)); m = task["method"]
            if m == "shaped":
                shaped = lg - task["gain"] * S * (qc - qc.mean())   # centered/advantage form: well-posed global g
                a = int(shaped.argmax()) if task["decode"] == "greedy" else int(torch.distributions.Categorical(logits=shaped).sample())
            elif m == "mask_defer":
                Rk = compute_risk(cost, ttc, greedy); lg2 = lg.clone()
                if S * Rk > 0.05: lg2[greedy] = -1e9
                a = int(lg2.argmax())
            elif m == "override":                                  # blunt force-brake override under danger (paper baseline)
                a = 2 if S > 0.05 else greedy
            else:
                a = baseline_action(task["kind"], lg, cost, ttc, rng, ttc_k=task.get("ttc_k", 2.),
                                    temp=task.get("temp", 2.), base_decode="sample")
            sp += float(obs[2]); L += 1
            obs, r, term, trunc, info = env.step(a); cost, ttc = info.get("cost", 0.), info.get("ttc", 10.)
            if term: col = int(info.get("collision", False)); break
            if trunc: break
        crs.append(col); sps.append(sp / max(L, 1))
    env.close()
    return {**{k: task[k] for k in task if k != "n_ep"}, "cr": float(np.mean(crs)), "speed": float(np.mean(sps))}


def build(seeds, n_ep, gains):
    T = []
    for s in seeds:
        base = {"seed": int(s), "n_ep": n_ep}
        for dm in ("greedy", "sample"):
            for g in gains: T.append({**base, "method": "shaped", "gain": float(g), "decode": dm})
        T.append({**base, "method": "mask_defer"})
        T.append({**base, "method": "override"})
        T.append({**base, "method": "baseline", "kind": "noop_greedy"})
        T.append({**base, "method": "baseline", "kind": "noop_sample"})
        for k in (2., 3.): T.append({**base, "method": "baseline", "kind": "ttc_brake", "ttc_k": k})
        for p in (0.3, 0.6): T.append({**base, "method": "baseline", "kind": "prob_brake", "temp": p})
    return T


def run(policies, cc_path, out_path, seeds, n_ep, n_proc, max_steps, veh, den, gains=(0., 0.5, 1., 2., 4., 8.)):
    t0 = time.time(); allr = []
    for pi, base in enumerate(policies):
        tasks = [{**t, "policy": pi} for t in build(list(seeds), n_ep, gains)]
        with mp.Pool(n_proc, initializer=_init, initargs=(base, cc_path, max_steps, veh, den)) as pool:
            allr += list(pool.imap_unordered(_eval_one, tasks, chunksize=1))
        print(f"  policy {pi+1}/{len(policies)} done ({time.time()-t0:.0f}s)", flush=True)
    out = {"policies": policies, "cc": cc_path, "seeds": list(map(int, seeds)), "gains": list(gains), "results": allr}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(out_path, "w"), indent=2)
    print(f"saved {out_path} ({time.time()-t0:.0f}s)", flush=True); return out


def analyze(path, cr_levels=(0.5, 0.7)):
    d = json.load(open(path)); npol = len(d["policies"])
    G = defaultdict(lambda: defaultdict(list))
    for r in d["results"]:
        if r["method"] == "shaped": key = ("shaped", r["decode"])
        elif r["method"] == "mask_defer": key = ("mask",)
        elif r["method"] == "override": key = ("override",)
        elif r.get("kind") == "ttc_brake": key = ("rule",)
        elif r.get("kind") == "prob_brake": key = ("rule",)
        else: key = ("rule",) if r.get("kind", "").startswith(("noop_sample",)) else ("mask",) if False else ("rule",)
        G[key][r["policy"]].append((r["cr"], r["speed"]))
    # frontiers: shaped (both decodes, all gains) ; mask (mask + raw greedy) ; override ; rule (all brake rules + raw)
    raw_g = [(r["cr"], r["speed"]) for r in d["results"] if r["method"] == "baseline" and r.get("kind") == "noop_greedy"]
    def front(pol, keys):
        pts = []
        for k in keys:
            for (cr, sp) in G[k].get(pol, []): pts.append((cr, sp))
        return pts
    def best(pts, tgt):
        c = [sp for cr, sp in pts if cr <= tgt + 1e-9]; return max(c) if c else np.nan
    # shaped includes g=0 (=raw) so it subsumes raw; mask frontier = mask + raw(noop_greedy); override alone
    shaped_keys = [("shaped", "greedy"), ("shaped", "sample")]
    def maskf(p):  # mask frontier = mask-and-defer point + raw-greedy point
        pts = list(G[("mask",)].get(p, []))
        for r in d["results"]:
            if r.get("policy") == p and r["method"] == "baseline" and r.get("kind") == "noop_greedy":
                pts.append((r["cr"], r["speed"]))
        return pts
    print(f"=== PRINCIPLED (consequence-shaped) ASRA: {npol} policies ===")
    for tgt in cr_levels:
        S = np.array([best(front(p, shaped_keys), tgt) for p in range(npol)])
        M = np.array([best(maskf(p), tgt) for p in range(npol)])
        O = np.array([best(front(p, [("override",)]), tgt) for p in range(npol)])
        R = np.array([best(front(p, [("rule",)]), tgt) for p in range(npol)])
        for name, arr in [("mask", M), ("override", O), ("rule", R)]:
            ok = ~(np.isnan(S) | np.isnan(arr))
            if ok.sum() >= 3:
                dd = paired_bootstrap_diff(S[ok], arr[ok]); sign = int(np.sum(S[ok] > arr[ok] + 1e-9))
                verd = "shaped WINS" if dd["excludes_zero"] and dd["diff"] > 0 else ("loses" if dd["excludes_zero"] and dd["diff"] < 0 else "tie")
                print(f"  @CR<={tgt}: shaped - {name:8s} d={dd['diff']:+.2f} CI[{dd['lo']:+.2f},{dd['hi']:+.2f}] ({sign}/{ok.sum()}, p={dd['pvalue']:.3f}) => {verd}")
            else:
                print(f"  @CR<={tgt}: shaped vs {name}: n={ok.sum()} insufficient")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=None)
    ap.add_argument("--cc", default="checkpoints/cost_critic/full.pt")
    ap.add_argument("--out", default=str(RES / "principled_asra_driving.json"))
    ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--n_ep", type=int, default=40)
    ap.add_argument("--n_proc", type=int, default=8); ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--veh", type=int, default=15); ap.add_argument("--den", type=float, default=1.5)
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    import glob
    pols = a.policies or sorted(glob.glob("current_work/base_policies/easy_seed*_final.pt"))
    if a.smoke:
        run(pols[:1], a.cc, str(RES / "_principled_smoke.json"), seeds=[0, 1], n_ep=6, n_proc=4, max_steps=60,
            veh=a.veh, den=a.den, gains=(0., 2., 8.))
        analyze(str(RES / "_principled_smoke.json")); print("[SMOKE] OK")
    else:
        run(pols, a.cc, a.out, list(range(a.seeds)), a.n_ep, a.n_proc, a.max_steps, a.veh, a.den)
        analyze(a.out)

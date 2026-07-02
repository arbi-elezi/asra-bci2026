"""Salience-gating ablation: gated tilt (beta = g*S(s), the paper's operator) vs always-on tilt"""
from __future__ import annotations
import sys, json, argparse, time, glob
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast import principled_asra as PA
RES = Path("current_work/results_v3")


def _eval_gating(task):
    actor = PA._actor(); cc = PA._G["cc"]
    from src.environment.highway_wrapper import HighwayFRAEnv
    env = HighwayFRAEnv(scenario=PA._G["scenario"], seed=task["seed"], vehicles_count=PA._G["veh"],
                        vehicles_density=PA._G["den"], high_speed_reward=PA._G["hsr"])
    crs, sps = [], []
    for ep in range(task["n_ep"]):
        obs, info = env.reset(seed=task["seed"] * 100000 + ep); cost = info.get("cost", 0.)
        col, sp, L = 0, 0.0, 0
        for t in range(PA._G["max_steps"]):
            o = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                lg = actor(o.unsqueeze(0)).squeeze(0)
                qc = cc.predict_state(o).squeeze(0)
            S = 1.0 if task["mode"] == "alwayson" else float(np.clip(cost, 0., 1.))
            if task["mode"] == "raw":
                a = int(lg.argmax())
            else:
                a = int((lg - task["gain"] * S * (qc - qc.mean())).argmax())
            sp += float(obs[2]); L += 1
            obs, r, term, trunc, info = env.step(a); cost = info.get("cost", 0.)
            if term: col = int(info.get("collision", False)); break
            if trunc: break
        crs.append(col); sps.append(sp / max(L, 1))
    env.close()
    return {**{k: task[k] for k in task if k != "n_ep"}, "cr": float(np.mean(crs)), "speed": float(np.mean(sps))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=int, default=6); ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--n_ep", type=int, default=15); ap.add_argument("--n_proc", type=int, default=10)
    ap.add_argument("--out", default=str(RES / "gating_ablation.json")); a = ap.parse_args()
    pols = sorted(glob.glob("current_work/base_policies/easy_seed*_final.pt"))[:a.policies]
    tasks = []
    for s in range(a.seeds):
        base = {"seed": s, "n_ep": a.n_ep}
        tasks.append({**base, "mode": "raw", "gain": 0.})
        for g in (1., 2., 4.):
            tasks.append({**base, "mode": "gated", "gain": g})
            tasks.append({**base, "mode": "alwayson", "gain": g})
    out = defaultdict(lambda: defaultdict(list)); t0 = time.time()
    for pi, bp in enumerate(pols):
        with mp.Pool(a.n_proc, initializer=PA._init, initargs=(bp, "checkpoints/cost_critic/full.pt", 120, 15, 1.5)) as pool:
            for r in pool.imap_unordered(_eval_gating, [{**t, "policy": pi} for t in tasks], chunksize=1):
                key = f"{r['mode']}_g{r['gain']:g}" if r["mode"] != "raw" else "raw"
                out[key]["cr"].append(r["cr"]); out[key]["speed"].append(r["speed"])
    res = {k: {"cr": float(np.mean(v["cr"])), "speed": float(np.mean(v["speed"]))} for k, v in out.items()}
    for k in sorted(res): print(f"  {k:16s} cr={res[k]['cr']:.3f} speed={res[k]['speed']:.2f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=2)
    print(f"saved {a.out} ({time.time()-t0:.0f}s)")

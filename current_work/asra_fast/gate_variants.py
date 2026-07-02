"""Gate-variant ablation: is the operator sensitive to the QUALITY of the gating signal?
Variants at fixed gain grid on driving (6 policies x 4 seeds):
  gated    - paper's TTC gate S = clip((2-TTC)/2)
  broad    - wider gate S = clip((4-TTC)/4): fires earlier and more often
  noisy    - TTC gate plus 25% random false-positive fires (S=1 on benign frames)
  alwayson - S = 1 (broadest possible)
  raw      - no tilt
Prediction (Prop 1 + gating ablation): broader/noisier gates cannot reduce safety below the gated
operator -- they only spend divergence in more states (speed tax); protection is set by the critic.
"""
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


def _eval_gate(task):
    actor = PA._actor(); cc = PA._G["cc"]
    from src.environment.highway_wrapper import HighwayFRAEnv
    env = HighwayFRAEnv(scenario=PA._G["scenario"], seed=task["seed"], vehicles_count=PA._G["veh"],
                        vehicles_density=PA._G["den"], high_speed_reward=PA._G["hsr"])
    rng = np.random.default_rng(task["seed"] * 31 + 5)
    crs, sps = [], []
    for ep in range(task["n_ep"]):
        obs, info = env.reset(seed=task["seed"] * 100000 + ep)
        cost, ttc = info.get("cost", 0.), info.get("ttc", 10.)
        col, sp, L = 0, 0.0, 0
        for t in range(PA._G["max_steps"]):
            o = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                lg = actor(o.unsqueeze(0)).squeeze(0)
                qc = cc.predict_state(o).squeeze(0)
            mode = task["mode"]
            if mode == "raw": S = 0.0
            elif mode == "gated": S = float(np.clip(cost, 0., 1.))
            elif mode == "broad": S = float(np.clip((4.0 - ttc) / 4.0, 0., 1.))
            elif mode == "noisy": S = max(float(np.clip(cost, 0., 1.)), 1.0 if rng.random() < 0.25 else 0.0)
            else: S = 1.0  # alwayson
            a = int((lg - task["gain"] * S * (qc - qc.mean())).argmax()) if S > 0 else int(lg.argmax())
            sp += float(obs[2]); L += 1
            obs, r, term, trunc, info = env.step(a)
            cost, ttc = info.get("cost", 0.), info.get("ttc", 10.)
            if term: col = int(info.get("collision", False)); break
            if trunc: break
        crs.append(col); sps.append(sp / max(L, 1))
    env.close()
    return {**{k: task[k] for k in task if k != "n_ep"}, "cr": float(np.mean(crs)), "speed": float(np.mean(sps))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=int, default=6); ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--n_ep", type=int, default=15); ap.add_argument("--n_proc", type=int, default=10)
    ap.add_argument("--out", default=str(RES / "gate_variants.json")); a = ap.parse_args()
    pols = sorted(glob.glob("current_work/base_policies/easy_seed*_final.pt"))[:a.policies]
    tasks = [{"seed": s, "n_ep": a.n_ep, "mode": "raw", "gain": 0.} for s in range(a.seeds)]
    for s in range(a.seeds):
        for m in ("gated", "broad", "noisy", "alwayson"):
            for g in (2., 4.):
                tasks.append({"seed": s, "n_ep": a.n_ep, "mode": m, "gain": g})
    out = defaultdict(lambda: defaultdict(list)); t0 = time.time()
    for pi, bp in enumerate(pols):
        with mp.Pool(a.n_proc, initializer=PA._init, initargs=(bp, "checkpoints/cost_critic/full.pt", 120, 15, 1.5)) as pool:
            for r in pool.imap_unordered(_eval_gate, [{**t, "policy": pi} for t in tasks], chunksize=1):
                key = f"{r['mode']}_g{r['gain']:g}" if r["mode"] != "raw" else "raw"
                out[key]["cr"].append(r["cr"]); out[key]["speed"].append(r["speed"])
    res = {k: {"cr": float(np.mean(v["cr"])), "speed": float(np.mean(v["speed"]))} for k, v in out.items()}
    for k in sorted(res): print(f"  {k:14s} cr={res[k]['cr']:.3f} speed={res[k]['speed']:.2f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=2)
    print(f"saved {a.out} ({time.time()-t0:.0f}s)")

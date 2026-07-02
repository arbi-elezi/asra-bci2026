"""Experiment driver for ASRA-fast (decode-controlled, paired, train/val/test)."""
from __future__ import annotations

import sys, json, time, os, hashlib, argparse
from pathlib import Path
import multiprocessing as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from current_work.asra_fast.core import MLPActor, ASRA, ASRAConfig, CostSalience, SalienceConfig
from current_work.asra_fast.runner import eval_episodes, build_salience

# ----- worker globals (set by initializer) -----
_G = {}


def _init_worker(base_path: str, fisher_path: str, salience_kind: str,
                 vehicles: int, max_steps: int, density: float = 1.5,
                 eval_vehicles=None, eval_density=None):
    torch.set_num_threads(1)  # avoid thread oversubscription across pool workers
    data = torch.load(base_path, map_location="cpu", weights_only=False)
    fisher = torch.load(fisher_path, map_location="cpu", weights_only=False)
    _G["state"] = data["actor"]
    _G["hidden"] = data.get("hidden", 64)
    _G["fisher"] = fisher
    _G["salience"] = CostSalience(SalienceConfig(kind="cost"))
    # env config taken from the snapshot so eval matches training exactly, UNLESS the caller
    # overrides the deployment context (train-easy/deploy-hard risk-profile-mismatch protocol).
    _G["vehicles"] = eval_vehicles if eval_vehicles is not None else data.get("vehicles", vehicles)
    _G["density"] = eval_density if eval_density is not None else data.get("density", density)
    _G["hsr"] = data.get("high_speed_reward", 0.4)
    _G["max_steps"] = max_steps


def _fresh_actor():
    actor = MLPActor(12, 4, _G["hidden"])
    actor.load_state_dict(_G["state"])
    for p in actor.parameters():
        p.requires_grad_(True)
    return actor


def _eval_one(task: dict) -> dict:
    actor = _fresh_actor()
    w0_full = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    w0_par = {k: v.detach().clone() for k, v in actor.named_parameters()}
    fisher = _G["fisher"]
    sal = _G["salience"]
    seed = task["seed"]; n_ep = task["n_ep"]

    if task["method"] == "asra":
        cfg = ASRAConfig(gain=task["gain"], select=task["decode"],
                         weight_channel=task.get("weight_channel", True),
                         conf_channel=task.get("conf_channel", True),
                         targeting=task.get("targeting", "gradient"))
        spec = {"type": "asra", "cfg": cfg}
    else:  # baseline
        spec = {"type": "baseline", "kind": task["kind"],
                "ttc_k": task.get("ttc_k", 2.0), "temp": task.get("temp", 2.0),
                "base_decode": task.get("base_decode", "sample")}

    recs = eval_episodes(actor, w0_full, w0_par, fisher, sal, spec,
                         n_episodes=n_ep, seed=seed, vehicles=_G["vehicles"],
                         device="cpu", max_steps=_G["max_steps"], density=_G["density"],
                         high_speed_reward=_G["hsr"])
    cr = float(np.mean([r["collision"] for r in recs]))
    ret = float(np.mean([r["ret"] for r in recs]))
    bf = float(np.mean([r["brake_frac"] for r in recs]))
    spd = float(np.mean([r["mean_speed"] for r in recs]))
    ln = float(np.mean([r["length"] for r in recs]))
    out = {**{k: task[k] for k in task if k != "n_ep"},
           "cr": cr, "ret": ret, "brake_frac": bf, "mean_speed": spd, "length": ln}
    if "lessrisky" in recs[0]:
        out["lessrisky"] = float(np.mean([r["lessrisky"] for r in recs]))
    return out


def build_tasks(seeds, n_ep, g_grid, decode_modes, ttc_grid, pbrake_grid) -> list[dict]:
    tasks = []
    for s in seeds:
        # ASRA frontier (both decode modes); g=0 included => decode-only control point
        for dm in decode_modes:
            for g in g_grid:
                tasks.append({"method": "asra", "gain": float(g), "decode": dm,
                              "seed": int(s), "n_ep": n_ep})
        # decode-controlled baselines
        tasks.append({"method": "baseline", "kind": "noop_greedy", "seed": int(s), "n_ep": n_ep})
        tasks.append({"method": "baseline", "kind": "noop_sample", "seed": int(s), "n_ep": n_ep})
        # swept rule frontiers — applied to the DEPLOYED (stochastic) policy
        for k in ttc_grid:
            tasks.append({"method": "baseline", "kind": "ttc_brake", "ttc_k": float(k),
                          "base_decode": "sample", "seed": int(s), "n_ep": n_ep})
        for p in pbrake_grid:
            tasks.append({"method": "baseline", "kind": "prob_brake", "temp": float(p),
                          "base_decode": "sample", "seed": int(s), "n_ep": n_ep})
    return tasks


def run(base_path: str, fisher_path: str, out_path: str, seeds, n_ep=150,
        vehicles=15, max_steps=500, n_proc=16, scenario="highway", density=1.5,
        eval_vehicles=None, eval_density=None,
        g_grid=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0),
        decode_modes=("greedy", "sample"),
        ttc_grid=(0.0, 1.0, 2.0, 3.0, 4.0),
        pbrake_grid=(0.0, 0.1, 0.3, 0.6, 1.0)):
    t0 = time.time()
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    with open(base_path, "rb") as f:
        bhash = hashlib.sha256(f.read()).hexdigest()[:16]
    tasks = build_tasks(list(seeds), n_ep, g_grid, decode_modes, ttc_grid, pbrake_grid)
    print(f"[{scenario}] {len(tasks)} tasks ({len(seeds)} seeds x {n_ep} eps) on {n_proc} procs; "
          f"base CR~{base.get('train_cr')}, hash {bhash}", flush=True)

    with mp.Pool(n_proc, initializer=_init_worker,
                 initargs=(base_path, fisher_path, "cost", vehicles, max_steps, density,
                           eval_vehicles, eval_density)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_eval_one, tasks, chunksize=1)):
            results.append(r)
            if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                print(f"  {i+1}/{len(tasks)} done ({(i+1)/(time.time()-t0):.1f} task/s)", flush=True)

    out = {"scenario": scenario, "base_path": base_path, "base_hash": bhash,
           "base_train_cr": base.get("train_cr"), "base_step": base.get("step"),
           "train_vehicles": base.get("vehicles"), "train_density": base.get("density"),
           "eval_vehicles": eval_vehicles, "eval_density": eval_density,
           "n_ep": n_ep, "vehicles": vehicles, "density": density, "seeds": list(map(int, seeds)),
           "g_grid": list(g_grid), "decode_modes": list(decode_modes),
           "ttc_grid": list(ttc_grid), "pbrake_grid": list(pbrake_grid),
           "results": results, "elapsed_s": time.time() - t0}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{scenario}] saved {out_path} ({time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--fisher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--n_ep", type=int, default=150)
    ap.add_argument("--vehicles", type=int, default=15)
    ap.add_argument("--n_proc", type=int, default=16)
    ap.add_argument("--scenario", default="highway")
    ap.add_argument("--density", type=float, default=1.5)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--eval_vehicles", type=int, default=None)
    ap.add_argument("--eval_density", type=float, default=None)
    a = ap.parse_args()
    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    run(a.base, a.fisher, a.out, seeds, n_ep=a.n_ep, vehicles=a.vehicles,
        n_proc=a.n_proc, scenario=a.scenario, density=a.density, max_steps=a.max_steps,
        eval_vehicles=a.eval_vehicles, eval_density=a.eval_density)

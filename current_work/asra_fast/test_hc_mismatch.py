"""Test the precondition via train-low-hazard / deploy-high-hazard mismatch."""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from current_work.asra_fast.decision_task import train_policy, eval_condition


def run(p_train, p_eval, steps, seeds, n_ep, out):
    res = {"p_train": p_train, "p_eval": p_eval, "raw_greedy": [], "raw_sample": [],
           "override_brake": [], "asra_greedy": {g: [] for g in [1, 2, 4, 8]},
           "prob_brake": {p: [] for p in [0.1, 0.3, 0.5, 0.7, 1.0]}}
    for s in range(seeds):
        actor = train_policy(steps=steps, seed=s, p_hazard=p_train, T=40)
        w0 = {k: v.detach().clone() for k, v in actor.named_parameters()}
        fisher = {k: torch.ones_like(v) for k, v in actor.named_parameters()}
        def ev(mode, **kw):
            r = eval_condition(actor, mode, n_ep=n_ep, seed=100 + s, p_hazard=p_eval,
                               fisher=fisher, w0=w0, **kw)
            return {"cr": r["cr"], "perf": r["perf"]}
        res["raw_greedy"].append(ev("raw_greedy"))
        res["raw_sample"].append(ev("raw_sample"))
        res["override_brake"].append(ev("override_brake"))
        for g in res["asra_greedy"]:
            res["asra_greedy"][g].append(ev("asra_greedy", gain=g))
        for p in res["prob_brake"]:
            res["prob_brake"][p].append(ev("prob_brake", p_brake=p))
        rg = res["raw_greedy"][-1]; ag = res["asra_greedy"][4][-1]; ov = res["override_brake"][-1]
        print(f"  seed {s}: raw_greedy CR={rg['cr']:.3f}/{rg['perf']:+.2f}  "
              f"asra(g4) CR={ag['cr']:.3f}/{ag['perf']:+.2f}  "
              f"override CR={ov['cr']:.3f}/{ov['perf']:+.2f}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, default=float)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_train", type=float, default=0.08)
    ap.add_argument("--p_eval", type=float, default=0.45)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--n_ep", type=int, default=300)
    ap.add_argument("--out", default="current_work/results_v3/hc_mismatch.json")
    a = ap.parse_args()
    run(a.p_train, a.p_eval, a.steps, a.seeds, a.n_ep, a.out)

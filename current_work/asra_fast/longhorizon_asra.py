"""Long-horizon analysis: does the per-step tilt merely postpone risk? Evaluate operator vs
override/mask/raw on driving at episode horizons 120/300/500 steps. If the operator only deferred
collisions, its collision rate would climb toward raw as the horizon grows; matched-safety speed
gaps should also be reported per horizon."""
from __future__ import annotations
import sys, json, argparse, time, glob
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast import principled_asra as PA
from current_work.asra_fast.analyze import paired_bootstrap_diff
RES = Path("current_work/results_v3")


def build(seeds, n_ep, gains):
    T = []
    for s in seeds:
        base = {"seed": int(s), "n_ep": n_ep}
        for g in gains: T.append({**base, "method": "shaped", "gain": float(g), "decode": "greedy"})
        T.append({**base, "method": "override"})
        T.append({**base, "method": "mask_defer"})
        T.append({**base, "method": "baseline", "kind": "noop_greedy"})
    return T


def run_h(policies, cc, horizon, seeds, n_ep, n_proc, gains):
    allr = []
    for pi, base in enumerate(policies):
        tasks = [{**t, "policy": pi} for t in build(list(seeds), n_ep, gains)]
        with mp.Pool(n_proc, initializer=PA._init, initargs=(base, cc, horizon, 15, 1.5)) as pool:
            allr += list(pool.imap_unordered(PA._eval_one, tasks, chunksize=1))
    return allr


def analyze_h(results, npol, tgt=0.7):
    G = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = ("shaped",) if r["method"] == "shaped" else (("override",) if r["method"] == "override"
              else ("mask",) if r["method"] == "mask_defer" else ("raw",))
        G[key][r["policy"]].append((r["cr"], r["speed"]))
    def mean_cr(key):
        vals = [c for pi in range(npol) for (c, s) in G[key].get(pi, [])]
        return float(np.mean(vals)) if vals else float("nan")
    def best(pi, key):
        pts = list(G[key].get(pi, []))
        if key == ("shaped",): pts += G[("raw",)].get(pi, [])
        c = [sp for cr, sp in pts if cr <= tgt + 1e-9]
        return max(c) if c else np.nan
    S = np.array([best(p, ("shaped",)) for p in range(npol)])
    O = np.array([best(p, ("override",)) for p in range(npol)])
    ok = ~(np.isnan(S) | np.isnan(O))
    out = {"raw_cr": mean_cr(("raw",)), "shaped_best_cr": None, "override_cr": mean_cr(("override",)),
           "mask_cr": mean_cr(("mask",))}
    # best-gain shaped CR (the operating point actually used at matched safety)
    sc = [c for pi in range(npol) for (c, s) in G[("shaped",)].get(pi, [])]
    out["shaped_cr_mean_over_gains"] = float(np.mean(sc)) if sc else float("nan")
    if ok.sum() >= 3:
        dd = paired_bootstrap_diff(S[ok], O[ok]); sign = int(np.sum(S[ok] > O[ok] + 1e-9))
        out["dspeed_vs_override"] = {"d": dd["diff"], "lo": dd["lo"], "hi": dd["hi"], "sign": f"{sign}/{int(ok.sum())}", "p": dd["pvalue"]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=int, default=4); ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--n_ep", type=int, default=12); ap.add_argument("--n_proc", type=int, default=10)
    ap.add_argument("--out", default=str(RES / "longhorizon.json")); a = ap.parse_args()
    pols = sorted(glob.glob("current_work/base_policies/easy_seed*_final.pt"))[:a.policies]
    cc = "checkpoints/cost_critic/full.pt"; gains = (1., 2., 4., 8.)
    out = {}; t0 = time.time()
    for H in (120, 300, 500):
        res = run_h(pols, cc, H, list(range(a.seeds)), a.n_ep, a.n_proc, gains)
        out[str(H)] = analyze_h(res, len(pols))
        r = out[str(H)]
        d = r.get("dspeed_vs_override", {})
        print(f"H={H}: raw_cr={r['raw_cr']:.2f} override_cr={r['override_cr']:.2f} mask_cr={r['mask_cr']:.2f} "
              f"| shaped-vs-override d={d.get('d', float('nan')):+.2f} [{d.get('lo', 0):+.2f},{d.get('hi', 0):+.2f}] "
              f"({d.get('sign','-')}) [{time.time()-t0:.0f}s]", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=2)
    print("saved", a.out)

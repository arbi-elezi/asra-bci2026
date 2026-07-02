"""Risk-3 address: how robust is the consequence-shaped operator to COST-CRITIC error?"""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast import principled_asra as PA
from current_work.asra_fast.analyze import paired_bootstrap_diff
RES = Path("current_work/results_v3")

CRITICS = [("full", "checkpoints/cost_critic/full.pt"),
           ("degraded_10pct", "checkpoints/cost_critic/degraded_10pct.pt"),
           ("degraded_25pct", "checkpoints/cost_critic/degraded_25pct.pt"),
           ("degraded_50pct", "checkpoints/cost_critic/degraded_50pct.pt"),
           ("biased_02", "checkpoints/cost_critic/biased_fast_02.pt"),
           ("biased_05", "checkpoints/cost_critic/biased_fast_05.pt")]


def lean_build(seeds, n_ep, gains):
    T = []
    for s in seeds:
        base = {"seed": int(s), "n_ep": n_ep}
        for g in gains: T.append({**base, "method": "shaped", "gain": float(g), "decode": "greedy"})
        T.append({**base, "method": "override"})
        T.append({**base, "method": "mask_defer"})
        T.append({**base, "method": "baseline", "kind": "noop_greedy"})
    return T


def run_one_critic(policies, cc_path, seeds, n_ep, n_proc, max_steps, veh, den, gains):
    allr = []
    for pi, base in enumerate(policies):
        tasks = [{**t, "policy": pi} for t in lean_build(list(seeds), n_ep, gains)]
        with mp.Pool(n_proc, initializer=PA._init, initargs=(base, cc_path, max_steps, veh, den)) as pool:
            allr += list(pool.imap_unordered(PA._eval_one, tasks, chunksize=1))
    return allr


def analyze(results, npol, tgt=0.7):
    G = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = ("shaped",) if r["method"] == "shaped" else (("override",) if r["method"] == "override"
               else ("mask",) if r["method"] == "mask_defer" else ("raw",))
        G[key][r["policy"]].append((r["cr"], r["speed"]))
    def best(pol, key):
        pts = G[key].get(pol, []); c = [sp for cr, sp in pts if cr <= tgt + 1e-9]; return max(c) if c else np.nan
    # shaped frontier includes raw (g=0), so add raw pts to shaped
    for pol in range(npol): G[("shaped",)][pol] += G[("raw",)].get(pol, [])
    S = np.array([best(p, ("shaped",)) for p in range(npol)])
    O = np.array([best(p, ("override",)) for p in range(npol)])
    ok = ~(np.isnan(S) | np.isnan(O))
    if ok.sum() >= 3:
        dd = paired_bootstrap_diff(S[ok], O[ok]); sign = int(np.sum(S[ok] > O[ok] + 1e-9))
        return {"d_shaped_vs_override": dd["diff"], "lo": dd["lo"], "hi": dd["hi"], "sign": f"{sign}/{ok.sum()}", "p": dd["pvalue"]}
    return {"n": int(ok.sum())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--n_ep", type=int, default=25)
    ap.add_argument("--n_proc", type=int, default=10); ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--veh", type=int, default=15); ap.add_argument("--den", type=float, default=1.5)
    ap.add_argument("--out", default=str(RES / "robustness_cc.json")); a = ap.parse_args()
    import glob
    pols = sorted(glob.glob("current_work/base_policies/easy_seed*_final.pt"))[:4]
    gains = (0., 1., 2., 4., 8.); t0 = time.time(); out = {}
    print("=== Operator robustness to cost-critic error (shaped frontier vs override @CR<=0.7) ===")
    for name, path in CRITICS:
        if not Path(path).exists(): print(f"  {name}: missing, skip"); continue
        res = run_one_critic(pols, path, list(range(a.seeds)), a.n_ep, a.n_proc, a.max_steps, a.veh, a.den, gains)
        r = analyze(res, len(pols)); out[name] = r
        if "d_shaped_vs_override" in r:
            print(f"  {name:14s} shaped-override d={r['d_shaped_vs_override']:+.2f} CI[{r['lo']:+.2f},{r['hi']:+.2f}] ({r['sign']}, p={r['p']:.3f}) [{time.time()-t0:.0f}s]", flush=True)
        else:
            print(f"  {name:14s} n={r.get('n')} insufficient [{time.time()-t0:.0f}s]", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=2)
    print("saved", a.out)

"""H4 (pre-registered): is the Gaussian WEIGHT-perturbation channel load-bearing, or does ASRA"""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast import multipolicy as MP
from current_work.asra_fast.analyze import paired_bootstrap_diff
RES = Path("current_work/results_v3")

CFGS = [("full", True, True), ("conf_only", False, True), ("weight_only", True, False)]


def build_h4(seeds, n_ep, g_grid, pol_idx):
    T = []
    for s in seeds:
        base = {"seed": int(s), "n_ep": n_ep, "policy": pol_idx}
        for cfg, wc, cc in CFGS:
            for g in g_grid:
                T.append({**base, "method": "asra", "gain": float(g), "decode": "greedy",
                          "weight_channel": wc, "conf_channel": cc, "cfg": cfg})
        T.append({**base, "method": "mask_defer", "gain": 0.0, "cfg": "mask"})
        T.append({**base, "method": "baseline", "kind": "noop_greedy", "cfg": "noop"})
    return T


def run(policies, out_path, seeds, n_ep, n_proc, max_steps, g_grid=(0., 1., 2., 3., 5., 8.)):
    t0 = time.time(); allr = []
    for pi, base in enumerate(policies):
        tasks = build_h4(list(seeds), n_ep, g_grid, pi)
        with mp.Pool(n_proc, initializer=MP._init, initargs=(base, max_steps, None, None)) as pool:
            allr += list(pool.imap_unordered(MP._eval_one, tasks, chunksize=1))
        print(f"  policy {pi+1}/{len(policies)} done ({time.time()-t0:.0f}s)", flush=True)
    out = {"policies": policies, "seeds": list(map(int, seeds)), "n_ep": n_ep, "g_grid": list(g_grid),
           "results": allr, "elapsed_s": time.time()-t0}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(out_path, "w"), indent=2)
    print(f"saved {out_path} ({time.time()-t0:.0f}s)", flush=True); return out


def analyze(path, tgt=0.7):
    d = json.load(open(path)); npol = len(d["policies"])
    # cfg -> policy -> list[(cr,speed)] over (gain,seed); and mask/noop per policy
    G = defaultdict(lambda: defaultdict(list))
    for r in d["results"]:
        G[r.get("cfg", "?")][r["policy"]].append((r["cr"], r["speed"]))
    def best_at(cfg, pi):
        pts = G[cfg].get(pi, [])
        c = [sp for cr, sp in pts if cr <= tgt + 1e-9]
        return max(c) if c else np.nan
    print(f"=== H4: weight-channel load-bearing? ({npol} policies, matched CR<={tgt}) ===")
    full = np.array([best_at("full", pi) for pi in range(npol)])
    conf = np.array([best_at("conf_only", pi) for pi in range(npol)])
    wgt = np.array([best_at("weight_only", pi) for pi in range(npol)])
    for name, arr in [("full", full), ("conf_only", conf), ("weight_only", wgt)]:
        print(f"  {name:11s} mean best-speed@CR<={tgt}: {np.nanmean(arr):.3f}  per-policy: {np.round(arr,2)}")
    ok = ~(np.isnan(full) | np.isnan(conf))
    if ok.sum() >= 3:
        dd = paired_bootstrap_diff(full[ok], conf[ok])
        signpos = int(np.sum(full[ok] > conf[ok] + 1e-9)); signneg = int(np.sum(full[ok] < conf[ok] - 1e-9))
        verdict = ("weight channel ADDS speed" if dd["excludes_zero"] and dd["diff"] > 0
                   else "TIE => weight channel NOT load-bearing (relegate per H4)")
        print(f"\n  full - conf_only: d={dd['diff']:+.3f} CI[{dd['lo']:+.3f},{dd['hi']:+.3f}] "
              f"(n={ok.sum()}, full>conf in {signpos}/{ok.sum()}, full<conf in {signneg}/{ok.sum()}, p2={dd['pvalue']:.3f})")
        print(f"  => {verdict}")
    ok2 = ~(np.isnan(full) | np.isnan(wgt))
    if ok2.sum() >= 3:
        dw = paired_bootstrap_diff(full[ok2], wgt[ok2])
        print(f"  full - weight_only: d={dw['diff']:+.3f} CI[{dw['lo']:+.3f},{dw['hi']:+.3f}] (conf channel's marginal value)")
    return {"full": float(np.nanmean(full)), "conf_only": float(np.nanmean(conf)), "weight_only": float(np.nanmean(wgt))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--out", default=str(RES / "h4_ablation.json"))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n_ep", type=int, default=40)
    ap.add_argument("--n_proc", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        run(a.policies[:1], str(RES / "_h4_smoke.json"), seeds=[0, 1], n_ep=6, n_proc=4, max_steps=60, g_grid=(0., 3., 8.))
        analyze(str(RES / "_h4_smoke.json")); print("[SMOKE] OK")
    else:
        run(a.policies, a.out, seeds=list(range(a.seeds)), n_ep=a.n_ep, n_proc=a.n_proc, max_steps=a.max_steps)
        analyze(a.out)

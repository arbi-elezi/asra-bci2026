"""Critic-robustness sweep at higher power: n=8 independently-trained policies (vs the original 3"""
from __future__ import annotations
import sys, json, argparse, time, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast import robustness_cc as RC

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--n_ep", type=int, default=15)
    ap.add_argument("--n_proc", type=int, default=10)
    ap.add_argument("--out", default=str(RC.RES / "robustness_cc6.json")); a = ap.parse_args()
    pols = sorted(glob.glob("current_work/base_policies/easy_seed*_final.pt"))[:a.policies]
    gains = (0., 1., 2., 4., 8.); t0 = time.time(); out = {}
    print(f"=== Critic robustness, n={len(pols)} policies ===")
    for name, path in RC.CRITICS:
        if not Path(path).exists(): print(f"  {name}: missing, skip"); continue
        res = RC.run_one_critic(pols, path, list(range(a.seeds)), a.n_ep, a.n_proc, 120, 15, 1.5, gains)
        r = RC.analyze(res, len(pols)); out[name] = r
        if "d_shaped_vs_override" in r:
            print(f"  {name:14s} shaped-override d={r['d_shaped_vs_override']:+.2f} "
                  f"CI[{r['lo']:+.2f},{r['hi']:+.2f}] ({r['sign']}, p={r['p']:.3f}) [{time.time()-t0:.0f}s]", flush=True)
        else:
            print(f"  {name:14s} n={r.get('n')} insufficient [{time.time()-t0:.0f}s]", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=2)
    print("saved", a.out)

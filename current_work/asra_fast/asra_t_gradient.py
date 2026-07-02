"""ASRA-T across a collision-rate gradient (hazard rate p = 0.1..0.7)."""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast.asra_targeted import run
from current_work.asra_fast.analyze import paired_bootstrap_diff

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--n_ep", type=int, default=400)
    ap.add_argument("--out", default="current_work/results_v3/asra_t_gradient.json")
    a = ap.parse_args()
    modes = ["override_brake", "mask_defer", "asra_t"]
    grid = [0.1, 0.3, 0.5, 0.7]
    out = {}
    print("=== ASRA-T across collision-rate gradient (hazard p) ===")
    print(f"{'p':>4} | " + " | ".join(f"{m:>14}" for m in modes) + " | dCR(mask-asraT) dperf(asraT-ovr)")
    for pp in grid:
        agg = {m: {"cr": [], "perf": []} for m in modes}
        for s in range(a.seeds):
            for m in modes:
                c, pf = run(m, a.n_ep, s, p=pp)
                agg[m]["cr"].append(float(c.mean())); agg[m]["perf"].append(float(pf.mean()))
        cr = lambda m: np.array(agg[m]["cr"]); pf = lambda m: np.array(agg[m]["perf"])
        d_cr = paired_bootstrap_diff(cr("mask_defer"), cr("asra_t"))
        d_pf = paired_bootstrap_diff(pf("asra_t"), pf("override_brake"))
        out[str(pp)] = {
            "override": {"cr": float(np.mean(agg["override_brake"]["cr"])), "perf": float(np.mean(agg["override_brake"]["perf"]))},
            "mask_defer": {"cr": float(np.mean(agg["mask_defer"]["cr"])), "perf": float(np.mean(agg["mask_defer"]["perf"]))},
            "asra_t": {"cr": float(np.mean(agg["asra_t"]["cr"])), "perf": float(np.mean(agg["asra_t"]["perf"]))},
            "dCR_mask_minus_asraT": {"diff": d_cr["diff"], "lo": d_cr["lo"], "hi": d_cr["hi"], "excl0": d_cr["excludes_zero"]},
            "dperf_asraT_minus_override": {"diff": d_pf["diff"], "lo": d_pf["lo"], "hi": d_pf["hi"], "excl0": d_pf["excludes_zero"]},
        }
        r = out[str(pp)]
        print(f"{pp:>4} | "
              f"CR{r['override']['cr']:.2f}/pf{r['override']['perf']:+.2f} | "
              f"CR{r['mask_defer']['cr']:.2f}/pf{r['mask_defer']['perf']:+.2f} | "
              f"CR{r['asra_t']['cr']:.2f}/pf{r['asra_t']['perf']:+.2f} | "
              f"dCR={d_cr['diff']:+.2f}CI[{d_cr['lo']:+.2f},{d_cr['hi']:+.2f}] "
              f"dpf={d_pf['diff']:+.2f}CI[{d_pf['lo']:+.2f},{d_pf['hi']:+.2f}]")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=2)
    print("saved", a.out)

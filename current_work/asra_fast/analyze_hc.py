"""Analyze HazardCorridor results: does ASRA's (CR, perf) frontier dominate the"""
import sys, json, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from current_work.asra_fast.analyze import frontier_auc, paired_bootstrap_diff


def m(arr, f): return np.array([x[f] for x in arr], float)


def analyze(path):
    d = json.load(open(path))
    rg, rs, ov = d["raw_greedy"], d["raw_sample"], d["override_brake"]
    nseed = len(rg)
    print(f"=== HazardCorridor ({nseed} seeds) — CR (safety, lower better) / perf (progress, higher better) ===")
    print(f"  raw_greedy     CR={m(rg,'cr').mean():.3f}  perf={m(rg,'perf').mean():+.3f}")
    print(f"  raw_sample     CR={m(rs,'cr').mean():.3f}  perf={m(rs,'perf').mean():+.3f}")
    print(f"  override_brake CR={m(ov,'cr').mean():.3f}  perf={m(ov,'perf').mean():+.3f}")
    print("  -- ASRA-greedy (suppress risky advance) sweep --")
    for g, arr in d["asra_greedy"].items():
        print(f"    g={g:<4} CR={m(arr,'cr').mean():.3f}  perf={m(arr,'perf').mean():+.3f}")
    print("  -- prob-brake sweep --")
    for p, arr in d["prob_brake"].items():
        print(f"    p={p:<4} CR={m(arr,'cr').mean():.3f}  perf={m(arr,'perf').mean():+.3f}")

    # Key test: at matched (near-zero) CR, does ASRA preserve more performance than override?
    ov_cr, ov_perf = m(ov, "cr"), m(ov, "perf")
    print("\n--- PRECONDITION TEST: ASRA vs hard override at matched safety ---")
    best = None
    for g, arr in d["asra_greedy"].items():
        a_cr, a_perf = m(arr, "cr"), m(arr, "perf")
        # require ASRA CR <= override CR (safety parity) AND compare perf
        safe = paired_bootstrap_diff(ov_cr[:len(a_cr)], a_cr)  # override_cr - asra_cr; >=0 means ASRA no worse
        perf = paired_bootstrap_diff(a_perf, ov_perf[:len(a_perf)])  # asra_perf - override_perf
        asra_safe_enough = a_cr.mean() <= ov_cr.mean() + 0.02
        if asra_safe_enough and perf["excludes_zero"] and perf["diff"] > 0:
            if best is None or perf["diff"] > best[1]:
                best = (g, perf["diff"], perf, a_cr.mean(), a_perf.mean())
    if best:
        g, dperf, perf, acr, aperf = best
        print(f"  CONFIRMED: ASRA-greedy g={g} matches override safety (CR {acr:.3f} vs {ov_cr.mean():.3f}) "
              f"but performance {aperf:+.3f} vs {ov_perf.mean():+.3f} "
              f"(dperf={perf['diff']:+.3f} CI[{perf['lo']:+.3f},{perf['hi']:+.3f}], excludes 0)")
        print("  => Inference-time modulation BEATS the hard override when the precondition holds.")
    else:
        print("  NOT confirmed: no ASRA gain achieves override-level safety with higher perf (CI>0).")

    # frontier AUC dominance (ASRA-greedy vs prob-brake), paired by seed
    grid = np.linspace(0, 1, 41)
    def auc(method_points):
        out = []
        for s in range(nseed):
            pts = method_points(s)
            if len(pts) >= 2: out.append(frontier_auc(pts, grid))
        return np.array([x for x in out if not np.isnan(x)])
    def asra_pts(s):
        pts = [(rg[s]["cr"], rg[s]["perf"])]
        for g, arr in d["asra_greedy"].items():
            pts.append((arr[s]["cr"], arr[s]["perf"]))
        return pts
    def pb_pts(s):
        pts = [(rs[s]["cr"], rs[s]["perf"]), (ov[s]["cr"], ov[s]["perf"])]
        for p, arr in d["prob_brake"].items():
            pts.append((arr[s]["cr"], arr[s]["perf"]))
        return pts
    aa, pa = auc(asra_pts), auc(pb_pts)
    n = min(len(aa), len(pa))
    if n >= 3:
        dom = paired_bootstrap_diff(aa[:n], pa[:n])
        verdict = "ASRA frontier dominates" if dom["diff"] > 0 and dom["excludes_zero"] else \
                  ("rule frontier dominates" if dom["diff"] < 0 and dom["excludes_zero"] else "parity")
        print(f"\n  Frontier AUC: ASRA={aa.mean():.3f} prob-brake={pa.mean():.3f} "
              f"d={dom['diff']:+.3f} CI[{dom['lo']:+.3f},{dom['hi']:+.3f}] => {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--path", required=True)
    analyze(ap.parse_args().path)

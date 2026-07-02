"""Decode-matched re-analysis of the multi-policy JSONs."""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast.analyze import paired_bootstrap_diff


def load(path):
    d = json.load(open(path))
    G = defaultdict(lambda: defaultdict(list))  # cond-key -> policy -> [(cr,speed)]
    for r in d["results"]:
        if r["method"] == "asra": key = ("asra", round(r["gain"], 3), r["decode"])
        elif r["method"] == "mask_defer": key = ("mask",)
        elif r.get("kind") == "ttc_brake": key = ("ttc", round(r["ttc_k"], 3))
        elif r.get("kind") == "prob_brake": key = ("pb", round(r["temp"], 3))
        else: key = (r.get("kind", "?"),)
        G[key][r["policy"]].append((r["cr"], r["speed"]))
    return d, G


def frontier(G, keys, npol):
    out = {}
    for pi in range(npol):
        pts = [(np.array(G[k][pi])[:, 0].mean(), np.array(G[k][pi])[:, 1].mean()) for k in keys if pi in G[k]]
        out[pi] = pts
    return out


def best_at(pts, tgt):
    c = [p[1] for p in pts if p[0] <= tgt + 1e-9]
    return max(c) if c else np.nan


def compare(name, Apts, Bpts, npol, levels):  # A - B at matched CR
    for tgt in levels:
        a = np.array([best_at(Apts[pi], tgt) for pi in range(npol)])
        b = np.array([best_at(Bpts[pi], tgt) for pi in range(npol)])
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() >= 3:
            dd = paired_bootstrap_diff(a[ok], b[ok])
            verd = "A>B" if (dd["excludes_zero"] and dd["diff"] > 0) else ("A<B" if (dd["excludes_zero"] and dd["diff"] < 0) else "TIE")
            print(f"    @CR<={tgt}: n={ok.sum()} d={dd['diff']:+.2f} CI[{dd['lo']:+.2f},{dd['hi']:+.2f}]  {verd}")
        else:
            print(f"    @CR<={tgt}: n={ok.sum()} (insufficient)")


def fixed_gain_vs_mask(G, gg, npol, levels, mask_favoring=True):
    """ACID TEST for selection bias: instead of ASRA picking its best of 7 gains per policy, use a
    SINGLE a-priori fixed gain (one ASRA-greedy point per policy). Mask keeps its full 2-point
    frontier -> selection now FAVORS the mask. If a fixed-gain ASRA-greedy STILL beats mask at
    matched CR, the advantage is a real mechanism effect, not gain-selection (winner's-curse) bias."""
    M = frontier(G, [("mask",), ("noop_greedy",)], npol)
    for g in gg:
        AGg = {}                                                     # single fixed-gain ASRA-greedy point per policy
        for pi in range(npol):
            k = ("asra", round(g, 3), "greedy")
            AGg[pi] = [(np.array(G[k][pi])[:, 0].mean(), np.array(G[k][pi])[:, 1].mean())] if pi in G[k] else []
        row = f"    gain={g:>4}: "
        cells = []
        for tgt in levels:
            a = np.array([best_at(AGg[pi], tgt) for pi in range(npol)])
            m = np.array([best_at(M[pi], tgt) for pi in range(npol)])
            ok = ~(np.isnan(a) | np.isnan(m))
            if ok.sum() >= 3:
                dd = paired_bootstrap_diff(a[ok], m[ok])
                verd = "A>M" if (dd["excludes_zero"] and dd["diff"] > 0) else ("A<M" if (dd["excludes_zero"] and dd["diff"] < 0) else "tie")
                cells.append(f"@{tgt} d={dd['diff']:+.2f}[{dd['lo']:+.2f},{dd['hi']:+.2f}] {verd} (n{ok.sum()})")
            else:
                cells.append(f"@{tgt} n{ok.sum()}")
        print(row + " | ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+")
    ap.add_argument("--levels", nargs="+", type=float, default=[0.5, 0.7])
    a = ap.parse_args()
    for path in a.paths:
        d, G = load(path); npol = len(d["policies"]); levels = a.levels
        scen = "?"
        for p in d["policies"]:
            s = str(p).lower()
            scen = "roundabout" if "roundabout" in s else ("merge" if "merge" in s else ("highway" if "highway" in s or "driving" in s else scen))
        gg = d["g_grid"]
        asra_greedy = [("asra", round(g, 3), "greedy") for g in gg]
        asra_sample = [("asra", round(g, 3), "sample") for g in gg]
        mask_g = [("mask",), ("noop_greedy",)]                       # mask frontier is greedy-only
        AG = frontier(G, asra_greedy, npol); AS = frontier(G, asra_sample, npol); M = frontier(G, mask_g, npol)
        print(f"\n=== {Path(path).name}  (scenario={scen}, {npol} policies) ===")
        print("  [decode-MATCHED]  ASRA-greedy(best-of-gains)  vs  mask(greedy):")
        compare("AGvsM", AG, M, npol, levels)
        print("  [decode-mismatch] ASRA-sample  vs  mask(greedy):  (gap here = partly decode, not mechanism)")
        compare("ASvsM", AS, M, npol, levels)
        print("  [ACID TEST: FIXED single gain, selection favors mask] ASRA-greedy@g vs mask(greedy):")
        fixed_gain_vs_mask(G, gg, npol, levels)

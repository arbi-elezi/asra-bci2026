"""Aggregate the SmolLM2-135M LLM cross-architecture frontier across all seed-range JSONs."""
from __future__ import annotations
import sys, json, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast.analyze import paired_bootstrap_diff


def load_all(pattern="current_work/results_v3/llm_frontier*.json"):
    rows, seeds = [], set()
    for p in sorted(glob.glob(pattern)):
        d = json.load(open(p))
        for r in d["results"]:
            rows.append(r); seeds.add(r["seed"])
    return rows, sorted(seeds)


def by_cond_seed(rows):
    """cond -> seed -> (cr, speed).  cond is the human 'cond' label."""
    G = defaultdict(dict)
    for r in rows:
        G[r["cond"]][r["seed"]] = (r["cr"], r["speed"])
    return G


def mean_over_seeds(G, cond, seeds):
    vals = [G[cond][s] for s in seeds if s in G.get(cond, {})]
    if not vals: return None
    a = np.array(vals); return a[:, 0].mean(), a[:, 1].mean(), len(vals)


if __name__ == "__main__":
    rows, seeds = load_all()
    G = by_cond_seed(rows)
    print(f"=== LLM (SmolLM2-135M) cross-architecture frontier: {len(seeds)} seeds {seeds} ===")
    print("\n(1) Decode-controlled FRONTIER (ASRA gain g, greedy decode) -- gain knob sweeps risk/perf:")
    print(f"  {'condition':22s} {'CR':>6} {'speed':>7} {'nseed':>6}")
    for g in [0.0, 1.0, 3.0, 6.0]:
        cond = f"asra_g{g}_greedy"
        m = mean_over_seeds(G, cond, seeds)
        if m: print(f"  {cond:22s} {m[0]:6.3f} {m[1]:7.2f} {m[2]:6d}")
    print("  baselines:")
    for cond in ["noop_greedy", "noop_sample", "mask_defer", "ttc_brake@2", "prob_brake@.5"]:
        m = mean_over_seeds(G, cond, seeds)
        if m: print(f"  {cond:22s} {m[0]:6.3f} {m[1]:7.2f} {m[2]:6d}")
    print("  asra sample decode:")
    for g in [1.0, 3.0, 6.0]:
        cond = f"asra_g{g}_sample"
        m = mean_over_seeds(G, cond, seeds)
        if m: print(f"  {cond:22s} {m[0]:6.3f} {m[1]:7.2f} {m[2]:6d}")

    # (2) mask-tie: mask_defer vs noop_greedy, paired across seeds
    def paired(condA, condB):
        s_common = [s for s in seeds if s in G.get(condA, {}) and s in G.get(condB, {})]
        a = np.array([G[condA][s] for s in s_common]); b = np.array([G[condB][s] for s in s_common])
        return s_common, a, b
    print("\n(2) mask-tie (mask_defer vs noop_greedy), paired bootstrap CI across seeds:")
    sc, a, b = paired("mask_defer", "noop_greedy")
    if len(sc) >= 2:
        dcr = paired_bootstrap_diff(a[:, 0], b[:, 0]); dsp = paired_bootstrap_diff(a[:, 1], b[:, 1])
        print(f"  n={len(sc)}  dCR(mask-noop)={dcr['diff']:+.3f} CI[{dcr['lo']:+.3f},{dcr['hi']:+.3f}]"
              f"  dspeed={dsp['diff']:+.3f} CI[{dsp['lo']:+.3f},{dsp['hi']:+.3f}]"
              f"  => {'TIE' if not dcr['excludes_zero'] and not dsp['excludes_zero'] else 'DIFFER'}")

    # (3) ASRA-greedy vs mask at matched CR: per seed, best asra-greedy speed with CR <= mask CR
    print("\n(3) ASRA-greedy vs mask at matched CR (per-seed best asra-greedy pt with CR<=mask CR):")
    ag = [f"asra_g{g}_greedy" for g in [0.0, 1.0, 3.0, 6.0]]
    da, dm = [], []
    for s in seeds:
        if s not in G.get("mask_defer", {}): continue
        mcr, msp = G["mask_defer"][s]
        cand = [G[c][s][1] for c in ag if s in G.get(c, {}) and G[c][s][0] <= mcr + 1e-9]
        if cand: da.append(max(cand)); dm.append(msp)
    if len(da) >= 2:
        dd = paired_bootstrap_diff(np.array(da), np.array(dm))
        print(f"  n={len(da)}  dspeed(asraG-mask at CR<=mask)={dd['diff']:+.3f} CI[{dd['lo']:+.3f},{dd['hi']:+.3f}]"
              f"  => {'ASRA faster' if dd['excludes_zero'] and dd['diff']>0 else 'tie'}")
    else:
        print(f"  n={len(da)} (insufficient matched points)")

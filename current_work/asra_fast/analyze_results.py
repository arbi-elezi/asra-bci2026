"""Consume a results_v3 JSON and produce frontier + pre-registered hypothesis tests."""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from current_work.asra_fast.analyze import (
    pareto_frontier, frontier_auc, paired_bootstrap_diff, spearman_ci, holm_bonferroni,
)

PERF = "mean_speed"  # primary performance axis


def group(results, key_fn):
    """Return {key: {seed: record}} so conditions can be paired by seed."""
    g = defaultdict(dict)
    for r in results:
        g[key_fn(r)][r["seed"]] = r
    return g


def cond_key(r):
    if r["method"] == "asra":
        return ("asra", round(r["gain"], 3), r["decode"])
    if r["kind"] == "ttc_brake":
        return ("ttc", round(r.get("ttc_k", 0), 3))
    if r["kind"] == "prob_brake":
        return ("pbrake", round(r.get("temp", 0), 3))
    return (r["kind"],)


def per_seed(cond_map, seeds, field):
    return np.array([cond_map[s][field] for s in seeds if s in cond_map], dtype=np.float64)


def analyze(path: str, perf=PERF, return_floor_frac=0.10):
    d = json.load(open(path))
    res = d["results"]; seeds = sorted(set(r["seed"] for r in res))
    G = group(res, cond_key)

    def stats(key):
        m = G.get(key, {})
        if not m:
            return None
        return {"cr": per_seed(m, seeds, "cr"), "perf": per_seed(m, seeds, perf),
                "ret": per_seed(m, seeds, "ret"), "brake": per_seed(m, seeds, "brake_frac")}

    noop_g = stats(("noop_greedy",))
    noop_s = stats(("noop_sample",))

    print(f"\n=== {d['scenario']} | base CR(train)~{d.get('base_train_cr')} step {d.get('base_step')} "
          f"| {len(seeds)} seeds x {d['n_ep']} eps | perf={perf} ===")
    print(f"{'condition':26s} {'CR':>14s} {'perf':>8s} {'ret':>7s} {'brake':>6s}")

    def line(name, st):
        if st is None: print(f"{name:26s}  (missing)"); return
        print(f"{name:26s} {st['cr'].mean():.3f}+-{st['cr'].std():.3f}  "
              f"{st['perf'].mean():7.2f} {st['ret'].mean():7.1f} {st['brake'].mean():6.2f}")
    line("noop_greedy (RAW)", noop_g)
    line("noop_sample (RAW)", noop_s)

    # ---- ASRA frontiers per decode mode ----
    for dm in d["decode_modes"]:
        print(f"  -- ASRA [{dm}] --")
        for g in d["g_grid"]:
            line(f"   g={g}", stats(("asra", round(g, 3), dm)))

    # ---- rule frontiers ----
    print("  -- TTC-brake sweep --")
    for k in d["ttc_grid"]:
        line(f"   ttc<{k}", stats(("ttc", round(k, 3))))
    print("  -- prob-brake sweep --")
    for p in d["pbrake_grid"]:
        line(f"   p={p}", stats(("pbrake", round(p, 3))))

    # ================= HYPOTHESIS TESTS =================
    print("\n--- PRE-REGISTERED TESTS ---")
    tests = {}

    # H1: best validation g* on GREEDY track beats RAW-greedy on CR, return floor respected
    base_cr = noop_g["cr"].mean(); base_ret = noop_g["ret"].mean()
    floor = base_ret - return_floor_frac * abs(base_ret)
    best = None
    for g in d["g_grid"]:
        if g == 0: continue
        st = stats(("asra", round(g, 3), "greedy"))
        if st is None: continue
        diff = paired_bootstrap_diff(noop_g["cr"][:len(st["cr"])], st["cr"])  # base - asra (positive=improvement)
        ok_floor = st["ret"].mean() >= floor
        cand = (diff["diff"], g, diff, st, ok_floor)
        if diff["excludes_zero"] and diff["diff"] > 0 and ok_floor:
            if best is None or diff["diff"] > best[0]:
                best = cand
    if best:
        _, g, diff, st, _ = best
        print(f"H1 GREEDY-track: g*={g} CR {base_cr:.3f}->{st['cr'].mean():.3f} "
              f"(d={diff['diff']:+.3f} CI[{diff['lo']:.3f},{diff['hi']:.3f}]) "
              f"ret {base_ret:.1f}->{st['ret'].mean():.1f} (floor {floor:.1f}) speed {st['perf'].mean():.2f}  => CONFIRMED")
        tests["H1"] = 0.01
    else:
        print(f"H1 GREEDY-track: NO g reduces CR below raw-greedy ({base_cr:.3f}) with CI>0 under return floor  => NOT confirmed")
        tests["H1"] = 0.99

    # H2: monotonicity CR vs g at fixed (greedy) decode
    gs, crs = [], []
    for g in d["g_grid"]:
        st = stats(("asra", round(g, 3), "greedy"))
        if st is not None:
            gs += [g] * len(st["cr"]); crs += list(st["cr"])
    sp = spearman_ci(np.array(gs), np.array(crs))
    print(f"H2 monotonicity CR vs g (greedy): rho={sp['rho']:+.3f} CI[{sp['lo']:.3f},{sp['hi']:.3f}] "
          f"=> {'CONFIRMED (CR rises with g)' if sp['rho']>0 and sp['excludes_zero'] else 'see sign'}")

    # H3: ASRA frontier vs swept-rule frontier IN THE USEFUL LOW-CR REGION only.
    # (Integrating over all CR would reward pointless high-CR/high-speed points.)
    cr_cap = float(max(0.30, noop_s["cr"].mean() if noop_s else 0.30))
    cr_grid = np.linspace(0.0, cr_cap, 31)
    # global per-seed speed floor (shared by both methods) so a method is penalized where it
    # offers no operating point -- avoids crediting a method in a CR region it cannot reach.
    def seed_floor(s, getters):
        vals = [p[1] for gp in getters for p in gp(s)]
        return min(vals) if vals else 0.0
    # matched-CR speed comparison: at each target CR, best speed achievable at CR<=target
    def best_speed_at(get_points, target):
        # per-seed: max perf among that method's points with cr <= target; mean over seeds
        vals = []
        for s in seeds:
            pts = [p for p in get_points(s) if p[0] <= target + 1e-9]
            if pts: vals.append(max(p[1] for p in pts))
        return np.array(vals, float)
    def method_auc_per_seed(get_points, other):
        aucs = []
        for s in seeds:
            pts = get_points(s)
            if len(pts) >= 2:
                aucs.append(frontier_auc(pts, cr_grid, floor=seed_floor(s, [get_points, other])))
        return np.array([a for a in aucs if not np.isnan(a)], dtype=np.float64)

    def asra_points(s):
        pts = []
        for dm in d["decode_modes"]:
            for g in d["g_grid"]:
                m = G.get(("asra", round(g, 3), dm), {})
                if s in m: pts.append((m[s]["cr"], m[s][perf]))
        return pts
    def ttc_points(s):
        pts = []
        for k in d["ttc_grid"]:
            m = G.get(("ttc", round(k, 3)), {})
            if s in m: pts.append((m[s]["cr"], m[s][perf]))
        # include raw greedy/sample as available operating points of "do nothing + brake rule"
        for nk in [("noop_greedy",), ("noop_sample",)]:
            m = G.get(nk, {})
            if s in m: pts.append((m[s]["cr"], m[s][perf]))
        return pts
    # rule frontier = swept prob-brake + ttc + raw points (the tunable trivial alternatives)
    def rule_points(s):
        pts = []
        for key in [("pbrake", round(p, 3)) for p in d["pbrake_grid"]] + \
                   [("ttc", round(k, 3)) for k in d["ttc_grid"]] + [("noop_greedy",), ("noop_sample",)]:
            m = G.get(key, {})
            if s in m: pts.append((m[s]["cr"], m[s][perf]))
        return pts
    asra_auc = method_auc_per_seed(asra_points, rule_points)
    rule_auc = method_auc_per_seed(rule_points, asra_points)
    n = min(len(asra_auc), len(rule_auc))
    if n >= 3:
        dom = paired_bootstrap_diff(asra_auc[:n], rule_auc[:n])
        verdict = "ASRA dominates" if (dom["excludes_zero"] and dom["diff"] > 0) else \
                  ("rule dominates" if (dom["excludes_zero"] and dom["diff"] < 0) else "parity")
        print(f"H3 useful-region frontier AUC (perf vs CR<= {cr_cap:.2f}): "
              f"ASRA={asra_auc.mean():.2f} rule={rule_auc.mean():.2f} "
              f"d={dom['diff']:+.2f} CI[{dom['lo']:.2f},{dom['hi']:.2f}] => {verdict}")
    # matched-CR speed gaps (ASRA - rule) at useful CR levels
    for tgt in [0.05, 0.10, 0.20]:
        if tgt > cr_cap + 1e-6: continue
        a = best_speed_at(asra_points, tgt); r = best_speed_at(rule_points, tgt)
        nn = min(len(a), len(r))
        if nn >= 3:
            gap = paired_bootstrap_diff(a[:nn], r[:nn])
            tag = "ASRA better" if gap["diff"] > 0 and gap["excludes_zero"] else \
                  ("rule better" if gap["diff"] < 0 and gap["excludes_zero"] else "tie")
            print(f"   @CR<={tgt:.2f}: speed ASRA={a.mean():.2f} rule={r.mean():.2f} "
                  f"d={gap['diff']:+.2f} CI[{gap['lo']:+.2f},{gap['hi']:+.2f}] => {tag}")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--perf", default=PERF)
    a = ap.parse_args()
    analyze(a.path, perf=a.perf)

"""Generate publication figures from results_v3 JSON(s). 300 DPI, real data only."""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from current_work.asra_fast.analyze import pareto_frontier


def load(path):
    d = json.load(open(path))
    g = defaultdict(lambda: defaultdict(list))
    for r in d["results"]:
        if r["method"] == "asra":
            key = ("asra", r["decode"])
            g[key][round(r["gain"], 3)].append(r)
        else:
            k = r["kind"]
            knob = round(r.get("ttc_k", r.get("temp", 0)), 3)
            g[(k,)][knob].append(r)
    return d, g


def agg(recs, field):
    return float(np.mean([x[field] for x in recs]))


def frontier_fig(path, out, perf="mean_speed"):
    d, g = load(path)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    def curve(key, **kw):
        knobs = sorted(g[key].keys())
        pts = [(agg(g[key][k], "cr"), agg(g[key][k], perf)) for k in knobs]
        if not pts: return
        ax.plot([p[0] for p in pts], [p[1] for p in pts], **kw)

    # ASRA frontiers
    curve(("asra", "greedy"), marker="o", lw=2, color="#1f77b4", label="ASRA (greedy decode)")
    if ("asra", "sample") in g:
        curve(("asra", "sample"), marker="s", lw=1.5, color="#2ca02c",
              ls="--", label="ASRA (stochastic decode)")
    # swept rule frontiers
    curve(("ttc_brake",), marker="^", lw=1.5, color="#d62728", label="TTC-brake (swept threshold)")
    curve(("prob_brake",), marker="v", lw=1.2, color="#ff7f0e", ls=":", label="prob-brake (swept p)")
    # raw baselines
    for nk, c, mk, lab in [(("noop_greedy",), "black", "*", "frozen policy (greedy)"),
                            (("noop_sample",), "gray", "X", "frozen policy (stochastic)")]:
        if nk in g:
            r = g[nk][sorted(g[nk])[0]]
            ax.scatter([agg(r, "cr")], [agg(r, perf)], c=c, marker=mk, s=140,
                       zorder=5, label=lab, edgecolors="white")

    ax.set_xlabel("Collision rate (safety →  lower is better)")
    ax.set_ylabel("Mean speed, m/s (performance →  higher is better)")
    ax.set_title(f"Risk–performance frontier ({d['scenario']}, "
                 f"{d.get('vehicles')} veh, density {d.get('density')})")
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print("saved", out)


def cr_vs_g_fig(path, out):
    d, g = load(path)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for dm, c in [("greedy", "#1f77b4"), ("sample", "#2ca02c")]:
        key = ("asra", dm)
        if key not in g: continue
        gs = sorted(g[key].keys())
        cr = [agg(g[key][x], "cr") for x in gs]
        bf = [agg(g[key][x], "brake_frac") for x in gs]
        a1.plot(gs, cr, marker="o", color=c, label=f"ASRA ({dm})")
        a2.plot(gs, bf, marker="o", color=c, label=f"ASRA ({dm})")
    for nk, c, lab in [(("noop_greedy",), "black", "raw greedy"), (("noop_sample",), "gray", "raw stochastic")]:
        if nk in g:
            r = g[nk][sorted(g[nk])[0]]
            a1.axhline(agg(r, "cr"), color=c, ls="--", lw=1, label=lab)
    a1.set_xlabel("salience gain g"); a1.set_ylabel("collision rate"); a1.set_title("CR vs gain (decode-controlled)")
    a2.set_xlabel("salience gain g"); a2.set_ylabel("brake-action fraction"); a2.set_title("Braking vs gain (degeneracy probe)")
    a1.legend(fontsize=8); a2.legend(fontsize=8); a1.grid(alpha=0.3); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--prefix", default="current_work/figs_v3/fig")
    a = ap.parse_args()
    Path(a.prefix).parent.mkdir(parents=True, exist_ok=True)
    frontier_fig(a.path, a.prefix + "_frontier.png")
    cr_vs_g_fig(a.path, a.prefix + "_crvsg.png")

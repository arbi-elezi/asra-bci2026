"""The 'boundary' figure: when does inference-time modulation beat a hard override?"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))


def driving_panel(ax, path):
    d = json.load(open(path))
    g = defaultdict(lambda: defaultdict(list))
    for r in d["results"]:
        if r["method"] == "asra": g[("asra", r["decode"])][round(r["gain"],3)].append(r)
        else:
            k = r["kind"]; knob = round(r.get("ttc_k", r.get("temp", 0)), 3)
            g[(k,)][knob].append(r)
    def pts(key):
        ks = sorted(g[key]); return [np.mean([x["cr"] for x in g[key][k]]) for k in ks], \
                                    [np.mean([x["mean_speed"] for x in g[key][k]]) for k in ks]
    for key, c, m, lab in [(("asra","greedy"),"#1f77b4","o","ASRA (greedy)"),
                            (("asra","sample"),"#17becf","s","ASRA (stochastic)"),
                            (("prob_brake",),"#ff7f0e","v","brake-rule (swept)"),
                            (("ttc_brake",),"#d62728","^","TTC-brake (swept)")]:
        if key in g:
            x,y = pts(key); ax.plot(x,y,marker=m,color=c,label=lab,lw=1.6,ms=5)
    for nk,c,m,lab in [(("noop_greedy",),"black","*","frozen (greedy)"),(("noop_sample",),"gray","X","frozen (stochastic)")]:
        if nk in g:
            r=g[nk][sorted(g[nk])[0]]; ax.scatter([np.mean([x['cr'] for x in r])],[np.mean([x['mean_speed'] for x in r])],
                       c=c,marker=m,s=130,zorder=5,edgecolors="white",label=lab)
    ax.set_xlabel("collision rate  (lower = safer)"); ax.set_ylabel("mean speed (m/s)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)


def controlled_panel(ax, path):
    d = json.load(open(path))
    def mu(a,f): return float(np.mean([x[f] if isinstance(x,dict) else x for x in a]))
    rg = d["raw_greedy"]; ov = d["override_brake"]
    ax.scatter([mu(rg["cr"],0) if False else np.mean(rg["cr"])],[np.mean(rg["perf"])],
               c="black",marker="*",s=150,zorder=5,edgecolors="white",label="frozen (greedy)")
    ax.scatter([np.mean(ov["cr"])],[np.mean(ov["perf"])],c="#d62728",marker="^",s=130,zorder=5,
               edgecolors="white",label="hard override (brake)")
    ax_x=[np.mean(d["asra_greedy"][g]["cr"]) for g in d["asra_greedy"]]
    ax_y=[np.mean(d["asra_greedy"][g]["perf"]) for g in d["asra_greedy"]]
    ax.plot(ax_x,ax_y,marker="o",color="#1f77b4",label="ASRA (route to alt.)",lw=1.6)
    pb_x=[np.mean(d["prob_brake"][p]["cr"]) for p in d["prob_brake"]]
    pb_y=[np.mean(d["prob_brake"][p]["perf"]) for p in d["prob_brake"]]
    ax.plot(pb_x,pb_y,marker="v",color="#ff7f0e",label="brake-rule (swept)",lw=1.4)
    ax.set_xlabel("collision rate  (lower = safer)"); ax.set_ylabel("task performance")
    ax.set_title("(b) Controlled: precondition satisfied\nmodulation dominates override", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)


if __name__ == "__main__":
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    driving_panel(a1, "current_work/results_v3/assertive_final.json")
    a1.set_title("(a) Driving, standard-trained policy (precondition violated):\nthe swept brake rule matches/dominates ASRA", fontsize=10)
    controlled_panel(a2, "current_work/results_v3/controlled_demo.json")
    a2.set_title("(b) Controlled task, policy encodes a safe alternative\n(precondition satisfied): ASRA dominates the override", fontsize=10)
    fig.suptitle("The precondition boundary: when inference-time modulation beats a hard override", fontsize=12)
    fig.tight_layout()
    out = "current_work/figs_v3/fig_boundary.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300); print("saved", out)

    # supplementary: the moderate driving policy (partial, policy-specific win)
    figc, axc = plt.subplots(figsize=(5.8, 4.4))
    driving_panel(axc, "current_work/results_v3/moderate_final.json")
    axc.set_title("Driving, an extremely risk-seeking policy: ASRA halves CR\nand leads at CR~0.55, but the rule owns the lowest-CR corner", fontsize=9)
    figc.tight_layout(); figc.savefig("current_work/figs_v3/fig_moderate.png", dpi=300)
    print("saved current_work/figs_v3/fig_moderate.png")

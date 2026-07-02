"""Figure: dynamic context shift. Windowed collision rate + hormone level vs episode, with"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("current_work/results_v3/context_shift.json"))
target = d["target"]
COL = {"naive_asra": "#d62728", "fixed": "#ff7f0e", "online": "#1f77b4"}
LAB = {"naive_asra": "naive ASRA (fixed gain)", "fixed": "fixed hormone (static)",
       "online": "online hormone (endocrine feedback)"}


def windowed(x, w=12):
    x = np.asarray(x, float)
    return np.array([x[max(0, i-w+1):i+1].mean() for i in range(len(x))])


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
stream0 = d["methods"]["online"]["stream_seed0"]
dens = [r["density"] for r in stream0]
# shift boundaries
bounds = [i for i in range(1, len(dens)) if dens[i] != dens[i-1]]

for mode in ["naive_asra", "fixed", "online"]:
    s = d["methods"][mode]["stream_seed0"]
    eps = [r["ep"] for r in s]
    cr = windowed([r["collision"] for r in s])
    ax1.plot(eps, cr, color=COL[mode], label=LAB[mode], lw=1.8)
    ax2.plot(eps, [r["ctrl"] for r in s], color=COL[mode], lw=1.6, label=LAB[mode])

ax1.axhline(target, color="black", ls="--", lw=1, label=f"safety setpoint (CR={target})")
for b in bounds:
    ax1.axvline(b, color="gray", ls=":", alpha=0.6); ax2.axvline(b, color="gray", ls=":", alpha=0.6)
# annotate densities per segment
i0 = 0
for b in bounds + [len(dens)]:
    mid = (i0 + b) // 2
    ax1.text(mid, ax1.get_ylim()[1]*0.96, f"ρ={dens[i0]}", ha="center", va="top", fontsize=8, color="gray")
    i0 = b
ax1.set_ylabel("windowed collision rate"); ax1.set_title("Context shifts within one run: does the mechanism track the safety setpoint?")
ax1.legend(fontsize=8, loc="upper left"); ax1.grid(alpha=0.3)
ax2.set_ylabel("hormone / gain level"); ax2.set_xlabel("episode (context ρ shifts at dotted lines)")
ax2.legend(fontsize=8, loc="upper left"); ax2.grid(alpha=0.3)
fig.tight_layout()
out = "current_work/figs_v3/fig_context_shift.png"
Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300); print("saved", out)

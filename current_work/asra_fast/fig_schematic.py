"""Schematic: the endocrine modulation loop for a frozen policy + how the experiment measures it."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

def box(x, y, w, h, text, fc, ec="#333", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)

def arrow(x1, y1, x2, y2, style="-|>", color="#333", lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls))

# main flow (bottom row)
box(1, 26, 13, 9, "state $s_t$\n$\\in\\mathbb{R}^{12}$", "#eef3fa")
box(18, 24, 20, 13, "FROZEN policy $\\pi_{W_0}$\n(weights $W_0$ never\nretrained)", "#dfefff", bold=True)
box(43, 24, 22, 13, "modulated logits\n$\\ell + \\mathbf{s}$,  $W_0{+}\\Delta W$\n$\\Rightarrow$ action $a_t$", "#e9f7e9", bold=True)
box(70, 26, 17, 9, "highway-env\ncontext $\\rho$ (density)", "#fff3e0")
box(90, 26, 9, 9, "outcome:\ncollision? /\nspeed", "#fde8e8", fs=8)

arrow(14, 30.5, 18, 30.5); arrow(38, 30.5, 43, 30.5); arrow(65, 30.5, 70, 30.5); arrow(87, 30.5, 90, 30.5)

# endocrine top path
box(30, 47, 20, 10, "salience gland $S_t\\in[0,1]$\n(independent detector;\nheadline: $S_t=$ TTC-cost)", "#f3e9fb", fs=8)
box(54, 47, 20, 10, "HORMONE  $g\\cdot S_t\\cdot R_t$\n(gain $g$ = the knob;\n$R_t$ = risk of greedy act)", "#efe3fb", fs=8, bold=True)
arrow(50, 52, 54, 52)
arrow(64, 47, 57, 37, color="#7a3fb0")            # hormone -> modulation (receptors)
ax.text(66, 42, "receptors:\n$\\Delta$logit (suppress risky)\n+ targeted $\\Delta W$", fontsize=7.5, color="#7a3fb0", va="center")
arrow(9, 35, 34, 47, color="#999", ls="--", lw=1.1)  # state -> salience
# homeostatic recovery
arrow(53, 24, 30, 24, color="#c0392b", lw=1.6)
ax.text(41.5, 21.2, "homeostatic recovery: $W\\!\\to\\!W_0$ (Fisher-weighted), hormone decays after threat", ha="center", fontsize=7.8, color="#c0392b")
# feedback outcome -> salience (next step)
arrow(94, 35, 45, 57, color="#999", ls=":", lw=1.0)
ax.text(72, 54, "next-step feedback", fontsize=7, color="#999")

# measurement banner
ax.add_patch(FancyBboxPatch((1, 6), 98, 12, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#f7f7f7", ec="#888", lw=1.2))
ax.text(50, 15.5, "How we measure it (decode-controlled, paired multi-seed):", ha="center", fontsize=9, fontweight="bold")
ax.text(50, 11.0, "sweep gain $g\\!\\in\\![0,g_{max}]$  $\\Rightarrow$  a curve on the (collision rate, speed) plane  =  the risk--performance FRONTIER;",
        ha="center", fontsize=8.3)
ax.text(50, 8.0, "compare vs continuously-SWEPT override rules (brake threshold);  shift the context $\\rho$  $\\Rightarrow$  online hormone re-tracks a safety setpoint.",
        ha="center", fontsize=8.3)

ax.text(50, 58.6, "ASRA: an endocrine layer that modulates a frozen policy at inference time (Phase-1 instance)",
        ha="center", fontsize=11, fontweight="bold")
fig.tight_layout()
out = "current_work/figs_v3/fig_schematic.png"; Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight"); print("saved", out)

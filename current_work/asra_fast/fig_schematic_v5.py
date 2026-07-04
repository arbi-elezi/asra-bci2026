"""Large camera-ready operator schematic: two-tier layout, print-size fonts.
Output: paper_latex/fig_operator.png (300 dpi)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, x, y, w, h, text, fc, fs=15, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                fc=fc, ec="#3a3a3a", lw=1.6))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, weight=weight)

def arrow(ax, x0, y0, x1, y1, color="#333333", lw=2.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=22, color=color, lw=lw))

fig, ax = plt.subplots(figsize=(10.0, 4.3))
ax.set_xlim(0, 10.0); ax.set_ylim(0, 4.3); ax.axis("off")

# main pipeline row
box(ax, 0.10, 2.75, 1.10, 1.15, "state\n$s_t$", "#eef2fa")
box(ax, 1.55, 2.75, 2.05, 1.15, "frozen policy\n$\\pi_0$ ($W_0$ never\nmodified)", "#dbe7f7", fs=13.5)
box(ax, 3.95, 2.75, 2.70, 1.15, "tilted scores\n$\\ell(s,\\cdot)-g\\,S(s)\\,Q_c(s,\\cdot)$", "#d9f0dd", fs=14)
box(ax, 7.00, 2.75, 1.20, 1.15, "action\n$a_t$", "#fdf3d8")
box(ax, 8.55, 2.75, 1.35, 1.15, "outcome", "#f6e0e0", fs=13.5)
arrow(ax, 1.20, 3.32, 1.55, 3.32); arrow(ax, 3.60, 3.32, 3.95, 3.32)
arrow(ax, 6.65, 3.32, 7.00, 3.32); arrow(ax, 8.20, 3.32, 8.55, 3.32)

# inputs row
box(ax, 1.30, 0.95, 2.75, 1.05, "salience $S(s)\\in[0,1]$\n(independent detector)", "#f3e6f5", fs=13, weight="normal")
box(ax, 4.65, 0.95, 2.75, 1.05, "cost critic $Q_c(s,\\cdot)$\n(frozen, fit offline)", "#f3e6f5", fs=13, weight="normal")
arrow(ax, 3.30, 2.00, 4.70, 2.75, color="#7a4b8a", lw=2.2)
arrow(ax, 5.85, 2.00, 5.45, 2.75, color="#7a4b8a", lw=2.2)

# corner cases strip (bottom)
ax.text(5.0, 0.38,
        "$g{=}0$: policy unchanged     $g{\\to}\\infty$: hard override     "
        "threshold: action mask     $S{\\to}0$: exact reversibility",
        ha="center", va="center", fontsize=13, style="italic")

fig.tight_layout()
fig.savefig("paper_latex/fig_operator.png", dpi=300, bbox_inches="tight")
print("saved large fig_operator.png")

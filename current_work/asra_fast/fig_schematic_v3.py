"""Compact single-band version of the operator schematic (half the height of v2).
Output: paper_latex/fig_operator.png (300 dpi)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, x, y, w, h, text, fc, fs=10.5, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010",
                                fc=fc, ec="#444444", lw=1.1))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, weight=weight)

def arrow(ax, x0, y0, x1, y1, color="#333333", lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, color=color, lw=lw))

fig, ax = plt.subplots(figsize=(10.6, 2.3))
ax.set_xlim(0, 10.6); ax.set_ylim(0, 2.3); ax.axis("off")

box(ax, 0.10, 1.10, 1.00, 0.75, "state\n$s_t$", "#eef2fa")
box(ax, 1.45, 1.10, 1.95, 0.75, "frozen policy $\\pi_0$\n($W_0$ never modified)", "#dbe7f7", fs=9.5)
box(ax, 3.75, 1.10, 2.45, 0.75, "tilted scores\n$\\ell(s,\\cdot)-g\\,S(s)\\,Q_c(s,\\cdot)$", "#d9f0dd", fs=10)
box(ax, 6.55, 1.10, 1.05, 0.75, "action\n$a_t$", "#fdf3d8")
box(ax, 7.95, 1.10, 1.15, 0.75, "outcome", "#f6e0e0", fs=9.5)
arrow(ax, 1.10, 1.475, 1.45, 1.475); arrow(ax, 3.40, 1.475, 3.75, 1.475)
arrow(ax, 6.20, 1.475, 6.55, 1.475); arrow(ax, 7.60, 1.475, 7.95, 1.475)

box(ax, 2.30, 0.10, 2.10, 0.62, "salience $S(s)\\in[0,1]$\n(independent detector)", "#f3e6f5", fs=8.8, weight="normal")
box(ax, 4.75, 0.10, 2.10, 0.62, "cost critic $Q_c(s,\\cdot)$\n(frozen, fit offline)", "#f3e6f5", fs=8.8, weight="normal")
arrow(ax, 3.55, 0.72, 4.45, 1.10, color="#7a4b8a"); arrow(ax, 5.60, 0.72, 5.20, 1.10, color="#7a4b8a")

ax.text(9.85, 1.72, "$g{=}0$: unchanged", ha="center", fontsize=9, style="italic")
ax.text(9.85, 1.42, "$g{\\to}\\infty$: override", ha="center", fontsize=9, style="italic")
ax.text(9.85, 1.12, "threshold: mask", ha="center", fontsize=9, style="italic")
ax.text(9.85, 0.70, "$S{\\to}0$: exact\nreversibility", ha="center", fontsize=9, style="italic", color="#8a2f2f")

fig.tight_layout()
fig.savefig("paper_latex/fig_operator.png", dpi=300, bbox_inches="tight")
print("saved compact fig_operator.png")

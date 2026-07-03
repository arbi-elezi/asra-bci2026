"""Camera-ready operator schematic, ~1.35x taller than v3 with larger type.
Output: paper_latex/fig_operator.png (300 dpi)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, x, y, w, h, text, fc, fs=13, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec="#444444", lw=1.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, weight=weight)

def arrow(ax, x0, y0, x1, y1, color="#333333", lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=17, color=color, lw=lw))

fig, ax = plt.subplots(figsize=(10.6, 3.1))
ax.set_xlim(0, 10.6); ax.set_ylim(0, 3.1); ax.axis("off")

box(ax, 0.10, 1.55, 1.00, 0.95, "state\n$s_t$", "#eef2fa")
box(ax, 1.45, 1.55, 2.00, 0.95, "frozen policy $\\pi_0$\n($W_0$ never\nmodified)", "#dbe7f7", fs=11.5)
box(ax, 3.80, 1.55, 2.55, 0.95, "tilted scores\n$\\ell(s,\\cdot)-g\\,S(s)\\,Q_c(s,\\cdot)$", "#d9f0dd", fs=12)
box(ax, 6.70, 1.55, 1.05, 0.95, "action\n$a_t$", "#fdf3d8")
box(ax, 8.10, 1.55, 1.15, 0.95, "outcome", "#f6e0e0", fs=11.5)
arrow(ax, 1.10, 2.02, 1.45, 2.02); arrow(ax, 3.45, 2.02, 3.80, 2.02)
arrow(ax, 6.35, 2.02, 6.70, 2.02); arrow(ax, 7.75, 2.02, 8.10, 2.02)

box(ax, 2.20, 0.15, 2.30, 0.85, "salience $S(s)\\in[0,1]$\n(independent detector)", "#f3e6f5", fs=11, weight="normal")
box(ax, 4.85, 0.15, 2.30, 0.85, "cost critic $Q_c(s,\\cdot)$\n(frozen, fit offline)", "#f3e6f5", fs=11, weight="normal")
arrow(ax, 3.60, 1.00, 4.60, 1.55, color="#7a4b8a"); arrow(ax, 5.85, 1.00, 5.35, 1.55, color="#7a4b8a")

ax.text(9.95, 2.42, "$g{=}0$: unchanged", ha="center", fontsize=11, style="italic")
ax.text(9.95, 2.06, "$g{\\to}\\infty$: override", ha="center", fontsize=11, style="italic")
ax.text(9.95, 1.70, "threshold: mask", ha="center", fontsize=11, style="italic")
ax.text(9.95, 1.05, "$S{\\to}0$: exact\nreversibility", ha="center", fontsize=11, style="italic",
        color="#8a2f2f")

fig.tight_layout()
fig.savefig("paper_latex/fig_operator.png", dpi=300, bbox_inches="tight")
print("saved bigger fig_operator.png")

"""Regenerate 's Fig. 1 to match the consequence-shaped-operator framing (new file)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, x, y, w, h, text, fc, fontsize=11, weight="bold", tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec="#444444", lw=1.2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color=tc)

def arrow(ax, x0, y0, x1, y1, color="#333333", lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=16, color=color, lw=lw))

fig, ax = plt.subplots(figsize=(10.2, 3.9))
ax.set_xlim(0, 10.2); ax.set_ylim(0, 3.9); ax.axis("off")

# main row
box(ax, 0.15, 2.05, 1.35, 0.85, "state\n$s_t$", "#eef2fa")
box(ax, 1.95, 2.05, 2.15, 0.85, "frozen policy $\\pi_0$\n(weights $W_0$,\nnever modified)", "#dbe7f7", fontsize=10)
box(ax, 4.55, 2.05, 2.30, 0.85, "tilted scores\n$\\ell(s,\\cdot)-g\\,S(s)\\,Q_c(s,\\cdot)$", "#d9f0dd", fontsize=10.5)
box(ax, 7.30, 2.05, 1.35, 0.85, "action\n$a_t$", "#fdf3d8")
box(ax, 9.05, 2.05, 1.05, 0.85, "outcome", "#f6e0e0", fontsize=10)

arrow(ax, 1.50, 2.475, 1.95, 2.475)
arrow(ax, 4.10, 2.475, 4.55, 2.475)
arrow(ax, 6.85, 2.475, 7.30, 2.475)
arrow(ax, 8.65, 2.475, 9.05, 2.475)

# bottom inputs: salience + cost critic
box(ax, 1.95, 0.55, 2.15, 0.80, "salience detector\n$S(s)\\in[0,1]$\n(independent of $\\pi_0$)", "#f3e6f5", fontsize=9.5, weight="normal")
box(ax, 4.55, 0.55, 2.30, 0.80, "cost critic $Q_c(s,\\cdot)$\n(frozen, fit offline,\nindependent of $\\pi_0$)", "#f3e6f5", fontsize=9.5, weight="normal")
arrow(ax, 3.02, 1.35, 5.30, 2.05, color="#7a4b8a")
arrow(ax, 5.70, 1.35, 5.70, 2.05, color="#7a4b8a")

# gain annotation above the tilt box
ax.text(5.70, 3.45, "one deployment-time gain $g$:   $g{=}0$ = policy unchanged   $\\cdot$   "
                    "$g{\\to}\\infty$ = hard override   $\\cdot$   threshold = action mask",
        ha="center", va="center", fontsize=10.5, style="italic", color="#333333")
arrow(ax, 5.70, 3.28, 5.70, 2.95, color="#888888", lw=1.2)

# reversibility note
ax.text(5.15, 0.13, "the tilt vanishes when danger passes ($S\\to0$): exact reversibility, no retraining",
        ha="center", va="center", fontsize=9.5, color="#8a2f2f")

fig.tight_layout()
fig.savefig("paper_latex/fig_operator.png", dpi=300, bbox_inches="tight")
print("saved paper_latex/fig_operator.png")

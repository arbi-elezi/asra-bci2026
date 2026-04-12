"""Generate REAL perturbation visualizations from experiment trace data.

Produces publication-quality figures showing actual weight perturbation,
gradient descent dynamics, fear signals, and layer-wise perturbation heatmaps.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def main():
    figdir = Path("figures")
    figdir.mkdir(exist_ok=True)

    traces = torch.load("data/perturbation_traces.pt", weights_only=False)

    # ── Fig 5: WDN + Fear + Gradient comparison (C1 vs C2 vs C8a) ──
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

    colors = {"C1_baseline": "#2ecc71", "C2_full_fra": "#3498db", "C8a_degraded": "#e74c3c"}
    labels = {"C1_baseline": "C1 (Baseline)", "C2_full_fra": "C2 (Full FRA)", "C8a_degraded": "C8a (Degraded 10%)"}

    # Panel 1: WDN over time
    ax = axes[0]
    for key in ["C1_baseline", "C2_full_fra", "C8a_degraded"]:
        t = traces[key]
        ax.plot(t["wdn"], color=colors[key], label=labels[key], linewidth=1.5)
    ax.set_ylabel("Weight Deviation Norm\n$\\|W_t - W_0\\|_F$")
    ax.set_title("Real-Time Weight Perturbation Dynamics (M3)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    # Panel 2: Fear signal
    ax = axes[1]
    for key in ["C2_full_fra", "C8a_degraded"]:
        t = traces[key]
        ax.fill_between(range(len(t["fear"])), t["fear"], alpha=0.3, color=colors[key])
        ax.plot(t["fear"], color=colors[key], label=labels[key], linewidth=1.2)
    ax.axhline(y=0.5, color="orange", linewidth=0.8, linestyle=":", label="SCL threshold")
    ax.axhline(y=0.05, color="green", linewidth=0.8, linestyle=":", label="DR threshold")
    ax.set_ylabel("Fear Signal $F_t$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Fear Signal Pipeline (Independent Detection)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    # Panel 3: Gradient norm
    ax = axes[2]
    for key in ["C2_full_fra", "C8a_degraded"]:
        t = traces[key]
        ax.plot(t["grad_norm"], color=colors[key], label=labels[key], linewidth=1.0)
    ax.set_ylabel("Gradient Norm\n$\\|G_t^{DR}\\|_F$")
    ax.set_xlabel("Timestep")
    ax.set_title("DR Gradient Magnitude (M13 — A1 Scope)")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(figdir / "fig5_perturbation_dynamics.png")
    plt.close()
    print("fig5_perturbation_dynamics.png")

    # ── Fig 6: Layer-wise perturbation heatmap ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))

    for ax, key, title in [
        (axes[0], "C2_full_fra", "C2 (Full FRA) — Layer Perturbation Over Time"),
        (axes[1], "C8a_degraded", "C8a (Degraded) — Layer Perturbation Over Time"),
    ]:
        t = traces[key]
        lp = np.array(t["layer_perts"])  # [timesteps, n_layers]
        if lp.ndim == 2 and lp.shape[0] > 1:
            im = ax.imshow(lp.T, aspect="auto", cmap="inferno",
                           extent=[0, lp.shape[0], lp.shape[1]-0.5, -0.5])
            ax.set_ylabel("Layer")
            ax.set_xlabel("Timestep")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label="$\\|\\Delta W_{layer}\\|_F$", fraction=0.02)
            ax.set_yticks(range(lp.shape[1]))
            ax.set_yticklabels([f"L{i}" for i in range(lp.shape[1])])

    plt.tight_layout()
    fig.savefig(figdir / "fig6_layer_heatmap.png")
    plt.close()
    print("fig6_layer_heatmap.png")

    # ── Fig 7: Fear-Cost-TTC phase space ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key, title in [
        (axes[0], "C2_full_fra", "C2 (Full FRA)"),
        (axes[1], "C8a_degraded", "C8a (Degraded)"),
    ]:
        t = traces[key]
        sc = ax.scatter(t["ttc"], t["fear"], c=t["cost"], cmap="RdYlGn_r",
                        s=15, alpha=0.7, edgecolors="none")
        ax.set_xlabel("TTC (seconds)")
        ax.set_ylabel("Fear Signal $F_t$")
        ax.set_title(title)
        ax.axvline(x=2.0, color="red", linewidth=0.8, linestyle="--", label="TTC=2s")
        ax.axhline(y=0.5, color="orange", linewidth=0.8, linestyle="--", label="SCL threshold")
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="Cost $c_t$", fraction=0.03)

    plt.suptitle("Fear-TTC-Cost Phase Space", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(figdir / "fig7_phase_space.png")
    plt.close()
    print("fig7_phase_space.png")

    # ── Fig 8: WDN recovery dynamics (zoom) ──
    fig, ax = plt.subplots(figsize=(10, 4))
    t = traces["C2_full_fra"]
    wdn = np.array(t["wdn"])
    fear = np.array(t["fear"])

    ax.plot(wdn, color="#3498db", linewidth=1.5, label="WDN $\\|W_t - W_0\\|_F$")
    ax2 = ax.twinx()
    ax2.fill_between(range(len(fear)), fear, alpha=0.15, color="red")
    ax2.plot(fear, color="#e74c3c", linewidth=0.8, alpha=0.6, label="Fear $F_t$")
    ax2.set_ylabel("Fear Signal", color="#e74c3c")
    ax2.set_ylim(-0.05, 1.05)

    # Mark spike-recovery cycles
    spikes = np.where(np.diff(fear > 0.3, prepend=False))[0]
    for sp in spikes[:10]:
        ax.axvline(x=sp, color="red", linewidth=0.3, alpha=0.3)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Weight Deviation Norm", color="#3498db")
    ax.set_title("FHR Homeostatic Recovery: Fear Spike → Weight Perturbation → Gradual Recovery")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(figdir / "fig8_recovery_dynamics.png")
    plt.close()
    print("fig8_recovery_dynamics.png")

    # ── Fig 9: Gradient descent visualization ──
    fig, ax = plt.subplots(figsize=(8, 6))

    # Show WDN trajectory in 2D projection (first 2 principal components of layer perturbations)
    t = traces["C2_full_fra"]
    lp = np.array(t["layer_perts"])
    if lp.shape[1] >= 2:
        x, y = lp[:, 0], lp[:, 1]
        fear_arr = np.array(t["fear"])

        # Color by fear
        sc = ax.scatter(x, y, c=fear_arr, cmap="RdYlGn_r", s=10, alpha=0.8)
        ax.plot(x, y, color="gray", linewidth=0.3, alpha=0.5)

        # Mark start/end
        ax.scatter([x[0]], [y[0]], color="green", s=100, marker="^", zorder=5, label="Start (W_0)")
        ax.scatter([x[-1]], [y[-1]], color="blue", s=100, marker="s", zorder=5, label="End")

        # Arrow for trajectory direction every N steps
        N = max(1, len(x) // 15)
        for i in range(0, len(x)-N, N):
            dx, dy = x[i+N]-x[i], y[i+N]-y[i]
            if abs(dx) + abs(dy) > 1e-6:
                ax.annotate("", xy=(x[i+N], y[i+N]), xytext=(x[i], y[i]),
                            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8, alpha=0.5))

        plt.colorbar(sc, ax=ax, label="Fear $F_t$", fraction=0.03)
        ax.set_xlabel("Layer 0 Perturbation $\\|\\Delta W_0\\|_F$")
        ax.set_ylabel("Layer 1 Perturbation $\\|\\Delta W_1\\|_F$")
        ax.set_title("Weight Space Trajectory (DR Push + FHR Recovery)")
        ax.legend()

    plt.tight_layout()
    fig.savefig(figdir / "fig9_weight_trajectory.png")
    plt.close()
    print("fig9_weight_trajectory.png")

    print(f"\nAll figures saved to {figdir}/")
    print(f"Total: {len(list(figdir.glob('*.png')))} PNG files")


if __name__ == "__main__":
    main()

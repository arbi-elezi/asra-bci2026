"""
generate_visuals.py — Produce publication-quality figures from FRA experiment results.

Reads actual data from:
  - results/{condition}/summary.json  (per-condition collision rates, CIs, rewards)
  - results/hypothesis_results.json   (hypothesis test outcomes)

Outputs to figures/:
  fig1_collision_rates.png  — Bar chart of CR across all 13 conditions with bootstrap CIs
  fig2_stress_test.png      — Dose-response curve for stress test conditions
  fig3_tradeoff.png         — Reward vs CR scatter (safety-performance tradeoff)
  fig4_hypotheses.png       — Hypothesis results table

Requires: matplotlib, numpy
"""

import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Condition ordering and metadata
# ---------------------------------------------------------------------------
PRIMARY_CONDITIONS = ["C1", "C2", "C3a", "C3b", "C4", "C5", "C6", "C7"]
STRESS_CONDITIONS = ["C8a", "C8b", "C8c", "C8d", "C8e"]
ALL_CONDITIONS = PRIMARY_CONDITIONS + STRESS_CONDITIONS

CONDITION_LABELS = {
    "C1": "C1\nBase Only",
    "C2": "C2\nFull FRA",
    "C3a": "C3a\nL2-HR",
    "C3b": "C3b\nNo BC",
    "C4": "C4\nNo GTCC",
    "C5": "C5\nNo FMS",
    "C6": "C6\nNo DR",
    "C7": "C7\nHard Override",
    "C8a": "C8a\n10% D_ref",
    "C8b": "C8b\n25% D_ref",
    "C8c": "C8c\n50% D_ref",
    "C8d": "C8d\nBias 0.2",
    "C8e": "C8e\nBias 0.5",
}

CONDITION_SHORT = {
    "C1": "C1", "C2": "C2", "C3a": "C3a", "C3b": "C3b",
    "C4": "C4", "C5": "C5", "C6": "C6", "C7": "C7",
    "C8a": "C8a", "C8b": "C8b", "C8c": "C8c", "C8d": "C8d", "C8e": "C8e",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_summary(condition: str) -> dict[str, Any]:
    """Load summary.json for a single condition."""
    path = RESULTS_DIR / condition / "summary.json"
    with open(path, "r") as f:
        return json.load(f)


def load_all_summaries() -> dict[str, dict[str, Any]]:
    """Load all condition summaries into a dict keyed by condition name."""
    summaries: dict[str, dict[str, Any]] = {}
    for cond in ALL_CONDITIONS:
        summaries[cond] = load_summary(cond)
    return summaries


def load_hypothesis_results() -> dict[str, Any]:
    """Load hypothesis test results."""
    path = RESULTS_DIR / "hypothesis_results.json"
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
def set_academic_style() -> None:
    """Configure matplotlib for clean academic figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 11,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def bar_color(condition: str) -> str:
    """Return color by condition category."""
    if condition == "C1":
        return "#4CAF50"  # green — baseline control
    elif condition.startswith("C8"):
        return "#E53935"  # red — stress tests
    else:
        return "#1E88E5"  # blue — FRA variants


# ---------------------------------------------------------------------------
# Figure 1: Collision Rate Bar Chart — All 13 Conditions
# ---------------------------------------------------------------------------
def fig1_collision_rates(summaries: dict[str, dict[str, Any]]) -> None:
    """Bar chart of collision rates with bootstrap CI error bars."""
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(ALL_CONDITIONS))
    crs = []
    ci_lowers = []
    ci_uppers = []
    colors = []

    for cond in ALL_CONDITIONS:
        s = summaries[cond]
        cr = s["M1_collision_rate"]
        ci_lo = s["M1_ci"]["ci_lower"]
        ci_hi = s["M1_ci"]["ci_upper"]
        crs.append(cr)
        ci_lowers.append(cr - ci_lo)
        ci_uppers.append(ci_hi - cr)
        colors.append(bar_color(cond))

    yerr = np.array([ci_lowers, ci_uppers])

    bars = ax.bar(x, crs, width=0.65, color=colors, edgecolor="black",
                  linewidth=0.5, zorder=3)
    ax.errorbar(x, crs, yerr=yerr, fmt="none", ecolor="black",
                elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in ALL_CONDITIONS],
                       fontsize=8, ha="center")
    ax.set_ylabel("Collision Rate (M1)", fontsize=12)
    ax.set_title("Collision Rates Across All Experimental Conditions",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

    # Add value labels above bars
    for i, (bar_obj, cr_val) in enumerate(zip(bars, crs)):
        label_y = cr_val + ci_uppers[i] + 0.02
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, label_y,
                f"{cr_val:.3f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold")

    # Add a vertical separator between primary and stress conditions
    sep_x = len(PRIMARY_CONDITIONS) - 0.5
    ax.axvline(sep_x, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(sep_x - 0.15, 1.08, "Primary", ha="right", fontsize=8,
            color="gray", fontstyle="italic")
    ax.text(sep_x + 0.15, 1.08, "Stress Tests", ha="left", fontsize=8,
            color="gray", fontstyle="italic")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", edgecolor="black", linewidth=0.5,
              label="Baseline (C1)"),
        Patch(facecolor="#1E88E5", edgecolor="black", linewidth=0.5,
              label="FRA Variants (C2--C7)"),
        Patch(facecolor="#E53935", edgecolor="black", linewidth=0.5,
              label="Stress Tests (C8a--C8e)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig1_collision_rates.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Stress Test Dose-Response
# ---------------------------------------------------------------------------
def fig2_stress_test(summaries: dict[str, dict[str, Any]]) -> None:
    """Dose-response plot showing sharp threshold between C8b and C8c."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Separate degraded (data scarcity) from biased conditions
    degraded = ["C8a", "C8b", "C8c"]
    biased = ["C8d", "C8e"]

    # X positions for degraded conditions
    deg_x = [10, 25, 50]
    deg_cr = [summaries[c]["M1_collision_rate"] for c in degraded]
    deg_lo = [summaries[c]["M1_collision_rate"] - summaries[c]["M1_ci"]["ci_lower"]
              for c in degraded]
    deg_hi = [summaries[c]["M1_ci"]["ci_upper"] - summaries[c]["M1_collision_rate"]
              for c in degraded]

    ax.errorbar(deg_x, deg_cr, yerr=[deg_lo, deg_hi],
                fmt="o-", color="#E53935", markersize=8, linewidth=2,
                capsize=5, capthick=1.5, label="Degraded D_ref", zorder=5)

    # Annotate each degraded point
    for i, (xv, yv, cond) in enumerate(zip(deg_x, deg_cr, degraded)):
        offset_y = 0.04 if yv < 0.9 else -0.06
        ax.annotate(f"{cond}\nCR={yv:.3f}",
                    xy=(xv, yv), xytext=(xv, yv + offset_y),
                    fontsize=9, ha="center", va="bottom" if offset_y > 0 else "top",
                    fontweight="bold")

    # Highlight the sharp threshold with a shaded region
    ax.axvspan(25, 50, alpha=0.08, color="red", zorder=1)
    ax.annotate("Sharp threshold\n(non-monotonic)",
                xy=(37.5, 0.75), fontsize=9, ha="center", color="#B71C1C",
                fontstyle="italic")

    # Baseline reference line
    c1_cr = summaries["C1"]["M1_collision_rate"]
    ax.axhline(c1_cr, color="#4CAF50", linewidth=1.2, linestyle="--",
               alpha=0.7, zorder=2)
    ax.text(52, c1_cr + 0.02, f"C1 Baseline ({c1_cr:.3f})",
            fontsize=9, color="#4CAF50", va="bottom")

    # Secondary x-axis region for bias conditions
    # Plot them at x=65 and x=75 (visual separation)
    bias_x = [65, 75]
    bias_cr = [summaries[c]["M1_collision_rate"] for c in biased]
    bias_lo = [summaries[c]["M1_collision_rate"] - summaries[c]["M1_ci"]["ci_lower"]
               for c in biased]
    bias_hi = [summaries[c]["M1_ci"]["ci_upper"] - summaries[c]["M1_collision_rate"]
               for c in biased]

    ax.errorbar(bias_x, bias_cr, yerr=[bias_lo, bias_hi],
                fmt="s-", color="#FF6F00", markersize=8, linewidth=2,
                capsize=5, capthick=1.5, label="Biased Cost Labels", zorder=5)

    for xv, yv, cond in zip(bias_x, bias_cr, biased):
        offset_y = -0.06 if yv > 0.9 else 0.04
        ax.annotate(f"{cond}\nCR={yv:.3f}",
                    xy=(xv, yv), xytext=(xv, yv + offset_y),
                    fontsize=9, ha="center", va="top" if offset_y < 0 else "bottom",
                    fontweight="bold")

    # Vertical separator between degraded and biased
    ax.axvline(57, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.text(35, -0.09, "Data Scarcity (% of D_ref)", ha="center",
            fontsize=10, transform=ax.get_xaxis_transform())
    ax.text(70, -0.09, "Cost Bias", ha="center",
            fontsize=10, transform=ax.get_xaxis_transform())

    ax.set_ylabel("Collision Rate (M1)", fontsize=12)
    ax.set_title("Stress Test: Cost Critic Degradation and Bias",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(0, 85)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

    # Custom x ticks
    ax.set_xticks([10, 25, 50, 65, 75])
    ax.set_xticklabels(["10%", "25%", "50%", "Bias\n0.2", "Bias\n0.5"],
                       fontsize=9)

    ax.legend(loc="center left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig2_stress_test.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Reward vs CR Scatter (Safety-Performance Tradeoff)
# ---------------------------------------------------------------------------
def fig3_tradeoff(summaries: dict[str, dict[str, Any]]) -> None:
    """Scatter plot of mean reward vs collision rate per condition."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for cond in ALL_CONDITIONS:
        s = summaries[cond]
        cr = s["M1_collision_rate"]
        reward = s["M8_mean_reward"]
        color = bar_color(cond)

        marker = "o"
        size = 80
        if cond == "C1":
            marker = "D"
            size = 100
        elif cond.startswith("C8"):
            marker = "s"
            size = 80

        ax.scatter(cr, reward, c=color, s=size, marker=marker,
                   edgecolors="black", linewidths=0.6, zorder=5)

        # Label each point
        # Offset labels to avoid overlap
        offset_x = 0.015
        offset_y = 1.5
        ha = "left"

        # Adjust offsets for crowded areas
        if cond in ("C3a", "C3b", "C5"):
            offset_y = -2.5
        elif cond in ("C6", "C7", "C8b"):
            offset_x = -0.015
            ha = "right"
        elif cond in ("C8a", "C8c", "C8d"):
            offset_x = -0.015
            ha = "right"
            if cond == "C8a":
                offset_y = 2.0
            elif cond == "C8d":
                offset_y = -2.0

        ax.annotate(CONDITION_SHORT[cond],
                    xy=(cr, reward),
                    xytext=(cr + offset_x, reward + offset_y),
                    fontsize=8, fontweight="bold", ha=ha, va="center",
                    color=color,
                    arrowprops=dict(arrowstyle="-", color="gray",
                                    lw=0.4, alpha=0.5) if abs(offset_x) > 0.01 else None)

    ax.set_xlabel("Collision Rate (M1)", fontsize=12)
    ax.set_ylabel("Mean Episode Reward (M8)", fontsize=12)
    ax.set_title("Safety-Performance Tradeoff Across Conditions",
                 fontsize=13, fontweight="bold", pad=12)

    ax.set_xlim(0.45, 1.08)
    ax.set_ylim(15, 115)

    # Ideal corner annotation
    ax.annotate("Ideal\n(low CR, high reward)",
                xy=(0.48, 110), fontsize=8, color="gray", fontstyle="italic",
                ha="center")
    ax.annotate("Worst\n(high CR, low reward)",
                xy=(1.02, 22), fontsize=8, color="gray", fontstyle="italic",
                ha="center")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#4CAF50",
               markeredgecolor="black", markersize=8, label="Baseline (C1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1E88E5",
               markeredgecolor="black", markersize=8, label="FRA Variants"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#E53935",
               markeredgecolor="black", markersize=8, label="Stress Tests"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig3_tradeoff.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Hypothesis Results Table
# ---------------------------------------------------------------------------
def fig4_hypotheses(hyp_results: dict[str, Any]) -> None:
    """Formatted table of hypothesis results."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    # Hypothesis descriptions
    hyp_descriptions = {
        "H1": "FRA reduces CR: CR(C2) < CR(C1)",
        "H8": "DR contributes on adversarial seeds",
        "H5a": "Fisher+BC reduces KL vs L2",
        "H15a": "Degraded 10% worse than baseline",
        "H15b": "Degraded 25% worse than baseline",
        "H15c": "Degraded 50% worse than baseline",
        "H16": "Severe bias (0.2) worse than baseline",
        "H17": "Moderate bias (0.5) worse than baseline",
    }

    # Column headers
    columns = ["Hypothesis", "Claim", "Delta", "95% CI", "Excludes 0?", "Result"]

    # Build table data in the order from the JSON
    # Sort by hypothesis name for consistent display
    hyp_order = ["H1", "H5a", "H8", "H15a", "H15b", "H15c", "H16", "H17"]
    table_data = []

    for h_key in hyp_order:
        if h_key not in hyp_results:
            continue
        h = hyp_results[h_key]
        delta = h["delta"]
        ci = h["ci"]
        excludes = h["excludes_zero"]
        confirmed = h.get("confirmed", None)

        if confirmed is True:
            result_str = "CONFIRMED"
        elif confirmed is False:
            result_str = "FALSIFIED"
        else:
            # H5a doesn't have 'confirmed' key — CI includes zero
            result_str = "FALSIFIED" if not excludes else "CONFIRMED"

        table_data.append([
            h_key,
            hyp_descriptions.get(h_key, h["claim"]),
            f"{delta:+.3f}",
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
            "Yes" if excludes else "No",
            result_str,
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header row
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_edgecolor("white")

    # Style data rows
    for i in range(len(table_data)):
        row_idx = i + 1  # +1 for header
        result = table_data[i][-1]

        for j in range(len(columns)):
            cell = table[row_idx, j]
            cell.set_edgecolor("#E0E0E0")

            # Alternate row shading
            if i % 2 == 0:
                cell.set_facecolor("#FAFAFA")
            else:
                cell.set_facecolor("#FFFFFF")

        # Color the result cell
        result_cell = table[row_idx, len(columns) - 1]
        if result == "CONFIRMED":
            result_cell.set_text_props(color="#2E7D32", fontweight="bold")
        else:
            result_cell.set_text_props(color="#C62828", fontweight="bold")

        # Color the excludes-zero cell
        excl_cell = table[row_idx, len(columns) - 2]
        if table_data[i][-2] == "Yes":
            excl_cell.set_text_props(color="#2E7D32")
        else:
            excl_cell.set_text_props(color="#C62828")

    # Adjust column widths
    col_widths = [0.06, 0.30, 0.08, 0.16, 0.10, 0.10]
    for j, w in enumerate(col_widths):
        for i in range(len(table_data) + 1):
            table[i, j].set_width(w)

    ax.set_title("Hypothesis Test Results (Bootstrap 95% CI, n=200 seeds)",
                 fontsize=13, fontweight="bold", pad=20, y=0.98)

    # Summary counts
    n_confirmed = sum(1 for row in table_data if row[-1] == "CONFIRMED")
    n_falsified = sum(1 for row in table_data if row[-1] == "FALSIFIED")
    fig.text(0.5, 0.02,
             f"Summary: {n_confirmed} confirmed, {n_falsified} falsified out of "
             f"{len(table_data)} tested hypotheses",
             ha="center", fontsize=10, fontstyle="italic", color="#616161")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = FIGURES_DIR / "fig4_hypotheses.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_academic_style()
    print("Loading experiment data...")
    summaries = load_all_summaries()
    hyp_results = load_hypothesis_results()

    print(f"Loaded {len(summaries)} condition summaries and "
          f"{len(hyp_results)} hypothesis results.\n")

    print("Generating figures:")

    print("  [1/4] Collision rate bar chart...")
    fig1_collision_rates(summaries)

    print("  [2/4] Stress test dose-response...")
    fig2_stress_test(summaries)

    print("  [3/4] Reward vs CR scatter...")
    fig3_tradeoff(summaries)

    print("  [4/4] Hypothesis results table...")
    fig4_hypotheses(hyp_results)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

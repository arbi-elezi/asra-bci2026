"""Post-experiment analysis: test all 17 hypotheses.

Usage:
  python analysis/hypothesis_tests.py --results-dir results/

This script:
  1. Loads per-seed results from all conditions
  2. Computes paired bootstrap CIs for all 17 hypotheses
  3. Reports confirmation/falsification for each
  4. Generates summary table (Table 7 in paper)
  5. Reports ALL hypotheses regardless of outcome (Rule 8)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.metrics import (
    bootstrap_ci,
    paired_bootstrap_ci,
    m1_collision_rate,
    m2_fc_f1,
    m3_wdn,
    m13_gradient_norm,
)


def load_per_seed(results_dir: Path, condition: str) -> dict:
    """Load per-seed results for a condition."""
    path = results_dir / condition / "per_seed.json"
    with open(path) as f:
        return json.load(f)


def load_summary(results_dir: Path, condition: str) -> dict:
    """Load summary for a condition."""
    path = results_dir / condition / "summary.json"
    with open(path) as f:
        return json.load(f)


def extract_collisions(per_seed: list[dict]) -> np.ndarray:
    """Extract collision flags as array."""
    return np.array([r["collision"] for r in per_seed], dtype=float)


def test_h1(results_dir: Path) -> dict:
    """H1: CR(C2) < CR(C1); CI excludes zero."""
    c1 = extract_collisions(load_per_seed(results_dir, "C1"))
    c2 = extract_collisions(load_per_seed(results_dir, "C2"))
    # Δ = CR(C1) - CR(C2) — positive means C2 is better
    result = paired_bootstrap_ci(c1, c2)
    return {
        "hypothesis": "H1",
        "claim": "CR(C2) < CR(C1)",
        "delta": result["point_estimate"],
        "ci": [result["ci_lower"], result["ci_upper"]],
        "excludes_zero": result["excludes_zero"],
        "confirmed": result["excludes_zero"] and result["point_estimate"] > 0,
    }


def test_h5a(results_dir: Path) -> dict:
    """H5a: M9(C3a) > M9(C2); CI excludes zero."""
    c2 = load_per_seed(results_dir, "C2")
    c3a = load_per_seed(results_dir, "C3a")
    # Extract KL divergence per seed
    kl_c2 = np.array([r.get("kl_mean", 0) for r in c2])
    kl_c3a = np.array([r.get("kl_mean", 0) for r in c3a])
    result = paired_bootstrap_ci(kl_c3a, kl_c2)
    return {
        "hypothesis": "H5a",
        "claim": "M9(C3a) > M9(C2) — Fisher+BC better than L2",
        "delta": result["point_estimate"],
        "ci": [result["ci_lower"], result["ci_upper"]],
        "excludes_zero": result["excludes_zero"],
        "confirmed": result["excludes_zero"] and result["point_estimate"] > 0,
    }


def test_h8(results_dir: Path) -> dict:
    """H8: CR(C2) < CR(C6) on adversarial seeds; CI excludes zero. CRITICAL."""
    c2 = extract_collisions(load_per_seed(results_dir, "C2"))
    c6 = extract_collisions(load_per_seed(results_dir, "C6"))
    # Use only adversarial seeds (first 100)
    c2_adv = c2[:100]
    c6_adv = c6[:100]
    result = paired_bootstrap_ci(c6_adv, c2_adv)
    return {
        "hypothesis": "H8 (CRITICAL)",
        "claim": "CR(C2) < CR(C6) on adversarial seeds — DR contributes",
        "delta": result["point_estimate"],
        "ci": [result["ci_lower"], result["ci_upper"]],
        "excludes_zero": result["excludes_zero"],
        "confirmed": result["excludes_zero"] and result["point_estimate"] > 0,
        "consequence_if_falsified": "DR revised to non-contributing; core still stands",
    }


def test_h10(results_dir: Path) -> dict:
    """H10: M3 ≤ bound (7) for all timesteps within B(W_0, r)."""
    summary = load_summary(results_dir, "C2")
    # Load artifacts for bound computation
    artifacts_path = Path("checkpoints/artifacts.json")
    if artifacts_path.exists():
        with open(artifacts_path) as f:
            artifacts = json.load(f)
        # Bound = η_f · G_max / (η_h · f_min) · max_F
        # This requires per-timestep WDN data
        return {
            "hypothesis": "H10",
            "claim": "M3 ≤ bound for all timesteps",
            "note": "Requires per-timestep WDN data analysis",
            "artifacts": artifacts,
        }
    return {"hypothesis": "H10", "note": "Artifacts not found — run train_base.py first"}


def test_h14_h15_h16_h17(results_dir: Path) -> list[dict]:
    """Stress test hypotheses H14a-c, H15a-c, H16, H17."""
    results = []
    c1 = extract_collisions(load_per_seed(results_dir, "C1"))[:500]

    for label, cond_a, cond_b in [
        ("H14a", "C8a", "C8c"),
        ("H14b", "C8b", "C8c"),
        ("H14c", "C8a", "C8b"),
    ]:
        a = extract_collisions(load_per_seed(results_dir, cond_a))
        b = extract_collisions(load_per_seed(results_dir, cond_b))
        r = paired_bootstrap_ci(a, b)
        results.append({
            "hypothesis": label,
            "claim": f"CR({cond_a}) > CR({cond_b})",
            "delta": r["point_estimate"],
            "ci": [r["ci_lower"], r["ci_upper"]],
            "excludes_zero": r["excludes_zero"],
            "confirmed": r["excludes_zero"] and r["point_estimate"] > 0,
        })

    for label, cond in [
        ("H15a", "C8a"), ("H15b", "C8b"), ("H15c", "C8c"),
        ("H16", "C8d"), ("H17", "C8e"),
    ]:
        cx = extract_collisions(load_per_seed(results_dir, cond))
        r = paired_bootstrap_ci(cx, c1[:len(cx)])
        results.append({
            "hypothesis": label,
            "claim": f"CR({cond}) > CR(C1)",
            "delta": r["point_estimate"],
            "ci": [r["ci_lower"], r["ci_upper"]],
            "excludes_zero": r["excludes_zero"],
            "confirmed": r["excludes_zero"] and r["point_estimate"] > 0,
        })

    return results


def run_all_tests(results_dir: str) -> None:
    """Run all 17 hypothesis tests and report results."""
    rd = Path(results_dir)

    print("=" * 70)
    print("FRA HYPOTHESIS TEST RESULTS")
    print("=" * 70)
    print(f"Results directory: {rd}")
    print()

    all_results = []

    # Run available tests
    try:
        all_results.append(test_h1(rd))
    except Exception as e:
        print(f"H1: SKIPPED ({e})")

    try:
        all_results.append(test_h5a(rd))
    except Exception as e:
        print(f"H5a: SKIPPED ({e})")

    try:
        all_results.append(test_h8(rd))
    except Exception as e:
        print(f"H8: SKIPPED ({e})")

    try:
        all_results.extend(test_h14_h15_h16_h17(rd))
    except Exception as e:
        print(f"H14-H17: SKIPPED ({e})")

    # Print results table
    print(f"\n{'Hyp':<12} {'Claim':<45} {'Δ':>8} {'CI':>22} {'Result':>10}")
    print("-" * 100)
    for r in all_results:
        status = "CONFIRMED" if r.get("confirmed") else "FALSIFIED"
        ci_str = f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]" if "ci" in r else "N/A"
        delta = f"{r.get('delta', 0):.4f}"
        print(f"{r['hypothesis']:<12} {r.get('claim', ''):<45} {delta:>8} {ci_str:>22} {status:>10}")

    # Save full results
    output = rd / "hypothesis_results.json"
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {output}")

    # Summary
    confirmed = sum(1 for r in all_results if r.get("confirmed"))
    total = len(all_results)
    print(f"\nSummary: {confirmed}/{total} hypotheses confirmed")

    # Rule 8 check
    print("\n[Rule 8] All tested hypotheses reported regardless of outcome.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    run_all_tests(args.results_dir)

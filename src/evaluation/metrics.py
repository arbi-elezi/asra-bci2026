"""Metrics M1–M14 and Bootstrap CI computation.

All hypothesis tests use bootstrap CIs per scientific_method.md.
Bootstrap resampling uses a SEPARATE seeded RNG from experiment RNG (coding rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


# ── Bootstrap CI ──────────────────────────────────────────────────────────


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 99999,  # Separate from experiment seeds
) -> dict[str, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: 1D array of per-seed differences (Δ_i).
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level.
        seed: RNG seed (SEPARATE from experiment seeds).

    Returns:
        Dict with point_estimate, ci_lower, ci_upper, excludes_zero.
    """
    rng = np.random.default_rng(seed)  # Separate RNG per coding rule
    n = len(data)
    means = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        means[b] = sample.mean()

    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)

    return {
        "point_estimate": float(data.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "excludes_zero": bool(lower > 0 or upper < 0),
        "n_samples": n,
        "n_bootstrap": n_bootstrap,
    }


def paired_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 99999,
) -> dict[str, float]:
    """Paired bootstrap CI for matched-seed comparisons.

    Δ_i = x_i - y_i per seed i. CI on mean(Δ).
    """
    assert len(x) == len(y), "Paired comparison requires same number of seeds"
    delta = x - y
    return bootstrap_ci(delta, n_bootstrap, ci, seed)


# ── Individual Metrics ────────────────────────────────────────────────────


def m1_collision_rate(episode_collisions: np.ndarray) -> float:
    """M1: Collision episodes / total episodes."""
    return float(episode_collisions.mean())


def m2_fc_f1(f1_history: list[float]) -> dict[str, float]:
    """M2: FC F1 score and slope."""
    if len(f1_history) < 2:
        return {"f1_final": 0.0, "f1_slope": 0.0}

    x = np.arange(len(f1_history))
    slope, intercept, _, _, _ = stats.linregress(x, f1_history)
    return {
        "f1_final": f1_history[-1],
        "f1_slope": float(slope),
    }


def m3_wdn(wdn_per_timestep: np.ndarray) -> dict[str, float]:
    """M3: Weight Deviation Norm statistics.

    Must be computed in float64 (coding rule — enforced in FHR).
    """
    return {
        "wdn_mean": float(wdn_per_timestep.mean()),
        "wdn_max": float(wdn_per_timestep.max()),
        "wdn_final": float(wdn_per_timestep[-1]) if len(wdn_per_timestep) > 0 else 0.0,
    }


def m4_recovery_time(
    wdn_per_timestep: np.ndarray,
    threshold_fraction: float = 0.05,
) -> list[int]:
    """M4: Steps for WDN to drop below 5% of peak.

    Returns list of recovery times per threat encounter.
    """
    peak = wdn_per_timestep.max()
    if peak == 0:
        return []

    threshold = peak * threshold_fraction
    recovery_times = []

    # Find peaks and measure recovery
    in_peak = False
    peak_step = 0

    for t in range(len(wdn_per_timestep)):
        if wdn_per_timestep[t] > peak * 0.5 and not in_peak:
            in_peak = True
            peak_step = t
        elif in_peak and wdn_per_timestep[t] < threshold:
            recovery_times.append(t - peak_step)
            in_peak = False

    return recovery_times


def m5_orhc(
    alpha_history: np.ndarray,
    w0_confidence: np.ndarray,
    conf_threshold: float = 0.99,
) -> int:
    """M5: Override-Reduction Hit Count.

    Count timesteps where α_t > 0.5 AND π_{W_0} confidence > 0.99.
    """
    return int(((alpha_history > 0.5) & (w0_confidence > conf_threshold)).sum())


def m6_fpr(
    fear_history: np.ndarray,
    ttc_history: np.ndarray,
    fear_threshold: float = 0.3,
    ttc_safe: float = 8.0,
) -> float:
    """M6: False Positive Rate — F_t > 0.3 when TTC > 8s."""
    safe_mask = ttc_history > ttc_safe
    if safe_mask.sum() == 0:
        return 0.0
    false_positives = (fear_history[safe_mask] > fear_threshold).sum()
    return float(false_positives / safe_mask.sum())


def m7_rlaf_reliability(rlaf_correct: int, rlaf_total: int) -> float:
    """M7: RLAF accuracy vs GTCC. Gate threshold: 0.70."""
    if rlaf_total == 0:
        return 0.0
    return rlaf_correct / rlaf_total


def m8_task_reward(episode_rewards: np.ndarray) -> float:
    """M8: Mean episode reward."""
    return float(episode_rewards.mean())


def m9_kl_divergence(kl_per_timestep: np.ndarray) -> dict[str, float]:
    """M9: D_KL(π_{W_0} || π_{W_t}). METRIC ONLY, NOT A BOUND."""
    return {
        "kl_mean": float(kl_per_timestep.mean()),
        "kl_max": float(kl_per_timestep.max()),
        "kl_final": float(kl_per_timestep[-1]) if len(kl_per_timestep) > 0 else 0.0,
    }


def m10_cost_critic_error(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> float:
    """M10: ||V̂^C - V^C_MC|| on holdout."""
    return float(np.sqrt(((predicted - actual) ** 2).mean()))


def m10s_stratified(
    predicted: np.ndarray,
    actual: np.ndarray,
    obstacle_classes: np.ndarray,
) -> dict[int, float]:
    """M10-s: Per-class cost critic error."""
    results = {}
    for cls in np.unique(obstacle_classes):
        mask = obstacle_classes == cls
        if mask.sum() > 0:
            results[int(cls)] = float(np.sqrt(((predicted[mask] - actual[mask]) ** 2).mean()))
    return results


def m11_fear_ranking(
    f_ca: np.ndarray,
    future_costs: np.ndarray,
) -> float:
    """M11: Spearman ρ between F_t^CA and actual future cost."""
    if len(f_ca) < 3:
        return 0.0
    rho, _ = stats.spearmanr(f_ca, future_costs)
    return float(rho) if not np.isnan(rho) else 0.0


def m12_fms_correction(delta_history: np.ndarray) -> dict[str, float]:
    """M12: FMS correction |δ_t| distribution."""
    abs_delta = np.abs(delta_history)
    return {
        "delta_abs_mean": float(abs_delta.mean()),
        "delta_abs_max": float(abs_delta.max()),
        "delta_abs_std": float(abs_delta.std()),
    }


def m13_gradient_norm(
    grad_norms: np.ndarray,
    g_max: float,
) -> dict[str, float]:
    """M13: A1 scope verification.

    Reports fraction of timesteps where ||G_t^DR||_F > G_max.
    """
    violation_mask = grad_norms > g_max
    return {
        "violation_fraction": float(violation_mask.mean()),
        "violation_count": int(violation_mask.sum()),
        "grad_norm_max": float(grad_norms.max()),
        "grad_norm_mean": float(grad_norms.mean()),
    }


def m14_degradation(
    m10_degraded: float,
    m10_full: float,
) -> float:
    """M14: C8 degradation = M10(degraded) - M10(full)."""
    return m10_degraded - m10_full

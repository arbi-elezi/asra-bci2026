"""FRA Experiment Runner — Executes conditions C1–C8e.

Usage:
  python run_experiment.py --config configs/c2_full_fra.yaml --seeds 0 1000
  python run_experiment.py --config configs/c1_baseline.yaml --seeds 0 100 --dry-run

This script:
  1. Loads config YAML
  2. Initializes environment (highway-env wrapper)
  3. Initializes LLM driver (TinyLlama + LoRA) or loads checkpoint
  4. Initializes FRA engine with all components
  5. Runs episodes with matched seeds
  6. Logs M1–M14 per seed
  7. Saves results, config copy, W_0 hash

Every run is reproducible: seed + config → identical results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from src.environment.highway_wrapper import HighwayDrivingEnv
from src.agents.llm_driver import LLMDriver, LLMDriverConfig
from src.agents.fra_engine import FRAEngine, FRAEngineConfig
from src.agents.cost_critic import CostCriticNet, train_cost_critic
from src.components.fear_detector import FearDetectorConfig
from src.components.td_fear import TDFearConfig
from src.components.scl import SCLConfig
from src.components.gtcc import GTCCConfig
from src.components.fc import FCConfig
from src.components.fms import FMSConfig
from src.evaluation.metrics import (
    m1_collision_rate,
    m3_wdn,
    m4_recovery_time,
    m6_fpr,
    m8_task_reward,
    m13_gradient_norm,
    bootstrap_ci,
    paired_bootstrap_ci,
)


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_fra_config(cfg: dict) -> FRAEngineConfig:
    """Build FRAEngineConfig from YAML dict."""
    fra = cfg.get("fra", {})
    if not fra.get("enabled", False):
        return FRAEngineConfig(enabled=False)

    components = fra.get("components", {})
    dr = components.get("dr", {})
    fhr = components.get("fhr", {})
    fc = components.get("fc", {})
    td = components.get("td_fear", {})
    scl = components.get("scl", {})
    fms_cfg = components.get("fms", {})
    gtcc = components.get("gtcc", {})

    return FRAEngineConfig(
        enabled=True,
        eta_f=dr.get("eta_f", 0.01),
        dr_enabled=dr.get("enabled", True),
        eta_h=fhr.get("eta_h", 1e-5),
        fhr_mode=fhr.get("type", "fisher"),
        bc_enabled=fhr.get("bc_enabled", True),
        eta_bc=fhr.get("eta_bc", 1e-5),
        fhr_enabled=fhr.get("enabled", True),
        fc_enabled=fc.get("enabled", True),
        fc_frozen=fc.get("frozen", False),
        td_fear_enabled=td.get("enabled", True),
        fms_enabled=fms_cfg.get("enabled", True),
        gtcc_enabled=gtcc.get("enabled", True),
        scl_enabled=scl.get("enabled", True),
        td_fear=TDFearConfig(gamma=td.get("gamma_f", 0.5)),
        scl=SCLConfig(
            tau=scl.get("tau", 0.5),
            k=scl.get("k", 10.0),
        ),
    )


def run_single_episode(
    env: HighwayDrivingEnv,
    llm: LLMDriver,
    fra_engine: FRAEngine | None,
    seed: int,
    condition: str,
) -> dict:
    """Run a single episode and collect metrics.

    Returns dict with per-timestep and episode-level metrics.
    """
    obs, info = env.reset(seed=seed)

    episode_reward = 0.0
    episode_collision = False
    step_data = []

    for t in range(env.max_steps):
        cost = info.get("cost", 0.0)
        ttc = info.get("ttc", 10.0)
        obstacle_class = info.get("nearest_class", -1)

        if fra_engine is not None:
            action, step_info = fra_engine.step(
                obs, cost, ttc, obstacle_class,
                state_text_fn=env.get_state_text,
            )
            step_data.append(step_info)
        else:
            # C1 baseline or C7 hard override
            action, _ = llm.get_action(env.get_state_text(obs))

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated:
            episode_collision = info.get("collision", True)
            break
        if truncated:
            break

    # Collect episode metrics
    result = {
        "seed": seed,
        "condition": condition,
        "collision": episode_collision,
        "reward": episode_reward,
        "n_steps": len(step_data) if step_data else t + 1,
    }

    # Per-timestep metrics for C2 (M3, M13 — Rule 6)
    if step_data:
        wdn_arr = np.array([s["wdn"] for s in step_data])
        grad_arr = np.array([s["grad_norm"] for s in step_data])
        fear_arr = np.array([s["fear_final"] for s in step_data])
        ttc_arr = np.array([s.get("ttc", 10.0) for s in step_data])

        result["wdn"] = m3_wdn(wdn_arr)
        result["recovery_times"] = m4_recovery_time(wdn_arr)
        result["grad_norm"] = m13_gradient_norm(grad_arr, fra_engine.cfg.g_max if fra_engine else 10.0)
        result["fpr"] = m6_fpr(fear_arr, ttc_arr)

        # Raw per-timestep (for Proposition 1 validation)
        result["wdn_per_step"] = wdn_arr.tolist()
        result["grad_per_step"] = grad_arr.tolist()

    return result


def run_condition(
    config_path: str,
    seed_start: int = 0,
    seed_end: int = 1000,
    dry_run: bool = False,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """Run a full experimental condition."""
    cfg = load_config(config_path)
    condition = cfg["condition"]
    save_dir = Path(cfg.get("logging", {}).get("save_dir", f"results/{condition}"))
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Running {condition}: {cfg.get('name', '')} ===")
    print(f"Seeds: {seed_start} to {seed_end}")
    print(f"Save dir: {save_dir}")

    if dry_run:
        print("DRY RUN — validating config only")
        print(f"Config valid: {condition}")
        print(f"FRA enabled: {cfg.get('fra', {}).get('enabled', False)}")
        return

    # ── 1. Initialize environment ──
    env = HighwayDrivingEnv(seed=seed_start)
    print(f"Environment: R^{env.observation_space.shape[0]}, |A|={env.action_space.n}")

    # ── 2. Initialize LLM ──
    device = cfg.get("gpu", {}).get("device", "cuda")
    llm_config = LLMDriverConfig(device=device)
    llm = LLMDriver(llm_config)
    print(f"LLM params: {llm.count_perturbable_params():,}")

    # ── 3. Initialize FRA engine ──
    fra_cfg = build_fra_config(cfg)
    fra_engine = None
    if fra_cfg.enabled:
        fra_engine = FRAEngine(llm, fra_cfg)
        print(f"FRA engine: DR={fra_cfg.dr_enabled}, FHR={fra_cfg.fhr_mode}, BC={fra_cfg.bc_enabled}")
    else:
        print("FRA disabled — running baseline")

    # ── 4. Save config copy and W_0 hash ──
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)
    w0_hash = llm.get_w0_hash()
    with open(save_dir / "w0_hash.txt", "w") as f:
        f.write(w0_hash)
    print(f"W_0 hash: {w0_hash[:16]}...")

    # ── 5. Run episodes ──
    all_results = []
    collisions = []
    rewards = []

    t_start = time.time()
    for seed in range(seed_start, seed_end):
        if fra_engine:
            fra_engine.reset()

        result = run_single_episode(env, llm, fra_engine, seed, condition)
        all_results.append(result)
        collisions.append(result["collision"])
        rewards.append(result["reward"])

        # Progress reporting every 100 seeds
        if (seed - seed_start + 1) % 100 == 0:
            cr_so_far = np.mean(collisions)
            elapsed = time.time() - t_start
            rate = (seed - seed_start + 1) / elapsed
            print(f"  [{seed - seed_start + 1}/{seed_end - seed_start}] "
                  f"CR={cr_so_far:.3f} | {rate:.1f} seeds/s")

        # Verify W_0 intact every 100 seeds
        if (seed - seed_start + 1) % 100 == 0:
            assert llm.verify_w0_intact(), f"W_0 CORRUPTED at seed {seed}!"

    elapsed = time.time() - t_start

    # ── 6. Compute summary metrics ──
    collisions_arr = np.array(collisions, dtype=float)
    rewards_arr = np.array(rewards)

    summary = {
        "condition": condition,
        "n_seeds": seed_end - seed_start,
        "seed_range": [seed_start, seed_end],
        "M1_collision_rate": m1_collision_rate(collisions_arr),
        "M8_mean_reward": m8_task_reward(rewards_arr),
        "w0_hash": w0_hash,
        "elapsed_seconds": elapsed,
        "seeds_per_second": (seed_end - seed_start) / elapsed,
    }

    # Bootstrap CI for CR
    cr_ci = bootstrap_ci(collisions_arr)
    summary["M1_ci"] = cr_ci

    print(f"\n=== {condition} Complete ===")
    print(f"CR = {summary['M1_collision_rate']:.4f} "
          f"[{cr_ci['ci_lower']:.4f}, {cr_ci['ci_upper']:.4f}]")
    print(f"Mean reward = {summary['M8_mean_reward']:.2f}")
    print(f"Time: {elapsed:.1f}s ({summary['seeds_per_second']:.1f} seeds/s)")

    # ── 7. Save results ──
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save per-seed results (for paired bootstrap across conditions)
    with open(save_dir / "per_seed.json", "w") as f:
        # Convert numpy types for JSON serialization
        serializable = []
        for r in all_results:
            sr = {k: v for k, v in r.items() if k not in ("wdn_per_step", "grad_per_step")}
            serializable.append(sr)
        json.dump(serializable, f, indent=2, default=str)

    # Save per-timestep data for C2 (Proposition 1 validation)
    if condition == "C2":
        timestep_dir = save_dir / "timestep_data"
        timestep_dir.mkdir(exist_ok=True)
        for r in all_results:
            if "wdn_per_step" in r:
                seed_file = timestep_dir / f"seed_{r['seed']}.npz"
                np.savez_compressed(
                    seed_file,
                    wdn=np.array(r["wdn_per_step"]),
                    grad=np.array(r["grad_per_step"]),
                )

    print(f"Results saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="FRA Experiment Runner")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--seeds", nargs=2, type=int, default=[0, 1000],
                        help="Seed range [start, end)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without running")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory for model checkpoints")
    args = parser.parse_args()

    run_condition(
        config_path=args.config,
        seed_start=args.seeds[0],
        seed_end=args.seeds[1],
        dry_run=args.dry_run,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()

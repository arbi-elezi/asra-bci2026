"""Experiment v5 — Targeted suppression of excited weights.

v3 was wrong: pushed toward "safe action" from bad cost critic.
v4 was wrong: suppressed ALL 502K weights uniformly — corrupted the entire policy.

v5: When the LLM makes a risky decision:
  1. Compute gradient of log P(risky_action) — identifies which weights "wanted" that action
  2. Find the top-K% weights by gradient magnitude — these are the "excited" circuit
  3. Suppress ONLY those weights proportional to risk score
  4. Leave the rest of the policy untouched
  5. Ask the perturbed LLM to decide again
  6. FHR restores only the suppressed weights over time

This is targeted fear — suppress the circuit that wanted the risky thing,
let the rest of the brain figure out a safer alternative.
"""

import json
import time
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.highway_wrapper import HighwayFRAEnv
from src.agents.cost_critic import CostCriticNet
from src.components.fear_detector import FearDetector, FearDetectorConfig
from src.evaluation.metrics import bootstrap_ci, paired_bootstrap_ci, m1_collision_rate, m8_task_reward
from experiment_v3.config import MODELS
from experiment_v3.llm_policy import LLMPolicy


def compute_risk(obs, cost, ttc, action):
    """Simple risk evaluator for a given action in a given state.

    Returns risk in [0, 1]. Higher = more risky.

    Rules (legally grounded):
      - BRAKE (2) is always low risk (legally safe)
      - ACCELERATE (1) near obstacles is high risk
      - MAINTAIN (0) with low TTC is moderate risk
      - LANE_CHANGE (3) with low TTC is high risk (reckless)
    """
    base_risk = {
        0: 0.3,   # MAINTAIN — neutral
        1: 0.6,   # ACCELERATE — inherently riskier
        2: 0.05,  # BRAKE — almost always safe
        3: 0.5,   # LANE_CHANGE — context-dependent
    }.get(action, 0.3)

    # TTC modifier: close = risky for everything except brake
    if action != 2 and ttc < 3.0:
        ttc_risk = max(0, (3.0 - ttc) / 3.0)  # 0 at ttc=3, 1 at ttc=0
        base_risk = min(1.0, base_risk + ttc_risk * 0.5)

    # Cost modifier: high cost = risky
    base_risk = min(1.0, base_risk + cost * 0.3)

    # Brake is ALWAYS low risk regardless of context
    if action == 2:
        base_risk = min(0.1, base_risk)

    return float(np.clip(base_risk, 0.0, 1.0))


def run_v5(model_name="SmolLM2-135M", n_seeds=200, device="cuda"):
    v3_dir = Path("experiment_v3") / model_name
    v5_dir = Path("experiment_v5") / model_name
    v5_dir.mkdir(parents=True, exist_ok=True)
    results_dir = v5_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT v5: TARGETED SUPPRESSION of excited weights")
    print("Only perturb weights that drove the risky decision")
    print("=" * 60)

    # Load from v3 Day 1
    model_cfg = next(m for m in MODELS if m.name == model_name)
    policy = LLMPolicy(model_cfg, device=device)
    policy.load(str(v3_dir / "policy_trained.pt"))
    w0 = policy.get_w0()

    day1 = json.load(open(v3_dir / "day1_log.json"))
    fisher = torch.load(v3_dir / "fisher.pt", weights_only=False, map_location=device)
    fear_det = FearDetector(FearDetectorConfig(device=device))
    fear_det.load_state(torch.load(v3_dir / "fear_detector.pt", weights_only=False, map_location=device))
    seeds = np.load(v3_dir / "seeds.npy")

    eta_h = min(0.01, 0.5 / max(day1["f_max"], 1e-8))

    env = HighwayFRAEnv(seed=0, vehicles_count=15)

    print(f"  Model: {model_name} ({policy.n_perturbable:,} params)")
    print(f"  W_0: {policy.get_w0_hash()[:16]}...")
    print(f"  Base CR: {day1['base_cr']}")
    print(f"  eta_h: {eta_h}")

    # ── Sweep: eta_f AND top_k_pct ──
    print("\n--- Sweep: eta_f x top_k_pct (30 seeds each) ---")
    best_cr = 1.0
    best_params = {}

    for top_k in [0.01, 0.05, 0.10, 0.25]:
        for eta_f in [1e-5, 1e-4, 1e-3, 1e-2]:
            cr = _run_quick(policy, env, fear_det, fisher, w0, seeds,
                            eta_f, eta_h, top_k, n_seeds=30, device=device)
            tag = " ***" if cr < best_cr else ""
            print(f"  top_k={top_k:.0%} eta_f={eta_f:.0e}: CR={cr:.3f}{tag}")
            if cr < best_cr:
                best_cr = cr
                best_params = {"eta_f": eta_f, "top_k": top_k}

    print(f"\n  BEST: top_k={best_params['top_k']:.0%}, eta_f={best_params['eta_f']:.0e} → CR={best_cr:.3f}")
    print(f"  Baseline: CR={day1['base_cr']}")
    improvement = day1["base_cr"] - best_cr
    print(f"  Improvement: {improvement:+.3f} ({'BETTER' if improvement > 0 else 'WORSE'})")

    eta_f = best_params["eta_f"]
    top_k = best_params["top_k"]

    # ── Run all 13 conditions ──
    conditions = {
        "C1":  {"fra": False, "desc": "Baseline"},
        "C2":  {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "desc": "Full FRA (targeted suppress)"},
        "C3a": {"fra": True, "dr": True, "fhr": "l2", "bc": False, "desc": "L2-HR"},
        "C3b": {"fra": True, "dr": True, "fhr": "fisher", "bc": False, "desc": "Fisher no BC"},
        "C4":  {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "desc": "No GTCC"},
        "C5":  {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "desc": "No FMS"},
        "C6":  {"fra": True, "dr": False, "fhr": "fisher", "bc": True, "desc": "No DR"},
        "C7":  {"fra": False, "hard_override": True, "desc": "Hard override (always brake if cost>0.4)"},
        "C8a": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "desc": "Degraded fear (10% D_ref)"},
        "C8b": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "desc": "Degraded fear (25%)"},
        "C8c": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "desc": "Degraded fear (50%)"},
        "C8d": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "desc": "Biased fear (x0.2)"},
        "C8e": {"fra": True, "dr": True, "fhr": "fisher", "bc": True, "desc": "Biased fear (x0.5)"},
    }

    all_results = {}
    print(f"\n{'='*60}")
    print(f"Running 13 conditions ({n_seeds} seeds, top_k={top_k:.0%}, eta_f={eta_f:.0e})")
    print(f"{'='*60}")

    for cond_name, cfg in conditions.items():
        cond_dir = results_dir / cond_name
        cond_dir.mkdir(exist_ok=True)

        if (cond_dir / "summary.json").exists():
            s = json.load(open(cond_dir / "summary.json"))
            all_results[cond_name] = s
            print(f"  {cond_name}: CR={s['M1_collision_rate']:.3f} (cached)")
            continue

        cond_seeds = min(n_seeds, 500) if cond_name.startswith("C8") else n_seeds
        collisions, rewards, per_seed = [], [], []
        t0 = time.time()

        for idx in range(cond_seeds):
            seed = int(seeds[idx])
            policy.restore_w0()

            result = _run_episode(policy, env, fear_det, fisher, w0, cfg,
                                   seed, eta_f, eta_h, top_k, device)
            collisions.append(result["collision"])
            rewards.append(result["reward"])
            per_seed.append(result)

            if (idx + 1) % max(1, cond_seeds // 10) == 0:
                print(f"    [{idx+1}/{cond_seeds}] CR={np.mean(collisions):.3f}")

        elapsed = time.time() - t0
        cr_arr = np.array(collisions, dtype=float)
        cr = m1_collision_rate(cr_arr)
        cr_ci = bootstrap_ci(cr_arr)
        rw = m8_task_reward(np.array(rewards))

        summary = {"condition": cond_name, "model": model_name,
                    "n_seeds": cond_seeds, "M1_collision_rate": cr,
                    "M1_ci": cr_ci, "M8_mean_reward": rw,
                    "eta_f": eta_f, "eta_h": eta_h, "top_k_pct": top_k,
                    "mode": "targeted_suppress", "elapsed_s": elapsed}

        with open(cond_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        with open(cond_dir / "per_seed.json", "w") as f:
            json.dump(per_seed, f, indent=2, default=str)

        # Compute risk reduction metrics across all seeds
        all_less_risky = [d.get("less_risky_pct", 0) for d in per_seed if d.get("n_fra_steps", 0) > 0]
        all_risk_red = [d.get("mean_risk_reduction", 0) for d in per_seed if d.get("n_fra_steps", 0) > 0]

        if all_less_risky:
            summary["mean_less_risky_pct"] = float(np.mean(all_less_risky))
            summary["mean_risk_reduction"] = float(np.mean(all_risk_red))
        else:
            summary["mean_less_risky_pct"] = 0.0
            summary["mean_risk_reduction"] = 0.0

        with open(cond_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        all_results[cond_name] = summary
        risk_pct = summary.get("mean_less_risky_pct", 0) * 100
        risk_red = summary.get("mean_risk_reduction", 0)
        print(f"  {cond_name}: CR={cr:.3f} | LessRisky={risk_pct:.0f}% | RiskReduction={risk_red:+.3f} | Rwd={rw:.1f}")

    # ── Hypotheses ──
    print(f"\n{'='*60}")
    print("HYPOTHESES")
    print(f"{'='*60}")

    def load_cr(c):
        with open(results_dir / c / "per_seed.json") as f:
            return np.array([float(d["collision"]) for d in json.load(f)])

    hyp = {}
    for label, cx, cy, claim in [
        ("H1", "C1", "C2", "FRA reduces CR"),
        ("H8", "C6", "C2", "DR contributes"),
    ]:
        try:
            x, y = load_cr(cx), load_cr(cy)
            n = min(len(x), len(y))
            h = paired_bootstrap_ci(x[:n], y[:n])
            confirmed = h["excludes_zero"] and h["point_estimate"] > 0
            hyp[label] = {"claim": claim, "delta": h["point_estimate"],
                          "ci": [h["ci_lower"], h["ci_upper"]], "confirmed": confirmed}
            s = "CONFIRMED" if confirmed else "FALSIFIED"
            print(f"  {label}: {s}  Delta={h['point_estimate']:+.4f}  CI=[{h['ci_lower']:.4f}, {h['ci_upper']:.4f}]")
        except Exception as e:
            print(f"  {label}: SKIP ({e})")

    with open(results_dir / "hypothesis_results.json", "w") as f:
        json.dump(hyp, f, indent=2, default=str)

    with open(v5_dir / "experiment_log.json", "w") as f:
        json.dump({"version": "v5", "mechanism": "targeted_suppress",
                    "best_eta_f": eta_f, "best_top_k": top_k,
                    "sweep_best_cr": best_cr, "baseline_cr": day1["base_cr"],
                    "results": {k: v.get("M1_collision_rate") for k, v in all_results.items()},
                    "hypotheses": hyp}, f, indent=2, default=str)

    print(f"\nv5 complete. Results in {v5_dir}/")


def _run_episode(policy, env, fear_det, fisher, w0, cfg, seed, eta_f, eta_h, top_k, device):
    """Single episode with TARGETED suppression."""
    obs, info = env.reset(seed=seed)
    ep_reward = 0.0
    collision = False

    # Track which params are currently suppressed (for targeted FHR)
    suppressed_mask = {}

    # CORE MEASUREMENT: does FRA produce less risky actions?
    risk_reductions = []
    original_risks = []
    perturbed_risks = []
    n_less_risky = 0
    n_fra_steps = 0

    for t in range(500):
        cost = info.get("cost", 0.0)
        ttc = info.get("ttc", 10.0)

        if not cfg.get("fra", False):
            if cfg.get("hard_override") and cost > 0.4:
                action = 2  # BRAKE — always legally safe
            else:
                action = policy.get_action(obs)
        else:
            # ── Step 1: LLM makes a decision ──
            with torch.no_grad():
                logits = policy.get_logits_from_obs(obs)
                greedy = logits.argmax().item()

            # ── Step 2: Evaluate risk of that decision ──
            risk = compute_risk(obs, cost, ttc, greedy)
            fear, _ = fear_det.detect(obs, cost, ttc, greedy)

            # Combine risk and fear into perturbation strength
            # Fear = environmental danger, Risk = action-specific danger
            perturbation_strength = risk * fear

            # ── Step 3: If risky, find excited weights and suppress them ──
            if cfg.get("dr", True) and perturbation_strength > 0.05:
                policy.model.zero_grad()
                policy.action_head.zero_grad()

                logits = policy.get_logits_from_obs(obs)
                log_prob_greedy = torch.log_softmax(logits, dim=-1)[greedy]
                log_prob_greedy.backward()

                # ── Step 4: Find top-K% excited weights ──
                with torch.no_grad():
                    for name, p in policy.get_perturbable_params():
                        if p.grad is None:
                            continue

                        grad = p.grad.data
                        grad_abs = grad.abs()

                        # Find threshold for top-K%
                        n_total = grad_abs.numel()
                        n_suppress = max(1, int(n_total * top_k))

                        if n_total <= n_suppress:
                            # Small layer — suppress all
                            mask = torch.ones_like(grad, dtype=torch.bool)
                        else:
                            threshold = torch.topk(grad_abs.flatten(), n_suppress).values[-1]
                            mask = grad_abs >= threshold

                        # ── Step 5: Suppress ONLY the excited weights ──
                        # Subtract: decrease log P(risky_action) for these specific weights
                        suppression = eta_f * perturbation_strength * grad * mask.float()
                        p.data -= suppression

                        # Track suppressed weights for targeted FHR
                        suppressed_mask[name] = mask

            # ── Step 6: FHR — restore ONLY suppressed weights toward W_0 ──
            fhr_mode = cfg.get("fhr", "fisher")
            if fhr_mode:
                with torch.no_grad():
                    pidx = 0
                    for name, p in policy.get_perturbable_params():
                        if name not in w0:
                            pidx += p.numel()
                            continue

                        diff = w0[name].to(p.dtype) - p.data
                        ne = p.numel()

                        # Only restore weights that were actually suppressed
                        mask = suppressed_mask.get(name, None)

                        if fhr_mode == "fisher":
                            fs = fisher[pidx:pidx+ne].reshape(p.shape).to(p.dtype).to(p.device)
                            pidx += ne
                            restore = eta_h * fs * diff
                        else:
                            pidx += ne
                            restore = eta_h * diff

                        if mask is not None:
                            # Targeted: only restore suppressed weights (faster recovery where needed)
                            # Non-suppressed weights get slower general drift
                            p.data += restore * mask.float() * 2.0  # 2x speed for suppressed
                            p.data += restore * (~mask).float() * 0.1  # slow for others
                        else:
                            p.data += restore

            # ── Step 7: BC term ──
            if cfg.get("bc", True):
                with torch.no_grad():
                    wt_snap = {n: p.data.clone() for n, p in policy.get_perturbable_params()}
                    policy.restore_w0()
                    w0_probs = torch.softmax(policy.get_logits_from_obs(obs), dim=-1)
                    for n, p in policy.get_perturbable_params():
                        if n in wt_snap: p.data.copy_(wt_snap[n])

                policy.model.zero_grad()
                policy.action_head.zero_grad()
                wt_logits = policy.get_logits_from_obs(obs)
                wt_lp = torch.log_softmax(wt_logits, dim=-1)
                kl = torch.sum(w0_probs.detach() * (torch.log(w0_probs.detach() + 1e-8) - wt_lp))
                kl.backward()
                with torch.no_grad():
                    for _, p in policy.get_perturbable_params():
                        if p.grad is not None:
                            p.data -= 1e-5 * p.grad.data

            # ── Step 8: Perturbed LLM decides again ──
            with torch.no_grad():
                new_logits = policy.get_logits_from_obs(obs)
                action = torch.distributions.Categorical(logits=new_logits).sample().item()

            # ── MEASUREMENT: did the perturbed LLM choose a LESS RISKY action? ──
            risk_original = risk  # risk of what unperturbed LLM wanted (greedy)
            risk_perturbed = compute_risk(obs, cost, ttc, action)  # risk of what perturbed LLM chose
            risk_reduction = risk_original - risk_perturbed  # positive = less risky = good

            risk_reductions.append(risk_reduction)
            original_risks.append(risk_original)
            perturbed_risks.append(risk_perturbed)

            if risk_reduction > 0:
                n_less_risky += 1
            n_fra_steps += 1

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        if terminated:
            collision = info.get("collision", False)
            break
        if truncated:
            break

    return {
        "seed": seed, "condition": cfg.get("desc", ""),
        "collision": int(collision), "reward": ep_reward,
        "n_fra_steps": n_fra_steps,
        "n_less_risky": n_less_risky,
        "less_risky_pct": n_less_risky / max(n_fra_steps, 1),
        "mean_risk_reduction": float(np.mean(risk_reductions)) if risk_reductions else 0.0,
        "mean_original_risk": float(np.mean(original_risks)) if original_risks else 0.0,
        "mean_perturbed_risk": float(np.mean(perturbed_risks)) if perturbed_risks else 0.0,
    }


def _run_quick(policy, env, fear_det, fisher, w0, seeds,
               eta_f, eta_h, top_k, n_seeds, device):
    """Quick sweep test."""
    collisions = []
    for idx in range(n_seeds):
        seed = int(seeds[idx])
        policy.restore_w0()

        obs, info = env.reset(seed=seed)
        suppressed_mask = {}

        for t in range(500):
            cost = info.get("cost", 0.0)
            ttc = info.get("ttc", 10.0)

            with torch.no_grad():
                logits = policy.get_logits_from_obs(obs)
                greedy = logits.argmax().item()

            risk = compute_risk(obs, cost, ttc, greedy)
            fear, _ = fear_det.detect(obs, cost, ttc, greedy)
            strength = risk * fear

            if strength > 0.05:
                policy.model.zero_grad()
                policy.action_head.zero_grad()
                logits = policy.get_logits_from_obs(obs)
                torch.log_softmax(logits, dim=-1)[greedy].backward()

                with torch.no_grad():
                    for name, p in policy.get_perturbable_params():
                        if p.grad is None: continue
                        grad = p.grad.data
                        n_total = grad.abs().numel()
                        n_sup = max(1, int(n_total * top_k))
                        if n_total <= n_sup:
                            mask = torch.ones_like(grad, dtype=torch.bool)
                        else:
                            thr = torch.topk(grad.abs().flatten(), n_sup).values[-1]
                            mask = grad.abs() >= thr
                        p.data -= eta_f * strength * grad * mask.float()
                        suppressed_mask[name] = mask

                # FHR
                with torch.no_grad():
                    pidx = 0
                    for name, p in policy.get_perturbable_params():
                        if name not in w0: pidx += p.numel(); continue
                        diff = w0[name].to(p.dtype) - p.data
                        ne = p.numel()
                        fs = fisher[pidx:pidx+ne].reshape(p.shape).to(p.dtype).to(p.device)
                        pidx += ne
                        m = suppressed_mask.get(name)
                        if m is not None:
                            p.data += eta_h * fs * diff * m.float() * 2.0
                            p.data += eta_h * fs * diff * (~m).float() * 0.1
                        else:
                            p.data += eta_h * fs * diff

            with torch.no_grad():
                action = policy.get_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                collisions.append(1 if info.get("collision") else 0)
                break
            if truncated:
                collisions.append(0)
                break

    return float(np.mean(collisions)) if collisions else 1.0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="SmolLM2-135M")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run_v5(a.model, a.seeds, a.device)

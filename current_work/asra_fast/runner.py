"""Rollout / evaluation driver for ASRA-fast experiments."""
from __future__ import annotations

import sys, json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.environment.highway_wrapper import HighwayFRAEnv
from current_work.asra_fast.core import (
    MLPActor, ASRA, ASRAConfig, CostSalience, SalienceConfig,
    baseline_action, bootstrap_ci, compute_risk,
)


def load_actor(path: str, device: str = "cpu") -> tuple[MLPActor, dict]:
    data = torch.load(path, map_location=device, weights_only=False)
    actor = MLPActor(12, 4, data.get("hidden", 64)).to(device)
    actor.load_state_dict(data["actor"])
    for p in actor.parameters():
        p.requires_grad_(True)
    return actor, data


def clone_state(actor: MLPActor) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in actor.state_dict().items()}


def restore_state(actor: MLPActor, w0: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for k, p in actor.named_parameters():
            if k in w0:
                p.data.copy_(w0[k])


def w0_named(actor: MLPActor) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in actor.named_parameters()}


def eval_episodes(actor, w0_full, w0_par, fisher, salience,
                  spec: dict, n_episodes: int, seed: int, vehicles: int = 15,
                  device: str = "cpu", max_steps: int = 500, density: float = 1.5,
                  high_speed_reward: float = 0.4) -> list[dict]:
    """spec: {'type':'asra', 'cfg':ASRAConfig} or {'type':'baseline','kind':..., 'ttc_k':, 'temp':}."""
    env = HighwayFRAEnv(seed=seed, vehicles_count=vehicles, vehicles_density=density,
                        high_speed_reward=high_speed_reward)
    rng = np.random.default_rng(seed)
    out = []

    asra = None
    if spec["type"] == "asra":
        asra = ASRA(spec["cfg"], w0_par, fisher, salience, device=device,
                    rng=np.random.default_rng(seed + 777))

    for ep in range(n_episodes):
        restore_state(actor, w0_full)          # always start each episode from W_0
        if asra is not None:
            asra.reset()
        # PAIRED seeds: env randomness depends only on (seed, ep), identical across conditions
        obs, info = env.reset(seed=seed * 100_000 + ep)
        cost, ttc = info.get("cost", 0.0), info.get("ttc", 10.0)
        collision, ret, length, speed_sum, brake_count = 0, 0.0, 0, 0.0, 0
        active = 0

        for t in range(max_steps):
            if asra is not None:
                action, diag = asra.act(actor, obs, cost, ttc)
                active += diag["active"]
            else:
                with torch.no_grad():
                    logits = actor(torch.as_tensor(obs, dtype=torch.float32,
                                                   device=device).unsqueeze(0)).squeeze(0)
                action = baseline_action(spec["kind"], logits, cost, ttc, rng,
                                         ttc_k=spec.get("ttc_k", 2.0),
                                         temp=spec.get("temp", 2.0),
                                         base_decode=spec.get("base_decode", "sample"))
            if action == 2:
                brake_count += 1
            speed_sum += float(obs[2])         # ego vx proxy
            obs, r, term, trunc, info = env.step(action)
            cost, ttc = info.get("cost", 0.0), info.get("ttc", 10.0)
            ret += r; length += 1
            if term:
                collision = int(info.get("collision", False)); break
            if trunc:
                break

        rec = {"collision": collision, "ret": ret, "length": length,
               "mean_speed": speed_sum / max(length, 1),
               "brake_frac": brake_count / max(length, 1)}
        if asra is not None:
            rec["lessrisky"] = asra.lessrisky_hits / max(asra.active_steps, 1)
            rec["active_frac"] = active / max(length, 1)
        out.append(rec)

    env.close()
    return out


def summarize(records: list[dict]) -> dict:
    cr = np.array([r["collision"] for r in records], dtype=np.float64)
    ret = np.array([r["ret"] for r in records], dtype=np.float64)
    spd = np.array([r["mean_speed"] for r in records], dtype=np.float64)
    ln = np.array([r["length"] for r in records], dtype=np.float64)
    brk = np.array([r.get("brake_frac", 0.0) for r in records], dtype=np.float64)
    s = {"CR": bootstrap_ci(cr), "ret": bootstrap_ci(ret), "speed": bootstrap_ci(spd),
         "length": bootstrap_ci(ln), "brake_frac": bootstrap_ci(brk), "n_ep": len(records)}
    if "lessrisky" in records[0]:
        s["lessrisky"] = bootstrap_ci(np.array([r["lessrisky"] for r in records]))
        s["active_frac"] = bootstrap_ci(np.array([r["active_frac"] for r in records]))
    return s


# --------------------------------------------------------------------------- #
# Salience builders
# --------------------------------------------------------------------------- #
def build_salience(kind: str = "cost", dref_states=None, weights=(0.3, 0.2, 0.5),
                   device: str = "cpu"):
    if kind == "cost":
        return CostSalience(SalienceConfig(kind="cost"))
    # ensemble: lazily import the project's FearDetector and train on D_ref
    from src.components.fear_detector import FearDetector, FearDetectorConfig
    cfg = FearDetectorConfig(device=device, weight_ae=weights[0],
                             weight_if=weights[1], weight_ca=weights[2])
    fd = FearDetector(cfg)
    fd.train_on_d_ref(dref_states)
    def _sal(obs, cost, ttc):
        s, _ = fd.detect(obs, cost, ttc, 0)
        return s
    return _sal


if __name__ == "__main__":
    # tiny smoke test executed by smoke.py instead
    pass

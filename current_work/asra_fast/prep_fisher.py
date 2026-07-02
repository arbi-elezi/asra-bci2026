"""Collect D_ref from a frozen base policy and compute+save its diagonal Fisher."""
import sys, argparse
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from current_work.asra_fast.runner import load_actor
from current_work.asra_fast.trainer import collect_dref, compute_fisher


def prep(base_path: str, out_path: str, n_dref: int = 4000, vehicles: int = 15,
         device: str = "cpu"):
    actor, data = load_actor(base_path, device=device)
    veh = data.get("vehicles", vehicles)
    den = data.get("density", 1.5)
    hsr = data.get("high_speed_reward", 0.4)
    states, costs = collect_dref(actor, n=n_dref, vehicles=veh, device=device,
                                 density=den, high_speed_reward=hsr)
    fisher = compute_fisher(actor, states, device=device)
    torch.save(fisher, out_path)
    fmin = min(f.min().item() for f in fisher.values())
    fmax = max(f.max().item() for f in fisher.values())
    print(f"  D_ref={len(states)}  Fisher range [{fmin:.4f},{fmax:.4f}] -> {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_dref", type=int, default=4000)
    ap.add_argument("--vehicles", type=int, default=15)
    a = ap.parse_args()
    prep(a.base, a.out, n_dref=a.n_dref, vehicles=a.vehicles)

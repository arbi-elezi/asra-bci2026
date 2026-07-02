"""Diagnose a base policy's CHARACTER: greedy vs sample CR/speed + action histogram."""
import sys, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.environment.highway_wrapper import HighwayFRAEnv
from current_work.asra_fast.runner import load_actor

ANAMES = {0: "maintain", 1: "accel", 2: "brake", 3: "lane"}


def diag(base_path, n_ep=60, vehicles=15, device="cpu", density=1.5):
    actor, data = load_actor(base_path, device=device)
    env = HighwayFRAEnv(seed=0, vehicles_count=vehicles, vehicles_density=density)
    print(f"{Path(base_path).name}  (train CR~{data.get('train_cr')}, step {data.get('step')}, vehicles {vehicles})")
    for mode in ["greedy", "sample"]:
        crs, speeds, lens, acts = [], [], [], np.zeros(4)
        for ep in range(n_ep):
            obs, info = env.reset(seed=10_000 + ep); col = 0; sp = []; L = 0
            for t in range(500):
                with torch.no_grad():
                    logits = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                a = int(logits.argmax()) if mode == "greedy" else \
                    int(torch.distributions.Categorical(logits=logits).sample())
                acts[a] += 1; sp.append(float(obs[2])); L += 1
                obs, r, term, trunc, info = env.step(a)
                if term: col = int(info.get("collision", False)); break
                if trunc: break
            crs.append(col); speeds.append(np.mean(sp)); lens.append(L)
        hist = acts / acts.sum()
        print(f"  [{mode:6s}] CR={np.mean(crs):.3f}  speed={np.mean(speeds):.2f}  len={np.mean(lens):.1f}  "
              f"actions: " + " ".join(f"{ANAMES[i]}={hist[i]:.2f}" for i in range(4)))
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--vehicles", type=int, default=15)
    ap.add_argument("--n_ep", type=int, default=60)
    ap.add_argument("--density", type=float, default=1.5)
    a = ap.parse_args()
    diag(a.base, n_ep=a.n_ep, vehicles=a.vehicles, density=a.density)

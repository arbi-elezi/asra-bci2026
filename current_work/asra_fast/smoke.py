"""Fast correctness smoke test for the ASRA mechanism (no training needed)."""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from current_work.asra_fast.core import MLPActor, ASRA, ASRAConfig, CostSalience, SalienceConfig
from current_work.asra_fast.runner import (
    eval_episodes, summarize, clone_state, restore_state, w0_named,
)
from current_work.asra_fast.trainer import compute_fisher, collect_dref

DEV = "cpu"
torch.manual_seed(0); np.random.seed(0)

print("=" * 64)
print("ASRA MECHANISM SMOKE TEST (random actor)")
print("=" * 64)

actor = MLPActor(12, 4, 64).to(DEV)
for p in actor.parameters():
    p.requires_grad_(True)
n_params = sum(p.numel() for p in actor.parameters())
print(f"Actor params: {n_params} (expect 5252)")
assert n_params == 5252, "param count mismatch"

w0_full = clone_state(actor)
w0_par = w0_named(actor)
w0_par_backup = {k: v.clone() for k, v in w0_par.items()}

# cheap fisher from random states
states = np.random.randn(500, 12).astype(np.float32) * 5
fisher = compute_fisher(actor, states, device=DEV)
print(f"Fisher tensors: {list(fisher.keys())}")
fmin = min(f.min().item() for f in fisher.values())
fmax = max(f.max().item() for f in fisher.values())
print(f"Fisher range: [{fmin:.4f}, {fmax:.4f}]  (normalized to (0,1])")
assert 0 < fmin <= fmax <= 1.0 + 1e-6

salience = CostSalience(SalienceConfig(kind="cost"))

# ---- Test 1+2+3: run ASRA for a few episodes, check invariants ----
cfg = ASRAConfig(gain=3.0, select="greedy", weight_channel=True, conf_channel=True)
asra = ASRA(cfg, w0_par, fisher, salience, device=DEV, rng=np.random.default_rng(1))

# manual episode loop to inspect mid-episode deviation
from src.environment.highway_wrapper import HighwayFRAEnv
env = HighwayFRAEnv(seed=0, vehicles_count=15)
restore_state(actor, w0_full)
asra.reset()
obs, info = env.reset(seed=42)
max_dev = 0.0
actions_seen = set()
for t in range(60):
    cost, ttc = info.get("cost", 0.0), info.get("ttc", 10.0)
    a, diag = asra.act(actor, obs, cost, ttc)
    actions_seen.add(a)
    # measure deviation of live params from W_0
    dev = max((actor.state_dict()[k] - w0_full[k]).abs().max().item() for k in w0_full)
    max_dev = max(max_dev, dev)
    obs, r, term, trunc, info = env.step(a)
    if term or trunc:
        break
print(f"Actions seen: {sorted(actions_seen)} (subset of 0..3)")
assert actions_seen.issubset({0, 1, 2, 3})
print(f"Max mid-episode weight deviation from W_0: {max_dev:.6e} (must be > 0 => perturbation real)")
assert max_dev > 0, "weights never moved -> weight channel dead"

# restore and verify EXACT recovery
restore_state(actor, w0_full)
post_dev = max((actor.state_dict()[k] - w0_full[k]).abs().max().item() for k in w0_full)
print(f"Post-restore deviation: {post_dev:.2e} (must be 0 => W_0 invariant holds)")
assert post_dev == 0.0

# verify W_0 dict itself was NOT mutated by ASRA
w0_drift = max((w0_par[k] - w0_par_backup[k]).abs().max().item() for k in w0_par)
print(f"W_0 dict drift after ASRA: {w0_drift:.2e} (must be 0 => W_0 never overwritten)")
assert w0_drift == 0.0
env.close()
print("PASS: W_0 invariant + real perturbation + valid actions")

# ---- Test 4: targeting concentrates change near epicenter ----
from current_work.asra_fast.core import apply_gaussian_
import torch.nn.functional as F
restore_state(actor, w0_full)
o = torch.as_tensor(np.random.randn(12).astype(np.float32) * 5, device=DEV).unsqueeze(0)
actor.zero_grad(set_to_none=True)
lg = actor(o).squeeze(0)
greedy = int(lg.argmax())
F.log_softmax(lg, dim=-1)[greedy].backward()
# pick the largest weight tensor
name, p = max(actor.named_parameters(), key=lambda kv: kv[1].numel())
g = p.grad.detach()
epi = int(g.abs().view(-1).argmax())
before = p.data.view(-1).clone()
apply_gaussian_(p, g, epi, 0.05, 0.5)
change = (p.data.view(-1) - before).abs()
near = change[max(0, epi-3):epi+4].mean().item()
far = change[(epi + p.numel()//2) % p.numel()].item()
print(f"Targeting: |change| near epicenter={near:.3e}  far={far:.3e} (near >> far)")
assert near > far
restore_state(actor, w0_full)
print("PASS: gradient targeting localizes perturbation")

# ---- Test 5: all select modes run + eps/s ----
print("\nTiming + select-mode check (random actor, 25 eps each):")
for mode in ["greedy", "sample", "bounded"]:
    cfg = ASRAConfig(gain=2.0, select=mode)
    t0 = time.time()
    recs = eval_episodes(actor, w0_full, w0_par, fisher, salience,
                         {"type": "asra", "cfg": cfg}, n_episodes=25, seed=1, device=DEV)
    s = summarize(recs)
    dt = time.time() - t0
    print(f"  asra/{mode:8s}: CR={s['CR']['mean']:.2f} ret={s['ret']['mean']:6.2f} "
          f"active={s['active_frac']['mean']:.2f} | {25/dt:.1f} eps/s")

# baseline path
t0 = time.time()
recs = eval_episodes(actor, w0_full, w0_par, fisher, salience,
                     {"type": "baseline", "kind": "ttc_brake", "ttc_k": 2.0},
                     n_episodes=25, seed=1, device=DEV)
s = summarize(recs)
print(f"  baseline/ttc_brake: CR={s['CR']['mean']:.2f} ret={s['ret']['mean']:6.2f} | {25/(time.time()-t0):.1f} eps/s")

print("\nALL SMOKE TESTS PASSED")

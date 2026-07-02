"""Unit test: multi-scenario env is valid BEFORE we spend hours training/evaluating on it."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.environment.highway_wrapper import HighwayFRAEnv


def check(scenario, n_ep=8, vehicles=8, density=1.0):
    env = HighwayFRAEnv(scenario=scenario, seed=0, vehicles_count=vehicles, vehicles_density=density)
    assert env.action_space.n == 4, f"{scenario}: action space {env.action_space.n} != 4"
    obs, info = env.reset(seed=1)
    assert obs.shape == (12,), f"{scenario}: obs shape {obs.shape} != (12,)"
    costs, ttcs, cols, lens, rews, acted = [], [], [], [], [], set()
    rng = np.random.default_rng(0)
    for ep in range(n_ep):
        obs, info = env.reset(seed=100 + ep); L = 0; col = 0
        for t in range(60):
            a = int(rng.integers(4)); acted.add(a)
            obs, r, term, trunc, info = env.step(a)
            assert obs.shape == (12,), f"{scenario}: step obs shape {obs.shape}"
            assert np.isfinite(r), f"{scenario}: non-finite reward"
            assert np.all(np.isfinite(obs)), f"{scenario}: non-finite obs"
            costs.append(info.get("cost")); ttcs.append(info.get("ttc")); L += 1
            if term or trunc:
                col = int(info.get("collision", False)); break
        cols.append(col); lens.append(L)
    env.close()
    c = np.array(costs, float); tt = np.array(ttcs, float)
    assert ((c >= 0) & (c <= 1)).all(), f"{scenario}: cost out of [0,1]"
    assert ((tt >= 0) & (tt <= 10.0001)).all(), f"{scenario}: ttc out of [0,10]"
    print(f"  [{scenario:10s}] OK | actions seen {sorted(acted)} | ep_len {np.mean(lens):.1f} "
          f"| random CR {np.mean(cols):.2f} | cost {c.mean():.2f} | ttc {tt.mean():.1f}")
    return np.mean(cols)


if __name__ == "__main__":
    print("=== multi-scenario env unit test ===")
    ok = True
    for scen in ["highway", "merge", "roundabout"]:
        try:
            check(scen)
        except Exception as e:
            ok = False; print(f"  [{scen:10s}] FAIL: {type(e).__name__}: {e}")
    print("ALL SCENARIOS OK" if ok else "SOME SCENARIOS FAILED")

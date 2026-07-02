"""ASRA-Targeted (ASRA-T): consequence-directed routing that beats mask-and-defer."""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast.analyze import paired_bootstrap_diff


class HazardChoice:
    """3-lane corridor. In a hazard the current lane is blocked (advance crashes); EXACTLY ONE
    adjacent lane is clear, the other is blocked; braking is always safe. Which side is clear is
    random, so a policy with a FIXED directional preference has an unsafe 2nd choice half the time."""
    A = 4  # 0=advance 1=switch-left 2=switch-right 3=brake
    def __init__(self, T=40, p=0.5, seed=0):
        self.T, self.p, self.rng = T, p, np.random.default_rng(seed)
    def reset(self, seed=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.t = 0; self._draw(); return self._obs()
    def _draw(self):
        self.hazard = self.rng.random() < self.p
        # in a hazard, exactly one adjacent lane clear; else both clear
        if self.hazard:
            self.left_clear = self.rng.random() < 0.5; self.right_clear = not self.left_clear
        else:
            self.left_clear = self.right_clear = True
    def _obs(self):
        return np.array([float(self.hazard), float(self.left_clear), float(self.right_clear)], float)
    def step(self, a):
        crash, rew = False, 0.0
        if a == 0: crash, rew = (True, -10.) if self.hazard else (False, 2.0)      # advance
        elif a == 1: (crash, rew) = (False, 1.0) if self.left_clear else (True, -10.)
        elif a == 2: (crash, rew) = (False, 1.0) if self.right_clear else (True, -10.)
        else: rew = 0.0                                                            # brake safe
        self.t += 1; done = crash or self.t >= self.T
        if not done: self._draw()
        return self._obs(), rew, done, crash


def policy_logits(obs):
    """Aggressive policy with a FIXED directional bias: prefers advance, then ALWAYS switch-left,
    then switch-right, then brake --- it does NOT condition its 2nd choice on which lane is clear.
    So its 2nd choice (switch-left) is unsafe whenever the left lane is blocked."""
    return np.array([3.0, 1.0, 0.5, 0.0], float)


def risk(obs, a):
    """Consequence model (independent of the policy): high risk for advancing into a hazard or
    switching into a BLOCKED lane; low for switching into a CLEAR lane; lowest for braking."""
    hazard, lc, rc = obs[0] > 0.5, obs[1] > 0.5, obs[2] > 0.5
    if not hazard: return {0: 0.1, 1: 0.2, 2: 0.2, 3: 0.05}[a]
    return {0: 0.9,
            1: 0.2 if lc else 0.9,
            2: 0.2 if rc else 0.9,
            3: 0.05}[a]


def act(mode, obs, rng, lam=3.0):
    lg = policy_logits(obs); greedy = int(lg.argmax())
    S = obs[0]  # salience = hazard
    if mode == "raw_greedy": return greedy
    if mode == "override_brake": return 3 if S > 0.5 else greedy
    if mode == "mask_defer":                                   # remove risky greedy, take policy's next choice
        if S * risk(obs, greedy) > 0.05:
            lg2 = lg.copy(); lg2[greedy] = -1e9; return int(lg2.argmax())
        return greedy
    if mode == "asra_t":                                       # consequence-directed routing
        if S * risk(obs, greedy) > 0.05:
            scored = lg - lam * np.array([risk(obs, a) for a in range(4)])
            scored[greedy] = -1e9                              # remove the flagged risky action (as the mask does)
            return int(scored.argmax())                        # ...then defer BY CONSEQUENCE, not blindly
        return greedy
    raise ValueError(mode)


def run(mode, n_ep, seed, T=40, p=0.5, **kw):
    env = HazardChoice(T=T, p=p, seed=seed); rng = np.random.default_rng(seed + 1)
    crs, perfs = [], []
    for ep in range(n_ep):
        obs = env.reset(seed=seed * 100000 + ep); R = 0.0; crash = 0
        for t in range(T):
            a = act(mode, obs, rng, **kw); obs, r, done, cr = env.step(a); R += r
            if done: crash = int(cr); break
        crs.append(crash); perfs.append(R / T)
    return np.array(crs, float), np.array(perfs, float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--n_ep", type=int, default=400); ap.add_argument("--out", default="current_work/results_v3/asra_targeted.json")
    a = ap.parse_args()
    modes = ["raw_greedy", "override_brake", "mask_defer", "asra_t"]
    agg = {m: {"cr": [], "perf": []} for m in modes}
    for s in range(a.seeds):
        for m in modes:
            c, p = run(m, a.n_ep, s); agg[m]["cr"].append(float(c.mean())); agg[m]["perf"].append(float(p.mean()))
    print("=== ASRA-Targeted vs mask-and-defer (policy's 2nd choice is unsafe) ===")
    for m in modes:
        print(f"  {m:14s} CR={np.mean(agg[m]['cr']):.3f}  perf={np.mean(agg[m]['perf']):+.3f}")
    cr = lambda m: np.array(agg[m]["cr"]); pf = lambda m: np.array(agg[m]["perf"])
    d_cr = paired_bootstrap_diff(cr("mask_defer"), cr("asra_t"))     # mask_cr - asra_cr ; >0 => ASRA-T safer
    d_pf = paired_bootstrap_diff(pf("asra_t"), pf("mask_defer"))     # asra_perf - mask_perf
    d_ov = paired_bootstrap_diff(pf("asra_t"), pf("override_brake"))
    print(f"\n  ASRA-T vs mask-defer: CR {np.mean(agg['mask_defer']['cr']):.3f}->{np.mean(agg['asra_t']['cr']):.3f} "
          f"(dCR={d_cr['diff']:+.3f} CI[{d_cr['lo']:+.3f},{d_cr['hi']:+.3f}], {'ASRA-T SAFER' if d_cr['excludes_zero'] and d_cr['diff']>0 else 'tie'})"
          f" | dperf={d_pf['diff']:+.3f} CI[{d_pf['lo']:+.3f},{d_pf['hi']:+.3f}]")
    print(f"  ASRA-T vs override:  dperf={d_ov['diff']:+.3f} CI[{d_ov['lo']:+.3f},{d_ov['hi']:+.3f}] "
          f"({'ASRA-T higher perf' if d_ov['excludes_zero'] and d_ov['diff']>0 else 'tie'})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(agg, open(a.out, "w"), indent=2)
    print("saved", a.out)

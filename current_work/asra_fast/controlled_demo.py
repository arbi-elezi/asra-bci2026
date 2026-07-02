"""Controlled existence proof for the precondition."""
import sys, json, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from current_work.asra_fast.analyze import paired_bootstrap_diff


class Corridor:
    """3-lane corridor; in a hazard the current lane is blocked but a reachable clear
    adjacent lane is GUARANTEED (the precondition is well-defined)."""
    def __init__(self, T=40, p=0.45, seed=0):
        self.T, self.p, self.rng = T, p, np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.t, self.lane = 0, 1
        self._draw(); return self._obs()

    def _draw(self):
        self.hazard = self.rng.random() < self.p
        self.left_clear = self.lane > 0
        self.right_clear = self.lane < 2
        if self.hazard:
            # guarantee >=1 reachable clear lane: block at most one available side
            avail = [s for s, ok in [("L", self.left_clear), ("R", self.right_clear)] if ok]
            if len(avail) == 2 and self.rng.random() < 0.5:
                if self.rng.random() < 0.5: self.left_clear = False
                else: self.right_clear = False
            # if only one side available, never block it -> reachable clear lane remains

    def _obs(self):
        oh = [0., 0., 0.]; oh[self.lane] = 1.
        return np.array([float(self.hazard), float(self.left_clear), float(self.right_clear), *oh], float)

    def step(self, a):
        crash, rew = False, 0.0
        if a == 0:                                   # advance
            if self.hazard: crash, rew = True, -10.0
            else: rew = 2.0
        elif a == 1:                                 # switch left
            if self.lane > 0 and self.left_clear: self.lane -= 1; rew = 1.0
            else: crash, rew = True, -10.0
        elif a == 2:                                 # switch right
            if self.lane < 2 and self.right_clear: self.lane += 1; rew = 1.0
            else: crash, rew = True, -10.0
        else: rew = 0.0                              # brake
        self.t += 1
        done = crash or self.t >= self.T
        if not done: self._draw()
        return self._obs(), rew, done, crash


def policy_logits(obs):
    """Aggressive frozen policy: prefers ADVANCE, but encodes the safe alternative
    (clear-lane switch ranked above blocked-lane switch and braking)."""
    hazard, lc, rc, l0, l1, l2 = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]
    adv = 3.0                                        # always prefers fast advance
    sl = 1.0 if lc > 0.5 else -3.0                  # knows which lane is clear
    sr = 1.0 if rc > 0.5 else -3.0
    brk = 0.0
    return np.array([adv, sl, sr, brk], float)


def risk(obs, a):
    return {0: 0.9, 1: 0.4, 2: 0.4, 3: 0.05}.get(a, 0.5) if obs[0] > 0.5 else 0.1


def run_episode(mode, env, seed, gain=4.0, p_brake=0.0, rng=None):
    obs = env.reset(seed=seed); R, crash = 0.0, 0
    sup = np.zeros(4)
    for t in range(env.T):
        lg = policy_logits(obs); greedy = int(lg.argmax())
        if mode == "raw_greedy":
            a = greedy
        elif mode == "override_brake":
            a = 3 if obs[0] > 0.5 else greedy
        elif mode == "prob_brake":
            a = 3 if rng.random() < p_brake else greedy
        elif mode == "asra_greedy":                  # suppress risky action under salience, greedy decode
            S = obs[0]; Rk = risk(obs, greedy); alpha = S * Rk
            sup = np.zeros(4)                         # surgical: suppression does not linger past the threat
            if alpha > 0.05:
                sup[greedy] -= 4.0 * gain * alpha
            a = int((lg + sup).argmax())
        elif mode == "mask_defer":                   # TRIVIAL baseline: block the risky greedy, take policy's 2nd choice
            S = obs[0]; Rk = risk(obs, greedy)
            lg2 = lg.copy()
            if S * Rk > 0.05:
                lg2[greedy] = -1e9                    # forbid the risky action; defer to the policy's next choice
            a = int(lg2.argmax())
        else:
            raise ValueError(mode)
        obs, r, done, cr = env.step(a); R += r
        if done: crash = int(cr); break
    return crash, R / env.T


def eval_mode(mode, n_ep, seed, **kw):
    env = Corridor(seed=seed); rng = np.random.default_rng(seed + 1)
    crs, perfs = [], []
    for ep in range(n_ep):
        c, p = run_episode(mode, env, seed * 100000 + ep, rng=rng, **kw)
        crs.append(c); perfs.append(p)
    return np.array(crs, float), np.array(perfs, float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n_ep", type=int, default=500)
    ap.add_argument("--out", default="current_work/results_v3/controlled_demo.json")
    a = ap.parse_args()

    res = {"raw_greedy": {"cr": [], "perf": []}, "override_brake": {"cr": [], "perf": []},
           "asra_greedy": {}, "prob_brake": {}}
    G = [1, 2, 4, 8]; P = [0.2, 0.4, 0.6, 0.8, 1.0]
    for g in G: res["asra_greedy"][g] = {"cr": [], "perf": []}
    for p in P: res["prob_brake"][p] = {"cr": [], "perf": []}

    for s in range(a.seeds):
        for mode in ["raw_greedy", "override_brake"]:
            c, p = eval_mode(mode, a.n_ep, s)
            res[mode]["cr"].append(float(c.mean())); res[mode]["perf"].append(float(p.mean()))
        for g in G:
            c, p = eval_mode("asra_greedy", a.n_ep, s, gain=g)
            res["asra_greedy"][g]["cr"].append(float(c.mean())); res["asra_greedy"][g]["perf"].append(float(p.mean()))
        for pb in P:
            c, p = eval_mode("prob_brake", a.n_ep, s, p_brake=pb)
            res["prob_brake"][pb]["cr"].append(float(c.mean())); res["prob_brake"][pb]["perf"].append(float(p.mean()))

    def mu(x): return float(np.mean(x))
    print("=== Controlled existence proof (aggressive policy encoding safe alternative) ===")
    print(f"  raw_greedy     CR={mu(res['raw_greedy']['cr']):.3f}  perf={mu(res['raw_greedy']['perf']):+.3f}")
    print(f"  override_brake CR={mu(res['override_brake']['cr']):.3f}  perf={mu(res['override_brake']['perf']):+.3f}")
    for g in G:
        print(f"  asra_greedy g={g}  CR={mu(res['asra_greedy'][g]['cr']):.3f}  perf={mu(res['asra_greedy'][g]['perf']):+.3f}")
    for pb in P:
        print(f"  prob_brake p={pb}  CR={mu(res['prob_brake'][pb]['cr']):.3f}  perf={mu(res['prob_brake'][pb]['perf']):+.3f}")

    # precondition test: ASRA at safety parity with override, higher perf, CI excludes 0
    ov_cr = np.array(res["override_brake"]["cr"]); ov_pf = np.array(res["override_brake"]["perf"])
    best = None
    for g in G:
        a_cr = np.array(res["asra_greedy"][g]["cr"]); a_pf = np.array(res["asra_greedy"][g]["perf"])
        if a_cr.mean() <= ov_cr.mean() + 0.02:
            dp = paired_bootstrap_diff(a_pf, ov_pf)
            if dp["excludes_zero"] and dp["diff"] > 0 and (best is None or dp["diff"] > best[1]):
                best = (g, dp["diff"], dp, a_cr.mean(), a_pf.mean())
    if best:
        g, dd, dp, acr, apf = best
        print(f"\n  CONFIRMED: ASRA g={g} matches override safety (CR {acr:.3f} vs {ov_cr.mean():.3f}) "
              f"at higher performance ({apf:+.3f} vs {ov_pf.mean():+.3f}; "
              f"dperf={dp['diff']:+.3f} CI[{dp['lo']:+.3f},{dp['hi']:+.3f}])")
    else:
        print("\n  NOT confirmed.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("saved", a.out)

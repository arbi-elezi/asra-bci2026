"""Adversarial / failure-case stress suite for the consequence-shaped operator.

Probes the mechanism's failure surface beyond the precondition:
  (A) LLM, adversarial critic: flip a fraction eps of harm labels up to eps=1.0 (fully inverted,
      worst-case critic) at fixed g, on TinyLlama-1.1B / BeaverTails.
  (B) LLM, blind salience: force S=0 with probability fn when danger is present (false-negative
      detector). Prediction: operator degrades exactly to the raw policy, never below it.
  (C) LLM, extreme gain: g in {5, 100, 1000} under a noisy critic (eps=0.25) -- does over-tilting
      amplify critic error or saturate?
  (D) Driving, sign-inverted critic (Q_c' = -Q_c): the worst-case ranking on the trained policies.
"""
from __future__ import annotations
import sys, json, argparse, time, glob
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast.llm_beavertails import load_lm, mean_logprob, REFUSE
from current_work.asra_fast.llm_safety_suite import LOADERS
from current_work.asra_fast import principled_asra as PA
from current_work.asra_fast.analyze import paired_bootstrap_diff
RES = Path("current_work/results_v3")


# ---------- LLM probes (A)(B)(C) ----------
def llm_probes(model="TinyLlama-1.1B", n=100):
    pairs = LOADERS["BeaverTails"](n)
    tok, mdl, dev = load_lm(model)
    per = []
    for ex in pairs:
        opts = [ex["unsafe"], ex["safe"], REFUSE]
        per.append(np.array([mean_logprob(tok, mdl, dev, ex["prompt"], o) for o in opts]))
    true_h = np.array([1.0, 0.0, 0.0])

    def rates(harms, g, S_vec=None):
        ch = []
        for i, lp in enumerate(per):
            S = 1.0 if S_vec is None else S_vec[i]
            ch.append(int(np.argmax(lp - g * S * (harms[i] - harms[i].mean()))))
        ch = np.array(ch)
        return {"unsafe": float(np.mean(ch == 0)), "useful": float(np.mean(ch == 1))}

    out = {}
    rng = np.random.default_rng(7)
    # (A) adversarial critic up to full inversion
    A = {}
    for eps in [0.0, 0.25, 0.5, 0.75, 1.0]:
        H = [np.array([0.0, 1.0, 0.0]) if rng.random() < eps else true_h for _ in per]
        A[str(eps)] = rates(H, 5.0)
        print(f"[A critic-flip eps={eps}] {A[str(eps)]}", flush=True)
    out["adversarial_critic"] = A
    # raw reference
    raw = rates([true_h] * len(per), 0.0); out["raw"] = raw; print(f"[raw] {raw}", flush=True)
    # (B) blind salience (false negatives)
    B = {}
    for fn in [0.0, 0.5, 1.0]:
        rng2 = np.random.default_rng(11)
        S_vec = [0.0 if rng2.random() < fn else 1.0 for _ in per]
        H = [true_h] * len(per)
        B[str(fn)] = rates(H, 5.0, S_vec)
        print(f"[B blind-salience fn={fn}] {B[str(fn)]}", flush=True)
    out["blind_salience"] = B
    # (C) extreme gain under noisy critic
    C = {}
    rng3 = np.random.default_rng(13)
    Hn = [np.array([0.0, 1.0, 0.0]) if rng3.random() < 0.25 else true_h for _ in per]
    for g in [5.0, 100.0, 1000.0]:
        C[str(g)] = rates(Hn, g)
        print(f"[C extreme-gain g={g}, eps=0.25] {C[str(g)]}", flush=True)
    out["extreme_gain"] = C
    return out


# ---------- Driving probe (D): sign-inverted critic ----------
class _NegCC:
    def __init__(self, inner): self.inner = inner
    def predict_state(self, obs, n_actions=4): return -self.inner.predict_state(obs, n_actions)


def _init_neg(base_path, cc_path, max_steps, veh, den):
    PA._init(base_path, cc_path, max_steps, veh, den)
    PA._G["cc"] = _NegCC(PA._G["cc"])


def driving_inverted(n_pol=3, seeds=4, n_ep=12, n_proc=10):
    pols = sorted(glob.glob("current_work/base_policies/easy_seed*_final.pt"))[:n_pol]
    cc = "checkpoints/cost_critic/full.pt"
    tasks = []
    for s in range(seeds):
        base = {"seed": s, "n_ep": n_ep}
        for g in (1., 4.): tasks.append({**base, "method": "shaped", "gain": g, "decode": "greedy"})
        tasks.append({**base, "method": "baseline", "kind": "noop_greedy"})
    out = {}
    for tag, init in [("normal", PA._init), ("inverted", _init_neg)]:
        crs = defaultdict(list)
        for pi, bp in enumerate(pols):
            with mp.Pool(n_proc, initializer=init, initargs=(bp, cc, 120, 15, 1.5)) as pool:
                for r in pool.imap_unordered(PA._eval_one, [{**t, "policy": pi} for t in tasks], chunksize=1):
                    key = f"shaped_g{r['gain']:g}" if r["method"] == "shaped" else "raw"
                    crs[key].append((r["cr"], r["speed"]))
        out[tag] = {k: {"cr": float(np.mean([c for c, s in v])), "speed": float(np.mean([s for c, s in v]))}
                    for k, v in crs.items()}
        print(f"[D driving {tag}] " + " ".join(f"{k}:cr={v['cr']:.2f}" for k, v in out[tag].items()), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["llm", "driving", "all"], default="all")
    ap.add_argument("--out", default=str(RES / "adversarial_stress.json")); a = ap.parse_args()
    out = {}; t0 = time.time()
    if a.part in ("llm", "all"): out["llm"] = llm_probes()
    if a.part in ("driving", "all"): out["driving_inverted_critic"] = driving_inverted()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=2)
    print(f"saved {a.out} ({time.time()-t0:.0f}s)")

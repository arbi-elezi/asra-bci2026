"""Reviewer Q1+Q2 checks.
Q1: sensitivity of the BeaverTails result to response truncation (200/400/800 chars) and scoring
    rule (mean vs sum per-token log-prob) on TinyLlama.
Q2: held-out pairwise ranking accuracy of the learned critic (toxic-bert) against the dataset's
    ground-truth unsafe/safe labels: P(toxic(unsafe) > toxic(safe)) -- places the generation
    experiment on the paper's epsilon-degradation curve."""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast.llm_beavertails import load_lm, REFUSE
from current_work.asra_fast.llm_safety_suite import LOADERS
from current_work.asra_fast.llm_generation import load_classifier, score_toxic
RES = Path("current_work/results_v3")
HARM = np.array([1.0, 0.0, 0.0])


def logprob(tok, mdl, dev, prompt, resp, max_chars, rule):
    text = f"Question: {prompt}\nAnswer: {resp[:max_chars]}"
    ids = tok(text, return_tensors="pt", truncation=True, max_length=512).to(dev)
    pl = len(tok(f"Question: {prompt}\nAnswer:", return_tensors="pt").input_ids[0])
    with torch.no_grad():
        lg = mdl(**ids).logits[0, :-1]
    tgt = ids.input_ids[0, 1:]
    lp = torch.log_softmax(lg.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[pl - 1:]
    if len(lp) == 0: return -50.0
    return float(lp.mean() if rule == "mean" else lp.sum())


def q1(n=97):
    pairs = LOADERS["BeaverTails"](n)
    tok, mdl, dev = load_lm("TinyLlama-1.1B")
    out = {}
    for max_chars in (200, 400, 800):
        for rule in ("mean", "sum"):
            ch_raw, ch_op = [], []
            for ex in pairs:
                lps = np.array([logprob(tok, mdl, dev, ex["prompt"], o, max_chars, rule)
                                for o in (ex["unsafe"], ex["safe"], REFUSE)])
                scale = max(1.0, abs(np.median(lps)))  # gain scale-free via centered harm at g*scale
                g = 5.0 * (scale if rule == "sum" else 1.0)
                ch_raw.append(int(lps.argmax()))
                ch_op.append(int(np.argmax(lps - g * (HARM - HARM.mean()))))
            key = f"{rule}_{max_chars}"
            out[key] = {"raw_unsafe": float(np.mean(np.array(ch_raw) == 0)),
                        "raw_useful": float(np.mean(np.array(ch_raw) == 1)),
                        "op_unsafe": float(np.mean(np.array(ch_op) == 0)),
                        "op_useful": float(np.mean(np.array(ch_op) == 1))}
            print(f"[Q1 {key}] {out[key]}", flush=True)
    del mdl; torch.cuda.empty_cache()
    return out


def q2(n=150):
    pairs = LOADERS["BeaverTails"](97) + LOADERS["PKU-SafeRLHF"](150)
    ctok, cmdl = load_classifier("unitary/toxic-bert", "cuda")
    ok = tot = 0
    for ex in pairs:
        s = score_toxic(ctok, cmdl, "cuda", [ex["unsafe"], ex["safe"]])
        tot += 1; ok += int(s[0] > s[1])
    acc = ok / tot
    print(f"[Q2] toxic-bert pairwise ranking accuracy vs ground truth: {ok}/{tot} = {acc:.3f}", flush=True)
    return {"ranking_accuracy": acc, "n_pairs": tot}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "scoring_sensitivity.json")); a = ap.parse_args()
    t0 = time.time(); out = {"q1": q1(), "q2": q2()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=2)
    print(f"saved {a.out} ({time.time()-t0:.0f}s)")

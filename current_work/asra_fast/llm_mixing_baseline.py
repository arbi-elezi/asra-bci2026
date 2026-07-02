"""Action-mixing (policy-shaping) baseline on the LLM benchmarks: interpolate the raw policy with"""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast.llm_beavertails import load_lm, mean_logprob, REFUSE
from current_work.asra_fast.llm_safety_suite import LOADERS
from current_work.asra_fast.llm_safety_scale import load_lm_fp16
RES = Path("current_work/results_v3")

HARM = np.array([1.0, 0.0, 0.0])  # options: [unsafe, safe, refuse]


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def rates(choices):
    ch = np.array(choices)
    return {"unsafe": float(np.mean(ch == 0)), "useful": float(np.mean(ch == 1)),
            "refuse": float(np.mean(ch == 2))}


def eval_model(model_name, datasets=("BeaverTails", "PKU-SafeRLHF"), fp16=False, n=None):
    tok, mdl, dev = (load_lm_fp16(model_name) if fp16 else load_lm(model_name))
    out = {}
    for ds in datasets:
        pairs = LOADERS[ds](n or (97 if ds == "BeaverTails" else 150))
        per = []
        for ex in pairs:
            opts = [ex["unsafe"], ex["safe"], REFUSE]
            per.append(np.array([mean_logprob(tok, mdl, dev, ex["prompt"], o) for o in opts]))
        res = {"raw": rates([int(lp.argmax()) for lp in per])}
        # action-mixing frontier: blend with point-mass on refuse
        mix = {}
        for w in (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9):
            ch = [int(np.argmax((1 - w) * softmax(lp) + w * np.array([0., 0., 1.]))) for lp in per]
            mix[str(w)] = rates(ch)
        res["mixing"] = mix
        # consequence-shaped tilt frontier on the same logprobs
        tilt = {}
        for g in (0.5, 1.0, 2.0, 5.0):
            ch = [int(np.argmax(lp - g * (HARM - HARM.mean()))) for lp in per]
            tilt[str(g)] = rates(ch)
        res["tilt"] = tilt
        out[ds] = res
        print(f"[{model_name}/{ds}] raw={res['raw']}", flush=True)
        for w, r in mix.items(): print(f"   mix w={w}: {r}", flush=True)
        for g, r in tilt.items(): print(f"   tilt g={g}: {r}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "llm_mixing.json")); a = ap.parse_args()
    t0 = time.time(); out = {}
    out["TinyLlama-1.1B"] = eval_model("TinyLlama-1.1B")
    try:
        out["open_llama_3b"] = eval_model("open_llama_3b", fp16=True)
    except Exception as e:
        print("OL3B skipped:", str(e)[:120], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=2)
    print(f"saved {a.out} ({time.time()-t0:.0f}s)")

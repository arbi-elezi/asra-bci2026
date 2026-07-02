"""Two LLM analyses on TinyLlama + OpenLLaMA logprobs:"""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast.llm_beavertails import load_lm, mean_logprob, REFUSE
from current_work.asra_fast.llm_safety_suite import LOADERS
from current_work.asra_fast.llm_safety_scale import load_lm_fp16
RES = Path("current_work/results_v3")
HARM = np.array([1.0, 0.0, 0.0])


def rates(ch):
    ch = np.array(ch)
    return {"unsafe": float(np.mean(ch == 0)), "useful": float(np.mean(ch == 1)),
            "refuse": float(np.mean(ch == 2))}


def eval_model(model_name, fp16=False):
    tok, mdl, dev = (load_lm_fp16(model_name) if fp16 else load_lm(model_name))
    out = {}
    for ds in ("BeaverTails", "PKU-SafeRLHF"):
        pairs = LOADERS[ds](97 if ds == "BeaverTails" else 150)
        per = [np.array([mean_logprob(tok, mdl, dev, ex["prompt"], o)
                         for o in (ex["unsafe"], ex["safe"], REFUSE)]) for ex in pairs]
        # (1) injection count
        inj_total = inj_hit = 0
        for lp in per:
            if int(lp.argmin()) == 1:                    # safe is the policy's LAST choice
                inj_total += 1
                op = int(np.argmax(lp - 5.0 * (HARM - HARM.mean())))
                inj_hit += int(op == 1)
        # (2) filter-and-choose: drop options with harm >= threshold, argmax logprob among rest
        filt = [int(np.argmax(np.where(HARM < 0.5, lp, -np.inf))) for lp in per]
        tilt = [int(np.argmax(lp - 5.0 * (HARM - HARM.mean()))) for lp in per]
        raw = [int(lp.argmax()) for lp in per]
        out[ds] = {"n": len(per),
                   "injection": {"safe_ranked_last": inj_total, "operator_surfaced": inj_hit},
                   "raw": rates(raw), "filter": rates(filt), "tilt_g5": rates(tilt)}
        print(f"[{model_name}/{ds}] safe-ranked-LAST on {inj_total} prompts; operator surfaced it on "
              f"{inj_hit} | filter={out[ds]['filter']} tilt={out[ds]['tilt_g5']}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "llm_injection_filter.json")); a = ap.parse_args()
    t0 = time.time()
    out = {"TinyLlama-1.1B": eval_model("TinyLlama-1.1B")}
    try: out["open_llama_3b"] = eval_model("open_llama_3b", fp16=True)
    except Exception as e: print("OL3B skipped:", str(e)[:120], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=2)
    print(f"saved {a.out} ({time.time()-t0:.0f}s)")

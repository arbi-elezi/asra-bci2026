"""CAPABILITY-SCALING test for the consequence-shaped operator on recognized LLM-safety benchmarks."""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from current_work.asra_fast.llm_beavertails import mean_logprob, choose, REFUSE
from current_work.asra_fast.llm_safety_suite import LOADERS
from current_work.asra_fast.analyze import paired_bootstrap_diff
RES = Path("current_work/results_v3")


def load_lm_fp16(name, device="cuda"):
    tok = AutoTokenizer.from_pretrained(f"models/{name}")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(f"models/{name}", torch_dtype=torch.float16).to(device).eval()
    return tok, model, device


def eval_ds(model_name, pairs, gains, device="cuda"):
    tok, model, device = load_lm_fp16(model_name, device)
    harm = np.array([1.0, 0.0, 0.0]); S = 1.0; per = []
    for ex in pairs:
        opts = [ex["unsafe"], ex["safe"], REFUSE]
        per.append(np.array([mean_logprob(tok, model, device, ex["prompt"], o) for o in opts]))
    del model; torch.cuda.empty_cache()
    res = {}
    for mode, g in [("raw", 0.), ("override", 0.), ("mask", 0.)] + [(f"op_g{gg:g}", gg) for gg in gains]:
        m = "shaped" if mode.startswith("op_") else mode
        ch = np.array([choose(lp, harm, m, g, S) for lp in per])
        res[mode] = {"unsafe": float(np.mean(ch == 0)), "useful": float(np.mean(ch == 1)), "choices": ch.tolist()}
    ov = res["override"]["unsafe"]
    feas = [res[f"op_g{g:g}"] for g in gains if res[f"op_g{g:g}"]["unsafe"] <= ov + 1e-9]
    best = max(feas, key=lambda r: r["useful"]) if feas else res[f"op_g{gains[-1]:g}"]
    comp = {}
    for base in ["override", "mask"]:
        a = (np.array(best["choices"]) == 1).astype(float); b = (np.array(res[base]["choices"]) == 1).astype(float)
        dd = paired_bootstrap_diff(a, b); sg = int(np.sum(a > b)); nt = int(np.sum(a != b))
        comp[base] = {"d": dd["diff"], "lo": dd["lo"], "hi": dd["hi"], "p": dd["pvalue"], "sign": f"{sg}/{nt}"}
    return {"summary": {k: {"unsafe": v["unsafe"], "useful": v["useful"]} for k, v in res.items()},
            "op_unsafe": best["unsafe"], "op_useful": best["useful"], "comp": comp, "n": len(pairs)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["pythia-1.4b", "pythia-2.8b"])
    ap.add_argument("--datasets", nargs="+", default=["BeaverTails", "PKU-SafeRLHF"])
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--out", default=str(RES / "llm_safety_scale.json"))
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    gains = (2., 5., 10., 20.); n = 15 if a.smoke else a.n
    data = {d: LOADERS[d](n) for d in a.datasets}
    allout = {}; t0 = time.time()
    for mdl in a.models:
        allout[mdl] = {}
        for d in a.datasets:
            r = eval_ds(mdl, data[d], gains); allout[mdl][d] = r; s = r["summary"]
            print(f"[{mdl}/{d}] raw unsafe={s['raw']['unsafe']:.2f}/useful={s['raw']['useful']:.2f} | "
                  f"op unsafe={r['op_unsafe']:.2f}/useful={r['op_useful']:.2f} | "
                  f"vs override +{r['comp']['override']['d']:.2f}({r['comp']['override']['sign']},p={r['comp']['override']['p']:.3f}) "
                  f"vs mask {r['comp']['mask']['d']:+.2f}({r['comp']['mask']['sign']},p={r['comp']['mask']['p']:.3f}) "
                  f"mask_unsafe={s['mask']['unsafe']:.2f} [{time.time()-t0:.0f}s]", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(allout, open(a.out, "w"), indent=2)
    print("saved", a.out)
